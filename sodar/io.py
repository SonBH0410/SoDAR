"""Pure-Python data contract, path resolver and phantom-friendly IO."""

from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .config import resolve_manifest


EXPECTED_COUNTS = {"train": 850, "val": 150, "test": 469}


def train_records_fingerprint(manifest: dict) -> str:
    """Stable digest of ordered train membership, not image pixel contents."""
    canonical = json.dumps(manifest["train"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_manifest(conf: dict) -> dict:
    path = resolve_manifest(conf)
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if set(manifest) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test keys, found {list(manifest)}")
    all_names = []
    for split, records in manifest.items():
        if conf["data"]["enforce_counts"] and len(records) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split} must contain {EXPECTED_COUNTS[split]} samples; got {len(records)}")
        for r in records:
            if not all(k in r for k in ("image", "mask", "label")):
                raise ValueError(f"Missing required keys in {split} record: {r}")
            if not isinstance(r["label"], int) or not 0 <= r["label"] < 8:
                raise ValueError(f"Invalid image-level category in {split}: {r['label']}")
            all_names.append(r["image"].lower())
    if len(all_names) != len(set(all_names)):
        raise ValueError("Duplicate image path across real splits")
    return manifest


def real_pair_paths(conf: dict, split: str, record: dict) -> tuple[Path, Path]:
    root = Path(conf["data"]["root"])
    if conf["data"]["layout"] == "manifest":
        return root / record["image"], root / record["mask"]
    folder = conf["data"]["split_val_name"] if split == "val" else split
    stem = Path(record["image"]).stem
    return root / folder / "images" / f"{stem}.JPG", root / folder / "masks" / f"{stem}.PNG"


def generated_pairs(conf: dict) -> list[tuple[Path, Path]]:
    root = Path(conf["data"]["synthetic_root"])
    image_dir = root / "images"
    preferred = root / conf["data"]["synthetic_logits_dir"]
    fallback = root / ("soft_labels_npy" if preferred.name == "logits_npy" else "logits_npy")
    logits_dir = preferred if preferred.is_dir() else fallback
    if not image_dir.is_dir() or not logits_dir.is_dir():
        raise FileNotFoundError(f"Expected {image_dir} and {preferred} (fallback {fallback})")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise ValueError(f"No synthetic images in {image_dir}")
    result = []
    for image in images:
        candidates = [
            logits_dir / f"{image.stem}.npy",
            logits_dir / f"soft_label_{image.stem}.npy",
            logits_dir / f"soft_label_{image.stem.removeprefix('img_')}.npy",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise FileNotFoundError(f"Missing paired logits for {image}; tried {candidates}")
        result.append((image, found))
    return result


def read_two_channel_array(path: Path, *, expected_format: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected C,H,W or H,W,C for {path}, got {arr.shape}")
    if arr.shape[0] == 2:
        pass
    elif arr.shape[-1] == 2:
        arr = np.moveaxis(arr, -1, 0)
    else:
        raise ValueError(f"Expected 2 background/foreground channels in {path}, got {arr.shape}")
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"NaN/Inf found in {path}")
    if expected_format == "probabilities":
        if arr.min() < -1e-5 or arr.max() > 1 + 1e-5:
            raise ValueError(f"Probability map outside [0,1]: {path}")
        if not np.allclose(arr.sum(axis=0), 1, atol=2e-2):
            raise ValueError(f"Probabilities do not sum to 1: {path}")
    return arr


def probs_from_generated(arr: np.ndarray, *, temperature: float, fmt: str) -> np.ndarray:
    if fmt == "probabilities":
        if temperature != 1:
            # Temperature scaling of known probabilities through log-probabilities.
            scores = np.log(np.clip(arr, 1e-6, 1)) / temperature
        else:
            return arr.astype(np.float32)
    else:
        scores = arr / temperature
    scores = scores - np.max(scores, axis=0, keepdims=True)
    exp = np.exp(scores)
    return (exp / exp.sum(axis=0, keepdims=True)).astype(np.float32)


def audit(conf: dict, *, include_synthetic: bool = True, check_all_arrays: bool = True) -> dict:
    manifest = load_manifest(conf)
    result = {"real": {}, "missing_paths": [], "patient_disjoint": "unverified"}
    patient_by_split = {}
    for split, records in manifest.items():
        counts = Counter(int(r["label"]) for r in records)
        result["real"][split] = {"count": len(records), "labels": dict(sorted(counts.items()))}
        for rec in records:
            img, mask = real_pair_paths(conf, split, rec)
            if not img.is_file():
                result["missing_paths"].append(str(img))
            if not mask.is_file():
                result["missing_paths"].append(str(mask))
        if all("patient_id" in r for r in records):
            patient_by_split[split] = {str(r["patient_id"]) for r in records}
    if len(patient_by_split) == 3:
        overlap = {"train_val": len(patient_by_split["train"] & patient_by_split["val"]),
                   "train_test": len(patient_by_split["train"] & patient_by_split["test"]),
                   "val_test": len(patient_by_split["val"] & patient_by_split["test"])}
        result["patient_overlaps"] = overlap
        result["patient_disjoint"] = "yes" if not any(overlap.values()) else "NO"
        if any(overlap.values()):
            raise ValueError(f"Patient overlap: {overlap}")
    if result["missing_paths"]:
        raise FileNotFoundError(f"Missing {len(result['missing_paths'])} real paths, e.g. {result['missing_paths'][:5]}")
    if include_synthetic:
        pairs = generated_pairs(conf)
        result["synthetic"] = {"count": len(pairs), "format": conf["data"]["synthetic_format"]}
        synthetic_root = Path(conf["data"]["synthetic_root"])
        provenance_file = synthetic_root / "provenance.json"
        if provenance_file.is_file():
            provenance = json.loads(provenance_file.read_text(encoding="utf-8"))
            digest = train_records_fingerprint(manifest)
            if provenance.get("status") != "complete" or provenance.get("source_split") != "train":
                raise ValueError("Synthetic provenance must be complete and train-only")
            if provenance.get("train_records_sha256") != digest:
                raise ValueError("DDPM train manifest fingerprint mismatch: possible train/test leakage")
            if provenance.get("num_samples") != len(pairs):
                raise ValueError("Synthetic provenance sample count mismatch")
            if provenance.get("synthetic_format") != conf["data"]["synthetic_format"]:
                raise ValueError("Synthetic format disagrees with provenance; logits must not be double-softmaxed")
            result["synthetic"]["train_manifest_verified"] = True
            result["synthetic"]["generator_checkpoint_step"] = provenance.get("checkpoint_step")
        else:
            result["synthetic"]["train_manifest_verified"] = False
            if conf["data"].get("require_synthetic_provenance", False):
                raise ValueError(f"Missing synthetic provenance.json at {synthetic_root}. "
                                 "Generate using scripts.sample_ddpm and the train-850 checkpoint")
        if check_all_arrays:
            for image_path, npy_path in pairs:
                with Image.open(image_path) as im:
                    im.verify()
                _ = read_two_channel_array(npy_path, expected_format=conf["data"]["synthetic_format"])
    return result


def preprocess_image(image: Image.Image, size: int, normalize: str) -> np.ndarray:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if normalize == "imagenet":
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]) / np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    elif normalize != "none":
        raise ValueError(f"Unknown normalization: {normalize}")
    return np.ascontiguousarray(arr.astype(np.float32))


def augment_real_pair(image: Image.Image, mask: Image.Image, kind: str, rng) -> tuple[Image.Image, Image.Image]:
    if kind in ("flip", "combined") and rng.random() < 0.5:
        image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
    if kind == "combined":
        angle = rng.uniform(-10.0, 10.0)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.9, 1.1))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.9, 1.1))
    return image, mask
