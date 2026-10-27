"""Numerically audit synthetic image/logit consistency before student training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from sodar.io import generated_pairs, probs_from_generated, read_two_channel_array


def run_audit(root: str | Path, *, expected_count: int | None = None) -> dict:
    root = Path(root)
    pairs = generated_pairs({"data": {"synthetic_root": str(root),
                                        "synthetic_logits_dir": "logits_npy"}})
    if expected_count is not None and len(pairs) != expected_count:
        raise ValueError(f"Expected {expected_count} generated pairs, found {len(pairs)}")
    provenance = root / "provenance.json"
    meta = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else None
    if meta is not None and (meta.get("status") != "complete" or meta.get("num_samples") != len(pairs)):
        raise ValueError("Synthetic provenance incomplete or count mismatch")
    entropies, area_ratios = [], []
    image_hashes = set()
    checked_masks = 0
    for path, npy_path in pairs:
        with Image.open(path) as image:
            image.verify()
        image_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        logits = read_two_channel_array(npy_path, expected_format="logits")
        with Image.open(path) as image:
            size = (image.height, image.width)
        if logits.shape[-2:] != size:
            raise ValueError(f"Image/logit size mismatch at {path}: {size} vs {logits.shape}")
        p = probs_from_generated(logits, temperature=1.0, fmt="logits")
        if not np.isfinite(p).all() or not np.allclose(p.sum(0), 1, atol=1e-5):
            raise ValueError(f"Invalid teacher probabilities: {npy_path}")
        entropies.append(float(-(p * np.log(np.clip(p, 1e-7, 1))).sum(0).mean()))
        mask = logits.argmax(0).astype(np.uint8)
        area_ratios.append(float(mask.mean()))
        hard_path = root / "hard_masks" / f"{path.stem}.png"
        if hard_path.is_file():
            with Image.open(hard_path) as hard:
                stored = np.asarray(hard.convert("L")) > 0
            if not np.array_equal(stored, mask > 0):
                raise ValueError(f"Hard preview mask disagrees with logits: {hard_path}")
            checked_masks += 1
    return {"result": "PASS", "root": str(root), "pairs": len(pairs),
            "identical_image_duplicates": len(pairs)-len(image_hashes),
            "hard_preview_masks_verified": checked_masks,
            "mean_teacher_entropy": float(np.mean(entropies)),
            "mean_foreground_ratio": float(np.mean(area_ratios)),
            "provenance": {k: meta.get(k) for k in
                           ("status", "checkpoint_step", "train_records_sha256", "source_split")}
            if meta is not None else None,
            "limitations": "File contracts only; NOT a clinical realism or segmentation quality metric"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.generated_root, expected_count=args.expected_count), indent=2))


if __name__ == "__main__":
    main()
