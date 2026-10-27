"""Generate entirely synthetic, non-patient phantom data to exercise the IO contract."""

from pathlib import Path
import json

import numpy as np
from PIL import Image


def make_phantom(root: str | Path, *, seed: int = 42) -> dict:
    root = Path(root)
    real_root = root / "real"
    img_dir = real_root / "images"
    mask_dir = real_root / "annotations"
    gen_root = root / "generated"
    for directory in (img_dir, mask_dir, gen_root / "images", gen_root / "logits_npy"):
        directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_train, n_val, n_test = 16, 8, 8
    manifest = {"train": [], "val": [], "test": []}
    yy, xx = np.mgrid[0:64, 0:64]
    def one_pair(index: int):
        cx = 30 + 5 * np.sin(index * 0.63)
        cy = 33 + 4 * np.cos(index * 0.71)
        r1 = 13 + (index % 5)
        r2 = 11 + (index % 4)
        signed = 1 - ((xx - cx) / r1) ** 2 - ((yy - cy) / r2) ** 2
        mask = signed > 0
        noise = rng.normal(0, 15, (64, 64))
        texture = np.clip(42 + (mask * 105) + noise + 8 * np.sin(xx / 3 + index), 0, 255).astype(np.uint8)
        image = np.stack([texture, texture, texture], axis=-1)
        return image, mask.astype(np.uint8), np.stack([-2.7 * signed, 2.7 * signed]).astype(np.float32)
    counter = 0
    for split, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for k in range(count):
            index = counter
            counter += 1
            img, mask, _ = one_pair(index)
            Image.fromarray(img).save(img_dir / f"{index:05d}.JPG")
            Image.fromarray(mask * 255).save(mask_dir / f"{index:05d}.PNG")
            manifest[split].append({"image": f"images/{index:05d}.JPG",
                                    "mask": f"annotations/{index:05d}.PNG",
                                    "label": k % 8, "patient_id": f"phantom_patient_{index:05d}"})
    for idx in range(12):
        img, _mask, logits = one_pair(idx + 100)
        Image.fromarray(img).save(gen_root / "images" / f"{idx:06d}.jpg")
        np.save(gen_root / "logits_npy" / f"{idx:06d}.npy", logits)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"path": str(root), "counts": {k: len(v) for k, v in manifest.items()}, "generated": 12}
