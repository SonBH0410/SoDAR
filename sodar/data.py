"""Torch datasets; all image/mask membership is manifest-driven."""

import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .io import (augment_real_pair, generated_pairs, load_manifest,
                 preprocess_image, probs_from_generated, read_two_channel_array,
                 real_pair_paths)


class RealDataset(Dataset):
    def __init__(self, conf: dict, split: str, *, augment: str = "none"):
        self.conf = conf
        self.split = split
        self.records = load_manifest(conf)[split]
        self.augment = augment if split == "train" else "none"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        image_path, mask_path = real_pair_paths(self.conf, self.split, record)
        with Image.open(image_path) as i, Image.open(mask_path) as m:
            image, mask = i.convert("RGB"), m.convert("L")
        image, mask = augment_real_pair(image, mask, self.augment, random)
        size = int(self.conf["data"]["size"])
        img = preprocess_image(image, size, self.conf["data"]["normalize"])
        mask = np.asarray(mask.resize((size, size), Image.Resampling.NEAREST), dtype=np.uint8)
        mask = np.ascontiguousarray((mask > 0).astype(np.float32)[None])
        return {"image": torch.from_numpy(img), "mask": torch.from_numpy(mask),
                "label": int(record["label"]), "id": str(record["image"])}


class SyntheticDataset(Dataset):
    def __init__(self, conf: dict):
        self.conf = conf
        self.pairs = generated_pairs(conf)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, logits_path = self.pairs[idx]
        with Image.open(image_path) as i:
            image = preprocess_image(i, int(self.conf["data"]["size"]), self.conf["data"]["normalize"])
        raw = read_two_channel_array(logits_path, expected_format=self.conf["data"]["synthetic_format"])
        # Resize the class channels BEFORE softmax; avoids artifacts from interpolated probabilities.
        raw_tensor = torch.from_numpy(raw).unsqueeze(0)
        size = int(self.conf["data"]["size"])
        if raw.shape[-2:] != (size, size):
            raw_tensor = torch.nn.functional.interpolate(raw_tensor, size=(size, size), mode="bilinear", align_corners=False)
        raw = raw_tensor.squeeze(0).numpy()
        p = probs_from_generated(raw, temperature=float(self.conf["synthetic"]["temperature"]), fmt=self.conf["data"]["synthetic_format"])
        # Hard mask independent of T. Soft target always (2,H,W).
        hard = (raw[1] > raw[0]).astype(np.float32)[None]
        return {"image": torch.from_numpy(image), "prob": torch.from_numpy(np.ascontiguousarray(p)),
                "hard": torch.from_numpy(np.ascontiguousarray(hard)), "id": image_path.stem}
