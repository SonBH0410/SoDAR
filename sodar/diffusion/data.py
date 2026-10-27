"""Train-only manifest loader and independently validated DDPM configuration."""

from copy import deepcopy
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from sodar.config import PROJECT_ROOT, deep_merge, read_yaml
from sodar.io import audit, load_manifest, real_pair_paths, train_records_fingerprint


DDPM_BASE = PROJECT_ROOT / "configs" / "ddpm.yaml"


def ddpm_config(*, path: str | None = None, data_root: str | None = None,
                manifest: str | None = None, steps: int | None = None,
                batch_size: int | None = None, grad_accum: int | None = None,
                save_root: str | None = None, seed: int | None = None,
                device: str | None = None, amp: bool | None = None) -> dict:
    conf = read_yaml(DDPM_BASE)
    if path:
        conf = deep_merge(conf, read_yaml(path))
    overrides = (("data", "root", data_root), ("data", "manifest", manifest),
                 ("train", "steps", steps), ("train", "batch_size", batch_size),
                 ("train", "grad_accum", grad_accum), ("train", "save_root", save_root),
                 ("train", "seed", seed), ("train", "device", device),
                 ("train", "amp", amp))
    for group, key, value in overrides:
        if value is not None:
            conf[group][key] = value
    validate_ddpm_config(conf)
    return conf


def validate_ddpm_config(conf: dict):
    d, m, p, tr = (conf[k] for k in ("data", "model", "diffusion", "train"))
    size = int(d["image_size"])
    factor = 2**(len(m["dim_mults"])-1)
    if size < 8 or size % factor:
        raise ValueError(f"data.image_size must be >=8 and divisible by {factor}")
    if not 0 <= float(d["flip_probability"]) <= 1:
        raise ValueError("data.flip_probability must be between 0 and 1")
    if p["objective"] != "pred_x0" or p["beta_schedule"] != "cosine":
        raise ValueError("This port implements exactly pred_x0 with a cosine schedule")
    if not 2 <= int(p["timesteps"]) or not 1 <= int(p["sampling_timesteps"]) <= int(p["timesteps"]):
        raise ValueError("Invalid diffusion timestep configuration")
    if not 0 < float(p["snr_balance_b"]) <= 1 or float(p["ddim_eta"]) < 0:
        raise ValueError("Invalid SNR balancing or DDIM eta")
    for name in ("steps", "batch_size", "grad_accum", "save_every", "log_every", "ema_update_every"):
        if int(tr[name]) < 1:
            raise ValueError(f"train.{name} must be positive")
    if not 0 < float(tr["ema_decay"]) < 1:
        raise ValueError("train.ema_decay must be in (0,1)")
    if not 0 < float(tr["lr"]) or int(tr["num_workers"]) < 0:
        raise ValueError("Invalid train.lr or train.num_workers")


def real_audit_and_fingerprint(conf: dict) -> tuple[dict, str]:
    """Does not read val/test pixel arrays or train diffusion with them."""
    data_conf = {"data": {"root": conf["data"]["root"], "layout": "manifest",
                          "manifest": conf["data"]["manifest"],
                          "enforce_counts": conf["data"]["enforce_counts"]}}
    report = audit(data_conf, include_synthetic=False)
    manifest = load_manifest(data_conf)
    digest = train_records_fingerprint(manifest)
    return report, digest


class JointDiffusionDataset(Dataset):
    def __init__(self, conf: dict):
        self.conf = deepcopy(conf)
        data_conf = {"data": {"root": conf["data"]["root"], "layout": "manifest",
                              "manifest": conf["data"]["manifest"],
                              "enforce_counts": conf["data"]["enforce_counts"]}}
        self.data_conf = data_conf
        self.records = load_manifest(data_conf)["train"]  # NEVER read val/test pixels
        self.size = int(conf["data"]["image_size"])
        self.flip_probability = float(conf["data"]["flip_probability"])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        image_path, mask_path = real_pair_paths(self.data_conf, "train", self.records[index])
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            rgb, mask = image.convert("RGB"), mask.convert("L")
        if self.flip_probability > 0 and random.random() < self.flip_probability:
            rgb, mask = ImageOps.mirror(rgb), ImageOps.mirror(mask)
        rgb = rgb.resize((self.size, self.size), Image.Resampling.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        img_arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        mask_arr = np.ascontiguousarray((np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)[None])
        return torch.from_numpy(img_arr), torch.from_numpy(mask_arr)
