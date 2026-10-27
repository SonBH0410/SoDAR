"""Reproducible joint diffusion training, checkpoint/resume and safe DDIM export."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from .data import JointDiffusionDataset, real_audit_and_fingerprint
from .model import JointUNet
from .process import JointGaussianDiffusion


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment")
    return device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_diffusion(conf: dict) -> JointGaussianDiffusion:
    model_conf, proc = conf["model"], conf["diffusion"]
    network = JointUNet(dim=int(model_conf["dim"]), dim_mults=model_conf["dim_mults"],
                        heads=int(model_conf["heads"]), dim_head=int(model_conf["dim_head"]),
                        dropout=float(model_conf["dropout"]))
    return JointGaussianDiffusion(network, image_size=int(conf["data"]["image_size"]),
                                  timesteps=int(proc["timesteps"]),
                                  sampling_timesteps=int(proc["sampling_timesteps"]),
                                  snr_balance_b=float(proc["snr_balance_b"]),
                                  ddim_eta=float(proc["ddim_eta"]))


def digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def save_json(path: Path, content: dict):
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(folder: Path, step: int, diffusion, ema, optimizer, scaler,
                    conf: dict, train_hash: str, elapsed: float):
    state = {"schema": "sodar.ddpm.v1", "step": step,
             "model": diffusion.model.state_dict(), "ema_model": ema.state_dict(),
             "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
             "config": conf, "train_records_sha256": train_hash,
             "rng": _rng_state(), "elapsed_seconds": elapsed}
    target = folder / f"step-{step:07d}.pt"
    staged = folder / f"step-{step:07d}.pt.incomplete"
    torch.save(state, staged)
    staged.replace(target)
    shutil.copy2(target, folder / "latest.pt.incomplete")
    (folder / "latest.pt.incomplete").replace(folder / "latest.pt")
    save_json(folder / "latest.json", {"step": step, "checkpoint": target.name,
                                      "train_records_sha256": train_hash})
    return target


def load_checkpoint(path: str | Path) -> dict:
    # weights_only=False is necessary for optimizer/RNG Python state. Only load
    # trusted checkpoints created locally by this repository, never strangers'.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("schema") != "sodar.ddpm.v1":
        raise ValueError("Not a SoDAR-v0.2 joint DDPM checkpoint")
    return ckpt


def _validate_resume(ckpt: dict, conf: dict, train_hash: str):
    if ckpt["train_records_sha256"] != train_hash:
        raise ValueError("Resume rejected: train-850 manifest changed")
    saved = ckpt["config"]
    for group in ("data", "model", "diffusion"):
        if saved[group] != conf[group]:
            raise ValueError(f"Resume rejected: {group} config has changed")
    for field in ("batch_size", "grad_accum", "lr", "weight_decay", "adam_betas",
                  "ema_decay", "ema_update_every", "amp"):
        if saved["train"][field] != conf["train"][field]:
            raise ValueError(f"Resume rejected: train.{field} changed")
    if int(conf["train"]["steps"]) <= int(ckpt["step"]):
        raise ValueError("--steps must exceed the saved checkpoint step")


def train_generator(conf: dict, *, resume: str | None = None) -> Path:
    """Train strictly on train records; write checkpoint artifacts separately."""
    from torch.optim import AdamW

    report, train_hash = real_audit_and_fingerprint(conf)
    set_seed(int(conf["train"]["seed"]))
    device = choose_device(str(conf["train"]["device"]))
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    use_amp = bool(conf["train"]["amp"]) and device.type == "cuda"
    diffusion = build_diffusion(conf).to(device)
    ema_model = deepcopy(diffusion.model).to(device).eval().requires_grad_(False)
    optimizer = AdamW(diffusion.model.parameters(), lr=float(conf["train"]["lr"]),
                      betas=tuple(float(x) for x in conf["train"]["adam_betas"]),
                      weight_decay=float(conf["train"]["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    folder = Path(conf["train"]["save_root"])
    if (folder / "latest.pt").exists() and resume is None:
        raise FileExistsError(f"{folder}/latest.pt exists; use --resume or a new save-root")
    step = 0
    prior_elapsed = 0.0
    if resume is not None:
        ckpt = load_checkpoint(resume)
        _validate_resume(ckpt, conf, train_hash)
        diffusion.model.load_state_dict(ckpt["model"], strict=True)
        ema_model.load_state_dict(ckpt["ema_model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step = int(ckpt["step"])
        prior_elapsed = float(ckpt.get("elapsed_seconds", 0))
        _restore_rng(ckpt["rng"])
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "config.json", conf)
    save_json(folder / "audit.json", {"real": report, "train_records_sha256": train_hash,
                                     "generator_train_split": "train", "n_train": len(JointDiffusionDataset(conf))})
    dataset = JointDiffusionDataset(conf)
    loader_generator = torch.Generator().manual_seed(int(conf["train"]["seed"]) + step)
    loader = DataLoader(dataset, batch_size=int(conf["train"]["batch_size"]), shuffle=True,
                        num_workers=int(conf["train"]["num_workers"]), drop_last=False,
                        generator=loader_generator, pin_memory=device.type == "cuda")
    iterator = iter(loader)
    total_steps = int(conf["train"]["steps"])
    accum = int(conf["train"]["grad_accum"])
    start = time.monotonic()
    diffusion.train()
    print(f"DDPM train: {len(dataset)} train-only cases; "+
          f"image_size={conf['data']['image_size']} device={device} "
          f"steps={step}->{total_steps} effective_batch={int(conf['train']['batch_size'])*accum}", flush=True)
    for iteration in range(step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total, image_loss, mask_loss = 0.0, 0.0, 0.0
        for _ in range(accum):
            try:
                image, mask = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                image, mask = next(iterator)
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                losses = diffusion.training_losses(image, mask)
                loss = losses["loss"] / accum
            scaler.scale(loss).backward()
            total += float(losses["loss"].detach()) / accum
            image_loss += float(losses["loss_img"].detach()) / accum
            mask_loss += float(losses["loss_mask"].detach()) / accum
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), float(conf["train"]["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        if iteration % int(conf["train"]["ema_update_every"]) == 0:
            decay = float(conf["train"]["ema_decay"])
            with torch.no_grad():
                for target, source in zip(ema_model.parameters(), diffusion.model.parameters()):
                    target.lerp_(source, 1 - decay)
                for target, source in zip(ema_model.buffers(), diffusion.model.buffers()):
                    target.copy_(source)
        if iteration % int(conf["train"]["log_every"]) == 0 or iteration == 1:
            row = {"step": iteration, "loss": total, "loss_img": image_loss,
                   "loss_mask": mask_loss, "elapsed_seconds": prior_elapsed+time.monotonic()-start}
            with (folder / "history.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(row) + "\n")
            print(f"[DDPM] step={iteration}/{total_steps} loss={total:.4f} "
                  f"img={image_loss:.4f} mask={mask_loss:.4f}", flush=True)
        if iteration % int(conf["train"]["save_every"]) == 0 or iteration == total_steps:
            saved = save_checkpoint(folder, iteration, diffusion, ema_model, optimizer, scaler,
                                    conf, train_hash, prior_elapsed+time.monotonic()-start)
            print(f"DDPM checkpoint: {saved}", flush=True)
    return folder / "latest.pt"


@torch.inference_mode()
def sample_generator(checkpoint: str | Path, *, output: str | Path,
                     num_samples: int = 1177, batch_size: int = 8,
                     seed: int = 2026, device_name: str = "auto",
                     resume_output: bool = False) -> dict:
    """Samples immutable paired images and raw logits; rejects mixed runs."""
    if num_samples < 1 or batch_size < 1:
        raise ValueError("num_samples and batch_size must be positive")
    checkpoint = Path(checkpoint).resolve()
    ckpt = load_checkpoint(checkpoint)
    conf = ckpt["config"]
    device = choose_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    diffusion = build_diffusion(conf).to(device)
    diffusion.model.load_state_dict(ckpt["ema_model"], strict=True)
    diffusion.eval()
    output = Path(output)
    image_folder = output / "images"
    logit_folder = output / "logits_npy"
    hard_folder = output / "hard_masks"
    metadata_path = output / "provenance.json"
    checkpoint_sha = digest_file(checkpoint)
    provenance = {"schema": "sodar.synthetic.v1", "checkpoint_sha256": checkpoint_sha,
                  "checkpoint_step": ckpt["step"], "train_records_sha256": ckpt["train_records_sha256"],
                  "source_split": "train", "generator_config": conf,
                  "num_samples": num_samples, "batch_size": batch_size, "seed": seed,
                  "uses_ema": True, "synthetic_format": "logits",
                  "sampling_timesteps": conf["diffusion"]["sampling_timesteps"]}
    files_exist = output.exists() and any(output.rglob("*.npy"))
    if metadata_path.exists():
        if not resume_output:
            raise FileExistsError(f"{output} already contains generated samples; use a new output path")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field, value in provenance.items():
            if existing.get(field) != value:
                raise ValueError(f"--resume-output rejected: provenance.{field} does not match")
    elif files_exist or (output.exists() and (image_folder.exists() or logit_folder.exists())):
        raise FileExistsError(f"{output} contains untracked data; refusing to mix synthetic samples")
    for folder in (image_folder, logit_folder, hard_folder):
        folder.mkdir(parents=True, exist_ok=True)
    if not metadata_path.exists():
        save_json(metadata_path, {**provenance, "status": "incomplete"})
    for start in range(0, num_samples, batch_size):
        count = min(batch_size, num_samples-start)
        required = [(image_folder / f"{idx:06d}.jpg", logit_folder / f"{idx:06d}.npy",
                     hard_folder / f"{idx:06d}.png") for idx in range(start, start+count)]
        if all(all(p.is_file() for p in triple) for triple in required):
            continue
        if any(any(p.exists() for p in triple) for triple in required):
            raise FileExistsError("Partial output batch exists; move incomplete batch files aside "
                                  "before --resume-output. No files were overwritten.")
        set_seed(seed+start)  # deterministic per-batch resume
        images, logits = diffusion.sample(count)
        images = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        logits = logits.float().cpu().numpy()
        for i, (image_path, logit_path, mask_path) in enumerate(required):
            Image.fromarray(images[i]).save(image_path, format="JPEG", quality=95)
            np.save(logit_path, logits[i].astype(np.float32))
            Image.fromarray(logits[i].argmax(axis=0).astype(np.uint8) * 255).save(mask_path)
        print(f"[DDIM] exported {start+count}/{num_samples}", flush=True)
    save_json(metadata_path, {**provenance, "status": "complete",
                              "finished_utc": datetime.now(timezone.utc).isoformat()})
    return {"output": str(output), "count": num_samples, "checkpoint_step": ckpt["step"],
            "train_records_sha256": ckpt["train_records_sha256"]}
