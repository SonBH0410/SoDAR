"""Training and real-only validation. Test evaluation is a separate command."""

from datetime import datetime, timezone
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import save_config
from .data import RealDataset, SyntheticDataset
from .io import audit
from .losses import hard_loss, synthetic_loss
from .metrics import SegmentationMeter
from .models import build_model
from .weighting import LambdaWeight, meta_weight_step


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_for(conf: dict) -> torch.device:
    name = conf["train"]["device"]
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def build_loader(dataset, conf, *, train=False, synthetic=False, seed_offset=0):
    bs = conf["train"]["batch_size_syn"] if synthetic else conf["train"]["batch_size_real"]
    generator = torch.Generator().manual_seed(int(conf["train"]["seed"]) + seed_offset)
    return DataLoader(dataset, batch_size=int(bs), shuffle=train, drop_last=False,
                      num_workers=int(conf["train"]["num_workers"]),
                      pin_memory=conf["train"]["device"] in ("cuda", "auto"),
                      generator=generator)


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def next_cycled(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


@torch.inference_mode()
def assess(model, dataloader, device):
    meter = SegmentationMeter()
    model.eval()
    for batch in dataloader:
        batch = to_device(batch, device)
        logits = model(batch["image"])
        meter.update(logits, batch["mask"], batch["label"], batch["id"])
    return meter.compute()


def train(conf: dict) -> Path:
    seed = int(conf["train"]["seed"])
    set_seed(seed)
    has_syn = conf["synthetic"]["mode"] != "none"
    audit_result = audit(conf, include_synthetic=has_syn, check_all_arrays=False)
    if audit_result["patient_disjoint"] == "NO":
        raise ValueError("Patient overlap in manifest")
    device = device_for(conf)
    model = build_model(conf).to(device)
    dataset_train = RealDataset(conf, "train", augment=conf["augment"])
    dataset_val = RealDataset(conf, "val")
    loader_real = build_loader(dataset_train, conf, train=True)
    loader_val = build_loader(dataset_val, conf, seed_offset=1)
    loader_syn = None
    if has_syn:
        dataset_syn = SyntheticDataset(conf)
        loader_syn = build_loader(dataset_syn, conf, train=True, synthetic=True, seed_offset=2)

    weight_type = conf["synthetic"]["weighting"]
    weight = None
    optimizer_meta = None
    if has_syn:
        weight = LambdaWeight(conf["synthetic"]["weight_init"], conf["synthetic"]["weight_max"]).to(device)
        if weight_type == "meta":
            optimizer_meta = torch.optim.Adam([weight.alpha], lr=float(conf["synthetic"]["meta_lr"]))
    parameters = list(model.parameters())
    groups = [{"params": parameters}]
    if has_syn and weight_type == "naive":
        groups.append({"params": [weight.alpha], "lr": float(conf["synthetic"]["meta_lr"]), "weight_decay": 0})
    opt = AdamW(groups, lr=float(conf["train"]["lr"]), weight_decay=float(conf["train"]["weight_decay"]))
    scheduler = CosineAnnealingLR(opt, T_max=int(conf["train"]["epochs"]), eta_min=1e-6)
    # The meta-real minibatch is independently shuffled from train-850;
    # this is not a patient-disjoint meta-validation subset.
    loader_meta = None
    if has_syn and weight_type == "meta":
        loader_meta = build_loader(RealDataset(conf, "train", augment="none"), conf, train=True, seed_offset=3)

    run_dir = Path(conf["train"]["save_root"]) / f"{conf['experiment']}_{conf['model']['name']}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(conf, run_dir / "config.json")
    (run_dir / "audit.json").write_text(json.dumps(audit_result, indent=2), encoding="utf-8")
    history = []
    best = -1.0
    start = time.time()
    global_step = 0
    for epoch in range(1, int(conf["train"]["epochs"]) + 1):
        model.train()
        syn_iter = iter(loader_syn) if loader_syn else None
        meta_iter = iter(loader_meta) if loader_meta else None
        sums = {"real": 0.0, "syn": 0.0, "lambda": 0.0, "meta": 0.0, "meta_n": 0}
        for real_batch in loader_real:
            global_step += 1
            real_batch = to_device(real_batch, device)
            pred_real = model(real_batch["image"])
            loss_real = hard_loss(pred_real, real_batch["mask"], float(conf["train"]["dice_weight"]))
            loss_syn = None
            if has_syn:
                syn_batch, syn_iter = next_cycled(syn_iter, loader_syn)
                syn_batch = to_device(syn_batch, device)
                pred_syn = model(syn_batch["image"])
                loss_syn = synthetic_loss(pred_syn, syn_batch, conf)
                if weight_type == "meta" and global_step % int(conf["synthetic"]["meta_every"]) == 0:
                    meta_batch, meta_iter = next_cycled(meta_iter, loader_meta)
                    meta_batch = to_device(meta_batch, device)
                    meta_val = meta_weight_step(model, weight, optimizer_meta, loss_real, loss_syn,
                                                meta_batch, conf, inner_lr=float(opt.param_groups[0]["lr"]))
                    sums["meta"] += meta_val
                    sums["meta_n"] += 1
                lam = weight() if weight_type == "naive" else weight().detach()
                loss = loss_real + lam * loss_syn
                sums["syn"] += float(loss_syn.detach())
                sums["lambda"] += float(lam.detach())
            else:
                loss = loss_real
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(conf["train"]["grad_clip"]))
            opt.step()
            sums["real"] += float(loss_real.detach())
        scheduler.step()
        val = assess(model, loader_val, device)
        n = len(loader_real)
        row = {"epoch": epoch, "real_loss": sums["real"]/n,
               "syn_loss": sums["syn"]/n if has_syn else None,
               "lambda": sums["lambda"]/n if has_syn else None,
               "meta_real_loss": sums["meta"]/sums["meta_n"] if sums["meta_n"] else None,
               "validation": val, "elapsed_seconds": time.time()-start}
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if val["macro_dice"] > best:
            best = val["macro_dice"]
            torch.save({"model": model.state_dict(), "config": conf, "epoch": epoch,
                        "best_validation": val,
                        "weight_alpha": weight.alpha.detach().cpu() if weight is not None else None},
                       run_dir / "best.pt")
        print(f"[{conf['experiment']}/{conf['model']['name']}/seed{seed}] "
              f"epoch={epoch:03d} real_loss={row['real_loss']:.4f} "
              f"lambda={row['lambda']} val_macro_dice={val['macro_dice']:.4f}", flush=True)
    (run_dir / "summary.json").write_text(json.dumps({"best_validation_macro_dice": best,
                "wall_seconds": time.time()-start, "epochs": len(history),
                "experiment": conf["experiment"], "model": conf["model"]["name"], "seed": seed}, indent=2), encoding="utf-8")
    return run_dir
