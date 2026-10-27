"""Hard/soft binary segmentation objectives with consistent KD temperature."""

import math

import torch
from torch.nn import functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    prediction = torch.sigmoid(logits)
    if prediction.shape != target.shape:
        raise ValueError(f"Dice shapes mismatch: {prediction.shape} vs {target.shape}")
    dims = (1, 2, 3)
    overlap = (prediction * target).sum(dim=dims)
    union = prediction.sum(dim=dims) + target.sum(dim=dims)
    return (1 - (2 * overlap + smooth) / (union + smooth)).mean()


def hard_loss(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target) + dice_weight * dice_loss(logits, target)


def _synthetic_per_sample(logits: torch.Tensor, q: torch.Tensor, conf: dict) -> torch.Tensor:
    kind = conf["synthetic"]["soft_loss"]
    temperature = float(conf["synthetic"]["temperature"])
    if q.shape[1] != 2 or logits.shape[1] != 1 or q.shape[-2:] != logits.shape[-2:]:
        raise ValueError(f"Expected student B,1,H,W and teacher B,2,H,W; got {logits.shape}, {q.shape}")
    if kind == "kd":
        log_p_fg = F.logsigmoid(logits / temperature)
        log_p_bg = F.logsigmoid(-logits / temperature)
        student_log_probs = torch.cat([log_p_bg, log_p_fg], dim=1)
        # Nonnegative KL; teacher entropy is included for mathematically clean reporting.
        teacher_log_probs = q.clamp_min(1e-7).log()
        kl = (q * (teacher_log_probs - student_log_probs)).sum(dim=1).mean(dim=(1, 2))
        return temperature ** 2 * kl.clamp_min(0)
    q_fg = q[:, 1:2]
    soft_bce = F.binary_cross_entropy_with_logits(logits, q_fg, reduction="none").mean(dim=(1, 2, 3))
    if kind == "softbce":
        return soft_bce
    if kind == "softbce_dice":
        pred = torch.sigmoid(logits)
        intersection = (pred * q_fg).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + q_fg.sum(dim=(1, 2, 3))
        return soft_bce + float(conf["train"]["dice_weight"]) * (1 - (2*intersection+1)/(union+1))
    raise ValueError(f"Unknown soft loss: {kind}")


def synthetic_loss(logits: torch.Tensor, batch: dict, conf: dict) -> torch.Tensor:
    if conf["synthetic"]["mode"] == "hard":
        return hard_loss(logits, batch["hard"], float(conf["train"]["dice_weight"]))
    q = batch["prob"]
    per_sample = _synthetic_per_sample(logits, q, conf)
    if not conf["synthetic"].get("entropy_weighting", False):
        return per_sample.mean()
    entropy = -(q.clamp_min(1e-7) * q.clamp_min(1e-7).log()).sum(dim=1)
    entropy = entropy.mean(dim=(1, 2)) / math.log(2)
    weights = torch.exp(-float(conf["synthetic"]["entropy_gamma"]) * entropy)
    weights = weights.clamp(min=float(conf["synthetic"]["entropy_floor"]))
    return (weights.detach() * per_sample).sum() / weights.detach().sum().clamp_min(1e-7)
