"""Positive bounded learnable synthetic weight and first-order hypergradient."""

import math

import torch
from torch import nn
from torch.func import functional_call

from .losses import hard_loss


class LambdaWeight(nn.Module):
    def __init__(self, weight_init: float, weight_max: float):
        super().__init__()
        frac = weight_init / weight_max
        self.alpha = nn.Parameter(torch.tensor(math.log(frac/(1-frac)), dtype=torch.float32))
        self.max_weight = float(weight_max)

    def forward(self):
        return self.max_weight * torch.sigmoid(self.alpha)


def meta_weight_step(model, weight: LambdaWeight, optimizer_meta,
                     loss_real, loss_syn, meta_batch: dict, conf: dict,
                     inner_lr: float) -> float:
    """FO one-step virtual update. g_r/g_s are stop-gradient; λ is differentiable.

    Meta batch comes from train, not the official val-150. Changes to model
    running stats are suppressed by eval mode in the virtual forward.
    """
    named = dict(model.named_parameters())
    params = list(named.values())
    g_real = torch.autograd.grad(loss_real, params, retain_graph=True, allow_unused=True)
    g_syn = torch.autograd.grad(loss_syn, params, retain_graph=True, allow_unused=True)
    lam = weight()
    updated = {}
    for (name, param), gr, gs in zip(named.items(), g_real, g_syn):
        dr = gr.detach() if gr is not None else torch.zeros_like(param)
        ds = gs.detach() if gs is not None else torch.zeros_like(param)
        updated[name] = param.detach() - inner_lr * (dr + lam * ds)
    was_training = model.training
    model.eval()
    try:
        virtual_logit = functional_call(model, updated, (meta_batch["image"],), strict=False)
        virtual_loss = hard_loss(virtual_logit, meta_batch["mask"], float(conf["train"]["dice_weight"]))
        optimizer_meta.zero_grad(set_to_none=True)
        virtual_loss.backward()
        optimizer_meta.step()
    finally:
        model.train(was_training)
    return float(virtual_loss.detach().item())
