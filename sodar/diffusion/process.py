"""Joint DDPM training loss and DDIM sampling following the old DAKTer variant.

The image head predicts x_0 (MSE). The segmentation head predicts two logits
(pixel cross-entropy). For the joint noising/denoising state, mask one-hot
channels are mapped to [-b,+b]. We do NOT apply a min-SNR loss-weight factor:
the provided notebook computed one but omitted it from its final loss.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def cosine_betas(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
    alphas = torch.cos(((t + offset) / (1 + offset)) * math.pi / 2).square()
    alphas = alphas / alphas[0]
    return (1 - alphas[1:] / alphas[:-1]).clamp(0, 0.999).float()


def extract(values: torch.Tensor, t: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values.gather(0, t).reshape(-1, *((1,) * (target.ndim - 1)))


class JointGaussianDiffusion(nn.Module):
    """Five-channel joint generative process with 3-channel image + mask logits."""

    def __init__(self, model: nn.Module, *, image_size: int = 256,
                 timesteps: int = 1000, sampling_timesteps: int = 200,
                 snr_balance_b: float = 0.684, ddim_eta: float = 0.0):
        super().__init__()
        if image_size < 8 or image_size % model.downsample_factor:
            raise ValueError(f"image_size must be >=8 and divisible by {model.downsample_factor}")
        if not (2 <= timesteps and 1 <= sampling_timesteps <= timesteps):
            raise ValueError("Require timesteps>=2 and 1<=sampling_timesteps<=timesteps")
        if not (0 < snr_balance_b <= 1) or ddim_eta < 0:
            raise ValueError("Require 0<snr_balance_b<=1 and ddim_eta>=0")
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.snr_balance_b = float(snr_balance_b)
        self.ddim_eta = float(ddim_eta)
        betas = cosine_betas(timesteps)
        alphas = (1 - betas).cumprod(dim=0)
        self.register_buffer("alphas_cumprod", alphas)
        self.register_buffer("sqrt_alphas_cumprod", alphas.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1 - alphas).sqrt())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (extract(self.sqrt_alphas_cumprod, t, x0) * x0
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise)

    def training_losses(self, img: torch.Tensor, mask: torch.Tensor,
                        t: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """img: float [B,3,H,W] in [0,1]; mask: float [B,1,H,W] in {0,1}."""
        if img.ndim != 4 or img.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected image shape (B,3,{self.image_size},{self.image_size})")
        if mask.shape != (len(img), 1, self.image_size, self.image_size):
            raise ValueError("Expected mask shape (B,1,H,W)")
        if t is None:
            t = torch.randint(0, self.timesteps, (len(img),), device=img.device)
        elif t.shape != (len(img),) or t.dtype != torch.long:
            raise ValueError("t must be long Tensor of shape (B,)")
        target_img = img * 2 - 1
        foreground = mask.clamp(0, 1)
        onehot = torch.cat((1 - foreground, foreground), dim=1)
        target_mask = (onehot * 2 - 1) * self.snr_balance_b
        x0 = torch.cat((target_img, target_mask), dim=1)
        xt = self.q_sample(x0, t, torch.randn_like(x0))
        prediction = self.model(xt, t)
        loss_img = F.mse_loss(prediction[:, :3], target_img)
        loss_mask = F.cross_entropy(prediction[:, 3:], foreground[:, 0].long())
        return {"loss": loss_img + loss_mask, "loss_img": loss_img,
                "loss_mask": loss_mask}

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.training_losses(img, mask)["loss"]

    @torch.inference_mode()
    def sample(self, batch_size: int = 1, *, progress: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Return images [0,1] and RAW two-channel mask logits.

        DDIM recursion uses an x0 mask built from two-channel softmax, whereas
        the returned teacher logits are the *unsoftmaxed final model output*.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.model.training:
            raise RuntimeError("Set the diffusion model to eval() before sampling")
        device = self.alphas_cumprod.device
        xt = torch.randn(batch_size, 5, self.image_size, self.image_size, device=device)
        times = torch.linspace(self.timesteps - 1, -1, self.sampling_timesteps + 1).long().tolist()
        pairs = list(zip(times[:-1], times[1:]))
        if progress:
            from tqdm.auto import tqdm
            pairs = tqdm(pairs, desc="DDIM", leave=False)
        final_logits = None
        for now, nxt in pairs:
            steps = torch.full((batch_size,), now, device=device, dtype=torch.long)
            pred = self.model(xt, steps)
            final_logits = pred[:, 3:].float()
            img_x0 = pred[:, :3].float().clamp(-1, 1)
            mask_x0 = (final_logits.softmax(dim=1) * 2 - 1) * self.snr_balance_b
            x0 = torch.cat((img_x0, mask_x0), dim=1)
            if nxt < 0:
                xt = x0
                break
            a = self.alphas_cumprod[now]
            anext = self.alphas_cumprod[nxt]
            pred_noise = (xt.float() - a.sqrt() * x0) / (1 - a).sqrt()
            sigma = self.ddim_eta * ((1-anext)/(1-a) * (1-a/anext)).clamp(min=0).sqrt()
            coefficient = (1 - anext - sigma.square()).clamp(min=0).sqrt()
            # Avoid consuming RNG state when deterministic eta=0.
            noise = torch.randn_like(xt) if self.ddim_eta else 0
            xt = anext.sqrt() * x0 + coefficient * pred_noise + sigma * noise
        return ((xt[:, :3] + 1) * 0.5).clamp(0, 1), final_logits
