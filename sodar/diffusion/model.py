"""PyTorch-only port of the joint 5-channel U-Net pattern in us-dakter.ipynb.

Uses RMSNorm, FiLM-conditioned residual blocks, spatial linear attention at
high resolutions and full attention at the bottleneck. No TensorFlow/einops.
This is a clean reimplementation, not checkpoint-compatible with the notebook.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def norm_groups(channels: int) -> int:
    return math.gcd(8, channels)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("Timestep embedding dim must be even and >=4")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        pos = t.float()[:, None] * freqs[None]
        return torch.cat((pos.sin(), pos.cos()), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = channels**0.5
        self.g = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.g


class Block(nn.Module):
    def __init__(self, dim: int, out_dim: int, dropout: float = 0):
        super().__init__()
        self.proj = nn.Conv2d(dim, out_dim, 3, padding=1)
        self.norm = RMSNorm(out_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (1 + scale) + shift
        return self.dropout(self.act(x))


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, time_dim: int, dropout: float = 0):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_dim * 2))
        self.block1 = Block(dim, out_dim, dropout)
        self.block2 = Block(out_dim, out_dim)
        self.res = nn.Conv2d(dim, out_dim, 1) if dim != out_dim else nn.Identity()

    def forward(self, x, time_emb):
        scale_shift = self.mlp(time_emb)[:, :, None, None].chunk(2, dim=1)
        return self.block2(self.block1(x, scale_shift)) + self.res(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, *, heads: int = 4, dim_head: int = 32):
        super().__init__()
        hidden = heads * dim_head
        self.heads, self.dim_head = heads, dim_head
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, 4))
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, _, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [z.reshape(b, self.heads, self.dim_head, h*w) for z in qkv]
        k = torch.cat((self.mem_kv[0][None].expand(b, -1, -1, -1), k), dim=-1)
        v = torch.cat((self.mem_kv[1][None].expand(b, -1, -1, -1), v), dim=-1)
        q = q.softmax(dim=-2) * self.dim_head**-0.5
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        return self.to_out(out.reshape(b, self.heads*self.dim_head, h, w))


class FullAttention(nn.Module):
    def __init__(self, dim: int, *, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        hidden = heads * dim_head
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, 4, dim_head))
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        qkv = self.to_qkv(self.norm(x)).chunk(3, dim=1)
        q, k, v = [z.reshape(b, self.heads, self.dim_head, h*w).transpose(2, 3)
                   for z in qkv]
        k = torch.cat((self.mem_kv[0][None].expand(b, -1, -1, -1), k), dim=2)
        v = torch.cat((self.mem_kv[1][None].expand(b, -1, -1, -1), v), dim=2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.to_out(out.transpose(2, 3).reshape(b, self.heads*self.dim_head, h, w))


class Downsample(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim * 4, out_dim, 1)

    def forward(self, x):
        return self.conv(F.pixel_unshuffle(x, 2))


class Upsample(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, out_dim, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class JointUNet(nn.Module):
    """Image(3) + foreground/background mask(2) diffusion denoiser.

    The final two channels are unconstrained classification logits and should
    never be pre-softmaxed when exported as SoDAR synthetic supervision.
    """

    def __init__(self, dim: int = 64, dim_mults=(1, 2, 4, 8),
                 heads: int = 4, dim_head: int = 32, dropout: float = 0):
        super().__init__()
        dim_mults = tuple(int(v) for v in dim_mults)
        if dim < 8 or dim % 2 or not dim_mults or any(v < 1 for v in dim_mults):
            raise ValueError("dim must be even >=8 and dim_mults must be nonempty positive integers")
        self.img_channels = 3
        self.mask_channels = 2
        self.channels = 5
        self.self_condition = False
        self.downsample_factor = 2**(len(dim_mults)-1)
        self.init_conv = nn.Conv2d(5, dim, 7, padding=3)
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, time_dim),
                                      nn.GELU(), nn.Linear(time_dim, time_dim))
        dims = [dim, *[dim * m for m in dim_mults]]
        pairs = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for idx, (inc, outc) in enumerate(pairs):
            attention = FullAttention if idx == len(pairs)-1 else LinearAttention
            self.downs.append(nn.ModuleList([
                ResnetBlock(inc, inc, time_dim, dropout),
                ResnetBlock(inc, inc, time_dim, dropout),
                attention(inc, heads=heads, dim_head=dim_head),
                nn.Conv2d(inc, outc, 3, padding=1) if idx == len(pairs)-1
                else Downsample(inc, outc),
            ]))
        mid = dims[-1]
        self.mid_block1 = ResnetBlock(mid, mid, time_dim, dropout)
        self.mid_attn = FullAttention(mid, heads=heads, dim_head=dim_head)
        self.mid_block2 = ResnetBlock(mid, mid, time_dim, dropout)
        for idx, (inc, outc) in enumerate(reversed(pairs)):
            attention = FullAttention if idx == 0 else LinearAttention
            self.ups.append(nn.ModuleList([
                ResnetBlock(outc+inc, outc, time_dim, dropout),
                ResnetBlock(outc+inc, outc, time_dim, dropout),
                attention(outc, heads=heads, dim_head=dim_head),
                nn.Conv2d(outc, inc, 3, padding=1) if idx == len(pairs)-1
                else Upsample(outc, inc),
            ]))
        self.final_block = ResnetBlock(dim*2, dim, time_dim, dropout)
        self.final_conv = nn.Conv2d(dim, 5, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor):
        if x.shape[1] != 5 or x.shape[-1] % self.downsample_factor or x.shape[-2] % self.downsample_factor:
            raise ValueError(f"Expected (B,5,H,W) with dimensions divisible by {self.downsample_factor}")
        x = self.init_conv(x)
        residual = x
        emb = self.time_mlp(timestep)
        skips = []
        for block1, block2, attn, down in self.downs:
            x = block1(x, emb)
            skips.append(x)
            x = block2(x, emb)
            x = attn(x) + x
            skips.append(x)
            x = down(x)
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, emb)
        for block1, block2, attn, up in self.ups:
            x = block1(torch.cat((x, skips.pop()), dim=1), emb)
            x = block2(torch.cat((x, skips.pop()), dim=1), emb)
            x = attn(x) + x
            x = up(x)
        x = self.final_block(torch.cat((x, residual), dim=1), emb)
        return self.final_conv(x)
