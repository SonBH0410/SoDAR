"""Binary segmentation backbones; all return [B,1,H,W] logits."""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, inc: int, outc: int):
        super().__init__(
            nn.Conv2d(inc, outc, 3, padding=1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc, 3, padding=1, bias=False), nn.BatchNorm2d(outc), nn.ReLU(inplace=True),
        )


class Unet(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = ConvBlock(3, c)
        self.downs = nn.ModuleList([ConvBlock(c, 2*c), ConvBlock(2*c, 4*c),
                                    ConvBlock(4*c, 8*c), ConvBlock(8*c, 16*c)])
        self.ups = nn.ModuleList([ConvBlock(24*c, 8*c), ConvBlock(12*c, 4*c),
                                  ConvBlock(6*c, 2*c), ConvBlock(3*c, c)])
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(F.max_pool2d(skips[-1], 2)))
        out = skips.pop()
        for decoder in self.ups:
            skip = skips.pop()
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = decoder(torch.cat([out, skip], dim=1))
        return self.out(out)


class AttentionGate(nn.Module):
    def __init__(self, g_channels, s_channels, hidden):
        super().__init__()
        self.g = nn.Conv2d(g_channels, hidden, 1, bias=False)
        self.s = nn.Conv2d(s_channels, hidden, 1, bias=False)
        self.norm = nn.BatchNorm2d(hidden)
        self.phi = nn.Conv2d(hidden, 1, 1)

    def forward(self, g, skip):
        weight = torch.sigmoid(self.phi(F.relu(self.norm(self.g(g) + self.s(skip)))))
        return skip * weight


class SPPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(channels, channels//2, 1), nn.BatchNorm2d(channels//2), nn.SiLU())
        self.pool = nn.MaxPool2d(5, stride=1, padding=2)
        self.expand = nn.Sequential(nn.Conv2d(2*channels, channels, 1), nn.BatchNorm2d(channels), nn.SiLU())

    def forward(self, x):
        x1 = self.reduce(x)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        x4 = self.pool(x3)
        return self.expand(torch.cat([x1, x2, x3, x4], dim=1))


class AttnDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.gate = AttentionGate(in_channels, skip_channels, out_channels)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, self.gate(x, skip)], dim=1))


class SovaSegNet(nn.Module):
    """VGG16+SPPF+attention U-Net variant aligned to supplied research notebook.

    Not a claim of bitwise equivalence to the ICIP TensorFlow code.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16
        except ImportError as exc:
            raise ImportError("Install matching torch/torchvision for SovaSeg-Net") from exc
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = vgg16(weights=weights).features
        self.indices = (3, 8, 15, 22, 29)
        self.sppf = SPPF(512)
        self.decoders = nn.ModuleList([
            AttnDecoder(512, 512, 512), AttnDecoder(512, 256, 256),
            AttnDecoder(256, 128, 128), AttnDecoder(128, 64, 64),
        ])
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        features = []
        for index, module in enumerate(self.encoder):
            x = module(x)
            if index in self.indices:
                features.append(x)
        out = self.sppf(features[-1])
        for decoder, skip in zip(self.decoders, reversed(features[:-1])):
            out = decoder(out, skip)
        return self.out(out)


class SegformerB2(nn.Module):
    def __init__(self, pretrained: bool, pretrained_id: str):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("Install `pip install -e '.[segformer]'` for SegFormer-B2") from exc
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_id, num_labels=1, ignore_mismatched_sizes=True,
            )
        else:
            conf = SegformerConfig(num_labels=1, hidden_sizes=[64, 128, 320, 512],
                                   depths=[3, 4, 6, 3], num_attention_heads=[1, 2, 5, 8],
                                   sr_ratios=[8, 4, 2, 1], decoder_hidden_size=256)
            self.model = SegformerForSemanticSegmentation(conf)

    def forward(self, x):
        out = self.model(pixel_values=x).logits
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(conf: dict, *, pretrained: bool | None = None) -> nn.Module:
    name = conf["model"]["name"]
    init = conf["model"]["pretrained"] if pretrained is None else pretrained
    if name == "unet":
        return Unet(base_channels=int(conf["model"]["base_channels"]))
    if name == "sovasegnet":
        return SovaSegNet(pretrained=bool(init))
    if name == "segformer_b2":
        return SegformerB2(pretrained=bool(init), pretrained_id=conf["model"]["segformer_pretrained_id"])
    raise ValueError(f"Unknown model: {name}")
