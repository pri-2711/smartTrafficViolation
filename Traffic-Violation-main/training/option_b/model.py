"""Option B network: MobileNetV3-Small backbone -> PAN-lite neck -> decoupled anchor-free head.

Outputs one (cls, reg, obj) tuple per level, for strides 8 / 16 / 32.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

STRIDES = (8, 16, 32)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=1, s=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )


class MobileNetBackbone(nn.Module):
    """ImageNet-pretrained MobileNetV3-Small, tapped at strides 8 (24ch), 16 (48ch), 32 (96ch)."""

    TAP_IDS = (3, 8, 11)
    CHANNELS = (24, 48, 96)

    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        full = torchvision.models.mobilenet_v3_small(weights=weights).features
        self.features = full[: self.TAP_IDS[-1] + 1]

    def forward(self, x):
        taps = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.TAP_IDS:
                taps.append(x)
        return taps


class PANLite(nn.Module):
    """Top-down (FPN) then bottom-up (PAN) feature fusion, all at `width` channels."""

    def __init__(self, in_channels=MobileNetBackbone.CHANNELS, width=128):
        super().__init__()
        self.lateral = nn.ModuleList(ConvBNAct(c, width, 1) for c in in_channels)
        self.td4 = ConvBNAct(width, width, 3)
        self.td3 = ConvBNAct(width, width, 3)
        self.down3 = ConvBNAct(width, width, 3, 2)
        self.down4 = ConvBNAct(width, width, 3, 2)
        self.bu4 = ConvBNAct(width, width, 3)
        self.bu5 = ConvBNAct(width, width, 3)

    def forward(self, feats):
        l3, l4, l5 = [m(f) for m, f in zip(self.lateral, feats)]
        t4 = self.td4(l4 + F.interpolate(l5, size=l4.shape[-2:], mode="nearest"))
        t3 = self.td3(l3 + F.interpolate(t4, size=l3.shape[-2:], mode="nearest"))
        n4 = self.bu4(t4 + self.down3(t3))
        n5 = self.bu5(l5 + self.down4(n4))
        return [t3, n4, n5]


class DecoupledHead(nn.Module):
    """Separate classification and regression towers, shared across levels.

    reg predicts (left, top, right, bottom) distances in units of stride (FCOS-style).
    obj is the objectness score, read from the regression tower (YOLOX-style).
    """

    def __init__(self, num_classes, width=128, depth=2):
        super().__init__()
        self.stem = ConvBNAct(width, width, 1)
        self.cls_tower = nn.Sequential(*[ConvBNAct(width, width, 3) for _ in range(depth)])
        self.reg_tower = nn.Sequential(*[ConvBNAct(width, width, 3) for _ in range(depth)])
        self.cls_pred = nn.Conv2d(width, num_classes, 1)
        self.reg_pred = nn.Conv2d(width, 4, 1)
        self.obj_pred = nn.Conv2d(width, 1, 1)
        prior_bias = -math.log((1 - 0.01) / 0.01)  # start every score at p=0.01
        nn.init.constant_(self.cls_pred.bias, prior_bias)
        nn.init.constant_(self.obj_pred.bias, prior_bias)

    def forward(self, feats):
        outs = []
        for f in feats:
            x = self.stem(f)
            c = self.cls_tower(x)
            r = self.reg_tower(x)
            outs.append((self.cls_pred(c), self.reg_pred(r), self.obj_pred(r)))
        return outs


class OptionBDetector(nn.Module):
    def __init__(self, num_classes, pretrained=True, width=128):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = MobileNetBackbone(pretrained)
        self.neck = PANLite(MobileNetBackbone.CHANNELS, width)
        self.head = DecoupledHead(num_classes, width)

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))

    def freeze_backbone(self, frozen):
        for p in self.backbone.parameters():
            p.requires_grad_(not frozen)


def flatten_outputs(outputs):
    """Concatenate all levels into (B, N, C) tensors, N = total grid points over the three strides."""
    cls = torch.cat([c.permute(0, 2, 3, 1).reshape(c.shape[0], -1, c.shape[1]) for c, _, _ in outputs], 1)
    reg = torch.cat([r.permute(0, 2, 3, 1).reshape(r.shape[0], -1, 4) for _, r, _ in outputs], 1)
    obj = torch.cat([o.permute(0, 2, 3, 1).reshape(o.shape[0], -1) for _, _, o in outputs], 1)
    return cls, reg, obj
