"""Triage CNN (5 ICDR grades) and Grad-CAM explanations.

Architectures: MobileNetV3-Large (default; PLAN.md), MobileNetV3-Small,
ResNet-50, and a tiny CNN used only by the software tests. ImageNet weights
come from torchvision when `pretrained=True` (downloaded on first use).

Grad-CAM (Selvaraju et al. 2017) shows which regions raised the chosen output,
here the log-odds of referable DR. It is a rough localisation aid for
inspecting the model, not a lesion detector, and it can look plausible even
for a wrong prediction.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..datasets.records import REFERABLE_MIN_GRADE

ARCHS = ("mobilenet_v3_large", "mobilenet_v3_small", "resnet50", "tiny")


class TinyCNN(nn.Module):
    """Four stride-2 conv blocks + global pooling: fast on CPU, for tests only."""

    def __init__(self, n_classes: int = 5, width: int = 24):
        super().__init__()
        layers, c = [], 3
        for w in (width, 2 * width, 4 * width, 4 * width):
            layers += [nn.Conv2d(c, w, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(c, n_classes)

    def forward(self, x):
        return self.classifier(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


def build_model(arch: str = "mobilenet_v3_large", n_classes: int = 5, pretrained: bool = False) -> nn.Module:
    if arch == "tiny":
        return TinyCNN(n_classes)
    import torchvision.models as tvm

    if arch in ("mobilenet_v3_large", "mobilenet_v3_small"):
        ctor = getattr(tvm, arch)
        weights = None
        if pretrained:
            enum = tvm.MobileNet_V3_Large_Weights if arch.endswith("large") else tvm.MobileNet_V3_Small_Weights
            weights = enum.IMAGENET1K_V1 if arch.endswith("small") else enum.IMAGENET1K_V2
        m = ctor(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_classes)
        return m
    if arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    raise ValueError(f"unknown architecture {arch!r}; choose from {ARCHS}")


def feature_module(model: nn.Module) -> nn.Module:
    """The last module with a spatial output (Grad-CAM is taken there)."""
    if hasattr(model, "layer4"):
        return model.layer4
    if hasattr(model, "features"):
        return model.features
    raise ValueError("model has no known feature module")


def referable_score(logits: torch.Tensor) -> torch.Tensor:
    """log P(grade >= 2) - log P(grade < 2), per image."""
    return (torch.logsumexp(logits[:, REFERABLE_MIN_GRADE:], 1) - torch.logsumexp(logits[:, :REFERABLE_MIN_GRADE], 1))


def grad_cam(model: nn.Module, x: torch.Tensor, target: int | str = "referable",
             module: nn.Module | None = None) -> np.ndarray:
    """Heat maps (B, H, W) in [0, 1] for `target` = a grade index or "referable"."""
    module = module or feature_module(model)
    store: dict = {}

    def hook(_, __, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = module.register_forward_hook(hook)
    was_training = model.training
    model.eval()
    try:
        x = x.detach().clone().requires_grad_(True)
        logits = model(x)
        score = referable_score(logits) if target == "referable" else logits[:, int(target)]
        model.zero_grad(set_to_none=True)
        score.sum().backward()
    finally:
        h.remove()
        model.train(was_training)
    act, grad = store["act"].detach(), store["grad"].detach()
    cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    peak = cam.flatten(1).amax(1).clamp_min(1e-12).view(-1, 1, 1)
    return (cam / peak).cpu().numpy()
