"""U-Net (Ronneberger et al., MICCAI 2015) with three sigmoid outputs.

Output channels: 0 = vessel, 1 = artery, 2 = vein. Artery and vein are
separate (multi-label) outputs, so a crossing pixel can be both, which keeps
both trees continuous through crossings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

OUT_CHANNELS = 3
VESSEL, ARTERY, VEIN = 0, 1, 2


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 4
    out_channels: int = OUT_CHANNELS
    base: int = 32          # channels at full resolution; doubles at each level
    depth: int = 4          # number of down-sampling steps; inputs must be divisible by 2**depth
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int, dropout: float):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, config: UNetConfig = UNetConfig()):
        super().__init__()
        self.config = config
        ch = [config.base * 2**k for k in range(config.depth + 1)]
        self.down = nn.ModuleList([_Block(config.in_channels, ch[0], 0.0)] +
                                  [_Block(ch[k - 1], ch[k], config.dropout) for k in range(1, config.depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[k], ch[k - 1], 2, stride=2)
                                 for k in range(config.depth, 0, -1)])
        self.dec = nn.ModuleList([_Block(2 * ch[k - 1], ch[k - 1], 0.0) for k in range(config.depth, 0, -1)])
        self.head = nn.Conv2d(ch[0], config.out_channels, 1)

    @property
    def multiple(self) -> int:
        return 2**self.config.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.multiple or x.shape[-2] % self.multiple:
            raise ValueError(f"input height and width must be divisible by {self.multiple}")
        skips = []
        for k, block in enumerate(self.down):
            x = block(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)  # logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
