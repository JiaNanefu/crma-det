"""S2-guided detail fusion for PResNet feature outputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from .common import get_activation

__all__ = ["S2GDF"]


class S2GDF(nn.Module):
    """Downsample S2 and add it to S3 through a learnable zero-init gate."""

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 128,
        alpha_init: float = 0.0,
        norm: str = "bn",
        act: str = "silu",
    ):
        super().__init__()
        norm = str(norm).lower()
        if norm not in ("bn", "batchnorm", "batchnorm2d"):
            raise ValueError(f"S2GDF only supports BatchNorm2d, got norm={norm}.")

        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(int(out_channels))
        self.act = get_activation(act)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))

    def forward(self, s2: torch.Tensor, s3: torch.Tensor) -> torch.Tensor:
        s2_down = self.act(self.norm(self.conv(s2)))
        if s2_down.shape != s3.shape:
            raise ValueError(
                "S2GDF shape mismatch: "
                f"S2_down={tuple(s2_down.shape)} vs S3={tuple(s3.shape)}"
            )
        return s3 + self.alpha.to(dtype=s2_down.dtype) * s2_down
