"""Lightweight dynamic upsampling for CCFM feature fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DySample(nn.Module):
    """Dynamic sampling based 2x upsampler.

    The offset branch is zero-initialized by default, so the initial behavior is
    nearest-neighbor upsampling. Offsets are learned in source-pixel units and
    normalized before ``grid_sample``.
    """

    def __init__(
        self,
        channels: int,
        scale_factor: int = 2,
        groups: int = 4,
        init_zero: bool = True,
        offset_scale: float = 0.25,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ):
        super().__init__()
        channels = int(channels)
        scale_factor = int(scale_factor)
        groups = int(groups)
        if scale_factor != 2:
            raise ValueError(f"DySample currently supports scale_factor=2, got {scale_factor}.")
        if groups <= 0:
            raise ValueError(f"DySample groups must be positive, got {groups}.")
        if channels % groups != 0:
            raise ValueError(f"DySample groups={groups} must divide channels={channels}.")

        self.channels = channels
        self.scale_factor = scale_factor
        self.groups = groups
        self.offset_scale = float(offset_scale)
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)
        self.offset = nn.Conv2d(channels, 2 * groups * scale_factor * scale_factor, 1)
        if init_zero:
            nn.init.zeros_(self.offset.weight)
            nn.init.zeros_(self.offset.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        scale = self.scale_factor
        out_h, out_w = h * scale, w * scale

        offset = self.offset(x)
        offset = F.pixel_shuffle(offset, scale)
        offset = offset.reshape(b, self.groups, 2, out_h, out_w) * self.offset_scale

        grid = self._base_grid(x, out_h, out_w)
        norm_x = 2.0 / max(w, 1)
        norm_y = 2.0 / max(h, 1)
        offset_x = offset[:, :, 0] * norm_x
        offset_y = offset[:, :, 1] * norm_y
        offset_grid = torch.stack((offset_x, offset_y), dim=-1)
        grid = grid + offset_grid

        x = x.reshape(b * self.groups, c // self.groups, h, w)
        grid = grid.reshape(b * self.groups, out_h, out_w, 2)
        out = F.grid_sample(
            x,
            grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return out.reshape(b, c, out_h, out_w)

    def _base_grid(self, x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        _, _, h, w = x.shape
        scale = self.scale_factor
        dtype = x.dtype
        device = x.device

        y = torch.arange(out_h, device=device, dtype=dtype).div(scale).floor()
        x_coord = torch.arange(out_w, device=device, dtype=dtype).div(scale).floor()
        y = (y + 0.5) * (2.0 / h) - 1.0
        x_coord = (x_coord + 0.5) * (2.0 / w) - 1.0
        yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)
        return grid.reshape(1, 1, out_h, out_w, 2)
