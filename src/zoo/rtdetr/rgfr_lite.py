"""RGFR-Lite: Residual Gated Feature Refinement Lite.

A lightweight post-hoc refinement module placed after the Hybrid Encoder.
For each output scale, it fuses adjacent-scale features through a gated
residual path:

    F_out = F + gamma * gate * delta

where delta is the compensation signal from neighbouring scales and gate is
a sigmoid-gated weighting.  The learnable gamma is initialised to 0 so the
module acts as identity at the start of training.

This module does NOT replace the CCFM or the Hybrid Encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register

__all__ = ["RGFRLite"]


# ---------------------------------------------------------------------------
#  Tiny Conv-BN-Act helper
# ---------------------------------------------------------------------------

class _ConvBNAct(nn.Module):
    """Conv2d -> BatchNorm2d -> SiLU"""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride,
            padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
#  RGFR-Lite
# ---------------------------------------------------------------------------

@register
class RGFRLite(nn.Module):
    """Residual Gated Feature Refinement Lite.

    Args:
        in_channels:  input channels of the three feature levels.
        out_channels: uniform output channel count.
        gamma_init:   initial value of the learnable residual scale.
    """

    def __init__(self, in_channels=(256, 256, 256), out_channels=256, gamma_init=0.0):
        super().__init__()
        assert len(in_channels) == 3, "RGFRLite expects exactly 3 input levels."

        # ---- channel projection (skip when in==out) ----
        self.proj = nn.ModuleList([
            _ConvBNAct(c, out_channels, 1) if c != out_channels else nn.Identity()
            for c in in_channels
        ])

        # ---- F8 branch (receives compensation from F16) ----
        self.fuse8 = _ConvBNAct(out_channels * 2, out_channels, 1)
        self.gate8 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # ---- F16 branch (receives compensation from F8 and F32) ----
        self.fuse16 = _ConvBNAct(out_channels * 3, out_channels, 1)
        self.gate16 = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # ---- F32 branch (receives compensation from F16) ----
        self.fuse32 = _ConvBNAct(out_channels * 2, out_channels, 1)
        self.gate32 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=True),
            nn.Sigmoid(),
        )

        # ---- learnable residual scale ----
        self.gamma = nn.Parameter(torch.ones(3) * float(gamma_init))

    # ------------------------------------------------------------------
    #  Resize helper
    # ------------------------------------------------------------------

    @staticmethod
    def _resize(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)) and len(feats) == 3

        f8, f16, f32 = feats

        # 1.  unify channels
        f8  = self.proj[0](f8)
        f16 = self.proj[1](f16)
        f32 = self.proj[2](f32)

        # 2.  F8 ← F16  (semantic compensation for high-resolution detail)
        f16_to8 = self._resize(f16, f8)
        cat8 = torch.cat([f8, f16_to8], dim=1)
        delta8 = self.fuse8(cat8)
        gate8  = self.gate8(cat8)
        out8   = f8 + self.gamma[0] * gate8 * delta8

        # 3.  F16 ← F8 + F32  (detail from above, semantics from below)
        f8_to16  = self._resize(f8,  f16)
        f32_to16 = self._resize(f32, f16)
        cat16 = torch.cat([f16, f8_to16, f32_to16], dim=1)
        delta16 = self.fuse16(cat16)
        gate16  = self.gate16(cat16)
        out16   = f16 + self.gamma[1] * gate16 * delta16

        # 4.  F32 ← F16  (structural compensation for semantic features)
        f16_to32 = self._resize(f16, f32)
        cat32 = torch.cat([f32, f16_to32], dim=1)
        delta32 = self.fuse32(cat32)
        gate32  = self.gate32(cat32)
        out32   = f32 + self.gamma[2] * gate32 * delta32

        return [out8, out16, out32]
