"""LC-PGNet-Main backbone for Carya maturity detection.

LC-PGNet means Local-Contrast Partial-Ghost Network.  This implementation is
designed as a moderate-size RT-DETR backbone: smaller than PResNet18-vd, but
with expansion bottlenecks for stronger fine-grained maturity representation.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn as nn

from src.core import register
from .common import ConvNormLayer, FrozenBatchNorm2d, get_activation


__all__ = [
    "make_divisible",
    "ConvBN",
    "ConvBNAct",
    "ECALayer",
    "PartialConv3x3",
    "GhostConv",
    "LocalContrastEnhance",
    "LCPGBlock",
    "LCPGNet",
]


def make_divisible(value, divisor=8, min_value=None):
    """Round channel counts to a hardware-friendly multiple of ``divisor``."""

    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}.")
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm2d + activation, reusing ConvNormLayer when possible."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        act="relu",
        bias=False,
    ):
        super().__init__()
        if groups == 1:
            self.layer = ConvNormLayer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                bias=bias,
                act=act,
            )
        else:
            self.layer = nn.Sequential(OrderedDict([
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                    ),
                ),
                ("norm", nn.BatchNorm2d(out_channels)),
                ("act", get_activation(act)),
            ]))

    def forward(self, x):
        return self.layer(x)


class ConvBN(nn.Module):
    """Conv2d + BatchNorm2d without activation for residual projections."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super().__init__()
        if groups == 1:
            self.layer = ConvNormLayer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=padding,
                bias=bias,
                act=None,
            )
        else:
            self.layer = nn.Sequential(OrderedDict([
                (
                    "conv",
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride,
                        padding=padding,
                        groups=groups,
                        bias=bias,
                    ),
                ),
                ("norm", nn.BatchNorm2d(out_channels)),
            ]))

    def forward(self, x):
        return self.layer(x)


class ECALayer(nn.Module):
    """Efficient Channel Attention for lightweight channel recalibration."""

    def __init__(self, channels, k_size=3):
        super().__init__()
        if k_size % 2 == 0:
            raise ValueError(f"k_size must be odd, got {k_size}.")
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class PartialConv3x3(nn.Module):
    """Apply 3x3 spatial convolution to the first channel partition only."""

    def __init__(self, channels, n_div=4, act="relu"):
        super().__init__()
        if n_div <= 0:
            raise ValueError(f"n_div must be positive, got {n_div}.")
        self.dim_conv = max(1, channels // n_div)
        self.dim_untouched = channels - self.dim_conv
        self.partial_conv = ConvBNAct(
            self.dim_conv,
            self.dim_conv,
            kernel_size=3,
            stride=1,
            padding=1,
            act=act,
        )

    def forward(self, x):
        if self.dim_untouched == 0:
            return self.partial_conv(x)
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        return torch.cat((x1, x2), dim=1)


class GhostConv(nn.Module):
    """Generate intrinsic features with 1x1 conv and cheap ghost features with DWConv."""

    def __init__(self, in_channels, out_channels, ratio=2, act="relu", use_act=True):
        super().__init__()
        ratio = int(ratio)
        if ratio < 1:
            raise ValueError(f"ratio must be >= 1, got {ratio}.")
        self.out_channels = int(out_channels)
        init_channels = int(math.ceil(out_channels / ratio))
        ghost_channels = init_channels * (ratio - 1)

        if use_act:
            self.primary_conv = ConvBNAct(
                in_channels,
                init_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                act=act,
            )
        else:
            self.primary_conv = ConvBN(
                in_channels,
                init_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

        if ghost_channels > 0 and use_act:
            self.cheap_operation = ConvBNAct(
                init_channels,
                ghost_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=init_channels,
                act=act,
            )
        elif ghost_channels > 0:
            self.cheap_operation = ConvBN(
                init_channels,
                ghost_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=init_channels,
            )
        else:
            self.cheap_operation = nn.Identity()

    def forward(self, x):
        intrinsic = self.primary_conv(x)
        ghost = self.cheap_operation(intrinsic)
        out = torch.cat((intrinsic, ghost), dim=1) if ghost is not intrinsic else intrinsic
        return out[:, : self.out_channels, :, :]


class LocalContrastEnhance(nn.Module):
    """Feature-level local contrast enhancement for subtle peel spots and edges."""

    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.pool = nn.AvgPool2d(3, 1, 1, count_include_pad=False)

    def forward(self, x):
        detail = x - self.pool(x)
        return x + self.gamma * detail


class LCPGBlock(nn.Module):
    """Expansion bottleneck block: C -> hidden -> C with LC/PConv/Ghost modules."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        expand_ratio=4.0,
        n_div=4,
        ghost_ratio=2,
        act="relu",
        use_pconv=True,
        use_ghost=True,
        use_lce=True,
        use_eca=True,
    ):
        super().__init__()
        hidden_channels = make_divisible(out_channels * float(expand_ratio), 8)
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}.")
        self.hidden_channels = hidden_channels

        self.shortcut = (
            ConvBN(in_channels, out_channels, 1, stride=stride, padding=0)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

        self.expand = ConvBNAct(
            out_channels, hidden_channels, kernel_size=1, stride=1, padding=0, act=act
        )
        self.spatial = (
            PartialConv3x3(hidden_channels, n_div=n_div, act=act)
            if use_pconv
            else ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                act=act,
            )
        )
        self.lce = LocalContrastEnhance(hidden_channels) if use_lce else nn.Identity()
        self.eca = ECALayer(hidden_channels) if use_eca else nn.Identity()
        self.project = (
            GhostConv(
                hidden_channels,
                out_channels,
                ratio=ghost_ratio,
                act=act,
                use_act=False,
            )
            if use_ghost
            else ConvBN(
                hidden_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.act = get_activation(act)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.expand(shortcut)
        out = self.spatial(out)
        out = self.lce(out)
        out = self.eca(out)
        out = self.project(out)
        return self.act(out + shortcut)


@register
class LCPGNet(nn.Module):
    """LC-PGNet-Main backbone with RT-DETR-compatible multi-scale outputs."""

    def __init__(
        self,
        widths=[64, 128, 256, 512],
        depths=[2, 2, 2, 2],
        return_idx=[1, 2, 3],
        num_stages=4,
        expand_ratio=4.0,
        n_div=4,
        ghost_ratio=2,
        use_pconv=True,
        use_ghost=True,
        use_lce=True,
        lce_position="last",
        use_eca=True,
        act="relu",
        freeze_at=-1,
        freeze_norm=True,
        pretrained=False,
    ):
        super().__init__()
        if pretrained:
            raise NotImplementedError("LC-PGNet does not provide pretrained weights.")
        if len(widths) != 4 or len(depths) != 4:
            raise ValueError("LCPGNet expects four widths and four depths.")
        if not 1 <= int(num_stages) <= 4:
            raise ValueError(f"num_stages must be in [1, 4], got {num_stages}.")

        self.widths = [int(v) for v in widths]
        self.depths = [int(v) for v in depths]
        self.return_idx = list(return_idx)
        self.num_stages = int(num_stages)
        self.expand_ratio = float(expand_ratio)
        self.lce_position = str(lce_position).lower()
        self.out_strides_all = [4, 8, 16, 32][: self.num_stages]

        if self.lce_position not in {"all", "last", "none"}:
            raise ValueError(
                f"lce_position must be one of 'all', 'last', 'none', got {lce_position}."
            )

        for idx in self.return_idx:
            if idx < 0 or idx >= self.num_stages:
                raise ValueError(
                    f"return_idx must be within [0, {self.num_stages - 1}], got {return_idx}."
                )
        for depth in self.depths[: self.num_stages]:
            if depth <= 0:
                raise ValueError(f"all stage depths must be positive, got {depth}.")

        self.stem = nn.Sequential(OrderedDict([
            ("conv1_1", ConvBNAct(3, 32, 3, stride=2, padding=1, act=act)),
            ("conv1_2", ConvBNAct(32, 32, 3, stride=1, padding=1, act=act)),
            ("conv1_3", ConvBNAct(32, 64, 3, stride=1, padding=1, act=act)),
        ]))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stages = []
        in_channels = 64
        for stage_idx in range(self.num_stages):
            out_channels = self.widths[stage_idx]
            stride = 1 if stage_idx == 0 else 2
            stages.append(
                self._make_stage(
                    in_channels,
                    out_channels,
                    self.depths[stage_idx],
                    stride,
                    expand_ratio=self.expand_ratio,
                    n_div=n_div,
                    ghost_ratio=ghost_ratio,
                    act=act,
                    use_pconv=use_pconv,
                    use_ghost=use_ghost,
                    use_lce=use_lce,
                    lce_position=self.lce_position,
                    use_eca=use_eca,
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)

        self.out_channels = [self.widths[i] for i in self.return_idx]
        self.out_strides = [self.out_strides_all[i] for i in self.return_idx]

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(int(freeze_at), self.num_stages)):
                self._freeze_parameters(self.stages[i])
        if freeze_norm:
            self._freeze_norm(self)

    def _make_stage(
        self,
        in_channels,
        out_channels,
        depth,
        stride,
        expand_ratio,
        n_div,
        ghost_ratio,
        act,
        use_pconv,
        use_ghost,
        use_lce,
        lce_position,
        use_eca,
    ):
        blocks = []
        for block_idx in range(depth):
            if not use_lce or lce_position == "none":
                block_use_lce = False
            elif lce_position == "all":
                block_use_lce = True
            else:
                block_use_lce = block_idx == depth - 1

            blocks.append(
                LCPGBlock(
                    in_channels if block_idx == 0 else out_channels,
                    out_channels,
                    stride=stride if block_idx == 0 else 1,
                    expand_ratio=expand_ratio,
                    n_div=n_div,
                    ghost_ratio=ghost_ratio,
                    act=act,
                    use_pconv=use_pconv,
                    use_ghost=use_ghost,
                    use_lce=block_use_lce,
                    use_eca=use_eca,
                )
            )
        return nn.Sequential(*blocks)

    def _freeze_parameters(self, module: nn.Module):
        for p in module.parameters():
            p.requires_grad = False

    def _freeze_norm(self, module: nn.Module):
        if isinstance(module, nn.BatchNorm2d):
            module = FrozenBatchNorm2d(module.num_features)
        else:
            for name, child in module.named_children():
                frozen_child = self._freeze_norm(child)
                if frozen_child is not child:
                    setattr(module, name, frozen_child)
        return module

    def forward(self, x):
        x = self.pool(self.stem(x))
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
