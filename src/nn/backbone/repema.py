"""RepEMA output enhancement modules for PResNet backbones.

These modules are intentionally placed after PResNet feature outputs instead of
replacing residual blocks, so pretrained PResNet weights remain reusable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvNormLayer, get_activation

__all__ = ["RepConv", "RepResidualEnhance", "EMA", "RepEMAEnhance"]


class RepConv(nn.Module):
    """Re-parameterizable convolution block.

    Training graph:
        Conv3x3-BN + Conv1x1-BN + Identity-BN

    Deploy graph:
        one fused Conv3x3 with bias
    """

    def __init__(self, channels: int, act: str | None = "relu", deploy: bool = False):
        super().__init__()
        self.channels = int(channels)
        self.deploy = bool(deploy)
        self.act = nn.Identity() if act is None else get_activation(act)
        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=True
            )
        else:
            self.rbr_dense = ConvNormLayer(self.channels, self.channels, 3, 1, act=None)
            self.rbr_1x1 = ConvNormLayer(self.channels, self.channels, 1, 1, padding=0, act=None)
            self.rbr_identity = nn.BatchNorm2d(self.channels)
            nn.init.zeros_(self.rbr_1x1.norm.weight)
            nn.init.zeros_(self.rbr_identity.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)
        return self.act(out)

    def switch_to_deploy(self):
        """Fuse training branches into one deploy-time 3x3 convolution."""
        if self.deploy:
            return self

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for name in ("rbr_dense", "rbr_1x1", "rbr_identity"):
            if hasattr(self, name):
                delattr(self, name)
        self.deploy = True
        return self

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_conv_bn(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_conv_bn(self.rbr_1x1)
        kernelid, biasid = self._fuse_identity_bn(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3(kernel: torch.Tensor) -> torch.Tensor:
        return F.pad(kernel, [1, 1, 1, 1])

    @staticmethod
    def _fuse_conv_bn(branch: ConvNormLayer):
        conv = branch.conv
        bn = branch.norm
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        bias = beta - running_mean * gamma / std
        return kernel * scale, bias

    def _fuse_identity_bn(self, branch: nn.BatchNorm2d):
        input_dim = self.channels
        kernel = torch.zeros(
            (input_dim, input_dim, 3, 3), dtype=branch.weight.dtype, device=branch.weight.device
        )
        for i in range(input_dim):
            kernel[i, i, 1, 1] = 1.0
        running_mean = branch.running_mean
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        bias = beta - running_mean * gamma / std
        return kernel * scale, bias


class RepResidualEnhance(nn.Module):
    """Residual output enhancement: out = x + gamma * RepConv(x)."""

    def __init__(self, channels: int, init_scale: float = 0.1, act: str = "relu", deploy: bool = False):
        super().__init__()
        self.rep = RepConv(channels, act=act, deploy=deploy)
        self.gamma = nn.Parameter(torch.ones(1) * float(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.rep(x)


class EMA(nn.Module):
    """Grouped Efficient Multi-scale Attention for spatial feature refinement."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        channels = int(channels)
        groups = int(groups)
        if groups <= 0:
            raise ValueError(f"EMA groups must be positive, got {groups}.")
        if channels % groups != 0:
            raise ValueError(f"EMA groups={groups} must divide channels={channels}.")

        self.channels = channels
        self.groups = groups
        group_channels = channels // groups
        self.conv1x1 = nn.Conv2d(group_channels, group_channels, kernel_size=1, bias=True)
        self.conv3x3 = nn.Conv2d(group_channels, group_channels, kernel_size=3, padding=1, bias=True)
        self.group_norm = nn.GroupNorm(1, group_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        group_x = x.reshape(b * self.groups, c // self.groups, h, w)

        x_h = F.adaptive_avg_pool2d(group_x, (h, 1))
        x_w = F.adaptive_avg_pool2d(group_x, (1, w)).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.group_norm(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)

        x11 = self.softmax(F.adaptive_avg_pool2d(x1, 1).reshape(b * self.groups, 1, -1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(F.adaptive_avg_pool2d(x2, 1).reshape(b * self.groups, 1, -1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class RepEMAEnhance(nn.Module):
    """Configurable RepResidualEnhance followed by EMA attention."""

    def __init__(
        self,
        channels: int,
        rep_enable: bool = True,
        ema_enable: bool = True,
        rep_init_scale: float = 0.1,
        rep_deploy: bool = False,
        ema_groups: int = 8,
        ema_init_scale: float = 0.1,
        act: str = "relu",
    ):
        super().__init__()
        self.ema_enable = bool(ema_enable)
        self.rep = (
            RepResidualEnhance(channels, init_scale=rep_init_scale, act=act, deploy=rep_deploy)
            if rep_enable
            else nn.Identity()
        )
        self.ema = EMA(channels, groups=ema_groups) if ema_enable else nn.Identity()
        self.ema_gamma = (
            nn.Parameter(torch.ones(1) * float(ema_init_scale)) if ema_enable else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rep(x)
        if self.ema_enable:
            attn = self.ema(out)
            out = out + self.ema_gamma * (attn - out)
        return out
