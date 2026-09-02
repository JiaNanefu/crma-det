"""Lightweight receptive-field block for P4 CCFM refinement."""

import torch
import torch.nn as nn

from .utils import get_activation


__all__ = ["P4RFBLite"]


class _ConvBNAct(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dilation=1, act="silu"):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        padding = dilation if kernel_size == 3 else 0
        self.conv = nn.Conv2d(
            int(ch_in),
            int(ch_out),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(int(ch_out))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class P4RFBLite(nn.Module):
    """Multi-dilation residual refinement for the final P4 encoder feature.

    The residual gate is initialized to zero, so enabling this module is
    numerically equivalent to the original encoder at initialization.
    """

    def __init__(
        self,
        in_channels=256,
        branch_channels=64,
        dilations=(1, 3, 5),
        beta_init=0.0,
        act="silu",
    ):
        super().__init__()
        in_channels = int(in_channels)
        branch_channels = int(branch_channels)
        dilations = [int(d) for d in dilations]
        if in_channels <= 0:
            raise ValueError(f"P4RFBLite in_channels must be positive, got {in_channels}.")
        if branch_channels <= 0:
            raise ValueError(f"P4RFBLite branch_channels must be positive, got {branch_channels}.")
        if not dilations:
            raise ValueError("P4RFBLite requires at least one dilation value.")
        if any(d <= 0 for d in dilations):
            raise ValueError(f"P4RFBLite dilations must be positive, got {dilations}.")

        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.dilations = dilations
        self.branches = nn.ModuleList([
            nn.Sequential(
                _ConvBNAct(in_channels, branch_channels, 1, act=act),
                _ConvBNAct(branch_channels, branch_channels, 3, dilation=d, act=act),
            )
            for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * len(dilations), in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"P4RFBLite expected {self.in_channels} channels, got {x.shape[1]}."
            )
        rfb = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        beta = self.beta.reshape(1, 1, 1, 1).to(dtype=x.dtype)
        return x + beta * rfb
