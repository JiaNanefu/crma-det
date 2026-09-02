"""Activation utilities.

Centralized `get_activation` helper used across nn/zoo modules to avoid duplication.
"""
from __future__ import annotations

import torch.nn as nn


def get_activation(act: str | nn.Module | None, inplace: bool = True) -> nn.Module:
    """Return an activation module by name.

    Args:
        act: Activation name ('silu', 'relu', 'leaky_relu', 'gelu', 'identity'/None) or an nn.Module.
        inplace: Passed to in-place capable activations when applicable.
    """
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act

    act_l = str(act).lower()
    if act_l in ("identity", "none"):
        return nn.Identity()
    if act_l == "silu":
        return nn.SiLU()
    if act_l == "relu":
        return nn.ReLU(inplace=inplace)
    if act_l in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(inplace=inplace)
    if act_l == "gelu":
        return nn.GELU()

    raise ValueError(f"Unsupported activation: {act}")
