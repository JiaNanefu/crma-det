"""Gated ECA channel recalibration for the final P4 encoder feature."""

import torch
import torch.nn as nn


__all__ = [
    "P4ECA",
    "GatedECAP4",
    "BalancedECAP4",
    "MeanPreservingECAP4",
    "EnhanceOnlyECAP4",
    "LeakyEnhanceECAP4",
]


class P4ECA(nn.Module):
    """ECA on final P4 with a zero-initialized residual gate.

    The default mode keeps the original gated residual behavior:
    out = x + beta * (eca(x) - x).

    The module is an exact identity when beta is initialized to 0.
    """

    def __init__(
        self,
        channels=256,
        kernel_size=5,
        beta_init=0.0,
        gated_residual=True,
    ):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        if channels <= 0:
            raise ValueError(f"P4ECA channels must be positive, got {channels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"P4ECA kernel_size must be a positive odd integer, got {kernel_size}.")
        if not bool(gated_residual):
            raise ValueError("P4ECA requires gated_residual=True for initial equivalence.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.gated_residual = bool(gated_residual)
        self.mode = "residual"
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.conv = nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        self.last_attention = None
        self.last_channel_scale = None

    def attention(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"P4ECA expected {self.channels} channels, got {x.shape[1]}.")
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y.transpose(1, 2).unsqueeze(-1)

    def forward(self, x):
        weight = self.attention(x)
        self.last_attention = weight.detach()
        beta = self.beta.reshape(1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        eca_x = x * weight
        self.last_channel_scale = (1.0 + beta * (weight - 1.0)).detach()
        return x + beta * (eca_x - x)


class BalancedECAP4(nn.Module):
    """Identity-initialized bidirectional ECA scaling for final P4.

    Positive beta_up amplifies high-attention channels, while positive
    beta_down suppresses low-attention channels. The tanh gates bound the
    scale to [1 - max_scale, 1 + max_scale], and beta_up=beta_down=0 keeps
    the module exactly equal to identity at initialization.
    """

    def __init__(
        self,
        channels=256,
        kernel_size=5,
        beta_init=0.0,
        gated_residual=True,
        max_scale=0.5,
    ):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        max_scale = float(max_scale)
        if channels <= 0:
            raise ValueError(f"BalancedECAP4 channels must be positive, got {channels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"BalancedECAP4 kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if not bool(gated_residual):
            raise ValueError("BalancedECAP4 requires gated_residual=True for initial equivalence.")
        if max_scale <= 0 or max_scale >= 1:
            raise ValueError(f"BalancedECAP4 max_scale must be in (0, 1), got {max_scale}.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.gated_residual = bool(gated_residual)
        self.mode = "balanced"
        self.max_scale = max_scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.conv = nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        init = torch.tensor(float(beta_init), dtype=torch.float32)
        self.beta_up = nn.Parameter(init.clone())
        self.beta_down = nn.Parameter(init.clone())
        self.last_attention = None
        self.last_channel_scale = None

    def attention(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"BalancedECAP4 expected {self.channels} channels, got {x.shape[1]}.")
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y.transpose(1, 2).unsqueeze(-1)

    def forward(self, x):
        weight = self.attention(x)
        self.last_attention = weight.detach()
        beta_up = self.beta_up.reshape(1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        beta_down = self.beta_down.reshape(1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        up = torch.tanh(beta_up) * self.max_scale
        down = torch.tanh(beta_down) * self.max_scale
        channel_scale = 1.0 + up * weight - down * (1.0 - weight)
        self.last_channel_scale = channel_scale.detach()
        return x * channel_scale


class MeanPreservingECAP4(nn.Module):
    """Identity-initialized mean-preserving ECA scaling for final P4.

    The ECA weights are normalized by their per-sample channel mean, so the
    module mainly redistributes channel responses while keeping the average
    P4 feature magnitude stable. beta=0 makes the module an exact identity.
    """

    def __init__(
        self,
        channels=256,
        kernel_size=5,
        beta_init=0.0,
        gated_residual=True,
        max_scale=0.25,
    ):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        max_scale = float(max_scale)
        if channels <= 0:
            raise ValueError(f"MeanPreservingECAP4 channels must be positive, got {channels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"MeanPreservingECAP4 kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if not bool(gated_residual):
            raise ValueError(
                "MeanPreservingECAP4 requires gated_residual=True for initial equivalence."
            )
        if max_scale <= 0 or max_scale >= 1:
            raise ValueError(f"MeanPreservingECAP4 max_scale must be in (0, 1), got {max_scale}.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.gated_residual = bool(gated_residual)
        self.mode = "mean_preserving"
        self.max_scale = max_scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.conv = nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        # Keep the trained ECA parameters available while allowing an
        # inference-only identity bypass for deployment experiments.
        self.inference_enabled = True
        self.last_attention = None
        self.last_channel_scale = None

    def attention(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"MeanPreservingECAP4 expected {self.channels} channels, got {x.shape[1]}."
            )
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y.transpose(1, 2).unsqueeze(-1)

    @torch.jit.unused
    def set_inference_enabled(self, enabled=True):
        """Enable/disable ECA computation only while the module is in eval mode."""
        self.inference_enabled = bool(enabled)
        return self

    def forward(self, x):
        if not self.training and not self.inference_enabled:
            return x
        weight = self.attention(x)
        self.last_attention = weight.detach()
        eps = torch.finfo(weight.dtype).eps
        weight_mean = weight.mean(dim=1, keepdim=True).clamp_min(eps)
        weight_norm = weight / weight_mean
        beta = self.beta.reshape(1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        gamma = torch.tanh(beta) * self.max_scale
        channel_scale = 1.0 + gamma * (weight_norm - 1.0)
        self.last_channel_scale = channel_scale.detach()
        return x * channel_scale


class EnhanceOnlyECAP4(nn.Module):
    """Identity-initialized ECA that only amplifies selected P4 channels.

    The ECA weights are normalized by their per-sample channel mean. Channels
    above that mean can be amplified, while channels at or below the mean keep
    scale 1.0. This avoids actively suppressing weak but useful transition
    cues for adjacent maturity classes.
    """

    def __init__(
        self,
        channels=256,
        kernel_size=5,
        beta_init=0.0,
        gated_residual=True,
        max_scale=0.2,
    ):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        max_scale = float(max_scale)
        if channels <= 0:
            raise ValueError(f"EnhanceOnlyECAP4 channels must be positive, got {channels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"EnhanceOnlyECAP4 kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if not bool(gated_residual):
            raise ValueError(
                "EnhanceOnlyECAP4 requires gated_residual=True for initial equivalence."
            )
        if max_scale <= 0 or max_scale >= 1:
            raise ValueError(f"EnhanceOnlyECAP4 max_scale must be in (0, 1), got {max_scale}.")

        self.channels = channels
        self.kernel_size = kernel_size
        self.gated_residual = bool(gated_residual)
        self.mode = "enhance_only"
        self.max_scale = max_scale
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.conv = nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        self.last_attention = None
        self.last_channel_scale = None

    def attention(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"EnhanceOnlyECAP4 expected {self.channels} channels, got {x.shape[1]}."
            )
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y.transpose(1, 2).unsqueeze(-1)

    def forward(self, x):
        weight = self.attention(x)
        self.last_attention = weight.detach()
        eps = torch.finfo(weight.dtype).eps
        weight_mean = weight.mean(dim=1, keepdim=True).clamp_min(eps)
        enhance_signal = torch.relu(weight / weight_mean - 1.0)
        beta = self.beta.reshape(1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        gamma = torch.clamp(torch.tanh(beta), min=0.0) * self.max_scale
        channel_scale = 1.0 + gamma * enhance_signal
        self.last_channel_scale = channel_scale.detach()
        return x * channel_scale


class LeakyEnhanceECAP4(nn.Module):
    """Identity-initialized ECA with a leaky gate for selected P4 channels.

    This keeps the enhance-only shape of the intervention: only channels whose
    ECA weight is above the per-sample channel mean are modulated. Unlike a hard
    positive clamp, the gate keeps a small negative slope when beta becomes
    negative, so the module can recover instead of getting stuck as identity.
    """

    def __init__(
        self,
        channels=256,
        kernel_size=5,
        beta_init=0.0,
        gated_residual=True,
        max_scale=0.2,
        negative_slope=0.05,
    ):
        super().__init__()
        channels = int(channels)
        kernel_size = int(kernel_size)
        max_scale = float(max_scale)
        negative_slope = float(negative_slope)
        if channels <= 0:
            raise ValueError(f"LeakyEnhanceECAP4 channels must be positive, got {channels}.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"LeakyEnhanceECAP4 kernel_size must be a positive odd integer, got {kernel_size}."
            )
        if not bool(gated_residual):
            raise ValueError(
                "LeakyEnhanceECAP4 requires gated_residual=True for initial equivalence."
            )
        if max_scale <= 0 or max_scale >= 1:
            raise ValueError(f"LeakyEnhanceECAP4 max_scale must be in (0, 1), got {max_scale}.")
        if negative_slope <= 0 or negative_slope >= 1:
            raise ValueError(
                f"LeakyEnhanceECAP4 negative_slope must be in (0, 1), got {negative_slope}."
            )

        self.channels = channels
        self.kernel_size = kernel_size
        self.gated_residual = bool(gated_residual)
        self.mode = "leaky_enhance"
        self.max_scale = max_scale
        self.negative_slope = negative_slope
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.conv = nn.Conv1d(
                1,
                1,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
                bias=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        self.beta = nn.Parameter(torch.tensor(float(beta_init), dtype=torch.float32))
        self.last_attention = None
        self.last_channel_scale = None

    def attention(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(
                f"LeakyEnhanceECAP4 expected {self.channels} channels, got {x.shape[1]}."
            )
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv(y)
        y = torch.sigmoid(y)
        return y.transpose(1, 2).unsqueeze(-1)

    def forward(self, x):
        weight = self.attention(x)
        self.last_attention = weight.detach()
        eps = torch.finfo(weight.dtype).eps
        weight_mean = weight.mean(dim=1, keepdim=True).clamp_min(eps)
        enhance_signal = torch.relu(weight / weight_mean - 1.0)
        beta = self.beta.reshape(1, 1, 1, 1).to(device=x.device, dtype=x.dtype)
        raw_gate = torch.tanh(beta)
        gate = torch.where(raw_gate >= 0, raw_gate, self.negative_slope * raw_gate)
        gamma = gate * self.max_scale
        channel_scale = 1.0 + gamma * enhance_signal
        self.last_channel_scale = channel_scale.detach()
        return x * channel_scale


GatedECAP4 = P4ECA
