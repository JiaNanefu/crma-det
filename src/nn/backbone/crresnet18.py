"""CR-ResNet18 — lightweight backbone for Carya maturity detection.

Builds on PResNet18-vd with three key improvements:
    - RepPConvBlock: partial 3×3 conv replaces stage 2/3/4's second BasicBlock
    - P4 MDC-Lite: dual-branch dilated depthwise conv for multi-scale detail
    - P5 LAMA: linear attention with local positional encoding for global context
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register
from .common import ConvNormLayer, get_activation
from .presnet import PResNet


__all__ = [
    "PartialConv3x3",
    "RepPConvBlock",
    "MDCLite",
    "P5FullLAMA",
    "LAMABottleneck",
    "CRResNet18",
]


# ---------------------------------------------------------------------------
#  PartialConv3x3
# ---------------------------------------------------------------------------

class PartialConv3x3(nn.Module):
    """Apply 3×3 spatial convolution to a channel subset, bypass the rest."""

    def __init__(self, channels, stride=1, pconv_ratio=0.25):
        super().__init__()
        if not 0.0 < pconv_ratio <= 1.0:
            raise ValueError(f"pconv_ratio must be in (0, 1], got {pconv_ratio}.")

        partial_channels = max(1, int(round(channels * pconv_ratio)))
        partial_channels = min(partial_channels, channels)
        self.partial_channels = partial_channels
        self.untouched_channels = channels - partial_channels
        self.stride = stride
        self.conv = nn.Conv2d(
            partial_channels, partial_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )

    def forward(self, x):
        x_conv = x[:, :self.partial_channels]
        x_conv = self.conv(x_conv)
        if self.untouched_channels == 0:
            return x_conv
        x_keep = x[:, self.partial_channels:]
        if self.stride == 2:
            x_keep = F.avg_pool2d(x_keep, kernel_size=2, stride=2, ceil_mode=True)
        return torch.cat([x_conv, x_keep], dim=1)


# ---------------------------------------------------------------------------
#  RepPConvBlock
# ---------------------------------------------------------------------------

class RepPConvBlock(nn.Module):
    """Residual 1×1 → partial 3×3 → 1×1 block."""

    def __init__(self, in_channels, out_channels, stride=1, act="relu",
                 pconv_ratio=0.5, resnet_variant="d"):
        super().__init__()
        self.pconv_ratio = float(pconv_ratio)
        self.use_identity = stride == 1 and in_channels == out_channels

        self.conv1 = ConvNormLayer(in_channels, out_channels, 1, 1, act=act)
        self.partial_conv = PartialConv3x3(out_channels, stride=stride, pconv_ratio=pconv_ratio)
        self.partial_norm = nn.BatchNorm2d(out_channels)
        self.partial_act = get_activation(act)
        self.conv2 = ConvNormLayer(out_channels, out_channels, 1, 1, act=None)

        if self.use_identity:
            self.short = nn.Identity()
        elif resnet_variant == "d" and stride == 2:
            self.short = nn.Sequential(OrderedDict([
                ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                ("conv", ConvNormLayer(in_channels, out_channels, 1, 1, act=None)),
            ]))
        else:
            self.short = ConvNormLayer(in_channels, out_channels, 1, stride, act=None)

        self.out_act = get_activation(act)

    def forward(self, x):
        identity = x if self.use_identity else self.short(x)
        out = self.conv1(x)
        out = self.partial_conv(out)
        out = self.partial_act(self.partial_norm(out))
        out = self.conv2(out)
        return self.out_act(out + identity)


# ---------------------------------------------------------------------------
#  MDC-Lite  (P4 detail enhancer)
# ---------------------------------------------------------------------------

class MDCLite(nn.Module):
    """Dual-branch dilated depthwise conv for multi-scale detail enhancement."""

    def __init__(self, channels, dilation=2, act="relu"):
        super().__init__()
        self.branch5 = nn.Sequential(
            nn.Conv2d(channels, channels, 5, 1, dilation * 2, dilation=dilation,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, dilation, dilation=dilation,
                      groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            get_activation(act),
        )
        self.fuse = nn.Conv2d(channels * 2, channels, 1, bias=False)

    def forward(self, x):
        out = torch.cat((self.branch5(x), self.branch3(x)), dim=1)
        return self.fuse(out) + x


# ---------------------------------------------------------------------------
#  LAMA  (P5 global context via linear attention)
# ---------------------------------------------------------------------------

class _DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ACLLinearAttention(nn.Module):
    """Linear attention with local positional enhancement."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, x_shape):
        b, n, c = x.shape
        h, w = x_shape[2:]
        if h * w != n:
            raise ValueError(f"token count mismatch: {n} vs {h}×{w}")
        if c % self.num_heads != 0:
            raise ValueError(f"dim={c} not divisible by num_heads={self.num_heads}")

        head_dim = c // self.num_heads
        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        q = q.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, self.num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)
        return x


class ACLLAMA(nn.Module):
    """LAMA block with conv position encoding + linear attention."""

    def __init__(self, dim, num_heads, mlp_ratio=2.0, qkv_bias=True,
                 drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = ACLLinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, x_shape):
        b, l, c = x.shape
        h, w = x_shape[2:]
        if h * w != l:
            raise ValueError(f"token count mismatch: {l} vs {h}×{w}")

        x = x + self.cpe1(x.reshape(b, h, w, c).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(b, h, w, c)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(b, l, c)
        x = self.attn(x, x_shape)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(b, h, w, c).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LAMABottleneck(nn.Module):
    """P5 bottleneck: reduce → LAMA → expand, with learnable residual scale."""

    def __init__(self, in_channels=512, hidden_dim=384, mlp_ratio=2.0,
                 num_heads=8, gamma_init=1e-4, act="relu"):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}")

        self.reduce = ConvNormLayer(in_channels, hidden_dim, 1, 1, act=act)
        self.lama = ACLLAMA(dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
        self.expand = ConvNormLayer(hidden_dim, in_channels, 1, 1, act=None)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        identity = x
        out = self.reduce(x)
        shape = out.shape
        tokens = out.flatten(2).permute(0, 2, 1)
        tokens = self.lama(tokens, shape)
        out = tokens.permute(0, 2, 1).reshape(shape).contiguous()
        out = self.expand(out)
        return identity + self.gamma * out


class P5FullLAMA(nn.Module):
    """Full-channel P5 LAMA with learnable residual scale."""

    def __init__(self, channels=512, mlp_ratio=2.0, num_heads=8, gamma_init=1e-4):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} not divisible by num_heads={num_heads}")
        self.lama = ACLLAMA(dim=channels, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        identity = x
        shape = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)
        tokens = self.lama(tokens, shape)
        out = tokens.permute(0, 2, 1).reshape(shape).contiguous()
        return identity + self.gamma * out


# ---------------------------------------------------------------------------
#  CRResNet18
# ---------------------------------------------------------------------------

def _validate_out_spec(out_channels, out_strides):
    expected_channels = [128, 256, 512]
    expected_strides = [8, 16, 32]
    if out_channels is not None and list(out_channels) != expected_channels:
        raise ValueError(f"Expected out_channels={expected_channels}, got {out_channels}.")
    if out_strides is not None and list(out_strides) != expected_strides:
        raise ValueError(f"Expected out_strides={expected_strides}, got {out_strides}.")


@register
class CRResNet18(PResNet):
    """ResNet-18-vd with P4 MDC-Lite and P5 LAMA global context.

    Stage 2/3/4 second blocks replaced by RepPConvBlock (partial conv).
    P4 output enhanced by MDC-Lite (dual-branch dilated depthwise conv).
    P5 output refined by LAMA (linear attention with conv position encoding).
    """

    _REPLACE_STAGE_BY_VARIANT = {"v2": {2, 3}}

    def __init__(
        self,
        variant="v2",
        resnet_variant="d",
        return_idx=[1, 2, 3],
        num_stages=4,
        act="relu",
        freeze_at=-1,
        freeze_norm=False,
        pretrained=False,
        out_channels=None,
        out_strides=None,
        light_block="RepPConvBlock",
        pconv_ratio=0.5,
        use_p5_global_context=False,
        p5_global_context_type="none",
        use_p4_mdc_lite=True,
        use_p5_lama=True,
        lama_hidden_dim=384,
        lama_mlp_ratio=2.0,
        lama_gamma_init=1e-4,
        lama_num_heads=8,
        p5_lama_mode="bottleneck",
    ):
        if pretrained:
            raise ValueError("CRResNet18 does not use external pretrained weights.")
        if variant not in self._REPLACE_STAGE_BY_VARIANT:
            raise ValueError(f"CRResNet18 supports variant='v2', got {variant}.")
        if light_block != "RepPConvBlock":
            raise ValueError(f"CRResNet18 only supports RepPConvBlock, got {light_block}.")
        if not 0.0 < float(pconv_ratio) <= 1.0:
            raise ValueError(f"pconv_ratio must be in (0, 1], got {pconv_ratio}.")
        if use_p5_global_context or p5_global_context_type != "none":
            raise ValueError("CRResNet18 expects P5 global context disabled.")
        if p5_lama_mode not in {"bottleneck", "full"}:
            raise ValueError(f"p5_lama_mode must be 'bottleneck' or 'full', got {p5_lama_mode}.")
        if p5_lama_mode == "full" and int(lama_hidden_dim) != 512:
            raise ValueError("P5 full LAMA expects lama_hidden_dim=512.")
        _validate_out_spec(out_channels, out_strides)

        super().__init__(
            depth=18, variant=resnet_variant, num_stages=num_stages,
            return_idx=return_idx, act=act, freeze_at=freeze_at,
            freeze_norm=False, pretrained=False,
        )

        self.variant = variant
        self.resnet_variant = resnet_variant
        self.pconv_ratio = float(pconv_ratio)
        self.light_block = light_block
        self.p5_lama_mode = p5_lama_mode
        self.replaced_stage_indices = sorted(self._REPLACE_STAGE_BY_VARIANT[variant])

        self._replace_second_blocks(act)

        self.use_p4_mdc_lite = bool(use_p4_mdc_lite)
        self.use_p5_lama = bool(use_p5_lama)
        self.p4_mdc_lite = MDCLite(256, dilation=2, act=act) if self.use_p4_mdc_lite else nn.Identity()

        if not self.use_p5_lama:
            self.p5_lama_bottleneck = nn.Identity()
        elif self.p5_lama_mode == "full":
            self.p5_lama_bottleneck = P5FullLAMA(
                channels=512, mlp_ratio=lama_mlp_ratio,
                num_heads=lama_num_heads, gamma_init=lama_gamma_init,
            )
        else:
            self.p5_lama_bottleneck = LAMABottleneck(
                in_channels=512, hidden_dim=lama_hidden_dim,
                mlp_ratio=lama_mlp_ratio, num_heads=lama_num_heads,
                gamma_init=lama_gamma_init, act=act,
            )

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])
        if freeze_norm:
            self._freeze_norm(self)

    def _replace_second_blocks(self, act):
        stage_channels = [64, 128, 256, 512]
        for stage_idx in self.replaced_stage_indices:
            stage = self.res_layers[stage_idx]
            if len(stage.blocks) < 2:
                raise ValueError(f"Stage {stage_idx} has no second block to replace.")
            channels = stage_channels[stage_idx]
            stage.blocks[1] = RepPConvBlock(
                channels, channels, stride=1, act=act,
                pconv_ratio=self.pconv_ratio, resnet_variant=self.resnet_variant,
            )

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        outs = []
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == 2:
                x = self.p4_mdc_lite(x)
            elif idx == 3:
                x = self.p5_lama_bottleneck(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs
