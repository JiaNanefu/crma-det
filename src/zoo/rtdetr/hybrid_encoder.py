'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation
from .rfb_lite import P4RFBLite
from .eca_lite import (
    BalancedECAP4,
    EnhanceOnlyECAP4,
    LeakyEnhanceECAP4,
    MeanPreservingECAP4,
    P4ECA,
)

from src.core import register
from src.nn.modules import DySample


__all__ = ['HybridEncoder', 'ScaleAdaptiveFusion']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


class ScaleAdaptiveFusion(nn.Module):
    """Normalized learnable weighted fusion for adjacent-scale features.

    The module is intentionally lightweight: it learns one scalar per input
    feature and optionally applies a 1x1 Conv-BN-Act refinement after the
    weighted sum. The 1x1 refine keeps SAF-CCFM small while preserving the
    original CCFM CSP blocks as the main feature refiner.
    """

    def __init__(self, channels, num_inputs=2, eps=1e-4, use_refine=True, act="silu"):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.eps = eps
        self.refine = (
            ConvNormLayer(channels, channels, 1, 1, padding=0, act=act)
            if use_refine else nn.Identity()
        )

    def normalized_weights(self):
        weights = F.relu(self.weights)
        return weights / (weights.sum() + self.eps)

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)), "ScaleAdaptiveFusion expects a list/tuple of features."
        assert len(feats) == self.weights.numel(), "Feature count must match SAF weight count."
        ref_shape = feats[0].shape
        for feat in feats[1:]:
            assert feat.shape == ref_shape, "All SAF input feature shapes must match."

        weights = self.normalized_weights().to(device=feats[0].device, dtype=feats[0].dtype)
        out = torch.zeros_like(feats[0])
        for idx, feat in enumerate(feats):
            out = out + weights[idx] * feat
        return self.refine(out)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 use_saf_ccfm=False,
                 saf_eps=1e-4,
                 saf_use_refine=True,
                 dysample_ccfm=None,
                 p4_rfb_ccfm=None,
                 p4_eca=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.use_saf_ccfm = use_saf_ccfm
        self.dysample_ccfm_cfg = self._normalize_dysample_cfg(dysample_ccfm)
        self.use_dysample_ccfm = self.dysample_ccfm_cfg["enable"]
        self.p4_rfb_ccfm_cfg = self._normalize_p4_rfb_cfg(p4_rfb_ccfm, hidden_dim)
        self.use_p4_rfb_ccfm = self.p4_rfb_ccfm_cfg["enable"]
        self.p4_eca_cfg = self._normalize_p4_eca_cfg(p4_eca, hidden_dim)
        self.use_p4_eca = self.p4_eca_cfg["enable"]
        if self.use_p4_rfb_ccfm and self.use_p4_eca:
            raise ValueError("P4-RFB-CCFM and ECA-P4 cannot be enabled at the same time.")

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_upsamples = nn.ModuleList()
        self.fpn_fusions = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            if self.use_dysample_ccfm:
                self.fpn_upsamples.append(
                    DySample(
                        hidden_dim,
                        scale_factor=self.dysample_ccfm_cfg["scale_factor"],
                        groups=self.dysample_ccfm_cfg["groups"],
                        init_zero=self.dysample_ccfm_cfg["init_zero"],
                    )
                )
            if use_saf_ccfm:
                self.fpn_fusions.append(
                    ScaleAdaptiveFusion(hidden_dim, num_inputs=2, eps=saf_eps,
                                        use_refine=saf_use_refine, act=act)
                )
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_fusions = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            if use_saf_ccfm:
                self.pan_fusions.append(
                    ScaleAdaptiveFusion(hidden_dim, num_inputs=2, eps=saf_eps,
                                        use_refine=saf_use_refine, act=act)
                )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self.p4_rfb = None
        if self.use_p4_rfb_ccfm:
            if len(in_channels) != 3:
                raise ValueError(
                    f"P4-RFB-CCFM expects exactly 3 encoder levels, got {len(in_channels)}."
                )
            if self.p4_rfb_ccfm_cfg["target_level"] != 1:
                raise ValueError(
                    "P4-RFB-CCFM is restricted to target_level=1 (final P4 / outs[1])."
                )
            if self.p4_rfb_ccfm_cfg["in_channels"] != hidden_dim:
                raise ValueError(
                    "P4-RFB-CCFM in_channels must match HybridEncoder hidden_dim "
                    f"{hidden_dim}, got {self.p4_rfb_ccfm_cfg['in_channels']}."
                )
            self.p4_rfb = P4RFBLite(
                in_channels=self.p4_rfb_ccfm_cfg["in_channels"],
                branch_channels=self.p4_rfb_ccfm_cfg["branch_channels"],
                dilations=self.p4_rfb_ccfm_cfg["dilations"],
                beta_init=self.p4_rfb_ccfm_cfg["beta_init"],
                act=self.p4_rfb_ccfm_cfg["act"],
            )

        self.p4_eca = None
        if self.use_p4_eca:
            if len(in_channels) != 3:
                raise ValueError(
                    f"ECA-P4 expects exactly 3 encoder levels, got {len(in_channels)}."
                )
            if self.p4_eca_cfg["target_level"] != 1:
                raise ValueError("ECA-P4 is restricted to target_level=1 (final P4 / outs[1]).")
            if self.p4_eca_cfg["channels"] != hidden_dim:
                raise ValueError(
                    "ECA-P4 channels must match HybridEncoder hidden_dim "
                    f"{hidden_dim}, got {self.p4_eca_cfg['channels']}."
                )
            if self.p4_eca_cfg["mode"] == "balanced":
                self.p4_eca = BalancedECAP4(
                    channels=self.p4_eca_cfg["channels"],
                    kernel_size=self.p4_eca_cfg["kernel_size"],
                    beta_init=self.p4_eca_cfg["beta_init"],
                    gated_residual=self.p4_eca_cfg["gated_residual"],
                    max_scale=self.p4_eca_cfg["max_scale"],
                )
            elif self.p4_eca_cfg["mode"] == "mean_preserving":
                self.p4_eca = MeanPreservingECAP4(
                    channels=self.p4_eca_cfg["channels"],
                    kernel_size=self.p4_eca_cfg["kernel_size"],
                    beta_init=self.p4_eca_cfg["beta_init"],
                    gated_residual=self.p4_eca_cfg["gated_residual"],
                    max_scale=self.p4_eca_cfg["max_scale"],
                )
            elif self.p4_eca_cfg["mode"] == "enhance_only":
                self.p4_eca = EnhanceOnlyECAP4(
                    channels=self.p4_eca_cfg["channels"],
                    kernel_size=self.p4_eca_cfg["kernel_size"],
                    beta_init=self.p4_eca_cfg["beta_init"],
                    gated_residual=self.p4_eca_cfg["gated_residual"],
                    max_scale=self.p4_eca_cfg["max_scale"],
                )
            elif self.p4_eca_cfg["mode"] == "leaky_enhance":
                self.p4_eca = LeakyEnhanceECAP4(
                    channels=self.p4_eca_cfg["channels"],
                    kernel_size=self.p4_eca_cfg["kernel_size"],
                    beta_init=self.p4_eca_cfg["beta_init"],
                    gated_residual=self.p4_eca_cfg["gated_residual"],
                    max_scale=self.p4_eca_cfg["max_scale"],
                    negative_slope=self.p4_eca_cfg["negative_slope"],
                )
            else:
                self.p4_eca = P4ECA(
                    channels=self.p4_eca_cfg["channels"],
                    kernel_size=self.p4_eca_cfg["kernel_size"],
                    beta_init=self.p4_eca_cfg["beta_init"],
                    gated_residual=self.p4_eca_cfg["gated_residual"],
                )

        self._reset_parameters()

    def _normalize_dysample_cfg(self, dysample_ccfm):
        cfg = {
            "enable": False,
            "scale_factor": 2,
            "groups": 4,
            "init_zero": True,
        }
        if dysample_ccfm is None:
            return cfg
        if not isinstance(dysample_ccfm, dict):
            raise TypeError(
                f"dysample_ccfm config must be a dict or None, got {type(dysample_ccfm)}."
            )
        cfg["enable"] = bool(dysample_ccfm.get("enable", False))
        cfg["scale_factor"] = int(dysample_ccfm.get("scale_factor", cfg["scale_factor"]))
        cfg["groups"] = int(dysample_ccfm.get("groups", cfg["groups"]))
        cfg["init_zero"] = bool(dysample_ccfm.get("init_zero", cfg["init_zero"]))
        return cfg

    def _normalize_p4_rfb_cfg(self, p4_rfb_ccfm, hidden_dim):
        cfg = {
            "enable": False,
            "target_level": 1,
            "in_channels": int(hidden_dim),
            "branch_channels": 64,
            "dilations": [1, 3, 5],
            "beta_init": 0.0,
            "act": "silu",
        }
        if p4_rfb_ccfm is None:
            return cfg
        if not isinstance(p4_rfb_ccfm, dict):
            raise TypeError(
                f"p4_rfb_ccfm config must be a dict or None, got {type(p4_rfb_ccfm)}."
            )
        cfg["enable"] = bool(p4_rfb_ccfm.get("enable", False))
        cfg["target_level"] = int(p4_rfb_ccfm.get("target_level", cfg["target_level"]))
        cfg["in_channels"] = int(p4_rfb_ccfm.get("in_channels", cfg["in_channels"]))
        cfg["branch_channels"] = int(
            p4_rfb_ccfm.get("branch_channels", cfg["branch_channels"])
        )
        cfg["dilations"] = [
            int(v) for v in p4_rfb_ccfm.get("dilations", cfg["dilations"])
        ]
        cfg["beta_init"] = float(p4_rfb_ccfm.get("beta_init", cfg["beta_init"]))
        cfg["act"] = str(p4_rfb_ccfm.get("act", cfg["act"]))
        return cfg

    def _normalize_p4_eca_cfg(self, p4_eca, hidden_dim):
        cfg = {
            "enable": False,
            "target_level": 1,
            "channels": int(hidden_dim),
            "kernel_size": 5,
            "beta_init": 0.0,
            "gated_residual": True,
            "mode": "residual",
            "max_scale": 0.5,
            "negative_slope": 0.05,
        }
        if p4_eca is None:
            return cfg
        if not isinstance(p4_eca, dict):
            raise TypeError(f"p4_eca config must be a dict or None, got {type(p4_eca)}.")
        cfg["enable"] = bool(p4_eca.get("enable", False))
        cfg["target_level"] = int(p4_eca.get("target_level", cfg["target_level"]))
        cfg["channels"] = int(p4_eca.get("channels", cfg["channels"]))
        cfg["kernel_size"] = int(p4_eca.get("kernel_size", cfg["kernel_size"]))
        cfg["beta_init"] = float(p4_eca.get("beta_init", cfg["beta_init"]))
        cfg["gated_residual"] = bool(p4_eca.get("gated_residual", cfg["gated_residual"]))
        cfg["mode"] = str(p4_eca.get("mode", cfg["mode"])).lower()
        cfg["max_scale"] = float(p4_eca.get("max_scale", cfg["max_scale"]))
        cfg["negative_slope"] = float(
            p4_eca.get("negative_slope", cfg["negative_slope"])
        )
        if cfg["mode"] not in {
            "residual",
            "balanced",
            "mean_preserving",
            "enhance_only",
            "leaky_enhance",
        }:
            raise ValueError(
                "ECA-P4 mode must be 'residual', 'balanced', "
                "'mean_preserving', 'enhance_only', or "
                f"'leaky_enhance', got {cfg['mode']}."
            )
        if cfg["max_scale"] <= 0:
            raise ValueError(f"ECA-P4 max_scale must be positive, got {cfg['max_scale']}.")
        if cfg["negative_slope"] <= 0 or cfg["negative_slope"] >= 1:
            raise ValueError(
                "ECA-P4 negative_slope must be in (0, 1), "
                f"got {cfg['negative_slope']}."
            )
        return cfg

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @torch.jit.unused
    def set_eca_inference(self, enabled=True):
        """Enable/disable the optional ECA-P4 branch for eval-time deployment."""
        if self.p4_eca is not None:
            setter = getattr(self.p4_eca, "set_inference_enabled", None)
            if callable(setter):
                setter(enabled)
        return self

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                    if pos_embed.device != src_flatten.device:
                        pos_embed = pos_embed.to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            fusion_idx = len(self.in_channels) - 1 - idx
            if self.use_dysample_ccfm:
                upsample_feat = self.fpn_upsamples[fusion_idx](feat_high)
            else:
                upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            if self.use_saf_ccfm:
                fused_feat = self.fpn_fusions[fusion_idx]([upsample_feat, feat_low])
                fpn_input = torch.concat([fused_feat, feat_low], dim=1)
            else:
                fpn_input = torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[fusion_idx](fpn_input)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            if self.use_saf_ccfm:
                fused_feat = self.pan_fusions[idx]([downsample_feat, feat_high])
                pan_input = torch.concat([fused_feat, feat_high], dim=1)
            else:
                pan_input = torch.concat([downsample_feat, feat_high], dim=1)
            out = self.pan_blocks[idx](pan_input)
            outs.append(out)

        if self.use_p4_rfb_ccfm:
            outs[1] = self.p4_rfb(outs[1])
        if self.use_p4_eca:
            outs[1] = self.p4_eca(outs[1])

        return outs
