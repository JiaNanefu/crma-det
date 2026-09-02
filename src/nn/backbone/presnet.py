'''by lyuwenyu
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d
from .repema import RepEMAEnhance, RepConv
from .s2_gdf import S2GDF

from src.core import register


__all__ = ['PResNet', 'CSPBasicStage', 'OSAStage']


ResNet_cfg = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    # 152: [3, 8, 36, 3],
}


donwload_url = {
    18: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet18_vd_pretrained_from_paddle.pth',
    34: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet34_vd_pretrained_from_paddle.pth',
    50: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet50_vd_ssld_v2_pretrained_from_paddle.pth',
    101: 'https://github.com/lyuwenyu/storage/releases/download/v0.1/ResNet101_vd_ssld_pretrained_from_paddle.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='b'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            self.blocks.append(
                block(
                    ch_in, 
                    ch_out,
                    stride=2 if i == 0 and stage_num != 2 else 1, 
                    shortcut=False if i == 0 else True,
                    variant=variant,
                    act=act)
            )

            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class CSPBasicStage(nn.Module):
    """CSP-style replacement for a BasicBlock stage.

    The stage downsamples/projects the input into a main branch and a shortcut
    branch, runs the main branch through BasicBlocks, then concatenates and
    fuses back to the original stage output channels.
    """

    def __init__(
        self,
        ch_in,
        ch_out,
        count,
        stage_num,
        act='relu',
        variant='b',
        main_ratio=0.75,
        keep_depth=True,
    ):
        super().__init__()
        main_ratio = float(main_ratio)
        if not 0.0 < main_ratio < 1.0:
            raise ValueError(f"CSP main_ratio must be in (0, 1), got {main_ratio}.")
        if not keep_depth:
            raise ValueError("CSPBasicStage currently expects keep_depth=True.")

        stride = 2 if stage_num != 2 else 1
        main_channels = max(1, int(round(ch_out * main_ratio)))
        main_channels = min(main_channels, ch_out - 1)
        shortcut_channels = ch_out - main_channels

        self.stage_num = int(stage_num)
        self.main_ratio = main_ratio
        self.main_channels = main_channels
        self.shortcut_channels = shortcut_channels

        self.main_proj = self._make_projection(ch_in, main_channels, stride, variant, act)
        self.shortcut_proj = self._make_projection(ch_in, shortcut_channels, stride, variant, act)
        self.main_blocks = nn.ModuleList([
            BasicBlock(
                main_channels,
                main_channels,
                stride=1,
                shortcut=True,
                variant=variant,
                act=act,
            )
            for _ in range(count)
        ])
        self.fuse = ConvNormLayer(
            main_channels + shortcut_channels,
            ch_out,
            1,
            1,
            padding=0,
            act=act,
        )

    @staticmethod
    def _make_projection(ch_in, ch_out, stride, variant, act):
        if variant == 'd' and stride == 2:
            return nn.Sequential(OrderedDict([
                ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                ('conv', ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=act)),
            ]))
        return ConvNormLayer(ch_in, ch_out, 1, stride, padding=0, act=act)

    def forward(self, x):
        main = self.main_proj(x)
        for block in self.main_blocks:
            main = block(main)
        shortcut = self.shortcut_proj(x)
        return self.fuse(torch.cat([main, shortcut], dim=1))


class OSAStage(nn.Module):
    """VoVNet/OSA-style replacement for a BasicBlock stage."""

    def __init__(
        self,
        ch_in,
        ch_out,
        stage_num,
        act='relu',
        variant='b',
        num_layers=2,
        mid_ratio=0.5,
        use_residual=True,
    ):
        super().__init__()
        num_layers = int(num_layers)
        mid_ratio = float(mid_ratio)
        if num_layers <= 0:
            raise ValueError(f"OSAStage num_layers must be positive, got {num_layers}.")
        if not 0.0 < mid_ratio <= 1.0:
            raise ValueError(f"OSAStage mid_ratio must be in (0, 1], got {mid_ratio}.")

        stride = 2 if stage_num != 2 else 1
        mid_channels = max(1, int(round(ch_out * mid_ratio)))

        self.stage_num = int(stage_num)
        self.out_channels = int(ch_out)
        self.mid_channels = int(mid_channels)
        self.num_layers = num_layers
        self.mid_ratio = mid_ratio
        self.use_residual = bool(use_residual)
        self.proj = self._make_projection(ch_in, ch_out, stride, variant, act)
        self.reduce = ConvNormLayer(ch_out, mid_channels, 1, 1, padding=0, act=act)
        self.layers = nn.ModuleList([
            ConvNormLayer(mid_channels, mid_channels, 3, 1, act=act)
            for _ in range(num_layers)
        ])
        self.aggregate = ConvNormLayer(
            ch_out + num_layers * mid_channels,
            ch_out,
            1,
            1,
            padding=0,
            act=None,
        )
        self.out_act = get_activation(act)

    @staticmethod
    def _make_projection(ch_in, ch_out, stride, variant, act):
        if variant == 'd' and stride == 2:
            return nn.Sequential(OrderedDict([
                ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                ('conv', ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=act)),
            ]))
        return ConvNormLayer(ch_in, ch_out, 1, stride, padding=0, act=act)

    def forward(self, x):
        base = self.proj(x)
        feat = self.reduce(base)
        features = [base]
        for layer in self.layers:
            feat = layer(feat)
            features.append(feat)
        out = self.aggregate(torch.cat(features, dim=1))
        if self.use_residual:
            out = out + base
        return self.out_act(out)


@register
class PResNet(nn.Module):
    def __init__(
        self, 
        depth, 
        variant='d', 
        num_stages=4, 
        return_idx=[0, 1, 2, 3], 
        act='relu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False,
        repema=None,
        repreplace=None,
        s2_gdf=None,
        csp=None,
        vov=None):
        super().__init__()

        block_nums = ResNet_cfg[depth]
        ch_in = 64
        if variant in ['c', 'd']:
            conv_def = [
                [3, ch_in // 2, 3, 2, "conv1_1"],
                [ch_in // 2, ch_in // 2, 3, 1, "conv1_2"],
                [ch_in // 2, ch_in, 3, 1, "conv1_3"],
            ]
        else:
            conv_def = [[3, ch_in, 7, 2, "conv1_1"]]

        self.conv1 = nn.Sequential(OrderedDict([
            (_name, ConvNormLayer(c_in, c_out, k, s, act=act)) for c_in, c_out, k, s, _name in conv_def
        ]))

        ch_out_list = [64, 128, 256, 512]
        block = BottleNeck if depth >= 50 else BasicBlock

        _out_channels = [block.expansion * v for v in ch_out_list]
        _out_strides = [4, 8, 16, 32]

        self.res_layers = nn.ModuleList()
        for i in range(num_stages):
            stage_num = i + 2
            self.res_layers.append(
                Blocks(block, ch_in, ch_out_list[i], block_nums[i], stage_num, act=act, variant=variant)
            )
            ch_in = _out_channels[i]

        self.csp_cfg = self._normalize_csp_cfg(csp)
        self.vov_cfg = self._normalize_vov_cfg(vov)
        if self.csp_cfg["enable"] and self.vov_cfg["enable"]:
            raise ValueError("PResNet csp and vov cannot be enabled at the same time.")
        self.csp_replaced_stage_indices = []
        self.vov_replaced_stage_indices = []
        if self.csp_cfg["enable"]:
            self._apply_csp(block, ch_out_list, _out_channels, block_nums, num_stages, act, variant)
        if self.vov_cfg["enable"]:
            self._apply_vov(block, ch_out_list, _out_channels, num_stages, act, variant)

        self.repreplace_cfg = self._normalize_repreplace_cfg(repreplace)
        self.repreplace_pretrain_map = []
        if self.repreplace_cfg["enable"]:
            self._apply_repreplace(block, num_stages)

        self.return_idx = return_idx
        self.out_channels = [_out_channels[_i] for _i in return_idx]
        self.out_strides = [_out_strides[_i] for _i in return_idx]
        self.s2_gdf_cfg = self._normalize_s2_gdf_cfg(
            s2_gdf,
            default_in_channels=_out_channels[0],
            default_out_channels=_out_channels[1],
        )
        self.s2_gdf = None
        if self.s2_gdf_cfg["enable"]:
            if num_stages < 2:
                raise ValueError("S2GDF requires PResNet stages S2 and S3.")
            if 1 not in return_idx:
                raise ValueError(
                    f"S2GDF enhances S3, but S3 stage index 1 is not in return_idx={return_idx}."
                )
            if self.s2_gdf_cfg["in_channels"] != _out_channels[0]:
                raise ValueError(
                    "S2GDF in_channels must match S2 channels "
                    f"{_out_channels[0]}, got {self.s2_gdf_cfg['in_channels']}."
                )
            if self.s2_gdf_cfg["out_channels"] != _out_channels[1]:
                raise ValueError(
                    "S2GDF out_channels must match S3 channels "
                    f"{_out_channels[1]}, got {self.s2_gdf_cfg['out_channels']}."
                )
            self.s2_gdf = S2GDF(
                in_channels=self.s2_gdf_cfg["in_channels"],
                out_channels=self.s2_gdf_cfg["out_channels"],
                alpha_init=self.s2_gdf_cfg["alpha_init"],
                norm=self.s2_gdf_cfg["norm"],
                act=self.s2_gdf_cfg["act"],
            )
        self.repema_cfg = self._normalize_repema_cfg(repema)
        self.repema_modules = nn.ModuleDict()
        if self.repema_cfg["enable"]:
            return_stage_nums = {int(idx) + 2 for idx in return_idx}
            for stage_num in self.repema_cfg["stages"]:
                if stage_num not in {3, 4}:
                    raise ValueError(
                        f"RepEMA only supports stages 3 and 4 (S3/S4), got stage {stage_num}."
                    )
                stage_idx = stage_num - 2
                if stage_idx < 0 or stage_idx >= num_stages:
                    raise ValueError(
                        f"RepEMA stage {stage_num} is invalid for num_stages={num_stages}."
                    )
                if stage_num not in return_stage_nums:
                    print(
                        f"Skip RepEMA stage {stage_num}: stage is not included in return_idx={return_idx}."
                    )
                    continue
                self.repema_modules[str(stage_num)] = RepEMAEnhance(
                    _out_channels[stage_idx],
                    rep_enable=self.repema_cfg["rep_enable"],
                    ema_enable=self.repema_cfg["ema_enable"],
                    rep_init_scale=self.repema_cfg["rep_init_scale"],
                    rep_deploy=self.repema_cfg["rep_deploy"],
                    ema_groups=self.repema_cfg["ema_groups"],
                    ema_init_scale=self.repema_cfg["ema_init_scale"],
                    act=act,
                )

        if freeze_at >= 0:
            self._freeze_parameters(self.conv1)
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.res_layers[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            state = torch.hub.load_state_dict_from_url(donwload_url[depth])
            self._load_pretrained_state(state, depth)

        if self.repreplace_cfg["deploy"]:
            self.switch_repreplace_to_deploy()
            
    def _normalize_repema_cfg(self, repema):
        cfg = {
            "enable": False,
            "stages": [3, 4],
            "rep_enable": True,
            "rep_init_scale": 0.1,
            "rep_deploy": False,
            "ema_enable": True,
            "ema_groups": 8,
            "ema_init_scale": 0.1,
        }
        if repema is None:
            return cfg
        if not isinstance(repema, dict):
            raise TypeError(f"repema config must be a dict or None, got {type(repema)}.")

        cfg["enable"] = bool(repema.get("enable", False))
        cfg["stages"] = [int(v) for v in repema.get("stages", cfg["stages"])]
        rep_cfg = repema.get("rep", {}) or {}
        ema_cfg = repema.get("ema", {}) or {}
        cfg["rep_enable"] = bool(rep_cfg.get("enable", True))
        cfg["rep_init_scale"] = float(rep_cfg.get("init_scale", 0.1))
        cfg["rep_deploy"] = bool(rep_cfg.get("deploy", False))
        cfg["ema_enable"] = bool(ema_cfg.get("enable", True))
        cfg["ema_groups"] = int(ema_cfg.get("groups", 8))
        cfg["ema_init_scale"] = float(ema_cfg.get("init_scale", 0.1))
        return cfg

    def _normalize_s2_gdf_cfg(self, s2_gdf, default_in_channels, default_out_channels):
        cfg = {
            "enable": False,
            "in_channels": int(default_in_channels),
            "out_channels": int(default_out_channels),
            "alpha_init": 0.0,
            "norm": "bn",
            "act": "silu",
        }
        if s2_gdf is None:
            return cfg
        if not isinstance(s2_gdf, dict):
            raise TypeError(f"s2_gdf config must be a dict or None, got {type(s2_gdf)}.")

        cfg["enable"] = bool(s2_gdf.get("enable", False))
        cfg["in_channels"] = int(s2_gdf.get("in_channels", cfg["in_channels"]))
        cfg["out_channels"] = int(s2_gdf.get("out_channels", cfg["out_channels"]))
        cfg["alpha_init"] = float(s2_gdf.get("alpha_init", cfg["alpha_init"]))
        cfg["norm"] = str(s2_gdf.get("norm", cfg["norm"]))
        cfg["act"] = str(s2_gdf.get("act", cfg["act"]))
        return cfg

    def _normalize_csp_cfg(self, csp):
        cfg = {
            "enable": False,
            "stages": [3, 4],
            "main_ratio": 0.75,
            "keep_depth": True,
        }
        if csp is None:
            return cfg
        if not isinstance(csp, dict):
            raise TypeError(f"csp config must be a dict or None, got {type(csp)}.")

        cfg["enable"] = bool(csp.get("enable", False))
        cfg["stages"] = [int(v) for v in csp.get("stages", cfg["stages"])]
        cfg["main_ratio"] = float(csp.get("main_ratio", cfg["main_ratio"]))
        cfg["keep_depth"] = bool(csp.get("keep_depth", cfg["keep_depth"]))
        return cfg

    def _normalize_vov_cfg(self, vov):
        cfg = {
            "enable": False,
            "stages": [3, 4],
            "num_layers": 2,
            "mid_ratio": 0.5,
            "use_residual": True,
        }
        if vov is None:
            return cfg
        if not isinstance(vov, dict):
            raise TypeError(f"vov config must be a dict or None, got {type(vov)}.")

        cfg["enable"] = bool(vov.get("enable", False))
        cfg["stages"] = [int(v) for v in vov.get("stages", cfg["stages"])]
        cfg["num_layers"] = int(vov.get("num_layers", cfg["num_layers"]))
        cfg["mid_ratio"] = float(vov.get("mid_ratio", cfg["mid_ratio"]))
        cfg["use_residual"] = bool(vov.get("use_residual", cfg["use_residual"]))
        return cfg

    def _apply_csp(self, block_type, ch_out_list, out_channels, block_nums, num_stages, act, variant):
        if block_type is not BasicBlock:
            raise ValueError("CSPBasicStage currently supports BasicBlock PResNet backbones only.")
        if not self.csp_cfg["keep_depth"]:
            raise ValueError("CSP-PResNet-S34 requires keep_depth=True.")

        stage_in_channels = [64] + list(out_channels[:-1])
        replaced = []
        for stage_num in self.csp_cfg["stages"]:
            if stage_num not in {3, 4}:
                raise ValueError(
                    f"CSP-PResNet-S34 only replaces stages 3 and 4, got stage {stage_num}."
                )
            stage_idx = stage_num - 2
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(
                    f"CSP stage {stage_num} is invalid for num_stages={num_stages}."
                )
            self.res_layers[stage_idx] = CSPBasicStage(
                stage_in_channels[stage_idx],
                ch_out_list[stage_idx],
                block_nums[stage_idx],
                stage_num,
                act=act,
                variant=variant,
                main_ratio=self.csp_cfg["main_ratio"],
                keep_depth=self.csp_cfg["keep_depth"],
            )
            replaced.append(stage_idx)
        self.csp_replaced_stage_indices = sorted(replaced)

    def _apply_vov(self, block_type, ch_out_list, out_channels, num_stages, act, variant):
        if block_type is not BasicBlock:
            raise ValueError("OSAStage currently supports BasicBlock PResNet backbones only.")

        stage_in_channels = [64] + list(out_channels[:-1])
        replaced = []
        for stage_num in self.vov_cfg["stages"]:
            if stage_num not in {3, 4}:
                raise ValueError(
                    f"VoV-PResNet-S34 only replaces stages 3 and 4, got stage {stage_num}."
                )
            stage_idx = stage_num - 2
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(
                    f"VoV stage {stage_num} is invalid for num_stages={num_stages}."
                )
            self.res_layers[stage_idx] = OSAStage(
                stage_in_channels[stage_idx],
                ch_out_list[stage_idx],
                stage_num,
                act=act,
                variant=variant,
                num_layers=self.vov_cfg["num_layers"],
                mid_ratio=self.vov_cfg["mid_ratio"],
                use_residual=self.vov_cfg["use_residual"],
            )
            replaced.append(stage_idx)
        self.vov_replaced_stage_indices = sorted(replaced)

    def _normalize_repreplace_cfg(self, repreplace):
        cfg = {
            "enable": False,
            "stages": [3, 4],
            "position": "last_block_conv2",
            "deploy": False,
        }
        if repreplace is None:
            return cfg
        if not isinstance(repreplace, dict):
            raise TypeError(f"repreplace config must be a dict or None, got {type(repreplace)}.")

        cfg["enable"] = bool(repreplace.get("enable", False))
        cfg["stages"] = [int(v) for v in repreplace.get("stages", cfg["stages"])]
        cfg["position"] = str(repreplace.get("position", cfg["position"]))
        cfg["deploy"] = bool(repreplace.get("deploy", False))
        return cfg

    def _apply_repreplace(self, block_type, num_stages):
        if block_type is not BasicBlock:
            raise ValueError("RepReplace currently supports BasicBlock backbones only.")
        if self.repreplace_cfg["position"] != "last_block_conv2":
            raise ValueError(
                "RepReplace currently supports position='last_block_conv2' only, "
                f"got {self.repreplace_cfg['position']}."
            )

        for stage_num in self.repreplace_cfg["stages"]:
            if stage_num not in {3, 4}:
                raise ValueError(
                    f"RepReplace only supports stages 3 and 4, got stage {stage_num}."
                )
            stage_idx = stage_num - 2
            if stage_idx < 0 or stage_idx >= num_stages:
                raise ValueError(
                    f"RepReplace stage {stage_num} is invalid for num_stages={num_stages}."
                )

            stage = self.res_layers[stage_idx]
            block_idx = len(stage.blocks) - 1
            target_block = stage.blocks[block_idx]
            target = target_block.branch2b
            if not isinstance(target, ConvNormLayer):
                raise TypeError(
                    f"Expected ConvNormLayer at stage {stage_num} last block branch2b, "
                    f"got {type(target)}."
                )

            conv = target.conv
            if conv.in_channels != conv.out_channels:
                raise ValueError(
                    "RepReplace requires equal input/output channels for the replaced conv, "
                    f"got {conv.in_channels}->{conv.out_channels}."
                )
            if conv.kernel_size != (3, 3) or conv.stride != (1, 1) or conv.padding != (1, 1):
                raise ValueError(
                    "RepReplace target must be a stride-1 3x3 conv with padding=1, "
                    f"got kernel={conv.kernel_size}, stride={conv.stride}, padding={conv.padding}."
                )

            old_prefix = f"res_layers.{stage_idx}.blocks.{block_idx}.branch2b"
            target_block.branch2b = RepConv(conv.in_channels, act=None, deploy=False)
            self.repreplace_pretrain_map.append((old_prefix, f"{old_prefix}.rbr_dense"))

    def _load_pretrained_state(self, state, depth):
        csp_enabled = self.csp_cfg["enable"]
        vov_enabled = self.vov_cfg["enable"]
        repema_enabled = self.repema_cfg["enable"]
        repreplace_enabled = self.repreplace_cfg["enable"]
        s2_gdf_enabled = self.s2_gdf_cfg["enable"]
        if not csp_enabled and not vov_enabled and not repema_enabled and not repreplace_enabled and not s2_gdf_enabled:
            self.load_state_dict(state)
            print(f'Load PResNet{depth} state_dict')
            return

        adapted_state = OrderedDict()
        repconv_init_count = 0
        csp_ignored_replaced_stage_keys = []
        vov_ignored_replaced_stage_keys = []
        for key, value in state.items():
            adapted_key = key
            for old_prefix, dense_prefix in self.repreplace_pretrain_map:
                prefix = old_prefix + "."
                if key.startswith(prefix):
                    suffix = key[len(prefix):]
                    if suffix.startswith("conv.") or suffix.startswith("norm."):
                        adapted_key = dense_prefix + "." + suffix
                        if suffix == "conv.weight":
                            repconv_init_count += 1
                    break
            adapted_state[adapted_key] = value

        current = self.state_dict()
        loadable = OrderedDict()
        skipped_shape = []
        unexpected_from_pretrain = []
        for key, value in adapted_state.items():
            if key in current and current[key].shape == value.shape:
                loadable[key] = value
            elif csp_enabled and self._is_csp_replaced_stage_key(key):
                csp_ignored_replaced_stage_keys.append(key)
            elif vov_enabled and self._is_vov_replaced_stage_key(key):
                vov_ignored_replaced_stage_keys.append(key)
            elif key in current:
                skipped_shape.append(key)
            else:
                unexpected_from_pretrain.append(key)

        missing_keys, unexpected_keys = self.load_state_dict(loadable, strict=False)
        unexpected_keys = list(unexpected_keys) + unexpected_from_pretrain
        new_module_keys = [key for key in missing_keys if key.startswith("repema_modules.")]
        csp_missing_keys = [key for key in missing_keys if self._is_csp_replaced_stage_key(key)]
        vov_missing_keys = [key for key in missing_keys if self._is_vov_replaced_stage_key(key)]
        stem_loaded = self._all_prefix_keys_loaded(state, loadable, "conv1.")
        stage2_loaded = self._all_prefix_keys_loaded(state, loadable, "res_layers.0.")
        stage5_loaded = self._all_prefix_keys_loaded(state, loadable, "res_layers.3.")
        features = []
        if csp_enabled:
            features.append("CSP")
        if vov_enabled:
            features.append("VoV")
        if repema_enabled:
            features.append("RepEMA")
        if repreplace_enabled:
            features.append("RepReplace")
        if s2_gdf_enabled:
            features.append("S2GDF")
        print(f"Load PResNet{depth} state_dict with {' + '.join(features)} (strict=False)")
        print(f'Loaded pretrained keys count: {len(loadable)}')
        print(f'Missing keys count: {len(missing_keys)}')
        print(f'Unexpected keys count: {len(unexpected_keys)}')
        print(f'Missing keys list: {list(missing_keys)}')
        print(f'Unexpected keys list: {list(unexpected_keys)}')
        if skipped_shape:
            print(f'Skipped shape-mismatch keys: {skipped_shape}')
        if csp_enabled:
            print(f'CSP ignored replaced-stage pretrained keys count: {len(csp_ignored_replaced_stage_keys)}')
            print(f'CSP ignored replaced-stage pretrained keys list: {csp_ignored_replaced_stage_keys}')
            print(f'CSP missing replaced-stage keys count: {len(csp_missing_keys)}')
            print(f'CSP missing replaced-stage keys list: {csp_missing_keys}')
            print(f'Stem loaded: {stem_loaded}')
            print(f'Stage2 loaded: {stage2_loaded}')
            print(f'Stage5 loaded: {stage5_loaded}')
        if vov_enabled:
            print(f'VoV ignored replaced-stage pretrained keys count: {len(vov_ignored_replaced_stage_keys)}')
            print(f'VoV ignored replaced-stage pretrained keys list: {vov_ignored_replaced_stage_keys}')
            print(f'VoV missing replaced-stage keys count: {len(vov_missing_keys)}')
            print(f'VoV missing replaced-stage keys list: {vov_missing_keys}')
            print(f'Stem loaded: {stem_loaded}')
            print(f'Stage2 loaded: {stage2_loaded}')
            print(f'Stage5 loaded: {stage5_loaded}')
        print(f'New module keys count: {len(new_module_keys)}')
        print(f'RepConv initialized from original conv count: {repconv_init_count}')

    def _is_csp_replaced_stage_key(self, key):
        return any(key.startswith(f"res_layers.{idx}.") for idx in self.csp_replaced_stage_indices)

    def _is_vov_replaced_stage_key(self, key):
        return any(key.startswith(f"res_layers.{idx}.") for idx in self.vov_replaced_stage_indices)

    @staticmethod
    def _all_prefix_keys_loaded(source_state, loadable_state, prefix):
        keys = [key for key in source_state if key.startswith(prefix)]
        return bool(keys) and all(key in loadable_state for key in keys)

    def switch_repreplace_to_deploy(self):
        for module in self.modules():
            if isinstance(module, RepConv):
                module.switch_to_deploy()
        return self

    def convert_to_deploy(self):
        return self.switch_repreplace_to_deploy()

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        conv1 = self.conv1(x)
        x = F.max_pool2d(conv1, kernel_size=3, stride=2, padding=1)
        outs = []
        s2 = None
        for idx, stage in enumerate(self.res_layers):
            x = stage(x)
            if idx == 0:
                s2 = x
            if idx in self.return_idx:
                stage_num = idx + 2
                stage_key = str(stage_num)
                out = x
                if idx == 1 and self.s2_gdf is not None:
                    if s2 is None:
                        raise RuntimeError("S2GDF expected S2 feature before S3.")
                    out = self.s2_gdf(s2, out)
                if stage_key in self.repema_modules:
                    out = self.repema_modules[stage_key](out)
                outs.append(out)
        return outs
