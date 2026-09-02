"""Probe no-retraining inference optimizations on one baseline/CRMA pair.

This does not alter checkpoints. It measures the model forward only, using the
same CUDA-event protocol as benchmark_bn_fusion_and_test.py.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SOURCE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_ROOT.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import benchmark_bn_fusion_and_test as bench


class AutocastWrapper(nn.Module):
    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, x: torch.Tensor):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.model(x)


class CudaGraphWrapper(nn.Module):
    """Replay a fixed-shape model forward without per-op Python launches."""

    def __init__(self, model: nn.Module, sample: torch.Tensor):
        super().__init__()
        self.model = model
        self.static_x = sample.clone()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            for _ in range(20):
                self.model(self.static_x)
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_x)

    def forward(self, x: torch.Tensor):
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_output


def measure(model: nn.Module, x: torch.Tensor, warmup: int, timed: int, repeats: int = 1) -> dict[str, Any]:
    model.eval()
    fps_runs = []
    latency_runs = []
    for _ in range(repeats):
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _ in range(timed):
                model(x)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / timed
        fps_runs.append(1000.0 / ms)
        latency_runs.append(ms)
    return {"FPS": statistics.fmean(fps_runs), "latency_ms": statistics.fmean(latency_runs), "FPS_runs": fps_runs}


def load_pair(device: torch.device, fuse_bn: bool = True, disable_ordinal: bool = True) -> dict[str, nn.Module]:
    specs = {
        "baseline": (bench.BASELINE_CONFIG, PROJECT_ROOT / "weights" / "baseline2best.pt", False),
        "CRMA-Det": (bench.CRMA_CONFIG, PROJECT_ROOT / "weights" / "crma_det2best.pt", True),
    }
    models: dict[str, nn.Module] = {}
    for name, (config, weight, disable_ordinal) in specs.items():
        ordinal_off = disable_ordinal if name == "CRMA-Det" else False
        _, model, _, _, _ = bench.load_model(config, weight, device, disable_ordinal=ordinal_off)
        if fuse_bn:
            bench.fuse_module(model)
        models[name] = model
    return models


def preload_static_tensors(model: nn.Module, device: torch.device) -> int:
    """Move plain tensor constants used by eval to CUDA once, not every forward."""
    moved = 0
    for module in model.modules():
        for name in list(module.__dict__.keys()):
            if not (name.startswith("pos_embed") or name in {"anchors", "valid_mask"}):
                continue
            value = getattr(module, name, None)
            if torch.is_tensor(value) and value.device != device:
                setattr(module, name, value.to(device))
                moved += 1
    return moved


def run_variant(name: str, base_models: dict[str, nn.Module], warmup: int, timed: int, repeats: int) -> dict[str, Any]:
    channels_last = name.startswith("channels_last")
    use_fp16 = name.endswith("fp16")
    result: dict[str, Any] = {"variant": name, "results": {}}
    for model_name, base_model in base_models.items():
        model = base_model
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        if use_fp16:
            model = AutocastWrapper(model).eval()
        x = torch.randn(1, 3, 640, 640, device="cuda", dtype=torch.float32)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        result["results"][model_name] = measure(model, x, warmup, timed, repeats)
        del model, x
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = bench.argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--preload-static", action="store_true")
    parser.add_argument("--eca-off", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--fp32-only", action="store_true")
    parser.add_argument("--no-bn-fusion", action="store_true")
    parser.add_argument("--keep-ordinal", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this probe.")
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    fuse_bn = not args.no_bn_fusion
    disable_ordinal = not args.keep_ordinal
    models = load_pair(device, fuse_bn=fuse_bn, disable_ordinal=disable_ordinal)
    if args.preload_static:
        moved = {name: preload_static_tensors(model, device) for name, model in models.items()}
    else:
        moved = {}
    if args.eca_off:
        for model in models.values():
            setter = getattr(getattr(model, "encoder", None), "set_eca_inference", None)
            if callable(setter):
                setter(False)
    results = []
    variants = ("fp32",) if args.fp32_only else ("fp32", "fp16", "channels_last_fp32", "channels_last_fp16")
    for variant in variants:
        results.append(run_variant(variant, models, args.warmup, args.timed, args.repeats))
    if args.compile:
        # Reload fresh contiguous models so the compiled comparison is not
        # affected by the channels-last probe above.
        compile_models = load_pair(device, fuse_bn=fuse_bn, disable_ordinal=disable_ordinal)
        compiled: dict[str, nn.Module] = {}
        for model_name, model in compile_models.items():
            compiled[model_name] = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        results.append(run_variant("torch_compile_fp32", compiled, args.warmup, args.timed, args.repeats))
    if getattr(args, "cuda_graph", False):
        graph_models = load_pair(device, fuse_bn=fuse_bn, disable_ordinal=disable_ordinal)
        if args.preload_static:
            graph_moved = {name: preload_static_tensors(model, device) for name, model in graph_models.items()}
            moved["cuda_graph"] = graph_moved
        if args.eca_off:
            for model in graph_models.values():
                setter = getattr(getattr(model, "encoder", None), "set_eca_inference", None)
                if callable(setter):
                    setter(False)
        sample = torch.randn(1, 3, 640, 640, device=device, dtype=torch.float32)
        graphed = {name: CudaGraphWrapper(model, sample) for name, model in graph_models.items()}
        results.append(run_variant("cuda_graph_fp32", graphed, args.warmup, args.timed, args.repeats))
    print(json.dumps({"protocol": {"warmup": args.warmup, "timed": args.timed, "repeats": args.repeats, "input": "1x3x640x640", "batch": 1, "bn_fused": fuse_bn, "crma_ordinal_inference": "off" if disable_ordinal else "original", "cudnn_benchmark": bool(args.cudnn_benchmark), "tf32": bool(args.tf32), "preload_static": bool(args.preload_static), "eca_off": bool(args.eca_off), "moved_static_tensors": moved}, "results": results}, indent=2))


if __name__ == "__main__":
    main()
