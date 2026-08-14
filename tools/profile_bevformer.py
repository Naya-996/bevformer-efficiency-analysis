#!/usr/bin/env python3
"""Profile BEVFormer inference on one GPU and write machine-readable JSON.

The timer excludes fetching a batch from the DataLoader. It includes the
CPU-to-GPU scatter performed by ``MMDataParallel``, the model forward pass,
and detector post-processing. CUDA is synchronized immediately before and
after every measured forward pass so the reported values are wall-clock
latencies rather than asynchronous kernel launch times.
"""

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MMDET3D_ROOT = PROJECT_ROOT / "mmdetection3d"
for import_root in (PROJECT_ROOT, LOCAL_MMDET3D_ROOT):
    import_root_str = str(import_root)
    if import_root.exists() and import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import mmcv  # noqa: E402
import mmdet  # noqa: E402
import mmdet3d  # noqa: E402
import mmseg  # noqa: E402
import torch  # noqa: E402
from mmcv import Config, DictAction  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint, wrap_fp16_model  # noqa: E402
from mmdet.datasets import replace_ImageToTensor  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile a BEVFormer-compatible config on one GPU. Data loading "
            "is excluded from latency measurements."
        )
    )
    parser.add_argument("config", help="model/test config file")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="number of unmeasured warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="number of measured iterations (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker processes; fetching is outside timing (default: 2)",
    )
    parser.add_argument(
        "--output",
        default="work_dirs/bevformer_profile.json",
        help=(
            "JSON output path, relative to the invocation directory; use '-' "
            "to only print JSON (default: work_dirs/bevformer_profile.json)"
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help=(
            "override config values as key=value pairs, for example "
            "model.pts_bbox_head.bev_h=100 "
            "model.pts_bbox_head.bev_w=100"
        ),
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    return args


def import_config_plugins(cfg, config_path):
    """Import custom modules and the OpenMMLab plugin declared by a config."""
    custom_imports = cfg.get("custom_imports")
    if custom_imports:
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**custom_imports)

    if not cfg.get("plugin", False):
        return None

    plugin_dir = cfg.get("plugin_dir")
    if plugin_dir:
        plugin_path = Path(plugin_dir).expanduser()
        if not plugin_path.is_absolute():
            project_candidate = PROJECT_ROOT / plugin_path
            config_candidate = config_path.parent / plugin_path
            plugin_path = (
                project_candidate
                if project_candidate.exists()
                else config_candidate
            )
    else:
        plugin_path = config_path.parent

    plugin_path = plugin_path.resolve()
    try:
        module_parts = plugin_path.relative_to(PROJECT_ROOT).parts
        python_root = PROJECT_ROOT
    except ValueError:
        module_parts = (plugin_path.name,)
        python_root = plugin_path.parent

    python_root_str = str(python_root)
    if python_root_str not in sys.path:
        sys.path.insert(0, python_root_str)

    module_name = ".".join(part for part in module_parts if part)
    if not module_name:
        raise ValueError(f"Cannot derive plugin module from {plugin_path}")
    importlib.import_module(module_name)
    return module_name


def prepare_test_data_cfg(cfg):
    """Apply the same test-mode handling as this repository's tools/test.py."""
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline
            )
    elif isinstance(cfg.data.test, (list, tuple)):
        for dataset_cfg in cfg.data.test:
            dataset_cfg.test_mode = True
        samples_per_gpu = max(
            dataset_cfg.pop("samples_per_gpu", 1)
            for dataset_cfg in cfg.data.test
        )
        if samples_per_gpu > 1:
            for dataset_cfg in cfg.data.test:
                dataset_cfg.pipeline = replace_ImageToTensor(
                    dataset_cfg.pipeline
                )
    else:
        raise TypeError(
            "cfg.data.test must be a mapping or a sequence of mappings, "
            f"got {type(cfg.data.test)!r}"
        )
    return int(samples_per_gpu)


def infinite_batches(data_loader):
    """Yield sequential batches forever, restarting only at dataset end."""
    while True:
        yielded = False
        for batch in data_loader:
            yielded = True
            yield batch
        if not yielded:
            raise RuntimeError("The test DataLoader produced no batches")


def count_parameters(model):
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(total - trainable),
        "total_millions": float(total / 1_000_000.0),
    }


def memory_stats(device, baseline_allocated, baseline_reserved):
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    current_allocated = int(torch.cuda.memory_allocated(device))
    current_reserved = int(torch.cuda.memory_reserved(device))

    def mib(value):
        return float(value / (1024.0 ** 2))

    return {
        "baseline_allocated_bytes": int(baseline_allocated),
        "baseline_allocated_mib": mib(baseline_allocated),
        "baseline_reserved_bytes": int(baseline_reserved),
        "baseline_reserved_mib": mib(baseline_reserved),
        "peak_allocated_bytes": peak_allocated,
        "peak_allocated_mib": mib(peak_allocated),
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_mib": mib(peak_reserved),
        "current_allocated_bytes": current_allocated,
        "current_allocated_mib": mib(current_allocated),
        "current_reserved_bytes": current_reserved,
        "current_reserved_mib": mib(current_reserved),
    }


def json_safe(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def main():
    args = parse_args()
    process_start_load = list(os.getloadavg())
    invocation_dir = Path.cwd()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output_path = None
    if args.output != "-":
        output_path = Path(args.output).expanduser()
        if not output_path.is_absolute():
            output_path = (invocation_dir / output_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BEVFormer profiling")

    os.chdir(PROJECT_ROOT)
    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    plugin_module = import_config_plugins(cfg, config_path)
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    if cfg.get("close_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    samples_per_gpu = prepare_test_data_cfg(cfg)
    dataset = build_dataset(cfg.data.test)
    if len(dataset) == 0:
        raise RuntimeError("The test dataset is empty")
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get("nonshuffler_sampler"),
    )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(
        model, str(checkpoint_path), map_location="cpu"
    )

    checkpoint_meta = checkpoint.get("meta", {})
    model.CLASSES = checkpoint_meta.get("CLASSES", dataset.CLASSES)
    if "PALETTE" in checkpoint_meta:
        model.PALETTE = checkpoint_meta["PALETTE"]
    elif hasattr(dataset, "PALETTE"):
        model.PALETTE = dataset.PALETTE

    parameters = count_parameters(model)
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    batch_stream = infinite_batches(data_loader)
    latencies_ms = []
    with torch.no_grad():
        for _ in range(args.warmup):
            batch = next(batch_stream)
            result = model(return_loss=False, rescale=True, **batch)
            torch.cuda.synchronize(device)
            del result, batch

        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        timing_start_load = list(os.getloadavg())

        for _ in range(args.iters):
            # Fetching/decoding/collating this batch is intentionally untimed.
            batch = next(batch_stream)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = model(return_loss=False, rescale=True, **batch)
            torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            del result, batch

    torch.cuda.synchronize(device)
    timing_end_load = list(os.getloadavg())
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    total_timed_seconds = float(latency_array.sum() / 1000.0)
    measured_samples = int(args.iters * samples_per_gpu)
    device_properties = torch.cuda.get_device_properties(device)

    report = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "plugin_module": plugin_module,
        "cfg_options": json_safe(args.cfg_options or {}),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mmcv": mmcv.__version__,
            "mmdet": mmdet.__version__,
            "mmdet3d": mmdet3d.__version__,
            "mmseg": mmseg.__version__,
        },
        "device": {
            "logical_index": 0,
            "name": device_properties.name,
            "compute_capability": list(
                torch.cuda.get_device_capability(device)
            ),
            "total_memory_bytes": int(device_properties.total_memory),
            "visible_device_count": int(torch.cuda.device_count()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "host": {
            "logical_cpu_count": os.cpu_count(),
            "load_average_process_start": process_start_load,
            "load_average_timing_start": timing_start_load,
            "load_average_timing_end": timing_end_load,
        },
        "settings": {
            "warmup_iterations": args.warmup,
            "measured_iterations": args.iters,
            "workers": args.workers,
            "samples_per_gpu": samples_per_gpu,
            "measured_samples": measured_samples,
            "dataset_size": int(len(dataset)),
            "data_loading_in_timing": False,
            "measurement_scope": (
                "MMDataParallel scatter + model forward + detector "
                "post-processing"
            ),
            "cuda_synchronize_before_and_after": True,
        },
        "parameters": parameters,
        "gpu_memory": memory_stats(
            device, baseline_allocated, baseline_reserved
        ),
        "latency_ms": {
            "avg": float(latency_array.mean()),
            "p50": float(np.percentile(latency_array, 50)),
            "p95": float(np.percentile(latency_array, 95)),
            "std": float(latency_array.std(ddof=0)),
            "min": float(latency_array.min()),
            "max": float(latency_array.max()),
            "per_iteration": [float(value) for value in latency_array],
        },
        "throughput": {
            "fps": float(measured_samples / total_timed_seconds),
            "iterations_per_second": float(
                args.iters / total_timed_seconds
            ),
            "total_timed_seconds": total_timed_seconds,
        },
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Profile JSON written to {output_path}", file=sys.stderr)
    print(rendered, end="")


if __name__ == "__main__":
    main()
