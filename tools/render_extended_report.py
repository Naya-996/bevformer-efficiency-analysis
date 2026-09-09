#!/usr/bin/env python3
"""Render the completed extended BEV-150 experiment as a concise report."""

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def portable_path(path):
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def metrics_from_manifest(path):
    manifest = load(path)
    metrics_path = Path(manifest["metrics_path"])
    if not metrics_path.is_absolute():
        metrics_path = ROOT / metrics_path
    return load(metrics_path)


def profile_values(path):
    report = load(path)
    return {
        "fps": float(report["throughput"]["fps"]),
        "latency": float(report["latency_ms"]["avg"]),
        "p95": float(report["latency_ms"]["p95"]),
        "memory": float(report["gpu_memory"]["peak_allocated_mib"]),
        "params": float(report["parameters"]["total_millions"]),
    }


def percent(new, old):
    return (new / old - 1.0) * 100.0


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--baseline-profile",
        type=Path,
        default=ROOT / "experiments/baseline/profile.json",
    )
    parser.add_argument(
        "--previous-profile",
        type=Path,
        default=ROOT / "experiments/lite_finetuned/profile.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = load(args.summary.resolve())
    if summary.get("status") != "complete":
        raise ValueError("extended summary is not complete")
    best = summary["best_metrics"]
    extended_nds = float(best["nd_score"])
    extended_map = float(best["mean_ap"])
    extended_profile = profile_values(args.profile.resolve())

    base_metrics = metrics_from_manifest(
        ROOT / "experiments/baseline/eval_manifest.json"
    )
    base_nds = float(base_metrics["nd_score"])
    base_map = float(base_metrics["mean_ap"])
    base_profile = profile_values(args.baseline_profile.resolve())
    previous_metrics = metrics_from_manifest(
        ROOT / "experiments/lite_finetuned/eval_manifest.json"
    )
    previous_nds = float(previous_metrics["nd_score"])
    previous_map = float(previous_metrics["mean_ap"])
    previous_profile = profile_values(args.previous_profile.resolve())

    best_record = next(
        item for item in summary["epochs"]
        if item["epoch"] == summary["best_epoch"]
    )
    best_metrics_path = Path(best_record["metrics_path"])
    if not best_metrics_path.is_absolute():
        best_metrics_path = ROOT / best_metrics_path
    best_raw_metrics = load(best_metrics_path)
    class_deltas = sorted(
        (
            float(best_raw_metrics["mean_dist_aps"][name])
            - float(previous_metrics["mean_dist_aps"][name]),
            name,
        )
        for name in previous_metrics["mean_dist_aps"]
    )
    largest_gains = ", ".join(
        f"{name} {delta * 100:+.2f} AP" for delta, name in reversed(class_deltas[-3:])
    )
    largest_regressions = ", ".join(
        f"{name} {delta * 100:+.2f} AP" for delta, name in class_deltas[:3]
    )

    text = f"""# BEV-150 extended fine-tuning result

Status: **complete**. Epoch {summary['best_epoch']} was selected by maximum NDS
with mAP as the tie-breaker after full evaluation of epochs 3--6.

| Variant | NDS | mAP | FPS | Mean / P95 latency (ms) | Peak allocated (MiB) | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Base | {base_nds:.4f} | {base_map:.4f} | {base_profile['fps']:.3f} | {base_profile['latency']:.2f} / {base_profile['p95']:.2f} | {base_profile['memory']:.1f} | {base_profile['params']:.3f} |
| L2 BEV-150, 2 ep | {previous_nds:.4f} | {previous_map:.4f} | {previous_profile['fps']:.3f} | {previous_profile['latency']:.2f} / {previous_profile['p95']:.2f} | {previous_profile['memory']:.1f} | {previous_profile['params']:.3f} |
| L3 BEV-150, best of 3--6 ep | {extended_nds:.4f} | {extended_map:.4f} | {extended_profile['fps']:.3f} | {extended_profile['latency']:.2f} / {extended_profile['p95']:.2f} | {extended_profile['memory']:.1f} | {extended_profile['params']:.3f} |

## Measured deltas

- L3 versus the two-epoch model: NDS {extended_nds - previous_nds:+.4f}, mAP {extended_map - previous_map:+.4f}.
- L3 versus Base: NDS {extended_nds - base_nds:+.4f}, mAP {extended_map - base_map:+.4f}.
- L3 versus Base efficiency: FPS {percent(extended_profile['fps'], base_profile['fps']):+.2f}%, mean latency {percent(extended_profile['latency'], base_profile['latency']):+.2f}%, parameters {percent(extended_profile['params'], base_profile['params']):+.2f}%.
- Largest class-level gains over L2: {largest_gains}.
- Largest class-level regressions from L2: {largest_regressions}.

## Provenance

- Selection rule: `{summary['selection_rule']}`
- Best checkpoint: `{summary['best_checkpoint']}`
- Checkpoint SHA256: `{summary['best_checkpoint_sha256']}`
- Full epoch metrics: `experiments/bev150_extended/summary.json`
- Paired Base profile: `{portable_path(args.baseline_profile)}`
- Paired L2 profile: `{portable_path(args.previous_profile)}`
- Paired L3 profile: `{portable_path(args.profile)}`

All accuracy values come from the complete 6,019-sample nuScenes validation
set. Performance was measured back-to-back on physical GPU1 with 20 warm-up
and 200 synchronized iterations per model; data loading is excluded.
"""
    atomic_write(args.output.resolve(), text)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
