#!/usr/bin/env python3
"""Validate, record, and select checkpoints for the extended BEV-150 run."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TP_ERROR_KEYS = ("trans_err", "scale_err", "orient_err", "vel_err", "attr_err")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path):
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def resolve_project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def validate_metrics(metrics):
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object")
    for key in ("nd_score", "mean_ap"):
        if not finite_number(metrics.get(key)) or not 0 <= float(metrics[key]) <= 1:
            raise ValueError(f"invalid metrics field: {key}")
    errors = metrics.get("tp_errors")
    if not isinstance(errors, dict):
        raise ValueError("metrics.tp_errors is missing")
    for key in TP_ERROR_KEYS:
        if not finite_number(errors.get(key)) or float(errors[key]) < 0:
            raise ValueError(f"invalid tp_errors field: {key}")


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_record(path, checkpoint, epoch):
    record = load_json(path)
    if record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ValueError("record is not complete schema version 1")
    if record.get("epoch") != epoch:
        raise ValueError("record epoch does not match")
    resolved = checkpoint.resolve()
    if resolve_project_path(record.get("checkpoint", "")) != resolved:
        raise ValueError("record checkpoint path does not match")
    if record.get("checkpoint_sha256") != sha256_file(resolved):
        raise ValueError("record checkpoint digest does not match")
    validate_metrics(record.get("metrics"))
    return record


def command_check(args):
    validate_record(args.record.resolve(), args.checkpoint.resolve(), args.epoch)
    print(f"Valid complete record: {args.record}")


def command_record(args):
    checkpoint = args.checkpoint.resolve()
    candidates = []
    for path in args.results_root.resolve().glob("*/pts_bbox/metrics_summary.json"):
        try:
            if path.stat().st_mtime_ns < args.started_ns:
                continue
            metrics = load_json(path)
            validate_metrics(metrics)
            candidates.append((path.stat().st_mtime_ns, path, metrics))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    if not candidates:
        raise RuntimeError("no new complete metrics_summary.json was produced")
    _, metrics_path, metrics = max(candidates, key=lambda item: item[0])
    record = {
        "schema_version": 1,
        "status": "complete",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "epoch": args.epoch,
        "checkpoint": portable_path(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metrics_path": portable_path(metrics_path),
        "metrics": {
            "nd_score": float(metrics["nd_score"]),
            "mean_ap": float(metrics["mean_ap"]),
            "tp_errors": {
                key: float(metrics["tp_errors"][key]) for key in TP_ERROR_KEYS
            },
        },
    }
    atomic_write_json(args.output.resolve(), record)
    print(f"Recorded epoch {args.epoch}: NDS={metrics['nd_score']:.6f}, "
          f"mAP={metrics['mean_ap']:.6f}")


def command_summarize(args):
    records = []
    for path in sorted(args.records_dir.resolve().glob("epoch_*_eval.json")):
        record = load_json(path)
        checkpoint = resolve_project_path(record.get("checkpoint", ""))
        epoch = record.get("epoch")
        if not isinstance(epoch, int):
            raise ValueError(f"invalid epoch in {path}")
        validate_record(path, checkpoint, epoch)
        records.append(record)
    if not records:
        raise RuntimeError("no complete epoch evaluation records found")
    best = max(
        records,
        key=lambda item: (item["metrics"]["nd_score"], item["metrics"]["mean_ap"]),
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "maximum NDS, then maximum mAP",
        "best_epoch": best["epoch"],
        "best_checkpoint": best["checkpoint"],
        "best_checkpoint_sha256": best["checkpoint_sha256"],
        "best_metrics": best["metrics"],
        "epochs": records,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(best["checkpoint"])


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="validate an existing record")
    check.add_argument("--record", type=Path, required=True)
    check.add_argument("--checkpoint", type=Path, required=True)
    check.add_argument("--epoch", type=int, required=True)
    check.set_defaults(func=command_check)

    record = subparsers.add_parser("record", help="record a newly produced eval")
    record.add_argument("--results-root", type=Path, required=True)
    record.add_argument("--started-ns", type=int, required=True)
    record.add_argument("--checkpoint", type=Path, required=True)
    record.add_argument("--epoch", type=int, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.set_defaults(func=command_record)

    summarize = subparsers.add_parser("summarize", help="select the best epoch")
    summarize.add_argument("--records-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.set_defaults(func=command_summarize)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
