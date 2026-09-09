#!/usr/bin/env python3
"""Scale optimizer learning rates in a trusted local MMCV checkpoint.

This is useful when extending a completed run with a lower learning-rate
schedule while retaining AdamW moments and runner progress.
"""

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="trusted local input checkpoint")
    parser.add_argument("output", help="output checkpoint")
    parser.add_argument(
        "--target-lr",
        type=float,
        required=True,
        help="new maximum initial_lr across optimizer parameter groups",
    )
    parser.add_argument("--force", action="store_true", help="overwrite output")
    args = parser.parse_args()
    if args.target_lr <= 0:
        parser.error("--target-lr must be positive")
    return args


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(value, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("input and output must be different files")
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force")

    checkpoint = torch_load(source)
    if not isinstance(checkpoint, dict) or "optimizer" not in checkpoint:
        raise ValueError("checkpoint does not contain optimizer state")
    optimizer = checkpoint["optimizer"]
    groups = optimizer.get("param_groups") if isinstance(optimizer, dict) else None
    if not isinstance(groups, list) or not groups:
        raise ValueError("optimizer.param_groups is missing or empty")

    initial_lrs = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict) or "lr" not in group:
            raise ValueError(f"invalid optimizer parameter group {index}")
        initial_lrs.append(float(group.get("initial_lr", group["lr"])))
    reference_lr = max(initial_lrs)
    if reference_lr <= 0:
        raise ValueError("checkpoint learning rates must be positive")
    scale = args.target_lr / reference_lr

    before = []
    after = []
    for group in groups:
        old_initial = float(group.get("initial_lr", group["lr"]))
        old_current = float(group["lr"])
        before.append({"initial_lr": old_initial, "lr": old_current})
        group["initial_lr"] = old_initial * scale
        group["lr"] = old_current * scale
        after.append(
            {"initial_lr": group["initial_lr"], "lr": group["lr"]}
        )

    meta = checkpoint.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("checkpoint meta must be a mapping")
    history = meta.setdefault("optimizer_lr_rebase", [])
    if not isinstance(history, list):
        history = [history]
        meta["optimizer_lr_rebase"] = history
    history.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source),
            "reference_initial_lr": reference_lr,
            "target_initial_lr": args.target_lr,
            "scale": scale,
            "groups_before": before,
            "groups_after": after,
        }
    )

    atomic_torch_save(checkpoint, output)
    print(f"Wrote {output}")
    print(f"Scaled {len(groups)} parameter groups by {scale:.8g}")


if __name__ == "__main__":
    main()
