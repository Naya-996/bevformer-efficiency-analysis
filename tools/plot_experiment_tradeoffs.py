#!/usr/bin/env python3
"""Plot BEVFormer accuracy/efficiency trade-offs from a summary JSON."""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path


DEFAULT_SUMMARY = "experiments/summary.json"
DEFAULT_OUTPUT_DIR = "docs/assets"


class PlotInputError(ValueError):
    """Raised when the summary cannot produce the requested plots."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate BEVFormer NDS-versus-FPS and NDS-versus-memory plots "
            "from experiments/summary.json."
        )
    )
    parser.add_argument(
        "--summary",
        default=DEFAULT_SUMMARY,
        help=f"experiment summary JSON (default: {DEFAULT_SUMMARY})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"directory for PNG and SVG outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def is_finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def load_experiments(summary_path):
    try:
        with summary_path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
    except FileNotFoundError as exc:
        raise PlotInputError(f"summary does not exist: {summary_path}") from exc
    except json.JSONDecodeError as exc:
        raise PlotInputError(
            f"invalid JSON in {summary_path}: {exc.msg} at line {exc.lineno}"
        ) from exc

    if not isinstance(summary, dict):
        raise PlotInputError("summary root must be a JSON object")
    experiments = summary.get("experiments")
    if not isinstance(experiments, list):
        raise PlotInputError("summary field 'experiments' must be a list")

    seen_ids = set()
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, dict):
            raise PlotInputError(f"experiments[{index}] must be an object")
        experiment_id = experiment.get("id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise PlotInputError(
                f"experiments[{index}] has no non-empty string ID"
            )
        if experiment_id in seen_ids:
            raise PlotInputError(f"duplicate experiment ID: {experiment_id}")
        seen_ids.add(experiment_id)
    return experiments


def select_points(experiments, x_field):
    points = []
    for experiment in experiments:
        if str(experiment.get("status", "")).lower() != "complete":
            continue
        if not is_finite_number(experiment.get("nds")):
            continue
        if not is_finite_number(experiment.get(x_field)):
            continue
        points.append(
            {
                "id": experiment["id"],
                "variant": str(experiment.get("variant", experiment["id"])),
                "x": float(experiment[x_field]),
                "nds": float(experiment["nds"]),
            }
        )
    return points


def require_enough_points(fps_points, memory_points):
    errors = []
    if len(fps_points) < 2:
        ids = ", ".join(point["id"] for point in fps_points) or "none"
        errors.append(
            "need at least 2 status=complete experiments with numeric NDS "
            f"and FPS; found {len(fps_points)} ({ids})"
        )
    if len(memory_points) < 2:
        ids = ", ".join(point["id"] for point in memory_points) or "none"
        errors.append(
            "need at least 2 status=complete experiments with numeric NDS "
            "and peak_allocated_mib; "
            f"found {len(memory_points)} ({ids})"
        )
    if errors:
        raise PlotInputError("; ".join(errors))


def style_for(experiment_id):
    if experiment_id == "B0":
        return {
            "group": "Official base",
            "color": "#D55E00",
            "marker": "*",
            "size": 190,
            "edgecolor": "#222222",
            "linewidth": 1.2,
            "zorder": 5,
        }
    if experiment_id.startswith("B"):
        return {
            "group": "BEV resolution",
            "color": "#0072B2",
            "marker": "o",
            "size": 95,
            "edgecolor": "white",
            "linewidth": 0.9,
            "zorder": 3,
        }
    if experiment_id.startswith("E"):
        return {
            "group": "Encoder depth",
            "color": "#009E73",
            "marker": "s",
            "size": 95,
            "edgecolor": "white",
            "linewidth": 0.9,
            "zorder": 3,
        }
    if experiment_id.startswith("T"):
        return {
            "group": "Temporal ablation",
            "color": "#CC79A7",
            "marker": "D",
            "size": 95,
            "edgecolor": "white",
            "linewidth": 0.9,
            "zorder": 3,
        }
    if experiment_id.startswith("L"):
        return {
            "group": "Fine-tuned Lite",
            "color": "#E69F00",
            "marker": "^",
            "size": 120,
            "edgecolor": "#222222",
            "linewidth": 0.9,
            "zorder": 4,
        }
    return {
        "group": "Other",
        "color": "#666666",
        "marker": "^",
        "size": 95,
        "edgecolor": "white",
        "linewidth": 0.9,
        "zorder": 3,
    }


def render_plot(plt, points, title, xlabel, png_path, svg_path):
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    legend_groups = set()
    try:
        for point in points:
            style = style_for(point["id"])
            group = style.pop("group")
            legend_label = group if group not in legend_groups else None
            legend_groups.add(group)
            axis.scatter(
                point["x"],
                point["nds"],
                label=legend_label,
                c=style.pop("color"),
                marker=style.pop("marker"),
                s=style.pop("size"),
                edgecolors=style.pop("edgecolor"),
                linewidths=style.pop("linewidth"),
                zorder=style.pop("zorder"),
            )
            axis.annotate(
                point["id"],
                (point["x"], point["nds"]),
                xytext=(7, 6),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold" if point["id"] == "B0" else "normal",
            )

        axis.set_title(title, fontsize=14, pad=12)
        axis.set_xlabel(xlabel, fontsize=11)
        axis.set_ylabel("nuScenes NDS (higher is better)", fontsize=11)
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        axis.set_axisbelow(True)
        axis.margins(x=0.12, y=0.15)
        axis.legend(frameon=True, fontsize=9)
        figure.tight_layout()
        figure.savefig(
            png_path,
            format="png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        figure.savefig(
            svg_path,
            format="svg",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    finally:
        plt.close(figure)


def commit_outputs(staged_outputs):
    """Replace all destinations, restoring previous files on commit failure."""
    backups = {}
    installed = []
    try:
        for staged_path, destination in staged_outputs:
            if destination.exists():
                backup = staged_path.with_name(staged_path.name + ".previous")
                os.replace(destination, backup)
                backups[destination] = backup
        for staged_path, destination in staged_outputs:
            os.replace(staged_path, destination)
            installed.append(destination)
    except Exception:
        for destination in installed:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        for destination, backup in backups.items():
            if backup.exists():
                os.replace(backup, destination)
        raise


def generate_plots(summary_path, output_dir):
    experiments = load_experiments(summary_path)
    fps_points = select_points(experiments, "fps")
    memory_points = select_points(experiments, "peak_allocated_mib")

    # Validate before creating an output directory or any staged image.
    require_enough_points(fps_points, memory_points)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tradeoff-plots-", dir=str(output_dir)
    ) as temporary_directory:
        stage_dir = Path(temporary_directory)
        names = (
            "accuracy_vs_fps.png",
            "accuracy_vs_fps.svg",
            "accuracy_vs_memory.png",
            "accuracy_vs_memory.svg",
        )
        staged = {name: stage_dir / name for name in names}

        render_plot(
            plt,
            fps_points,
            "BEVFormer Accuracy vs Throughput",
            "Throughput (FPS, higher is better)",
            staged["accuracy_vs_fps.png"],
            staged["accuracy_vs_fps.svg"],
        )
        render_plot(
            plt,
            memory_points,
            "BEVFormer Accuracy vs GPU Memory",
            "Peak allocated GPU memory (MiB, lower is better)",
            staged["accuracy_vs_memory.png"],
            staged["accuracy_vs_memory.svg"],
        )

        staged_outputs = [
            (staged[name], output_dir / name) for name in names
        ]
        commit_outputs(staged_outputs)
    return [output_dir / name for name in names]


def main(argv=None):
    args = parse_args(argv)
    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        outputs = generate_plots(summary_path, output_dir)
    except (PlotInputError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
