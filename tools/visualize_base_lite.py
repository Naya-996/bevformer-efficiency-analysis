#!/usr/bin/env python3
"""Create objective GT/Base/Lite nuScenes camera comparisons.

The script selects four validation keyframes from different scenes using only
ground-truth metadata: dense traffic, ego turning, low visibility, and distant
objects. Predictions never influence scene selection.
"""

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import BoxVisibility, box_in_image
from nuscenes.eval.detection.utils import category_to_detection_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)
DETECTION_CLASSES = {
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
}
ROW_COLORS = {
    "GT": "#00B050",
    "Base": "#0072B2",
    "Lite": "#E69F00",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render objective nuScenes GT/Base/Lite camera comparisons."
    )
    parser.add_argument(
        "--dataroot",
        default="data/nuscenes",
        help="nuScenes root (default: data/nuscenes)",
    )
    parser.add_argument(
        "--base-results",
        default=(
            "test/bevformer_base/Mon_Aug_10_17_02_09_2026/"
            "pts_bbox/results_nusc.json"
        ),
    )
    parser.add_argument(
        "--lite-results",
        default=(
            "test/bev150/Wed_Aug_12_12_18_15_2026/"
            "pts_bbox/results_nusc.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="docs/assets/visualizations",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--manifest",
        default="experiments/lite/visualization_manifest.json",
    )
    args = parser.parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1]")
    if args.max_distance <= 0:
        parser.error("--max-distance must be positive")
    return args


def absolute_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_results(path):
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    results = document.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError(f"{path}: missing non-empty results mapping")
    return document, results


def wrap_angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def ego_pose(nusc, sample):
    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    return nusc.get("ego_pose", lidar_sd["ego_pose_token"])


def sample_frame_index(nusc, sample):
    index = 0
    previous = sample["prev"]
    while previous:
        index += 1
        previous = nusc.get("sample", previous)["prev"]
    return index


def annotation_stats(nusc, sample, ego_xy):
    dense = 0
    far = 0
    occluded = 0
    valid = 0
    for token in sample["anns"]:
        annotation = nusc.get("sample_annotation", token)
        detection_name = category_to_detection_name(annotation["category_name"])
        if detection_name not in DETECTION_CLASSES:
            continue
        distance = float(
            np.linalg.norm(np.asarray(annotation["translation"][:2]) - ego_xy)
        )
        if distance > 50.0:
            continue
        valid += 1
        if distance <= 30.0:
            dense += 1
        if 35.0 <= distance <= 50.0:
            far += 1
        if annotation.get("visibility_token") in {"1", "2"}:
            occluded += 1
    return valid, dense, far, occluded


def turning_score(nusc, sample):
    earlier = sample
    later = sample
    for _ in range(2):
        if earlier["prev"]:
            earlier = nusc.get("sample", earlier["prev"])
        if later["next"]:
            later = nusc.get("sample", later["next"])
    earlier_pose = ego_pose(nusc, earlier)
    later_pose = ego_pose(nusc, later)
    earlier_yaw = Quaternion(earlier_pose["rotation"]).yaw_pitch_roll[0]
    later_yaw = Quaternion(later_pose["rotation"]).yaw_pitch_roll[0]
    yaw_change = abs(wrap_angle(later_yaw - earlier_yaw))
    displacement = float(
        np.linalg.norm(
            np.asarray(later_pose["translation"][:2])
            - np.asarray(earlier_pose["translation"][:2])
        )
    )
    return yaw_change, displacement


def select_scenes(nusc, eligible_tokens):
    candidates = []
    for token in eligible_tokens:
        sample = nusc.get("sample", token)
        frame_index = sample_frame_index(nusc, sample)
        if frame_index < 3:
            continue
        pose = ego_pose(nusc, sample)
        ego_xy = np.asarray(pose["translation"][:2])
        valid, dense, far, occluded = annotation_stats(nusc, sample, ego_xy)
        yaw_change, displacement = turning_score(nusc, sample)
        scene = nusc.get("scene", sample["scene_token"])
        candidates.append(
            {
                "sample_token": token,
                "scene_token": sample["scene_token"],
                "scene_name": scene["name"],
                "frame_index": frame_index,
                "valid_gt_50m": valid,
                "dense_gt_30m": dense,
                "far_gt_35_50m": far,
                "low_visibility_gt_50m": occluded,
                "yaw_change_rad": yaw_change,
                "displacement_m": displacement,
            }
        )

    selectors = (
        ("dense", lambda row: (row["dense_gt_30m"], row["valid_gt_50m"])),
        (
            "turning",
            lambda row: (
                row["yaw_change_rad"] if row["displacement_m"] >= 1.0 else -1.0,
                row["dense_gt_30m"],
            ),
        ),
        (
            "occlusion",
            lambda row: (
                row["low_visibility_gt_50m"],
                row["valid_gt_50m"],
            ),
        ),
        ("far", lambda row: (row["far_gt_35_50m"], row["valid_gt_50m"])),
    )
    selected = []
    used_scenes = set()
    for reason, key in selectors:
        available = [row for row in candidates if row["scene_token"] not in used_scenes]
        if not available:
            raise RuntimeError(f"no unique-scene candidate available for {reason}")
        winner = max(available, key=lambda row: (key(row), row["sample_token"]))
        winner = dict(winner)
        winner["selection_reason"] = reason
        selected.append(winner)
        used_scenes.add(winner["scene_token"])
    return selected


def global_prediction_boxes(records, ego_xy, threshold, max_distance):
    boxes = []
    for record in records:
        score = float(record.get("detection_score", -1.0))
        name = record.get("detection_name")
        if score < threshold or name not in DETECTION_CLASSES:
            continue
        distance = float(
            np.linalg.norm(np.asarray(record["translation"][:2]) - ego_xy)
        )
        if distance > max_distance:
            continue
        boxes.append(
            Box(
                center=record["translation"],
                size=record["size"],
                orientation=Quaternion(record["rotation"]),
                name=name,
                score=score,
                token="prediction",
            )
        )
    return boxes


def transform_to_camera(box, pose, calibration):
    transformed = box.copy()
    transformed.translate(-np.asarray(pose["translation"]))
    transformed.rotate(Quaternion(pose["rotation"]).inverse)
    transformed.translate(-np.asarray(calibration["translation"]))
    transformed.rotate(Quaternion(calibration["rotation"]).inverse)
    return transformed


def boxes_for_camera(nusc, sample, camera, row_name, records, threshold, max_distance):
    sample_data = nusc.get("sample_data", sample["data"][camera])
    calibration = nusc.get(
        "calibrated_sensor", sample_data["calibrated_sensor_token"]
    )
    pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
    intrinsic = np.asarray(calibration["camera_intrinsic"])
    image_size = (sample_data["width"], sample_data["height"])
    ego_xy = np.asarray(pose["translation"][:2])

    if row_name == "GT":
        global_boxes = []
        for annotation_token in sample["anns"]:
            annotation = nusc.get("sample_annotation", annotation_token)
            name = category_to_detection_name(annotation["category_name"])
            if name not in DETECTION_CLASSES:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(annotation["translation"][:2]) - ego_xy
                )
            )
            if distance > max_distance:
                continue
            box = nusc.get_box(annotation_token)
            box.name = name
            global_boxes.append(box)
    else:
        global_boxes = global_prediction_boxes(
            records, ego_xy, threshold, max_distance
        )

    visible = []
    for box in global_boxes:
        camera_box = transform_to_camera(box, pose, calibration)
        if box_in_image(
            camera_box,
            intrinsic,
            image_size,
            vis_level=BoxVisibility.ANY,
        ):
            visible.append(camera_box)
    return sample_data, intrinsic, visible


def render_scene(nusc, selection, base_records, lite_records, output, threshold, max_distance):
    sample = nusc.get("sample", selection["sample_token"])
    rows = (
        ("GT", None),
        ("Base", base_records),
        ("Lite", lite_records),
    )
    figure, axes = plt.subplots(3, 6, figsize=(24, 11.2), dpi=120)
    for row_index, (row_name, records) in enumerate(rows):
        for column_index, camera in enumerate(CAMERAS):
            axis = axes[row_index, column_index]
            sample_data, intrinsic, boxes = boxes_for_camera(
                nusc,
                sample,
                camera,
                row_name,
                records,
                threshold,
                max_distance,
            )
            image_path = nusc.get_sample_data_path(sample_data["token"])
            with Image.open(image_path) as image:
                axis.imshow(image)
                width, height = image.size
            color = ROW_COLORS[row_name]
            for box in boxes:
                box.render(
                    axis,
                    view=intrinsic,
                    normalize=True,
                    colors=(color, color, color),
                    linewidth=1.4,
                )
            axis.set_xlim(0, width)
            axis.set_ylim(height, 0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(camera.replace("CAM_", ""), fontsize=10)
            if column_index == 0:
                axis.text(
                    -0.035,
                    0.5,
                    row_name,
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=13,
                    fontweight="bold",
                    color=color,
                )

    reason = selection["selection_reason"]
    figure.suptitle(
        f"{reason.title()} | {selection['scene_name']} | "
        f"frame {selection['frame_index']} | token {selection['sample_token'][:8]}…\n"
        f"score ≥ {threshold:.2f}, range ≤ {max_distance:.0f} m",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.01, 1.0, 0.94), h_pad=0.5, w_pad=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, dpi=180, bbox_inches="tight", facecolor="white")
        os.replace(temporary, output)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def atomic_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    args = parse_args()
    dataroot = absolute_path(args.dataroot)
    base_path = absolute_path(args.base_results)
    lite_path = absolute_path(args.lite_results)
    output_dir = absolute_path(args.output_dir)
    manifest_path = absolute_path(args.manifest)
    for path in (dataroot, base_path, lite_path):
        if not path.exists():
            raise FileNotFoundError(path)

    base_document, base_results = load_results(base_path)
    lite_document, lite_results = load_results(lite_path)
    if set(base_results) != set(lite_results):
        raise ValueError("Base and Lite result token sets differ")

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(dataroot), verbose=False)
    eligible_tokens = sorted(base_results)
    selections = select_scenes(nusc, eligible_tokens)
    for selection in selections:
        token = selection["sample_token"]
        filename = f"{selection['selection_reason']}_{token}.png"
        output = output_dir / filename
        render_scene(
            nusc,
            selection,
            base_results[token],
            lite_results[token],
            output,
            args.score_threshold,
            args.max_distance,
        )
        selection["output"] = output.relative_to(PROJECT_ROOT).as_posix()
        print(output)

    manifest = {
        "schema_version": 1,
        "selection_uses_predictions": False,
        "dataroot": str(dataroot),
        "base_results": str(base_path),
        "lite_results": str(lite_path),
        "base_meta": base_document.get("meta"),
        "lite_meta": lite_document.get("meta"),
        "score_threshold": args.score_threshold,
        "max_distance_m": args.max_distance,
        "rows": ["GT", "Base", "Lite"],
        "selections": selections,
    }
    atomic_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
