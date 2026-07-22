r"""Run ROI inference and export LabelMe JSON plus visualization images.

Examples (from project root ``dinov3_zby``)::

    python shipdraft_mtl/tools/predict_e2e.py ^
      --config shipdraft_mtl/configs/stage1_keypoints.yml ^
      --weights outputs/shipdraft_mtl/stage1_keypoints/best.pth ^
      --input E:/data/PingLuRiver/dataset/ShipDraft_e2e/test ^
      --output-dir outputs/shipdraft_mtl/stage1_keypoints/predict

On Windows, prefer forward slashes in paths, or quote the argument, so that
``\b`` in ``\best.pth`` is not eaten by shell/string escaping.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shipdraft_mtl.engine.predictor import DraftFormerPredictor, get_image_files


def normalize_user_path(path: str) -> Path:
    """Turn a user path into an absolute Path (safe on Windows)."""
    text = str(path).strip().strip('"').strip("'")
    # If the string was already corrupted by Python ``\b`` in source defaults,
    # callers should pass CLI args (shell does not interpret \b like Python does).
    text = text.replace("\\", "/")
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image or image directory")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and visualization files")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML, e.g. shipdraft_mtl/configs/stage1_keypoints.yml",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Checkpoint, e.g. outputs/shipdraft_mtl/stage1_keypoints/best.pth",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--score-thresh", type=float, default=0.1, help="Character confidence threshold")
    parser.add_argument("--waterline-thresh", type=float, default=0.3, help="Waterline score threshold")
    parser.add_argument("--waterline-spacing", type=int, default=12, help="Legacy mask-export spacing (px)")
    parser.add_argument("--valid-thresh", type=float, default=0.5, help="Draft validity threshold")
    return parser.parse_args()


def _longest_true_run(values: np.ndarray) -> Tuple[int, int] | None:
    indices = np.flatnonzero(values)
    if not len(indices):
        return None
    splits = np.where(np.diff(indices) > 1)[0] + 1
    groups = np.split(indices, splits)
    longest = max(groups, key=len)
    return int(longest[0]), int(longest[-1])


def _median_smooth(values: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float32)
    kernel_size = min(kernel_size, len(values) if len(values) % 2 else len(values) - 1)
    kernel_size = max(kernel_size, 3)
    pad = kernel_size // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + kernel_size]) for index in range(len(values))],
        dtype=np.float32,
    )


def extract_waterline(
    probability: np.ndarray,
    threshold: float,
    spacing: int,
) -> Tuple[List[List[float]], np.ndarray, float]:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim == 3:
        probability = probability[0]
    if probability.ndim != 2 or probability.size == 0:
        shape = probability.shape[-2:] if probability.ndim >= 2 else (0, 0)
        return [], np.zeros(shape, dtype=np.uint8), 0.0

    maximum = probability.max(axis=0)
    effective_threshold = float(threshold)
    valid_columns = maximum >= effective_threshold
    run = _longest_true_run(valid_columns)
    if run is None and float(probability.max()) > 0:
        effective_threshold = max(0.05, float(probability.max()) * 0.5)
        valid_columns = maximum >= effective_threshold
        run = _longest_true_run(valid_columns)
    binary_mask = (probability >= effective_threshold).astype(np.uint8)
    if run is None:
        return [], binary_mask, 0.0

    start, end = run
    dense_x = np.arange(start, end + 1, dtype=np.int32)
    dense_y = probability[:, dense_x].argmax(axis=0).astype(np.float32)
    dense_y = _median_smooth(dense_y)
    sample_indices = list(range(0, len(dense_x), max(int(spacing), 1)))
    if sample_indices[-1] != len(dense_x) - 1:
        sample_indices.append(len(dense_x) - 1)
    points = [[float(dense_x[index]), float(dense_y[index])] for index in sample_indices]
    confidence = float(np.mean(probability[np.rint(dense_y).astype(np.int32), dense_x]))
    return points, binary_mask, confidence


def character_class_names(class_names: Sequence[str]) -> List[str]:
    return [name for name in class_names if str(name).lower() != "waterline"]


def character_shapes(
    points: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> List[Dict]:
    names = character_class_names(class_names)
    shapes = []
    for point, score, label in zip(points, scores, labels):
        if float(score) < threshold:
            continue
        class_id = int(label)
        class_name = names[class_id] if class_id < len(names) else str(class_id)
        shapes.append(
            {
                "label": class_name,
                "points": [[float(point[0]), float(point[1])]],
                "group_id": None,
                "description": f"confidence={float(score):.6f}",
                "shape_type": "point",
                "flags": {"score": float(score)},
            }
        )
    return shapes


def build_labelme_json(
    image_path: Path,
    json_path: Path,
    image_shape: Tuple[int, int],
    shapes: List[Dict],
    waterline_points: List[List[float]],
    waterline_confidence: float,
    draft_depth: float,
    draft_confidence: float,
    valid_threshold: float,
    has_depth: bool,
) -> Dict:
    all_shapes = list(shapes)
    if waterline_points:
        all_shapes.append(
            {
                "label": "waterline",
                "points": waterline_points,
                "group_id": None,
                "description": f"confidence={waterline_confidence:.6f}",
                "shape_type": "linestrip" if len(waterline_points) >= 2 else "point",
                "flags": {"score": waterline_confidence},
            }
        )
    height, width = image_shape
    relative_image = os.path.relpath(image_path.resolve(), json_path.parent.resolve()).replace("\\", "/")
    payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": all_shapes,
        "imagePath": relative_image,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }
    if has_depth:
        payload["draft depth"] = float(draft_depth)
        payload["draft_depth_valid"] = bool(draft_confidence >= valid_threshold)
        payload["draft_depth_confidence"] = float(draft_confidence)
    else:
        payload["draft depth"] = None
        payload["draft_depth_valid"] = False
        payload["draft_depth_confidence"] = 0.0
        payload["flags"]["stage_without_depth"] = True
    return payload


def draw_visualization(
    image: np.ndarray,
    shapes: Sequence[Dict],
    waterline_mask: np.ndarray,
    draft_depth: float,
    draft_confidence: float,
    has_depth: bool,
) -> np.ndarray:
    canvas = image.copy()
    if waterline_mask is not None and waterline_mask.shape == image.shape[:2] and waterline_mask.any():
        overlay = canvas.copy()
        overlay[waterline_mask > 0] = (40, 210, 80)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)

    for shape in shapes:
        points = np.rint(np.asarray(shape["points"], dtype=np.float32)).astype(np.int32)
        if shape["label"] == "waterline":
            if len(points) >= 2:
                cv2.polylines(canvas, [points], False, (30, 240, 70), 2, cv2.LINE_AA)
            elif len(points) == 1:
                cv2.circle(canvas, tuple(points[0]), 4, (30, 240, 70), -1, cv2.LINE_AA)
            continue
        x, y = points[0].tolist()
        cv2.circle(canvas, (x, y), 4, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 6, (10, 30, 30), 1, cv2.LINE_AA)
        label = str(shape["label"])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
        text_x = x + 7
        if text_x + text_size[0] >= canvas.shape[1]:
            text_x = max(1, x - text_size[0] - 7)
        cv2.putText(
            canvas,
            label,
            (text_x, max(y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if has_depth:
        text_lines = [f"Draft: {draft_depth:.3f} m", f"Valid: {draft_confidence:.3f}"]
    else:
        text_lines = ["Draft: N/A (no depth head)", "Characters + waterline only"]
    base_scale = 0.5
    widest = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, base_scale, 1)[0][0] for line in text_lines)
    font_scale = base_scale * min(1.0, max((canvas.shape[1] - 18) / max(widest, 1), 0.6))
    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0] for line in text_lines]
    panel_width = min(max(width for width, _ in sizes) + 14, canvas.shape[1] - 4)
    line_height = max(height for _, height in sizes) + 7
    cv2.rectangle(canvas, (4, 4), (4 + panel_width, 8 + line_height * len(text_lines)), (20, 20, 20), -1)
    for index, line in enumerate(text_lines):
        cv2.putText(
            canvas,
            line,
            (9, 4 + line_height * (index + 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _tensor_to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def predict_image(
    predictor: DraftFormerPredictor,
    image_path: Path,
    output_dir: Path,
    args,
    has_depth: bool,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    output = predictor(image)

    points = output.get("points")
    scores = output.get("scores")
    labels = output.get("labels")
    if points is None:
        points = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
    else:
        points = points.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

    segmentation = output.get("sem_seg")
    if segmentation is not None:
        segmentation = segmentation.detach().cpu().numpy()
    else:
        segmentation = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float32)

    draft_depth = _tensor_to_float(output.get("draft_depth"), 0.0)
    draft_confidence = _tensor_to_float(output.get("draft_valid"), 0.0)
    if not has_depth:
        draft_depth = 0.0
        draft_confidence = 0.0

    if "waterline_points" in output and len(output["waterline_points"]) > 0:
        wl = output["waterline_points"].detach().cpu().numpy()
        wl_scores = (
            output["waterline_scores"].detach().cpu().numpy()
            if "waterline_scores" in output and len(output.get("waterline_scores", []))
            else None
        )
        waterline_points = [[float(x), float(y)] for x, y in wl]
        waterline_confidence = float(wl_scores.mean()) if wl_scores is not None and len(wl_scores) else 1.0
        if segmentation.ndim == 3:
            waterline_mask = (segmentation[0] > args.waterline_thresh).astype(np.uint8)
        else:
            waterline_mask = (segmentation > args.waterline_thresh).astype(np.uint8)
    else:
        waterline_points, waterline_mask, waterline_confidence = extract_waterline(
            segmentation,
            threshold=args.waterline_thresh,
            spacing=args.waterline_spacing,
        )

    shapes = character_shapes(
        points,
        scores,
        labels,
        predictor.class_names,
        threshold=args.score_thresh,
    )
    json_path = output_dir / f"{image_path.stem}.json"
    label = build_labelme_json(
        image_path=image_path,
        json_path=json_path,
        image_shape=image.shape[:2],
        shapes=shapes,
        waterline_points=waterline_points,
        waterline_confidence=waterline_confidence,
        draft_depth=draft_depth,
        draft_confidence=draft_confidence,
        valid_threshold=args.valid_thresh,
        has_depth=has_depth,
    )
    json_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    visualization = draw_visualization(
        image,
        label["shapes"],
        waterline_mask,
        draft_depth,
        draft_confidence,
        has_depth=has_depth,
    )
    visualization_path = output_dir / f"{image_path.stem}_vis.jpg"
    if not cv2.imwrite(str(visualization_path), visualization):
        raise ValueError(f"Failed to write visualization: {visualization_path}")

    draft_msg = f"draft={draft_depth:.3f}m valid={draft_confidence:.3f}" if has_depth else "draft=N/A"
    print(
        f"{image_path.name}: {draft_msg} "
        f"characters={len(shapes)} waterline_points={len(waterline_points)} "
        f"-> {visualization_path.name}"
    )


def main() -> None:
    args = parse_args()
    config_path = normalize_user_path(args.config)
    weights_path = normalize_user_path(args.weights)
    output_dir = normalize_user_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    image_files = [Path(path) for path in get_image_files(args.input)]
    if not image_files:
        # Try resolving relative to project root.
        try:
            image_files = [Path(path) for path in get_image_files(str(normalize_user_path(args.input)))]
        except Exception:
            image_files = []
    if not image_files:
        raise ValueError(f"No input images found: {args.input}")

    stems = [path.stem.casefold() for path in image_files]
    duplicate_stems = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicate_stems:
        raise ValueError(
            "Input images would overwrite outputs because they share a filename stem: "
            + ", ".join(duplicate_stems)
        )
    input_paths = {path.resolve() for path in image_files}
    output_root = output_dir.resolve()
    if any(path.parent.resolve() == output_root for path in image_files):
        raise ValueError("Output directory must differ from the input image directory")
    output_paths = {
        target.resolve()
        for path in image_files
        for target in (output_dir / f"{path.stem}.json", output_dir / f"{path.stem}_vis.jpg")
    }
    collisions = sorted(input_paths & output_paths)
    if collisions:
        raise ValueError(
            "Output paths would overwrite input images: " + ", ".join(str(path) for path in collisions)
        )

    print(f"config : {config_path}")
    print(f"weights: {weights_path}")
    print(f"images : {len(image_files)}")

    predictor = DraftFormerPredictor(
        config_file=str(config_path),
        weights_path=str(weights_path),
        device=args.device,
        opts={
            "Architecture": {"return_auxiliary_outputs": True},
            "PostProcess": {
                "score_thresh": args.score_thresh,
                "waterline_score_thresh": args.waterline_thresh,
            },
        },
    )
    # Whether this checkpoint/config has a draft head is detected from the model.
    has_depth = getattr(predictor.model, "direct_depth_head", None) is not None
    print(f"has_depth_head: {has_depth}")

    for image_path in image_files:
        predict_image(predictor, image_path, output_dir, args, has_depth=has_depth)


if __name__ == "__main__":
    main()
