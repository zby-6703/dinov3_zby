r"""Run ROI inference and export LabelMe JSON plus visualization images.

Usage from the project root:
    python shipdraft_mtl\tools\predict_e2e.py --input E:\data\PingLuRiver\dataset\ShipDraft_e2e\test\pl_3.jpg --output-dir outputs\predict_e2e --device cuda

Directory inference uses the same command with ``--input`` set to an image
directory. The default config and checkpoint are defined below; use
``--config`` and ``--weights`` to override them.
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


DEFAULT_CONFIG = PROJECT_ROOT / "shipdraft_mtl" / "configs" / "default_e2e.yml"
DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "outputs"
    / "shipdraft_mtl"
    / "direct_depth_mtl_dinov3_convnext_tiny"
    / "best.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image or image directory")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and visualization files")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Model YAML config")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Model checkpoint")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--score-thresh", type=float, default=0.1, help="Character confidence threshold")
    parser.add_argument("--waterline-thresh", type=float, default=0.3, help="Waterline probability threshold")
    parser.add_argument("--waterline-spacing", type=int, default=12, help="Exported waterline point spacing in pixels")
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
    return np.asarray([
        np.median(padded[index : index + kernel_size]) for index in range(len(values))
    ], dtype=np.float32)


def extract_waterline(
    probability: np.ndarray,
    threshold: float,
    spacing: int,
) -> Tuple[List[List[float]], np.ndarray, float]:
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim == 3:
        probability = probability[0]
    if probability.ndim != 2 or probability.size == 0:
        return [], np.zeros(probability.shape[-2:], dtype=np.uint8), 0.0

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


def character_shapes(
    points: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    threshold: float,
) -> List[Dict]:
    shapes = []
    for point, score, label in zip(points, scores, labels):
        if float(score) < threshold:
            continue
        class_id = int(label)
        class_name = class_names[class_id] if class_id < len(class_names) else str(class_id)
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
    return {
        "version": "5.5.0",
        "flags": {},
        "shapes": all_shapes,
        "imagePath": relative_image,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
        "draft depth": float(draft_depth),
        "draft_depth_valid": bool(draft_confidence >= valid_threshold),
        "draft_depth_confidence": float(draft_confidence),
    }


def draw_visualization(
    image: np.ndarray,
    shapes: Sequence[Dict],
    waterline_mask: np.ndarray,
    draft_depth: float,
    draft_confidence: float,
) -> np.ndarray:
    canvas = image.copy()
    if waterline_mask.shape == image.shape[:2] and waterline_mask.any():
        overlay = canvas.copy()
        overlay[waterline_mask > 0] = (40, 210, 80)
        canvas = cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0)

    for shape in shapes:
        points = np.rint(np.asarray(shape["points"], dtype=np.float32)).astype(np.int32)
        if shape["label"] == "waterline":
            if len(points) >= 2:
                cv2.polylines(canvas, [points], False, (30, 240, 70), 2, cv2.LINE_AA)
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

    text_lines = [f"Draft: {draft_depth:.3f} m", f"Valid: {draft_confidence:.3f}"]
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


def predict_image(predictor: DraftFormerPredictor, image_path: Path, output_dir: Path, args) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    output = predictor(image)
    points = output["points"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    segmentation = output["sem_seg"].detach().cpu().numpy()
    draft_depth = float(output["draft_depth"].detach().cpu())
    draft_confidence = float(output["draft_valid"].detach().cpu())

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
    )
    json_path.write_text(json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8")
    visualization = draw_visualization(image, label["shapes"], waterline_mask, draft_depth, draft_confidence)
    visualization_path = output_dir / f"{image_path.stem}_vis.jpg"
    if not cv2.imwrite(str(visualization_path), visualization):
        raise ValueError(f"Failed to write visualization: {visualization_path}")
    print(
        f"{image_path.name}: draft={draft_depth:.3f}m valid={draft_confidence:.3f} "
        f"characters={len(shapes)} waterline_points={len(waterline_points)}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [Path(path) for path in get_image_files(args.input)]
    if not image_files:
        raise ValueError(f"No input images found: {args.input}")

    predictor = DraftFormerPredictor(
        config_file=args.config,
        weights_path=args.weights,
        device=args.device,
        opts={
            "Architecture": {"return_auxiliary_outputs": True},
            "PostProcess": {"score_thresh": args.score_thresh},
        },
    )
    for image_path in image_files:
        predict_image(predictor, image_path, output_dir, args)


if __name__ == "__main__":
    main()
