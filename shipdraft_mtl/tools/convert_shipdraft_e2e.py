#!/usr/bin/env python3
"""Convert ShipDraft LabelMe labels for end-to-end draft reading.

Transformations applied to each JSON label:
1. Water polygon -> waterline linestrip (upper boundary of the water region).
2. Character rectangles -> point at the bottom-center of each box.
3. ``draft depth`` equal to 0 is rewritten as JSON null (unknown / unreadable).
4. Other metadata fields are preserved; images are copied or hardlinked.

Example:
  python shipdraft_mtl/tools/convert_shipdraft_e2e.py \\
    --src E:/data/PingLuRiver/dataset/ShipDraft \\
    --dst E:/data/PingLuRiver/dataset/ShipDraft_e2e
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
WATER_LABELS = {"water", "waterline"}
DEFAULT_WATERLINE_LABEL = "waterline"


def _as_xy(point: Sequence[Any]) -> List[float]:
    return [float(point[0]), float(point[1])]


def _dedupe_closed_ring(points: Sequence[Sequence[float]]) -> List[List[float]]:
    pts = [_as_xy(p) for p in points]
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    return pts


def extract_upper_boundary(
    points: Sequence[Sequence[float]],
    *,
    bottom_band_ratio: float = 0.20,
) -> List[List[float]]:
    """Extract the free-surface waterline from a water polygon.

    Original water polygons usually look like::

        (top-left) ---- (top-mid) ---- (top-right)
            |                              |
        (bottom-left) -------------- (bottom-right)

    Left/right extremes are often **bottom corners** (image edge + water bed).
    Walking leftmost→rightmost therefore pulls the polyline into a triangle that
    includes the bed (LabelMe failure on samples like 000255).

    Strategy (bottom rejection, not top-band keep):
    1. Drop vertices in the bottom band of the polygon bbox (near max y).
       These are water-bed corners / the bottom edge of the mask.
    2. Keep the remaining vertices — free surface, including **sloped** waterlines
       that span a large fraction of the bbox height.
    3. Sort left→right; if several points share nearly the same x, keep the
       highest one (smallest y).

    Why not "per-column min-y of the filled polygon"?
    When the mask is loose, edge columns may only contain the bottom corner, so
    column-wise min-y is the bed, not the free surface.
    """
    pts = _dedupe_closed_ring(points)
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [pts[0]]

    ys = [p[1] for p in pts]
    min_y = min(ys)
    max_y = max(ys)
    height = max(max_y - min_y, 1e-6)

    # Anything near the polygon floor is water bed / bottom edge — never waterline.
    bottom_y_min = max_y - float(bottom_band_ratio) * height
    surface_pts = [p for p in pts if p[1] < bottom_y_min - 1e-6]

    if len(surface_pts) < 2:
        # Flat / degenerate polygon: keep the top-most distinct vertices only.
        ordered = sorted(pts, key=lambda p: (p[1], p[0]))
        surface_pts = []
        for p in ordered:
            if p[1] >= bottom_y_min - 1e-6:
                continue
            surface_pts.append(p)
            if len(surface_pts) >= max(2, min(len(pts) - 1, 8)):
                break
        if len(surface_pts) < 2:
            surface_pts = ordered[:2]

    # Left → right; near-duplicate x keeps the higher free-surface point.
    surface_pts = sorted(surface_pts, key=lambda p: (p[0], p[1]))
    cleaned: List[List[float]] = []
    for p in surface_pts:
        q = [float(p[0]), float(p[1])]
        if cleaned and abs(cleaned[-1][0] - q[0]) < 1e-3:
            if q[1] < cleaned[-1][1]:
                cleaned[-1] = q
            continue
        cleaned.append(q)

    return cleaned


def rectangle_bottom_center(points: Sequence[Sequence[float]]) -> List[float]:
    """Bottom-center of a LabelMe rectangle defined by two opposite corners."""
    if len(points) < 2:
        raise ValueError("Rectangle requires two corner points")
    xs = [float(p[0]) for p in points[:2]]
    ys = [float(p[1]) for p in points[:2]]
    x_min, x_max = min(xs), max(xs)
    y_max = max(ys)  # larger y is the bottom edge in image coordinates
    return [(x_min + x_max) / 2.0, y_max]


def normalize_draft_depth(value: Any) -> Any:
    """Keep draft depth, but map numeric 0 to null (unreadable waterline reading)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number == 0.0:
        return None
    return number


def _make_point_shape(label: str, xy: Sequence[float], group_id=None, flags=None) -> Dict[str, Any]:
    return {
        "label": str(label),
        "points": [[float(xy[0]), float(xy[1])]],
        "group_id": group_id,
        "description": "",
        "shape_type": "point",
        "flags": flags or {},
    }


def _make_waterline_shape(
    points: Sequence[Sequence[float]],
    label: str = DEFAULT_WATERLINE_LABEL,
    group_id=None,
    flags=None,
) -> Dict[str, Any]:
    shape_type = "linestrip" if len(points) >= 2 else "point"
    return {
        "label": label,
        "points": [[float(p[0]), float(p[1])] for p in points],
        "group_id": group_id,
        "description": "upper boundary of original water polygon",
        "shape_type": shape_type,
        "flags": flags or {},
    }


def convert_shapes(
    shapes: Sequence[MappingLike],
    *,
    waterline_label: str = DEFAULT_WATERLINE_LABEL,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Convert LabelMe shapes; returns new shapes and a stats counter."""
    stats: Counter = Counter()
    new_shapes: List[Dict[str, Any]] = []

    for shape in shapes:
        shape_type = str(shape.get("shape_type", "")).lower()
        label = str(shape.get("label", "")).strip()
        points = shape.get("points") or []

        if shape_type == "rectangle":
            try:
                bottom_center = rectangle_bottom_center(points)
            except ValueError:
                stats["rectangle_invalid"] += 1
                continue
            new_shapes.append(
                _make_point_shape(
                    label=label,
                    xy=bottom_center,
                    group_id=shape.get("group_id"),
                    flags=shape.get("flags") or {},
                )
            )
            stats["rectangle_to_point"] += 1
            continue

        if shape_type == "polygon" and label.lower() in WATER_LABELS:
            waterline = extract_upper_boundary(points)
            if len(waterline) == 0:
                stats["water_empty"] += 1
                continue
            new_shapes.append(
                _make_waterline_shape(
                    waterline,
                    label=waterline_label,
                    group_id=shape.get("group_id"),
                    flags=shape.get("flags") or {},
                )
            )
            stats["water_to_waterline"] += 1
            stats["waterline_points"] += len(waterline)
            continue

        # Drop legacy auxiliary points / other shapes; e2e labels only keep
        # character points + waterline derived above.
        stats[f"dropped_{shape_type or 'unknown'}"] += 1

    return new_shapes, stats


# For typing without importing Mapping from typing repeatedly in runtime hot path
MappingLike = Dict[str, Any]


def convert_labelme_json(
    data: Dict[str, Any],
    *,
    waterline_label: str = DEFAULT_WATERLINE_LABEL,
) -> Tuple[Dict[str, Any], Counter]:
    out = dict(data)
    shapes, stats = convert_shapes(data.get("shapes") or [], waterline_label=waterline_label)
    out["shapes"] = shapes

    if "draft depth" in out:
        original = out["draft depth"]
        converted = normalize_draft_depth(original)
        out["draft depth"] = converted
        if original is not None and converted is None:
            stats["draft_depth_zero_to_null"] += 1
        elif converted is None:
            stats["draft_depth_null"] += 1
        else:
            stats["draft_depth_kept"] += 1
    else:
        stats["draft_depth_missing"] += 1

    # Optional explicit flag for training code.
    out["draft_depth_valid"] = out.get("draft depth") is not None
    return out, stats


def _resolve_image_path(label_path: Path, data: Dict[str, Any]) -> Optional[Path]:
    image_path = data.get("imagePath")
    if image_path:
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = label_path.parent / candidate
        if candidate.is_file():
            return candidate.resolve()

    stem = label_path.with_suffix("")
    for ext in IMAGE_EXTENSIONS:
        candidate = Path(str(stem) + ext)
        if candidate.is_file():
            return candidate.resolve()
    return None


def _copy_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def iter_split_dirs(src_root: Path) -> Iterable[Path]:
    preferred = ["train", "val", "valid", "test"]
    found = []
    for name in preferred:
        path = src_root / name
        if path.is_dir():
            found.append(path)
    if found:
        return found
    # Flat dataset: treat root as one split.
    if any(src_root.glob("*.json")):
        return [src_root]
    return []


def convert_dataset(
    src_root: Path,
    dst_root: Path,
    *,
    waterline_label: str = DEFAULT_WATERLINE_LABEL,
    image_mode: str = "copy",
    dry_run: bool = False,
) -> Counter:
    src_root = src_root.resolve()
    dst_root = dst_root.resolve()
    if src_root == dst_root:
        raise ValueError("dst must be different from src to avoid overwriting the original dataset")

    totals: Counter = Counter()
    split_dirs = list(iter_split_dirs(src_root))
    if not split_dirs:
        raise FileNotFoundError(f"No split directories or JSON labels under {src_root}")

    for split_dir in split_dirs:
        rel_split = split_dir.relative_to(src_root) if split_dir != src_root else Path(".")
        label_paths = sorted(split_dir.rglob("*.json"))
        if not label_paths:
            print(f"[warn] no json in {split_dir}")
            continue

        for label_path in label_paths:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            converted, stats = convert_labelme_json(data, waterline_label=waterline_label)
            totals.update(stats)
            totals["json_total"] += 1

            rel = label_path.relative_to(src_root)
            dst_label = dst_root / rel

            image_src = _resolve_image_path(label_path, data)
            if image_src is None:
                totals["image_missing"] += 1
            else:
                # Keep original imagePath basename when possible.
                image_name = Path(converted.get("imagePath") or image_src.name).name
                converted["imagePath"] = image_name
                dst_image = dst_label.parent / image_name
                if not dry_run:
                    _copy_image(image_src, dst_image, image_mode)
                totals["image_copied"] += 1

            if not dry_run:
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_label, "w", encoding="utf-8") as f:
                    json.dump(converted, f, ensure_ascii=False, indent=2)

            if totals["json_total"] <= 3 or totals["json_total"] % 500 == 0:
                n_pts = sum(1 for s in converted["shapes"] if s["shape_type"] == "point")
                n_wl = sum(1 for s in converted["shapes"] if s["label"] == waterline_label)
                print(
                    f"[{totals['json_total']}] {rel.as_posix()} "
                    f"points={n_pts} waterline={n_wl} "
                    f"draft={converted.get('draft depth')!r}"
                )

        print(f"split {rel_split.as_posix() or '.'}: {len(label_paths)} labels")

    return totals


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(r"E:\data\PingLuRiver\dataset\ShipDraft"),
        help="Source ShipDraft dataset root",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(r"E:\data\PingLuRiver\dataset\ShipDraft_e2e"),
        help="Destination dataset root (must differ from --src)",
    )
    parser.add_argument(
        "--waterline-label",
        default=DEFAULT_WATERLINE_LABEL,
        help="Label name for extracted waterline linestrip",
    )
    parser.add_argument(
        "--image-mode",
        choices=["copy", "hardlink", "symlink"],
        default="copy",
        help="How to materialize images in the destination dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and convert in memory only; do not write files",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.src.is_dir():
        print(f"Source not found: {args.src}", file=sys.stderr)
        return 1

    print(f"src: {args.src}")
    print(f"dst: {args.dst}")
    print(f"waterline_label: {args.waterline_label}")
    print(f"image_mode: {args.image_mode} dry_run={args.dry_run}")

    totals = convert_dataset(
        args.src,
        args.dst,
        waterline_label=args.waterline_label,
        image_mode=args.image_mode,
        dry_run=args.dry_run,
    )

    print("\n=== conversion summary ===")
    for key in sorted(totals.keys()):
        print(f"  {key}: {totals[key]}")
    if not args.dry_run:
        print(f"\nDone. New dataset at: {args.dst.resolve()}")
    else:
        print("\nDry-run finished (no files written).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
