from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def compute_resize_padding_params(height: int, width: int, target_size_wh: Tuple[int, int]) -> Dict[str, float]:
    target_w, target_h = target_size_wh
    scale = min(target_w / width, target_h / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    pad_left = max(0, (target_w - new_w) // 2)
    pad_top = max(0, (target_h - new_h) // 2)
    pad_right = max(0, target_w - new_w - pad_left)
    pad_bottom = max(0, target_h - new_h - pad_top)
    return {
        "scale": float(scale),
        "new_w": int(new_w),
        "new_h": int(new_h),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
        "pad_right": int(pad_right),
        "pad_bottom": int(pad_bottom),
        "target_w": int(target_w),
        "target_h": int(target_h),
    }


def resize_with_padding(image: np.ndarray, target_size_wh: Tuple[int, int], pad_value: int = 0):
    params = compute_resize_padding_params(image.shape[0], image.shape[1], target_size_wh)
    resized = cv2.resize(image, (params["new_w"], params["new_h"]), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(
        resized,
        params["pad_top"],
        params["pad_bottom"],
        params["pad_left"],
        params["pad_right"],
        borderType=cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )
    return padded, params
