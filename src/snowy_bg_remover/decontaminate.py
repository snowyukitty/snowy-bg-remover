from __future__ import annotations

import numpy as np
from PIL import Image

from .alpha import linear_to_srgb, srgb_to_linear
from .masks import normalize_alpha


def _accumulate_shift(
    acc: np.ndarray,
    count: np.ndarray,
    rgb: np.ndarray,
    filled: np.ndarray,
    remaining: np.ndarray,
    dy: int,
    dx: int,
) -> None:
    h, w = filled.shape
    dst_y0 = max(0, dy)
    dst_y1 = h + min(0, dy)
    dst_x0 = max(0, dx)
    dst_x1 = w + min(0, dx)
    src_y0 = max(0, -dy)
    src_y1 = h - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = w - max(0, dx)

    src_filled = filled[src_y0:src_y1, src_x0:src_x1]
    dst_remaining = remaining[dst_y0:dst_y1, dst_x0:dst_x1]
    update = src_filled & dst_remaining
    if not np.any(update):
        return
    acc_view = acc[dst_y0:dst_y1, dst_x0:dst_x1]
    count_view = count[dst_y0:dst_y1, dst_x0:dst_x1]
    rgb_src = rgb[src_y0:src_y1, src_x0:src_x1]
    acc_view[update] += rgb_src[update]
    count_view[update] += 1


def bleed_edge_rgb_from_opaque(
    image: Image.Image,
    alpha: np.ndarray,
    *,
    opaque_threshold: float = 0.98,
    max_radius: int = 24,
) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    alpha_f = normalize_alpha(alpha)
    if alpha_f.shape != arr.shape[:2]:
        raise ValueError("alpha shape does not match image size")

    rgb = arr[:, :, :3].astype(np.float32)
    known = alpha_f >= opaque_threshold
    target = (alpha_f > 0.0) & (alpha_f < opaque_threshold)
    if not np.any(target) or not np.any(known):
        arr[alpha_f <= 0.0, :3] = 0
        return Image.fromarray(arr, mode="RGBA")

    filled = known.copy()
    remaining = target.copy()
    out_rgb = rgb.copy()

    for _ in range(max(0, max_radius)):
        if not np.any(remaining):
            break
        acc = np.zeros_like(out_rgb, dtype=np.float32)
        count = np.zeros(alpha_f.shape, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                _accumulate_shift(acc, count, out_rgb, filled, remaining, dy, dx)
        update = remaining & (count > 0)
        if not np.any(update):
            break
        out_rgb[update] = acc[update] / count[update, None]
        filled[update] = True
        remaining[update] = False

    arr[target, :3] = np.clip(out_rgb[target] + 0.5, 0, 255).astype(np.uint8)
    arr[alpha_f <= 0.0, :3] = 0
    return Image.fromarray(arr, mode="RGBA")


def estimate_foreground_rgb(
    image: Image.Image,
    alpha: np.ndarray,
    *,
    opaque_threshold: float = 0.98,
    fallback_radius: int = 24,
) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).copy()
    alpha_f = normalize_alpha(alpha)
    if alpha_f.shape != arr.shape[:2]:
        raise ValueError("alpha shape does not match image size")

    target = (alpha_f > 0.0) & (alpha_f < opaque_threshold)
    if not np.any(target):
        arr[alpha_f <= 0.0, :3] = 0
        return Image.fromarray(arr, mode="RGBA")

    try:
        from pymatting import estimate_foreground_ml
    except Exception:
        return bleed_edge_rgb_from_opaque(
            rgba,
            alpha_f,
            opaque_threshold=opaque_threshold,
            max_radius=fallback_radius,
        )

    try:
        rgb_srgb = arr[:, :, :3].astype(np.float32) / 255.0
        rgb_linear = srgb_to_linear(rgb_srgb)
        foreground_linear = estimate_foreground_ml(rgb_linear, alpha_f)
        foreground_linear = np.nan_to_num(
            foreground_linear,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        foreground_srgb = linear_to_srgb(np.clip(foreground_linear, 0.0, 1.0))
        arr[target, :3] = np.clip(
            foreground_srgb[target] * 255.0 + 0.5,
            0,
            255,
        ).astype(np.uint8)
        arr[alpha_f <= 0.0, :3] = 0
        return Image.fromarray(arr, mode="RGBA")
    except Exception:
        return bleed_edge_rgb_from_opaque(
            rgba,
            alpha_f,
            opaque_threshold=opaque_threshold,
            max_radius=fallback_radius,
        )
