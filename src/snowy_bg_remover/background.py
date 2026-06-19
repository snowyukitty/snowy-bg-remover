"""Neutral-backdrop suppression by region growing.

The learned matte (especially fast ``isnet-anime``) keeps flat backdrop pixels in
concave regions — the gaps between hair strands, around ears — when a pale subject
sits on a low-contrast matte backdrop. Those pixels are often fully opaque, so
alpha confidence cannot separate them from the subject.

A fixed-color chroma key fails here because an AI-generated backdrop is NOT a
single value: it is shaded darker inside the gaps between hair layers, so the
trapped gray spans a wide luminance range. Instead we model the backdrop as
"neutral (low chroma), smooth (low local texture), within an adaptive luminance
window around the border background" and **region-grow** it from the image border.
Growing through local similarity follows the gray as it darkens into the gaps and
stops at bright/tinted hair, so it is color-value agnostic.

Safety comes from connectivity + the neutral/smooth/luminance gates:
- White hair is too bright (above the window) and is excluded.
- Tinted features (skin, lavender eyes, pink hair shading) are chromatic.
- Dark features (sunglasses, pupils, outlines) are far below the background
  luminance and are excluded; interior ones are also unreachable from the border.
A conservative enclosed-pocket pass removes fully-walled gaps only when they are
smooth, neutral, and at (not merely near) the background luminance.

Active only when the border ring has a dominant neutral cluster; otherwise a
no-op, leaving scene/gradient backgrounds entirely to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# A pixel counts as backdrop when it is neutral, smooth, and within a luminance
# window around the detected border background.
NEUTRAL_CHROMA_MARGIN = 22.0  # max chroma above the background's own chroma
SMOOTH_STD_MAX = 6.0  # max local luminance std (background is smooth, hair is not)
LUM_WINDOW_DOWN = 105.0  # how far below border luminance gaps may be shaded
LUM_WINDOW_UP = 80.0  # how far above (kept below bright hair)
# Enclosed (border-unreachable) pockets are only removed when their median
# luminance is THIS close to the background luminance, protecting darker features
# such as sunglasses lenses that are also smooth and neutral.
ENCLOSED_LUM_BAND = 35.0
ENCLOSED_MIN_PX = 25


@dataclass(frozen=True)
class BackgroundEstimate:
    color: np.ndarray  # (3,) float32, 0-255
    luminance: float  # median luminance of the border neutral cluster
    chroma: float  # median chroma of the border neutral cluster
    cluster_fraction: float  # share of border pixels that are neutral backdrop
    uniform: bool


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb.mean(axis=2)


def _chroma(rgb: np.ndarray) -> np.ndarray:
    return rgb.max(axis=2) - rgb.min(axis=2)


def _local_std(lum: np.ndarray, radius: int = 2) -> np.ndarray:
    size = 2 * radius + 1
    mean = ndimage.uniform_filter(lum, size)
    mean_sq = ndimage.uniform_filter(lum * lum, size)
    return np.sqrt(np.clip(mean_sq - mean * mean, 0.0, None))


def estimate_background(
    rgb: np.ndarray,
    *,
    margin_ratio: float = 0.04,
    neutral_chroma: float = 18.0,
    min_cluster_fraction: float = 0.30,
) -> BackgroundEstimate:
    """Estimate a dominant neutral background from the border ring."""
    height, width = rgb.shape[:2]
    margin = max(1, int(round(min(height, width) * margin_ratio)))
    lum = _luminance(rgb)
    chroma = _chroma(rgb)

    def ring(arr: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [
                arr[:margin].reshape(-1),
                arr[-margin:].reshape(-1),
                arr[:, :margin].reshape(-1),
                arr[:, -margin:].reshape(-1),
            ]
        )

    ring_chroma = ring(chroma)
    ring_lum = ring(lum)
    ring_rgb = np.concatenate(
        [
            rgb[:margin].reshape(-1, 3),
            rgb[-margin:].reshape(-1, 3),
            rgb[:, :margin].reshape(-1, 3),
            rgb[:, -margin:].reshape(-1, 3),
        ]
    ).astype(np.float32)

    neutral = ring_chroma < neutral_chroma
    cluster_fraction = float(neutral.mean())
    if neutral.any():
        bg_lum = float(np.median(ring_lum[neutral]))
        bg_chroma = float(np.median(ring_chroma[neutral]))
        color = np.median(ring_rgb[neutral], axis=0).astype(np.float32)
    else:
        bg_lum, bg_chroma = float(np.median(ring_lum)), float(np.median(ring_chroma))
        color = np.median(ring_rgb, axis=0).astype(np.float32)

    return BackgroundEstimate(
        color=color,
        luminance=bg_lum,
        chroma=bg_chroma,
        cluster_fraction=cluster_fraction,
        uniform=cluster_fraction >= min_cluster_fraction,
    )


def suppress_background_color(
    alpha: np.ndarray,
    rgb: np.ndarray,
    background: BackgroundEstimate,
) -> tuple[np.ndarray, dict[str, float]]:
    """Region-grow the neutral backdrop from the border and zero its alpha.

    ``alpha`` is expected normalized to float32 [0, 1] and the same height/width as
    ``rgb``. Returns the new alpha and a small metrics dict.
    """
    if not background.uniform:
        return alpha, {"backgroundSuppressed": 0.0}

    rgb = rgb.astype(np.float32)
    lum = _luminance(rgb)
    chroma = _chroma(rgb)
    std = _local_std(lum)

    # "Backdrop-like": neutral, smooth, and within an adaptive luminance window
    # around the border background (wide downward to follow shaded gaps, capped
    # below bright hair). Color-value agnostic by design.
    backdrop = (
        (chroma < background.chroma + NEUTRAL_CHROMA_MARGIN)
        & (lum > background.luminance - LUM_WINDOW_DOWN)
        & (lum < background.luminance + LUM_WINDOW_UP)
        & (std < SMOOTH_STD_MAX)
    )

    height, width = alpha.shape
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    # (1) Border-connected backdrop: grown from the frame edge through backdrop-like
    # pixels. Interior features cannot be reached because the surrounding bright /
    # tinted subject blocks propagation.
    reachable = ndimage.binary_propagation(backdrop & border, mask=backdrop)

    # (2) Conservative enclosed pockets: backdrop-like regions fully walled off by
    # the subject, removed only when sizeable and AT the background luminance, so
    # darker smooth features (sunglasses lenses, etc.) are preserved.
    enclosed = backdrop & ~reachable
    enclosed_kill = np.zeros_like(reachable)
    labels, count = ndimage.label(enclosed)
    if count:
        for index in range(1, count + 1):
            component = labels == index
            if component.sum() < ENCLOSED_MIN_PX:
                continue
            if abs(float(np.median(lum[component])) - background.luminance) < ENCLOSED_LUM_BAND:
                enclosed_kill |= component

    kill = reachable | enclosed_kill
    # Grab the 1px anti-aliased rim just inside the backdrop boundary.
    kill = ndimage.binary_dilation(kill, iterations=1) & ndimage.binary_dilation(
        backdrop, iterations=2
    )

    if not kill.any():
        return alpha, {"backgroundSuppressed": 0.0}

    new_alpha = (alpha * (~kill)).astype(np.float32)

    # Safety: if this would erase almost the whole declared subject (e.g. subject
    # the same flat neutral as the backdrop), the detection is unreliable — bail.
    original_fg = alpha >= 0.5
    original_fg_count = int(original_fg.sum())
    if original_fg_count > 0:
        erased = int((original_fg & (new_alpha < 0.5)).sum())
        if erased / original_fg_count > 0.9:
            return alpha, {"backgroundSuppressed": 0.0, "backgroundAborted": 1.0}

    metrics = {
        "backgroundSuppressed": 1.0,
        "backgroundLuminance": float(background.luminance),
        "backgroundChroma": float(background.chroma),
        "backgroundClusterFraction": float(background.cluster_fraction),
        "backgroundReachablePx": float(int(reachable.sum())),
        "backgroundEnclosedPx": float(int(enclosed_kill.sum())),
        "backgroundKilledPx": float(int(kill.sum())),
    }
    return new_alpha, metrics
