"""Uniform-background color suppression.

The learned matte (especially the fast ``isnet-anime`` model) tends to keep
flat, low-contrast background pixels in concave regions such as the gaps between
hair strands or around ears when the subject is pale-on-pale against a solid
matte backdrop. Those trapped pixels are often fully opaque, so alpha confidence
cannot separate them from the real subject.

This module removes them with a color + topology rule that is safe by
construction: it only ever *reduces* alpha for pixels whose color is close to the
detected background color AND which are connected to the image border through a
continuous run of background-colored pixels. The character's own neutral elements
(eyes, pearls, silver) are interior and walled off by non-background-colored
pixels, so they are never reachable from the border and stay untouched.

It activates only when a single dominant uniform background is detected in the
border ring; for scene/gradient backgrounds it is a no-op and segmentation is
left entirely to the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BackgroundEstimate:
    color: np.ndarray  # (3,) float32, 0-255
    tolerance: float  # RGB euclidean catch radius around ``color``
    cluster_fraction: float  # share of border pixels belonging to the bg cluster
    uniform: bool


def estimate_background(
    rgb: np.ndarray,
    *,
    margin_ratio: float = 0.04,
    cluster_gate: float = 60.0,
    min_cluster_fraction: float = 0.30,
    tol_floor: float = 60.0,
    tol_ceiling: float = 82.0,
) -> BackgroundEstimate:
    """Estimate a single dominant uniform background color from the border ring."""
    height, width = rgb.shape[:2]
    margin = max(1, int(round(min(height, width) * margin_ratio)))
    ring = np.concatenate(
        [
            rgb[:margin].reshape(-1, 3),
            rgb[-margin:].reshape(-1, 3),
            rgb[:, :margin].reshape(-1, 3),
            rgb[:, -margin:].reshape(-1, 3),
        ]
    ).astype(np.float32)

    median = np.median(ring, axis=0)
    dist = np.sqrt(((ring - median) ** 2).sum(axis=1))
    cluster = dist < cluster_gate
    cluster_fraction = float(cluster.mean())
    spread = float(np.percentile(dist[cluster], 85)) if cluster.any() else 0.0
    tolerance = float(np.clip(tol_floor + 1.2 * spread, tol_floor, tol_ceiling))
    uniform = cluster_fraction >= min_cluster_fraction
    return BackgroundEstimate(
        color=median.astype(np.float32),
        tolerance=tolerance,
        cluster_fraction=cluster_fraction,
        uniform=uniform,
    )


def suppress_background_color(
    alpha: np.ndarray,
    rgb: np.ndarray,
    background: BackgroundEstimate,
) -> tuple[np.ndarray, dict[str, float]]:
    """Scale alpha toward 0 for border-connected background-colored pixels.

    ``alpha`` is expected normalized to float32 [0, 1] and the same height/width
    as ``rgb``. Returns the new alpha and a small metrics dict.
    """
    if not background.uniform:
        return alpha, {"backgroundSuppressed": 0.0}

    diff = rgb.astype(np.float32) - background.color[None, None, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    bg_colored = dist < background.tolerance

    height, width = alpha.shape
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    # Background = bg-colored pixels reachable from the border through other
    # bg-colored pixels. Interior neutral features cannot be reached because the
    # surrounding non-bg-colored subject blocks propagation.
    reachable = ndimage.binary_propagation(bg_colored & border, mask=bg_colored)
    if not reachable.any():
        return alpha, {"backgroundSuppressed": 0.0}

    # Feathered kill: full at the background color, fading to 0 at the tolerance
    # edge, so the cut blends into the anti-aliased hair boundary. ``bg_like`` is
    # 0 wherever ``dist >= tolerance``, so non-background pixels are never touched.
    bg_like = np.clip(1.0 - dist / max(background.tolerance, 1e-3), 0.0, 1.0)
    kill = np.where(reachable, bg_like, 0.0).astype(np.float32)
    new_alpha = (alpha * (1.0 - kill)).astype(np.float32)

    # Safety: if the subject is the same flat color as the background, the border
    # flood reaches through the whole "subject" and would erase it. Treat that as
    # an unreliable detection and leave the alpha untouched.
    original_fg = alpha >= 0.5
    original_fg_count = int(original_fg.sum())
    if original_fg_count > 0:
        erased_fg = int((original_fg & (new_alpha < 0.5)).sum())
        if erased_fg / original_fg_count > 0.9:
            return alpha, {"backgroundSuppressed": 0.0, "backgroundAborted": 1.0}

    touched = kill > 0.01
    metrics = {
        "backgroundSuppressed": 1.0,
        "backgroundColor": [float(c) for c in background.color],
        "backgroundTolerance": float(background.tolerance),
        "backgroundClusterFraction": float(background.cluster_fraction),
        "backgroundReachablePx": float(int(reachable.sum())),
        "backgroundTouchedPx": float(int(touched.sum())),
    }
    return new_alpha, metrics
