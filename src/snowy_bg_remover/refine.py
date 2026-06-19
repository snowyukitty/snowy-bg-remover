from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .masks import normalize_alpha

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - scipy is a required dependency.
    ndimage = None


@dataclass(frozen=True)
class AlphaRefineResult:
    alpha: np.ndarray
    metrics: dict[str, float | int | str | bool]


def _resize_float(values: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(values.astype(np.float32), mode="F").resize(size, resample)
    ).astype(np.float32)


def refine_alpha_closed_form(
    image: Image.Image,
    alpha: np.ndarray,
    *,
    max_size: int = 640,
    foreground_threshold: float = 0.98,
    background_threshold: float = 0.02,
    unknown_radius: int = 10,
    core_erosion: int = 4,
) -> AlphaRefineResult:
    started = time.perf_counter()
    alpha_f = normalize_alpha(alpha)
    metrics: dict[str, float | int | str | bool] = {
        "alphaRefineApplied": False,
        "alphaRefineMethod": "closed_form",
    }

    if ndimage is None:
        metrics["alphaRefineSkipped"] = "scipy_unavailable"
        return AlphaRefineResult(alpha_f, metrics)
    if max_size <= 0:
        metrics["alphaRefineSkipped"] = "disabled_size"
        return AlphaRefineResult(alpha_f, metrics)

    support = alpha_f > background_threshold
    core = alpha_f >= foreground_threshold
    if not np.any(support) or not np.any(core):
        metrics["alphaRefineSkipped"] = "missing_support_or_core"
        return AlphaRefineResult(alpha_f, metrics)

    scale = min(1.0, float(max_size) / max(image.size))
    working_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    radius = max(1, int(round(unknown_radius * scale)))
    erosion = max(1, int(round(core_erosion * scale)))

    image_small = image.convert("RGB").resize(working_size, Image.Resampling.LANCZOS)
    alpha_small = _resize_float(alpha_f, working_size, Image.Resampling.BILINEAR)
    support_small = alpha_small > background_threshold
    core_small = alpha_small >= foreground_threshold

    unknown_small = (
        ndimage.binary_dilation(support_small, iterations=radius)
        & ~ndimage.binary_erosion(core_small, iterations=erosion)
    ) | ((alpha_small > background_threshold) & (alpha_small < foreground_threshold))
    foreground_small = core_small & ~unknown_small
    background_small = (
        ~ndimage.binary_dilation(support_small, iterations=radius)
    ) & ~unknown_small

    fg_count = int(np.count_nonzero(foreground_small))
    bg_count = int(np.count_nonzero(background_small))
    unknown_count = int(np.count_nonzero(unknown_small))
    metrics.update(
        {
            "alphaRefineWorkingWidth": working_size[0],
            "alphaRefineWorkingHeight": working_size[1],
            "alphaRefineKnownForeground": fg_count,
            "alphaRefineKnownBackground": bg_count,
            "alphaRefineUnknown": unknown_count,
        }
    )
    if fg_count == 0 or bg_count == 0 or unknown_count == 0:
        metrics["alphaRefineSkipped"] = "degenerate_trimap"
        return AlphaRefineResult(alpha_f, metrics)

    trimap = np.full(alpha_small.shape, 0.5, dtype=np.float64)
    trimap[background_small] = 0.0
    trimap[foreground_small] = 1.0

    try:
        from pymatting import estimate_alpha_cf

        image_np = np.asarray(image_small).astype(np.float64) / 255.0
        refined_small = estimate_alpha_cf(
            image_np,
            trimap,
            cg_kwargs={"maxiter": 300},
        )
    except Exception as exc:
        metrics["alphaRefineSkipped"] = "solver_error"
        metrics["alphaRefineError"] = str(exc)
        return AlphaRefineResult(alpha_f, metrics)

    refined = _resize_float(
        np.clip(refined_small, 0.0, 1.0).astype(np.float32),
        image.size,
        Image.Resampling.BILINEAR,
    )
    unknown_full = (
        ndimage.binary_dilation(support, iterations=unknown_radius)
        & ~ndimage.binary_erosion(core, iterations=core_erosion)
    ) | ((alpha_f > background_threshold) & (alpha_f < foreground_threshold))
    outside = ~ndimage.binary_dilation(support, iterations=unknown_radius + 2)

    refined_alpha = alpha_f.copy()
    refined_alpha[unknown_full] = refined[unknown_full]
    refined_alpha[outside] = 0.0
    refined_alpha[core & ~unknown_full] = alpha_f[core & ~unknown_full]
    refined_alpha = np.clip(refined_alpha, 0.0, 1.0).astype(np.float32)

    metrics["alphaRefineApplied"] = True
    metrics["alphaRefineMs"] = int(round((time.perf_counter() - started) * 1000))
    metrics["alphaRefineUnknownCoverage"] = float(
        np.count_nonzero(unknown_full) / max(alpha_f.size, 1)
    )
    return AlphaRefineResult(refined_alpha, metrics)
