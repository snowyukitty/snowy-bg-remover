from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

import numpy as np

from .adapters.base import ModelAdapter
from .adapters.registry import create_adapter
from .alpha import apply_alpha, resize_rgba_linear_premultiplied
from .atomic_write import atomic_save_png
from .background import estimate_background, suppress_background_color
from .contracts import CutoutOptions, CutoutResult
from .decontaminate import estimate_foreground_rgb
from .explain import save_explain_artifacts
from .framing import frame_image
from .image_io import LoadedImage, load_image
from .masks import (
    TopologyResult,
    analyze_soft_alpha,
    bbox_from_mask,
    contract_alpha,
    normalize_alpha,
)
from .refine import refine_alpha_closed_form


def _elapsed_ms(started_at: float) -> int:
    return max(0, int(round((time.perf_counter() - started_at) * 1000)))


def _bbox_json(bbox: tuple[int, int, int, int] | None) -> list[int] | None:
    if bbox is None:
        return None
    return [int(value) for value in bbox]


def _metrics(topology: TopologyResult) -> dict[str, Any]:
    return {
        "seedCoverage": topology.seed_coverage,
        "supportCoverage": topology.support_coverage,
        "edgeUncertaintyScore": topology.edge_uncertainty_score,
        "removedBlobCount": topology.removed_blob_count,
        "removedBlobArea": topology.removed_blob_area,
        "holeFillCount": topology.hole_fill_count,
        "holeFillArea": topology.hole_fill_area,
        "seedComponentCount": topology.seed_component_count,
        "mainSeedArea": topology.main_seed_area,
        "secondSeedArea": topology.second_seed_area,
        "seedComponentDominance": topology.seed_component_dominance,
    }


def _artifact_flags(topology: TopologyResult, had_alpha: bool) -> list[str]:
    flags: list[str] = []
    if had_alpha:
        flags.append("existing_alpha")
    if topology.removed_blob_count > 0:
        flags.append("removed_detached_blobs")
    if topology.hole_fill_count > 0:
        flags.append("filled_interior_holes")
    if topology.edge_uncertainty_score > 0.5:
        flags.append("high_edge_uncertainty")
    return flags


def _failure(
    *,
    options: CutoutOptions,
    started_at: float,
    reason: str,
    message: str,
    width: int | None = None,
    height: int | None = None,
    had_alpha: bool = False,
    model: str | None = None,
    device: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> CutoutResult:
    return CutoutResult(
        ok=False,
        reason=reason,
        input=str(options.input_path),
        output=str(options.output_path) if options.output_path else None,
        width=width,
        height=height,
        hadAlpha=had_alpha,
        elapsedMs=_elapsed_ms(started_at),
        message=message,
        model=model,
        device=device,
        metrics=metrics or {},
    )


def _alpha_from_adapter(
    image: Image.Image, adapter: ModelAdapter
) -> tuple[Any, str, str, dict[str, Any]]:
    adapter.load()
    prediction = adapter.predict_alpha(image.convert("RGB"))
    return (
        prediction.alpha,
        prediction.model_name,
        prediction.device,
        dict(prediction.raw_scores),
    )


def process_image(
    options: CutoutOptions, adapter: ModelAdapter | None = None
) -> CutoutResult:
    started_at = time.perf_counter()
    input_path = Path(options.input_path)

    if not input_path.exists():
        return _failure(
            options=options,
            started_at=started_at,
            reason="input_missing",
            message="input file does not exist",
        )

    try:
        loaded = load_image(input_path)
    except Exception as exc:
        return _failure(
            options=options,
            started_at=started_at,
            reason="decode_failure",
            message=f"failed to decode image: {exc}",
        )

    # Re-cut support: flatten an already-cut RGBA back onto a flat gray backdrop and
    # drop its alpha, so the model path (with background suppression) runs again
    # instead of trusting the old, possibly-flawed alpha. A no-op for opaque input.
    if options.flatten_bg is not None and loaded.alpha is not None:
        level = int(np.clip(options.flatten_bg, 0, 255))
        flat = Image.alpha_composite(
            Image.new("RGBA", loaded.image.size, (level, level, level, 255)),
            loaded.image.convert("RGBA"),
        ).convert("RGB")
        loaded = LoadedImage(image=flat, had_alpha=loaded.had_alpha, alpha=None)

    width, height = loaded.image.size
    model_name = "input-alpha"
    device = "none"
    raw_metrics: dict[str, Any] = {}

    if loaded.alpha is not None and not options.force_model:
        source_alpha = loaded.alpha
    elif adapter is not None:
        try:
            source_alpha, model_name, device, raw_metrics = _alpha_from_adapter(
                loaded.image, adapter
            )
        except Exception as exc:
            return _failure(
                options=options,
                started_at=started_at,
                reason="model_unavailable",
                message=f"model prediction failed: {exc}",
                width=width,
                height=height,
                had_alpha=loaded.had_alpha,
                model=options.model,
            )
    else:
        try:
            adapter = create_adapter(
                options.model,
                cache_dir=options.model_cache_dir,
                allow_download=options.allow_download,
                offline=not options.allow_download,
                device=options.device,
                threads=options.threads,
            )
            if adapter is None:
                raise RuntimeError("selected model does not provide segmentation")
            source_alpha, model_name, device, raw_metrics = _alpha_from_adapter(
                loaded.image, adapter
            )
        except Exception as exc:
            return _failure(
                options=options,
                started_at=started_at,
                reason="model_unavailable",
                message=str(exc),
                width=width,
                height=height,
                had_alpha=loaded.had_alpha,
                model=options.model,
            )

    # Suppress a flat uniform background the model left in concave hair/ear gaps.
    # Only on the model path (input alpha is trusted as-is) and only when a single
    # dominant uniform border background is detected; otherwise a no-op.
    used_model = loaded.alpha is None or options.force_model
    if options.background_suppression and used_model:
        rgb_arr = np.asarray(loaded.image.convert("RGB"))
        alpha01 = normalize_alpha(source_alpha)
        if rgb_arr.shape[:2] != alpha01.shape:
            rgb_arr = np.asarray(
                loaded.image.convert("RGB").resize((alpha01.shape[1], alpha01.shape[0]))
            )
        background = estimate_background(rgb_arr)
        source_alpha, bg_metrics = suppress_background_color(alpha01, rgb_arr, background)
        raw_metrics.update(bg_metrics)

    try:
        topology = analyze_soft_alpha(
            source_alpha,
            high_threshold=options.high_threshold,
            low_threshold=options.low_threshold,
            bbox_threshold=options.bbox_threshold,
            max_hole_area_ratio=options.max_hole_area_ratio,
        )
    except Exception as exc:
        return _failure(
            options=options,
            started_at=started_at,
            reason="degenerate_alpha",
            message=f"invalid alpha map: {exc}",
            width=width,
            height=height,
            had_alpha=loaded.had_alpha,
            model=model_name,
            device=device,
        )

    metrics = _metrics(topology)
    metrics.update(raw_metrics)
    flags = _artifact_flags(topology, loaded.alpha is not None)
    if raw_metrics.get("backgroundSuppressed"):
        flags.append("suppressed_background")

    if options.alpha_refine and topology.bbox is not None:
        refined = refine_alpha_closed_form(
            loaded.image,
            topology.alpha,
            max_size=options.alpha_refine_size,
        )
        topology.alpha = refined.alpha
        topology.bbox = bbox_from_mask(
            topology.alpha >= max(options.low_threshold, options.bbox_threshold)
        )
        topology.subject_coverage = float(
            (topology.alpha > options.low_threshold).sum() / max(topology.alpha.size, 1)
        )
        metrics.update(refined.metrics)
        if refined.metrics.get("alphaRefineApplied"):
            flags.append("alpha_refined")

    if topology.bbox is None or topology.seed_coverage <= 0:
        result = CutoutResult(
            ok=False,
            reason="no_confident_subject",
            input=str(input_path),
            output=str(options.output_path) if options.output_path else None,
            width=width,
            height=height,
            bbox=_bbox_json(topology.bbox),
            subjectCoverage=topology.subject_coverage,
            hadAlpha=loaded.had_alpha,
            elapsedMs=_elapsed_ms(started_at),
            message="no high-confidence subject core was found",
            model=model_name,
            device=device,
            artifactFlags=flags,
            metrics=metrics,
        )
        _maybe_write_explain(options, source_alpha, topology, result)
        return result

    if topology.subject_coverage < options.min_subject_coverage:
        reason = "no_confident_subject"
        message = "subject coverage is below the minimum threshold"
    elif topology.subject_coverage > options.max_subject_coverage:
        reason = "low_confidence"
        message = "subject coverage is above the maximum threshold"
    elif (
        topology.seed_component_dominance is not None
        and topology.seed_component_dominance < 1.5
    ):
        reason = "low_confidence"
        message = "multiple high-confidence subject cores were detected"
    elif topology.edge_uncertainty_score > 0.75:
        reason = "low_confidence"
        message = "alpha map is too uncertain around the kept subject"
    else:
        reason = None
        message = "ok"

    if reason is not None:
        result = CutoutResult(
            ok=False,
            reason=reason,
            input=str(input_path),
            output=str(options.output_path) if options.output_path else None,
            width=width,
            height=height,
            bbox=_bbox_json(topology.bbox),
            subjectCoverage=topology.subject_coverage,
            hadAlpha=loaded.had_alpha,
            elapsedMs=_elapsed_ms(started_at),
            message=message,
            model=model_name,
            device=device,
            artifactFlags=flags,
            metrics=metrics,
        )
        _maybe_write_explain(options, source_alpha, topology, result)
        return result

    if options.check:
        result = CutoutResult(
            ok=True,
            input=str(input_path),
            output=None,
            width=width,
            height=height,
            bbox=_bbox_json(topology.bbox),
            subjectCoverage=topology.subject_coverage,
            hadAlpha=loaded.had_alpha,
            elapsedMs=_elapsed_ms(started_at),
            message=message,
            model=model_name,
            device=device,
            artifactFlags=flags,
            metrics=metrics,
        )
        _maybe_write_explain(options, source_alpha, topology, result)
        return result

    if options.edge_contract > 0:
        topology.alpha = contract_alpha(topology.alpha, options.edge_contract)

    final_image = apply_alpha(loaded.image, topology.alpha)
    if options.decontaminate_edges:
        final_image = estimate_foreground_rgb(
            final_image,
            topology.alpha,
            fallback_radius=options.decontaminate_radius,
        )
    final_image = frame_image(
        final_image,
        topology.bbox,
        trim=options.trim,
        pad=options.pad,
        square=options.square,
    )
    if options.emit_size is not None:
        final_image = resize_rgba_linear_premultiplied(
            final_image, (options.emit_size, options.emit_size)
        )

    output = str(options.output_path) if options.output_path else None
    if options.output_path is None:
        return _failure(
            options=options,
            started_at=started_at,
            reason="usage_error",
            message="--output is required unless --check is used",
            width=width,
            height=height,
            had_alpha=loaded.had_alpha,
            model=model_name,
            device=device,
            metrics=metrics,
        )
    try:
        atomic_save_png(final_image, Path(options.output_path))
    except Exception as exc:
        return _failure(
            options=options,
            started_at=started_at,
            reason="write_failure",
            message=f"failed to write output atomically: {exc}",
            width=width,
            height=height,
            had_alpha=loaded.had_alpha,
            model=model_name,
            device=device,
            metrics=metrics,
        )

    result = CutoutResult(
        ok=True,
        input=str(input_path),
        output=output,
        width=width,
        height=height,
        bbox=_bbox_json(topology.bbox),
        subjectCoverage=topology.subject_coverage,
        hadAlpha=loaded.had_alpha,
        elapsedMs=_elapsed_ms(started_at),
        message=message,
        model=model_name,
        device=device,
        artifactFlags=flags,
        metrics=metrics,
    )
    _maybe_write_explain(options, source_alpha, topology, result)
    return result


def _maybe_write_explain(
    options: CutoutOptions,
    source_alpha,
    topology: TopologyResult,
    result: CutoutResult,
) -> None:
    if options.explain_dir is None:
        return
    try:
        save_explain_artifacts(
            Path(options.explain_dir),
            source_alpha=source_alpha,
            topology=topology,
            payload=result.to_json_dict(),
        )
    except Exception as exc:
        result.artifactFlags.append("explain_write_failed")
        result.metrics["explainError"] = str(exc)
