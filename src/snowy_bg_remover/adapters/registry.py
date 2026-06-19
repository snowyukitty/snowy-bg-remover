from __future__ import annotations

from pathlib import Path

from ..model_specs import get_model_spec, resolve_model_id
from .base import ModelAdapter
from .onnx import OnnxSegmentationAdapter


def create_adapter(
    model_id: str,
    *,
    cache_dir: Path | None = None,
    allow_download: bool = False,
    offline: bool = True,
    device: str = "cpu",
    threads: int | None = None,
) -> ModelAdapter | None:
    resolved = resolve_model_id(model_id)
    if resolved in {"none", "input-alpha"}:
        return None
    spec = get_model_spec(resolved)
    return OnnxSegmentationAdapter(
        spec.model_id,
        cache_dir=cache_dir,
        allow_download=allow_download,
        offline=offline,
        device=device,
        threads=threads,
    )
