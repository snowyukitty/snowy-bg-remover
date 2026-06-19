from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .masks import TopologyResult, normalize_alpha


def _save_luma(path: Path, values: np.ndarray) -> None:
    arr = np.clip(normalize_alpha(values) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def save_explain_artifacts(
    explain_dir: Path,
    *,
    source_alpha: np.ndarray,
    topology: TopologyResult,
    payload: dict[str, Any],
) -> None:
    explain_dir.mkdir(parents=True, exist_ok=True)
    _save_luma(explain_dir / "source-alpha.png", source_alpha)
    _save_luma(explain_dir / "final-alpha.png", topology.alpha)
    _save_luma(explain_dir / "keep-mask.png", topology.keep_mask.astype(np.float32))
    _save_luma(
        explain_dir / "protected-core.png",
        topology.protected_core.astype(np.float32),
    )
    with (explain_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
