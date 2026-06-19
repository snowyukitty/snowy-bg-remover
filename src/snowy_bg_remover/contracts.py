from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_INPUT_MISSING = 10
EXIT_DECODE_FAILURE = 11
EXIT_MODEL_MISSING = 12
EXIT_MODEL_VERIFY_FAILURE = 13
EXIT_NO_CONFIDENT_SUBJECT = 20
EXIT_DEGENERATE_ALPHA = 21
EXIT_WRITE_FAILURE = 30
EXIT_INTERNAL_ERROR = 40


REASON_TO_EXIT_CODE = {
    None: EXIT_SUCCESS,
    "usage_error": EXIT_USAGE,
    "input_missing": EXIT_INPUT_MISSING,
    "decode_failure": EXIT_DECODE_FAILURE,
    "model_unavailable": EXIT_MODEL_MISSING,
    "model_verify_failure": EXIT_MODEL_VERIFY_FAILURE,
    "no_confident_subject": EXIT_NO_CONFIDENT_SUBJECT,
    "low_confidence": EXIT_NO_CONFIDENT_SUBJECT,
    "degenerate_alpha": EXIT_DEGENERATE_ALPHA,
    "write_failure": EXIT_WRITE_FAILURE,
    "internal_error": EXIT_INTERNAL_ERROR,
}


@dataclass(frozen=True)
class CutoutOptions:
    input_path: Path
    output_path: Path | None = None
    check: bool = False
    profile: str = "emote"
    model: str = "auto"
    high_threshold: float = 0.85
    low_threshold: float = 0.05
    bbox_threshold: float = 0.12
    min_subject_coverage: float = 0.01
    max_subject_coverage: float = 0.98
    max_hole_area_ratio: float = 0.02
    alpha_refine: bool = False
    alpha_refine_size: int = 640
    force_model: bool = False
    allow_download: bool = False
    device: str = "cpu"
    threads: int | None = None
    model_cache_dir: Path | None = None
    explain_dir: Path | None = None
    decontaminate_edges: bool = True
    decontaminate_radius: int = 24
    background_suppression: bool = True
    edge_contract: int = 0
    flatten_bg: int | None = None
    trim: bool = False
    pad: str | None = None
    square: bool = False
    emit_size: int | None = None


@dataclass
class CutoutResult:
    ok: bool
    input: str
    output: str | None
    width: int | None = None
    height: int | None = None
    bbox: list[int] | None = None
    subjectCoverage: float | None = None
    hadAlpha: bool = False
    elapsedMs: int = 0
    message: str = ""
    reason: str | None = None
    model: str | None = None
    device: str | None = None
    schemaVersion: int = 1
    artifactFlags: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return REASON_TO_EXIT_CODE.get(self.reason, EXIT_INTERNAL_ERROR)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schemaVersion,
            "ok": self.ok,
            "reason": self.reason,
            "input": self.input,
            "output": self.output,
            "width": self.width,
            "height": self.height,
            "bbox": self.bbox,
            "subjectCoverage": self.subjectCoverage,
            "hadAlpha": self.hadAlpha,
            "elapsedMs": self.elapsedMs,
            "message": self.message,
            "model": self.model,
            "device": self.device,
            "artifactFlags": self.artifactFlags,
            "metrics": self.metrics,
        }
