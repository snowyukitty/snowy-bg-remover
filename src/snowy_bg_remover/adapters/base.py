from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskResult:
    alpha: np.ndarray
    model_name: str
    device: str = "cpu"
    elapsed_ms: int = 0
    raw_scores: dict[str, float] = field(default_factory=dict)


class ModelAdapter(Protocol):
    name: str

    def load(self) -> None:
        ...

    def predict_alpha(self, image: Image.Image) -> MaskResult:
        ...
