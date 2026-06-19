from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class LoadedImage:
    image: Image.Image
    had_alpha: bool
    alpha: np.ndarray | None


def load_image(path: Path) -> LoadedImage:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        had_alpha = "A" in img.getbands()
        rgba = img.convert("RGBA")

    alpha = np.asarray(rgba.getchannel("A")).astype(np.float32) / 255.0
    if not has_meaningful_alpha(alpha):
        return LoadedImage(image=rgba, had_alpha=had_alpha, alpha=None)
    return LoadedImage(image=rgba, had_alpha=had_alpha, alpha=alpha)


def has_meaningful_alpha(alpha: np.ndarray, epsilon: float = 1.0 / 255.0) -> bool:
    if alpha.ndim != 2 or alpha.size == 0:
        return False
    min_alpha = float(alpha.min())
    max_alpha = float(alpha.max())
    if max_alpha <= epsilon:
        return False
    if min_alpha >= 1.0 - epsilon:
        return False
    return (max_alpha - min_alpha) > epsilon
