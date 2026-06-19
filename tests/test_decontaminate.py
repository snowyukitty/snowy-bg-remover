from __future__ import annotations

import numpy as np
from PIL import Image

from snowy_bg_remover.decontaminate import bleed_edge_rgb_from_opaque


def test_bleed_edge_rgb_from_opaque_preserves_alpha_and_opaque_pixels() -> None:
    image = Image.new("RGBA", (5, 5), (0, 0, 0, 0))
    image.putpixel((2, 2), (240, 10, 20, 255))
    image.putpixel((2, 1), (10, 10, 240, 128))
    alpha = np.zeros((5, 5), dtype=np.float32)
    alpha[2, 2] = 1.0
    alpha[1, 2] = 0.5

    result = bleed_edge_rgb_from_opaque(image, alpha, max_radius=2)

    assert result.getpixel((2, 2)) == (240, 10, 20, 255)
    assert result.getpixel((2, 1))[3] == 128
    assert result.getpixel((2, 1))[:3] == (240, 10, 20)
    assert result.getpixel((0, 0))[:3] == (0, 0, 0)
