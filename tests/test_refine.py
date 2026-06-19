from __future__ import annotations

import numpy as np
from PIL import Image

from snowy_bg_remover.refine import refine_alpha_closed_form


def test_refine_alpha_closed_form_preserves_shape_and_range() -> None:
    image = Image.new("RGB", (16, 16), (240, 240, 240))
    for y in range(4, 12):
        for x in range(4, 12):
            image.putpixel((x, y), (20, 80, 220))
    alpha = np.zeros((16, 16), dtype=np.float32)
    alpha[4:12, 4:12] = 1.0
    alpha[3:13, 3] = 0.4

    result = refine_alpha_closed_form(image, alpha, max_size=16)

    assert result.alpha.shape == alpha.shape
    assert float(result.alpha.min()) >= 0.0
    assert float(result.alpha.max()) <= 1.0
    assert result.alpha[8, 8] >= 0.9
