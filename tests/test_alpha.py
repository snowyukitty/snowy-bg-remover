from __future__ import annotations

import numpy as np
from PIL import Image

from snowy_bg_remover.alpha import resize_rgba_linear_premultiplied


def test_resize_rgba_linear_premultiplied_preserves_rgba_contract() -> None:
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    for y in range(2, 6):
        for x in range(2, 6):
            image.putpixel((x, y), (255, 0, 0, 255))

    resized = resize_rgba_linear_premultiplied(image, (4, 4))

    assert resized.mode == "RGBA"
    assert resized.size == (4, 4)
    assert np.asarray(resized.getchannel("A")).max() > 0
