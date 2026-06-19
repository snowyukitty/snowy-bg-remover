from __future__ import annotations

from PIL import Image

from snowy_bg_remover.framing import frame_image


def test_padding_can_extend_beyond_source_bounds() -> None:
    image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    for y in range(0, 5):
        for x in range(0, 5):
            image.putpixel((x, y), (255, 0, 0, 255))

    framed = frame_image(image, (0, 0, 5, 5), trim=True, pad="2px", square=False)

    assert framed.size == (9, 9)
    assert framed.getpixel((0, 0))[3] == 0
    assert framed.getpixel((2, 2)) == (255, 0, 0, 255)
