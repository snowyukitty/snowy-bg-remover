from __future__ import annotations

from PIL import Image


def parse_pad(pad: str | None, basis: int) -> int:
    if not pad:
        return 0
    value = pad.strip()
    if value.endswith("%"):
        return max(0, round(basis * float(value[:-1]) / 100.0))
    if value.endswith("px"):
        value = value[:-2]
    return max(0, int(value))


def frame_image(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    trim: bool = False,
    pad: str | None = None,
    square: bool = False,
) -> Image.Image:
    if bbox is None or not (trim or square or pad):
        return image

    rgba = image.convert("RGBA")
    x, y, w, h = bbox
    basis = max(w, h)
    pad_px = parse_pad(pad, basis)
    left = x - pad_px
    top = y - pad_px
    right = x + w + pad_px
    bottom = y + h + pad_px

    if trim or pad:
        canvas = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(rgba.width, right)
        src_bottom = min(rgba.height, bottom)
        if src_right > src_left and src_bottom > src_top:
            crop = rgba.crop((src_left, src_top, src_right, src_bottom))
            canvas.alpha_composite(crop, (src_left - left, src_top - top))
        cropped = canvas
    else:
        cropped = rgba

    if not square:
        return cropped

    side = max(cropped.width, cropped.height)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    canvas.alpha_composite(cropped, offset)
    return canvas
