from __future__ import annotations

import numpy as np
from PIL import Image


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * (values ** (1.0 / 2.4)) - 0.055,
    )


def apply_alpha(image: Image.Image, alpha: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha_u8, mode="L"))
    return rgba


def resize_rgba_linear_premultiplied(
    image: Image.Image, size: tuple[int, int], resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    rgba = image.convert("RGBA")
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    rgb_linear = srgb_to_linear(arr[:, :, :3])
    alpha = arr[:, :, 3:4]
    premultiplied = rgb_linear * alpha

    resized_channels = []
    for index in range(3):
        channel = Image.fromarray(premultiplied[:, :, index].astype(np.float32), mode="F")
        resized_channels.append(np.asarray(channel.resize(size, resample=resample)))

    alpha_img = Image.fromarray(alpha[:, :, 0].astype(np.float32), mode="F")
    alpha_resized = np.asarray(alpha_img.resize(size, resample=resample))

    premultiplied_resized = np.stack(resized_channels, axis=2)
    alpha_safe = np.maximum(alpha_resized[:, :, None], 1e-6)
    rgb_linear_resized = np.where(
        alpha_resized[:, :, None] > 1e-6,
        premultiplied_resized / alpha_safe,
        0.0,
    )
    rgb_srgb = linear_to_srgb(rgb_linear_resized)

    out = np.dstack([rgb_srgb, np.clip(alpha_resized, 0.0, 1.0)])
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="RGBA")
