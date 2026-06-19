import numpy as np

from snowy_bg_remover.background import estimate_background, suppress_background_color


def _gray_scene():
    # Flat gray background with a non-gray subject block in the center, a gray
    # gap inside the subject that is OPEN to the border (a hair-gap analog), and
    # an enclosed gray pocket fully walled off by the subject (eye/pearl analog).
    rgb = np.full((40, 40, 3), 128, dtype=np.float32)
    rgb[8:32, 8:32] = (230, 90, 90)  # subject (clearly non-gray)
    rgb[8:20, 18:22] = 128  # gray channel open to the top border through the subject edge
    rgb[24:28, 24:28] = 128  # enclosed gray pocket inside the subject
    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[8:32, 8:32] = 1.0  # model kept the whole block opaque, gaps included
    return rgb, alpha


def test_estimate_background_detects_uniform_gray():
    rgb, _ = _gray_scene()
    bg = estimate_background(rgb)
    assert bg.uniform
    assert np.allclose(bg.color, 128, atol=2)


def test_suppress_removes_border_connected_gap_keeps_enclosed_pocket():
    rgb, alpha = _gray_scene()
    bg = estimate_background(rgb)
    out, metrics = suppress_background_color(alpha, rgb, bg)
    assert metrics["backgroundSuppressed"] == 1.0
    # Outer background removed.
    assert out[0, 0] == 0.0
    # The border-connected gray gap is removed.
    assert out[9, 20] < 0.5
    # The enclosed gray pocket (walled by subject) is preserved.
    assert out[26, 26] > 0.9
    # Real (non-gray) subject pixels are untouched.
    assert out[16, 12] == 1.0


def test_suppress_aborts_when_subject_matches_background():
    rgb = np.full((16, 16, 3), 255, dtype=np.float32)
    alpha = np.ones((16, 16), dtype=np.float32)
    bg = estimate_background(rgb)
    out, metrics = suppress_background_color(alpha, rgb, bg)
    assert metrics["backgroundSuppressed"] == 0.0
    assert np.array_equal(out, alpha)
