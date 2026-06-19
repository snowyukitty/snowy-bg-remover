import numpy as np

from snowy_bg_remover.background import estimate_background, suppress_background_color


def _gray_scene():
    # Flat neutral-gray background with a non-gray subject block. Inside the subject:
    # a gray channel open to the border (connected backdrop, removed), a fully
    # enclosed neutral-gray pocket at background luminance (trapped backdrop,
    # removed), and an enclosed tinted spot (lavender eye/pearl analog, preserved).
    rgb = np.full((80, 80, 3), 128, dtype=np.float32)
    rgb[16:64, 16:64] = (230, 90, 90)  # subject (clearly non-gray)
    rgb[16:40, 36:44] = 128  # gray channel open to the top border (connected)
    rgb[44:60, 24:40] = 128  # enclosed neutral-gray pocket at bg luminance
    rgb[44:60, 46:62] = (150, 110, 185)  # enclosed TINTED feature (eye/pearl analog)
    alpha = np.zeros((80, 80), dtype=np.float32)
    alpha[16:64, 16:64] = 1.0  # model kept the whole block opaque, pockets included
    return rgb, alpha


def test_estimate_background_detects_uniform_gray():
    rgb, _ = _gray_scene()
    bg = estimate_background(rgb)
    assert bg.uniform
    assert np.allclose(bg.color, 128, atol=2)


def test_suppress_removes_backdrop_keeps_tinted_feature():
    rgb, alpha = _gray_scene()
    bg = estimate_background(rgb)
    out, metrics = suppress_background_color(alpha, rgb, bg)
    assert metrics["backgroundSuppressed"] == 1.0
    # Outer background removed.
    assert out[0, 0] == 0.0
    # The border-connected gray channel is removed.
    assert out[20, 40] < 0.5
    # The enclosed neutral-gray pocket (trapped backdrop) is removed.
    assert out[52, 32] < 0.5
    # The enclosed TINTED feature (eye/pearl analog) is preserved.
    assert out[52, 54] > 0.9
    # Real (non-gray) subject pixels are untouched.
    assert out[52, 20] == 1.0


def test_suppress_aborts_when_subject_matches_background():
    rgb = np.full((16, 16, 3), 255, dtype=np.float32)
    alpha = np.ones((16, 16), dtype=np.float32)
    bg = estimate_background(rgb)
    out, metrics = suppress_background_color(alpha, rgb, bg)
    assert metrics["backgroundSuppressed"] == 0.0
    assert np.array_equal(out, alpha)


def test_contract_alpha_shrinks_outer_ring_only():
    import numpy as np
    from snowy_bg_remover.masks import contract_alpha

    alpha = np.zeros((40, 40), dtype=np.float32)
    alpha[10:30, 10:30] = 1.0
    out = contract_alpha(alpha, 3)
    # Outer ring (halo analog) is reduced.
    assert out[10, 20] < 0.5
    # Deep interior is essentially untouched.
    assert out[20, 20] > 0.99
    # No-op when pixels <= 0.
    assert np.array_equal(contract_alpha(alpha, 0), alpha)
