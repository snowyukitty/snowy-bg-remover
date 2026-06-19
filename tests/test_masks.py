from __future__ import annotations

import numpy as np

from snowy_bg_remover.masks import analyze_soft_alpha, fill_interior_holes


def test_hysteresis_keeps_soft_bridge_and_drops_detached_blob() -> None:
    alpha = np.zeros((12, 12), dtype=np.float32)
    alpha[2:5, 2:5] = 1.0
    alpha[3, 5:8] = 0.10
    alpha[2:5, 8:10] = 0.60
    alpha[9:11, 9:11] = 1.0

    result = analyze_soft_alpha(alpha, high_threshold=0.85, low_threshold=0.05)

    assert result.keep_mask[3, 6]
    assert result.alpha[3, 6] == np.float32(0.10)
    assert result.keep_mask[3, 8]
    assert result.alpha[3, 8] == np.float32(0.60)
    assert not result.keep_mask[9, 9]
    assert result.alpha[9, 9] == np.float32(0.0)
    assert result.removed_blob_count == 1
    assert result.seed_component_count == 2
    assert result.seed_component_dominance == 9 / 4


def test_interior_hole_is_filled_but_border_connected_gap_is_not() -> None:
    mask = np.ones((7, 7), dtype=bool)
    mask[3, 3] = False
    mask[0:4, 5] = False

    filled, fill_count, fill_area = fill_interior_holes(mask, max_hole_area=4)

    assert filled[3, 3]
    assert not filled[1, 5]
    assert fill_count == 1
    assert fill_area == 1


def test_analyze_soft_alpha_repairs_small_interior_alpha_hole() -> None:
    alpha = np.ones((7, 7), dtype=np.float32)
    alpha[3, 3] = 0.0

    result = analyze_soft_alpha(alpha, high_threshold=0.85, low_threshold=0.05)

    assert result.hole_fill_count == 1
    assert result.alpha[3, 3] == np.float32(1.0)


def test_analyze_soft_alpha_suppresses_distant_secondary_seed_artifact() -> None:
    alpha = np.zeros((80, 80), dtype=np.float32)
    alpha[20:55, 20:55] = 1.0
    alpha[62:66, 62:66] = 1.0
    alpha[55:62, 55:62] = 0.08

    result = analyze_soft_alpha(alpha, high_threshold=0.85, low_threshold=0.05)

    assert result.alpha[30, 30] == np.float32(1.0)
    assert result.alpha[63, 63] == np.float32(0.0)
