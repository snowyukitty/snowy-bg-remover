from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

try:
    from scipy import ndimage
except Exception:  # pragma: no cover - exercised only in minimal installs.
    ndimage = None


@dataclass
class TopologyResult:
    alpha: np.ndarray
    keep_mask: np.ndarray
    protected_core: np.ndarray
    bbox: tuple[int, int, int, int] | None
    subject_coverage: float
    seed_coverage: float
    support_coverage: float
    edge_uncertainty_score: float
    removed_blob_count: int
    removed_blob_area: int
    hole_fill_count: int
    hole_fill_area: int
    seed_component_count: int
    main_seed_area: int
    second_seed_area: int
    seed_component_dominance: float | None


NEIGHBORS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

STRUCTURE_8 = np.ones((3, 3), dtype=bool)


def normalize_alpha(alpha: np.ndarray) -> np.ndarray:
    arr = np.asarray(alpha)
    if arr.ndim != 2:
        raise ValueError("alpha must be a 2D array")
    arr = arr.astype(np.float32, copy=False)
    if arr.size == 0:
        return arr
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def reconstruct_from_seed(seed: np.ndarray, support: np.ndarray) -> np.ndarray:
    if seed.shape != support.shape:
        raise ValueError("seed and support shapes differ")
    if ndimage is not None:
        return ndimage.binary_propagation(
            seed & support,
            structure=STRUCTURE_8,
            mask=support,
        ).astype(bool)

    keep = np.zeros(seed.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    h, w = seed.shape

    for y, x in np.argwhere(seed & support):
        yi = int(y)
        xi = int(x)
        keep[yi, xi] = True
        queue.append((yi, xi))

    while queue:
        y, x = queue.popleft()
        for dy, dx in NEIGHBORS_8:
            ny = y + dy
            nx = x + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= w:
                continue
            if keep[ny, nx] or not support[ny, nx]:
                continue
            keep[ny, nx] = True
            queue.append((ny, nx))

    return keep


def _count_components(mask: np.ndarray) -> tuple[int, int]:
    if ndimage is not None:
        labels, count = ndimage.label(mask, structure=STRUCTURE_8)
        if count == 0:
            return 0, 0
        return int(count), int(np.count_nonzero(labels))

    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    area = 0

    for start_y, start_x in np.argwhere(mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        count += 1
        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            area += 1
            for dy, dx in NEIGHBORS_8:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

    return count, area


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    if ndimage is not None:
        labels, count = ndimage.label(mask, structure=STRUCTURE_8)
        if count == 0:
            return np.zeros(mask.shape, dtype=bool), 0, 0, 0
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        largest_label = int(np.argmax(areas))
        largest_area = int(areas[largest_label])
        nonzero_areas = np.sort(areas[areas > 0])
        second_area = int(nonzero_areas[-2]) if nonzero_areas.size > 1 else 0
        return labels == largest_label, int(count), largest_area, second_area

    visited = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    largest_area = 0
    second_area = 0
    largest_pixels: list[tuple[int, int]] = []

    for start_y, start_x in np.argwhere(mask):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        count += 1
        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            for dy, dx in NEIGHBORS_8:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

        area = len(pixels)
        if area > largest_area:
            second_area = largest_area
            largest_area = area
            largest_pixels = pixels
        elif area > second_area:
            second_area = area

    largest = np.zeros(mask.shape, dtype=bool)
    for py, px in largest_pixels:
        largest[py, px] = True
    return largest, count, largest_area, second_area


def fill_interior_holes(
    mask: np.ndarray, max_hole_area: int
) -> tuple[np.ndarray, int, int]:
    if max_hole_area <= 0:
        return mask.copy(), 0, 0

    if ndimage is not None:
        background = ~mask
        labels, count = ndimage.label(background, structure=STRUCTURE_8)
        if count == 0:
            return mask.copy(), 0, 0

        border_labels = np.unique(
            np.concatenate(
                [
                    labels[0, :],
                    labels[-1, :],
                    labels[:, 0],
                    labels[:, -1],
                ]
            )
        )
        border = np.zeros(count + 1, dtype=bool)
        border[border_labels] = True

        areas = np.bincount(labels.ravel(), minlength=count + 1)
        fill_labels = (areas <= max_hole_area) & ~border
        fill_labels[0] = False
        fill = fill_labels[labels]
        filled = mask | fill
        return filled, int(np.count_nonzero(fill_labels)), int(np.count_nonzero(fill))

    background = ~mask
    visited = np.zeros(mask.shape, dtype=bool)
    filled = mask.copy()
    h, w = mask.shape
    fill_count = 0
    fill_area = 0

    for start_y, start_x in np.argwhere(background):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue

        queue: deque[tuple[int, int]] = deque([(y, x)])
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        touches_border = False

        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            if cy == 0 or cx == 0 or cy == h - 1 or cx == w - 1:
                touches_border = True
            for dy, dx in NEIGHBORS_8:
                ny = cy + dy
                nx = cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or not background[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((ny, nx))

        if not touches_border and len(pixels) <= max_hole_area:
            fill_count += 1
            fill_area += len(pixels)
            for py, px in pixels:
                filled[py, px] = True

    return filled, fill_count, fill_area


def suppress_detached_seed_artifacts(
    *,
    seed_all: np.ndarray,
    main_seed: np.ndarray,
    support: np.ndarray,
    keep: np.ndarray,
    max_area_ratio: float = 0.008,
    min_distance_ratio: float = 0.025,
    clearance_ratio: float = 0.018,
) -> tuple[np.ndarray, int, int]:
    if ndimage is None:
        return keep, 0, 0
    secondary = seed_all & ~main_seed
    labels, count = ndimage.label(secondary, structure=STRUCTURE_8)
    if count == 0:
        return keep, 0, 0

    h, w = seed_all.shape
    diagonal = float((h * h + w * w) ** 0.5)
    max_area = max(1, int(seed_all.size * max_area_ratio))
    min_distance = max(2.0, diagonal * min_distance_ratio)
    clearance = max(2, int(round(diagonal * clearance_ratio)))

    distance_to_main = ndimage.distance_transform_edt(~main_seed)
    protected_zone = ndimage.binary_dilation(
        main_seed,
        structure=STRUCTURE_8,
        iterations=clearance,
    )
    remove = np.zeros(seed_all.shape, dtype=bool)
    suppressed_count = 0

    for label in range(1, count + 1):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area > max_area:
            continue
        if float(distance_to_main[component].min()) < min_distance:
            continue
        local_support = support & ~protected_zone
        artifact_region = reconstruct_from_seed(component, local_support)
        artifact_region &= keep
        artifact_region &= distance_to_main >= clearance
        if not np.any(artifact_region):
            continue
        remove |= artifact_region
        suppressed_count += 1

    if suppressed_count == 0:
        return keep, 0, 0
    suppressed_area = int(np.count_nonzero(remove & keep))
    return keep & ~remove, suppressed_count, suppressed_area


def analyze_soft_alpha(
    alpha: np.ndarray,
    high_threshold: float = 0.85,
    low_threshold: float = 0.05,
    bbox_threshold: float = 0.12,
    max_hole_area_ratio: float = 0.02,
) -> TopologyResult:
    alpha_f = normalize_alpha(alpha)
    if alpha_f.size == 0:
        raise ValueError("alpha is empty")

    seed_all = alpha_f >= high_threshold
    seed, seed_component_count, main_seed_area, second_seed_area = _largest_component(
        seed_all
    )
    support = alpha_f >= low_threshold
    keep = reconstruct_from_seed(seed, support)
    keep, _, _ = suppress_detached_seed_artifacts(
        seed_all=seed_all,
        main_seed=seed,
        support=support,
        keep=keep,
    )

    removed_mask = support & ~keep
    removed_count, removed_area = _count_components(removed_mask)

    if max_hole_area_ratio <= 0:
        max_hole_area = 0
    else:
        max_hole_area = max(1, int(alpha_f.size * max_hole_area_ratio))
    keep_before_hole_fill = keep
    keep, hole_count, hole_area = fill_interior_holes(keep, max_hole_area)
    filled_holes = keep & ~keep_before_hole_fill

    final_alpha = alpha_f * keep.astype(np.float32)
    if hole_count > 0:
        final_alpha = np.where(filled_holes, 1.0, final_alpha)
    bbox = bbox_from_mask(final_alpha >= max(low_threshold, bbox_threshold))
    subject_coverage = float(np.count_nonzero(final_alpha > low_threshold) / alpha_f.size)
    seed_coverage = float(np.count_nonzero(seed) / alpha_f.size)
    support_coverage = float(np.count_nonzero(support) / alpha_f.size)
    uncertainty = (alpha_f > low_threshold) & (alpha_f < high_threshold) & keep
    edge_uncertainty = float(np.count_nonzero(uncertainty) / max(np.count_nonzero(keep), 1))
    seed_dominance = (
        float(main_seed_area / second_seed_area) if second_seed_area > 0 else None
    )

    return TopologyResult(
        alpha=final_alpha,
        keep_mask=keep,
        protected_core=seed & keep,
        bbox=bbox,
        subject_coverage=subject_coverage,
        seed_coverage=seed_coverage,
        support_coverage=support_coverage,
        edge_uncertainty_score=edge_uncertainty,
        removed_blob_count=removed_count,
        removed_blob_area=removed_area,
        hole_fill_count=hole_count,
        hole_fill_area=hole_area,
        seed_component_count=seed_component_count,
        main_seed_area=main_seed_area,
        second_seed_area=second_seed_area,
        seed_component_dominance=seed_dominance,
    )
