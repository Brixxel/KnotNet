"""Skeleton extraction utility functions."""

import numpy as np
import cv2
from skimage.morphology import (
    skeletonize,
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk,
)
from skimage.measure import label as measure_label
from scipy import ndimage


def preprocess_for_sam2(img_rgb: np.ndarray) -> np.ndarray:
    """CLAHE + Bilateral Filter preprocessing."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    img_eq = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(img_eq, 7, 50, 50)


def clean_mask(
    mask: np.ndarray, min_hole: int = 400, min_obj: int = 300
) -> np.ndarray:
    """Clean binary mask."""
    cleaned = remove_small_holes(mask, area_threshold=min_hole)
    cleaned = remove_small_objects(cleaned, min_size=min_obj)
    cleaned = binary_opening(cleaned, disk(1))
    cleaned = binary_closing(cleaned, disk(2))
    return cleaned.astype(bool)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(skel.astype(np.uint8), -1, k)


def _remove_spurs(skel: np.ndarray, min_len: int = 15) -> np.ndarray:
    """Remove short branches from skeleton."""
    skel = skel.astype(bool).copy()
    for _ in range(5):
        nc = _neighbor_count(skel)
        endpoints = list(zip(*np.where(skel & (nc == 1))))
        junctions = set(zip(*np.where(skel & (nc >= 3))))
        if not endpoints:
            break
        changed = False
        for ep in endpoints:
            path = [ep]
            visited = {ep}
            cy, cx = ep
            for _ in range(min_len + 5):
                nbrs = [
                    (cy + dy, cx + dx)
                    for dy in (-1, 0, 1)
                    for dx in (-1, 0, 1)
                    if (dy or dx)
                    and 0 <= cy + dy < skel.shape[0]
                    and 0 <= cx + dx < skel.shape[1]
                    and skel[cy + dy, cx + dx]
                    and (cy + dy, cx + dx) not in visited
                ]
                if not nbrs:
                    break
                path.append(nbrs[0])
                visited.add(nbrs[0])
                if nbrs[0] in junctions:
                    break
                cy, cx = nbrs[0]
            if len(path) < min_len and path[-1] in junctions:
                for y, x in path[:-1]:
                    skel[y, x] = False
                changed = True
        if not changed:
            break
    return skel


def skeletonize_full(
    mask: np.ndarray,
    min_spur_len: int = 15,
    keep_largest: bool = True,
) -> np.ndarray:
    """Full skeletonization pipeline."""
    skel = skeletonize(mask, method="lee").astype(bool)
    skel = _remove_spurs(skel, min_len=min_spur_len)
    if keep_largest:
        lbl = measure_label(skel, connectivity=2)
        if lbl.max() > 1:
            sizes = ndimage.sum(skel, lbl, range(1, lbl.max() + 1))
            skel = lbl == (np.argmax(sizes) + 1)
    return skel.astype(np.uint8)