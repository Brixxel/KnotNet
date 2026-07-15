"""Stage 2: Skeleton Extraction via SAM2 + YOLO Prompts."""

from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PIL import Image

from ..utils.skeleton_ops import (
    preprocess_for_sam2,
    clean_mask,
    skeletonize_full,
)


def extract_skeleton(
    img_full: np.ndarray,
    crossings: list[dict],
    endpoints: list[dict],
    config,
    cache_key: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract skeleton from image using SAM2 + detection prompts.

    Args:
        img_full: (1024, 1024, 3) uint8
        crossings: detected crossings
        endpoints: detected endpoints
        config: PipelineConfig instance
        cache_key: string for cache filename (e.g. image stem)

    Returns:
        (skel_full, skel_model_input)
        - skel_full: (1024, 1024) uint8 binary
        - skel_model_input: (img_size, img_size) float32
    """
    img_size = config.img_size
    skel_threshold = config.skel_threshold

    # Check cache
    if cache_key and config.use_skeleton_cache:
        cache_path = config.cache_dir / f"_skel_{cache_key}.png"
        if cache_path.exists():
            skel_full = np.array(Image.open(cache_path).convert("L"))
            skel_full = (skel_full > 127).astype(np.uint8)
            skel_model = _resize_skeleton(skel_full, img_size, skel_threshold)
            return skel_full, skel_model

    # Try SAM2
    try:
        from knotcv.segmentation import RopeSegmenter
        skel_full = _skeleton_via_sam2(
            img_full, crossings, endpoints, config
        )
    except ImportError:
        skel_full = _skeleton_fallback(img_full)

    # Cache
    if cache_key and config.use_skeleton_cache:
        cache_path = config.cache_dir / f"_skel_{cache_key}.png"
        Image.fromarray((skel_full * 255).astype(np.uint8)).save(cache_path)

    skel_model = _resize_skeleton(skel_full, img_size, skel_threshold)
    return skel_full, skel_model


def _resize_skeleton(
    skel_full: np.ndarray, img_size: int, threshold: float
) -> np.ndarray:
    """Resize skeleton to model input size."""
    skel_resized = cv2.resize(
        skel_full.astype(np.float32),
        (img_size, img_size),
        interpolation=cv2.INTER_AREA,
    )
    return (skel_resized > threshold).astype(np.float32)


def _skeleton_via_sam2(
    img_rgb: np.ndarray,
    crossings: list[dict],
    endpoints: list[dict],
    config,
) -> np.ndarray:
    """SAM2-based skeleton extraction."""
    from knotcv.segmentation import RopeSegmenter

    H0, W0 = img_rgb.shape[:2]
    target_long_side = config.skeleton_target_long_side

    # Collect prompts
    pts = []
    for e in endpoints:
        pts.append([e["x"], e["y"]])
    for c in crossings:
        if c.get("kps") is not None:
            for k_idx in range(4):
                kp = c["kps"][k_idx]
                if isinstance(kp, (list, np.ndarray)) and len(kp) >= 3:
                    kx, ky, kconf = kp[0], kp[1], kp[2]
                else:
                    continue
                if kconf > 0.1:
                    pts.append([float(kx), float(ky)])
        else:
            x1, y1, x2, y2 = c["box"]
            pts.append([(x1 + x2) / 2, (y1 + y2) / 2])

    if not pts:
        return np.zeros((H0, W0), dtype=np.uint8)

    pts = np.array(pts)

    # Dedupe
    kept = []
    for p in pts:
        if not kept or np.min(np.linalg.norm(np.array(kept) - p, axis=1)) > 30:
            kept.append(p)
    pts = np.array(kept)

    # Resize for SAM2
    long_side = max(H0, W0)
    if long_side > target_long_side:
        scale = target_long_side / long_side
        new_W, new_H = int(W0 * scale), int(H0 * scale)
        img_small = cv2.resize(
            img_rgb, (new_W, new_H), interpolation=cv2.INTER_AREA
        )
        pts_small = pts * scale
    else:
        scale = 1.0
        img_small = img_rgb
        pts_small = pts
        new_H, new_W = H0, W0

    # Preprocess + SAM2
    img_proc = preprocess_for_sam2(img_small)
    margin = 20
    negative_xy = np.array(
        [
            [margin, margin],
            [new_W - margin, margin],
            [margin, new_H - margin],
            [new_W - margin, new_H - margin],
        ],
        dtype=np.float32,
    )

    segmenter = RopeSegmenter(variant="small", device=str(config.device))
    rope_mask = segmenter.segment(
        image_rgb=img_proc,
        positive_xy=pts_small.astype(np.float32),
        negative_xy=negative_xy,
        multimask=True,
        refine=True,
    )
    mask_small = rope_mask.mask

    # Resize mask back
    if scale != 1.0:
        mask_full = cv2.resize(
            mask_small.astype(np.uint8),
            (W0, H0),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    else:
        mask_full = mask_small

    mask_full = clean_mask(mask_full)
    skel_full = skeletonize_full(mask_full, min_spur_len=15)

    return skel_full


def _skeleton_fallback(img_rgb: np.ndarray) -> np.ndarray:
    """Otsu-based fallback skeleton extraction."""
    from skimage.morphology import skeletonize

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return skeletonize(binary > 0).astype(np.uint8)