"""Stage 1: Image Preprocessing & Normalization."""

import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def preprocess_image(
    image_path: str | Path,
    target_full_size: int = 1024,
    target_model_size: int = 224,
) -> dict:
    """
    Load and normalize image.

    Returns:
        dict with keys:
            - img_full:  (H, W, 3) uint8 at target_full_size
            - img_small: (h, w, 3) uint8 at target_model_size
            - original_size: (W_orig, H_orig)
            - crop_offset: (left, top)
            - scale: float
    """
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    # Center crop to square
    side = min(W, H)
    left = (W - side) // 2
    top = (H - side) // 2
    img_sq = img.crop((left, top, left + side, top + side))

    # Resize to target
    img_full = img_sq.resize(
        (target_full_size, target_full_size), Image.BILINEAR
    )
    img_full_np = np.array(img_full)

    # Small version for model input
    img_small_np = cv2.resize(
        img_full_np,
        (target_model_size, target_model_size),
        interpolation=cv2.INTER_AREA,
    )

    scale = target_full_size / side

    return {
        "img_full": img_full_np,
        "img_small": img_small_np,
        "original_size": (W, H),
        "crop_offset": (left, top),
        "crop_side": side,
        "scale": scale,
    }