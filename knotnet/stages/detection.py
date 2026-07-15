"""Stage 3+4: YOLO-based Crossing and Endpoint Detection."""

from pathlib import Path
from typing import Optional
import numpy as np
import tempfile
from PIL import Image


def detect_crossings(
    img_full: np.ndarray,
    weights_path: Path,
    conf: float = 0.3,
    iou: float = 0.4,
    cache_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Detect crossings using YOLO pose model.

    Args:
        img_full: (1024, 1024, 3) uint8 normalized image
        weights_path: Path to YOLO crossing weights
        conf: confidence threshold
        iou: IoU threshold for NMS

    Returns:
        List of crossing dicts with keys:
            cid, box, cls, conf, kps, cx, cy
    """
    from ultralytics import YOLO

    # Save temp image for YOLO
    if cache_dir:
        tmp_path = cache_dir / "_tmp_detection_input.jpg"
    else:
        tmp_path = Path(tempfile.mktemp(suffix=".jpg"))

    Image.fromarray(img_full).save(tmp_path)

    yolo_model = YOLO(str(weights_path))
    results = yolo_model.predict(
        str(tmp_path), conf=conf, iou=iou, verbose=False
    )[0]

    crossings = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes
        keypoints = results.keypoints

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[j].cpu().numpy()
            cls_id = int(boxes.cls[j].cpu())
            confidence = float(boxes.conf[j].cpu())

            kps = None
            if keypoints is not None:
                kps = keypoints.data[j].cpu().numpy()  # (4, 3)

            crossings.append({
                "cid": j,
                "box": (float(x1), float(y1), float(x2), float(y2)),
                "cls": cls_id,
                "conf": confidence,
                "kps": kps.tolist() if kps is not None else None,
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
            })

    # Cleanup temp
    if not cache_dir and tmp_path.exists():
        tmp_path.unlink()

    return crossings


def detect_endpoints(
    img_full: np.ndarray,
    weights_path: Path,
    conf: float = 0.25,
    iou: float = 0.5,
    cache_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Detect rope endpoints using YOLO.

    Returns:
        List of endpoint dicts (max 2, sorted by x, labeled 'a'/'z')
    """
    from ultralytics import YOLO

    if cache_dir:
        tmp_path = cache_dir / "_tmp_detection_input.jpg"
    else:
        tmp_path = Path(tempfile.mktemp(suffix=".jpg"))

    if not tmp_path.exists():
        Image.fromarray(img_full).save(tmp_path)

    yolo_model = YOLO(str(weights_path))
    results = yolo_model.predict(
        str(tmp_path), conf=conf, iou=iou, verbose=False
    )[0]

    endpoints = []
    if results.boxes is not None and len(results.boxes) > 0:
        for j in range(len(results.boxes)):
            x1, y1, x2, y2 = results.boxes.xyxy[j].cpu().numpy()
            confidence = float(results.boxes.conf[j].cpu())
            endpoints.append({
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "conf": confidence,
                "box": (float(x1), float(y1), float(x2), float(y2)),
            })

    # Sort by confidence, keep top 2
    endpoints.sort(key=lambda e: -e["conf"])
    endpoints = endpoints[:2]

    # Assign labels by x-position
    if len(endpoints) == 2:
        endpoints.sort(key=lambda e: e["x"])
        endpoints[0]["label"] = "a"
        endpoints[1]["label"] = "z"
    elif len(endpoints) == 1:
        endpoints[0]["label"] = "a"

    # Cleanup
    if not cache_dir and tmp_path.exists():
        tmp_path.unlink()

    return endpoints


def locations_from_gt(
    gt_locations: dict,
    original_size: tuple[int, int],
    crop_offset: tuple[int, int],
    crop_side: int,
    scale: float,
) -> tuple[list[dict], list[dict]]:
    """
    Convert ground-truth locations to detection format.

    Args:
        gt_locations: dict with keys 'crossings' and 'endpoints'
            crossings: list of dicts with box_x, box_y, box_w, box_h,
                       over_1_x, over_1_y, ..., crossing_idx
            endpoints: list of dicts with x, y

    Returns:
        (crossings_detected, endpoints_detected)
    """
    crop_left, crop_top = crop_offset

    def gold_to_norm(x, y):
        return (
            float((x - crop_left) * scale),
            float((y - crop_top) * scale),
        )

    def gold_box_to_norm(bx, by, bw, bh):
        x1, y1 = gold_to_norm(bx, by)
        x2, y2 = gold_to_norm(bx + bw, by + bh)
        return x1, y1, x2, y2

    # Crossings
    crossings = []
    for row in gt_locations.get("crossings", []):
        bx = float(row["box_x"])
        by = float(row["box_y"])
        bw = float(row["box_w"])
        bh = float(row["box_h"])

        x1_n, y1_n, x2_n, y2_n = gold_box_to_norm(bx, by, bw, bh)
        cx_center = (x1_n + x2_n) / 2
        cy_center = (y1_n + y2_n) / 2

        kps = []
        for prefix in ["over_1", "over_2", "under_1", "under_2"]:
            kx = row.get(f"{prefix}_x")
            ky = row.get(f"{prefix}_y")
            if kx is not None and ky is not None:
                xn, yn = gold_to_norm(float(kx), float(ky))
                kps.append([xn, yn, 1.0])
            else:
                kps.append([0.0, 0.0, 0.0])

        crossings.append({
            "cid": int(row.get("crossing_idx", len(crossings))),
            "box": (x1_n, y1_n, x2_n, y2_n),
            "cls": 0,
            "conf": 1.0,
            "kps": kps,
            "cx": cx_center,
            "cy": cy_center,
        })

    # Endpoints
    endpoints = []
    for ep in gt_locations.get("endpoints", []):
        xn, yn = gold_to_norm(float(ep["x"]), float(ep["y"]))
        endpoints.append({
            "x": xn,
            "y": yn,
            "conf": 1.0,
            "box": (xn - 10, yn - 10, xn + 10, yn + 10),
        })

    # Label assignment
    endpoints = endpoints[:2]
    if len(endpoints) == 2:
        endpoints.sort(key=lambda e: e["x"])
        endpoints[0]["label"] = "a"
        endpoints[1]["label"] = "z"
    elif len(endpoints) == 1:
        endpoints[0]["label"] = "a"

    return crossings, endpoints