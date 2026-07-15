"""Stage 5: Token Construction from Detections."""

import numpy as np


def build_tokens(
    crossings: list[dict],
    endpoints: list[dict],
    img_size: int = 224,
    max_tokens: int = 40,
) -> list[dict]:
    """
    Construct token list from detections.

    Args:
        crossings: detected crossings with keypoints
        endpoints: detected endpoints with labels
        img_size: model input size
        max_tokens: maximum number of tokens

    Returns:
        List of token dicts with keys:
            x_full, y_full, x_norm, y_norm, type, pair, cid, label
    """
    SCALE = img_size / 1024.0
    tokens = []

    # Endpoint 'a'
    ep_a = next((e for e in endpoints if e.get("label") == "a"), None)
    if ep_a:
        tokens.append({
            "x_full": ep_a["x"],
            "y_full": ep_a["y"],
            "x_norm": ep_a["x"] * SCALE / img_size,
            "y_norm": ep_a["y"] * SCALE / img_size,
            "type": 1,
            "pair": 4,
            "cid": -1,
            "label": "a",
        })

    # Crossing keypoints
    label_letters = "bcdefghijklmnopqrstuvwxy"
    for c in crossings:
        cid = c["cid"]
        letter = label_letters[cid] if cid < len(label_letters) else f"x{cid}"

        if c.get("kps") is None:
            continue

        for k_idx in range(4):
            kp = c["kps"][k_idx]
            if isinstance(kp, (list, np.ndarray)) and len(kp) >= 3:
                kx, ky, kconf = float(kp[0]), float(kp[1]), float(kp[2])
            else:
                continue

            if kconf < 0.1:
                continue

            tokens.append({
                "x_full": kx,
                "y_full": ky,
                "x_norm": kx * SCALE / img_size,
                "y_norm": ky * SCALE / img_size,
                "type": 0,
                "pair": k_idx,
                "cid": cid,
                "label": f"{letter}{k_idx + 1}",
                "kp_conf": kconf,
            })

    # Endpoint 'z'
    ep_z = next((e for e in endpoints if e.get("label") == "z"), None)
    if ep_z:
        tokens.append({
            "x_full": ep_z["x"],
            "y_full": ep_z["y"],
            "x_norm": ep_z["x"] * SCALE / img_size,
            "y_norm": ep_z["y"] * SCALE / img_size,
            "type": 1,
            "pair": 5,
            "cid": -1,
            "label": "z",
        })

    # Truncate
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]

    return tokens