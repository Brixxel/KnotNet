"""Stage 6: Sequence Prediction using KnotGraphNet V5."""

import numpy as np
import torch
from typing import Optional

from ..models.knotgraphnet import KnotGraphNet


def predict_sequence(
    tokens: list[dict],
    img_small: np.ndarray,
    skel_model: np.ndarray,
    model: KnotGraphNet,
    config,
) -> dict:
    """
    Predict traversal sequence from tokens using KnotGraphNet.

    Returns:
        dict with keys:
            - sequence: list of token labels in order
            - crossings_order: list of crossing-level dicts
            - edge_logits: (n_tok, n_tok) numpy array
            - path_indices: list of token indices
    """
    device = config.device
    max_tokens = config.max_tokens
    n_tok = len(tokens)

    if n_tok < 3:
        return {
            "sequence": [],
            "crossings_order": [],
            "edge_logits": None,
            "path_indices": [],
        }

    # Build input tensors
    token_xy = torch.zeros(1, max_tokens, 2)
    token_type = torch.zeros(1, max_tokens, dtype=torch.long)
    token_pair = torch.zeros(1, max_tokens, dtype=torch.long)
    token_cid = torch.full((1, max_tokens), -1, dtype=torch.long)
    tok_mask = torch.zeros(1, max_tokens, dtype=torch.bool)

    for i, t in enumerate(tokens):
        token_xy[0, i, 0] = float(t["x_norm"])
        token_xy[0, i, 1] = float(t["y_norm"])
        token_type[0, i] = int(t["type"])
        token_pair[0, i] = int(t["pair"])
        token_cid[0, i] = int(t["cid"])
        tok_mask[0, i] = True

    img_t = torch.from_numpy(img_small).permute(2, 0, 1).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)
    skel_t = torch.from_numpy(skel_model).unsqueeze(0).unsqueeze(0).float().to(device)

    token_xy = token_xy.to(device)
    token_type = token_type.to(device)
    token_pair = token_pair.to(device)
    token_cid = token_cid.to(device)
    tok_mask = tok_mask.to(device)

    # Forward pass
    with torch.no_grad():
        edge_logits = model(
            img_t, skel_t, token_xy, token_type, token_cid, token_pair, tok_mask
        )

    edge_logits_np = edge_logits[0, :n_tok, :n_tok].cpu().numpy()

    # Symmetrize
    edge_logits_np = 0.5 * (edge_logits_np + edge_logits_np.T)
    edge_logits_safe = np.where(np.isinf(edge_logits_np), -1e6, edge_logits_np)

    # Build graph
    sequence, path_indices, crossings_order = _build_path(
        tokens, edge_logits_safe, n_tok
    )

    return {
        "sequence": sequence,
        "crossings_order": crossings_order,
        "edge_logits": edge_logits_np,
        "path_indices": path_indices,
    }


def _build_path(
    tokens: list[dict],
    edge_logits: np.ndarray,
    n_tok: int,
) -> tuple[list[str], list[int], list[dict]]:
    """Build path through token graph using greedy symmetric matching."""
    adj = {i: set() for i in range(n_tok)}

    # Implicit edges (same crossing, same strand)
    crossings = {}
    for i, t in enumerate(tokens):
        if t["cid"] < 0:
            continue
        crossings.setdefault(t["cid"], {})[t["pair"]] = i

    for cid, slots in crossings.items():
        if 0 in slots and 1 in slots:
            adj[slots[0]].add(slots[1])
            adj[slots[1]].add(slots[0])
        if 2 in slots and 3 in slots:
            adj[slots[2]].add(slots[3])
            adj[slots[3]].add(slots[2])

    # Target degrees
    target_deg = {}
    for i, t in enumerate(tokens):
        target_deg[i] = 1 if t["label"] in ("a", "z") else 2

    needed = {i: target_deg[i] - len(adj[i]) for i in range(n_tok)}

    # Forbidden intra-crossing pairs
    intra_crossing_pairs = set()
    for cid, slots in crossings.items():
        idxs = list(slots.values())
        for i in idxs:
            for j in idxs:
                if i != j:
                    intra_crossing_pairs.add((min(i, j), max(i, j)))

    # Collect candidates
    candidates = []
    for i in range(n_tok):
        for j in range(i + 1, n_tok):
            if (i, j) in intra_crossing_pairs:
                continue
            if j in adj[i]:
                continue
            candidates.append((edge_logits[i, j], i, j))

    candidates.sort(reverse=True)

    # Greedy matching
    for score, i, j in candidates:
        if needed[i] > 0 and needed[j] > 0:
            adj[i].add(j)
            adj[j].add(i)
            needed[i] -= 1
            needed[j] -= 1

    # Walk path from 'a'
    a_idx = next((i for i, t in enumerate(tokens) if t["label"] == "a"), None)
    z_idx = next((i for i, t in enumerate(tokens) if t["label"] == "z"), None)

    if a_idx is None:
        return [], [], []

    path = [a_idx]
    visited = {a_idx}
    current = a_idx

    for _ in range(n_tok + 5):
        unvisited = [n for n in adj[current] if n not in visited]
        if not unvisited:
            break
        if len(unvisited) > 1:
            unvisited.sort(key=lambda n: -edge_logits[current, n])
        current = unvisited[0]
        path.append(current)
        visited.add(current)
        if current == z_idx:
            break

    sequence = [tokens[i]["label"] for i in path]

    # Extract crossing-level order
    crossings_order = []
    seen_cids = set()
    for i in path:
        t = tokens[i]
        if t["cid"] < 0 or t["cid"] in seen_cids:
            continue
        seen_cids.add(t["cid"])
        cx_tokens = [tokens[j] for j in range(n_tok) if tokens[j]["cid"] == t["cid"]]
        mx = float(np.mean([tk["x_norm"] for tk in cx_tokens]))
        my = float(np.mean([tk["y_norm"] for tk in cx_tokens]))
        crossings_order.append({
            "cid": t["cid"],
            "label": f"c{t['cid']}",
            "x_norm": mx,
            "y_norm": my,
        })

    return sequence, path, crossings_order