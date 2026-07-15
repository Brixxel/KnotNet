"""Stage 7+: PD-Code, Gauss-Code, Writhe, Jones Polynomial."""

import numpy as np
from typing import Optional
from itertools import product as iterproduct
from collections import defaultdict

# ─── NEU: Am Anfang der Datei nach den Imports ergänzen ──────────────

def compute_knot_notation(
    tokens: list[dict],
    pred_sequence: list[str],
    crossings_order: list[dict],
) -> dict:
    """
    Berechnet die "Knot Notation" — die menschenlesbare Traversal-Notation
    VOR dem PD-Code.

    Format pro Crossing-Durchgang:
        <crossing_id>.<over|under>.<direction>
    
    Beispiel:
        a → c0.over.in → c0.over.out → c1.under.in → c1.under.out → ... → z

    Returns:
        dict with:
            - notation_sequence: list of notation strings
            - notation_str: formatted single-line string
            - notation_detailed: list of dicts with full info per step
            - crossing_visits: ordered list of {cid, strand_type, entry_dir}
    """
    label_to_idx = {t["label"]: i for i, t in enumerate(tokens)}

    notation_sequence = []
    notation_detailed = []
    crossing_visits = []

    # Track which crossings we've seen (for numbering visits)
    cid_visit_count = {}

    for lbl in pred_sequence:
        idx = label_to_idx.get(lbl)
        if idx is None:
            continue
        tok = tokens[idx]

        if tok["type"] == 1:
            # Endpoint
            notation_sequence.append(lbl)
            notation_detailed.append({
                "label": lbl,
                "type": "endpoint",
                "token_idx": idx,
            })
            continue

        cid = tok["cid"]
        pair = tok["pair"]

        # pair 0 = over_1 (over-strand entry)
        # pair 1 = over_2 (over-strand exit)
        # pair 2 = under_1 (under-strand entry)
        # pair 3 = under_2 (under-strand exit)
        strand_type = "over" if pair in (0, 1) else "under"
        direction = "in" if pair in (0, 2) else "out"

        notation_token = f"c{cid}.{strand_type}.{direction}"
        notation_sequence.append(notation_token)

        detail = {
            "label": lbl,
            "notation": notation_token,
            "type": "crossing_kp",
            "cid": cid,
            "strand_type": strand_type,
            "direction": direction,
            "pair": pair,
            "token_idx": idx,
            "x_norm": tok["x_norm"],
            "y_norm": tok["y_norm"],
        }
        notation_detailed.append(detail)

        # Track crossing visits (only on entry)
        if direction == "in":
            visit_num = cid_visit_count.get(cid, 0) + 1
            cid_visit_count[cid] = visit_num
            crossing_visits.append({
                "cid": cid,
                "strand_type": strand_type,
                "visit_number": visit_num,
                "entry_label": lbl,
            })

    notation_str = " → ".join(notation_sequence)

    # Kompakte Crossing-Level Notation
    # Nur die Crossings (ohne Endpoints), mit über/unter
    compact_parts = []
    seen_strands = {}  # cid → list of strand_types
    for visit in crossing_visits:
        cid = visit["cid"]
        st = visit["strand_type"]
        compact_parts.append(f"c{cid}({'O' if st == 'over' else 'U'})")
        seen_strands.setdefault(cid, []).append(st)

    compact_str = " → ".join(compact_parts)

    return {
        "notation_sequence": notation_sequence,
        "notation_str": notation_str,
        "notation_detailed": notation_detailed,
        "crossing_visits": crossing_visits,
        "compact_notation": compact_str,
        "compact_parts": compact_parts,
    }

def compute_pd_code(
    tokens: list[dict],
    pred_sequence: list[str],
    crossings_order: list[dict],
) -> dict:
    """
    Compute PD-Code from predicted sequence.

    Returns:
        dict with keys:
            pd_crossings: list of {cid, pd, sign, label}
            pd_code_str: formatted string
            writhe: int
    """
    label_to_idx = {t["label"]: i for i, t in enumerate(tokens)}

    # KP-only path (closed knot: remove endpoints a/z)
    kp_only = [lbl for lbl in pred_sequence if lbl not in ("a", "z")]
    N_kp = len(kp_only)

    if N_kp == 0:
        return {"pd_crossings": [], "pd_code_str": "", "writhe": 0}

    kp_pos = {lbl: i for i, lbl in enumerate(kp_only)}

    def arc_in(lbl):
        i = kp_pos[lbl]
        return ((i - 1) % N_kp) + 1

    def arc_out(lbl):
        i = kp_pos[lbl]
        return (i % N_kp) + 1

    pd_crossings = []
    for c_info in crossings_order:
        cid = c_info["cid"]
        cx_tok = {
            t["pair"]: t
            for t in tokens
            if t["cid"] == cid and t["label"] in kp_pos
        }

        if len(cx_tok) < 4:
            continue

        t_o1 = cx_tok[0]
        t_o2 = cx_tok[1]
        t_u1 = cx_tok[2]
        t_u2 = cx_tok[3]

        l = arc_in(t_o1["label"])
        j = arc_out(t_o2["label"])
        i = arc_in(t_u1["label"])
        k = arc_out(t_u2["label"])

        # Sign from cross product
        over_vec = np.array([
            t_o2["x_norm"] - t_o1["x_norm"],
            t_o2["y_norm"] - t_o1["y_norm"],
        ])
        under_vec = np.array([
            t_u2["x_norm"] - t_u1["x_norm"],
            t_u2["y_norm"] - t_u1["y_norm"],
        ])
        cross_z = over_vec[0] * under_vec[1] - over_vec[1] * under_vec[0]
        sign = "+" if cross_z > 0 else "-"

        pd_entry = [i, j, k, l]
        if k < i:
            pd_entry = [k, l, i, j]

        pd_crossings.append({
            "cid": cid,
            "pd": pd_entry,
            "sign": sign,
            "label": f"c{cid}",
        })

    # Writhe
    writhe = sum(1 if c["sign"] == "+" else -1 for c in pd_crossings)

    pd_code_str = "  ".join(
        f"X[{','.join(map(str, c['pd']))}]" for c in pd_crossings
    )

    return {
        "pd_crossings": pd_crossings,
        "pd_code_str": pd_code_str,
        "writhe": writhe,
    }


def compute_topology(
    pd_crossings: list[dict],
    tokens: list[dict],
    pred_sequence: list[str],
) -> dict:
    """
    Compute full topological representations.

    Returns:
        dict with:
            gauss_code, dt_notation, writhe, n_crossings,
            jones_terms, jones_str, alexander_str
    """
    label_to_idx = {t["label"]: i for i, t in enumerate(tokens)}
    pd_raw = [c["pd"] for c in pd_crossings]
    signs = [c["sign"] for c in pd_crossings]
    n_crossings = len(pd_crossings)
    writhe = sum(1 if s == "+" else -1 for s in signs if s in ("+", "-"))

    # Gauss code
    gauss_code = _compute_gauss_code(tokens, pred_sequence, pd_crossings)

    # DT notation
    dt_notation = _compute_dt_notation(tokens, pred_sequence, pd_crossings)

    # Jones polynomial (Kauffman bracket)
    jones_terms = {}
    jones_str = "1"
    if n_crossings > 0 and n_crossings <= 12:
        jones_terms, jones_str = _compute_jones(pd_raw, signs, writhe)

    return {
        "gauss_code": gauss_code,
        "dt_notation": dt_notation,
        "writhe": writhe,
        "n_crossings": n_crossings,
        "jones_terms": jones_terms,
        "jones_str": jones_str,
    }


def _compute_gauss_code(
    tokens: list[dict],
    pred_sequence: list[str],
    pd_crossings: list[dict],
) -> str:
    """Compute extended Gauss code."""
    label_to_idx = {t["label"]: i for i, t in enumerate(tokens)}
    parts = []

    for lbl in pred_sequence:
        idx = label_to_idx.get(lbl)
        if idx is None:
            continue
        tok = tokens[idx]
        if tok["type"] == 1:
            continue

        cid = tok["cid"]
        is_over = tok["pair"] in (0, 1)
        cx_sign = next(
            (c["sign"] for c in pd_crossings if c["cid"] == cid), "+"
        )
        prefix = "O" if is_over else "U"
        parts.append(f"{prefix}{cid + 1}{cx_sign}")

    return " ".join(parts)


def _compute_dt_notation(
    tokens: list[dict],
    pred_sequence: list[str],
    pd_crossings: list[dict],
) -> list[int]:
    """Compute Dowker-Thistlethwaite notation."""
    label_to_idx = {t["label"]: i for i, t in enumerate(tokens)}
    all_steps = []
    step_counter = 1

    for lbl in pred_sequence:
        idx = label_to_idx.get(lbl)
        if idx is None:
            continue
        tok = tokens[idx]
        if tok["type"] == 1:
            continue
        all_steps.append({
            "step": step_counter,
            "cid": tok["cid"],
            "is_over": tok["pair"] in (0, 1),
        })
        step_counter += 1

    dt_pairs = {}
    for entry in all_steps:
        dt_pairs.setdefault(entry["cid"], []).append(entry["step"])

    dt_notation = []
    for cid in sorted(dt_pairs.keys()):
        steps = dt_pairs[cid]
        if len(steps) == 2:
            even_s = steps[1] if steps[1] % 2 == 0 else steps[0]
            cx_sign = next(
                (c["sign"] for c in pd_crossings if c["cid"] == cid), "+"
            )
            signed_even = even_s if cx_sign == "+" else -even_s
            dt_notation.append(signed_even)

    return dt_notation


def _compute_jones(
    pd_list: list[list[int]],
    signs: list[str],
    writhe: int,
) -> tuple[dict, str]:
    """Compute Jones polynomial via Kauffman bracket state sum."""
    try:
        import sympy as sp

        A = sp.Symbol("A")
        n = len(pd_list)

        bracket = sp.Integer(0)
        d_loop = -A**2 - A ** (-2)

        for state in iterproduct([0, 1], repeat=n):
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                if parent[x] != x:
                    parent[x] = find(parent[x])
                return parent[x]

            def union(x, y):
                parent[find(x)] = find(y)

            for idx, (pd, s) in enumerate(zip(pd_list, state)):
                i_a, j_a, k_a, l_a = pd
                if s == 0:
                    union(i_a, l_a)
                    union(j_a, k_a)
                else:
                    union(i_a, j_a)
                    union(k_a, l_a)

            all_arcs = set(a for pd in pd_list for a in pd)
            n_loops = len({find(a) for a in all_arcs})

            n0 = state.count(0)
            n1 = state.count(1)
            weight = A ** (n0 - n1) * d_loop ** (n_loops - 1)
            bracket = bracket + weight

        bracket = sp.expand(bracket)
        jones = sp.expand((-A**3) ** (-writhe) * bracket)

        # Extract terms
        poly = sp.Poly(jones, A)
        terms = {m[0]: int(c) for m, c in zip(poly.monoms(), poly.coeffs())}

        # Format
        jones_str = _format_laurent(terms, "A")

        return terms, jones_str

    except Exception as e:
        return {}, f"Error: {e}"


def _format_laurent(terms: dict, var_name: str = "A") -> str:
    """Format Laurent polynomial."""
    if not terms:
        return "0"
    parts = []
    for exp in sorted(terms.keys(), reverse=True):
        c = terms[exp]
        if c == 0:
            continue
        if exp == 0:
            parts.append(f"{c:+d}")
        elif abs(c) == 1:
            sign = "+" if c > 0 else "-"
            parts.append(f"{sign}{var_name}^{exp}")
        else:
            parts.append(f"{c:+d}{var_name}^{exp}")
    return " ".join(parts).lstrip("+").strip() or "0"