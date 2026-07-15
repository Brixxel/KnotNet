"""
Visualization utilities for KnotNet pipeline results.

Erzeugt ein Übersichtsbild mit:
  - Panel oben links: Detections (Crossings farblich Over/Under + Endpoints)
  - Panel oben rechts: Traversal-Pfad mit Pfeilen und Verbindungen
  - Panel unten: Compact Notation + PD-Code als Text
"""

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend für Server/Module
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe


# ═══════════════════════════════════════════════════════════════════════
# FARBEN
# ═══════════════════════════════════════════════════════════════════════

COLORS = {
    "over_1": "#0066FF",
    "over_2": "#00CCFF",
    "under_1": "#FF0066",
    "under_2": "#FF99CC",
    "over_strand": "#0088FF",
    "under_strand": "#FF3366",
    "endpoint": "#00FF88",
    "box": "#FFFF00",
    "traversal_arrow": "#FF8800",
    "traversal_line": "#FFAA00",
    "text_bg": "#1E1E2E",
}


def visualize_result(
    result: dict,
    img_full: np.ndarray,
    img_small: np.ndarray,
    tokens: list[dict],
    output_path: Optional[str | Path] = None,
    figsize: tuple = (20, 14),
    dpi: int = 120,
) -> Optional[plt.Figure]:
    """
    Erzeugt ein Übersichtsbild der Pipeline-Ergebnisse.

    Args:
        result: Pipeline output dict (von pipe.run())
        img_full: (1024, 1024, 3) uint8 — normalisiertes Bild
        img_small: (224, 224, 3) uint8 — Model-Input Bild
        tokens: Token-Liste (aus intermediates oder neu berechnet)
        output_path: Speicherpfad (.png/.jpg). None → nur Figure zurückgeben.
        figsize: Figure-Größe
        dpi: Auflösung

    Returns:
        matplotlib Figure (oder None wenn gespeichert + geschlossen)
    """
    fig = plt.figure(figsize=figsize, facecolor="#0D1117")
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.20,
                          height_ratios=[3, 1.2])

    crossings = result["detections"]["crossings"]
    endpoints = result["detections"]["endpoints"]
    sequence = result["sequence"]
    pd_code = result.get("pd_code", {})
    notation = result.get("knot_notation", {})
    topology = result.get("topology", {})

    img_size = img_small.shape[0]
    label_to_token = {t["label"]: t for t in tokens}

    # ═══ Panel 1: Detection Overview (Over/Under farblich) ════════════
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_detections(ax1, img_full, crossings, endpoints)

    # ═══ Panel 2: Traversal mit Pfeilen ══════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_traversal(ax2, img_small, tokens, sequence, crossings,
                    endpoints, img_size, notation)

    # ═══ Panel 3: Notation + PD-Code Text ═════════════════════════════
    ax3 = fig.add_subplot(gs[1, :])
    _draw_text_panel(ax3, result, notation, pd_code, topology, sequence)

    # ─── Titel ────────────────────────────────────────────────────────
    image_name = result.get("image_name", "unknown")
    mode_str = result.get("mode", "inference").upper()
    n_cx = topology.get("n_crossings", len(crossings))

    plt.suptitle(
        f"KnotNet Result [{mode_str}]: {image_name}  |  "
        f"{n_cx} Crossings  |  Writhe: {topology.get('writhe', '?'):+d}",
        fontsize=13, fontweight="bold", color="white", y=0.98,
    )

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return None

    return fig


# ═══════════════════════════════════════════════════════════════════════
# PANEL-ZEICHENFUNKTIONEN
# ═══════════════════════════════════════════════════════════════════════


def _draw_detections(ax, img_full, crossings, endpoints):
    """Panel 1: Bild mit Crossing-Boxen, KPs (over=blau, under=rot), Endpoints."""
    ax.imshow(img_full)
    ax.set_title("Detections\n(Over = blau, Under = rot)",
                 fontsize=11, fontweight="bold", color="white")

    kp_styles = [
        ("over_1",  COLORS["over_1"],  "o", 90),
        ("over_2",  COLORS["over_2"],  "o", 90),
        ("under_1", COLORS["under_1"], "s", 90),
        ("under_2", COLORS["under_2"], "s", 90),
    ]

    for c in crossings:
        x1, y1, x2, y2 = c["box"]

        # Box
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=COLORS["box"], facecolor="none",
        )
        ax.add_patch(rect)

        # Label
        ax.text(
            x1, y1 - 8, f"c{c['cid']} ({c['conf']:.2f})",
            fontsize=8, color="black", fontweight="bold",
            bbox=dict(facecolor=COLORS["box"], alpha=0.85,
                      pad=1.5, edgecolor="none"),
        )

        # Keypoints + Strang-Verbindungen
        if c.get("kps") is not None:
            kps = c["kps"]
            drawn_kps = {}

            for k_idx, (name, color, marker, size) in enumerate(kp_styles):
                kp = kps[k_idx]
                if isinstance(kp, (list, np.ndarray)) and len(kp) >= 3:
                    kx, ky, kconf = float(kp[0]), float(kp[1]), float(kp[2])
                else:
                    continue
                if kconf > 0.1:
                    ax.scatter(
                        kx, ky, c=color, s=size, marker=marker,
                        edgecolors="white", linewidths=1.5, zorder=5,
                    )
                    drawn_kps[k_idx] = (kx, ky)

            # Over-Strang Linie (blau)
            if 0 in drawn_kps and 1 in drawn_kps:
                ax.plot(
                    [drawn_kps[0][0], drawn_kps[1][0]],
                    [drawn_kps[0][1], drawn_kps[1][1]],
                    color=COLORS["over_strand"], linewidth=3,
                    alpha=0.85, solid_capstyle="round", zorder=4,
                )

            # Under-Strang Linie (rot, gestrichelt)
            if 2 in drawn_kps and 3 in drawn_kps:
                ax.plot(
                    [drawn_kps[2][0], drawn_kps[3][0]],
                    [drawn_kps[2][1], drawn_kps[3][1]],
                    color=COLORS["under_strand"], linewidth=3,
                    alpha=0.75, linestyle="--",
                    solid_capstyle="round", zorder=3,
                )

    # Endpoints
    for e in endpoints:
        ax.scatter(
            e["x"], e["y"], c=COLORS["endpoint"], s=400, marker="*",
            edgecolors="black", linewidths=2, zorder=6,
        )
        ax.text(
            e["x"] + 15, e["y"] + 5, e.get("label", "?"),
            fontsize=16, color="black", fontweight="bold",
            bbox=dict(facecolor=COLORS["endpoint"], alpha=0.9,
                      pad=3, edgecolor="black"),
        )

    # Legende
    legend_elements = [
        Line2D([0], [0], color=COLORS["over_strand"], lw=3,
               label="Over-Strang"),
        Line2D([0], [0], color=COLORS["under_strand"], lw=3,
               linestyle="--", label="Under-Strang"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=COLORS["endpoint"], markersize=12,
               label="Endpoint"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=9, framealpha=0.85)
    ax.axis("off")


def _draw_traversal(ax, img_small, tokens, sequence, crossings,
                    endpoints, img_size, notation):
    """Panel 2: Traversal-Pfad mit dünnen Pfeilen und Farbverlauf."""
    ax.imshow(img_small, alpha=0.45)
    ax.set_title("Traversal Sequence\n(Pfeil = Richtung, Farbe = Fortschritt)",
                 fontsize=11, fontweight="bold", color="white")

    label_to_xy = {}
    for t in tokens:
        label_to_xy[t["label"]] = (
            t["x_norm"] * img_size,
            t["y_norm"] * img_size,
        )

    n_steps = len(sequence)
    cmap = plt.cm.plasma

    # ─── Verbindungslinien mit Farbverlauf + Pfeile ───────────────────
    for i in range(n_steps - 1):
        lbl_a = sequence[i]
        lbl_b = sequence[i + 1]

        if lbl_a not in label_to_xy or lbl_b not in label_to_xy:
            continue

        x1, y1 = label_to_xy[lbl_a]
        x2, y2 = label_to_xy[lbl_b]

        progress = i / max(n_steps - 1, 1)
        color = cmap(progress)

        # Dünne Linie
        ax.plot(
            [x1, x2], [y1, y2],
            color=color, linewidth=1.8, alpha=0.8,
            solid_capstyle="round", zorder=3,
        )

        # Pfeil in der Mitte der Linie
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        dx = (x2 - x1) * 0.15
        dy = (y2 - y1) * 0.15

        ax.annotate(
            "",
            xy=(mid_x + dx, mid_y + dy),
            xytext=(mid_x - dx, mid_y - dy),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=1.5,
                mutation_scale=12,
            ),
            zorder=4,
        )

    # ─── Token-Marker ────────────────────────────────────────────────
    for i, lbl in enumerate(sequence):
        if lbl not in label_to_xy:
            continue
        x, y = label_to_xy[lbl]
        tok = next((t for t in tokens if t["label"] == lbl), None)
        if tok is None:
            continue

        progress = i / max(n_steps - 1, 1)

        if tok["type"] == 1:
            # Endpoint
            ax.scatter(
                x, y, c=COLORS["endpoint"], s=250, marker="*",
                edgecolors="black", linewidths=1.5, zorder=7,
            )
            ax.text(
                x + 6, y - 8, lbl,
                fontsize=12, fontweight="bold", color="white",
                path_effects=[
                    pe.withStroke(linewidth=3, foreground="black")
                ],
                zorder=8,
            )
        else:
            # Crossing KP
            pair = tok["pair"]
            if pair in (0, 1):
                color = COLORS["over_strand"]
                marker = "o"
            else:
                color = COLORS["under_strand"]
                marker = "s"

            ax.scatter(
                x, y, c=color, s=60, marker=marker,
                edgecolors="white", linewidths=1, zorder=6,
            )

            # Schrittnummer (klein)
            ax.text(
                x + 4, y + 4, f"{i}",
                fontsize=6, color="#CCCCCC", alpha=0.8,
                zorder=7,
            )

    # ─── Start/Ende markieren ─────────────────────────────────────────
    if sequence and sequence[0] in label_to_xy:
        sx, sy = label_to_xy[sequence[0]]
        ax.annotate(
            "START", (sx, sy),
            fontsize=8, fontweight="bold", color="#00FF88",
            xytext=(-20, -20), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#00FF88", lw=1.5),
            zorder=9,
        )

    if len(sequence) > 1 and sequence[-1] in label_to_xy:
        ex, ey = label_to_xy[sequence[-1]]
        ax.annotate(
            "END", (ex, ey),
            fontsize=8, fontweight="bold", color="#FF8800",
            xytext=(15, 15), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="#FF8800", lw=1.5),
            zorder=9,
        )

    # Colorbar für Fortschritt
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Traversal-Fortschritt", fontsize=8, color="white")
    cbar.ax.tick_params(labelsize=7, colors="white")

    ax.set_xlim(0, img_size)
    ax.set_ylim(img_size, 0)
    ax.axis("off")


def _draw_text_panel(ax, result, notation, pd_code, topology, sequence):
    """Panel 3: Notation, PD-Code, Jones als Text."""
    ax.set_facecolor(COLORS["text_bg"])
    ax.axis("off")

    # ─── Text zusammenbauen ───────────────────────────────────────────
    mode = result.get("mode", "?").upper()
    image_name = result.get("image_name", "?")

    # Knot Notation
    notation_full = notation.get("full", "N/A")
    notation_compact = notation.get("compact", "N/A")

    # PD-Code
    pd_str = pd_code.get("pd_code_str", "N/A")
    writhe = pd_code.get("writhe", "?")

    # Topology
    gauss = topology.get("gauss_code", "N/A")
    dt = topology.get("dt_notation", [])
    jones = topology.get("jones_str", "N/A")
    n_cx = topology.get("n_crossings", "?")

    # Timing
    timing_total = result.get("timing_total", 0)

    # Sequenz (gekürzt wenn zu lang)
    seq_str = " → ".join(sequence)
    if len(seq_str) > 100:
        seq_str = seq_str[:97] + "..."

    text = (
        f"{'━' * 90}\n"
        f"  MODE: {mode}   |   IMAGE: {image_name}   |   "
        f"TOTAL TIME: {timing_total:.2f}s\n"
        f"{'━' * 90}\n\n"
        f"  TRAVERSAL SEQUENCE ({len(sequence)} tokens):\n"
        f"    {seq_str}\n\n"
        f"  KNOT NOTATION (compact):\n"
        f"    {notation_compact}\n\n"
        f"  KNOT NOTATION (full):\n"
        f"    {notation_full[:120]}{'...' if len(notation_full) > 120 else ''}\n\n"
        f"  PD-CODE ({n_cx} crossings, writhe={writhe:+d}):\n"
        f"    {pd_str}\n\n"
        f"  GAUSS CODE:\n"
        f"    {gauss}\n\n"
        f"  DT NOTATION:\n"
        f"    {dt}\n\n"
        f"  JONES POLYNOMIAL V(A):\n"
        f"    {jones}\n"
        f"{'━' * 90}"
    )

    ax.text(
        0.02, 0.95, text,
        transform=ax.transAxes,
        fontsize=8.5, va="top", fontfamily="monospace",
        color="#E0E0FF",
        bbox=dict(facecolor=COLORS["text_bg"], alpha=0.95,
                  boxstyle="round,pad=0.5", edgecolor="#444"),
    )


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION (direkt aus Pipeline-Result)
# ═══════════════════════════════════════════════════════════════════════


def save_visualization(
    result: dict,
    image_path: str | Path,
    output_path: str | Path,
    img_size: int = 224,
    dpi: int = 120,
):
    """
    Convenience: Erzeugt Visualisierung direkt aus Pipeline-Result + Image.

    Args:
        result: Output von pipe.run()
        image_path: Original-Bild (wird erneut geladen + normalisiert)
        output_path: Speicherpfad für das Ergebnisbild
        img_size: Model-Input-Größe
        dpi: Auflösung
    """
    from ..stages.preprocessing import preprocess_image
    from ..stages.tokenization import build_tokens

    prep = preprocess_image(image_path, target_full_size=1024,
                            target_model_size=img_size)

    # Tokens rekonstruieren (falls nicht in intermediates)
    if "intermediates" in result and "tokens" in result["intermediates"]:
        tokens = result["intermediates"]["tokens"]
    else:
        tokens = build_tokens(
            result["detections"]["crossings"],
            result["detections"]["endpoints"],
            img_size=img_size,
        )

    visualize_result(
        result=result,
        img_full=prep["img_full"],
        img_small=prep["img_small"],
        tokens=tokens,
        output_path=output_path,
        dpi=dpi,
    )