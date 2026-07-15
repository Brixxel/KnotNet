"""
Setup-Skript: Kopiert alle notwendigen Modelle und Dateien
in ein eigenständiges Deployment-Verzeichnis.

Usage:
    python -m knotnet.setup_deployment \\
        --source-root /path/to/knot-cv \\
        --target-dir ./knotnet_deployment

Oder programmatisch:
    from knotnet.setup_deployment import setup_deployment
    setup_deployment(source_root, target_dir)
"""

import argparse
import shutil
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# DEFAULT SOURCE PATHS (relativ zum knot-cv Repo-Root)
# Passe diese an, wenn sich Pfade ändern!
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_SOURCE_MAPPINGS = {
    # target_relative_path → source_relative_path (vom Repo-Root aus)
    "weights/knotgraphnet_v5_best.pt": (
        "10_data/10_train/derived/results/knotgraphnet_v5_best.pt"
    ),
    "weights/yolo_crossings_best.pt": (
        "30_models/crossing/"
        "arch_v26xlarge_yolo26x-pose_20260625_054512/weights/best.pt"
    ),
    "weights/yolo_endpoints_best.pt": (
        "30_models/endpoint/"
        "ep_v8s_box100_20260707_181519/weights/best.pt"
    ),
}

# Optionale Dateien (werden kopiert falls vorhanden)
OPTIONAL_FILES = {
    "config/knot_lookup_table.json": None,  # Wird inline generiert
}


def setup_deployment(
    source_root: str | Path,
    target_dir: str | Path,
    source_mappings: dict | None = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    Kopiert alle notwendigen Dateien in das Deployment-Verzeichnis.

    Args:
        source_root: Root-Pfad des knot-cv Repositories
        target_dir: Ziel-Verzeichnis für Deployment
        source_mappings: Optional dict {target_rel: source_rel}
                        Wenn None, wird DEFAULT_SOURCE_MAPPINGS verwendet.
        force: Wenn True, überschreibt existierende Dateien.
        verbose: Wenn True, gibt Fortschrittsmeldungen aus.

    Returns:
        Path zum Deployment-Verzeichnis
    """
    source_root = Path(source_root).resolve()
    target_dir = Path(target_dir).resolve()
    mappings = source_mappings or DEFAULT_SOURCE_MAPPINGS

    if verbose:
        print("=" * 70)
        print("  KnotNet Deployment Setup")
        print("=" * 70)
        print(f"  Source root: {source_root}")
        print(f"  Target dir:  {target_dir}")
        print(f"  Force:       {force}")
        print()

    # Erstelle Verzeichnisstruktur
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "weights").mkdir(exist_ok=True)
    (target_dir / "cache").mkdir(exist_ok=True)
    (target_dir / "outputs").mkdir(exist_ok=True)
    (target_dir / "config").mkdir(exist_ok=True)

    # Kopiere Dateien
    success = 0
    failed = 0
    skipped = 0

    for target_rel, source_rel in mappings.items():
        target_path = target_dir / target_rel
        source_path = source_root / source_rel

        if target_path.exists() and not force:
            if verbose:
                print(f"  ⏭️  SKIP (exists): {target_rel}")
            skipped += 1
            continue

        if not source_path.exists():
            print(f"  ❌ NOT FOUND: {source_path}")
            failed += 1
            continue

        # Erstelle Ziel-Verzeichnis
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Kopiere
        shutil.copy2(source_path, target_path)
        size_mb = target_path.stat().st_size / (1024 * 1024)

        if verbose:
            print(f"  ✅ COPIED: {target_rel} ({size_mb:.1f} MB)")
        success += 1

    # Erstelle Standard-Config
    _write_default_config(target_dir)

    # Zusammenfassung
    if verbose:
        print()
        print("─" * 70)
        print(f"  ✅ Kopiert: {success}")
        print(f"  ⏭️  Übersprungen: {skipped}")
        print(f"  ❌ Fehlend: {failed}")
        print()

        if failed == 0:
            print(f"  🎉 Deployment bereit in: {target_dir}")
            print()
            print("  Verwendung:")
            print(f"    from knotnet import KnotNetPipeline, PipelineConfig")
            print(f"    config = PipelineConfig(deployment_dir='{target_dir}')")
            print(f"    pipe = KnotNetPipeline(config=config)")
            print(f"    result = pipe.run('mein_bild.jpg')")
        else:
            print(f"  ⚠️  {failed} Datei(en) fehlen!")
            print(f"      Bitte Quellpfade prüfen oder manuell kopieren.")

    return target_dir


def _write_default_config(target_dir: Path):
    """Schreibt eine Standard-Config-Datei ins Deployment."""
    import json

    config_path = target_dir / "config" / "deployment_info.json"
    info = {
        "version": "0.1.0",
        "model_architecture": {
            "name": "KnotGraphNet_V5",
            "img_size": 224,
            "max_tokens": 40,
            "num_fourier": 8,
            "d_model": 64,
            "num_heads": 2,
            "num_layers": 2,
        },
        "detection": {
            "crossing_model": "YOLOv26x-pose (crossings)",
            "endpoint_model": "YOLOv8s (endpoints)",
        },
        "files": {
            "v5_weights": "weights/knotgraphnet_v5_best.pt",
            "yolo_crossings": "weights/yolo_crossings_best.pt",
            "yolo_endpoints": "weights/yolo_endpoints_best.pt",
        },
    }
    with open(config_path, "w") as f:
        json.dump(info, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Setup KnotNet Deployment Directory"
    )
    parser.add_argument(
        "--source-root",
        "-s",
        type=str,
        required=True,
        help="Root of the knot-cv repository",
    )
    parser.add_argument(
        "--target-dir",
        "-t",
        type=str,
        default="./knotnet_deployment",
        help="Target deployment directory",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing files",
    )
    args = parser.parse_args()

    setup_deployment(
        source_root=args.source_root,
        target_dir=args.target_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()