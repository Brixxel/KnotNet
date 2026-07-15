"""
KnotNet Pipeline — Main orchestrator class.

Usage:
    from knotnet import KnotNetPipeline, PipelineConfig

    config = PipelineConfig(deployment_dir="/path/to/deployment")
    pipe = KnotNetPipeline(config=config)

    # Full inference mode
    result = pipe.run("image.jpg")

    # Gold/GT mode — provide ground-truth locations
    result = pipe.run("image.jpg", gt_locations={
        "crossings": [
            {"crossing_idx": 0, "box_x": 100, "box_y": 200, "box_w": 80, "box_h": 80,
             "over_1_x": 110, "over_1_y": 210, "over_2_x": 170, "over_2_y": 220,
             "under_1_x": 130, "under_1_y": 195, "under_2_x": 140, "under_2_y": 270},
            ...
        ],
        "endpoints": [
            {"x": 50, "y": 300},
            {"x": 900, "y": 100},
        ]
    })
"""

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import PipelineConfig
from .models.knotgraphnet import KnotGraphNet
from .stages.preprocessing import preprocess_image
from .stages.detection import (
    detect_crossings,
    detect_endpoints,
    locations_from_gt,
)
from .stages.skeleton import extract_skeleton
from .stages.tokenization import build_tokens
from .stages.sequence import predict_sequence
from .stages.topology import compute_pd_code, compute_topology, compute_knot_notation


class KnotNetPipeline:
    """End-to-end knot inference pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: PipelineConfig instance. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
        self._model: Optional[KnotGraphNet] = None
        self._model_loaded = False

        # Validate deployment
        self._validate_deployment()

    def _validate_deployment(self):
        """Check that deployment directory has required files."""
        missing = []
        for name, path in [
            ("V5 weights", self.config.v5_weights),
            ("YOLO crossings", self.config.yolo_crossings_weights),
            ("YOLO endpoints", self.config.yolo_endpoints_weights),
        ]:
            if not path.exists():
                missing.append(f"  ❌ {name}: {path}")

        if missing:
            print(f"⚠️  Missing model files in {self.config.deployment_dir}:")
            for m in missing:
                print(m)
            print(
                "\nRun `python -m knotnet.setup_deployment` to copy all "
                "required files."
            )

    def _load_model(self) -> KnotGraphNet:
        """Load KnotGraphNet V5 model (lazy, once)."""
        if self._model is not None:
            return self._model

        model = KnotGraphNet(
            d=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_tokens=self.config.max_tokens,
            num_fourier=self.config.num_fourier,
            dropout=self.config.dropout,
            max_neighbor_dist=self.config.max_neighbor_dist,
            soft_dist_penalty=self.config.soft_dist_penalty,
        ).to(self.config.device)

        weights_path = self.config.v5_weights
        if weights_path.exists():
            state = torch.load(weights_path, map_location=self.config.device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

        model.eval()
        self._model = model
        self._model_loaded = True
        return model

    def run(
        self,
        image_path: str | Path,
        gt_locations: Optional[dict] = None,
        return_intermediates: bool = False,
    ) -> dict:
        """
        Run full pipeline on a single image.

        Args:
            image_path: Path to input image (.jpg/.png)
            gt_locations: Optional ground-truth locations dict.
                If provided, skips YOLO detection and uses GT instead.
                Format:
                    {
                        "crossings": [
                            {"crossing_idx": 0,
                             "box_x": ..., "box_y": ...,
                             "box_w": ..., "box_h": ...,
                             "over_1_x": ..., "over_1_y": ...,
                             "over_2_x": ..., "over_2_y": ...,
                             "under_1_x": ..., "under_1_y": ...,
                             "under_2_x": ..., "under_2_y": ...},
                            ...
                        ],
                        "endpoints": [
                            {"x": ..., "y": ...},
                            {"x": ..., "y": ...},
                        ]
                    }
            return_intermediates: If True, includes intermediate data
                (tokens, edge_logits, etc.) in result.

        Returns:
            dict with keys:
                mode: "inference" | "gold"
                image_path: str
                sequence: list of token labels
                crossings_order: list of crossing traversal order
                pd_code: {pd_crossings, pd_code_str, writhe}
                topology: {gauss_code, dt_notation, writhe,
                           n_crossings, jones_terms, jones_str}
                detections: {crossings, endpoints}
                timing: dict of stage durations
                (+ intermediates if requested)
        """
        image_path = Path(image_path)
        mode = "gold" if gt_locations is not None else "inference"
        timings = {}

        # ─── Stage 1: Preprocessing ──────────────────────────────────
        t0 = time.time()
        prep = preprocess_image(
            image_path,
            target_full_size=1024,
            target_model_size=self.config.img_size,
        )
        timings["preprocessing"] = time.time() - t0

        img_full = prep["img_full"]
        img_small = prep["img_small"]

        # ─── Stage 3+4: Detection ────────────────────────────────────
        t0 = time.time()
        if gt_locations is not None:
            crossings, endpoints = locations_from_gt(
                gt_locations,
                original_size=prep["original_size"],
                crop_offset=prep["crop_offset"],
                crop_side=prep["crop_side"],
                scale=prep["scale"],
            )
        else:
            crossings = detect_crossings(
                img_full,
                self.config.yolo_crossings_weights,
                conf=self.config.crossing_conf,
                iou=self.config.crossing_iou,
                cache_dir=self.config.cache_dir,
            )
            endpoints = detect_endpoints(
                img_full,
                self.config.yolo_endpoints_weights,
                conf=self.config.endpoint_conf,
                iou=self.config.endpoint_iou,
                cache_dir=self.config.cache_dir,
            )
        timings["detection"] = time.time() - t0

        # ─── Stage 2: Skeleton ────────────────────────────────────────
        t0 = time.time()
        cache_key = image_path.stem if self.config.use_skeleton_cache else None
        skel_full, skel_model = extract_skeleton(
            img_full, crossings, endpoints, self.config, cache_key=cache_key
        )
        timings["skeleton"] = time.time() - t0

        # ─── Stage 5: Tokenization ───────────────────────────────────
        t0 = time.time()
        tokens = build_tokens(
            crossings,
            endpoints,
            img_size=self.config.img_size,
            max_tokens=self.config.max_tokens,
        )
        timings["tokenization"] = time.time() - t0

        # ─── Stage 6: Sequence Prediction ─────────────────────────────
        t0 = time.time()
        model = self._load_model()
        seq_result = predict_sequence(
            tokens, img_small, skel_model, model, self.config
        )
        timings["sequence_prediction"] = time.time() - t0


        # ─── Stage 6.5: Knot Notation (pre-PD) ───────────────────────
        t0 = time.time()
        notation_result = compute_knot_notation(
            tokens, seq_result["sequence"], seq_result["crossings_order"]
        )
        timings["notation"] = time.time() - t0


        # ─── Stage 7+: Topology ───────────────────────────────────────
        t0 = time.time()
        pd_result = compute_pd_code(
            tokens, seq_result["sequence"], seq_result["crossings_order"]
        )
        topo_result = compute_topology(
            pd_result["pd_crossings"],
            tokens,
            seq_result["sequence"],
        )
        timings["topology"] = time.time() - t0

        # ─── Assemble output ──────────────────────────────────────────
        result = {
            "mode": mode,
            "image_path": str(image_path),
            "image_name": image_path.name,
            # Main outputs
            "sequence": seq_result["sequence"],
            "crossings_order": seq_result["crossings_order"],
            "knot_notation": {
                "full": notation_result["notation_str"],
                "compact": notation_result["compact_notation"],
                "sequence": notation_result["notation_sequence"],
                "crossing_visits": notation_result["crossing_visits"],
                "detailed": notation_result["notation_detailed"],
            },
            "pd_code": {
                "crossings": [
                    {"cid": c["cid"], "pd": c["pd"], "sign": c["sign"]}
                    for c in pd_result["pd_crossings"]
                ],
                "pd_code_str": pd_result["pd_code_str"],
                "writhe": pd_result["writhe"],
            },
            "topology": {
                "gauss_code": topo_result["gauss_code"],
                "dt_notation": topo_result["dt_notation"],
                "writhe": topo_result["writhe"],
                "n_crossings": topo_result["n_crossings"],
                "jones_terms": topo_result["jones_terms"],
                "jones_str": topo_result["jones_str"],
            },
            # Detections (for inspection/debugging)
            "detections": {
                "crossings": crossings,
                "endpoints": endpoints,
            },
            "timing": timings,
            "timing_total": sum(timings.values()),
        }

        if return_intermediates:
            result["intermediates"] = {
                "tokens": tokens,
                "edge_logits": (
                    seq_result["edge_logits"].tolist()
                    if seq_result["edge_logits"] is not None
                    else None
                ),
                "path_indices": seq_result["path_indices"],
                "skeleton_pixels": int(skel_full.sum()),
            }
        
        # ─── Optional: Visualisierung speichern ──────────────────────
        if self.config.save_visualizations:
            from .utils.visualization import visualize_result

            viz_path = self.config.outputs_dir / \
                f"result_{image_path.stem}.png"
            visualize_result(
                result=result,
                img_full=img_full,
                img_small=img_small,
                tokens=tokens,
                output_path=viz_path,
            )
            result["visualization_path"] = str(viz_path)

        return result

    def run_batch(
        self,
        image_paths: list[str | Path],
        gt_locations_list: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Run pipeline on multiple images.

        Args:
            image_paths: list of image paths
            gt_locations_list: optional list of GT dicts (same length)

        Returns:
            list of result dicts
        """
        results = []
        for i, img_path in enumerate(image_paths):
            gt = gt_locations_list[i] if gt_locations_list else None
            try:
                result = self.run(img_path, gt_locations=gt)
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": str(img_path),
                    "error": str(e),
                })
        return results
    
    def visualize(
        self,
        result: dict,
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Optional[Path]:
        """
        Erzeuge Visualisierung für ein bestehendes Result.

        Args:
            result: Output von self.run()
            image_path: Pfad zum Original-Bild
            output_path: Speicherpfad. None → auto in outputs_dir.

        Returns:
            Path zum gespeicherten Bild (oder None wenn inline).
        """
        from .utils.visualization import save_visualization

        if output_path is None:
            stem = Path(image_path).stem
            output_path = self.config.outputs_dir / f"result_{stem}.png"

        save_visualization(
            result=result,
            image_path=image_path,
            output_path=output_path,
            img_size=self.config.img_size,
        )
        return Path(output_path)