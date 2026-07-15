"""
Zentrale Konfiguration für die KnotNet Pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import torch


@dataclass
class PipelineConfig:
    """Alle konfigurierbaren Parameter der Pipeline."""

    # ─── Deployment Paths ─────────────────────────────────────────────
    deployment_dir: Path = Path("./knotnet_deployment")

    # Relative Pfade innerhalb deployment_dir (werden automatisch aufgelöst)
    v5_weights_rel: str = "weights/knotgraphnet_v5_best.pt"
    yolo_crossings_weights_rel: str = "weights/yolo_crossings_best.pt"
    yolo_endpoints_weights_rel: str = "weights/yolo_endpoints_best.pt"

    # ─── Model Architecture Params ───────────────────────────────────
    img_size: int = 224
    max_tokens: int = 40
    num_fourier: int = 8
    d_model: int = 64
    num_heads: int = 2
    num_layers: int = 2
    dropout: float = 0.3
    max_neighbor_dist: float = 0.85
    soft_dist_penalty: float = -8.0

    # ─── Detection Thresholds ─────────────────────────────────────────
    crossing_conf: float = 0.3
    crossing_iou: float = 0.4
    endpoint_conf: float = 0.25
    endpoint_iou: float = 0.5

    # ─── Skeleton ─────────────────────────────────────────────────────
    skel_threshold: float = 0.3
    skeleton_target_long_side: int = 1536
    use_skeleton_cache: bool = True
    skeleton_cache_dir: Optional[Path] = None  # None → deployment_dir/cache/

    # ─── Visualization ────────────────────────────────────────────────
    save_visualizations: bool = False
    viz_dpi: int = 120
    viz_figsize: tuple = (20, 14)

    # ─── Device ───────────────────────────────────────────────────────
    device_override: Optional[str] = None  # None → auto-detect

    # ─── Output ───────────────────────────────────────────────────────
    save_visualizations: bool = False
    output_dir: Optional[Path] = None  # None → deployment_dir/outputs/

    def __post_init__(self):
        self.deployment_dir = Path(self.deployment_dir)

    @property
    def device(self) -> torch.device:
        if self.device_override:
            return torch.device(self.device_override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @property
    def v5_weights(self) -> Path:
        return self.deployment_dir / self.v5_weights_rel

    @property
    def yolo_crossings_weights(self) -> Path:
        return self.deployment_dir / self.yolo_crossings_weights_rel

    @property
    def yolo_endpoints_weights(self) -> Path:
        return self.deployment_dir / self.yolo_endpoints_weights_rel

    @property
    def cache_dir(self) -> Path:
        d = self.skeleton_cache_dir or (self.deployment_dir / "cache")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def outputs_dir(self) -> Path:
        d = self.output_dir or (self.deployment_dir / "outputs")
        d.mkdir(parents=True, exist_ok=True)
        return d