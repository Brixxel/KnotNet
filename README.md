# KnotNet — End-to-End Knot Recognition Pipeline

<p align="center">
  <em>From photograph to knot type — fully automated.</em>
</p>

---

## Overview

**KnotNet** is a modular Python pipeline that takes a single photograph of a knotted rope and outputs:

- **Traversal Sequence** — the order in which crossings are visited along the rope
- **Knot Notation** — compact human-readable representation (`c0(O) → c1(U) → c2(O) → ...`)
- **PD-Code** — Planar Diagram code for rigorous topological analysis
- **Jones Polynomial** — computed via Kauffman bracket state-sum
- **Knot Type Classification** — matched against a lookup table of known knots

The pipeline consists of the following stages:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  1. Image   │───▶│ 2. Skeleton  │───▶│ 3. Detection  │───▶│ 4. Tokens    │
│  Preprocess │    │  (SAM2)      │    │  (YOLO)       │    │  Construct   │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
                                                                     │
                                                                     ▼
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌──────────────┐
│  8. Knot    │◀───│ 7. Jones     │◀───│ 6. PD-Code    │◀───│ 5. Sequence  │
│  Classify   │    │  Polynomial  │    │  + Topology   │    │  (KnotNet)   │
└─────────────┘    └──────────────┘    └───────────────┘    └──────────────┘
```

## Architecture

### Detection (Stages 2–4)

| Component | Model | Purpose |
|-----------|-------|---------|
| Segmentation | SAM2 (small) | Rope mask extraction from point prompts |
| Crossing Detection | YOLOv26x-pose | Bounding boxes + 4 keypoints per crossing |
| Endpoint Detection | YOLOv8s | Rope start/end localization |

### Sequence Prediction (Stage 5)

**KnotGraphNet V5** — a custom Transformer-based architecture:

- **Image Encoder**: 4-layer CNN (`RGB + Skeleton → D-dim feature map`)
- **Token Embedding**: Fourier-encoded positions + type/pair/partner embeddings
- **Cross-Attention**: Tokens attend to image features
- **Self-Attention**: Token-to-token reasoning (2 layers, 2 heads)
- **Edge Prediction**: Bilinear scoring `Q·K^T` with skeleton bias and distance penalties
- **Decoding**: Greedy path-walk on symmetric graph (implicit + learned edges)

### Topology (Stages 6–8)

PD-Code, Gauss code, DT notation, writhe, and Jones polynomial are computed purely algorithmically from the predicted traversal — no neural networks involved.

## Installation

```bash
# Clone and install
git clone <repo-url>
cd knotnet
pip install -e .

# Dependencies
pip install torch torchvision ultralytics
pip install numpy scipy scikit-image opencv-python pillow matplotlib
pip install pandas sympy

# Optional (for SAM2 segmentation):
pip install knotcv  # or install from local path
```

## Quick Start

### Setup (one-time)

Copy model weights into a self-contained deployment directory:

```bash
python -m knotnet.setup_deployment \
    --source-root /path/to/knot-cv \
    --target-dir ./knotnet_deployment
```

This creates:

```
knotnet_deployment/
├── weights/
│   ├── knotgraphnet_v5_best.pt     # Sequence prediction model
│   ├── yolo_crossings_best.pt      # Crossing detector
│   └── yolo_endpoints_best.pt      # Endpoint detector
├── cache/                           # Skeleton cache
├── outputs/                         # Pipeline outputs
└── config/
    └── deployment_info.json
```

### Python API

```python
from knotnet import KnotNetPipeline, PipelineConfig

# Initialize
config = PipelineConfig(deployment_dir="./knotnet_deployment")
pipe = KnotNetPipeline(config=config)

# Run on an image (full inference)
result = pipe.run("my_knot_photo.jpg")

# Access results
print(result["knot_notation"]["compact"])
# → c0(O) → c1(U) → c2(O) → c0(U) → c1(O)

print(result["pd_code"]["pd_code_str"])
# → X[1,4,2,5]  X[3,6,4,1]  X[5,2,6,3]

print(result["topology"]["jones_str"])
# → -A^4 +A^3 +A^1

print(result["topology"]["writhe"])
# → -3
```

### Ground-Truth Mode

If you have annotated crossing/endpoint locations (e.g. for evaluation), you can bypass detection and only run the sequence prediction + topology stages:

```python
gt_locations = {
    "crossings": [
        {
            "crossing_idx": 0,
            "box_x": 412, "box_y": 305, "box_w": 95, "box_h": 88,
            "over_1_x": 420, "over_1_y": 320,
            "over_2_x": 495, "over_2_y": 340,
            "under_1_x": 445, "under_1_y": 298,
            "under_2_x": 460, "under_2_y": 385,
        },
        # ... more crossings
    ],
    "endpoints": [
        {"x": 120, "y": 500},
        {"x": 880, "y": 150},
    ],
}

result = pipe.run("my_knot.jpg", gt_locations=gt_locations)
print(result["mode"])  # → "gold"
```

### Command Line

```bash
# Full inference
python -m knotnet --image photo.jpg --deployment-dir ./knotnet_deployment

# With visualization output
python -m knotnet -i photo.jpg -d ./knotnet_deployment --visualize

# With ground-truth locations
python -m knotnet -i photo.jpg --gt-locations annotations.json

# Output is saved as <image_stem>_result.json
```

### Visualization

```python
# Generate comparison visualization
viz_path = pipe.visualize(result, "my_knot.jpg")

# Or enable auto-save in config:
config = PipelineConfig(
    deployment_dir="./knotnet_deployment",
    save_visualizations=True,
)
```

## Output Format

The pipeline returns a JSON-serializable dictionary:

```json
{
  "mode": "inference",
  "image_name": "trefoil.jpg",
  
  "sequence": ["a", "b1", "b2", "c3", "c4", "d1", "d2", "b3", "b4", "z"],
  
  "knot_notation": {
    "compact": "c0(O) → c1(U) → c2(O) → c0(U) → c1(O)",
    "full": "a → c0.over.in → c0.over.out → c1.under.in → ...",
    "crossing_visits": [
      {"cid": 0, "strand_type": "over", "visit_number": 1},
      {"cid": 1, "strand_type": "under", "visit_number": 1}
    ]
  },
  
  "pd_code": {
    "crossings": [
      {"cid": 0, "pd": [1, 4, 2, 5], "sign": "-"},
      {"cid": 1, "pd": [3, 6, 4, 1], "sign": "-"},
      {"cid": 2, "pd": [5, 2, 6, 3], "sign": "-"}
    ],
    "pd_code_str": "X[1,4,2,5]  X[3,6,4,1]  X[5,2,6,3]",
    "writhe": -3
  },
  
  "topology": {
    "gauss_code": "O1- U2- O2- U3- O3- U1-",
    "dt_notation": [-4, -6, -2],
    "writhe": -3,
    "n_crossings": 3,
    "jones_terms": {"4": -1, "3": 1, "1": 1},
    "jones_str": "-A^4 +A^3 +A^1"
  },
  
  "detections": {
    "crossings": [...],
    "endpoints": [...]
  },
  
  "timing": {
    "preprocessing": 0.05,
    "detection": 1.2,
    "skeleton": 0.8,
    "tokenization": 0.001,
    "sequence_prediction": 0.15,
    "topology": 0.3
  },
  "timing_total": 2.5
}
```

## Configuration

All parameters are controlled via `PipelineConfig`:

```python
from knotnet import PipelineConfig

config = PipelineConfig(
    # Paths
    deployment_dir="./knotnet_deployment",
    
    # Model architecture (must match training!)
    img_size=224,
    max_tokens=40,
    num_fourier=8,
    d_model=64,
    num_heads=2,
    num_layers=2,
    dropout=0.3,
    
    # Detection thresholds
    crossing_conf=0.3,
    crossing_iou=0.4,
    endpoint_conf=0.25,
    endpoint_iou=0.5,
    
    # Skeleton
    skel_threshold=0.3,
    skeleton_target_long_side=1536,
    use_skeleton_cache=True,
    
    # Inference
    max_neighbor_dist=0.85,
    soft_dist_penalty=-8.0,
    
    # Device (auto-detected if None)
    device_override=None,  # "cuda", "mps", "cpu"
    
    # Output
    save_visualizations=False,
)
```

## Module Structure

```
knotnet/
├── __init__.py              # Public API: KnotNetPipeline, PipelineConfig
├── __main__.py              # CLI entry point (python -m knotnet)
├── pipeline.py              # Main orchestrator class
├── config.py                # Configuration dataclass
├── setup_deployment.py      # Model copying script
├── models/
│   ├── __init__.py
│   ├── knotgraphnet.py      # KnotGraphNet V5 architecture
│   └── image_encoder.py     # CNN encoder
├── stages/
│   ├── __init__.py
│   ├── preprocessing.py     # Image normalization
│   ├── skeleton.py          # SAM2-based skeleton extraction
│   ├── detection.py         # YOLO crossing/endpoint detection
│   ├── tokenization.py      # Token construction
│   ├── sequence.py          # Graph-based sequence prediction
│   └── topology.py          # PD-code, Jones polynomial, etc.
└── utils/
    ├── __init__.py
    ├── skeleton_ops.py      # Morphological operations
    └── visualization.py     # Result plotting
```

## How It Works

### Token Representation

Each point of interest on the rope becomes a **token**:

| Token Type | Pair ID | Meaning |
|------------|---------|---------|
| Endpoint | 4 | Rope start (`a`) |
| Crossing KP | 0 | Over-strand entry (`over_1`) |
| Crossing KP | 1 | Over-strand exit (`over_2`) |
| Crossing KP | 2 | Under-strand entry (`under_1`) |
| Crossing KP | 3 | Under-strand exit (`under_2`) |
| Endpoint | 5 | Rope end (`z`) |

### Graph Decoding

The model predicts **edge logits** (an N×N score matrix) indicating which tokens should be adjacent in the traversal. The decoder builds a path using:

1. **Implicit edges**: Within each crossing, `over_1↔over_2` and `under_1↔under_2` are always connected (same strand passes through).
2. **Learned edges**: The model predicts which tokens from *different* crossings should be connected. A greedy symmetric matching ensures each token gets the correct degree (endpoints: 1, crossing KPs: 2).
3. **Path walk**: Starting from endpoint `a`, walk the graph until reaching `z`.

### PD-Code Convention

```
X[i, j, k, l]
  i = incoming under-strand arc
  j = outgoing over-strand arc
  k = outgoing under-strand arc
  l = incoming over-strand arc
```

The knot is treated as closed (connecting `z` back to `a`), and arc numbers are assigned along the traversal path.

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- ultralytics (YOLO)
- numpy, scipy, scikit-image, opencv-python
- matplotlib, Pillow
- sympy (for Jones polynomial computation)
- pandas (for data loading in evaluation)

Optional:
- `knotcv` package (for SAM2-based segmentation; falls back to Otsu if unavailable)
- CUDA or MPS for GPU acceleration

## Citation

If you use KnotNet in your work, please cite:

```bibtex
@software{knotnet2025,
  title={KnotNet: End-to-End Knot Recognition from Photographs},
  author={Regler et al.},
  year={2026},
  url={https://github.com/Brixxel/knotnet}
}
```

## License

MIT License — see `LICENSE` file.