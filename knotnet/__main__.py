"""
Ermöglicht: python -m knotnet --image path/to/image.jpg [--gt-locations ...]
"""

import argparse
import json
from pathlib import Path

from .pipeline import KnotNetPipeline
from .config import PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        description="KnotNet — End-to-End Knot Inference Pipeline"
    )
    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="Path to input image (.jpg/.png)"
    )
    parser.add_argument(
        "--deployment-dir", "-d", type=str, default=None,
        help="Path to deployment directory with model weights. "
             "Defaults to ./knotnet_deployment/"
    )
    parser.add_argument(
        "--gt-locations", "-g", type=str, default=None,
        help="JSON file or inline JSON string with ground-truth locations. "
             "Format: {\"crossings\": [...], \"endpoints\": [...]}"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output JSON path. Defaults to <image_stem>_result.json"
    )
    parser.add_argument(
        "--no-skeleton-cache", action="store_true",
        help="Force skeleton recomputation (ignore cache)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'mps', 'cpu'. Auto-detected if omitted."
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Generate visualization image alongside JSON output"
    )
    args = parser.parse_args()

    # Config
    config = PipelineConfig()
    if args.deployment_dir:
        config.deployment_dir = Path(args.deployment_dir)
    if args.device:
        config.device_override = args.device
    if args.no_skeleton_cache:
        config.use_skeleton_cache = False

    # GT Locations
    gt_locations = None
    if args.gt_locations:
        gt_path = Path(args.gt_locations)
        if gt_path.exists():
            with open(gt_path) as f:
                gt_locations = json.load(f)
        else:
            gt_locations = json.loads(args.gt_locations)

    # Run
    pipe = KnotNetPipeline(config=config)
    result = pipe.run(args.image, gt_locations=gt_locations)

    # Visualization
    if args.visualize:
        viz_path = output_path.with_suffix(".png")
        pipe.visualize(result, args.image, output_path=viz_path)
        print(f"  🖼️  Viz:    {viz_path}")

    # Output
    output_path = Path(args.output) if args.output else \
        Path(args.image).with_name(Path(args.image).stem + "_result.json")
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n{'═'*60}")
    print(f"Ergebnis: {result.get('knot_type', 'unknown')}")
    print(f"PD-Code:  {result.get('pd_code_str', 'N/A')}")
    print(f"Output:   {output_path}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()