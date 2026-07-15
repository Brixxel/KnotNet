"""
KnotNet — End-to-End Knot Inference Pipeline
=============================================

Usage:
    from knotnet import KnotNetPipeline

    pipe = KnotNetPipeline("/path/to/deployment")
    result = pipe.run("image.jpg")               # full inference
    result = pipe.run("image.jpg", gt_locations=locations)  # GT mode
"""

from .pipeline import KnotNetPipeline
from .config import PipelineConfig

__version__ = "0.1.0"
__all__ = ["KnotNetPipeline", "PipelineConfig"]