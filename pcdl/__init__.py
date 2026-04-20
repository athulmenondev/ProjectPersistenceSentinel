"""
PSCDL - Persistence Scene Change Detection Library

A unified pipeline for detecting persistent encroachment events in video streams.
Designed for the PSCDL 2026 Challenge.

Modules:
    pcdl.io - Video preprocessing and data loading (Athul)
    pcdl.spatial - Neural network segmentation models (Lekshmi)
    pcdl.temporal - Temporal persistence filtering (Jyotsna)
    pcdl.postprocess - Mask refinement and evaluation (Goutham)
"""

from pcdl.io import VideoPreprocessor
from pcdl.spatial import UNetModel, predict_mask, get_binary_mask
from pcdl.temporal import TemporalFilter
from pcdl.postprocess import (
    MaskRefiner, 
    pixel_f1, 
    pixel_iou, 
    pixel_precision, 
    pixel_recall,
    evaluate_directory,
    validate_submission
)
from pcdl.pipeline import PersistenceSentinelPipeline

__version__ = "1.0.0"
__all__ = [
    "VideoPreprocessor",
    "UNetModel",
    "predict_mask",
    "get_binary_mask",
    "TemporalFilter",
    "MaskRefiner",
    "pixel_f1",
    "pixel_iou",
    "pixel_precision",
    "pixel_recall",
    "evaluate_directory",
    "validate_submission",
    "PersistenceSentinelPipeline",
]
