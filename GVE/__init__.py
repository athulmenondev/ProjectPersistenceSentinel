"""
Module 4: Semantic Refinement & Metrics Dashboard
Owner: Goutham

Post-processing, evaluation, and submission layer for the PSCDL 2026 pipeline.
"""

from .metrics import (
    pixel_precision,
    pixel_recall,
    pixel_f1,
    pixel_iou,
    confusion_matrix,
    evaluate_batch,
    evaluate_directory,
)
from .mask_refiner import MaskRefiner
from .threshold_tuner import ThresholdTuner
from .submission_exporter import export_mask, export_batch, validate_submission
from .dashboard import DashboardGenerator
from .mock_generator import MockGenerator

__all__ = [
    "pixel_precision",
    "pixel_recall",
    "pixel_f1",
    "pixel_iou",
    "confusion_matrix",
    "evaluate_batch",
    "evaluate_directory",
    "MaskRefiner",
    "ThresholdTuner",
    "export_mask",
    "export_batch",
    "validate_submission",
    "DashboardGenerator",
    "MockGenerator",
]
