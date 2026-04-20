"""
Confidence Threshold Tuner.

Sweeps binarization thresholds on probability maps to find the value
that maximizes F1-Score (or any target metric) against ground truth.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .metrics import pixel_precision, pixel_recall, pixel_f1, pixel_iou


@dataclass
class ThresholdResult:
    """Result of a threshold sweep."""

    optimal_threshold: float
    optimal_f1: float
    optimal_precision: float
    optimal_recall: float
    optimal_iou: float
    thresholds: List[float] = field(repr=False)
    f1_scores: List[float] = field(repr=False)
    precisions: List[float] = field(repr=False)
    recalls: List[float] = field(repr=False)
    ious: List[float] = field(repr=False)

    def summary(self) -> str:
        return (
            f"Optimal Threshold: {self.optimal_threshold:.3f}\n"
            f"  F1-Score:  {self.optimal_f1:.4f}\n"
            f"  Precision: {self.optimal_precision:.4f}\n"
            f"  Recall:    {self.optimal_recall:.4f}\n"
            f"  IoU:       {self.optimal_iou:.4f}"
        )


class ThresholdTuner:
    """
    Finds the optimal binarization threshold that maximizes F1-Score.

    Usage:
        tuner = ThresholdTuner()
        result = tuner.tune(prob_maps, gt_masks)
        print(result.summary())
        print(f"Use threshold = {result.optimal_threshold}")
    """

    def __init__(
        self,
        thresholds: Optional[np.ndarray] = None,
        target_metric: str = "f1",
    ):
        """
        Args:
            thresholds: Array of thresholds to sweep. Default: 0.05 to 0.95
                        in steps of 0.05.
            target_metric: Which metric to optimize ('f1', 'precision',
                           'recall', 'iou').
        """
        if thresholds is None:
            self.thresholds = np.arange(0.05, 1.0, 0.05)
        else:
            self.thresholds = np.asarray(thresholds)

        self.target_metric = target_metric

    def tune(
        self,
        prob_maps: List[np.ndarray],
        gt_masks: List[np.ndarray],
    ) -> ThresholdResult:
        """
        Sweep thresholds and find the one that maximizes the target metric.

        Args:
            prob_maps: List of probability maps (H, W) in [0.0, 1.0].
            gt_masks:  List of binary ground truth masks (H, W) in {0, 1}.

        Returns:
            ThresholdResult with optimal threshold and full sweep data.
        """
        assert len(prob_maps) == len(gt_masks), (
            f"Mismatch: {len(prob_maps)} probability maps vs "
            f"{len(gt_masks)} ground truths"
        )

        all_f1 = []
        all_precision = []
        all_recall = []
        all_iou = []

        for t in self.thresholds:
            # Binarize all probability maps at this threshold
            batch_f1 = []
            batch_p = []
            batch_r = []
            batch_iou = []

            for prob, gt in zip(prob_maps, gt_masks):
                pred = (prob >= t).astype(np.uint8)
                batch_f1.append(pixel_f1(pred, gt))
                batch_p.append(pixel_precision(pred, gt))
                batch_r.append(pixel_recall(pred, gt))
                batch_iou.append(pixel_iou(pred, gt))

            all_f1.append(np.mean(batch_f1))
            all_precision.append(np.mean(batch_p))
            all_recall.append(np.mean(batch_r))
            all_iou.append(np.mean(batch_iou))

        # Find optimal threshold
        metric_map = {
            "f1": all_f1,
            "precision": all_precision,
            "recall": all_recall,
            "iou": all_iou,
        }
        target_scores = metric_map[self.target_metric]
        best_idx = int(np.argmax(target_scores))

        return ThresholdResult(
            optimal_threshold=float(self.thresholds[best_idx]),
            optimal_f1=all_f1[best_idx],
            optimal_precision=all_precision[best_idx],
            optimal_recall=all_recall[best_idx],
            optimal_iou=all_iou[best_idx],
            thresholds=[float(t) for t in self.thresholds],
            f1_scores=all_f1,
            precisions=all_precision,
            recalls=all_recall,
            ious=all_iou,
        )

    def tune_single(
        self,
        prob_map: np.ndarray,
        gt_mask: np.ndarray,
    ) -> ThresholdResult:
        """
        Convenience method for tuning on a single sample.
        """
        return self.tune([prob_map], [gt_mask])
