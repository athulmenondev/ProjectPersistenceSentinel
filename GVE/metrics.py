"""
Pixel-Level Evaluation Metrics for PSCDL 2026.

Implements Precision, Recall, F1-Score, and IoU at the pixel level,
exactly matching the competition judging criteria.

All functions expect binary NumPy arrays of shape (H, W) with values {0, 1}.
"""

import numpy as np
import os
import cv2
from typing import Dict, List, Tuple, Optional


# ====================================================================== #
#  Core Metric Functions
# ====================================================================== #

def confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """
    Compute pixel-level confusion matrix counts.

    Args:
        pred: Predicted binary mask (H, W), values {0, 1}.
        gt:   Ground truth binary mask (H, W), values {0, 1}.

    Returns:
        Dictionary with keys 'TP', 'FP', 'FN', 'TN'.
    """
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def pixel_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Pixel-level Precision = TP / (TP + FP).

    Returns 1.0 if both pred and gt are all-zero (no positives predicted,
    none expected — perfect precision). Returns 0.0 if pred has positives
    but gt has none.
    """
    cm = confusion_matrix(pred, gt)
    tp, fp = cm["TP"], cm["FP"]

    if tp + fp == 0:
        # No positive predictions — precision is 1.0 if GT also has no positives
        return 1.0 if cm["FN"] == 0 else 0.0
    return tp / (tp + fp)


def pixel_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Pixel-level Recall = TP / (TP + FN).

    Returns 1.0 if gt has no positive pixels (nothing to recall).
    """
    cm = confusion_matrix(pred, gt)
    tp, fn = cm["TP"], cm["FN"]

    if tp + fn == 0:
        return 1.0  # Nothing to recall
    return tp / (tp + fn)


def pixel_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Pixel-level F1-Score = 2 · (Precision · Recall) / (Precision + Recall).

    Returns 1.0 if both pred and gt are all-zero.
    Returns 0.0 if one is all-zero and the other is not.
    """
    p = pixel_precision(pred, gt)
    r = pixel_recall(pred, gt)

    if p + r == 0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def pixel_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Pixel-level Intersection over Union = TP / (TP + FP + FN).

    Also called Jaccard Index. Returns 1.0 if both masks are empty.
    """
    cm = confusion_matrix(pred, gt)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]

    denominator = tp + fp + fn
    if denominator == 0:
        return 1.0  # Both masks empty — perfect overlap
    return tp / denominator


# ====================================================================== #
#  Batch Evaluation
# ====================================================================== #

def evaluate_single(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """
    Compute all metrics for a single prediction-ground truth pair.

    Returns:
        Dictionary with keys: 'precision', 'recall', 'f1', 'iou',
        'tp', 'fp', 'fn', 'tn'.
    """
    cm = confusion_matrix(pred, gt)
    p = pixel_precision(pred, gt)
    r = pixel_recall(pred, gt)
    f1 = 2.0 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    iou = pixel_iou(pred, gt)

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "iou": iou,
        "tp": cm["TP"],
        "fp": cm["FP"],
        "fn": cm["FN"],
        "tn": cm["TN"],
    }


def evaluate_batch(
    pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]
) -> Dict[str, object]:
    """
    Compute per-sample and aggregate metrics across a batch.

    Args:
        pred_masks: List of predicted binary masks.
        gt_masks:   List of ground truth binary masks.

    Returns:
        Dictionary with:
          - 'per_sample': list of per-sample metric dicts
          - 'aggregate': dict with mean metrics across all samples
          - 'micro': dict with micro-averaged metrics (sum TP/FP/FN first)
    """
    assert len(pred_masks) == len(gt_masks), (
        f"Mismatch: {len(pred_masks)} predictions vs {len(gt_masks)} ground truths"
    )

    per_sample = []
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0

    for pred, gt in zip(pred_masks, gt_masks):
        result = evaluate_single(pred, gt)
        per_sample.append(result)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]
        total_tn += result["tn"]

    n = len(per_sample)

    # Macro average (mean of per-sample metrics)
    aggregate = {
        "precision": sum(r["precision"] for r in per_sample) / n,
        "recall": sum(r["recall"] for r in per_sample) / n,
        "f1": sum(r["f1"] for r in per_sample) / n,
        "iou": sum(r["iou"] for r in per_sample) / n,
    }

    # Micro average (compute metrics from summed TP/FP/FN)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_p * micro_r / (micro_p + micro_r)
        if (micro_p + micro_r) > 0
        else 0.0
    )
    micro_iou = (
        total_tp / (total_tp + total_fp + total_fn)
        if (total_tp + total_fp + total_fn) > 0
        else 1.0
    )

    micro = {
        "precision": micro_p,
        "recall": micro_r,
        "f1": micro_f1,
        "iou": micro_iou,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_tn": total_tn,
    }

    return {"per_sample": per_sample, "aggregate": aggregate, "micro": micro}


# ====================================================================== #
#  Directory-Based Evaluation
# ====================================================================== #

def _load_mask(path: str) -> np.ndarray:
    """Load a mask image as binary (H, W) with values {0, 1}."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {path}")
    # Binarize: anything > 127 becomes 1
    return (mask > 127).astype(np.uint8)


def evaluate_directory(
    pred_dir: str,
    gt_dir: str,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".bmp"),
) -> Dict[str, object]:
    """
    Load predicted and ground truth masks from directories and evaluate.

    Matching is done by filename: for each file in pred_dir, finds the
    corresponding file in gt_dir with the same base name.

    Args:
        pred_dir: Directory containing predicted mask images.
        gt_dir:   Directory containing ground truth mask images.
        extensions: Tuple of allowed file extensions.

    Returns:
        Same structure as evaluate_batch, plus 'filenames' key.
    """
    pred_files = sorted(
        [
            f
            for f in os.listdir(pred_dir)
            if os.path.splitext(f)[1].lower() in extensions
        ]
    )

    if not pred_files:
        raise ValueError(f"No mask files found in {pred_dir}")

    pred_masks = []
    gt_masks = []
    matched_files = []

    for fname in pred_files:
        pred_path = os.path.join(pred_dir, fname)

        # Try to find matching ground truth file (same basename, any extension)
        base = os.path.splitext(fname)[0]
        gt_path = None
        for ext in extensions:
            candidate = os.path.join(gt_dir, base + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            print(f"Warning: No ground truth found for {fname}, skipping.")
            continue

        pred_masks.append(_load_mask(pred_path))
        gt_masks.append(_load_mask(gt_path))
        matched_files.append(fname)

    if not pred_masks:
        raise ValueError("No matching prediction-ground truth pairs found.")

    result = evaluate_batch(pred_masks, gt_masks)
    result["filenames"] = matched_files
    return result
