"""
Pixel-Level Evaluation Metrics for PSCDL 2026.

Implements Precision, Recall, F1-Score, and IoU at the pixel level,
exactly matching the competition judging criteria.
"""

import numpy as np
import os
import cv2
from typing import Dict, List, Tuple, Optional


def confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> Dict[str, int]:
    """Compute pixel-level confusion matrix counts."""
    pred_bool = pred.astype(bool)
    gt_bool = gt.astype(bool)

    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    tn = int(np.logical_and(~pred_bool, ~gt_bool).sum())

    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def pixel_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level Precision = TP / (TP + FP)."""
    cm = confusion_matrix(pred, gt)
    tp, fp = cm["TP"], cm["FP"]

    if tp + fp == 0:
        return 1.0 if cm["FN"] == 0 else 0.0
    return tp / (tp + fp)


def pixel_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level Recall = TP / (TP + FN)."""
    cm = confusion_matrix(pred, gt)
    tp, fn = cm["TP"], cm["FN"]

    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)


def pixel_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level F1-Score."""
    p = pixel_precision(pred, gt)
    r = pixel_recall(pred, gt)

    if p + r == 0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def pixel_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Pixel-level Intersection over Union (Jaccard Index)."""
    cm = confusion_matrix(pred, gt)
    tp, fp, fn = cm["TP"], cm["FP"], cm["FN"]

    denominator = tp + fp + fn
    if denominator == 0:
        return 1.0
    return tp / denominator


def evaluate_single(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Compute all metrics for a single prediction-ground truth pair."""
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
    """Compute per-sample and aggregate metrics across a batch."""
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
    if n == 0:
        return {"per_sample": [], "aggregate": {}, "micro": {}}

    aggregate = {
        "precision": sum(r["precision"] for r in per_sample) / n,
        "recall": sum(r["recall"] for r in per_sample) / n,
        "f1": sum(r["f1"] for r in per_sample) / n,
        "iou": sum(r["iou"] for r in per_sample) / n,
    }

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


def evaluate_directory(
    pred_dir: str,
    gt_dir: str,
    extensions: Tuple[str, ...] = (".png", ".jpg", ".bmp"),
) -> Dict[str, object]:
    """Load predicted and ground truth masks from directories and evaluate."""
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
        base = os.path.splitext(fname)[0]
        gt_path = None
        for ext in extensions:
            candidate = os.path.join(gt_dir, base + ext)
            if os.path.exists(candidate):
                gt_path = candidate
                break

        if gt_path is None:
            continue

        mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            pred_masks.append((mask > 127).astype(np.uint8))
            
            mask_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if mask_gt is not None:
                gt_masks.append((mask_gt > 127).astype(np.uint8))
                matched_files.append(fname)

    if not pred_masks:
        raise ValueError("No matching prediction-ground truth pairs found.")

    result = evaluate_batch(pred_masks, gt_masks)
    result["filenames"] = matched_files
    return result
