"""
Submission Exporter for PSCDL 2026.

Converts refined binary masks into competition-format .png files
and validates the submission directory.
"""

import numpy as np
import cv2
import os
from typing import Callable, List, Optional, Tuple


def export_mask(
    mask: np.ndarray,
    output_path: str,
    target_size: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Save a single binary mask as a .png file.

    Args:
        mask: Binary mask (H, W) with values {0, 1}, dtype uint8.
        output_path: Full path to save the .png file.
        target_size: (width, height) to resize the mask before saving.
                     Use this to match original video resolution if needed.

    Returns:
        The output path.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Convert {0, 1} to {0, 255}
    mask_img = (mask * 255).astype(np.uint8)

    # Resize if target size specified
    if target_size is not None:
        mask_img = cv2.resize(
            mask_img,
            target_size,
            interpolation=cv2.INTER_NEAREST,  # Preserve binary values
        )

    # Save
    cv2.imwrite(output_path, mask_img)
    return output_path


def export_batch(
    masks: List[np.ndarray],
    output_dir: str,
    naming_fn: Optional[Callable[[int], str]] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> List[str]:
    """
    Batch export binary masks to a directory.

    Args:
        masks: List of binary masks.
        output_dir: Directory to save masks into.
        naming_fn: Function(index) -> filename (without extension).
                   Default: "mask_00001", "mask_00002", etc.
        target_size: Optional (width, height) to resize all masks.

    Returns:
        List of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    if naming_fn is None:
        naming_fn = lambda i: f"mask_{i + 1:05d}"

    paths = []
    for i, mask in enumerate(masks):
        fname = naming_fn(i) + ".png"
        path = os.path.join(output_dir, fname)
        export_mask(mask, path, target_size=target_size)
        paths.append(path)

    print(f"Exported {len(paths)} masks to {output_dir}")
    return paths


def validate_submission(
    submission_dir: str,
    expected_count: Optional[int] = None,
    expected_size: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Validate a submission directory to ensure all masks are valid.

    Checks:
      - Files exist and are loadable
      - Images are single-channel (grayscale)
      - Pixel values are strictly binary ({0, 255})
      - Dimensions match expected_size (if provided)
      - Count matches expected_count (if provided)

    Args:
        submission_dir: Path to submission directory.
        expected_count: Expected number of mask files.
        expected_size: Expected (width, height) of each mask.

    Returns:
        Dictionary with 'valid' (bool), 'errors' (list), 'warnings' (list),
        and 'file_count' (int).
    """
    errors = []
    warnings = []

    # Find all PNG files
    files = sorted(
        [f for f in os.listdir(submission_dir) if f.lower().endswith(".png")]
    )

    if not files:
        errors.append(f"No .png files found in {submission_dir}")
        return {"valid": False, "errors": errors, "warnings": warnings, "file_count": 0}

    # Check count
    if expected_count is not None and len(files) != expected_count:
        errors.append(
            f"Expected {expected_count} masks, found {len(files)}"
        )

    for fname in files:
        path = os.path.join(submission_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            errors.append(f"{fname}: Could not load image")
            continue

        # Check single channel
        if len(img.shape) == 3:
            warnings.append(f"{fname}: Image has {img.shape[2]} channels (expected 1)")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Check binary values
        unique_vals = set(np.unique(img))
        if not unique_vals.issubset({0, 255}):
            non_binary = unique_vals - {0, 255}
            errors.append(
                f"{fname}: Non-binary pixel values found: {non_binary}"
            )

        # Check dimensions
        if expected_size is not None:
            actual_size = (img.shape[1], img.shape[0])  # (W, H)
            if actual_size != expected_size:
                errors.append(
                    f"{fname}: Size {actual_size} != expected {expected_size}"
                )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "file_count": len(files),
    }
