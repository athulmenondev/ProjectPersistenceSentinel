"""
Mask Refiner: Multi-Stage Post-Processing Pipeline.

Implements the "Award-Winning Twist" — Morphological SNR Optimization
plus CRF (Conditional Random Field) edge refinement to snap predicted
mask edges to actual visual object boundaries.

Pipeline stages:
  1. Morphological Cleanup (erosion, dilation, opening, closing)
  2. Connected Component Analysis (area & aspect-ratio filtering)
  3. CRF Edge Refinement (DenseCRF or GrabCut fallback)
  4. Median Filter Smoothing
"""

import numpy as np
import cv2
from typing import Optional, Tuple

# Optional CRF import
try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_softmax,
        create_pairwise_bilateral,
        create_pairwise_gaussian,
    )
    HAS_CRF = True
except ImportError:
    HAS_CRF = False


class MaskRefiner:
    """
    Multi-stage mask refinement pipeline.

    Usage:
        refiner = MaskRefiner(morph_kernel_size=5, min_area=500, use_crf=True)
        clean_mask = refiner.refine(probability_map, rgb_frame=original_image)
    """

    def __init__(
        self,
        morph_kernel_size: int = 5,
        min_area: int = 500,
        max_aspect_ratio: float = 10.0,
        use_crf: bool = True,
        crf_iterations: int = 5,
        crf_gaussian_sxy: int = 3,
        crf_bilateral_sxy: int = 50,
        crf_bilateral_srgb: int = 13,
        median_ksize: int = 5,
        threshold: float = 0.5,
    ):
        """
        Args:
            morph_kernel_size: Kernel size for morphological operations.
            min_area: Minimum connected component area (pixels) to keep.
            max_aspect_ratio: Maximum bounding-box aspect ratio to keep.
            use_crf: Whether to use CRF refinement (requires pydensecrf OR
                     falls back to GrabCut).
            crf_iterations: Number of CRF inference iterations.
            crf_gaussian_sxy: CRF Gaussian pairwise spatial sigma.
            crf_bilateral_sxy: CRF bilateral pairwise spatial sigma.
            crf_bilateral_srgb: CRF bilateral pairwise color sigma.
            median_ksize: Kernel size for final median filter smoothing.
            threshold: Probability threshold for initial binarization.
        """
        self.morph_kernel_size = morph_kernel_size
        self.min_area = min_area
        self.max_aspect_ratio = max_aspect_ratio
        self.use_crf = use_crf
        self.crf_iterations = crf_iterations
        self.crf_gaussian_sxy = crf_gaussian_sxy
        self.crf_bilateral_sxy = crf_bilateral_sxy
        self.crf_bilateral_srgb = crf_bilateral_srgb
        self.median_ksize = median_ksize
        self.threshold = threshold

    # ------------------------------------------------------------------ #
    #  Full Pipeline
    # ------------------------------------------------------------------ #

    def refine(
        self,
        prob_map: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Full refinement pipeline:
          prob_map → binarize → morphology → CCA → CRF → smooth.

        Args:
            prob_map: Probability map (H, W) with values in [0.0, 1.0].
            rgb_frame: Original RGB frame (H, W, 3) for CRF edge snapping.
                       If None, CRF stage is skipped.
            threshold: Override binarization threshold (uses self.threshold
                       if None).

        Returns:
            Binary mask (H, W) with values {0, 1}, dtype uint8.
        """
        t = threshold if threshold is not None else self.threshold

        # Stage 0: Binarize the probability map
        binary = (prob_map >= t).astype(np.uint8)

        # Stage 1: Morphological cleanup
        binary = self.morphological_cleanup(binary)

        # Stage 2: Connected component filtering
        binary = self.filter_components(binary)

        # Stage 3: CRF edge refinement (if enabled and RGB available)
        if self.use_crf and rgb_frame is not None:
            binary = self.crf_refine(prob_map, rgb_frame)
            # Re-apply morphological cleanup after CRF (CRF can re-introduce noise)
            binary = self.morphological_cleanup(binary)
            binary = self.filter_components(binary)

        # Stage 4: Median filter smoothing
        if self.median_ksize > 1:
            binary = cv2.medianBlur(binary, self.median_ksize)

        return binary

    # ------------------------------------------------------------------ #
    #  Stage 1: Morphological Cleanup
    # ------------------------------------------------------------------ #

    def morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to remove noise and fill holes.

        Operations applied in order:
          1. Opening (erosion → dilation): removes small false-positive blobs
          2. Closing (dilation → erosion): fills small holes within detections

        Args:
            binary_mask: Binary mask (H, W) with values {0, 1}, dtype uint8.

        Returns:
            Cleaned binary mask (H, W), dtype uint8.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )

        # Opening: remove small noise blobs
        result = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Closing: fill small holes
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)

        return result

    # ------------------------------------------------------------------ #
    #  Stage 2: Connected Component Analysis
    # ------------------------------------------------------------------ #

    def filter_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Filter connected components by area and aspect ratio.

        Removes blobs that are too small (likely noise) or have extreme
        aspect ratios (unlikely to be real encroachments).

        Args:
            binary_mask: Binary mask (H, W), dtype uint8.

        Returns:
            Filtered binary mask (H, W), dtype uint8.
        """
        # Ensure mask is proper uint8 for connectedComponentsWithStats
        mask_255 = (binary_mask * 255).astype(np.uint8)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_255, connectivity=8
        )

        result = np.zeros_like(binary_mask, dtype=np.uint8)

        for label_id in range(1, n_labels):  # Skip background (label 0)
            area = stats[label_id, cv2.CC_STAT_AREA]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]

            # Area filter
            if area < self.min_area:
                continue

            # Aspect ratio filter
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect_ratio:
                continue

            # Keep this component
            result[labels == label_id] = 1

        return result

    # ------------------------------------------------------------------ #
    #  Stage 3: CRF Edge Refinement
    # ------------------------------------------------------------------ #

    def crf_refine(
        self, prob_map: np.ndarray, rgb_frame: np.ndarray
    ) -> np.ndarray:
        """
        Use Dense CRF to snap mask edges to visual object boundaries.

        Falls back to bilateral-filter-based refinement if pydensecrf
        is not installed.

        Args:
            prob_map: Probability map (H, W) in [0.0, 1.0].
            rgb_frame: Original RGB image (H, W, 3), uint8.

        Returns:
            Refined binary mask (H, W), dtype uint8.
        """
        if HAS_CRF:
            return self._densecrf_refine(prob_map, rgb_frame)
        else:
            return self._bilateral_fallback(prob_map, rgb_frame)

    def _densecrf_refine(
        self, prob_map: np.ndarray, rgb_frame: np.ndarray
    ) -> np.ndarray:
        """DenseCRF-based refinement (requires pydensecrf)."""
        h, w = prob_map.shape

        # Build 2-class softmax: [background_prob, foreground_prob]
        fg_prob = np.clip(prob_map, 1e-6, 1.0 - 1e-6).astype(np.float32)
        bg_prob = 1.0 - fg_prob
        softmax = np.stack([bg_prob, fg_prob], axis=0)  # (2, H, W)

        # Create unary potentials
        unary = unary_from_softmax(softmax)

        # Initialize DenseCRF
        d = dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(unary)

        # Pairwise Gaussian (encourages spatial smoothness)
        d.addPairwiseGaussian(
            sxy=self.crf_gaussian_sxy,
            compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        # Pairwise Bilateral (encourages color-consistent boundaries)
        img_c = np.ascontiguousarray(rgb_frame)
        d.addPairwiseBilateral(
            sxy=self.crf_bilateral_sxy,
            srgb=self.crf_bilateral_srgb,
            rgbim=img_c,
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        # Inference
        q = d.inference(self.crf_iterations)
        map_result = np.argmax(np.array(q).reshape(2, h, w), axis=0)

        return map_result.astype(np.uint8)

    def _bilateral_fallback(
        self, prob_map: np.ndarray, rgb_frame: np.ndarray
    ) -> np.ndarray:
        """
        Fallback when pydensecrf is not available.

        Uses bilateral filtering on the probability map guided by the
        RGB frame's edges, then applies adaptive thresholding.
        """
        h, w = prob_map.shape

        # Convert prob_map to uint8 for bilateral filter
        prob_uint8 = (prob_map * 255).astype(np.uint8)

        # Apply joint bilateral filter (uses RGB image to guide smoothing)
        # OpenCV's bilateralFilter only uses the input image for guidance,
        # so we use a weighted approach:
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        # Edge-aware smoothing of the probability map
        filtered = cv2.bilateralFilter(prob_uint8, d=9, sigmaColor=75, sigmaSpace=75)

        # Detect edges in the RGB frame
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        # Near edges, prefer sharper boundaries
        # Away from edges, smooth aggressively
        smooth = cv2.GaussianBlur(prob_uint8, (11, 11), 3)
        result = np.where(edges_dilated > 0, filtered, smooth)

        # Binarize
        binary = (result > int(self.threshold * 255)).astype(np.uint8)

        return binary

    # ------------------------------------------------------------------ #
    #  Utility: Get Pipeline Stage Names
    # ------------------------------------------------------------------ #

    def get_pipeline_description(self) -> str:
        """Return a human-readable description of the active pipeline."""
        stages = [
            f"1. Binarize at threshold={self.threshold}",
            f"2. Morphological cleanup (kernel={self.morph_kernel_size})",
            f"3. CCA filtering (min_area={self.min_area}, max_ar={self.max_aspect_ratio})",
        ]
        if self.use_crf:
            backend = "DenseCRF" if HAS_CRF else "Bilateral Fallback"
            stages.append(f"4. CRF refinement ({backend})")
        stages.append(f"5. Median smoothing (k={self.median_ksize})")
        return "\n".join(stages)
