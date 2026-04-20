"""
Mask Refiner: Multi-Stage Post-Processing Pipeline.

Implements Morphological SNR Optimization and CRF (Conditional Random Field) 
edge refinement to snap predicted mask edges to actual visual object boundaries.
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

    Stages:
      1. Morphological Cleanup (erosion, dilation, opening, closing)
      2. Connected Component Analysis (area & aspect-ratio filtering)
      3. CRF Edge Refinement (DenseCRF or joint bilateral fallback)
      4. Median Filter Smoothing
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
            use_crf: Whether to use CRF refinement.
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

    def refine(
        self,
        prob_map: np.ndarray,
        rgb_frame: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Full refinement pipeline:
          prob_map -> binarize -> morphology -> CCA -> CRF -> smooth.
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
            # Re-apply morphological cleanup after CRF
            binary = self.morphological_cleanup(binary)
            binary = self.filter_components(binary)

        # Stage 4: Median filter smoothing
        if self.median_ksize > 1:
            # Kernel size must be odd
            ksize = self.median_ksize if self.median_ksize % 2 == 1 else self.median_ksize + 1
            binary = cv2.medianBlur(binary, ksize)

        return binary

    def morphological_cleanup(self, binary_mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to remove noise and fill holes."""
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morph_kernel_size, self.morph_kernel_size),
        )
        # Opening: remove small noise blobs
        result = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing: fill small holes
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
        return result

    def filter_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Filter connected components by area and aspect ratio."""
        mask_255 = (binary_mask * 255).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_255, connectivity=8
        )

        result = np.zeros_like(binary_mask, dtype=np.uint8)

        for label_id in range(1, n_labels):  # Skip background
            area = stats[label_id, cv2.CC_STAT_AREA]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]

            if area < self.min_area:
                continue

            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect_ratio:
                continue

            result[labels == label_id] = 1

        return result

    def crf_refine(
        self, prob_map: np.ndarray, rgb_frame: np.ndarray
    ) -> np.ndarray:
        """Use Dense CRF to snap mask edges to visual object boundaries."""
        if HAS_CRF:
            return self._densecrf_refine(prob_map, rgb_frame)
        else:
            return self._bilateral_fallback(prob_map, rgb_frame)

    def _densecrf_refine(
        self, prob_map: np.ndarray, rgb_frame: np.ndarray
    ) -> np.ndarray:
        """DenseCRF-based refinement."""
        h, w = prob_map.shape
        fg_prob = np.clip(prob_map, 1e-6, 1.0 - 1e-6).astype(np.float32)
        bg_prob = 1.0 - fg_prob
        softmax = np.stack([bg_prob, fg_prob], axis=0)

        unary = unary_from_softmax(softmax)
        d = dcrf.DenseCRF2D(w, h, 2)
        d.setUnaryEnergy(unary)

        d.addPairwiseGaussian(
            sxy=self.crf_gaussian_sxy,
            compat=3,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        img_c = np.ascontiguousarray(rgb_frame)
        d.addPairwiseBilateral(
            sxy=self.crf_bilateral_sxy,
            srgb=self.crf_bilateral_srgb,
            rgbim=img_c,
            compat=10,
            kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC,
        )

        q = d.inference(self.crf_iterations)
        map_result = np.argmax(np.array(q).reshape(2, h, w), axis=0)
        return map_result.astype(np.uint8)

    def _bilateral_fallback(
        self, prob_map: np.ndarray, rgb_frame: np.ndarray
    ) -> np.ndarray:
        """Fallback: edge-aware smoothing using bilateral filter."""
        prob_uint8 = (prob_map * 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        filtered = cv2.bilateralFilter(prob_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(gray, 50, 150)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        smooth = cv2.GaussianBlur(prob_uint8, (11, 11), 3)
        result = np.where(edges_dilated > 0, filtered, smooth)
        binary = (result > int(self.threshold * 255)).astype(np.uint8)
        return binary

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
