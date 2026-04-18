"""
Mock Data Generator for Module 4 Standalone Testing.

Generates synthetic probability maps, ground truth masks, and temporal
sequences so Module 4 can be developed and tested independently of
Modules 2 (Segmentation) and 3 (Temporal Filtering).
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class MockGenerator:
    """
    Generates synthetic test data for the PSCDL pipeline.

    Usage:
        gen = MockGenerator(height=224, width=224)
        gt = gen.generate_ground_truth(n_objects=2)
        pred = gen.generate_noisy_prediction(gt, noise_level=0.15)
    """

    def __init__(self, height: int = 224, width: int = 224, seed: Optional[int] = 42):
        self.height = height
        self.width = width
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------ #
    #  Ground Truth Mask Generation
    # ------------------------------------------------------------------ #

    def generate_ground_truth(
        self,
        n_objects: int = 1,
        min_radius: int = 15,
        max_radius: int = 50,
        shape_types: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Create a binary ground truth mask with realistic blob shapes.

        Args:
            n_objects: Number of objects to place in the mask.
            min_radius: Minimum object radius in pixels.
            max_radius: Maximum object radius in pixels.
            shape_types: List of shape types to use. Options: 'circle',
                         'rectangle', 'ellipse', 'polygon'. If None,
                         randomly picks from all.

        Returns:
            Binary mask of shape (H, W) with values {0, 1}, dtype uint8.
        """
        if shape_types is None:
            shape_types = ["circle", "rectangle", "ellipse", "polygon"]

        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        for _ in range(n_objects):
            shape = self.rng.choice(shape_types)
            r = self.rng.randint(min_radius, max_radius + 1)

            # Keep objects away from edges
            cx = self.rng.randint(r + 5, self.width - r - 5)
            cy = self.rng.randint(r + 5, self.height - r - 5)

            if shape == "circle":
                cv2.circle(mask, (cx, cy), r, 1, thickness=-1)

            elif shape == "rectangle":
                half_w = self.rng.randint(r // 2, r)
                half_h = self.rng.randint(r // 2, r)
                cv2.rectangle(
                    mask,
                    (cx - half_w, cy - half_h),
                    (cx + half_w, cy + half_h),
                    1,
                    thickness=-1,
                )

            elif shape == "ellipse":
                axes = (
                    self.rng.randint(r // 2, r),
                    self.rng.randint(r // 2, r),
                )
                angle = self.rng.randint(0, 180)
                cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1, thickness=-1)

            elif shape == "polygon":
                n_pts = self.rng.randint(5, 10)
                angles = np.sort(self.rng.uniform(0, 2 * np.pi, n_pts))
                radii = self.rng.uniform(r * 0.5, r, n_pts)
                pts = np.column_stack(
                    [
                        cx + (radii * np.cos(angles)).astype(int),
                        cy + (radii * np.sin(angles)).astype(int),
                    ]
                ).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)

        return mask

    # ------------------------------------------------------------------ #
    #  Noisy Prediction Generation
    # ------------------------------------------------------------------ #

    def generate_noisy_prediction(
        self,
        gt_mask: np.ndarray,
        noise_level: float = 0.15,
        false_positive_rate: float = 0.02,
        edge_blur_sigma: float = 3.0,
    ) -> np.ndarray:
        """
        Generate a realistic noisy probability map from a ground truth mask.

        Simulates imperfect model output:
          - Gaussian blur on edges (imprecise boundaries)
          - Random noise throughout
          - Small false positive blobs

        Args:
            gt_mask: Binary ground truth mask (H, W) with values {0, 1}.
            noise_level: Standard deviation of Gaussian noise (0.0 - 1.0).
            false_positive_rate: Fraction of background pixels that become
                                 false-positive blobs.
            edge_blur_sigma: Sigma for Gaussian blur applied to edges.

        Returns:
            Probability map of shape (H, W) with values in [0.0, 1.0], dtype float32.
        """
        h, w = gt_mask.shape[:2]

        # Start from ground truth as float
        prob_map = gt_mask.astype(np.float32)

        # Blur edges to simulate imprecise segmentation
        if edge_blur_sigma > 0:
            ksize = int(edge_blur_sigma * 4) | 1  # Ensure odd kernel size
            prob_map = cv2.GaussianBlur(prob_map, (ksize, ksize), edge_blur_sigma)

        # Add Gaussian noise
        noise = self.rng.normal(0, noise_level, (h, w)).astype(np.float32)
        prob_map = prob_map + noise

        # Add false positive blobs
        if false_positive_rate > 0:
            n_fp_pixels = int(h * w * false_positive_rate)
            fp_mask = np.zeros((h, w), dtype=np.float32)
            n_blobs = max(1, n_fp_pixels // 100)

            for _ in range(n_blobs):
                blob_r = self.rng.randint(3, 12)
                bx = self.rng.randint(blob_r, w - blob_r)
                by = self.rng.randint(blob_r, h - blob_r)
                cv2.circle(fp_mask, (bx, by), blob_r, 1.0, thickness=-1)

            # Scale false positives to a moderate confidence
            fp_mask *= self.rng.uniform(0.3, 0.7)
            prob_map = np.maximum(prob_map, fp_mask)

        # Clamp to [0, 1]
        prob_map = np.clip(prob_map, 0.0, 1.0)

        return prob_map

    # ------------------------------------------------------------------ #
    #  Temporal Sequence Generation
    # ------------------------------------------------------------------ #

    def generate_temporal_sequence(
        self,
        n_frames: int = 30,
        persistent_object: bool = True,
        transient_object: bool = True,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate a temporal sequence of probability maps containing:
          - A stationary persistent object (should be detected)
          - A moving transient object (should be filtered out)

        Args:
            n_frames: Number of frames in the sequence.
            persistent_object: Whether to include a stationary object.
            transient_object: Whether to include a moving object.

        Returns:
            Tuple of:
              - List of probability maps (n_frames × H × W), float32
              - Ground truth mask for the persistent object only (H × W), uint8
        """
        h, w = self.height, self.width
        prob_maps = []
        gt_persistent = np.zeros((h, w), dtype=np.uint8)

        # Persistent object location (stationary)
        p_cx, p_cy, p_r = w // 4, h // 2, 25
        if persistent_object:
            cv2.circle(gt_persistent, (p_cx, p_cy), p_r, 1, thickness=-1)

        # Transient object parameters (moves across the frame)
        t_r = 20
        t_start_x = 10
        t_end_x = w - 10
        t_cy = h // 4

        for i in range(n_frames):
            frame_map = np.zeros((h, w), dtype=np.float32)

            # Draw persistent object (always present)
            if persistent_object:
                cv2.circle(frame_map, (p_cx, p_cy), p_r, 0.9, thickness=-1)

            # Draw transient object (moves left to right)
            if transient_object:
                progress = i / max(n_frames - 1, 1)
                t_cx = int(t_start_x + progress * (t_end_x - t_start_x))
                cv2.circle(frame_map, (t_cx, t_cy), t_r, 0.85, thickness=-1)

            # Add light noise
            noise = self.rng.normal(0, 0.05, (h, w)).astype(np.float32)
            frame_map = np.clip(frame_map + noise, 0.0, 1.0)

            prob_maps.append(frame_map)

        return prob_maps, gt_persistent

    # ------------------------------------------------------------------ #
    #  RGB Frame Generation (for CRF testing)
    # ------------------------------------------------------------------ #

    def generate_rgb_frame(
        self,
        gt_mask: np.ndarray,
        bg_color: Tuple[int, int, int] = (40, 60, 80),
        obj_color: Tuple[int, int, int] = (180, 120, 60),
    ) -> np.ndarray:
        """
        Generate a synthetic RGB frame where the object region has a
        distinct color from the background. Useful for testing CRF
        edge refinement.

        Args:
            gt_mask: Binary mask (H, W) with values {0, 1}.
            bg_color: RGB tuple for background.
            obj_color: RGB tuple for object.

        Returns:
            RGB image of shape (H, W, 3), dtype uint8.
        """
        h, w = gt_mask.shape[:2]
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Background
        frame[:, :] = bg_color

        # Object region
        frame[gt_mask == 1] = obj_color

        # Add some texture noise to make it realistic
        texture = self.rng.randint(0, 15, (h, w, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + texture.astype(np.int16), 0, 255).astype(
            np.uint8
        )

        return frame
