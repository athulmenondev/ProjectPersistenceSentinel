"""
Temporal Persistence Engine

Filters out transient motion (cars, pedestrians) and keeps only persistent changes.
"""

import numpy as np


class TemporalFilter:
    """
    Temporal filter that tracks pixel persistence over time.

    A pixel is marked as "persistent change" only if it deviates from
    the background for K consecutive frames.
    """
    def __init__(self, width: int, height: int, k_frames_threshold: int):
        """
        Args:
            width: Width of input frames.
            height: Height of input frames.
            k_frames_threshold: Number of consecutive frames a pixel must be
                                active to be considered an encroachment.
        """
        self.width = width
        self.height = height
        self.k_frames = k_frames_threshold

        # Matrix to keep track of how many consecutive frames a pixel has been active
        self.persistence_matrix = np.zeros((height, width), dtype=np.int32)

    def reset(self):
        """Reset the persistence matrix (useful for new video sequences)."""
        self.persistence_matrix = np.zeros((self.height, self.width), dtype=np.int32)

    def process_frame(self, current_mask: np.ndarray) -> np.ndarray:
        """
        Process a single frame mask and return filtered persistent mask.

        Args:
            current_mask: Input mask (H, W) with values in [0, 1] or binary.

        Returns:
            Filtered mask where only persistent changes are marked.
        """
        # Increment counter where mask is active (prob > 0.5), reset where inactive
        self.persistence_matrix = np.where(
            current_mask > 0.5,
            self.persistence_matrix + 1,
            0
        )

        # Output mask: 1 only where persistence >= threshold
        persistent_mask = (self.persistence_matrix >= self.k_frames).astype(np.float32)

        return persistent_mask


def generate_mock_frame(frame_idx, width=100, height=100):
    """
    Utility: Simulates a probability mask from a spatial model.
    - A moving square representing a car/person (transient).
    - A stationary square representing a dropped bag (persistent, appears at frame 20).
    """
    frame = np.zeros((height, width))
    
    # 1. Transient Object (e.g., car moving across the screen)
    car_x = (frame_idx * 2) % width
    car_y = (10 + frame_idx) % height
    
    x_end = min(car_x + 10, width)
    y_end = min(car_y + 10, height)
    frame[car_y:y_end, car_x:x_end] = 1.0

    # 2. Persistent Object (e.g., dropped luggage)
    if frame_idx >= 20:
        box_x, box_y = 60, 40
        frame[box_y:box_y+8, box_x:box_x+8] = 1.0
        
    return frame
