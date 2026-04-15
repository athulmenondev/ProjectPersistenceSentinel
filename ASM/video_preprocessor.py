import cv2
import numpy as np
import os
import glob
from typing import Tuple, Union, List

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

class VideoPreprocessor:
    """
    VideoPreprocessor converts raw .mp4 videos into standardized tensors for ML pipelines.
    Supports NumPy, PyTorch, and TensorFlow backends.
    """
    def __init__(
        self,
        input_dir: str,
        output_size: Tuple[int, int] = (224, 224),
        fps: int = 10,
        backend: str = "numpy",
        normalization_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Args:
            input_dir: Directory containing .mp4 files.
            output_size: Target resolution (width, height).
            fps: Target frames per second for extraction.
            backend: Output format ("numpy", "torch", "tensorflow").
            normalization_range: Range to normalize pixel values to (e.g., (0.0, 1.0) or (-1.0, 1.0)).
        """
        self.input_dir = input_dir
        self.output_size = output_size
        self.fps = fps
        self.backend = backend.lower()
        self.normalization_range = normalization_range

        # Discover all mp4 files recursively in subdirectories
        self.video_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True))

        if not self.video_paths:
            print(f"Warning: No .mp4 files found in {input_dir}")

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame pixels to the specified range."""
        low, high = self.normalization_range
        # Standardize to [0, 1] first
        frame = frame.astype(np.float32) / 255.0
        # Map [0, 1] to [low, high]
        return frame * (high - low) + low

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize, convert color space, and normalize a single frame.
        """
        # Resize
        frame = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_AREA)
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        frame = self._normalize(frame)
        return frame

    def load_video(self, path: str) -> List[np.ndarray]:
        """
        Extracts frames from a video file at the target FPS.
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {path}")
            return []

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps == 0:
            print(f"Error: Could not determine FPS for {path}")
            return []

        # Calculate the interval to pick frames to match target FPS
        # interval = native_fps / target_fps
        interval = native_fps / self.fps

        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames based on interval
            if count % int(round(interval)) == 0 or (native_fps <= self.fps):
                # For videos with native_fps << target target_fps, we just take all frames
                # and let the downstream model handle temporal interpolation if needed.
                processed = self.preprocess_frame(frame)
                frames.append(processed)

            count += 1

        cap.release()
        return frames

    def _convert_to_backend(self, frames: List[np.ndarray]) -> Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']:
        """
        Convert a list of processed frames to the requested backend format.
        """
        data = np.array(frames) # Shape: (T, H, W, C)

        if self.backend == "numpy":
            return data

        elif self.backend == "torch":
            if torch is None:
                raise ImportError("PyTorch is not installed, but backend='torch' was requested.")
            # Convert to (T, C, H, W)
            tensor = torch.from_numpy(data).permute(0, 3, 1, 2).float()
            return tensor

        elif self.backend == "tensorflow":
            if tf is None:
                raise ImportError("TensorFlow is not installed, but backend='tensorflow' was requested.")
            return tf.convert_to_tensor(data, dtype=tf.float32)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'numpy', 'torch', or 'tensorflow'.")

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, index: int) -> Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']:
        """
        Lazy loads and preprocesses a video by index.
        """
        path = self.video_paths[index]
        frames = self.load_video(path)
        if not frames:
            # Return empty array of correct shape if video is corrupt/empty
            return self._convert_to_backend([])

        return self._convert_to_backend(frames)

    def get_frame_pairs(self, index: int) -> Union[np.ndarray, 'torch.Tensor', 'tf.Tensor']:
        """
        Returns (baseline, target) frame pairs for a video.
        Simple implementation: splits the sequence in half or takes every other frame.
        """
        frames = self.__getitem__(index)
        # Example: Every second frame as target, every first as baseline
        # This is a placeholder for specific project pair logic
        baseline = frames[0::2]
        target = frames[1::2]

        # Trim to match length
        min_len = min(len(baseline), len(target))
        return (baseline[:min_len], target[:min_len])
