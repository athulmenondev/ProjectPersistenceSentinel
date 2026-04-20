"""
Persistence Sentinel Pipeline

Main orchestrator that connects all four modules:
    Athul (IO) → Lekshmi (Spatial) → Jyotsna (Temporal) → Goutham (Post-process)
"""

import torch
import numpy as np
import cv2
from typing import Optional, Tuple, List

from pcdl.io import VideoPreprocessor
from pcdl.spatial import UNetModel, predict_mask
from pcdl.temporal import TemporalFilter
from pcdl.postprocess import MaskRefiner


class PersistenceSentinelPipeline:
    """
    Complete PSCDL pipeline for persistent encroachment detection.

    Usage:
        pipeline = PersistenceSentinelPipeline(
            model_path="model.pth",
            k_frames_threshold=30,
            input_size=(256, 256)
        )
        masks = pipeline.process_video("video.mp4")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 256),
        k_frames_threshold: int = 30,
        fps: int = 10,
        backend: str = "torch",
        threshold: float = 0.5,
        min_area: int = 100,
        use_crf: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to trained model weights (.pth file).
            input_size: (width, height) for frame resizing.
            k_frames_threshold: Frames needed for persistence detection.
            fps: Target frames per second for extraction.
            backend: "torch", "tensorflow", or "numpy".
            threshold: Binarization threshold for masks.
            min_area: Minimum connected component area to keep.
            use_crf: Whether to use CRF edge refinement.
            device: "cuda", "cpu", or None (auto-detect).
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Initialize spatial model
        self.model = UNetModel().to(self.device)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model weights from {model_path}")
        self.model.eval()

        # Initialize temporal filter
        self.temporal_filter = TemporalFilter(
            width=input_size[0],
            height=input_size[1],
            k_frames_threshold=k_frames_threshold,
        )

        # Initialize post-processor
        self.refiner = MaskRefiner(
            min_area=min_area,
            use_crf=use_crf,
            threshold=threshold,
        )

        # Initialize preprocessor
        self.preprocessor: Optional[VideoPreprocessor] = None
        self.input_size = input_size
        self.fps = fps
        self.backend = backend

    def reset(self):
        """Reset temporal state for processing a new video."""
        self.temporal_filter.reset()

    def process_frame(
        self,
        frame: np.ndarray,
        return_prob_map: bool = False,
    ) -> np.ndarray:
        """
        Process a single frame through the complete pipeline.

        Args:
            frame: Input frame (H, W, 3) RGB, values [0, 255] or [0, 1].
            return_prob_map: If True, also return the probability map.

        Returns:
            Binary mask (H, W) with values {0, 1}.
        """
        # Preprocess frame
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
        frame_resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)

        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # 1. Spatial inference (Lekshmi's module)
        prob_map = predict_mask(self.model, frame_tensor)
        prob_map = prob_map.squeeze().cpu().numpy()

        # 2. Mask refinement (Goutham's module - stage 1)
        refined_mask = self.refiner.refine(prob_map, rgb_frame=(frame_resized * 255).astype(np.uint8))

        # 3. Temporal filtering (Jyotsna's module)
        final_mask = self.temporal_filter.process_frame(refined_mask)

        if return_prob_map:
            return final_mask, prob_map
        return final_mask

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> List[np.ndarray]:
        """
        Process an entire video file.

        Args:
            video_path: Path to .mp4 video file.
            output_dir: If provided, save masks to this directory.
            verbose: Print progress messages.

        Returns:
            List of binary masks, one per frame.
        """
        import os
        from pcdl.postprocess import export_mask

        # Reset temporal state
        self.reset()

        # Initialize preprocessor for this video
        self.preprocessor = VideoPreprocessor(
            input_dir=os.path.dirname(video_path),
            output_size=self.input_size,
            fps=self.fps,
            backend=self.backend,
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        original_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        masks = []
        frame_idx = 0

        if verbose:
            print(f"Processing video: {video_path}")
            print(f"Original size: {original_size}, Target: {self.input_size}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process through pipeline
            mask = self.process_frame(frame_rgb)

            # Resize back to original if needed
            if mask.shape != original_size[::-1]:
                mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

            masks.append(mask)

            # Save if output directory specified
            if output_dir:
                export_mask(
                    mask,
                    os.path.join(output_dir, f"mask_{frame_idx:05d}.png"),
                )

            frame_idx += 1

            if verbose and frame_idx % 30 == 0:
                print(f"  Processed {frame_idx} frames...")

        cap.release()

        if verbose:
            print(f"Complete. Processed {frame_idx} frames.")

        return masks

    def process_frames_batch(
        self,
        frames: List[np.ndarray],
        return_prob_maps: bool = False,
    ) -> List[np.ndarray]:
        """
        Process a batch of frames (useful for testing or custom pipelines).

        Args:
            frames: List of RGB frames (H, W, 3).
            return_prob_maps: Also return intermediate probability maps.

        Returns:
            List of binary masks.
        """
        self.reset()
        masks = []
        prob_maps = [] if return_prob_maps else None

        for frame in frames:
            if return_prob_maps:
                mask, prob = self.process_frame(frame, return_prob_map=True)
                prob_maps.append(prob)
            else:
                mask = self.process_frame(frame)
            masks.append(mask)

        if return_prob_maps:
            return masks, prob_maps
        return masks


def run_demo(video_path: Optional[str] = None):
    """Run a demo of the pipeline."""
    print("=" * 60)
    print("Persistence Sentinel Pipeline Demo")
    print("=" * 60)

    pipeline = PersistenceSentinelPipeline(
        model_path=None,  # Will use untrained weights
        k_frames_threshold=30,
        input_size=(256, 256),
    )

    if video_path:
        print(f"\nProcessing video: {video_path}")
        masks = pipeline.process_video(video_path, output_dir=None)
        print(f"Generated {len(masks)} masks")
    else:
        print("\nNo video specified. Pipeline initialized and ready.")
        print("Usage: pipeline.process_video('path/to/video.mp4')")

    return pipeline


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(video)
