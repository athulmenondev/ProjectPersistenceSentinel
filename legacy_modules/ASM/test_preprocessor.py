import os
import numpy as np
import cv2
from video_preprocessor import VideoPreprocessor

def create_dummy_video(path, duration=2, fps=30, width=640, height=480):
    """Creates a simple .mp4 video for testing purposes."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(duration * fps):
        # Create a frame with a moving white square
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (i*2 % width, 100), (i*2 % width + 50, 150), (255, 255, 255), -1)
        out.write(frame)

    out.release()
    print(f"Created dummy video at {path}")

def test_preprocessor():
    # Setup test environment
    test_dir = "test_videos"
    os.makedirs(test_dir, exist_ok=True)
    video_path = os.path.join(test_dir, "test_video.mp4")
    create_dummy_video(video_path)

    print("\n--- Testing NumPy Backend ---")
    prep_np = VideoPreprocessor(test_dir, output_size=(224, 224), fps=10, backend="numpy")
    data_np = prep_np[0]
    print(f"NumPy shape: {data_np.shape}") # Expected: (T, 224, 224, 3)
    print(f"Value range: {np.min(data_np):.2f} to {np.max(data_np):.2f}")
    assert data_np.shape[1:] == (224, 224, 3)
    assert 0.0 <= np.min(data_np) <= 1.0 and 0.0 <= np.max(data_np) <= 1.0

    print("\n--- Testing PyTorch Backend ---")
    try:
        prep_torch = VideoPreprocessor(test_dir, output_size=(224, 224), fps=10, backend="torch")
        data_torch = prep_torch[0]
        print(f"Torch shape: {data_torch.shape}") # Expected: (T, 3, 224, 224)
        assert data_torch.shape[1] == 3
        assert data_torch.shape[2:] == (224, 224)
        print("Torch backend verified.")
    except ImportError:
        print("PyTorch not installed, skipping...")

    print("\n--- Testing TensorFlow Backend ---")
    try:
        prep_tf = VideoPreprocessor(test_dir, output_size=(224, 224), fps=10, backend="tensorflow")
        data_tf = prep_tf[0]
        print(f"TF shape: {data_tf.shape}") # Expected: (T, 224, 224, 3)
        assert data_tf.shape[1:] == (224, 224, 3)
        print("TF backend verified.")
    except ImportError:
        print("TensorFlow not installed, skipping...")

    print("\n--- Testing Frame Pairs ---")
    pairs = prep_np.get_frame_pairs(0)
    print(f"Pair shapes: Baseline {pairs[0].shape}, Target {pairs[1].shape}")
    assert pairs[0].shape == pairs[1].shape

    print("\nAll tests passed successfully!")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_preprocessor()
