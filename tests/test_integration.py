#!/usr/bin/env python3
"""
Integration Test for PSCDL Pipeline

Tests that all four modules work together correctly.
Run: python test_integration.py
"""

import numpy as np
import torch


def test_module_1_io():
    """Test Athul's Video Preprocessor module."""
    print("Testing Module 1 (IO)... ", end="")

    from pcdl.io import VideoPreprocessor

    # Test with dummy data (no actual video needed)
    # We test the class structure and methods
    assert hasattr(VideoPreprocessor, 'preprocess_frame')
    assert hasattr(VideoPreprocessor, 'load_video')
    assert hasattr(VideoPreprocessor, '_convert_to_backend')

    # Test frame preprocessing
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    preprocessor = VideoPreprocessor(input_dir="/tmp", output_size=(224, 224))
    processed = preprocessor.preprocess_frame(dummy_frame)

    assert processed.shape == (224, 224, 3), f"Expected (224, 224, 3), got {processed.shape}"
    assert 0.0 <= processed.min() <= 1.0, "Values should be normalized"

    print("PASSED")
    return True


def test_module_2_spatial():
    """Test Lekshmi's Spatial Segmentation model."""
    print("Testing Module 2 (Spatial)... ", end="")

    from pcdl.spatial import UNetModel, predict_mask, dice_loss

    # Create model
    model = UNetModel()

    # Test forward pass
    dummy_input = torch.randn(1, 3, 256, 256)
    output = model(dummy_input)

    assert output.shape == (1, 1, 256, 256), f"Expected (1, 1, 256, 256), got {output.shape}"
    assert output.min() >= 0 and output.max() <= 1, "Output should be sigmoid probabilities"

    # Test predict_mask function
    prob_map = predict_mask(model, dummy_input)
    assert prob_map.shape == (1, 1, 256, 256)

    # Test dice_loss
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    loss = dice_loss(output, target)
    assert 0 <= loss <= 1, f"Loss should be in [0, 1], got {loss}"

    print("PASSED")
    return True


def test_module_3_temporal():
    """Test Jyotsna's Temporal Filter."""
    print("Testing Module 3 (Temporal)... ", end="")

    from pcdl.temporal import TemporalFilter

    # Create filter
    width, height = 100, 100
    k_threshold = 5
    engine = TemporalFilter(width, height, k_threshold)

    # Test with moving object (should be filtered out)
    for i in range(20):
        # Moving square
        mask = np.zeros((height, width))
        x = i % width
        mask[10:20, x:x+10] = 1.0

        result = engine.process_frame(mask)

        # Moving object should NOT appear in persistent output
        assert result.max() == 0, "Transient object should be filtered out"

    # Reset and test with persistent object
    engine.reset()

    for i in range(k_threshold + 5):
        # Stationary square
        mask = np.zeros((height, width))
        mask[50:60, 50:60] = 1.0

        result = engine.process_frame(mask)

        # After k_threshold frames, persistent object should appear
        if i >= k_threshold:
            assert result.max() > 0, "Persistent object should be detected"

    print("PASSED")
    return True


def test_module_4_postprocess():
    """Test Goutham's Post-Processing module."""
    print("Testing Module 4 (Post-Processing)... ", end="")

    from pcdl.postprocess import MaskRefiner, pixel_f1, pixel_iou

    # Test MaskRefiner
    refiner = MaskRefiner(min_area=50, use_crf=False)

    # Create noisy probability map
    prob_map = np.random.rand(100, 100).astype(np.float32)
    refined = refiner.refine(prob_map)

    assert refined.shape == (100, 100)
    assert set(np.unique(refined)).issubset({0, 1}), "Output should be binary"

    # Test metrics
    pred = np.zeros((100, 100))
    pred[20:40, 20:40] = 1
    gt = np.zeros((100, 100))
    gt[20:40, 20:40] = 1

    f1 = pixel_f1(pred, gt)
    iou = pixel_iou(pred, gt)

    assert f1 == 1.0, f"Perfect match should give F1=1.0, got {f1}"
    assert iou == 1.0, f"Perfect match should give IoU=1.0, got {iou}"

    # Test with no overlap
    pred_empty = np.zeros((100, 100))
    f1_no_overlap = pixel_f1(pred_empty, pred)
    assert f1_no_overlap == 0.0, f"No overlap should give F1=0.0"

    print("PASSED")
    return True


def test_full_pipeline():
    """Test the complete integrated pipeline."""
    print("Testing Full Pipeline Integration... ", end="")

    from pcdl.pipeline import PersistenceSentinelPipeline

    # Initialize with untrained model (just testing structure)
    pipeline = PersistenceSentinelPipeline(
        model_path=None,
        k_frames_threshold=5,
        input_size=(256, 256),
        use_crf=False,
    )

    # Test process_frame with dummy frames
    dummy_frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(10)
    ]

    masks = pipeline.process_frames_batch(dummy_frames)

    assert len(masks) == 10, f"Expected 10 masks, got {len(masks)}"
    assert masks[0].shape == (480, 640), f"Mask shape mismatch"

    print("PASSED")
    return True


def main():
    print("=" * 50)
    print("PSCDL Integration Tests")
    print("=" * 50)
    print()

    tests = [
        ("Module 1: IO Pipeline", test_module_1_io),
        ("Module 2: Spatial Model", test_module_2_spatial),
        ("Module 3: Temporal Filter", test_module_3_temporal),
        ("Module 4: Post-Processing", test_module_4_postprocess),
        ("Full Pipeline Integration", test_full_pipeline),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print()
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
