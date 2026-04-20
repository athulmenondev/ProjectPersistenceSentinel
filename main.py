#!/usr/bin/env python3
"""
PSCDL 2026 - Persistence Scene Change Detection
Main Entry Point

Usage:
    python main.py --video <path> --output <dir> --model <path>
    python main.py --demo
    python main.py --eval --pred <dir> --gt <dir>
"""

import argparse
import os
import sys
import json

from pcdl import (
    PersistenceSentinelPipeline,
    evaluate_directory,
    pixel_f1,
    pixel_iou,
    validate_submission,
)


def main():
    parser = argparse.ArgumentParser(
        description="PSCDL 2026 - Persistence Scene Change Detection Pipeline"
    )

    # Input/Output
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_masks", help="Output directory for masks")
    parser.add_argument("--model", type=str, help="Path to trained model weights (.pth)")

    # Pipeline parameters
    parser.add_argument("--k-frames", type=int, default=30, help="Persistence threshold (frames)")
    parser.add_argument("--fps", type=int, default=10, help="Target FPS for frame extraction")
    parser.add_argument("--input-size", type=str, default="256x256", help="Input frame size (WxH)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum connected component area")

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="Run evaluation mode")
    parser.add_argument("--pred", type=str, help="Directory with predicted masks (for eval)")
    parser.add_argument("--gt", type=str, help="Directory with ground truth masks (for eval)")

    # Misc
    parser.add_argument("--demo", action="store_true", help="Run demo mode")
    parser.add_argument("--no-crf", action="store_true", help="Disable CRF edge refinement")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Parse input size
    try:
        w, h = args.input_size.split("x")
        input_size = (int(w), int(h))
    except ValueError:
        print("Error: --input-size must be in format WxH (e.g., 256x256)")
        sys.exit(1)

    # Device selection
    device = "cpu" if args.cpu else None

    # Evaluation mode
    if args.eval:
        if not args.pred or not args.gt:
            print("Error: --eval requires --pred and --gt directories")
            sys.exit(1)

        print(f"Evaluating predictions: {args.pred}")
        print(f"Against ground truth: {args.gt}")
        print("-" * 40)

        results = evaluate_directory(args.pred, args.gt)

        print("\n=== Results ===")
        print(f"Samples evaluated: {len(results.get('filenames', []))}")
        print(f"Precision: {results['aggregate']['precision']:.4f}")
        print(f"Recall:    {results['aggregate']['recall']:.4f}")
        print(f"F1-Score:  {results['aggregate']['f1']:.4f}")
        print(f"IoU:       {results['aggregate']['iou']:.4f}")
        print(f"\nMicro F1:  {results['micro']['f1']:.4f}")
        print(f"Micro IoU: {results['micro']['iou']:.4f}")

        # Save detailed results
        with open("evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\nDetailed results saved to evaluation_results.json")

        return 0

    # Demo mode
    if args.demo:
        from pcdl.pipeline import run_demo
        run_demo(args.video)
        return 0

    # Processing mode
    if args.video:
        print(f"Processing video: {args.video}")

        pipeline = PersistenceSentinelPipeline(
            model_path=args.model,
            input_size=input_size,
            k_frames_threshold=args.k_frames,
            fps=args.fps,
            threshold=args.threshold,
            min_area=args.min_area,
            use_crf=not args.no_crf,
            device=device,
        )

        os.makedirs(args.output, exist_ok=True)

        masks = pipeline.process_video(
            args.video,
            output_dir=args.output,
            verbose=args.verbose or True,
        )

        print(f"\nGenerated {len(masks)} masks in {args.output}/")

        # Validate output
        validation = validate_submission(args.output)
        if validation["valid"]:
            print("Output validation: PASSED")
        else:
            print("Output validation: ISSUES FOUND")
            for err in validation["errors"]:
                print(f"  - {err}")

        return 0

    # No mode specified - show help
    print("PSCDL 2026 Pipeline")
    print("-" * 40)
    print("Modes:")
    print("  Process video:  python main.py --video <path> --model <path>")
    print("  Evaluate:       python main.py --eval --pred <dir> --gt <dir>")
    print("  Demo:           python main.py --demo")
    print("")
    print("Run 'python main.py --help' for all options")

    return 0


if __name__ == "__main__":
    sys.exit(main())
