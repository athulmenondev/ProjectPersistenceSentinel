"""
Demo Script — Module 4: Semantic Refinement & Metrics Dashboard
================================================================
Generates synthetic data via MockGenerator, runs it through all
pipeline stages, and produces a self-contained HTML dashboard
you can open in any browser.

Run:
    python -m GVE.run_demo
"""

import numpy as np
import os
import webbrowser

from .mock_generator import MockGenerator
from .metrics import evaluate_batch
from .threshold_tuner import ThresholdTuner
from .mask_refiner import MaskRefiner
from .dashboard import DashboardGenerator


def main():
    print("=" * 60)
    print("  PSCDL 2026 — Module 4 Demo")
    print("  Semantic Refinement & Metrics Dashboard")
    print("=" * 60)
    print()

    # ── 1. Generate synthetic test data ──────────────────────────
    print("[1/5] Generating synthetic test data ...")
    gen = MockGenerator(height=256, width=256, seed=42)

    n_samples = 8
    gt_masks = []
    prob_maps = []
    rgb_frames = []
    filenames = []

    for i in range(n_samples):
        # Re-seed for variety but reproducibility
        gen_i = MockGenerator(height=256, width=256, seed=42 + i)

        gt = gen_i.generate_ground_truth(
            n_objects=np.random.randint(1, 4),
            min_radius=20,
            max_radius=60,
        )
        pred = gen_i.generate_noisy_prediction(
            gt,
            noise_level=0.12 + i * 0.02,  # Increasing noise per sample
            false_positive_rate=0.01 + i * 0.005,
        )
        rgb = gen_i.generate_rgb_frame(
            gt,
            bg_color=(30 + i * 5, 50 + i * 3, 70 + i * 2),
            obj_color=(180 - i * 10, 120 + i * 5, 60 + i * 8),
        )

        gt_masks.append(gt)
        prob_maps.append(pred)
        rgb_frames.append(rgb)
        filenames.append(f"scene_{i+1:03d}.png")

    print(f"   ✓ Generated {n_samples} synthetic samples (256×256)")

    # ── 2. Threshold Tuning ──────────────────────────────────────
    print("[2/5] Running threshold sweep ...")
    tuner = ThresholdTuner(
        thresholds=np.arange(0.05, 0.95, 0.025),
        target_metric="f1",
    )
    threshold_result = tuner.tune(prob_maps, gt_masks)

    print(f"   ✓ Optimal threshold: {threshold_result.optimal_threshold:.3f}")
    print(f"     F1 = {threshold_result.optimal_f1:.4f}  |  "
          f"Precision = {threshold_result.optimal_precision:.4f}  |  "
          f"Recall = {threshold_result.optimal_recall:.4f}")

    # ── 3. Mask Refinement ───────────────────────────────────────
    print("[3/5] Refining masks ...")
    refiner = MaskRefiner(
        morph_kernel_size=5,
        min_area=200,
        use_crf=True,
        threshold=threshold_result.optimal_threshold,
    )
    print(f"   Pipeline:\n   {refiner.get_pipeline_description()}")

    refined_masks = []
    for prob, rgb in zip(prob_maps, rgb_frames):
        refined = refiner.refine(
            prob,
            rgb_frame=rgb,
            threshold=threshold_result.optimal_threshold,
        )
        refined_masks.append(refined)

    print(f"   ✓ Refined {len(refined_masks)} masks")

    # ── 4. Evaluate refined masks ────────────────────────────────
    print("[4/5] Computing metrics ...")

    # Before refinement
    raw_preds = [(p >= threshold_result.optimal_threshold).astype(np.uint8) for p in prob_maps]
    raw_results = evaluate_batch(raw_preds, gt_masks)

    # After refinement
    refined_results = evaluate_batch(refined_masks, gt_masks)

    raw_f1 = raw_results["aggregate"]["f1"]
    ref_f1 = refined_results["aggregate"]["f1"]
    improvement = ref_f1 - raw_f1

    print(f"   ✓ Raw F1:      {raw_f1:.4f}")
    print(f"   ✓ Refined F1:  {ref_f1:.4f}  "
          f"({'↑' if improvement >= 0 else '↓'} {abs(improvement):.4f})")

    # ── 5. Generate Dashboard ────────────────────────────────────
    print("[5/5] Generating HTML dashboard ...")

    # Prepare sample images for visualization (use first 6)
    sample_images = []
    for i in range(min(6, n_samples)):
        sample_images.append((rgb_frames[i], refined_masks[i], gt_masks[i]))

    dashboard = DashboardGenerator(
        title="PSCDL 2026 — Module 4 Demo Dashboard"
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "demo_dashboard.html")

    dashboard.generate(
        eval_results=refined_results,
        threshold_result=threshold_result,
        sample_images=sample_images,
        filenames=filenames,
        output_path=output_path,
    )

    print(f"   ✓ Dashboard saved to: {output_path}")
    print()
    print("=" * 60)
    print("  DONE!  Opening dashboard in your browser ...")
    print("=" * 60)

    # Open in the default browser
    webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    main()
