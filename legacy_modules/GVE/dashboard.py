"""
Metrics Dashboard Generator for PSCDL 2026.

Generates a self-contained HTML performance report with:
  - KPI cards (F1, Precision, Recall, IoU)
  - Per-sample breakdown table
  - Threshold optimization chart (inline SVG)
  - Sample visualizations (base64-embedded images)

No external dependencies — opens in any browser.
"""

import numpy as np
import cv2
import os
import base64
from typing import Dict, List, Optional
from datetime import datetime


class DashboardGenerator:
    """
    Generate a self-contained HTML performance dashboard.

    Usage:
        gen = DashboardGenerator()
        html = gen.generate(
            eval_results=results_from_evaluate_batch,
            threshold_result=result_from_tuner,
            sample_images=[(rgb, pred, gt), ...],
            output_path="report.html"
        )
    """

    def __init__(self, title: str = "PSCDL 2026 — Performance Dashboard"):
        self.title = title

    def _encode_image(self, img: np.ndarray) -> str:
        """Encode numpy image to base64 PNG string for HTML embedding."""
        if len(img.shape) == 2:
            # Grayscale: convert to visible format
            img_vis = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            _, buf = cv2.imencode(".png", img_vis)
        else:
            # Color: assume RGB, convert to BGR for cv2
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
            _, buf = cv2.imencode(".png", img_bgr)
        return base64.b64encode(buf).decode("utf-8")

    def _make_overlay(
        self, rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray
    ) -> np.ndarray:
        """Create an overlay image showing TP (green), FP (red), FN (blue)."""
        h, w = pred.shape[:2]
        overlay = rgb.copy()
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))

        # True positives: green
        tp_mask = (pred > 0) & (gt > 0)
        overlay[tp_mask] = (
            overlay[tp_mask] * 0.4 + np.array([0, 220, 0]) * 0.6
        ).astype(np.uint8)

        # False positives: red
        fp_mask = (pred > 0) & (gt == 0)
        overlay[fp_mask] = (
            overlay[fp_mask] * 0.4 + np.array([220, 50, 50]) * 0.6
        ).astype(np.uint8)

        # False negatives: blue
        fn_mask = (pred == 0) & (gt > 0)
        overlay[fn_mask] = (
            overlay[fn_mask] * 0.4 + np.array([50, 80, 220]) * 0.6
        ).astype(np.uint8)

        return overlay

    def _svg_threshold_chart(
        self, thresholds, f1_scores, precisions, recalls, optimal_t
    ) -> str:
        """Generate an inline SVG chart for threshold vs metrics."""
        chart_w, chart_h = 600, 300
        pad_l, pad_r, pad_t, pad_b = 60, 20, 30, 50

        plot_w = chart_w - pad_l - pad_r
        plot_h = chart_h - pad_t - pad_b

        def to_x(val):
            return pad_l + val * plot_w

        def to_y(val):
            return pad_t + (1.0 - val) * plot_h

        def polyline(xs, ys, color, label):
            points = " ".join(f"{to_x(x):.1f},{to_y(y):.1f}" for x, y in zip(xs, ys))
            return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" />'

        # Normalize thresholds to [0, 1] for the x-axis
        t_min, t_max = min(thresholds), max(thresholds)
        t_range = t_max - t_min if t_max > t_min else 1.0
        t_norm = [(t - t_min) / t_range for t in thresholds]

        lines = []
        lines.append(
            f'<svg viewBox="0 0 {chart_w} {chart_h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;max-width:{chart_w}px;height:auto;background:#1a1a2e;border-radius:12px;padding:8px;">'
        )

        # Grid lines
        for i in range(6):
            y = to_y(i / 5)
            lines.append(
                f'<line x1="{pad_l}" y1="{y}" x2="{chart_w - pad_r}" y2="{y}" '
                f'stroke="#333" stroke-width="0.5" />'
            )
            lines.append(
                f'<text x="{pad_l - 8}" y="{y + 4}" fill="#888" '
                f'font-size="11" text-anchor="end">{i / 5:.1f}</text>'
            )

        # X-axis labels
        for i in range(0, len(thresholds), max(1, len(thresholds) // 6)):
            x = to_x(t_norm[i])
            lines.append(
                f'<text x="{x}" y="{chart_h - 10}" fill="#888" '
                f'font-size="11" text-anchor="middle">{thresholds[i]:.2f}</text>'
            )

        # Plot lines
        lines.append(polyline(t_norm, f1_scores, "#00d4aa", "F1"))
        lines.append(polyline(t_norm, precisions, "#ff6b6b", "Precision"))
        lines.append(polyline(t_norm, recalls, "#4ecdc4", "Recall"))

        # Optimal threshold marker
        opt_norm = (optimal_t - t_min) / t_range
        opt_x = to_x(opt_norm)
        lines.append(
            f'<line x1="{opt_x}" y1="{pad_t}" x2="{opt_x}" y2="{chart_h - pad_b}" '
            f'stroke="#ffd93d" stroke-width="1.5" stroke-dasharray="6,4" />'
        )
        lines.append(
            f'<text x="{opt_x}" y="{pad_t - 8}" fill="#ffd93d" '
            f'font-size="11" text-anchor="middle">t={optimal_t:.2f}</text>'
        )

        # Legend
        legend_items = [
            ("#00d4aa", "F1-Score"),
            ("#ff6b6b", "Precision"),
            ("#4ecdc4", "Recall"),
            ("#ffd93d", "Optimal"),
        ]
        for i, (color, name) in enumerate(legend_items):
            lx = pad_l + 10 + i * 110
            lines.append(
                f'<rect x="{lx}" y="{chart_h - 32}" width="14" height="3" fill="{color}" rx="1" />'
            )
            lines.append(
                f'<text x="{lx + 20}" y="{chart_h - 28}" fill="#ccc" font-size="11">{name}</text>'
            )

        # Axis labels
        lines.append(
            f'<text x="{chart_w // 2}" y="{chart_h - 2}" fill="#aaa" '
            f'font-size="12" text-anchor="middle">Threshold</text>'
        )
        lines.append(
            f'<text x="14" y="{chart_h // 2}" fill="#aaa" font-size="12" '
            f'text-anchor="middle" transform="rotate(-90,14,{chart_h // 2})">Score</text>'
        )

        lines.append("</svg>")
        return "\n".join(lines)

    def generate(
        self,
        eval_results: Dict,
        threshold_result=None,
        sample_images: Optional[List[tuple]] = None,
        filenames: Optional[List[str]] = None,
        output_path: str = "dashboard.html",
    ) -> str:
        """
        Generate the complete HTML dashboard.

        Args:
            eval_results: Output from metrics.evaluate_batch().
            threshold_result: Output from ThresholdTuner.tune() (optional).
            sample_images: List of (rgb, pred_mask, gt_mask) tuples for
                          visualization (optional, max 6 shown).
            filenames: List of filenames for the per-sample table.
            output_path: Path to save the HTML file.

        Returns:
            The output_path.
        """
        agg = eval_results.get("aggregate", {})
        micro = eval_results.get("micro", {})
        per_sample = eval_results.get("per_sample", [])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ---- Build HTML ----
        html_parts = []
        html_parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{self.title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'Inter', -apple-system, sans-serif;
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    color: #e0e0e0;
    min-height: 100vh;
    padding: 32px;
  }}

  .container {{ max-width: 1200px; margin: 0 auto; }}

  .header {{
    text-align: center;
    margin-bottom: 40px;
    padding: 24px;
    background: rgba(255,255,255,0.04);
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
  }}

  .header h1 {{
    font-size: 28px;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4aa, #4ecdc4, #a8edea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }}

  .header .timestamp {{
    font-size: 13px;
    color: #888;
  }}

  /* KPI Cards */
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }}

  .kpi-card {{
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    backdrop-filter: blur(8px);
  }}

  .kpi-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(0, 212, 170, 0.15);
  }}

  .kpi-card .label {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #888;
    margin-bottom: 8px;
  }}

  .kpi-card .value {{
    font-size: 36px;
    font-weight: 700;
  }}

  .kpi-card .sub {{
    font-size: 12px;
    color: #666;
    margin-top: 4px;
  }}

  .kpi-f1 .value {{ color: #00d4aa; }}
  .kpi-precision .value {{ color: #ff6b6b; }}
  .kpi-recall .value {{ color: #4ecdc4; }}
  .kpi-iou .value {{ color: #ffd93d; }}

  /* Section */
  .section {{
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 32px;
    backdrop-filter: blur(8px);
  }}

  .section h2 {{
    font-size: 18px;
    font-weight: 600;
    color: #ccc;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }}

  /* Table */
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }}

  .data-table th {{
    text-align: left;
    padding: 10px 14px;
    color: #888;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 11px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }}

  .data-table td {{
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
  }}

  .data-table tr:hover {{
    background: rgba(255,255,255,0.03);
  }}

  .score-high {{ color: #00d4aa; font-weight: 600; }}
  .score-mid {{ color: #ffd93d; font-weight: 600; }}
  .score-low {{ color: #ff6b6b; font-weight: 600; }}

  /* Visualizations */
  .viz-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 16px;
  }}

  .viz-card {{
    background: rgba(0,0,0,0.3);
    border-radius: 12px;
    overflow: hidden;
  }}

  .viz-card img {{
    width: 100%;
    display: block;
    image-rendering: pixelated;
  }}

  .viz-card .viz-label {{
    padding: 8px 12px;
    font-size: 11px;
    color: #888;
    text-align: center;
  }}

  /* Chart */
  .chart-container {{
    display: flex;
    justify-content: center;
    padding: 16px 0;
  }}

  .badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
  }}

  .badge-macro {{ background: rgba(0, 212, 170, 0.15); color: #00d4aa; }}
  .badge-micro {{ background: rgba(78, 205, 196, 0.15); color: #4ecdc4; }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>{self.title}</h1>
  <div class="timestamp">Generated: {timestamp}</div>
</div>
""")

        # ---- KPI Cards ----
        html_parts.append('<div class="kpi-grid">')

        kpi_data = [
            ("kpi-f1", "F1-Score", agg.get("f1", 0), micro.get("f1", 0)),
            ("kpi-precision", "Precision", agg.get("precision", 0), micro.get("precision", 0)),
            ("kpi-recall", "Recall", agg.get("recall", 0), micro.get("recall", 0)),
            ("kpi-iou", "IoU", agg.get("iou", 0), micro.get("iou", 0)),
        ]

        for cls, label, macro_val, micro_val in kpi_data:
            html_parts.append(f"""
  <div class="kpi-card {cls}">
    <div class="label">{label}</div>
    <div class="value">{macro_val:.4f}</div>
    <div class="sub">
      <span class="badge badge-macro">Macro</span> {macro_val:.4f}
      &nbsp;|&nbsp;
      <span class="badge badge-micro">Micro</span> {micro_val:.4f}
    </div>
  </div>""")

        html_parts.append("</div>")

        # ---- Threshold Chart ----
        if threshold_result is not None:
            html_parts.append('<div class="section">')
            html_parts.append("<h2>Threshold Optimization</h2>")
            html_parts.append(f'<p style="color:#888;font-size:13px;margin-bottom:16px;">'
                            f'Optimal threshold: <strong style="color:#ffd93d">'
                            f'{threshold_result.optimal_threshold:.3f}</strong> '
                            f'(F1 = {threshold_result.optimal_f1:.4f})</p>')
            html_parts.append('<div class="chart-container">')
            html_parts.append(
                self._svg_threshold_chart(
                    threshold_result.thresholds,
                    threshold_result.f1_scores,
                    threshold_result.precisions,
                    threshold_result.recalls,
                    threshold_result.optimal_threshold,
                )
            )
            html_parts.append("</div></div>")

        # ---- Per-Sample Table ----
        if per_sample:
            html_parts.append('<div class="section">')
            html_parts.append("<h2>Per-Sample Breakdown</h2>")
            html_parts.append("""
<table class="data-table">
<thead>
  <tr>
    <th>#</th>
    <th>File</th>
    <th>F1</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>IoU</th>
    <th>TP</th>
    <th>FP</th>
    <th>FN</th>
  </tr>
</thead>
<tbody>""")

            for i, row in enumerate(per_sample):
                fname = filenames[i] if filenames and i < len(filenames) else f"Sample {i + 1}"
                f1_cls = (
                    "score-high" if row["f1"] >= 0.8
                    else "score-mid" if row["f1"] >= 0.5
                    else "score-low"
                )
                html_parts.append(f"""
  <tr>
    <td>{i + 1}</td>
    <td>{fname}</td>
    <td class="{f1_cls}">{row['f1']:.4f}</td>
    <td>{row['precision']:.4f}</td>
    <td>{row['recall']:.4f}</td>
    <td>{row['iou']:.4f}</td>
    <td>{row['tp']:,}</td>
    <td>{row['fp']:,}</td>
    <td>{row['fn']:,}</td>
  </tr>""")

            html_parts.append("</tbody></table></div>")

        # ---- Sample Visualizations ----
        if sample_images:
            html_parts.append('<div class="section">')
            html_parts.append("<h2>Sample Visualizations</h2>")
            html_parts.append(
                '<p style="color:#888;font-size:13px;margin-bottom:16px;">'
                '<span style="color:#00d4aa">■ Green</span> = True Positive &nbsp; '
                '<span style="color:#ff6b6b">■ Red</span> = False Positive &nbsp; '
                '<span style="color:#4488dd">■ Blue</span> = False Negative</p>'
            )

            for idx, (rgb, pred, gt) in enumerate(sample_images[:6]):
                overlay = self._make_overlay(rgb, pred, gt)

                html_parts.append('<div class="viz-grid" style="margin-bottom:20px;">')

                for img, label in [
                    (rgb, "Original"),
                    ((pred * 255).astype(np.uint8), "Prediction"),
                    ((gt * 255).astype(np.uint8), "Ground Truth"),
                    (overlay, "Overlay (TP/FP/FN)"),
                ]:
                    b64 = self._encode_image(img)
                    html_parts.append(f"""
  <div class="viz-card">
    <img src="data:image/png;base64,{b64}" alt="{label}">
    <div class="viz-label">{label}</div>
  </div>""")

                html_parts.append("</div>")

        # ---- Footer ----
        html_parts.append("""
<div style="text-align:center;padding:24px;color:#555;font-size:12px;">
  Persistence Sentinel — PSCDL 2026 Competition Pipeline<br>
  Module 4: Semantic Refinement &amp; Metrics Dashboard
</div>

</div>
</body>
</html>""")

        html = "\n".join(html_parts)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Dashboard saved to {output_path}")
        return output_path
