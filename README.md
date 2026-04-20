# PSCDL 2026 - Persistence Scene Change Detection

A complete pipeline for detecting persistent encroachment events in video streams, designed for the [PSCDL 2026 Challenge](https://pscdl.org/).

## Overview

This system detects objects that have been **persistently** introduced into a scene (e.g., dropped luggage, debris) while filtering out transient motion (pedestrians, vehicles).

### Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   Module 1  │ ──→ │   Module 2   │ ──→ │   Module 3  │ ──→ │   Module 4  │
│   (ASM)     │     │    (LH)      │     │    (JM)     │     │    (GVE)    │
│             │     │              │     │             │     │             │
│ Video IO    │     │ Spatial      │     │ Temporal    │     │ Post-       │
│ Preprocessor│     │ Segmentation │     │ Filtering   │     │ Processing  │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
       │                   │                    │                   │
       │                   │                    │                   │
       ▼                   ▼                    ▼                   ▼
  Raw frames →      Probability       Persistent          Clean binary
  Tensors           maps [H,W]        masks [H,W]         masks + metrics
```

### API Contract (Data Flow)

| Stage                  | Input           | Output                 | Shape                            |
| ---------------------- | --------------- | ---------------------- | -------------------------------- |
| **Module 1 (Athul)**   | Raw .mp4 video  | Normalized tensors     | `(T, H, W, 3)` or `(T, 3, H, W)` |
| **Module 2 (Lekshmi)** | Image tensor    | Probability map        | `(H, W)` values ∈ [0, 1]         |
| **Module 3 (Jyotsna)** | Probability map | Persistent mask        | `(H, W)` binary {0, 1}           |
| **Module 4 (Goutham)** | Filtered mask   | Refined mask + metrics | `(H, W)` binary + F1/IoU         |

---

## Team

| Module                    | Assignee    | Responsibility                        |
| ------------------------- | ----------- | ------------------------------------- |
| Module 1: IO Pipeline     | **Athul**   | Video preprocessing, frame extraction |
| Module 2: Spatial Model   | **Lekshmi** | U-Net segmentation architecture       |
| Module 3: Temporal Engine | **Jyotsna** | Persistence filtering logic           |
| Module 4: Post-Processing | **Goutham** | Mask refinement, evaluation, export   |

---

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

```bash
# Place training data in LH/data/train/images and LH/data/train/masks
cd LH
python train.py
# Outputs: model.pth
```

### 3. Process a Video

```bash
# From project root
python main.py --video path/to/video.mp4 --model LH/model.pth --output output_masks
```

### 4. Evaluate Results

```bash
python main.py --eval --pred output_masks --gt ground_truth_masks
```

---

## Usage

### Command Line

```bash
# Process a video
python main.py --video <video.mp4> \
               --model <model.pth> \
               --output <output_dir> \
               --k-frames 30 \
               --fps 10 \
               --input-size 256x256 \
               --verbose

# Evaluate against ground truth
python main.py --eval --pred <pred_dir> --gt <gt_dir>

# Run demo (no video)
python main.py --demo
```

### Python API

```python
from pcdl import PersistenceSentinelPipeline

# Initialize pipeline
pipeline = PersistenceSentinelPipeline(
    model_path="LH/model.pth",
    k_frames_threshold=30,  # Persistence threshold
    input_size=(256, 256),
    fps=10,
)

# Process video
masks = pipeline.process_video("video.mp4", output_dir="output")

# Or process frame-by-frame
import cv2
cap = cv2.VideoCapture("video.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mask = pipeline.process_frame(frame_rgb)
```

### Individual Modules

```python
# Module 1: Video Preprocessing
from pcdl import VideoPreprocessor

preprocessor = VideoPreprocessor(
    input_dir="videos/",
    output_size=(256, 256),
    fps=10,
    backend="torch"  # or "numpy", "tensorflow"
)
frames = preprocessor[0]  # Load first video

# Module 2: Spatial Model
from pcdl import UNetModel, predict_mask

model = UNetModel()
model.load_state_dict(torch.load("model.pth"))
prob_map = predict_mask(model, input_tensor)

# Module 3: Temporal Filtering
from pcdl import TemporalFilter

filter = TemporalFilter(width=256, height=256, k_frames_threshold=30)
persistent_mask = filter.process_frame(prob_map)

# Module 4: Metrics & Export
from pcdl import MaskRefiner, pixel_f1, export_mask

refiner = MaskRefiner()
clean_mask = refiner.refine(prob_map, rgb_frame=frame)

f1 = pixel_f1(pred_mask, gt_mask)
export_mask(clean_mask, "output/mask_001.png")
```

---

## Configuration

### Key Parameters

| Parameter            | Description                   | Default |
| -------------------- | ----------------------------- | ------- |
| `k_frames_threshold` | Frames needed for persistence | 30      |
| `fps`                | Frame extraction rate         | 10      |
| `input_size`         | Model input resolution        | 256x256 |
| `threshold`          | Mask binarization             | 0.5     |
| `min_area`           | Minimum blob area (pixels)    | 100     |
| `use_crf`            | Enable CRF refinement         | True    |

### K-Frames to Minutes Conversion

The persistence threshold K in frames converts to time:

```
K_minutes = (k_frames_threshold / fps) / 60
```

For example, with `fps=10` and `k_frames_threshold=30`:

- 30 frames = 3 seconds of persistence
- For 1 minute: `k_frames_threshold = 60 * 10 = 600`

---

## Project Structure

```
ProjectPersistenceSentinel/
├── pcdl/                 # Unified package (Core Project)
├── data/
│   ├── dataset/
│   │   ├── train/        # Video 1 to 4 (Used for Training)
│   │   └── test/         # Video 5 (Reserved for Final Testing)
│   ├── docs/             # PDF and documentation
│   └── temp/             # Temporary/Mock data
├── legacy_modules/       # Original individual modules
├── tests/                # System tests and utilities
├── main.py               # CLI entry point
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

---

## Evaluation Metrics

The pipeline implements the official PSCDL metrics:

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)
- **IoU**: TP / (TP + FP + FN)

All metrics are computed at the **pixel level** as required by the competition.

---

## Troubleshooting

### "PyTorch not installed"

```bash
pip install torch torchvision
```

### "No .mp4 files found"

Ensure videos are in the specified directory with `.mp4` extension (case-sensitive).

### CRF import warning

The `pydensecrf` package requires compilation. If unavailable, the pipeline falls back to bilateral filtering.

### CUDA out of memory

Reduce `--input-size` or add `--cpu` flag.

---

## License

MIT License - See LICENSE file for details.

## Acknowledgments

This project is developed for the PSCDL 2026 Challenge by Team PSCDL.
