# Solution Strategy for PSCDL 2026 Challenge

## Problem Analysis
The goal is to build a **Persistence Scene Change Detection (PSCD)** pipeline. The system must detect an "encroachment" event in a video and generate a pixel-wise change mask, but only after the encroachment has persisted for a specific duration $K$ minutes.

### Key Constraints:
- **Baseline**: The start of the video serves as the ground truth (no encroachment).
- **Persistence**: Mask generation must be delayed until $K$ minutes after the start of the event.
- **False Positives**: Must ignore transient changes (noise, lighting, moving objects) and only detect persistent ones.
- **Performance**: Real-time or near real-time applicability.
- **Evaluation**: F1 Score based on the generated mask vs. ground truth.

---

## Proposed Solution Approach

### 1. Pre-processing & Background Modeling
Since the initial segment is guaranteed to be the baseline, we can establish a robust reference model.
- **Initial Reference**: Average the first few seconds of the video to create a "Clean Background" image.
- **Dynamic Background Update**: Use a Gaussian Mixture Model (GMM) or a running average to handle slow lighting changes and minor noise, ensuring the background model evolves without absorbing the encroachment event.

### 2. Change Detection Engine
To identify candidate encroachment areas:
- **Frame Differencing**: Compute the absolute difference between the current frame and the reference background.
- **Temporal Accumulation**: Instead of instant detection, maintain a "persistence map" (a counter for each pixel). A pixel is marked as "changed" only if it deviates from the background for a significant number of consecutive frames.
- **Morphological Filtering**: Use erosion and dilation to remove small noise particles and fill holes in the detected encroachment area.

### 3. Persistence Logic (The $K$-Minute Trigger)
This is the core requirement of the challenge.
- **Event Timestamping**: Once a cluster of pixels is detected as a "persistent change" (using a threshold of consecutive frames), mark the start time $T_{start}$.
- **Timer Mechanism**: Start a timer from $T_{start}$.
- **Mask Generation**: 
    - For $t < T_{start} + K$: Output no mask (or an empty mask).
    - For $t \geq T_{start} + K$: Generate the pixel-wise change mask based on the accumulated difference map.

### 4. Refinement & Post-processing
To maximize the F1 Score:
- **Connected Component Analysis (CCA)**: Filter out small blobs that don't meet a minimum area requirement for an "encroachment."
- **Edge Smoothing**: Apply a median filter to the final mask to produce cleaner edges.

---

## Implementation Roadmap

### Phase 1: Development
1. **Data Exploration**: Analyze the provided `PSCDL_2026.zip` development videos and metadata to determine the typical size and appearance of encroachments.
2. **Baseline Algorithm**: Implement a basic Background Subtraction $\rightarrow$ Temporal Accumulation $\rightarrow$ Masking pipeline.
3. **Tune $K$-Trigger**: Implement the logic to delay mask output by $K$ minutes.
4. **Evaluation**: Run against development ground truth masks and calculate F1 Score.

### Phase 2: Optimization
1. **Robustness**: Test against videos with varying lighting or camera shake.
2. **Speed**: Optimize the pipeline using OpenCV (GPU acceleration if possible) to ensure near real-time performance.
3. **Refine Thresholds**: Tune the persistence threshold (number of frames) and area threshold to balance precision and recall.

## Recommended Tools & Libraries
- **Language**: Python
- **Libraries**: 
    - `OpenCV`: For image processing and background subtraction.
    - `NumPy`: For efficient array manipulations (persistence maps).
    - `scikit-image`: For advanced morphological operations and CCA.
    - `PyTorch/TensorFlow` (Optional): If utilizing pre-trained models from the VL-CMU or PSCD datasets for better feature extraction.
