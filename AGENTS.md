# Agents Specification (agents.md)

This document defines the agents (modules), responsibilities, models, and data contracts for the local, CPU-only image processing pipeline.

The system is designed for:

* Old / low-power laptops
* Fully local execution
* Deterministic results
* Minimal dependencies

No cloud inference is required.

---

## Global Principles

* CPU-only execution (no CUDA)
* Models are downloaded once and reused
* One face analysis pass; downstream agents reuse outputs
* Prefer classical computer vision over large neural models
* Hugging Face is used only as a model registry, not as runtime logic
* ONNX Runtime is preferred over PyTorch where available

---

## Shared Data Contract

All agents communicate using structured Python objects (or JSON if run as subprocesses).

### FaceAnalysisResult

```python
{
  "image_path": str,
  "face_bbox": [x1, y1, x2, y2],
  "landmarks": {
    "left_eye": [x, y],
    "right_eye": [x, y],
    "nose": [x, y],
    "mouth_left": [x, y],
    "mouth_right": [x, y],
    "chin": [x, y]
  },
  "embedding": [float, ...],  # 512-D
  "face_area_ratio": float
}
```

---

## Agent 1 — Identity Grouping Agent

### Purpose

Group photos that contain the same person.

### Responsibilities

* Detect faces
* Extract face embeddings
* Extract facial landmarks
* Cluster images by identity

### Model

**InsightFace — buffalo_l**

Reasons:

* Single model provides detection, landmarks, and embeddings
* CPU-optimized
* ONNX-compatible
* No training required

### Dependencies

* insightface
* onnxruntime
* opencv-python
* numpy
* scikit-learn

### Algorithm

1. Load InsightFace model once
2. For each image:

   * Detect primary face
   * Extract landmarks and 512-D embedding
3. Cluster embeddings using DBSCAN (cosine distance)

### Output

```python
{
  "person_0": [FaceAnalysisResult, ...],
  "person_1": [FaceAnalysisResult, ...]
}
```

---

## Agent 2 — Best Photo Selection Agent

### Purpose

Select the best 2 photos per person.

### Responsibilities

* Score image quality
* Rank images per identity group
* Select top 2

### Models

No additional ML models.

### Metrics Used

* **Sharpness**: Laplacian variance
* **Face size**: face area / image area
* **Pose quality**: eye alignment symmetry
* **Expression**: mouth openness (landmark-based)

Optional prompt-based rules may adjust weights (no LLMs).

### Scoring Formula (example)

```text
score =
  0.4 * sharpness +
  0.3 * face_size +
  0.2 * pose_quality +
  0.1 * expression_score
```

### Output

```python
{
  "person_0": [best_img_1, best_img_2],
  "person_1": [best_img_1, best_img_2]
}
```

---

## Agent 3 — Portrait Cropping Agent

### Purpose

Crop standardized 13 cm × 18 cm portraits using face dimensions as reference for margins.

### Framing Approach

The cropping uses face dimensions directly to determine margins:

* **Top margin**: Half of face height above the top of the head
* **Left/Right margins**: Face width on each side of the face
* **Aspect ratio**: Maintains 13 cm width × 18 cm height ratio

### Crop Calculation

1. Extract face bounding box dimensions:
   * Face width: `right - left` from face bbox
   * Face height: `bottom - top` from face bbox

2. Calculate crop boundaries:
   * Upper bound: `top - (face_height × 0.5)`
   * Left margin: `face_width` pixels from left edge of face
   * Right margin: `face_width` pixels from right edge of face
   * Desired width: `face_width + left_margin + right_margin`
   * Desired height: Calculated to maintain 13:18 aspect ratio

3. Center horizontally on face center point

4. Clamp to image boundaries to ensure valid crop

### Output Size

* Physical dimensions: 13 cm × 18 cm at 300 DPI
* Pixel dimensions: 1535 × 2126 pixels (configurable via DPI)

### File Size Control

Images are saved with automatic quality adjustment to meet file size requirements:

* Target file size: 1000–1500 KB
* JPEG quality is automatically adjusted (50–100) to meet size constraints
* Uses binary search approach to find optimal quality setting

### Dependencies

* opencv-python
* numpy

---

## Summary

* Cropping uses anatomy-informed proportional heuristics
* "Lowest rib" is an estimated photographic reference, not anatomical detection
* The pipeline remains fast, local, and deterministic
