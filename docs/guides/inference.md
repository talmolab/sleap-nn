# Running Inference

Run predictions on videos and label files.

!!! info "Using uv workflow"
    - If using `uvx`, no installation needed
    - If using `uv sync`, prefix commands with `uv run`:
      ```bash
      uv run sleap-nn track ...
      ```

---

## Quick Start

!!! success "TL;DR - Just want predictions?"
    ```bash
    # Single model (single-instance or bottom-up)
    sleap-nn track -i video.mp4 -m models/my_model/

    # Top-down (two models required)
    sleap-nn track -i video.mp4 -m models/centroid/ -m models/centered_instance/
    ```

    **Output:** `video.mp4.predictions.slp`

    **If results aren't great**, try these common fixes:

    - Too many false detections? Add `--filter_min_instance_score 0.3`
    - Duplicate detections on same animal? Add `--filter_overlapping`
    - Missing detections? Lower `--peak_threshold 0.1`
    - Known number of animals? Add `--max_instances N`
    - Want tracking IDs? Add `--tracking`

---

## Basic Inference

```bash
sleap-nn track --data_path video.mp4 --model_paths models/my_model/
```

Output: `video.mp4.predictions.slp`

See the [CLI Reference](../reference/cli.md) for all available parameters.

### Viewing Results

After inference, you'll have a `.slp` file containing predictions. To view them:

**Option 1: Open in SLEAP GUI**

```bash
sleap predictions.slp
```

This opens the predictions in the SLEAP labeling interface where you can visualize skeletons overlaid on video frames.

**Option 2: Quick inspection with Python**

```python
import sleap_io as sio

labels = sio.load_slp("video.mp4.predictions.slp")
print(f"Frames with predictions: {len(labels)}")
print(f"Total instances: {sum(len(lf.instances) for lf in labels)}")

# Check a specific frame
lf = labels[0]
print(f"Frame {lf.frame_idx}: {len(lf.instances)} instances")
for inst in lf.instances:
    print(f"  Score: {inst.score:.3f}, Points: {inst.numpy().shape}")
```

**Option 3: Export to analysis formats**

```python
# Export to Analysis HDF5 for downstream analysis
labels.export("predictions.analysis.h5")

# Or convert to CSV/DataFrames
# See sleap-io documentation for more export options
```

---

## Essential Parameters

| Parameter | Description | Values | Default |
|-----------|-------------|--------|---------|
| `--data_path` / `-i` | Video or labels file | `PATH` | Required |
| `--model_paths` / `-m` | Model directory (repeat for top-down) | `PATH` | Required* |
| `--output_path` / `-o` | Output file path | `PATH` | `<input>.predictions.slp` |
| `--device` / `-d` | Compute device | `auto`, `cuda`, `cuda:0`, `cpu`, `mps` | `auto` |
| `--batch_size` / `-b` | Frames per batch | `INT` | `4` |
| `--max_instances` / `-n` | Max instances per frame (forward pass only) | `INT` | `None` |
| `--tracking` / `-t` | Enable tracking | Flag | `False` |
| `--peak_threshold` | Min confidence for peaks | `FLOAT` | `0.2` |

*Not required for track-only mode.

!!! tip "Device selection"
    The `--device` parameter accepts:

    - `auto` - Automatically detect best available device
    - `cuda` - Use default CUDA GPU
    - `cuda:0`, `cuda:1`, etc. - Use a specific GPU by index
    - `cpu` - Use CPU
    - `mps` - Use Apple Metal (macOS)

!!! note "Max instances"
    The `--max_instances` / `-n` parameter is only applied during the model forward pass. When running in **track-only mode** (without model inference), this parameter has no effect since no peak detection is performed.

For all parameters including image pre-processing and data selection options, see the [CLI Reference](../reference/cli.md).

---

## Model Types

### Single Instance

For videos with exactly one animal:

```bash
sleap-nn track -i video.mp4 -m models/single_instance/
```

**How parameters apply:**

- `--peak_threshold`: Keypoints below threshold become **NaN** (instance still exists, but with missing keypoints)
- `--max_instances`: Not applicable (always outputs one instance)
- Post-processing filters: Generally not needed, but can filter out frames with poor detection

### Bottom-Up

For multi-animal videos using part affinity fields:

```bash
sleap-nn track -i video.mp4 -m models/bottomup/
```

**How parameters apply:**

- `--peak_threshold`: Peaks below threshold are **not detected** and won't be used for PAF grouping. Lower values detect more peaks but may create spurious instances.
- `--max_instances`: Limits instances **after** PAF grouping. Keeps the top N instances by instance score.
- Post-processing filters: Particularly useful for bottom-up models to clean up the output:
    - Use overlap filter (`--filter_overlapping`) to remove duplicate detections
    - Use node count filter to remove partial instances from grouping errors

!!! note "When max_instances applies in bottom-up"
    In bottom-up models, `--max_instances` is applied **after** PAF grouping assembles instances (not during peak detection). All peaks are detected, PAF grouping creates instances, then the top N instances by score are kept.

### Top-Down

For multi-animal videos using centroid + centered instance approach:

```bash
sleap-nn track -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/
```

**How parameters apply:**

- `--peak_threshold`: Applies to **both** the centroid model and the centered instance model, but behaves differently:
    - **Centroid model**: Centroids below threshold are **removed entirely** (no instance created)
    - **Centered instance model**: Keypoints below threshold become **NaN** (instance still exists, but with missing keypoints)
- `--max_instances`: Limits the number of centroids (and thus instances). Takes the top N centroids by confidence score.
- Post-processing filters: Apply to assembled instances after centered instance prediction

!!! tip "Controlling instance count in top-down models"
    You have two ways to control how many instances are detected:

    1. **`--max_instances N`**: Hard limit on centroids. Only the top N highest-confidence centroids generate instances.
    2. **`--peak_threshold`**: Soft limit. Centroids below this confidence are not detected.

    Use `--max_instances` when you know exactly how many animals are in the video. Use `--peak_threshold` when the number varies.

!!! tip "Avoiding NaN keypoints in top-down predictions"
    When `--peak_threshold` is too high, good instances may have some keypoints set to NaN (below threshold).

    **Solution:** Keep `--peak_threshold` low enough to detect all keypoints, then use `--filter_min_instance_score` to remove low-quality instances entirely. This preserves complete skeletons for good instances.

    ```bash
    # Instead of raising peak_threshold (which creates NaN keypoints)
    sleap-nn track -i video.mp4 -m models/centroid/ -m models/ci/ \
        --peak_threshold 0.1 \
        --filter_min_instance_score 0.3
    ```

---

## Filtering Instances

Post-processing filters remove low-quality or duplicate predictions before tracking.

### Node Count Filter

Remove instances with too few detected keypoints:

```bash
# Require at least 3 visible nodes
sleap-nn track -i video.mp4 -m models/ --filter_min_visible_nodes 3

# Require at least 50% of skeleton nodes to be visible
sleap-nn track -i video.mp4 -m models/ --filter_min_visible_node_fraction 0.5
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter_min_visible_nodes` | Minimum number of visible keypoints | `0` (disabled) |
| `--filter_min_visible_node_fraction` | Minimum fraction of skeleton nodes | `0.0` (disabled) |

### Confidence Score Filter

Remove instances with low confidence scores:

```bash
# Require mean node confidence >= 0.4
sleap-nn track -i video.mp4 -m models/ --filter_min_mean_node_score 0.4

# Require instance score >= 0.3
sleap-nn track -i video.mp4 -m models/ --filter_min_instance_score 0.3
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter_min_mean_node_score` | Minimum mean confidence across visible nodes | `0.0` (disabled) |
| `--filter_min_instance_score` | Minimum overall instance score | `0.0` (disabled) |

!!! note "Instance score differs by model type"
    The instance score comes from different sources depending on model type:

    - **Top-down**: Instance score is the **centroid confidence** (how confident the model was that an animal exists at that location)
    - **Bottom-up**: Instance score is derived from **PAF grouping** (how well the keypoints connected together)

### Overlap Filter

Remove duplicate detections with greedy NMS:

```bash
# Enable with default IOU
sleap-nn track -i video.mp4 -m models/ --filter_overlapping

# Use OKS with custom threshold
sleap-nn track -i video.mp4 -m models/ \
    --filter_overlapping \
    --filter_overlapping_method oks \
    --filter_overlapping_threshold 0.5
```

| Method | Description |
|--------|-------------|
| `iou` | Bounding box intersection-over-union |
| `oks` | Object Keypoint Similarity (pose-aware) |

| Threshold | Effect |
|-----------|--------|
| 0.3 | Aggressive filtering |
| 0.5 | Moderate |
| 0.8 | Permissive (default) |

### Combining Filters

All filters can be combined. They are applied in order: node count → confidence → overlap.

**Example: Strict filtering for clean output**

```bash
sleap-nn track -i video.mp4 -m models/ \
    --filter_min_visible_nodes 2 \
    --filter_min_visible_node_fraction 0.25 \
    --filter_min_mean_node_score 0.3 \
    --filter_overlapping \
    --filter_overlapping_threshold 0.5
```

### Common Use Cases

**Use case 1: Known number of animals (top-down)**

You know there are exactly 3 mice in the video:

```bash
sleap-nn track -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/ \
    --max_instances 3
```

**Use case 2: Remove false positives (bottom-up)**

Your bottom-up model produces spurious partial detections:

```bash
sleap-nn track -i video.mp4 -m models/bottomup/ \
    --filter_min_visible_node_fraction 0.5 \
    --filter_min_instance_score 0.3
```

**Use case 3: Crowded scenes with overlapping animals**

Animals frequently overlap, causing duplicate detections:

```bash
sleap-nn track -i video.mp4 -m models/ \
    --filter_overlapping \
    --filter_overlapping_method oks \
    --filter_overlapping_threshold 0.4
```

**Use case 4: High-quality predictions only (top-down)**

Keep all keypoints but remove low-confidence instances:

```bash
sleap-nn track -i video.mp4 \
    -m models/centroid/ \
    -m models/centered_instance/ \
    --peak_threshold 0.1 \
    --filter_min_instance_score 0.4 \
    --filter_min_visible_node_fraction 0.75
```

!!! note "Inference only"
    Filters are only applied during inference. When running in **track-only mode** (without model paths), these parameters have no effect.

---

## Processing Order

!!! abstract "Deep dive"
    This section explains the internal pipeline in detail. **You don't need to read this to use inference** - the [Quick Start](#quick-start) and [Troubleshooting](#troubleshooting) sections cover most use cases. Read on if you want to understand exactly how parameters interact.

When running inference, operations are applied in a specific order. Understanding this order helps you choose the right parameters to control your predictions.

### Overview

```
1. Model Forward Pass
   ├── peak_threshold: Filter low-confidence peaks
   └── max_instances: Limit instances by score
       ├── Top-down: limits centroids before instance prediction
       └── Bottom-up: limits instances after PAF grouping

2. Post-processing Filters (in order)
   ├── Node count filter
   ├── Confidence score filter
   └── Overlap filter (NMS)

3. Tracking
   └── Instance identity assignment across frames
```

### Step 1: Model Forward Pass

The first filtering happens during the model's peak detection phase.

**`--peak_threshold`** (default: 0.2)

- Filters out peaks with confidence values below this threshold
- Applied during peak detection, before instances are assembled
- **Effect depends on what's being detected:**
    - **Centroids** (top-down): Below-threshold centroids are **removed** — no instance is created
    - **Keypoints**: Below-threshold keypoints become **NaN** — instance exists but with missing points

**`--max_instances`** (default: unlimited)

- Limits the number of instances per frame based on confidence score
- Keeps the top N highest-scoring instances, discards the rest
- **Top-down**: Limits centroids before centered instance model runs
- **Bottom-up**: Limits instances after PAF grouping assembles them
- **Single instance**: Not applicable (always outputs one instance)

### Step 2: Post-processing Filters

After instances are assembled, post-processing filters are applied **in this order**:

1. **Node Count Filter** → removes instances with too few keypoints
2. **Confidence Score Filter** → removes low-confidence instances
3. **Overlap Filter (NMS)** → removes duplicate detections

See [Filtering Instances](#filtering-instances) for detailed parameter documentation and examples.

### Step 3: Tracking

After filtering, tracking assigns consistent identities across frames. See [Tracking](tracking.md) for details.

!!! note "Why filtering happens before tracking"
    Filtering is applied **before** tracking to prevent spurious track creation. If low-quality instances were tracked first and then filtered out, their track IDs would be lost, causing track switches in subsequent frames.

In **track-only mode** (no model paths), filtering is still applied before tracking on existing predictions.

---

## Python API

### Basic Usage

```python
from sleap_nn.predict import run_inference

labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/my_model/"],
    output_path="predictions.slp",
    make_labels=True,
)
```

### Get Raw Outputs

```python
results = run_inference(
    data_path="video.mp4",
    model_paths=["models/my_model/"],
    make_labels=False,
    return_confmaps=True,
)
```

---

## Provenance Metadata

Output files include metadata about how predictions were generated:

```python
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
provenance = labels.provenance

print(f"sleap-nn version: {provenance.get('sleap_nn_version')}")
print(f"Model type: {provenance.get('model_type')}")
print(f"Runtime: {provenance.get('runtime_sec')}s")
```

### Recorded Information

| Category | Fields |
|----------|--------|
| **Timestamps** | `timestamp_start`, `timestamp_end`, `runtime_sec` |
| **Versions** | `sleap_nn_version`, `sleap_io_version`, `torch_version` |
| **Model** | `model_paths`, `model_type`, `head_type` |
| **Input** | `source_path`, `source_video_paths` |
| **Config** | `peak_threshold`, `batch_size`, `max_instances` |
| **System** | `device`, `python_version`, `cuda_version`, `gpu_names` |

---

## Legacy SLEAP Model Support

Run inference with SLEAP <=v1.4 models (UNet only):

```bash
sleap-nn track -i video.mp4 -m /path/to/sleap_model/
```

The directory should contain:
- `best_model.h5`
- `training_config.json`

!!! warning "UNet only"
    Only UNet backbone models from SLEAP ≤1.4 are supported.

---

## Troubleshooting

### No Predictions / Empty Output

??? question "Getting zero predictions on all frames"
    **Possible causes and solutions:**

    1. **Peak threshold too high** - The model is detecting peaks but they're being filtered out
       ```bash
       sleap-nn track -i video.mp4 -m models/ --peak_threshold 0.05
       ```

    2. **Wrong model type** - Using a single-instance model on multi-animal video (or vice versa)
       - Check your training config to confirm model type matches your data

    3. **Preprocessing mismatch** - Video has different properties than training data
       - Check if training used grayscale vs RGB: `--ensure_grayscale` or `--ensure_rgb`
       - Check if training used different input scaling: `--input_scale 0.5`

    4. **Model didn't train properly** - Check training loss curves and validation metrics

??? question "Getting predictions on some frames but not others"
    This is usually normal - frames without confident detections won't have predictions.

    To get predictions on more frames:
    ```bash
    sleap-nn track -i video.mp4 -m models/ --peak_threshold 0.1
    ```

    To keep empty frames in output (useful for analysis):
    ```bash
    # Empty frames are kept by default; use --no_empty_frames to remove them
    ```

### Too Many Predictions

??? question "Getting many false positive detections"
    **Solution 1:** Filter by instance confidence score
    ```bash
    sleap-nn track -i video.mp4 -m models/ --filter_min_instance_score 0.3
    ```

    **Solution 2:** Filter by number of visible keypoints
    ```bash
    sleap-nn track -i video.mp4 -m models/ --filter_min_visible_node_fraction 0.5
    ```

    **Solution 3:** If you know the exact number of animals
    ```bash
    sleap-nn track -i video.mp4 -m models/ --max_instances 3
    ```

??? question "Getting duplicate detections on the same animal"
    Enable overlap filtering with NMS:
    ```bash
    # Basic overlap filter
    sleap-nn track -i video.mp4 -m models/ --filter_overlapping

    # More aggressive filtering
    sleap-nn track -i video.mp4 -m models/ \
        --filter_overlapping \
        --filter_overlapping_threshold 0.5

    # Pose-aware filtering (better for overlapping animals)
    sleap-nn track -i video.mp4 -m models/ \
        --filter_overlapping \
        --filter_overlapping_method oks
    ```

### Too Few Predictions

??? question "Missing animals that are clearly visible"
    **Solution 1:** Lower the peak threshold
    ```bash
    sleap-nn track -i video.mp4 -m models/ --peak_threshold 0.1
    ```

    **Solution 2:** For top-down models, check if centroids are being detected
    - The centroid model might be missing animals
    - Try lowering peak threshold significantly: `--peak_threshold 0.05`

??? question "Predictions have missing keypoints (NaN values)"
    This happens when individual keypoint confidence is below the peak threshold.

    **Solution:** Lower peak threshold and filter by instance score instead
    ```bash
    sleap-nn track -i video.mp4 -m models/ \
        --peak_threshold 0.1 \
        --filter_min_instance_score 0.3
    ```

    This detects more keypoints while still removing low-quality instances.

### Wrong Predictions

??? question "Predictions are in the wrong location / shifted"
    **Possible causes:**

    1. **Input scaling mismatch** - Training used different scale than inference
       ```bash
       # Check training config for scale value, then match it
       sleap-nn track -i video.mp4 -m models/ --input_scale 0.5
       ```

    2. **Crop size mismatch** (top-down only)
       ```bash
       # Check training config for crop_hw value
       sleap-nn track -i video.mp4 -m models/ --crop_size 256
       ```

??? question "Skeletons are jumbled / keypoints assigned to wrong body parts"
    This usually indicates a model training issue rather than inference settings.

    - Check if training data had consistent labeling
    - Verify skeleton definition matches between training and inference
    - Consider retraining with more data or data augmentation

### Performance Issues

??? question "Out of GPU memory (CUDA OOM)"
    **Solution 1:** Reduce batch size
    ```bash
    sleap-nn track -i video.mp4 -m models/ --batch_size 2
    ```

    **Solution 2:** Use CPU (slower but no memory limit)
    ```bash
    sleap-nn track -i video.mp4 -m models/ --device cpu
    ```

    **Solution 3:** Process fewer frames at once
    ```bash
    sleap-nn track -i video.mp4 -m models/ --frames 0-1000
    ```

??? question "Inference is very slow"
    **Check GPU is being used:**
    ```bash
    sleap-nn system  # Verify CUDA is available
    sleap-nn track -i video.mp4 -m models/ --device cuda
    ```

    **Increase batch size** (if memory allows):
    ```bash
    sleap-nn track -i video.mp4 -m models/ --batch_size 16
    ```

    **For production speed**, consider [ONNX/TensorRT export](export.md).

??? question "Progress bar not moving / seems stuck"
    - Large videos take time to process - check GPU utilization with `nvidia-smi`
    - First batch may be slow due to model compilation
    - Try a smaller test: `--frames 0-100`

---

## Next Steps

- [:octicons-arrow-right-24: Evaluation](evaluation.md) - Assess model performance
- [:octicons-arrow-right-24: Tracking](tracking.md) - Assign IDs across frames
- [:octicons-arrow-right-24: Export](export.md) - Deploy models
