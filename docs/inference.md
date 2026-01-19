# Running Inference

SLEAP-NN provides powerful inference capabilities for pose estimation with support for multiple model types, tracking, and legacy SLEAP model compatibility. SLEAP-NN supports inference on:

- **Videos**: Any format supported by OpenCV
- **SLEAP Labels**: `.slp` files

!!! info "Using uv workflow"
    This section assumes you have `sleap-nn` installed. If not, refer to the [installation guide](installation.md).
    
    - If you're using the `uvx` workflow, you do **not** need to install anything. (See [installation using uvx](installation.md#installation-using-uvx) for more details.)
    
    - If you are using `uv sync` or `uv add` installation methods, add `uv run` as a prefix to all CLI commands shown below, for example:

          `uv run sleap-nn track ...`

## Run Inference with CLI

For a quick reference of all CLI options, see the [CLI Reference](cli.md#sleap-nn-track).


```bash
sleap-nn track \
    --data_path video.mp4 \
    --model_paths models/ckpt_folder/
```

To run inference on a specific cuda device (say cuda:0 - first gpu),
```bash
sleap-nn track \
    --data_path video.mp4 \
    --model_paths models/ckpt_folder/
    --device cuda:0
```

To run inference on video files with specific frames
```bash
sleap-nn track \
    --data_path video.mp4 \
    --frames "0-100,200-300" \
    --model_paths models/ckpt_folder/
```

To run inference with different backbone weights than the one in `models/ckpt_folder/`
```bash
sleap-nn track \
    --data_path video.mp4 \
    --frames "0-100,200-300" \
    --model_paths models/ckpt_folder/ \
    --backbone_ckpt_path models/backbone.ckpt
```

!!! info "Inference on TopDown models"

    For two-stage models (topdown and multiclass topdown), both the centroid and centered-instance model ckpts should be provided as given below:
        ```bash
        sleap-nn track \
            --data_path video.mp4 \
            --model_paths models/centroid_unet/ \
            --model_paths models/centered_instance_unet/
        ```

!!! note "Centroid-only / Centered-instance-only inference"
    You can run inference using only the centroid model or only the centered-instance model. However, both modes require ground-truth (GT) instances to be present in the `.slp` file, so this type of inference can only be performed on datasets that already have GT labels.
    
    **_Note_**: In centroid-only inference, each predicted centroid (single-keypoint) is matched (by Euclidean distance) to the nearest ground-truth instance, and the ground-truth keypoints are copied for display (thus, the pipeline always expects ground-truth pose data to be available when running inference on just the centroid model). As a result, an OKS mAP of 1.0 for a centroid model simply means all instances were detected—it does **not** measure pose/keypoint accuracy. To evaluate keypoints, run the second stage (the pose model) instead of centroid-only inference.

### Arguments for CLI

#### Essential Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` / `-i` | Path to `.slp` file or `.mp4` to run inference on | Required |
| `--model_paths` / `-m` | List of paths to the directory where the best.ckpt and training_config.yaml are saved | Required |
| `--output_path` / `-o` | The output filename to use for the predicted data. If not provided, defaults to '[data_path].slp' | `<data_path>.predictions.slp` |
| `--device` / `-d` | Device on which torch.Tensor will be allocated. One of ('cpu', 'cuda', 'mps', 'auto', 'opencl', 'ideep', 'hip', 'msnpu'). Default: 'auto' (based on available backend either cuda, mps or cpu is chosen) | `auto` |
| `--batch_size` / `-b` | Number of frames to predict at a time. Larger values result in faster inference speeds, but require more memory | `4` |
| `--max_instances` / `-n` | Limit maximum number of instances in multi-instance models. Not available for ID models | `None` |
| `--tracking` / `-t` | If True, runs tracking on the predicted instances | `False` |
| `--peak_threshold` | Minimum confidence map value to consider a peak as valid | `0.2` |
| `--integral_refinement` | If `None`, returns the grid-aligned peaks with no refinement. If `'integral'`, peaks will be refined with integral regression. Default: 'integral'. | `integral` |

#### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backbone_ckpt_path` | To run inference on any `.ckpt` other than `best.ckpt` from the `model_paths` dir, the path to the `.ckpt` file should be passed here | `None` |
| `--head_ckpt_path` | Path to `.ckpt` file if a different set of head layer weights are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt from `backbone_ckpt_path` if provided) | `None` |

#### Image Pre-processing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_height` | Maximum height the image should be padded to. If not provided, the values from the training config are used. Default: None. | `None` |
| `--max_width` | Maximum width the image should be padded to. If not provided, the values from the training config are used. Default: None. | `None` |
| `--input_scale` | Scale factor to apply to the input image. If not provided, the values from the training config are used. Default: None. | `None` |
| `--ensure_rgb` | True if the input image should have 3 channels (RGB image). If input has only one channel when this is set to `True`, then the images from single-channel is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. If not provided, the values from the training config are used. Default: `None`. | `False` |
| `--ensure_grayscale` | True if the input image should only have a single channel. If input has three channels (RGB) and this is set to True, then we convert the image to grayscale (single-channel) image. If the source image has only one channel and this is set to False, then we retain the single channel input. If not provided, the values from the training config are used. Default: `None`. | `False` |
| `--crop_size` | Crop size in **original image coordinates**. The crop is extracted first at this size, then resized by `input_scale` if provided. If not provided, the crop size from training_config.yaml is used. | `None` |
| `--anchor_part` | The node name to use as the anchor for the centroid. If not provided, the anchor part in the `training_config.yaml` is used. Default: `None`. | `None` |

!!! warning "Breaking Change in v0.1.0: crop_size semantics"
    Prior to v0.1.0, `crop_size` referred to the crop region size after scaling (resize-then-crop).

    **New behavior**: `crop_size` now represents the size in **original image coordinates**—the image is cropped first at this size, then resized by `input_scale`.

    **Migration**: If you were using `crop_size=128` with `input_scale=0.5`, you may need to use `crop_size=256` to maintain the same effective crop region.

#### Data Selection

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--only_labeled_frames` | `True` if inference should be run only on user-labeled frames | `False` |
| `--only_suggested_frames` | `True` if inference should be run only on unlabeled suggested frames | `False` |
| `--video_index` | Integer index of video in .slp file to predict on. To be used with an .slp path as an alternative to specifying the video path | `None` |
| `--video_dataset` | The dataset for HDF5 videos | `None` |
| `--video_input_format` | The input_format for HDF5 videos | `channels_last` |
| `--frames` | List of frames indices. If `None`, all frames in the video are used | All frames |
| `--no_empty_frames` | If `True`, removes frames with no predicted instances from the output labels | `False` |


#### Performance

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--queue_maxsize` | Maximum size of the frame buffer queue | `8` |

#### Filtering Overlapping Instances

SLEAP-NN can filter out duplicate/overlapping predictions after inference using greedy non-maximum suppression (NMS). This is useful for removing redundant detections without enabling full tracking.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter_overlapping` | Enable filtering of overlapping instances after inference | `False` |
| `--filter_overlapping_method` | Similarity metric: `iou` (bounding box) or `oks` (keypoint-based) | `iou` |
| `--filter_overlapping_threshold` | Similarity threshold above which instances are considered duplicates | `0.8` |

**Methods:**

- **`iou`**: Uses bounding box intersection-over-union. Fast and position-based.
- **`oks`**: Uses Object Keypoint Similarity. Pose-aware, considers keypoint distances.

**Example usage:**

```bash
# Enable filtering with default IOU method
sleap-nn track -i video.mp4 -m model/ --filter_overlapping

# Use OKS method with custom threshold
sleap-nn track -i video.mp4 -m model/ \
    --filter_overlapping \
    --filter_overlapping_method oks \
    --filter_overlapping_threshold 0.5
```

**Threshold guidelines:**

| Value | Effect |
|-------|--------|
| 0.3 | Aggressive - removes instances with >30% similarity |
| 0.5 | Moderate - balanced filtering |
| 0.8 | Permissive (default) - only removes highly similar instances |

!!! note "Filtering vs Tracking"
    This filtering is independent of tracking and runs before the tracking step. You can use both together—filtering removes duplicates first, then tracking assigns IDs to remaining instances.


## Run inference with API

To return predictions as list of dictionaries:

```python linenums="1"
from sleap_nn.predict import run_inference

# Run inference
labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/unet/"],
    make_labels=False,
    return_confmaps=True
)
```
Setting `return_confmaps=True` will also return the raw confidence maps generated by the model, allowing you to inspect or analyze the model's outputs alongside the predicted poses.

To return predictions as a `sleap_io.Labels` object, 

```python linenums="1"
from sleap_nn.predict import run_inference

# Run inference
labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/unet/"],
    output_path="predictions.slp",
    make_labels=True,
)
```

For a detailed explanation of all available parameters for `run_inference`, see the API docs for [run_inference()](../api/predict/#sleap_nn.predict.run_inference).

## Provenance Metadata

Output prediction files (`.slp`) now include comprehensive provenance metadata that records how predictions were generated. This is useful for reproducibility and debugging.

### Accessing Provenance

```python linenums="1"
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
| **Input** | `source_path`, `source_video_paths`, `input_provenance` |
| **Frames** | `frame_selection_method`, `frames_predicted`, `total_frames` |
| **Config** | `peak_threshold`, `batch_size`, `integral_refinement`, `max_instances` |
| **System** | `device`, `python_version`, `cuda_version`, `gpu_names` |
| **Tracking** | `tracker_method`, `tracker_window`, etc. (when tracking enabled) |

!!! note "Input Provenance Preservation"
    If the input was an SLP file with existing provenance, that information is preserved in the `input_provenance` field, maintaining the full chain of processing history.

## Tracking

SLEAP-NN provides advanced tracking capabilities for multi-instance pose estimation. When running inference, the output labels object (or `.slp` file) will include track IDs assigned to the predicted instances (or user-labeled instances). 

When using the `sleap-nn track` CLI command with both `--model_paths` and `--tracking` specified, the tool will perform both pose prediction and track assignment in a single step—automatically generating pose predictions and associating them into tracks.

### Tracking Methods

#### Tracking Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tracking` / `-t` | If True, runs tracking on the predicted instances | `False` |
| `--tracking_window_size` | Number of frames to look for in the candidate instances to match with the current detections | `5` |
| `--min_new_track_points` | We won't spawn a new track for an instance with fewer than this many points | `0` |
| `--candidates_method` | Either of `fixed_window` or `local_queues`. In fixed window method, candidates from the last `window_size` frames. In local queues, last `window_size` instances for each track ID is considered for matching against the current detection | `fixed_window` |
| `--min_match_points` | Minimum non-NaN points for match candidates | `0` |
| `--features` | Feature representation for the candidates to update current detections. One of [`keypoints`, `centroids`, `bboxes`, `image`] | `keypoints` |
| `--scoring_method` | Method to compute association score between features from the current frame and the previous tracks. One of [`oks`, `cosine_sim`, `iou`, `euclidean_dist`] | `oks` |
| `--scoring_reduction` | Method to aggregate and reduce multiple scores if there are several detections associated with the same track. One of [`mean`, `max`, `robust_quantile`] | `mean` |
| `--robust_best_instance` | If the value is between 0 and 1 (excluded), use a robust quantile similarity score for the track. If the value is 1, use the max similarity (non-robust). For selecting a robust score, 0.95 is a good value | `1.0` |
| `--track_matching_method` | Track matching algorithm. One of `hungarian`, `greedy` | `hungarian` |
| `--max_tracks` | Maximum number of new tracks to be created to avoid redundant tracks (only for local queues candidate) | `None` |
| `--use_flow` | If True, `FlowShiftTracker` is used, where the poses are matched using optical flow shifts | `False` |
| `--of_img_scale` | Factor to scale the images by when computing optical flow. Decrease this to increase performance at the cost of finer accuracy. Sometimes decreasing the image scale can improve performance with fast movements (only if `use_flow` is True) | `1.0` |
| `--of_window_size` | Optical flow window size to consider at each pyramid scale level (only if `use_flow` is True) | `21` |
| `--of_max_levels` | Number of pyramid scale levels to consider. This is different from the scale parameter, which determines the initial image scaling (only if `use_flow` is True) | `3` |
| `--post_connect_single_breaks` | If True and `max_tracks` is not None with local queues candidate method, connects track breaks when exactly one track is lost and exactly one new track is spawned in the frame | `False` |

!!! warning "Tracking cleaning and pre-cull parameters"

    The parameters `--tracking_pre_cull_to_target`, `--tracking_target_instance_count`, `tracking_pre_cull_iou_threshold`, `tracking_clean_iou_threshold` and `--tracking_clean_instance_count` are provided for backwards compatibility with legacy SLEAP workflows and **may be deprecated in future releases**. 

    - To restrict the number of instances per frame, use the `--max_instances` parameter, which selects the top instances with the highest prediction scores.

    We recommend using `--max_instances` for controlling the number of predicted instances per frame in new projects.

#### Fixed Window Tracking

This method maintains a fixed-size queue with the last N frames and uses all instances from those frames as candidates for matching.

```bash
sleap-nn track \
    -i video.mp4 \
    -m models/bottomup_unet/ \
    -t \
    --candidates_method fixed_window \
    --tracking_window_size 10
```

#### Local Queues Tracking

This method maintains separate queues for each track ID, keeping the last N instances per track. It's more robust to track breaks but requires more memory and computation.

```bash
sleap-nn track \
    -i video.mp4 \
    -m models/bottomup_unet/ \
    -t \
    --candidates_method local_queues \
    --tracking_window_size 5
```

#### Optical Flow Tracking

This method uses optical flow to shift the candidates onto the frame to be tracked and then associates the untracked instances to the shifted instances.

```bash
sleap-nn track \
    -i video.mp4 \
    -m models/bottomup_unet/ \
    -t \
    --use_flow
```

### Track-only workflow
You can perform tracking on existing user-labeled instances—without running inference to get new predictions—by enabling tracking (`--tracking`) and omitting the `--model-paths` argument.

```bash
sleap-nn track \
    -i labels.slp \
    -t \
```

To run track-only workflow on select frame indices and video index in a slp file:

```bash
sleap-nn track \
    -i labels.slp \
    -t \
    --frames 0-100 \
    --video_index 0
```

## Legacy SLEAP Model Support

SLEAP-NN supporting running inference with models trained using the original SLEAP TensorFlow/Keras backend (sleap <= 1.4), **but only for UNet backbone models**.
To run inference with legacy SLEAP models (trained with TensorFlow/Keras, sleap ≤ 1.4), simply use the same CLI or API workflows described above in the [With CLI](#with-cli) and [With API](#with-api) sections. Just provide the path to the directory containing both `best_model.h5` and `training_config.json` from your SLEAP training run as your `model_paths` argument. SLEAP-NN will automatically detect and load these legacy models for inference.

!!! warning "Legacy support Limitation"
    **Currently, only UNet backbone models are supported for inference when converting legacy SLEAP (≤1.4) models.**  

## Evaluation Metrics

SLEAP-NN provides comprehensive evaluation capabilities to assess model performance against ground truth labels.

Using CLI:
```bash
sleap-nn eval \
    --ground_truth_path gt_labels.slp \
    --predicted_path pred_labels.slp \
    --save_metrics pred_metrics.npz \
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--ground_truth_path` / `-g` | Path to ground truth labels file (.slp) | Required |
| `--predicted_path` / `-p` | Path to predicted labels file (.slp) | Required |
| `--save_metrics` / `-s` | Path to save metrics (.npz file) | `None` |
| `--oks_stddev` | Standard deviation for OKS calculation | `0.05` |
| `--oks_scale` | Scale factor for OKS calculation | `None` |
| `--match_threshold` | Threshold for instance matching | `0.0` |
| `--user_labels_only` | Only evaluate user-labeled frames | `False` |

Using `Evaluator` API:
```python linenums="1"
import sleap_io as sio
from sleap_nn.evaluation import Evaluator

# Load ground truth and predictions
gt_labels = sio.load_slp("gt_labels.slp")
pred_labels = sio.load_slp("pred_labels.slp")

# Create evaluator and compute metrics
evaluator = Evaluator(gt_labels, pred_labels)
metrics = evaluator.evaluate()

# Print results
print(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']}")
print(f"mOKS: {metrics['mOKS']}")
print(f"Dist. error @90th percentile: {metrics['distance_metrics']['p90']}")
```



## Troubleshooting

### Common Issues

#### Out of Memory
- Reduce `--batch_size`
- Use `--device cpu` for CPU-only inference

#### Slow Inference
- Increase `--batch_size` (if memory allows)
- Use `--device cuda` for GPU acceleration

#### Poor Predictions
- Adjust `--peak_threshold`
- Check model compatibility
- Verify input preprocessing matches training

#### Tracking Issues
- Adjust `--tracking_window_size`
- Try different feature representation and scoring method.


## Next Steps

- [Training Models](training.md)
- [Configuration Guide](config.md)
- [Model Architectures](models.md)
- [API Reference](api/index.md)
