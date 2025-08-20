# Running Inference

SLEAP-NN provides powerful inference capabilities for pose estimation with support for multiple model types, tracking, and legacy SLEAP model compatibility. SLEAP-NN supports inference on:

- **Videos**: Any format supported by OpenCV
- **SLEAP Labels**: `.slp` files

## Run Inference with CLI

```bash
sleap-nn-track \
    --data_path video.mp4 \
    --model_paths models/ckpt_folder/
```

To run inference on video files with specific frames
```bash
sleap-nn-track \
    --data_path video.mp4 \
    --frames "1-100,200-300" \
    --model_paths models/ckpt_folder/
```

To run inference with different backbone weights than the one in `models/ckpt_folder/`
```bash
sleap-nn-track \
    --data_path video.mp4 \
    --frames "1-100,200-300" \
    --model_paths models/ckpt_folder/ \
    --backbone_ckpt_path models/backbone.ckpt
```

For two-stage models (topdown and multiclass topdown), both the centroid and centered-instance model ckpts should be provided as given below:
```bash
sleap-nn-track \
    --data_path video.mp4 \
    --model_paths models/centroid_unet/ \
    --model_paths models/centered_instance_unet/
```

!!! note
    **_Note (for centroid-only inference)_**: The centroid model is essentially the first stage of TopDown model workflow, which only predicts centers, not keypoints. In centroid-only inference, each predicted centroid is matched (by Euclidean distance) to the nearest ground-truth instance, and the ground-truth keypoints are copied for display. Therefore, an OKS mAP of 1.0 just means all instances were detected—it does not reflect pose/keypoint accuracy. To evaluate keypoints, run the second stage (the pose model) rather than centroid-only inference.

### Arguments for CLI

#### Essential Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_path` | Path to `.slp` file or `.mp4` to run inference on | Required |
| `--model_paths` | List of paths to the directory where the best.ckpt and training_config.yaml are saved | Required |
| `--output_path` | The output filename to use for the predicted data. If not provided, defaults to '[data_path].slp' | `[input].slp` |
| `--device` | Device on which torch.Tensor will be allocated. One of ('cpu', 'cuda', 'mps', 'auto', 'opencl', 'ideep', 'hip', 'msnpu'). Default: 'auto' (based on available backend either cuda, mps or cpu is chosen) | `auto` |
| `--batch_size` | Number of frames to predict at a time. Larger values result in faster inference speeds, but require more memory | `4` |
| `--peak_threshold` | Minimum confidence map value to consider a peak as valid | `0.2` |
| `--integral_refinement` | If `None`, returns the grid-aligned peaks with no refinement. If `'integral'`, peaks will be refined with integral regression. Default: 'integral'. | `integral` |

#### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--backbone_ckpt_path` | To run inference on any `.ckpt` other than `best.ckpt` from the `model_paths` dir, the path to the `.ckpt` file should be passed here | `None` |
| `--head_ckpt_path` | Path to `.ckpt` file if a different set of head layer weights are to be used. If `None`, the `best.ckpt` from `model_paths` dir is used (or the ckpt from `backbone_ckpt_path` if provided) | `None` |
| `--max_instances` | Limit maximum number of instances in multi-instance models. Not available for ID models | `None` |

#### Image Pre-processing

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--max_height` | Maximum height the image should be padded to. If not provided, the values from the training config are used. Default: None. | `None` |
| `--max_width` | Maximum width the image should be padded to. If not provided, the values from the training config are used. Default: None. | `None` |
| `--input_scale` | Scale factor to apply to the input image. If not provided, the values from the training config are used. Default: None. | `None` |
| `--ensure_rgb` | True if the input image should have 3 channels (RGB image). If input has only one channel when this is set to `True`, then the images from single-channel is replicated along the channel axis. If the image has three channels and this is set to False, then we retain the three channels. If not provided, the values from the training config are used. Default: `None`. | `False` |
| `--ensure_grayscale` | True if the input image should only have a single channel. If input has three channels (RGB) and this is set to True, then we convert the image to grayscale (single-channel) image. If the source image has only one channel and this is set to False, then we retain the single channel input. If not provided, the values from the training config are used. Default: `None`. | `False` |
| `--crop_size` | Crop size. If not provided, the crop size from training_config.yaml is used | `None` |
| `--anchor_part` | The node name to use as the anchor for the centroid. If not provided, the anchor part in the `training_config.yaml` is used. Default: `None`. | `None` |

#### Data Selection

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--only_labeled_frames` | `True` if inference should be run only on user-labeled frames | `False` |
| `--only_suggested_frames` | `True` if inference should be run only on unlabeled suggested frames | `False` |
| `--video_index` | Integer index of video in .slp file to predict on. To be used with an .slp path as an alternative to specifying the video path | `None` |
| `--video_dataset` | The dataset for HDF5 videos | `None` |
| `--video_input_format` | The input_format for HDF5 videos | `channels_last` |
| `--frames` | List of frames indices. If `None`, all frames in the video are used | All frames |

#### Performance

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--queue_maxsize` | Maximum size of the frame buffer queue | `8` |

#### Tracking Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--tracking` | If True, runs tracking on the predicted instances | `False` |
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


## Run inference with API

To return predictions as list of dictionaries, 

```python
from sleap_nn.predict import run_inference

# Run inference
labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/unet/"],
    make_labels=False,
    return_confmaps=True
)
```

To return predictions as a `sleap_io.Labels` object, 

```python
from sleap_nn.predict import run_inference

# Run inference
labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/unet/"],
    output_path="predictions.slp",
    make_labels=True,
)
```

For a detailed explanation of all available parameters for `run_inference`, see the API docs for [run_inference()](../api/inference/predictors/#sleap_nn.inference.predictors.run_inference).

## Tracking

SLEAP-NN includes sophisticated tracking capabilities for multi-instance scenarios. The output labels object (or `.slp` file) will have track IDs associated to the predicted instances (or user-labeled instances).

### Tracking Methods

#### Fixed Window Tracking

This method maintains a fixed-size window of the last N frames and uses all instances from those frames as candidates for matching.

```bash
sleap-nn-track \
    --data_path video.mp4 \
    --model_paths models/bottomup_unet/ \
    --tracking \
    --candidates_method fixed_window \
    --tracking_window_size 10
```

#### Local Queues Tracking

This method maintains separate queues for each track ID, keeping the last N instances per track. It's more robust to track breaks but requires more memory and computation.

```bash
sleap-nn-track \
    --data_path video.mp4 \
    --model_paths models/bottomup_unet/ \
    --tracking \
    --candidates_method local_queues \
    --tracking_window_size 5
```

#### Optical Flow Tracking

This method uses optical flow to shift the candidates onto the frame to be tracked and then associates the untracked instances to the shifted instances.

```bash
sleap-nn-track \
    --data_path video.mp4 \
    --model_paths models/bottomup_unet/ \
    --tracking \
    --use_flow
```

### Track-only workflow
You can perform tracking on existing user-labeled instances—without running inference to get new predictions—by enabling tracking (`--tracking`) and omitting the `--model-paths` argument. This will associate tracks using only the provided labels.

```bash
sleap-nn-track \
    --data_path video.mp4 \
    --tracking \
    --candidates_method fixed_window \
    --tracking_window_size 10
```

## Legacy SLEAP Model Support

SLEAP-NN supporting running inference with models trained using the original SLEAP TensorFlow/Keras backend (sleap <= 1.4), **but only for UNet backbone models**.
To run inference with legacy SLEAP models (trained with TensorFlow/Keras, sleap ≤ 1.4), simply use the same CLI or API workflows described above in the [With CLI](#with-cli) and [With API](#with-api) sections. 

Just provide the path to the directory containing both `best_model.h5` and `training_config.json` from your SLEAP training run as your `--model-paths` (CLI) or `model_paths` (API) argument. SLEAP-NN will automatically detect and load these legacy models for inference.


## Evaluation Metrics

SLEAP-NN provides comprehensive evaluation capabilities to assess model performance against ground truth labels.

Using CLI:
```bash
sleap-nn-eval \
    --ground_truth_path gt_labels.slp \
    --predicted_path pred_labels.slp \
    --save_metrics pred_metrics.npz \
```

Using `Evaluator` API:
```python
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
