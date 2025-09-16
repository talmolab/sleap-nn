# Step-by-Step Training Tutorial

This tutorial will walk you through the complete process of training a pose estimation model from scratch.

## 📋 Prerequisites

Before starting, make sure you have `sleap-nn` installed (Refer [`Installation docs`](installation.md))

!!! note "API-based Tutorial"

    In this tutorial, we use the **Python API** for all steps, which is ideal for running in a notebook or Python script. The `uvx` workflow **will not work** with the API-based approach.  

      - **Installation:** Make sure you have installed `sleap-nn` using either [pip](installation.md#installation-using-pip) or the [uv sync workflow](installation.md#installation-using-uv-sync).

      - **Command Line Interface (CLI):** If you prefer using the CLI, or want to see all available CLI options, refer to the [Training Guide](training.md) and [Inference Guide](inference.md).

---

## 🚀 Step 1: Configuration Setup

The first step is to set-up our configuration file, which configures the parameters required to train a pose estimation model with sleap-nn.

### 1.1 Load a Sample Configuration

Start by loading a sample (`.yaml`) configuration:

- [Available sample configs](config.md#available-sample-configurations)

```yaml
data_config:
  train_labels_path: 
    - path/to/your/training_data.slp
  val_labels_path:
    - path/to/your/validation_data.slp
  validation_fraction: 0.1
  user_instances_only: true
  data_pipeline_fw: torch_dataset
  preprocessing:
    ensure_rgb: false
    ensure_grayscale: false
    scale: 1.0
    crop_size: null
    min_crop_size: 100
  use_augmentations_train: true
  augmentation_config:
    intensity:
      contrast_p: 0.5
      brightness_p: 0.5
    geometric:
      rotation_min: -15.0
      rotation_max: 15.0
      scale_min: 0.9
      scale_max: 1.1
      affine_p: 1.0
model_config:
  pretrained_backbone_weights: null
  pretrained_head_weights: null
  backbone_config:
    unet:
      in_channels: 1
      kernel_size: 3
      filters: 16
      filters_rate: 2.0
      max_stride: 16
      middle_block: true
      up_interpolate: true
      stacks: 1
      convs_per_block: 2
      output_stride: 2
    convnext: null
    swint: null
  head_configs:
    single_instance:
      confmaps:
        part_names: null
        sigma: 2.5
        output_stride: 2
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
trainer_config:
  train_data_loader:
    batch_size: 4
    shuffle: true
    num_workers: 0
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0
  model_ckpt:
    save_top_k: 1
    save_last: false
  trainer_devices:
  trainer_device_indices:
  trainer_accelerator: auto
  min_train_steps_per_epoch: 200
  visualize_preds_during_training: true
  keep_viz: false
  max_epochs: 200
  use_wandb: false
  save_ckpt: true
  ckpt_dir: <ckpt_dir>
  run_name: <run_name>
  optimizer_name: Adam
  optimizer:
    lr: 0.0001
    amsgrad: false
  lr_scheduler:
    step_lr: null
    reduce_lr_on_plateau:
      threshold: 1.0e-06
      threshold_mode: rel
      cooldown: 3
      patience: 5
      factor: 0.5
      min_lr: 1.0e-08
  early_stopping:
    min_delta: 1.0e-08
    patience: 10
    stop_training_on_plateau: true
```

### 1.2 Understanding the Configuration Structure

Your config file has three main sections:

```yaml
data_config:      # How to load and process your data
model_config:     # What model architecture to use
trainer_config:   # How to train the model, setup hyparameters
```

### 1.3 Key Parameters to Modify

#### **Data Configuration (`data_config`)**

> Download sample [`train.pkg.slp`](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp) and [`val.pkg.slp`](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp).

Set the `train_labels_path` to the path of your training `.slp` file, or a list of `.slp` files if you have multiple datasets. The `val_labels_path` is optional—if you leave it out, the training data will be automatically split into training and validation sets based on the `validation_fraction` parameter. 

Choose the appropriate `data_pipeline_fw` based on your dataset size and hardware:

- Use `torch_dataset_cache_img_memory` for small datasets that fit comfortably in RAM. This will cache all source images in memory for faster training.
- Use `torch_dataset_cache_img_disk` for larger datasets that don't fit in memory. This caches images to disk, enabling efficient loading even for very large datasets. You can reuse the disk cache across different model types, since only the raw source images are cached (not model-specific data).

You can customize data loading, preprocessing, and augmentation options in this section. For a full explanation of all available parameters and augmentation settings, see the [Data config](config.md#data-configuration-data_config) section of the Configuration Guide.

```yaml
data_config:
  train_labels_path: 
    - path/to/your/training_data.slp
  val_labels_path:
    - path/to/your/validation_data.slp
  validation_fraction: 0.1
  data_pipeline_fw: torch_dataset
  preprocessing:
    ensure_rgb: false
    ensure_grayscale: false
    scale: 1.0
    crop_size: null # only for centered-instance model
    min_crop_size: 100 # only for centered-instance model
  use_augmentations_train: true
  augmentation_config:
    intensity:
      contrast_p: 0.5
      brightness_p: 0.5
    geometric:
      rotation_min: -15.0
      rotation_max: 15.0
      scale_min: 0.9
      scale_max: 1.1
      affine_p: 1.0
```

#### **Model Configuration (`model_config`)**
When configuring your model, you’ll need to select both a backbone architecture and a model type:

- **Backbone options:** `unet`, `swint`, or `convnext`
- **Model type options:** `single_instance`, `centroid`, `centered_instance`, or `bottomup`

For a detailed explanation of each backbone and model type, see the [Model Architectures Guide](models.md).

**Tips for configuring your model:**

- **Input channels (`in_channels`):** Set this to match your input image format (e.g., 1 for grayscale, 3 for RGB). The training pipeline will also infer and adjust this automatically.
- **Max stride (`max_stride`):** This parameter controls the number of downsampling (encoder) blocks in the backbone, which directly affects the receptive field size. For a deeper dive into how receptive field is affected, check out the [Receptive Field Guide](example_notebooks.md).
- **Special note for `convnext` and `swint`:** For these backbones, `max_stride` is determined by `stem_patch_stride * 16` and cannot be set arbitrarily.

For ready-to-use configuration examples for each backbone and model type, see the [Model Config Guide](config.md#model-configuration-model_config).

```yaml
model_config:
  pretrained_backbone_weights: null
  pretrained_head_weights: null
  backbone_config:
    unet:
      in_channels: 1
      kernel_size: 3
      filters: 16
      filters_rate: 2.0
      max_stride: 16
      middle_block: true
      up_interpolate: true
      stacks: 1
      convs_per_block: 2
      output_stride: 2
    convnext: null
    swint: null
  head_configs:
    single_instance:
      confmaps:
        part_names: null
        sigma: 2.5
        output_stride: 2
    centroid: null
    centered_instance: null
    bottomup: null
    multi_class_bottomup: null
    multi_class_topdown: null
```

#### **Trainer Configuration (`trainer_config`)**
The `trainer_config` section controls the training process, including key hyperparameters and device settings.

**Key tips for configuring `trainer_config`:**

- **Data Loader Workers (`num_workers`):**  
    - For the default data pipeline (`torch_dataset`), set `num_workers: 0` because `.slp` video objects cannot be pickled for multiprocessing.
    - If you use a caching data pipeline (e.g., `torch_dataset_cache_img_memory` or `torch_dataset_cache_img_disk` for `data_config.data_pipeline_fw`), you can increase `num_workers` (>0) to speed up data loading.

- **Epochs and Checkpoints:**  
    - Set `max_epochs` to control how many epochs to train for.
    - Use `ckpt_dir` and `run_name` to specify where model checkpoints are saved. If both are `None`, a default folder will be created in the working directory using a timestamp and model type.
    - For multi-GPU training, always set a static `run_name` so all workers write to the same location.

- **Device and Accelerator:**  
    - `trainer_accelerator` can be `"cpu"`, `"gpu"`, `"mps"`, or `"auto"`.  
            - `"auto"` lets Lightning choose the best device based on your hardware.
    - `trainer_device_indices` is a list of ints used to set the device indices.
    - `trainer_devices` can be set to specify the number of devices (e.g., GPUs) to use. If `None`, the number of devices is inferred from the underlying hardware in the training workflow. 

- **Other Tips:**  
    - Adjust `batch_size` and learning rate (`optimizer.lr`) as needed for your dataset and hardware.
    - Enable `visualize_preds_during_training` to see predictions during training.
    - Use `use_wandb: true` to log training metrics to Weights & Biases (optional).

For a full list of options and explanations for the `trainer_config` parameters, see the [Config Guide](config.md#trainer-configuration-trainer_config).

```yaml
trainer_config:
  train_data_loader:
    batch_size: 4
    shuffle: true
    num_workers: 0
  val_data_loader:
    batch_size: 4
    shuffle: false
    num_workers: 0
  model_ckpt:
    save_top_k: 1
    save_last: false
  trainer_devices:
  trainer_device_indices:
  trainer_accelerator: auto
  min_train_steps_per_epoch: 200
  visualize_preds_during_training: true
  keep_viz: false
  max_epochs: 200
  use_wandb: false
  save_ckpt: true
  ckpt_dir: my_model_ckpt_dir
  run_name: my_run_1
  optimizer_name: Adam
  optimizer:
    lr: 0.0001
    amsgrad: false
  lr_scheduler:
    step_lr: null
    reduce_lr_on_plateau:
      threshold: 1.0e-06
      threshold_mode: rel
      cooldown: 3
      patience: 5
      factor: 0.5
      min_lr: 1.0e-08
  early_stopping:
    min_delta: 1.0e-08
    patience: 10
    stop_training_on_plateau: true
```

---

## 🤖 Step 2: Training Your Model

Now that you have your configuration file, let's train your model!

### 2.1 Training with Python API

```python linenums="1"
from omegaconf import OmegaConf
from sleap_nn.training.model_trainer import ModelTrainer

# Load configuration
config = OmegaConf.load("my_config.yaml")

# Create trainer
trainer = ModelTrainer.get_model_trainer_from_config(config=config)

# Start training
trainer.train()
```

If you want to use custom `sleap_io.Labels` objects,

```python linenums="1"
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_io import Labels

# Load your labels
train_labels = Labels.load("my_data.slp")
val_labels = Labels.load("my_validation.slp")

# Create trainer with custom labels
trainer = ModelTrainer.get_model_trainer_from_config(
    config=config,
    train_labels=[train_labels],
    val_labels=[val_labels]
)

# Train
trainer.train()
```

> For more details and advanced training options, see the [Training Guide](training.md).


### 2.2 Training Output

After training, you'll find:
```
my_model_ckpt_dir/my_run_1
├── best.ckpt                  # Best model weights
├── initial_config.yaml        # Initial training configuration
├── training_config.yaml       # Final training configuration
├── labels_train_gt_0.slp      # Ground-truth train data split
├── labels_val_gt_0.slp        # Ground-truth val data split
├── pred_train_0.slp           # Predictions on training data
├── pred_val_0.slp             # Predictions on validation data
├── train_0_pred_metrics.npz   # Metrics on train preds
├── val_0_pred_metrics.npz     # Metrics on val preds
└── training_log.csv           # CSV that tracks the train/ val losses and epoch time
```

---

## 🔍 Step 3: Running Inference

Now that you have a trained model, let's use it to make predictions on new data!

### 3.1 Inference

To run inference on a `.slp` file, 

```python linenums="1"
from sleap_nn.predict import run_inference

pred_labels = run_inference(
  data_path="test.slp",
  model_paths=["/path/to/model/dir"],
  output_path="preds.slp",
)
```

To run inference on a video on specific frames, 

```python linenums="1"
from sleap_nn.predict import run_inference

pred_labels = run_inference(
  data_path="test.mp4",
  model_paths=["/path/to/model/dir"],
  output_path="preds.slp",
  frames=list(range(100)), # run on the first 100 frames
)
```

To run inference on a video with tracking, 

```python linenums="1"
from sleap_nn.predict import run_inference

pred_labels = run_inference(
  data_path="test.mp4",
  model_paths=["/path/to/model/dir"],
  output_path="preds.slp",
  tracking=True
)
```

> For more details and advanced inference options, see the [Inference Guide](inference.md).

### 3.2 Inference Parameters

#### **Essential Parameters:**
- `--data_path`: Input video or labels file
- `--model_paths`: Path to your trained model directory
- `--output_path`: Where to save predictions
- `--batch_size`: Number of frames to process at once
- `--device`: Hardware to use (cpu, cuda, mps, auto)
- `--peak_threshold`: Confidence threshold for detections
- `--frames`: Specific frame ranges (e.g., "1-100,200-300")
- `--tracking`: To enable tracking

---

## 📊 Step 4: Evaluation and Visualization

Let's evaluate how well your model performed and visualize the results!

### 4.1 Evaluating Model Performance

```python
from sleap_nn.evaluation import Evaluator
import sleap_io as sio

# Load labels
ground_truth = sio.load_slp("ground_truth.slp")
predictions = sio.load_slp("predictions.slp")

# Create evaluator
evaluator = Evaluator(ground_truth, predictions)

# Run evaluation
metrics = evaluator.evaluate()

# Print results
print(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']:.3f}")
print(f"Dist p90: {metrics['distance_metrics']['p90']:.3f}")
```

### 4.2 Visualizing Results

```python
import sleap_io as sio
import matplotlib.pyplot as plt

def plot_preds(gt_labels, pred_labels, lf_index):
    _fig, _ax = plt.subplots(1, 1, figsize=(5 * 1, 5 * 1))

    # Plot each frame
    gt_lf = gt_labels[lf_index]
    pred_lf = pred_labels[lf_index]

    # Ensure we're plotting keypoints for the same frame
    assert (
        gt_lf.frame_idx == pred_lf.frame_idx
    ), f"Frame mismatch at {lf_index}: GT={gt_lf.frame_idx}, Pred={pred_lf.frame_idx}"

    _ax.imshow(gt_lf.image, cmap="gray")
    _ax.set_title(
        f"Frame {gt_lf.frame_idx} (lf idx: {lf_index})",
        fontsize=12,
        fontweight="bold",
    )

    # Plot ground truth instances
    for idx, instance in enumerate(gt_lf.instances):
        if not instance.is_empty:
            gt_pts = instance.numpy()
            _ax.plot(
                gt_pts[:, 0],
                gt_pts[:, 1],
                "go",
                markersize=6,
                alpha=0.8,
                label="GT" if idx == 0 else "",
            )

    # Plot predicted instances
    for idx, instance in enumerate(pred_lf.instances):
        if not instance.is_empty:
            pred_pts = instance.numpy()
            _ax.plot(
                pred_pts[:, 0],
                pred_pts[:, 1],
                "rx",
                markersize=6,
                alpha=0.8,
                label="Pred" if idx == 0 else "",
            )

    # Add legend
    _ax.legend(loc="upper right", fontsize=8)

    _ax.axis("off")

    plt.suptitle(f"Ground Truth vs Predictions", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.show()
    return


# Overlay results
gt_labels = sio.load_slp("groundtruth.slp")
pred_labels = sio.load_slp("my_predictions.slp")
plot_preds(gt_labels, pred_labels, lf_index=0)
```

### 4.3 Metrics Interpretation

#### **Key Metrics to Understand:**
- **PCK (Percentage of Correct Keypoints)**: How many keypoints are within a certain distance threshold
- **OKS (Object Keypoint Similarity)**: How similar are the predicted keypoints to the ground-truth
- **mAP (mean Average Precision)**: Mean of average precisions across match thresholds (where OKS or PCK could be the matching score).
- **Distance Metrics**: Average euclidean distance between predicted and true keypoints

---

## ✨ Next Steps

Now that you have the basics, you can:

1. **Experiment with different model architectures** (UNet, ConvNeXt, SwinT)
2. **Try different detection methods** (single instance, bottom-up, top-down)
3. **Optimize hyperparameters** for better performance
4. **Use data augmentation** to improve model robustness

## 📚 Additional Resources

- **[Configuration Guide](config.md)**: Detailed configuration options
- **[Training Documentation](training.md)**: Advanced training features
- **[Inference Guide](inference.md)**: Complete inference options
- **[Model Architectures](models.md)**: Available model types
- **[Example Notebooks](example_notebooks.md)**: Interactive tutorials

Happy SLEAPiNNg! 🐭🐭
