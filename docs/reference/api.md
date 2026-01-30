# Python API

Use SLEAP-NN programmatically in Python.

---

## Training

### Using Config File

```python
from omegaconf import OmegaConf
from sleap_nn.training.model_trainer import ModelTrainer

# Load config
config = OmegaConf.load("config.yaml")

# Create trainer and train
trainer = ModelTrainer.get_model_trainer_from_config(config=config)
trainer.train()
```

### With Custom Labels

```python
import sleap_io as sio
from omegaconf import OmegaConf
from sleap_nn.training.model_trainer import ModelTrainer

# Load labels
train_labels = sio.load_slp("train.slp")
val_labels = sio.load_slp("val.slp")

# Load config and train
config = OmegaConf.load("config.yaml")
trainer = ModelTrainer.get_model_trainer_from_config(
    config=config,
    train_labels=[train_labels],
    val_labels=[val_labels]
)
trainer.train()
```

### Simplified Training

```python
from sleap_nn.train import train

train(
    train_labels_path=["train.slp"],
    backbone_config="unet_medium_rf",
    head_configs="bottomup",
    save_ckpt=True,
    max_epochs=100,
)
```

---

## Inference

### Basic Inference

```python
from sleap_nn.predict import run_inference

labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/bottomup/"],
    output_path="predictions.slp",
    make_labels=True,
)
```

### Get Raw Outputs

```python
results = run_inference(
    data_path="video.mp4",
    model_paths=["models/bottomup/"],
    make_labels=False,
    return_confmaps=True,
)

# results is a list of dicts with predictions per frame
for frame_result in results:
    peaks = frame_result["peaks"]
    confmaps = frame_result["confmaps"]
```

### With Options

```python
labels = run_inference(
    data_path="video.mp4",
    model_paths=["models/centroid/", "models/instance/"],
    output_path="predictions.slp",
    batch_size=8,
    device="cuda:0",
    peak_threshold=0.3,
    make_labels=True,
)
```

---

## Evaluation

```python
import sleap_io as sio
from sleap_nn.evaluation import Evaluator

# Load labels
gt = sio.load_slp("ground_truth.slp")
pred = sio.load_slp("predictions.slp")

# Evaluate
evaluator = Evaluator(gt, pred)
metrics = evaluator.evaluate()

# Access results
print(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']:.3f}")
print(f"Mean OKS: {metrics['mOKS']:.3f}")
print(f"Distance 90th %ile: {metrics['distance_metrics']['p90']:.2f} px")
```

---

## Export

### To ONNX

```python
from sleap_nn.export import export_to_onnx

export_to_onnx(
    model,
    output_path="model.onnx",
    input_shape=(1, 1, 192, 192),
    input_dtype="uint8",
)
```

### To TensorRT

```python
from sleap_nn.export import export_to_tensorrt

export_to_tensorrt(
    model,
    output_path="model.trt",
    input_shape=(1, 1, 192, 192),
    precision="fp16",
)
```

### Inference with Exported Model

```python
from sleap_nn.export.predictors import ONNXPredictor
import numpy as np

predictor = ONNXPredictor("model.onnx")

# Prepare frames (uint8, NCHW format)
frames = np.random.randint(0, 256, (4, 1, 192, 192), dtype=np.uint8)

# Predict
outputs = predictor.predict(frames)
peaks = outputs["peaks"]      # (B, N_nodes, 2)
peak_vals = outputs["peak_vals"]  # (B, N_nodes)
```

---

## Configuration

### Load Config

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")

# Access values
print(config.trainer_config.max_epochs)
print(config.model_config.backbone_config.unet.filters)
```

### Modify Config

```python
config.trainer_config.max_epochs = 200
config.trainer_config.optimizer.lr = 0.0005
```

### Save Config

```python
OmegaConf.save(config, "modified_config.yaml")
```

### Convert Legacy Config

```python
from sleap_nn.config.training_job_config import TrainingJobConfig
from omegaconf import OmegaConf

config = TrainingJobConfig.load_sleap_config("config.json")
OmegaConf.save(config, "config.yaml")
```

---

## Data I/O

SLEAP-NN uses [sleap-io](https://io.sleap.ai) for data handling.

### Load Labels

```python
import sleap_io as sio

labels = sio.load_slp("labels.slp")
print(f"Videos: {len(labels.videos)}")
print(f"Labeled frames: {len(labels)}")
print(f"Skeleton: {labels.skeleton}")
```

### Access Predictions

```python
for lf in labels:
    print(f"Frame {lf.frame_idx}:")
    for inst in lf.instances:
        print(f"  Instance: {len(inst.points)} points")
```

### Save Labels

```python
sio.save_slp(labels, "output.slp")
```

---

## Full API Reference

For complete API documentation, see the auto-generated reference:

[:octicons-arrow-right-24: Full API Documentation](https://nn.sleap.ai/api/)
