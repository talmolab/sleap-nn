# Quick Start

Train a model and run inference in under 5 minutes.

---

## Prerequisites

- [SLEAP-NN installed](../installation.md)
- A training dataset (`.slp` or `.pkg.slp` file)

!!! tip "Sample Data"
    Download sample data to try it out:

    - [Train data](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp)
    - [Validation data](https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp)

---

## Step 1: Create a Config File

Create `config.yaml`:

```yaml title="config.yaml"
data_config:
  train_labels_path:
    - train.pkg.slp
  val_labels_path:
    - val.pkg.slp

model_config:
  backbone_config:
    unet:
      filters: 32
      max_stride: 16
  head_configs:
    single_instance:
      confmaps:
        sigma: 5.0

trainer_config:
  max_epochs: 50
  save_ckpt: true
  ckpt_dir: models
  run_name: my_first_model
```

Or grab a [sample config](../configuration/samples.md) for your model type.

---

## Step 2: Train

```bash
sleap-nn train --config config.yaml
```

That's it! Training will:

1. Load your data
2. Build the model
3. Train for 50 epochs
4. Save the best checkpoint to `models/my_first_model/`

---

## Step 3: Run Inference

```bash
sleap-nn track --data_path val.pkg.slp --model_paths models/my_first_model/ -o val.predictions.slp
```

This creates `val.predictions.slp` with your predictions.

---

## Step 4: View Results

Open the predictions in the [SLEAP](https://docs.sleap.ai/latest/) GUI:

```bash
sleap-label val.predictions.slp
```

Or load in Python:

```python
import sleap_io as sio

labels = sio.load_slp("val.predictions.slp")
print(f"Found {len(labels)} frames with predictions")
```

---

## What's Next?

<div class="grid cards" markdown>

-   **Train a multi-animal model**

    [:octicons-arrow-right-24: Model types](../reference/models.md)

-   **Customize your config**

    [:octicons-arrow-right-24: Configuration guide](../configuration/index.md)

-   **Enable tracking**

    [:octicons-arrow-right-24: Tracking guide](../guides/tracking.md)

-   **Export for production**

    [:octicons-arrow-right-24: ONNX/TensorRT export](../guides/export.md)

</div>

