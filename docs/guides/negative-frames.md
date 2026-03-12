# Negative Frames

Teach your model to not hallucinate detections on empty backgrounds.

---

## What Are Negative Frames?

Negative frames are frames that the user has explicitly marked as containing **no animals**. When included in training, the model sees these frames paired with all-zero confidence maps, learning to suppress false-positive detections on backgrounds where nothing is present.

Without negative frames, models only ever see frames with animals during training. This can cause the model to predict phantom keypoints on background-only regions at inference time — especially in scenes with textured backgrounds, equipment, or visual clutter.

---

## Marking Frames as Negative

Negative frames are marked in the **SLEAP labeling GUI** (the `sleap` frontend). When you mark a frame as negative, it is stored as a `LabeledFrame` with `is_negative=True` in the `.slp` file. These frames have no instance annotations — they represent confirmed empty backgrounds.

!!! important
    Only frames explicitly marked by the user are used. Unlabeled frames (frames that simply haven't been annotated yet) are **never** treated as negatives, since they may contain animals that haven't been labeled.

### What Makes a Good Negative Frame?

Choose frames that:

- Show the environment without any animals present (e.g., before animals enter the arena)
- Cover diverse backgrounds that appear in your videos (different lighting, positions, equipment)
- Represent the visual variety the model will encounter at inference time

The more diverse your negative frames, the better the model generalizes. A few negatives from one background teaches "don't predict on *this* background." Many negatives across varied backgrounds teaches "don't predict when there are no animals."

---

## Enabling in Config

```yaml
data_config:
  use_negative_frames: true      # Include negative frames in training
  negative_loss_weight: 1.0      # Relative weight for negative sample loss
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_negative_frames` | `bool` | `false` | Include all user-confirmed negative frames in training |
| `negative_loss_weight` | `float` | `1.0` | Multiplier on per-sample loss for negatives. Values > 1 amplify the negative gradient signal; < 1 dampen it. Must be > 0. |

### Python API

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
config.data_config.use_negative_frames = True
config.data_config.negative_loss_weight = 1.0
```

---

## Supported Model Types

| Model Type | Supported | Notes |
|------------|-----------|-------|
| Single Instance | Yes | |
| Centroid | Yes | |
| Bottom-Up | Yes | |
| Bottom-Up Multi-Class | Yes | |
| Centered Instance (Top-Down) | No | Operates on instance crops, not full frames |
| Multi-Class Top-Down | No | Operates on instance crops, not full frames |

For unsupported model types, a warning is logged and the feature is automatically disabled:

```
use_negative_frames is enabled but model_type='centered_instance' operates at
instance-crop level and does not support frame-level negatives. Negative frames
will be disabled.
```

---

## How It Works

### Data Pipeline

1. **Collection:** `_collect_negative_frames()` iterates over `labels.negative_frames` for each label file and builds sample dicts with `is_negative=True`.

2. **Train/Val Split:** Negative frames are `LabeledFrame` objects in the `.slp` file. When `make_training_splits()` splits labeled frames into train and val, negative frames are split proportionally alongside positives.

3. **Dataset:** Negative samples are appended to `lf_idx_list`. Each negative frame appears **once** — there is no oversampling. The balance between positives and negatives is controlled by how many frames the user marks as negative during annotation.

4. **Shuffling:** The standard PyTorch `DataLoader` with `shuffle=True` interleaves negatives with positives across batches.

5. **Image Loading:** `_load_negative_sample()` reads the image from the video at the specified frame index. In caching mode, negatives are loaded dynamically via stored video references (not cached, since they have no `lf_idx`).

6. **Confidence Maps:** `process_negative_lf()` creates a sample with all-NaN instances. These flow through the existing confidence map and PAF generation code, which converts NaN values to zeros via `torch.nan_to_num()`. The result is an all-zero target.

### Loss Computation

When `is_negative` is present in the batch:

- Per-sample MSE is computed: `(y_pred - y_target)^2` averaged over spatial dims
- Positive samples get weight `1.0`; negative samples get weight `negative_loss_weight`
- The weighted per-sample losses are averaged to produce the final loss

When `negative_loss_weight=1.0` (default), this is numerically identical to `nn.MSELoss()`.

### Feature Gating

When `use_negative_frames=False` (default), **no code paths change**:

- No `is_negative` key in sample dicts or batch tensors
- No additional metrics logged
- Loss computation uses `nn.MSELoss()` directly

---

## Monitoring

When the feature is active, these additional metrics are logged to WandB/TensorBoard:

| Metric | Aggregation | What to Look For |
|--------|-------------|------------------|
| `train/loss` | epoch mean | Combined loss (always logged, unchanged) |
| `train/loss_positive` | epoch mean | MSE on positive frames — should decrease as model learns poses |
| `train/loss_negative` | epoch mean | MSE on negative frames — should approach 0 as model learns to suppress |
| `train/n_positive` | epoch sum | Total positive samples seen in epoch |
| `train/n_negative` | epoch sum | Total negative samples seen in epoch |

!!! tip "Is It Working?"
    Watch `train/loss_negative` — it should decrease toward 0 over training. If it stays flat, the model isn't learning from negatives. Consider increasing `negative_loss_weight` or adding more diverse negative frames.

---

## SLP File Format

Negative frames are stored in `.slp` files (HDF5-based) in two places:

1. **`/frames` dataset:** Like all labeled frames, with an empty instance range (`instance_id_start == instance_id_end`)
2. **`/negative_frames` dataset:** A sidecar structured array of `(video_id, frame_idx)` tuples that marks which frames are explicitly negative

The `sleap-io` library exposes these via:

```python
import sleap_io as sio

labels = sio.load_slp("data.slp")

# All user-confirmed negative frames
neg_frames = labels.negative_frames  # List[LabeledFrame] where is_negative=True

# Check a specific frame
lf = labels[0]
print(lf.is_negative)  # True or False
```

---

## Practical Tips

- **Start with 10-20% negatives.** If you have 100 labeled frames, mark 10-20 background frames as negative. This gives a reasonable signal without dominating training.

- **Diverse backgrounds matter more than quantity.** 10 negatives from 10 different camera angles or time points are more valuable than 50 negatives from the same static background.

- **Watch the batch composition.** With 100 positives and 10 negatives, ~9% of samples are negative. In a batch of 8, most batches will have 0-1 negatives. This is fine — the loss weight amplifies the signal when negatives do appear.

- **Use `negative_loss_weight` to tune.** If negatives are rare relative to positives and `train/loss_negative` isn't decreasing, try `negative_loss_weight: 2.0` or higher to amplify the gradient signal.

- **Check val metrics too.** Negative frames in the validation set (via the train/val split) contribute to `val/loss`. A model that hallucinates on empty backgrounds will have higher val loss.
