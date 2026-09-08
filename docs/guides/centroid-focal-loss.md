# Centroid Focal Loss

An opt-in alternative to plain MSE for the centroid confidence-map head, aimed at dense-animal scenes where nearby instances can blur together.

---

## What Problem Does This Solve?

Centroid confidence maps are dominated by background: a single sparse Gaussian peak per instance spread across an otherwise-empty (often downsampled) frame. Plain `MSELoss` treats every pixel equally, so the loss signal is mostly driven by the huge number of easy background pixels rather than the handful of foreground pixels that actually matter.

In scenes with multiple animals close together (dense tracking, small enclosures, swarms), this can produce a specific failure mode: **confmap collapse**, where two nearby instances' peaks blur into a single merged blob under MSE's smooth quadratic penalty, causing duplicate/missed detections at inference time.

Centroid focal loss is a CenterNet/CornerNet-style penalty-reduced pixelwise focal loss ([Lin et al. 2017](https://arxiv.org/abs/1708.02002), [Zhou et al. 2019](https://arxiv.org/abs/1904.07850)) that down-weights already-confident (easy) pixels on both the foreground and background side, concentrating gradient signal on ambiguous pixels — including where two peaks sit close together.

!!! warning "Not a universal upgrade"
    Results are dataset-dependent: a clear win on some dense-instance datasets, roughly neutral-to-slightly-worse on others. Treat this as an experimental tool to try when you suspect confmap collapse, not a default recommendation.

---

## The Math

For a pixel with predicted probability $\hat y \in (0,1)$ and ground-truth confmap value $y \in [0,1]$ (continuous Gaussian target), and a positive-pixel threshold $\tau$ (`focal_loss_pos_threshold`):

$$
\ell(\hat y, y) =
\begin{cases}
-(1-\hat y)^{\alpha}\,\log(\hat y) & \text{if } y \ge \tau \quad \text{(positive pixel)}\\[6pt]
-(1-y)^{\beta}\,\hat y^{\alpha}\,\log(1-\hat y) & \text{if } y < \tau \quad \text{(negative pixel)}
\end{cases}
$$

- **$(1-\hat y)^\alpha$ / $\hat y^\alpha$** (the focusing terms) shrink toward 0 for pixels the model already predicts confidently and correctly, so training concentrates on pixels still being learned.
- **$(1-y)^\beta$** (the penalty-reduction term) softens the penalty for background pixels that sit close to — but just below — a true peak, since they're not really "wrong," just imprecise.

sleap-nn's targets are continuous sub-pixel Gaussians (unlike the original CenterNet formulation's integer-snapped peaks), so "positive" is defined as `y >= pos_threshold` rather than exact equality.

---

## Enabling in Config

```yaml
model_config:
  head_configs:
    centroid:
      confmaps:
        use_sigmoid_activation: true    # required: calibrates head output to (0, 1)
        focal_loss_alpha: 2.0           # 0.0 disables (default); 2.0 is the standard CenterNet value
        focal_loss_beta: 4.0            # default
        focal_loss_pos_threshold: 0.5   # default
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_sigmoid_activation` | `bool` | `false` | Applies a sigmoid to the head's raw output, producing a calibrated `(0, 1)` probability. **Required** for focal loss — its `log(ŷ)`/`log(1-ŷ)` terms need bounded input. |
| `focal_loss_alpha` | `float` | `0.0` | Focal exponent. `0.0` disables focal loss entirely (plain MSE, byte-identical to every existing config). Nonzero enables it. |
| `focal_loss_beta` | `float` | `4.0` | Penalty-reduction exponent for negative pixels near a true peak. Only has an effect when `focal_loss_alpha != 0`. |
| `focal_loss_pos_threshold` | `float` | `0.5` | Minimum target confmap value for a pixel to count as "positive." Only has an effect when `focal_loss_alpha != 0`. |

All four fields live together on the centroid head's confmaps config (`CentroidConfMapsConfig`), not on `data_config` — this is loss/architecture configuration specific to the centroid head, not a data-pipeline setting.

### Disabling

Set `focal_loss_alpha: 0.0` (or omit it). `CentroidLightningModule._compute_loss` checks this first and falls straight back to the original plain-MSE path, ignoring `beta`/`pos_threshold`.

For a fully byte-identical revert to pre-feature behavior, also set `use_sigmoid_activation: false` — leaving it `true` with `alpha: 0.0` is a valid but *different* config (plain MSE trained against a sigmoid-bounded head output, not the original unbounded regression output).

### Python API

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
config.model_config.head_configs.centroid.confmaps.use_sigmoid_activation = True
config.model_config.head_configs.centroid.confmaps.focal_loss_alpha = 2.0
```

---

## Supported Model Types

| Model Type | Supported | Notes |
|------------|-----------|-------|
| Centroid | Yes | |
| Single Instance | No | Not yet implemented — see [issue #736](https://github.com/talmolab/sleap-nn/issues/736) |
| Centered Instance (Top-Down) | No | Not yet implemented — see [issue #736](https://github.com/talmolab/sleap-nn/issues/736) |
| Bottom-Up | No | Not yet implemented — see [issue #736](https://github.com/talmolab/sleap-nn/issues/736) |

---

## How It Works

### Head Output

`CentroidConfmapsHead.use_sigmoid_activation` swaps the head's output activation from `identity` to `sigmoid`, baked into the model graph at **train time**. Because the activation lives in the model itself (not applied only during loss computation), a checkpoint trained with this flag on works with the existing inference path (`predict()`, `peak_threshold`) with **no changes needed** — the raw confmap output is already a calibrated probability.

### Bias Initialization

A freshly initialized sigmoid head starts at ~0.5 everywhere. Against a target that's >99% background, that start point gives weak, roughly-symmetric gradients that can leave training stuck for many epochs. To fix this, when `focal_loss_alpha != 0`, the head's first-conv-layer bias is initialized to a large negative logit (RetinaNet/CenterNet "prior probability" bias init, Lin et al. 2017 §4.1) so the head starts predicting near-0 everywhere, giving the loss a much stronger initial gradient toward learning the sparse foreground.

### Loss Computation

`CentroidLightningModule._compute_loss` (in `sleap_nn/training/lightning_modules.py`):

- **`train`, `alpha == 0`:** identical to the pre-existing plain-MSE path (optionally negative-frame-weighted).
- **`train`, `alpha != 0`:** uses `compute_centroid_focal_loss` in place of MSE, then applies the same negative-frame weighting on top.
- **`val`/`eval`, always:** plain unweighted MSE regardless of `alpha`, so `ModelCheckpoint`/`EarlyStopping` and cross-experiment comparisons remain meaningful even as you vary `alpha`.

### Feature Gating

When `focal_loss_alpha == 0.0` (default), no code paths change: the loss computation is the exact same call as before this feature existed.

---

## Monitoring

A diagnostic foreground/background confmap MSE split is logged **regardless of whether focal loss is enabled** — it's useful both to decide if you *should* try focal loss, and to check whether it's helping once enabled. These reach WandB and `training_log.csv` identically:

| Metric | What to Look For |
|--------|------------------|
| `train/confmap_loss_fg` / `val/confmap_loss_fg` | MSE over pixels where `y > 0.5` (near a true peak). Should decrease as training progresses. |
| `train/confmap_loss_bg` / `val/confmap_loss_bg` | MSE over pixels where `y < 0.5`. Typically small relative to `_fg` given the pixel-count imbalance. |
| `train/confmap_fg_frac` / `val/confmap_fg_frac` | Fraction of pixels that are foreground — a direct measure of how imbalanced your data is. A very small fraction is exactly the scenario focal loss targets. |

!!! tip "Is It Helping?"
    Compare `confmap_loss_fg` between an `alpha=0` (MSE) run and an `alpha=2` (focal) run on the same data. If dense/close-together instances are your problem, also inspect predicted confmaps visually (via the training viz callback) for merged vs. separated peaks — the diagnostic loss values alone won't show peak-collapse directly.

---

## Practical Tips

- **Try this only if you suspect confmap collapse.** Symptoms: duplicate or missed centroid detections specifically in frames where instances are close together, while isolated instances are detected fine.
- **Start with the CenterNet defaults** (`alpha=2.0`, `beta=4.0`, `pos_threshold=0.5`) before tuning — these are well-studied values from the object-detection literature.
- **Watch `val/loss` for model selection, not the focal training loss.** Since val always uses plain MSE, `val/loss` stays comparable across `alpha` settings — use it (not the focal-loss-influenced train loss) to judge whether a given `alpha` actually improved the model.
- **Expect dataset-dependent results.** This isn't a guaranteed win — evaluate empirically on your own held-out data (e.g. centroid distance/precision/recall via `--match_method centroid`) rather than assuming it helps.
