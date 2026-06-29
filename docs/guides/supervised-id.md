# Supervised ID models

Predict pose **and** a persistent identity for each animal in a single model —
no separate tracking step needed. These are the `multi_class_topdown` and
`multi_class_bottomup` model types.

Use them when your animals have **distinct, consistent appearances** — visual
markers, fur color, ear-clips, dye spots — and you want identities assigned
directly by the network instead of linked frame-to-frame by an
[unsupervised tracker](tracking.md).

---

## When to use

A supervised ID model learns to map each animal's appearance to an **identity
class** you define (e.g. `"male"` / `"female"`, `"dark_fur"` / `"light_fur"`).
At inference it predicts the pose as usual and also assigns each instance to one
of those classes by appearance.

| | Supervised ID | Unsupervised tracking |
|---|---|---|
| Identities | Fixed classes you label | Anonymous tracks linked across frames |
| Needs distinct appearance | **Yes** | No |
| Proofreading | Minimal — IDs come from appearance, not motion | Often needed (ID swaps on occlusion) |
| Extra labeling | Assign a track to every user instance | None |
| Training | Finicky — balances pose vs. identity objectives | N/A |

**Choose supervised ID when** the individuals look reliably different and you
want to eliminate most ID-swap proofreading. **Stick with standard pose models +
[tracking](tracking.md) when** the animals are visually similar (the classifier
can't separate appearances it can't see) or you don't need persistent IDs.

!!! warning "These models can be finicky to train"
    You're optimizing two different objectives at once (accurate pose *and*
    correct identity). Expect to spend some time tuning the classification
    [loss weight](#tuning-the-loss-weight). If it doesn't converge well, fall
    back to a standard model plus tracking.

---

## Step 1: Label identities as tracks

Supervised ID reuses SLEAP's **track** mechanism: each track *name* becomes an
identity *class*. Assign a track to every user instance in the legacy
[SLEAP GUI](https://github.com/talmolab/sleap):

1. Open a frame with labeled animals.
2. Click an instance to select it (it should show **Track: none**).
3. **Tracks → Set Instance Track → New Track**.
4. In the **Instances** panel, double-click the new track and rename it to the
   identity class — e.g. `male`, `female`, `dark_fur`, `neck_dye`.
5. Repeat for every animal in the frame.
6. For the remaining frames, select each instance and assign it to a track with
   the **Ctrl+1 – Ctrl+9** shortcuts (hold **Ctrl** to see which shortcut maps
   to which track).

!!! important "Assign a track to every instance you want identified"
    Training uses **user instances only** (`user_instances_only: true`), and an
    instance **without a track carries no identity label** — it contributes
    nothing to the identity objective. So assign a track to every labeled
    instance, and use **consistent track names** across the whole project: the
    set of distinct names defines your classes. (Training errors out if it finds
    no tracks at all.)

The class list is inferred from your track names automatically (the head's
`classes` field is left empty / `null` in the config).

---

## Step 2: Pick a model type

Both variants mirror their non-ID counterparts — same pose architecture, plus a
classification head. Choose using the same logic as in
[Choosing a Model](../reference/models.md#choosing-a-model):

| Model type | Based on | Models to train | Best when |
|---|---|---|---|
| `multi_class_topdown` | Top-Down | centroid + `multi_class_topdown` | Animals separated; size small relative to frame |
| `multi_class_bottomup` | Bottom-Up | one model | Animals overlap / touch; flexible bodies |

The simplest way to convert an existing project is to take your current
`centered_instance` (top-down) or `bottomup` config and swap the head type —
keep the backbone and pose parameters the same.

### `multi_class_topdown`

The centered-instance stage of a top-down pipeline, with a `class_vectors`
classification head. It still needs a **centroid model** (train one exactly as
for standard top-down). Starter:
[`config_topdown_multi_class_centered_instance_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_topdown_multi_class_centered_instance_unet.yaml).

```yaml
head_configs:
  multi_class_topdown:
    confmaps:
      anchor_part: null      # anchor node for cropping (e.g. "thorax")
      sigma: 1.5
      output_stride: 2
    class_vectors:
      classes: null          # inferred from track names
      num_fc_layers: 3
      num_fc_units: 64
      global_pool: true
      output_stride: 16
      loss_weight: 0.01      # tune this — see below
```

### `multi_class_bottomup`

A single bottom-up model with a `class_maps` head alongside the confidence-map
head. Starter:
[`config_multi_class_bottomup_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_multi_class_bottomup_unet.yaml).

```yaml
head_configs:
  multi_class_bottomup:
    confmaps:
      part_names: null
      sigma: 1.5
      output_stride: 2
      loss_weight: 1.0
    class_maps:
      classes: null          # inferred from track names
      sigma: 50
      output_stride: 4
      loss_weight: 1.0       # tune this — see below
```

See [Model Types → Multi-Class](../reference/models.md#multi-class-identity-models)
and the [model config reference](../configuration/model.md) for the full head
schemas.

---

## Step 3: Train

Train exactly like any other model:

```bash
# Bottom-up ID: one model
sleap-nn train --config-name config_multi_class_bottomup_unet.yaml

# Top-down ID: a centroid model + the multi-class centered-instance model
sleap-nn train --config-name config_centroid_unet.yaml
sleap-nn train --config-name config_topdown_multi_class_centered_instance_unet.yaml
```

---

## Tuning the loss weight

The most important knob is the **`loss_weight` on the classification head**
(`class_vectors.loss_weight` for top-down, `class_maps.loss_weight` for
bottom-up). It balances the pose objective against the identity objective:

| Symptom | Adjustment |
|---|---|
| Starting point | `loss_weight: 1e-3` (`0.001`) |
| **Poses degrade** | **Decrease** toward `1e-4` (`0.0001`) |
| **Identities don't separate** | **Increase** toward `1e-2` (`0.01`) |

!!! tip "Pose vs. identity tension"
    Pushing the classification weight up sharpens identity separation but can
    blur the pose; pushing it down protects the pose at the cost of fuzzier
    IDs. Adjust in roughly 10× steps and watch both the pose metrics and the
    identity assignments before changing anything else.

---

## Step 4: Run inference

Identities are assigned **by the model** — you do **not** pass any
`--tracking.*` / tracker arguments.

```bash
# Bottom-up ID: single model directory
sleap-nn predict \
    --data_path video.mp4 \
    --model_paths models/multi_class_bottomup/ \
    -o predictions.slp

# Top-down ID: centroid model + multi-class centered-instance model
sleap-nn predict \
    --data_path video.mp4 \
    --model_paths models/centroid/ \
    --model_paths models/multi_class_topdown/ \
    -o predictions.slp
```

`sleap-nn predict` auto-detects the model type from the model directory and
assigns each predicted instance to its identity class. The output `.slp` carries
the predicted track (identity) on every instance, so no proofreading-by-motion
step is required.

---

## References

- [Choosing a Model](../reference/models.md#choosing-a-model) — when to pick ID models vs. standard pose + tracking
- [Tracking](tracking.md) — the unsupervised alternative
- [Model config reference](../configuration/model.md) — full `multi_class_*` head schemas
- [SLEAP paper](https://www.nature.com/articles/s41592-022-01426-1) — architecture background for supervised identity
