# Distillation

`trainer_config.distillation` trains a model with help from another, already
trained model, the **teacher**. It follows DWPose ([Yang et al. 2023](https://arxiv.org/abs/2307.15880)):

- **Stage 1, teacher → student.** A big or high-resolution teacher (say ViT-B at
  384×288) teaches a smaller or cheaper student (ViT-S at 256×192). The student
  keeps its usual supervised loss and also learns to match the teacher's
  features and heatmaps.
- **Stage 2, head self-distillation.** The trained student becomes its own
  teacher: its backbone is frozen, its head is re-initialized and trained to match
  the old head's heatmaps. DWPose reports a further small gain for little cost.

Distillation is off by default. It works for `single_instance`,
`centered_instance` and `multi_class_topdown` models (the confidence-map head;
`multi_class_topdown`'s class vectors are trained as usual).

---

## Quick start

Train the student with a teacher:

```yaml
trainer_config:
  distillation:
    teacher: /runs/vitpose_b_384     # model directory (best.ckpt + training_config.yaml)
    teacher_input_scale: 1.5         # 384x288 teacher, 256x192 student
```

```bash
sleap-nn train student.yaml
```

Then refine the student's head with its own predictions:

```yaml
trainer_config:
  max_steps: 10000                   # stage 2 is short
  distillation:
    stage: 2
    teacher: /runs/vit_s_stage1      # the stage-1 student
```

---

## How it works

### Stage 1

The loss is

```text
loss = supervised + r(t) * (alpha * L_feat + beta * L_heatmap)
```

- **`L_heatmap`** is the MSE between the student's and the teacher's confidence
  maps over the nodes the two skeletons share **by name**. A COCO-17 teacher can
  teach a student with a larger skeleton; nodes the teacher lacks only get the
  supervised loss. The teacher's maps are resized to the student's output grid.
- **`L_feat`** is the MSE between the teacher's deepest backbone feature and the
  student's, mapped to the teacher's channel count by a trainable 1×1 conv
  (the *projector*). For a plain-ViT `pretrained` backbone the feature is the
  last tapped token map (after its tap norm); for other backbones it is the
  bottleneck. The teacher's feature is resized to the student's grid.
  `alpha: 0` skips this term and builds no projector.
- **`r(t) = 1 - t / T`** (`decay: linear`) fades both terms out over the run's
  optimizer steps, so the end of training is purely supervised. `decay: none`
  keeps them constant.

`loss` and `train/loss` still log the **supervised** loss, so runs with and
without distillation stay comparable. The distillation terms are logged as
epoch means: `train/distill_heatmap`, `train/distill_feat` and
`train/distill_weight` (`r(t)`). Validation is unchanged (`val/loss` is the
supervised loss).

### Stage 2

At the start of the run:

1. The student is loaded from `teacher` (the stage-1 model; leave `teacher` unset
   to keep the weights `model_config.init_from` loaded).
2. A frozen copy of the network becomes the teacher.
3. Everything outside `trainable` (default `[head]`) is frozen, and `trainable` is
   re-initialized.

The loss is `gamma * L_heatmap`, with no supervised loss (`supervised_weight: 0`,
as in DWPose) and no decay (`decay` defaults to `none` at stage 2, since the
distillation term is the whole loss). When the whole backbone is frozen it stays
in eval mode (no drop path, no BatchNorm updates) and the teacher's head reads the
student's own backbone features, so the backbone runs once per step.

With a ViTPose-style head (`vit_decoder: classic`) the deconvolution decoder is
part of the backbone, so `trainable: [head]` retrains only the final 1×1 conv. To
retrain the decoder too:

```yaml
trainer_config:
  distillation:
    stage: 2
    teacher: /runs/vit_s_stage1
    trainable: [head, "backbone.vit_dec.*"]
```

`trainable` takes the same selectors as `model_config.init_from` (see
[Selectors](init-from.md#selectors)). Stage 2 cannot be combined with
`trainer_config.freeze_schedule`.

---

## Partially labeled data

With per-node loss masking active (see [Partially labeled skeletons](partial-labels.md)),
the heatmap term uses the same `node_weights` as the supervised loss: a node the
sample's source did not label (or, under `unlabeled_points: ignore`, a node
without a location) gets no teacher signal either. Set
`mask_unlabeled_nodes: false` to distill every shared node everywhere, which is
DWPose's choice: the teacher then fills in nodes the annotators left out.

---

## Teacher resolution

A teacher trained at a higher resolution should see more pixels than the
student. `teacher_input_scale` resamples the student's crop by that factor
before it goes to the teacher (rounded to a multiple of the teacher's
`max_stride`), and resizes the teacher's outputs back to the student's grids.
This only upsamples the student's pixels: the teacher sees the same image
content at a larger size, not extra detail.

A data pipeline can instead put the same crop window, rendered at the teacher's
resolution, in the batch under `teacher_image`; the teacher then uses it as is
(`teacher_input_scale` is ignored). The built-in data pipeline does not emit it
yet.

The log warns when the teacher's heatmaps, after the resize, are much wider or
narrower than the student's targets (`sigma`). A 384-crop teacher with
`sigma: 3` teaching a 256-crop student with `sigma: 2` is consistent.

---

## The teacher at training time

- It is frozen, in eval mode, and run under bf16 autocast on CUDA
  (`teacher_precision: "32"` runs it in the training precision instead).
- It is not part of the student module: it is not in the `state_dict`, the
  optimizer, EMA or checkpoints. It follows the student across devices.
- Only the teacher's `model_config` is read, so teachers trained with data or
  trainer options this version lacks still load. Its checkpoint `state_dict` is
  loaded as is: for a teacher trained with weight EMA, that is the EMA weights.
- Under DDP each rank loads its own copy; DDP neither wraps nor syncs it. The
  projector is a normal trainable parameter and is all-reduced like the rest of
  the student.

The projector is training-only. Checkpoints keep it under
`checkpoint["distillation"]` rather than in `state_dict`, so they load into a
plain model for inference, export or `init_from` with no extra keys. Resuming a
distillation run (`resume_ckpt_path`) restores it. Resuming stage 2 needs
`teacher` set, since the resumed weights are not the teacher.

---

## Options

| Key | Default | Description |
|-----|---------|-------------|
| `teacher` | `null` | Teacher model directory or `.ckpt` next to its `training_config.yaml`. Required for stage 1 |
| `stage` | `1` | `1` teacher → student, `2` head self-distillation |
| `alpha` | `1.0` | Feature loss weight (stage 1); `0` disables it |
| `beta` | `1.0` | Heatmap loss weight (stage 1) |
| `gamma` | `1.0` | Heatmap loss weight (stage 2) |
| `supervised_weight` | `null` | Supervised loss weight; `null` = 1 at stage 1, 0 at stage 2 |
| `decay` | `null` | `linear` (`r(t) = 1 - t/T`) or `none`; `null` = `linear` at stage 1, `none` at stage 2 |
| `teacher_input_scale` | `1.0` | Student crop → teacher input resampling factor |
| `mask_unlabeled_nodes` | `true` | Weight the heatmap term by `node_weights` |
| `trainable` | `null` | Stage 2: what is re-initialized and trained (`[head]`) |
| `teacher_precision` | `bf16` | `bf16` or `32` |

The feature and heatmap losses have different scales: heatmap MSE is about the
size of the supervised loss (~1e-3), while the feature MSE of LayerNorm'd ViT
tokens starts near 0.05–1. `alpha` and `beta` are not tuned yet; check the
`train/distill_*` logs to see what each term contributes.
