# Initializing from Another Model

`model_config.init_from` starts a new training run from the weights of another
sleap-nn model, even when the two models differ: a grown or reordered skeleton, a
different crop size, a re-initialized head. It copies what fits, leaves the rest at
its fresh init, and logs a report of every tensor it touched.

Use it to:

- **Grow a skeleton.** Seed a 46-node model from a 17-node COCO model: the 17 shared
  nodes keep their trained head channels, the 29 new nodes start fresh.
- **Re-initialize part of a model.** Keep a trained backbone and reset the head, or
  reset the last transformer blocks.
- **Fine-tune at another resolution.** Load a 256-crop model into a 384-crop run.

For continuing an interrupted run (same model, optimizer state included), use
`trainer_config.resume_ckpt_path` instead; see
[Resuming & Fine-Tuning](resume-finetune.md). `init_from` is ignored when
`resume_ckpt_path` is set.

---

## Quick start

```yaml
model_config:
  init_from:
    path: /runs/coco17_vit_s     # a model directory, a .ckpt or a raw state dict
```

That is all it takes to grow a skeleton: per-node head channels are matched by
**node name**, so the new model's `part_names` can add, drop or reorder nodes.

The log then shows what happened. Loading a COCO-17 ViT-S model into the 46-node
union skeleton:

```text
Weight transfer from /runs/vitpose_s_coco_210e:
  loaded          176 tensors
  remapped          2
      model.head_layers.0.CenteredInstanceConfmapsHead.0.weight: 17/46 channels copied by node name from 17 source channels
      model.head_layers.0.CenteredInstanceConfmapsHead.0.bias: 17/46 channels copied by node name from 17 source channels
  resized           0
  reinitialized     1
      model.head_layers.0 (channels): 29 new node(s) keep fresh init: left_big_toe, right_big_toe, ...
  skipped           0
  missing           0
  unexpected        0 (source-only keys)
```

| Line | Meaning |
|------|---------|
| `loaded` | Same key and shape: copied as-is. |
| `remapped` | A head's per-node channels, copied by node name. |
| `resized` | A learned positional embedding resampled to a new token grid. |
| `reinitialized` | New nodes (or modules named in `reinit`) left at a fresh init. |
| `skipped` | Present in both models but the shapes differ: the target keeps its init. Logged as a WARNING. |
| `missing` | In the new model but not in the source: kept at init. |
| `unexpected` | In the source but not in the new model: ignored. |

---

## All options

```yaml
model_config:
  init_from:
    path: /runs/coco17_vit_s
    include: null                 # selectors to load (null = everything)
    exclude: null                 # selectors to leave at their init
    remap_nodes: true             # copy head channels by node name
    node_map: null                # {target node: source node} overrides
    source_part_names: null       # source node names, if the checkpoint has none
    interpolate_pos_embed: true   # resize learned position embeddings
    reinit: null                  # selectors to re-initialize AFTER loading
    reinit_layer_scale: 1.0e-5    # LayerScale for re-initialized ViT blocks
    strict: false                 # raise if any tensor had to be skipped
```

| Option | Default | Description |
|--------|---------|-------------|
| `path` | `null` | Model directory (its `best.ckpt`), a Lightning `.ckpt`, or a raw `state_dict` file. `null` loads nothing, but `reinit` still applies. |
| `include` | `null` | [Selectors](#selectors) of the weights to load. `null` means everything. |
| `exclude` | `null` | Selectors of weights to leave at their init. |
| `remap_nodes` | `true` | Copy per-node head channels by node name. With `false`, they are copied by position when the shapes match. |
| `node_map` | `null` | `{target node: source node}`. Seeds a new node from a related source node, or bridges a renamed node. |
| `source_part_names` | `null` | Source node names, for checkpoints that record none (see [Where node names come from](#where-node-names-come-from)). |
| `interpolate_pos_embed` | `true` | Bicubically resize learned positional embeddings on a different square grid, keeping CLS/register tokens. |
| `reinit` | `null` | Selectors re-initialized after loading, e.g. `["head"]` or `["blocks[-2:]", "head"]`. |
| `reinit_layer_scale` | `1e-5` | LayerScale value for re-initialized ViT blocks, so they start close to identity on the residual stream. |
| `strict` | `false` | Raise instead of warning when a tensor is skipped for a shape mismatch. |

`init_from` runs after every other weight source: `init_weights`, the backbone's
own pretrained weights, and `pretrained_backbone_weights` /
`pretrained_head_weights`.

---

## Selectors

`include`, `exclude`, `reinit` and `trainer_config.freeze_schedule.targets` all take
the same selectors:

| Selector | Picks |
|----------|-------|
| `all` | The whole model. |
| `backbone` | The backbone (encoder + decoder). |
| `encoder` | The backbone's encoder, as `freeze: true` defines it: the HuggingFace model of a `pretrained` backbone, the ImageNet encoder of `convnext` / `swint`, the stem and encoder blocks of a `unet`. |
| `decoder` | The backbone's decoder. |
| `neck` | A plain ViT's pyramid neck and per-tap norms. |
| `head` / `heads` | The output heads. |
| `embeddings` | A plain ViT's patch and position embeddings and prefix (CLS/register) tokens. |
| `blocks[a:b]` | A Python slice over a plain ViT's transformer blocks, e.g. `blocks[-2:]` or `blocks[:6]`. |
| a glob | State-dict keys, e.g. `backbone.enc.*norm*` (with or without the `model.` prefix). |

`embeddings` cannot be re-initialized.

---

## Recipes

### Grow a skeleton and seed new nodes

```yaml
model_config:
  init_from:
    path: /runs/coco17
    node_map:
      left_heel: left_ankle     # start the heel channel from the ankle channel
```

New nodes without a `node_map` entry keep their fresh init.

### Keep the backbone, re-initialize the head

```yaml
model_config:
  init_from:
    path: /runs/previous
    reinit: [head]
```

`reinit` uses the model's `init_weights` scheme (`xavier` or `default`) for convs
and linear layers. ViT blocks picked by `blocks[a:b]` get ViT-style init instead:
truncated normal (std 0.02), zero bias and `reinit_layer_scale` for LayerScale.

!!! note "Head init and the ViTPose `classic` head"
    In a 1,500-step A/B on COCO, `init_weights: xavier` on the ViTPose `classic`
    head learned half as fast as `init_weights: default` (mOKS 0.24 vs 0.48). Prefer
    `default` for that head until a full run says otherwise.

### Re-initialize the last transformer blocks

```yaml
model_config:
  init_from:
    path: /runs/previous
    reinit: ["blocks[-2:]", neck, decoder, head]
```

If the backbone taps fewer layers than it has, blocks after the last tapped layer
do not run, and re-initializing them has no effect (a warning says so).

### Load only the backbone

```yaml
model_config:
  init_from:
    path: /runs/previous
    include: [backbone]
```

### Fine-tune at a higher resolution

Nothing special is needed. HuggingFace DINOv2 interpolates its stored position
embeddings on every call and timm DINOv3 uses RoPE, so a 256-crop checkpoint loads
into a 384-crop model with nothing skipped. Learned position embeddings stored at a
different grid size are resized (`interpolate_pos_embed`).

---

## Where node names come from

Name matching needs the source model's node names. sleap-nn looks, in order, at:

1. `sleap_nn_heads` inside the checkpoint. Checkpoints written by this version on
   record each head's output-channel names there.
2. The `training_config.yaml` next to the checkpoint (older checkpoints).
3. `init_from.source_part_names`, which overrides both.

When none is available (a raw state dict on its own), head channels are copied by
position if the shapes match, and the report carries a WARNING. Check the node
order, or set `source_part_names`.

Heads are paired by head type: a `centered_instance` head seeds a
`centered_instance` head. Channels that are not per-node (PAFs, offsets) are matched
by shape.

---

## Freeze schedule

`trainer_config.freeze_schedule` freezes part of the model for the first steps of
training. The usual use is to warm up a new head on a frozen pretrained encoder,
then fine-tune everything:

```yaml
trainer_config:
  freeze_schedule:
    targets: [encoder]        # default; any selectors from the table above
    unfreeze_at_step: 1000    # null keeps them frozen for the whole run
```

- Freezing happens after the optimizer is built, so frozen parameters keep their
  optimizer groups (with `encoder_lr_scale` / `layer_decay`) and train once
  unfrozen. AdamW skips parameters without a gradient, weight decay included.
- `encoder` also keeps the encoder in eval mode while frozen (BatchNorm statistics,
  stochastic depth), as `freeze: true` does.
- Do not combine `unfreeze_at_step` with the backbone's `freeze: true`. That keeps
  the encoder out of the optimizer, so it could never train; the run refuses to
  start.
- Under DDP, `find_unused_parameters` is turned on automatically, because frozen
  parameters get no gradient. An explicit
  `trainer_strategy: ddp_find_unused_parameters_false` is refused, and so are
  multi-process strategies other than DDP.
- A run resumed after `unfreeze_at_step` does not freeze again.

---

## Changes to `pretrained_backbone_weights` / `pretrained_head_weights`

The `.ckpt` forms of these keys now go through the same loader (`include:
[backbone]` and `include: [head]`), so they log the same report and match head
channels by node name. `pretrained_head_weights` no longer fails when the skeleton
grows, and no longer copies a reordered skeleton's channels blindly.

**Behaviour change:** a tensor whose shape differs used to raise an error. It is now
skipped with a WARNING, and the target keeps its init. Set
`model_config.init_from.strict: true` (no `path` needed) to fail on any mismatch
again. Legacy SLEAP `.h5` weights load as before.
