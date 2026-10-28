# DETR box detector (`bbox_detr`)

`bbox_detr` is a box detector modeled on [RF-DETR](https://github.com/roboflow/rf-detr)
(Robinson et al., ICLR 2026). It predicts a **set** of boxes per frame: each of a
fixed number of queries either claims one animal or predicts "nothing". Training
matches queries to animals one to one (Hungarian matching), so the model learns
not to predict duplicates, and there is no NMS step at inference. Two animals that
touch or overlap stay two boxes; with CenterNet or SSD, NMS or merged heatmap
peaks can turn them into one.

Use it as:

- a standalone detector (boxes written as `sio.PredictedBoundingBox`), or
- the first stage of a top-down pose model, in place of the centroid model
  ([Top-down with a box detector](bbox-topdown.md)), or
- the source of detector boxes for training crops (`sleap-nn detect-boxes`).

It is detection only: no keypoint or mask heads.

| | `bbox_centernet` / `bbox_ssd` | `bbox_detr` |
|---|---|---|
| Backbone | UNet / ConvNeXt / SwinT | pretrained plain ViT (DINOv3, or RF-DETR's windowed DINOv2) |
| Output | dense maps, decoded + NMS | 100–300 boxes with scores, no NMS |
| Touching animals | can merge | stay separate |
| GPU memory (training) | 1–4 GB | ~4–8 GB for ViT-S at batch 4–8, more for ViT-B or large inputs |
| Data needed | a few hundred frames | a few hundred frames, but more epochs (> 100) |

The [Bounding-box detection guide](bbox-detection.md) covers what all `bbox_*`
models share: where training boxes come from, evaluation (`eval/val/bbox_mAP`,
`sleap-nn eval --match_method bbox`) and the output format.

## How it works

```
frame (letterboxed to max_height x max_width)
  -> ViT (DINOv3 ViT-S/B, or windowed DINOv2-S), 16 x 16 px tokens; blocks 3, 6, 9, 12 tapped
  -> projector: concat the 4 taps -> C2f -> one 256-channel map at stride 16
  -> two-stage proposals: every token proposes a box; the top num_queries become queries
  -> decoder (2-4 layers): self-attention between queries, deformable attention into the map
  -> per query: class scores (sigmoid) + box (cx, cy, w, h)
```

Training uses RF-DETR's recipe: 13 query groups (Group DETR), IoU-aware BCE class
loss plus L1 and GIoU box losses on every decoder layer and on the proposals,
AdamW, a lower learning rate for the backbone with layer-wise decay, weight EMA,
gradient clipping and bf16. Each class bias starts at the 0.01 prior, so the model
starts by predicting "nothing" everywhere.

## Train an animal detector from a DINO backbone

"From scratch" here means: pretrained DINOv3 ViT, randomly initialized projector
and decoder, no detection checkpoint. This is the default for animals.

The quickest start is the config generator, which sizes everything from your
labels:

```bash
sleap-nn config labels.slp --auto --pipeline bbox_detr -o detector.yaml
# or the pair (detector + box-scaled centered-instance model):
sleap-nn config labels.slp --auto --pipeline topdown_bbox_detr -o mice.yaml
#   -> mice_bbox_detr.yaml + mice_centered_instance.yaml
sleap-nn train detector.yaml
```

Or start from the sample config
[`config_bbox_detr_dinov3_s_mouse.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_detr_dinov3_s_mouse.yaml),
which explains each value. It was written for the SLEAP paper's 2-mouse dataset
(1,180 training frames at 1280 x 1024). The parts that matter:

```yaml
data_config:
  bbox_source: instances   # one box per pose instance: its visible nodes
  bbox_padding: 9.0        # px on every side; 5% of the median animal size
  preprocessing:
    ensure_rgb: true       # DINOv3 reads RGB; gray frames are repeated
    max_height: 512        # input size: see "Choosing the input size"
    max_width: 640
    resize_mode: letterbox
  augmentation_config:
    geometric: null        # no rotation: it inflates keypoint boxes
    detr:                  # RF-DETR's crop / flip / multi-scale, scaled to the input
      crop_p: 0.5
      shortest_sides: [512]
      crop_range: [341, 512]
      flip_p: 0.5
      multi_scale: true
      multi_scale_steps: 5
model_config:
  backbone_config:
    pretrained:
      model_name: timm/vit_small_patch16_dinov3.lvd1689m
      weights: true
      mode: tokens
      layers: [3, 6, 9, 12]
      drop_path_rate: 0.1
      max_stride: 32
  head_configs:
    bbox_detr:
      detr:
        categories: null   # one class, filled from the boxes ("object")
        num_queries: 100
        num_select: 100
        dec_layers: 3
trainer_config:
  max_epochs: 200
  min_train_steps_per_epoch: 1
  optimizer_name: AdamW
  optimizer: {lr: 1.0e-4, weight_decay: 1.0e-4, encoder_lr_scale: 1.5, layer_decay: 0.8}
  lr_scheduler: {reduce_lr_on_plateau: null}   # constant LR
  early_stopping: {stop_training_on_plateau: false}
  ema: {enabled: true, decay: 0.993, warmup_steps: 100}
  gradient_clip_val: 0.1
  precision: bf16-mixed
  eval: {enabled: true, frequency: 1}           # keeps the best eval/val/bbox_mAP
```

The `pretrained` backbone needs the `backbones` extra (`pip install
"sleap-nn[backbones]"`) and downloads the DINOv3 weights from the HF Hub on first
use (see [Pretrained Backbones](pretrained-backbones.md) for offline use).

**Boxes.** A pose project has no boxes, so they are derived from the poses:
the tight box of each instance's visible nodes, plus `bbox_padding` pixels on
every side. Keypoints sit inside the animal's outline (a mouse's snout, ears and
tail base do not reach its flanks), so a little padding makes the box cover the
body. Use the same `bbox_source` and `bbox_padding` in the centered-instance
model of a top-down pair: its crops are then the same boxes the detector learns.

**Classes.** Leave `categories: null` for one class. Boxes derived from poses
have no category, so the class is named `object`. If your labels carry user
boxes with categories, each category becomes a class (only boxes of the listed
categories are ground truth when you set `categories` yourself).

### Choosing the input size

The ViT sees the frame as 16 x 16 px tokens, and the detector reasons about
animals in tokens. An animal under about 3 tokens across is often missed. The
config generator's rule (`recommend_detr_resolution`):

1. Take the animal size as `sqrt(w * h)` of the padded training boxes (it varies
   less with the animal's orientation than the longer side does).
2. Pick the smallest scale at which the small animals (5th percentile) are at
   least 3 tokens (48 px) across and the median animal at least 4 tokens, never
   above the frame's own resolution.
3. Round the letterboxed frame up to multiples of 32; at least 384 px on the long
   side, at most 1280 px.

| Dataset | Frames | Animals (p5 / median, padded) | Input | Small / median animal |
|---|---|---|---|---|
| SLEAP mice HC (2 mice) | 1280 x 1024 | 97 / 191 px | 640 x 512 | 3.0 / 6.0 tokens |
| MABe 2023 (2 mice) | 720 x 480 | 79 / 104 px | 448 x 320 | 3.1 / 4.0 tokens |
| MoSAIC (6 mice, large arena) | 3934 x 3490 | 87 / 169 px | needs 2170; capped at 1280 x 1152 | 1.8 / 3.4 tokens |

Compute cost grows with the number of tokens (global attention: with the square
of it, though flash attention keeps memory linear), so the rule does not go
bigger than it needs to. The `Input:` line of `sleap-nn config --auto` output
(and the TUI's DETR panel) shows the result and a warning when the cap was hit.
For that case see [Large arenas](#large-arenas-and-tiling).

`num_queries` must not exceed the tokens of the smallest training size (100
queries need at least 160 x 160 px). Multi-scale training drops sizes that would
have fewer tokens than queries.

### Backbone sizes

`ConfigGenerator.detr(preset=...)` (or the TUI's DETR panel) switches the
backbone and decoder. Parameter counts are for one class and 100 queries; GPU
memory is the generator's estimate for training at the preset's default input with
multi-scale on (calibrated on a measured RF-DETR-M run; expect +/- 30%).

| Preset | Backbone | Decoder layers | Default input | Parameters | GPU memory, batch 4 / 8 |
|---|---|---|---|---|---|
| `dinov3_s` (default) | DINOv3 ViT-S/16 | 3 | from the data (640 above) | 30.7M | 4.4 / 8.1 GB at 640 |
| `dinov3_b` | DINOv3 ViT-B/16 | 4 | from the data (704 for people) | 96.7M | 10.4 / 18.9 GB at 704 |
| `rfdetr_n` | windowed DINOv2-S | 2 | 384 | 29.5M | 2.4 / 4.1 GB |
| `rfdetr_s` | windowed DINOv2-S | 3 | 512 | 31.1M | 3.3 / 6.0 GB |
| `rfdetr_m` | windowed DINOv2-S | 4 | 576 | 32.7M | 3.9 / 7.2 GB |
| `rfdetr_l` | windowed DINOv2-S | 4 | 704 | 32.9M | 5.1 / 9.5 GB |

Without `init_from`, the `rfdetr_*` presets start the windowed DINOv2-S from the
plain `facebook/dinov2-small` weights with a fresh decoder. With `init_from`
pointing at an imported RF-DETR model they fine-tune it (next section).

## Import RF-DETR

RF-DETR's Apache-2.0 checkpoints (`nano`, `small`, `medium`, `large`) convert to
sleap-nn model directories. The other RF-DETR sizes (the XL / 2XL and "plus"
models) are under a different license (PML 1.0) and are not supported.

```bash
sleap-nn import-rfdetr medium -o models/rfdetr-medium      # downloads Roboflow/rf-detr-medium
sleap-nn predict -m models/rfdetr-medium -i video.mp4      # 91 COCO-id classes
```

The import needs `huggingface_hub` and `safetensors` (the `backbones` extra). The
output loads like a trained `bbox_detr` model. It keeps RF-DETR's square
**squash** resize (`resize_mode: squash`) at the checkpoint's resolution (384 /
512 / 576 / 704 for N / S / M / L): RF-DETR's weights degrade with letterboxing.
The imported RF-DETR-M scores 54.7 box AP on COCO val2017, matching RF-DETR's
published number.

## Fine-tune to new classes

Start a training run from any `bbox_detr` model with `model_config.init_from`:
tensors with the same name and shape are copied, and the class heads' rows are
copied **by class name**. Keep the source's architecture (backbone, `dec_layers`,
`num_queries`) to reuse all of it: a different query count leaves the learned
query embeddings fresh. A class that the source model does not have starts
fresh at the 0.01 prior; the rest of the detector (backbone, projector, decoder,
box heads) is reused.

```yaml
data_config:
  preprocessing:
    max_height: 576
    max_width: 576
    resize_mode: squash          # as the imported weights were trained
model_config:
  init_from:
    path: models/rfdetr-medium
    # Optional: seed rows from related source classes, {target: source}.
    # __extra__ is the never-positive extra logit; RF-DETR's person row is a good seed.
    node_map:
      __extra__: person
  backbone_config:
    dinov2_windowed: {...}       # as in the imported model's training_config.yaml
  head_configs:
    bbox_detr:
      detr:
        categories: [mouse]      # "mouse" is new: its row starts fresh
        dec_layers: 4
        num_queries: 300         # RF-DETR's; fewer and its query embeddings start fresh
        num_select: 300
```

The config generator writes this for you (an `rfdetr_*` preset with `init_from`
switches to squash and 300 queries):

```python
from sleap_nn.config_generator import ConfigGenerator

gen = ConfigGenerator.from_slp("labels.slp").auto().pipeline("bbox_detr")
gen.detr(preset="rfdetr_m", init_from="models/rfdetr-medium").save("detector.yaml")
```

Worked examples:
[`config_bbox_detr_rfdetr_m_person.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_detr_rfdetr_m_person.yaml)
(RF-DETR-M cut to a person-only COCO detector, 20 epochs at 4 x 4 images per step).
RF-DETR's own fine-tuning advice applies: if training does not converge, halve the
learning rate.

Another `bbox_detr` model works as the source the same way, e.g. a human detector
to start a mouse detector. Its backbone must have the same shape (same preset).

### Our human detector (two stages)

The human detector trains in two stages on DINOv3 ViT-B/16 at 704 px, letterboxed:

1. [`config_bbox_detr_dinov3_b_person.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_detr_dinov3_b_person.yaml):
   fresh projector and 4-layer decoder on the person boxes of Objects365 +
   Open Images, 12 epochs at 4 x 8 images per step, lr 2e-4.
2. [`config_bbox_detr_dinov3_b_person_stage2.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_detr_dinov3_b_person_stage2.yaml):
   every tensor from stage 1 (`init_from`), then COCO + CrowdHuman + CrowdPose +
   JRDB-Pose + Human-Art mixed per sample by `trainer_config.balanced_sampler`
   (`per_sample: true`, weights 0.4 / 0.2 / 0.1 / 0.1 / 0.2), 50 epochs of 100K
   samples, lr 1e-4.

Only boxes of the configured categories are ground truth (`categories:
[person]`): faces, heads, crowd and ignore regions in those datasets are skipped,
and frames left with no person box are not sampled.

## Top-down pose with a DETR first stage

Train the detector and a box-scaled centered-instance model (the config generator's
`topdown_bbox_detr` writes both), then pass both to `predict`:

```bash
sleap-nn train mice_bbox_detr.yaml
sleap-nn train mice_centered_instance.yaml
sleap-nn predict -i video.mp4 -m models/mice_bbox_detr -m models/mice_centered_instance
sleap-nn eval --ground_truth_path test.slp --predicted_path test.predictions.slp
```

Each detected box is padded, squared and resized to the instance model's crop
size (`crop_mode: bbox`), and kept in the output linked to the pose instance it
seeded. `--peak_threshold` is the detection score threshold and `--max_instances`
caps the boxes per frame. Everything else is as in
[Top-down with a box detector](bbox-topdown.md).

## Detector boxes for training crops

At inference the instance model sees crops from detector boxes, which are
slightly off from the ground-truth boxes it trained on. `sleap-nn detect-boxes`
runs a detector over the labeled frames and stores its boxes, matched to the
instances, so the instance model can train on detector-box crops:

```bash
sleap-nn detect-boxes -i train.pkg.slp -m models/mice_bbox_detr --category mouse
# -> train.pkg.bboxdetr.slp: one PredictedBoundingBox per detection (score >= 0.3),
#    linked to its matched instance; match statistics in <output>.json
```

Matching is Hungarian on IoU per frame, against each instance's user box or, in a
pose project, its keypoint extent. A single-class detector's class is used as
`--category`, which also names the new boxes. Then train the instance model on the
new file with the warp data path and `crop_bbox_source: detector` (see
[Detector-box crops](warp-data-path.md#detector-box-crops)).

## Large arenas and tiling

When the token rule asks for more than 1280 px (small animals in a large arena),
you have three options today:

- **Raise the input size.** Set `max_height` / `max_width` past the cap
  (`ConfigGenerator.detr(max_side=...)`). Memory grows with the token count: the
  estimate for ViT-S at batch 4 is ~11 GB at 1280 px and ~23 GB at 1920 px;
  lower the batch size to fit.
- **Crop the arena** to the area the animals can reach, if the camera is fixed.
- **Use a centroid first stage with tiling** ([Tiling](tiling.md)) for frames
  far beyond that.

Tiled inference is not implemented for `bbox_*` models in `sleap-nn predict` yet.
`sleap-nn detect-boxes --tile_width W` has a full-height tiled mode for very wide
frames (panoramas), merging the tiles' boxes with an edge-aware NMS.

Training already sees zoomed-in views: half of the training samples go through
RF-DETR's crop branch, a random square crop of 2/3 to all of the frame's short
side, resized to the input size.

## Small datasets

Advice for a few hundred to a few thousand labeled frames, as in the mouse sample:

- **Train long.** The RF-DETR paper says small datasets need more than 100 epochs.
  The sample uses 200 (one epoch = one pass over the frames,
  `min_train_steps_per_epoch: 1`). Keep the LR constant and early stopping off; the
  EMA weights keep improving after the loss flattens, and the checkpoint with the
  best `eval/val/bbox_mAP` is kept.
- **Use fewer queries.** 100 instead of RF-DETR's 300. Frames hold a few
  animals; fewer queries train faster and need fewer tokens.
- **Prefer ViT-S.** DINOv3 ViT-S with a 3-layer decoder has 31M parameters, most
  of them pretrained. ViT-B is worth it for varied scenes (many setups, lighting,
  species), not for one arena.
- **No rotation.** Rotating a keypoint box makes it bigger than the rotated
  animal. RF-DETR's crop and flip, multi-scale and brightness / contrast give
  enough variety. Keep `geometric: null` explicitly: an omitted section falls back
  to the default (+/- 15 degree rotation).
- **Check the boxes.** Look at a few training frames with their derived boxes (e.g.
  render `LabeledFrame.bboxes`); raise `bbox_padding` if boxes cut off the body.
- **Keep the batch, adjust the LR.** The sample uses batch 8 at lr 1e-4
  (RF-DETR: 16 per step at 1e-4). Halve the LR if the loss spikes or
  `bbox_mAP` stays near 0 after 20-30 epochs.
- **Start from another detector** when one exists for a similar domain
  (`init_from`, previous section). Whether a human detector helps a mouse
  detector is one of the M5 experiments.

## Configuration reference

All `bbox_detr` keys are in the configuration reference:
[`head_configs.bbox_detr.detr`](../configuration/model.md#bbox_detrdetr),
[`backbone_config.dinov2_windowed`](../configuration/model.md#dinov2windowedconfig),
[`backbone_config.pretrained.mode: tokens`](../configuration/model.md#pretrainedconfig),
[`init_from.node_map`](../configuration/model.md#initfromconfig),
[`preprocessing.resize_mode`](../configuration/data.md#preprocessingconfig),
[`augmentation_config.detr`](../configuration/data.md#detraugmentationconfig-augmentation_configdetr),
[`trainer_config.balanced_sampler.per_sample`](../configuration/trainer.md#trainerconfig)
and the optimizer's [`encoder_lr_scale` / `layer_decay`](../configuration/trainer.md#optimizerconfig).

## Limits

- Detection only: no per-query keypoint or mask heads.
- No tiled inference in `sleap-nn predict` (see above).
- `sleap-nn export` traces `bbox_detr` to ONNX, but `sleap-nn predict` runs
  exported CenterNet / SSD detectors only; run an exported `bbox_detr` graph in
  your own runtime for now.
- `precision: bf16-mixed` needs an Ampere or newer GPU; use `"32"` otherwise.
