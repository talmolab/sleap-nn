# Bounding-box detection

Train a whole-frame **object detector** that predicts one axis-aligned box (and a
class) per animal, and write the predictions to `.slp` as
`sio.PredictedBoundingBox` on `LabeledFrame.bboxes`. Two dense detection styles share one
data pipeline, one output type and one COCO-style evaluation (a third,
`bbox_detr`, predicts a set of boxes with a pretrained ViT and no NMS: see the
[DETR box detector guide](bbox-detr.md)):

| | `bbox_centernet` (CenterNet-style) | `bbox_ssd` (SSD-style) |
|---|---|---|
| Idea | anchor-free: a box = a center peak + its regressed size | anchor-based: classify + refine a dense set of prior boxes |
| Heads | center heatmap (1 channel per class) + size (w, h) + sub-cell offset, all at one stride | one multi-scale head over pyramid levels (default strides 8/16/32): per anchor, class logits (incl. background) + 4 box deltas |
| Targets | Gaussian peak at each box center (CenterNet's size-adaptive radius), size/offset at the center cell | anchors matched by IoU (each box's best anchor always positive), SSD `(cx, cy, w, h)` encoding |
| Loss | penalty-reduced focal loss (normalized by #objects) + masked L1 | softmax CE with 3:1 hard-negative mining + smooth L1 |
| Decoding | 3x3 max-pool peak NMS, top-k, threshold (IoU-NMS optional) | softmax, threshold, top-k, class-wise IoU-NMS (0.45) |
| Typical use | one size range, crowded scenes, simplest knobs | wide size ranges, many aspect ratios |

Sample configs:
[`config_bbox_centernet_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_centernet_unet.yaml),
[`config_bbox_ssd_unet.yaml`](https://github.com/talmolab/sleap-nn/blob/main/docs/sample_configs/config_bbox_ssd_unet.yaml).

## Training data: boxes, or poses / masks to derive them from

Ground-truth boxes are `sio.UserBoundingBox` objects on `LabeledFrame.bboxes`. You
do **not** need to annotate boxes if you already have poses or masks:
`data_config.bbox_source` derives them in memory through sleap-io at load time.

| `data_config.bbox_source` | Boxes come from |
|---|---|
| `auto` (default) | per labels file: user boxes if there are any, else user pose instances, else user masks |
| `bboxes` | `LabeledFrame.bboxes` only |
| `instances` | a tight box around each user instance's visible points (`sio.Instance.to_bbox`) |
| `masks` | each user mask's extent (`sio.SegmentationMask.to_bbox`) |

`data_config.bbox_padding` (pixels, default `0`) inflates derived boxes on every
side — keypoints sit inside the animal's outline, so a tight keypoint box is a bit
smaller than the animal. Existing user boxes are never padded.

To make a box-only dataset from poses yourself (what the benchmark below does):

```python
import sleap_io as sio

labels = sio.load_slp("poses.pkg.slp")
for lf in labels.user_labeled_frames:
    lf.bboxes = [inst.to_bbox(mode="tight") for inst in lf.user_instances]
    lf.instances = []
sio.save_slp(labels, "boxes.pkg.slp", embed="user")
```

**Classes.** Each box's `category` (a `sio.Category`) is its class. The head's
`categories` list is filled in at training setup from the training boxes (first
seen first) and saved with the model; boxes without a category form the single
class `"object"`, which is also what predictions carry (`category=None`).

**Negative frames.** With `data_config.use_negative_frames: true`, frames the user
marked as empty are added as all-background samples.

## Configuration

The quickest start is the config generator, which writes a detector config for any
pose project (boxes are derived from the poses):

```bash
sleap-nn config labels.slp --auto --pipeline bbox_ssd -o detector.yaml   # or bbox_centernet
sleap-nn config labels.slp   # interactive: pick "Bounding Boxes (CenterNet)" or "Bounding Boxes (SSD)"
```

It uses `preprocessing.scale: 0.5`, `output_stride` 4 (CenterNet) / 8 (SSD, with
`max_stride` >= 32), turns rotation augmentation off and evaluation on (so the best
checkpoint is picked on box mAP, see below). The sample configs
`docs/sample_configs/config_bbox_{centernet,ssd}_unet.yaml` spell out every knob:

```yaml
model_config:
  backbone_config:
    unet: {filters: 32, filters_rate: 1.5, max_stride: 32, output_stride: 4}
  head_configs:
    bbox_centernet:
      center: {output_stride: 4}   # categories, sigma, min_overlap, focal_alpha/beta
      size:   {output_stride: 4, loss_weight: 0.1}
      offset: {output_stride: 4, loss_weight: 1.0}
```

```yaml
model_config:
  backbone_config:
    unet: {filters: 32, filters_rate: 1.5, max_stride: 32, output_stride: 8}
  head_configs:
    bbox_ssd:
      ssd:
        output_stride: 8        # finest pyramid level
        num_levels: 3           # levels at strides 8, 16, 32
        anchor_sizes: null      # auto: geometric between the boxes' p5..p95 sqrt(area)
        anchor_scales: [1.0, 1.26, 1.587]
        aspect_ratios: [0.5, 1.0, 2.0]
        pos_iou_threshold: 0.5
        neg_iou_threshold: 0.4
        neg_pos_ratio: 3.0
```

Notes:

- **CenterNet** heads must share one `output_stride`. `sigma: null` (default) uses
  CenterNet's size-adaptive Gaussian radius; a number fixes sigma in input pixels.
- **SSD** binds to the backbone feature maps at `output_stride * 2**i`: decoder
  outputs for the finer levels and the bottleneck for the deepest one. The deepest
  level may not exceed `max_stride` (it is raised to fit). Leave `anchor_sizes:
  null` to size the anchors from your training boxes (in model-input pixels, i.e.
  after `preprocessing.scale`); the resolved sizes are written into the saved
  `training_config.yaml` so inference uses identical anchors.
- **Augmentation:** boxes are carried through the geometric augmentation as their
  four corners and re-boxed afterwards; boxes pushed mostly out of frame are dropped.
  Prefer **little or no rotation**: the axis-aligned box of rotated corners is larger
  than the rotated animal, so rotation inflates the targets. Scale, translation and
  `flip_p` are exact.
- **Zoom-out:** the `zoom_out_*` keys of `augmentation_config.detr` work here as for
  `bbox_detr`: with probability `zoom_out_p` a training frame is shrunk onto a black
  canvas, or onto another training frame (`zoom_out_canvas: frame`), so the detector
  sees animals smaller than the labels show. Off by default; the section's other
  keys are `bbox_detr` only.

  ```yaml
  data_config:
    augmentation_config:
      detr: {zoom_out_p: 0.4, zoom_out_min: 0.3, zoom_out_max: 1.0, zoom_out_canvas: frame}
  ```
- **Tiling** is not supported for detectors in `sleap-nn predict` yet (a clear error
  is raised). `sleap-nn detect-boxes --tile_width` has a full-height tiled mode for
  wide panoramas (see [detector-box crops](warp-data-path.md#detector-box-crops)).

## Training and monitoring

```bash
sleap-nn train --config config_bbox_centernet_unet.yaml \
  --labels train.pkg.slp --val-labels val.pkg.slp
```

Logged per epoch: `train|val/center_loss, size_loss, offset_loss` (CenterNet) or
`train|val/cls_loss, box_loss` (SSD). With `trainer_config.eval.enabled: true`, every
`eval.frequency` epochs the validation set is decoded with the same decoder as
inference and scored with COCO-style metrics, logged to the CSV log, wandb and
`callback_metrics` as `eval/val/bbox_mAP`, `bbox_AP50`, `bbox_AP75`, `bbox_AR`,
`bbox_precision`, `bbox_recall`, `bbox_f1`, `bbox_f1_threshold`, `bbox_mean_iou`,
`bbox_n_tp/fp/fn` (wandb also keeps `best/` summaries). Training visualizations
draw ground-truth boxes in green and predictions in red (CenterNet also overlays its
center heatmap).

**Checkpoint selection.** With evaluation on, `best.ckpt` and early stopping follow
`eval/val/bbox_mAP` (checked on evaluation epochs, so early-stopping `patience` counts
evaluations), not the validation loss: the loss tracks box quality only loosely, and
in the benchmark several runs were still improving their mAP when the loss stopped
them. Choose another metric with `trainer_config.eval.select_metric`: `bbox_mAP`,
`bbox_AP50`, `bbox_AP75`, `bbox_AR`, `bbox_f1`, or `val_loss` for the old behavior.
With evaluation off, detectors select on `val/loss`.

From Python, `sleap_nn.train.train(..., head_configs="bbox_ssd")` builds the same
config; it takes `bbox_source=` / `bbox_padding=` and, like `sleap-nn train`, defaults
`geometry_aug` to rotation, so pass e.g. `geometry_aug="scale"` for a detector.

After training, when checkpoints are saved (`trainer_config.save_ckpt: true`),
every split (and `data_config.test_file_path`) is predicted and evaluated
automatically; metrics are saved as `metrics.<split>.npz/.json` and
logged to the wandb summary as `eval/<split>/bbox_*` (plus `predict_fps`). Val and
test run first. For a large training set, skip the train split
(`trainer_config.eval.post_train_splits: [val, test]`) or score a sample of it
(`post_train_max_train_frames`); see the
[EvalConfig reference](../configuration/trainer.md#evalconfig).

## Experimental training options

Four opt-in options. Each is off by default; a config that does not set it trains,
saves and predicts as before, and its `training_config.yaml` gets no new key.

**Ignore boxes** (`bbox_detr` only). In a partly labelled frame, a detector that
finds an unlabelled animal is trained to call it background. Draw a box of a
dedicated category around such regions and name that category:

```yaml
data_config:
  ignore_box_category: ignore   # boxes of this category are not objects
  ignore_box_min_cover: 0.5     # share of a query's box inside an ignore box
```

Ignore boxes go through the same augmentation as the other boxes (crop, flips,
90 degree turns, zoom-out, rotation) and leave the data path as a separate target.
They are never matched to a query, never counted in the loss normalizer and never
ground truth. In the score loss (RF-DETR's IoU-aware BCE), a query that matches no
object and whose predicted box lies inside one ignore box by at least
`ignore_box_min_cover` of its own area gets no background term, so it is neither
pushed towards background nor towards an object. Matched queries are unaffected.
The box metrics (during training and after it) use the same rule: a detection
that lies inside an ignore box by at least `ignore_box_min_cover` of its own area
(COCO's crowd rule), and matches no labelled box, is neither a hit nor a false
positive. A frame whose only boxes are ignore boxes is a training frame (everything
outside them is background).

The animals can be labelled with poses or masks: with `bbox_source: instances` or
`masks` (or `auto` on a file whose only user boxes are ignore boxes), the object
boxes are derived from them and the ignore boxes are kept. The category must not
be one of the detector's `categories`, nor `object` (the name boxes without a
category get); with `categories` unset it is left out of the categories read from
the labels. With zoom-out onto another training frame, an ignore box of the canvas
frame that the shrunk frame covers by less than half is kept whole, so it can
reach over the shrunk frame (object boxes of the canvas behave the same way).

**Negative frames in validation** (`bbox_*`). Validation normally leaves out
negative frames, so the box metrics during training say nothing about false boxes
on empty scenes. `data_config.val_negative_frames: true` keeps the validation
labels' user-confirmed negative frames (see [Negative frames](negative-frames.md)).
Detections on them count as false positives in `eval/val/bbox_*`, and their
background terms enter `val/loss`. A `bbox_detr` set loss divides by the number of
labelled boxes in the batch, clamped at 1, so a validation batch of only negative
frames has a much larger loss than a batch with boxes. By default the best
checkpoint (`trainer_config.model_ckpt.monitor`), early stopping
(`trainer_config.early_stopping`) and `trainer_config.lr_scheduler.reduce_lr_on_plateau`
follow `val/loss`; with this option on, turn on `trainer_config.eval.enabled` so
checkpoints and early stopping follow a box metric (`eval.select_metric`, e.g.
`bbox_mAP`), and consider a fixed LR schedule. Training logs a warning when the
checkpoint monitor is still `val/loss`. The post-training metrics
(`metrics.val.*`) skip frames with no labelled box, so they do not count false
boxes on the negative frames.

**Per-frame sampling weights** (`bbox_*`, with
`trainer_config.balanced_sampler.per_sample: true`). Give frames a relative weight
within their file in the frame metadata, and name the key:

```python
lf.metadata["sample_weight"] = 3.0   # drawn 3x as often as a frame of weight 1
```

```yaml
trainer_config:
  balanced_sampler: {per_sample: true, weights: [1, 1], frame_weight_key: sample_weight}
```

Frames without the key weigh 1, and weight 0 never draws a frame. Each file keeps
its share of the samples; the weights multiply with `overlap_weight`.

**Encoder-only reduced precision at prediction.** `sleap-nn predict --precision
bf16-encoder` (or `fp16-encoder`) runs only the backbone under autocast. A
`bbox_detr` detector's projector, transformer decoder and box and score heads run
in fp32 on fp32 features. It applies to every `bbox_detr` backbone (DINOv2,
DINOv3 and other HF or timm ViTs). See
[Reduced precision](inference-performance.md#reduced-precision-precision).

## Inference

```bash
sleap-nn predict --data_path video.mp4 --model_paths models/my_detector \
  --peak_threshold 0.3 --max_instances 10
```

A detector can also be the first stage of a top-down pose model, in place of a
centroid model: pass it with a centered-instance model directory (`--model_paths
<detector> --model_paths <centered_instance>`). See
[Top-down pose with a box detector](bbox-topdown.md).

`sleap-nn predict` detects the model type from its directory. For detectors,
`--peak_threshold` is the detection score threshold, `--max_instances` caps
detections per frame (default 100) and `--bbox_nms_iou` sets the class-wise IoU-NMS
threshold (default 0.45 for SSD, off for CenterNet; `1.0` disables it). Output
frames carry `sio.PredictedBoundingBox(x1, y1, x2, y2, score, category)` in original
image pixels on `LabeledFrame.bboxes`. From Python:

```python
from sleap_nn.inference.predictor import Predictor

predictor = Predictor.from_model_paths(["models/my_detector"], peak_threshold=0.3)
labels = predictor.predict("video.mp4")
boxes = labels[0].bboxes
```

On a `.slp` input, a detector with a fixed input size (`max_height` and
`max_width`) reads its frames already fitted to that size. Large JPEGs of an image
list are decoded at 1/2, 1/4 or 1/8 size, as in `bbox_detr` training. Each frame is
then letterboxed (or squashed) on the CPU, on 8 threads for image lists. Every
frame comes out the same size, so batches fill whatever the mix of image sizes.
The boxes are mapped back to each frame's original pixels.

Tracking is not supported for detector output (the tracker works on poses and
masks). The legacy `sleap-nn track` / `run_inference` pipeline refuses detectors and
points to `sleap-nn predict`.

### Export (ONNX / TensorRT)

```bash
sleap-nn export models/my_detector -o models/my_detector/exported --format onnx
sleap-nn predict -m models/my_detector/exported -i video.mp4 -o boxes.slp
```

A detector exports its **network only**: the graph returns the decoder-ready head
outputs (CenterNet's center probabilities + size/offset maps; SSD's per-anchor
logits + deltas; `bbox_detr`'s final-layer logits and boxes, see
[DETR detection](bbox-detr.md#export-onnx-tensorrt)), and the exported-model adapter decodes them with the same
`BBoxDecoder` as checkpoint inference, rebuilt from the `training_config.yaml`
copied into the export directory. Anchors, NMS and thresholds therefore live in one
place, exported and native boxes match, and the graph's spatial axes stay dynamic.
The `input_scale` resize and the max-stride padding also stay out of the graph (a
resize traced into it would be frozen to the trace input's size): the adapter applies
them on the host with the checkpoint path's own preprocessing (including the size
matcher, `max_height` / `max_width`, so mixed-resolution videos work as they do for
the checkpoint), then clips boxes to the frame.

## Evaluation

```bash
sleap-nn eval --ground_truth_path test.pkg.slp --predicted_path preds.slp \
  --match_method bbox --save_metrics metrics.npz
```

`--match_method auto` picks `bbox` when the predictions hold boxes only. Ground
truth without boxes (e.g. the pose file a detector was trained from) has them
derived first, like training does; pass `--bbox_source` / `--bbox_padding` to match
the model's `data_config.bbox_source` / `bbox_padding` (post-training eval does this
itself). Classes are matched by category name, except that when the ground truth and
the predictions each carry at most one name the comparison is class-agnostic: a
single-class detector maps every box to its one class whatever it is called.

Metrics follow the COCO protocol: per class and IoU threshold, each frame's top-100
detections are greedily matched (in descending score order) to the highest-IoU
unmatched ground-truth box; AP is the 101-point interpolated area under the
precision-recall curve over all frames. `bbox_mAP` averages IoU 0.50:0.05:0.95 and
classes; `bbox_AR` is the matching max recall. The operating point
(`bbox_precision/recall/f1`, `bbox_mean_iou` of the matches) is reported at IoU 0.5
and the score threshold that maximizes F1 (`bbox_f1_threshold`), so models are
compared at their own best threshold rather than an arbitrary fixed one.

`sleap-nn info models/my_detector` shows the detector's head and these metrics for a
trained model directory.

## How it fits together

- `sleap_nn/data/bboxes.py` — GT derivation (`add_bboxes_to_labels`), corner
  geometry, `CenterNetTargetEncoder`, `SSDTargetEncoder` (anchors, matching, box
  coding).
- `sleap_nn/architectures/heads.py` — `BBoxCenterHead`, `BBoxSizeHead`,
  `BBoxOffsetHead`, and `SSDHead`, a `MultiScaleHead` that `Model` binds to several
  backbone strides.
- `sleap_nn/training/lightning_modules.py` — `BBoxCenterNetLightningModule`,
  `BBoxSSDLightningModule` (shared `_BBoxLightningModel` base).
- `sleap_nn/inference/bbox.py` — `decode_centernet`, `decode_ssd`, `BBoxDecoder`
  (used by both training-time eval and inference); `inference/layers/bbox.py` —
  `BBoxLayer`.
- `sleap_nn/evaluation.py` — `compute_bbox_metrics`, `run_evaluation(match_method="bbox")`;
  `training/callbacks.py` — `BBoxEvaluationCallback`.
