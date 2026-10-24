# Top-down pose with a box detector

A top-down pose pipeline finds each animal first, then runs a centered-instance
model on a crop around it. Stage 1 is usually a `centroid` model. It can instead be a
bounding-box detector (`bbox_centernet` or `bbox_ssd`, see
[Bounding-box detection](bbox-detection.md)), and the centered-instance model can then
crop each animal by its **box** — resized to a fixed crop size — instead of a
fixed-size window around a point:

```bash
sleap-nn predict --data_path video.mp4 \
    --model_paths models/bbox_centernet \
    --model_paths models/centered_instance
```

The detections that seeded the crops are kept in the output: each predicted instance
comes with a `sio.PredictedBoundingBox` on `LabeledFrame.bboxes`, linked to it
(`box.instance`). The same works for `multi_class_topdown` and (with fixed crops)
`centered_instance_segmentation` models.

Boxes can also come from outside sleap-nn: `sleap-nn predict -m models/centered_instance
-i labels.slp --boxes {dets.json | boxes.slp | gt}` seeds the crop model from a COCO
detection file, the predicted boxes in a `.slp`, or the GT boxes (see
[COCO-Style Evaluation](coco-evaluation.md#top-down-inference-from-given-boxes)).

To generate both configs from your labels:

```bash
sleap-nn config labels.slp --auto --pipeline topdown_bbox_centernet -o config.yaml
# -> config_bbox_centernet.yaml + config_centered_instance.yaml (crop_mode: bbox)
sleap-nn train config_bbox_centernet.yaml
sleap-nn train config_centered_instance.yaml
```

(`topdown_bbox_ssd` uses the SSD-style detector.) The two configs share
`bbox_source` / `bbox_padding`, as they must (below). The TUI (`sleap-nn config
labels.slp`) offers the same pair.

## Crop modes

The centered-instance model's `data_config.preprocessing.crop_mode` decides how a box
becomes a crop. Inference reads it from the model's saved config, so it always crops
the way the model was trained.

| `crop_mode` | `crop_bbox_aspect` | Crop | Animal in the crop |
|---|---|---|---|
| `fixed` (default) | – | `crop_size` px window centered on the box center | native pixel size, like the centroid pipeline |
| `bbox` | `square` (default) | padded box, squared on its longest side, resized to `crop_size` | fills the crop at any size, aspect kept |
| `bbox` | `letterbox` | as `square`, with pixels outside the padded box zeroed | as `square`, neighbors masked out |
| `bbox` | `stretch` | padded box resized to the square crop per axis | fills the crop, aspect distorted |

Box-scaled crops (`crop_mode: bbox`) normalize animal size: a pup and an adult, a
curled and a stretched mouse, or a near and a far animal all fill the crop the same
way. Fixed crops keep pixel size and need a crop big enough for the largest animal.

`crop_mode: bbox` is supported for `centered_instance` and `multi_class_topdown`.

## Training a box-scaled centered-instance model

```yaml
data_config:
  bbox_source: auto       # where boxes come from: the detector's own setting
  bbox_padding: 0.0       # pixels added to derived boxes: the detector's own setting
  preprocessing:
    crop_mode: bbox
    crop_bbox_aspect: square
    crop_bbox_padding: 0.1   # margin on every side, fraction of the box's longest side
    crop_bbox_min_size: null # smallest window side (px); auto: a quarter of the median
    crop_size: null          # auto: median padded box side, rounded up to max_stride
    scale: 1.0
model_config:
  head_configs:
    centered_instance:
      confmaps:
        anchor_part: null
        centroid_method: bbox_center
```

- **Use the detector's box definition.** Each training box is derived exactly as the
  detector's ground truth is: `bbox_source` and `bbox_padding` mean the same thing in
  both configs (a user box linked to the instance, the user mask linked to it, or the
  tight box of its visible keypoints). Train both models with the same two values —
  `sleap-nn predict` warns when they differ.
- **`crop_size` sets resolution, not coverage.** Every box is resized to the crop, so
  the crop no longer has to hold the largest animal. Left unset it is the median
  padded box side (rounded up to the backbone's max stride), so a typical animal
  keeps its pixel size; larger animals are downsampled and smaller ones upsampled.
- **`crop_bbox_min_size`** floors the window side so a tiny or partial box (one
  labeled keypoint) is not magnified without bound. Left unset it is a quarter of the
  median window side. Both values are written into the saved config.
- **Augmentation** works as for fixed crops (rotation, scale, intensity …): it is
  applied to the scale-normalized crop.

The `sleap-nn train` log prints the resolved crop size, the p5/p50/p95 window sides
and the resulting scale range.

### Fixed crops behind a detector

A fixed-crop model can also follow a detector; crops are then centered on the box
centers. Train it with `centroid_method: bbox_center` (and no `anchor_part`) so its
training crops are centered the same way — `sleap-nn predict` warns when the model was
trained on another centroid.

## How training and inference stay aligned

One rule turns a box into a crop window (`BBoxCropSpec` in
`sleap_nn/data/instance_cropping.py`), and both the training datasets and the
inference layer call it. Training cuts crops with Skia and inference with
`grid_sample`; both put pixel centers at the same coordinates and treat the frame edge
the same way, so the two crops of a box agree to within ~2/255 gray levels. Training
cuts a √2-larger window at the final scale for rotation headroom, augments it, and
re-crops on whole pixels, so an un-augmented training crop is exactly the inference
crop. Keypoints are mapped into and out of the crop by the same affine transform. The
frame is size-matched to the centered-instance model's own `max_height`/`max_width`
before cropping.

## Evaluation

- `sleap-nn eval` scores the pose predictions as for any top-down model.
- A box-scaled model run **alone** on labeled frames (`sleap-nn predict -m
  centered_instance -i labels.slp`) uses ground-truth boxes, derived from the labeled
  keypoints with the model's `bbox_padding` — the box analog of the GT-centroid
  fallback. Post-training evaluation uses this, so its numbers measure pose quality
  given perfect boxes.

## Limits

- A centroid model cannot feed a box-scaled model (it has no boxes); that pairing is
  refused.
- One first stage per pipeline: a centroid model and a detector together are refused.
- `sleap-nn export` exports each model on its own; a combined detector + top-down
  bundle is not supported yet.
- The legacy `sleap-nn track` pipeline does not run detectors; use
  `sleap-nn predict --tracking`.
