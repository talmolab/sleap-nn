# COCO-Style Keypoint Evaluation

Report keypoint AP/AR that can be compared with published human pose numbers
(COCO, OCHuman, CrowdPose), and run top-down models on the person boxes papers use.

sleap-nn's original OKS metrics (`oks_voc.mAP`, `oks_voc.mAR`) are kept for
back-compat but are **legacy**: they do not follow COCOeval and read low or high
depending on the data (on COCO val2017 with the default `oks_stddev=0.025` they
read about 0.35 AP low). Use the `coco.*` metrics below whenever you compare with
a paper.

---

## What `coco.*` measures

`sleap_nn.evaluation_coco` is a numpy port of `pycocotools`'
`COCOeval(iouType="keypoints")` that works directly on `.slp` labels. On COCO val2017
it matches pycocotools to 4 decimals. It follows COCOeval in every respect:

- **Per-threshold matching**: detections are greedily matched to GT separately at
  each OKS threshold (0.50:0.05:0.95), in score order.
- **Every prediction is scored**: extra detections, detections with OKS 0 to every
  GT and detections on frames with no GT people are false positives.
- **Ignore regions**: crowd boxes and people with no labeled keypoint (a user box
  with no instance, or an instance with no labeled point and a linked box) are
  "ignore" GT: a detection matched to one is neither a TP nor an FP.
- **Per-node sigmas** instead of one `oks_stddev`.
- **Object area** from the source annotation's segmentation area when the file
  carries it (`/human_meta`, written by the human-pose dataset converter), else
  0.53 × the GT box area (xtcocotools' rule), else the visible-keypoint box.
- **maxDets = 20** per image and the medium / large area ranges (`AP_M`, `AP_L`).

| Metric | Meaning |
|--------|---------|
| `coco.AP` | mean AP over OKS 0.50:0.05:0.95 (the headline number) |
| `coco.AP50`, `coco.AP75` | AP at OKS 0.5 / 0.75 |
| `coco.AP_M`, `coco.AP_L` | AP for medium (32²–96² px²) / large (> 96² px²) people |
| `coco.AR`, `coco.AR50`, `coco.AR75`, `coco.AR_M`, `coco.AR_L` | the same for recall |
| `coco.n_gt`, `coco.n_gt_ignore`, `coco.n_dt`, `coco.n_images` | counts |
| `coco.sigmas` | the per-node sigmas used |

---

## Evaluating predictions

```bash
sleap-nn eval -g coco_val2017.slp -p predictions.slp --coco
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--coco` | Also compute the `coco.*` metrics (saved under `metrics["coco_metrics"]`) | off |
| `--oks_sigmas` | Per-node sigmas: `coco`, `crowdpose`, `ochuman`, `provenance` (the GT file's `node_sigmas`), `auto`, one float, or a comma-separated list (one per node). Also replaces `--oks_stddev` in the legacy metrics | by node name |
| `--coco_protocol` | Dataset protocol preset: `auto` (from the GT file's provenance `dataset`), `coco`, `ochuman`, `crowdpose`, or `none` | `auto` |
| `--coco_area_source` | GT area: `meta`, `bbox` (0.53 × box), `keypoints`, `auto` (meta → bbox → keypoints) | the protocol's |
| `--count_unmatched_as_fp` | Count unmatched predictions as FPs in the legacy `oks_voc` metrics | off |

Without `--oks_sigmas`, each node's sigma comes from the GT file's provenance
`node_sigmas`, then the COCO-17 sigmas by node name, then `--oks_stddev`.

**Protocols.** `coco` and `ochuman` score every GT person with a labeled keypoint
(xtcocotools `keypoints`, as mmpose does) and take the area from the annotation.
`crowdpose` ignores people whose source `num_keypoints` is 0 and uses 0.53 × box
area (xtcocotools `keypoints_crowd`); it needs `/human_meta`. On OCHuman, 1,431 of
6,546 annotations have `num_keypoints == 0` despite labeled keypoints, so the
pycocotools and mmpose rules give different AP there; sleap-nn reports the mmpose
rule.

From Python:

```python
import sleap_io as sio
from sleap_nn.evaluation_coco import coco_keypoint_metrics

gt = sio.load_slp("coco_val2017.slp")
pred = sio.load_slp("predictions.slp")
m = coco_keypoint_metrics(gt, pred, protocol="auto")
print(m["coco.AP"], m["coco.AR"])

# Cross-check against pycocotools (pip install 'sleap-nn[coco]'):
ref = coco_keypoint_metrics(gt, pred, protocol="auto", backend="pycocotools")
```

---

## Top-down inference from given boxes

Top-down papers report AP with a fixed set of person boxes (for COCO val2017, the
56 AP SimpleBaseline detections `COCO_val2017_detections_AP_H_56_person.json`) or
with the GT boxes. `sleap-nn predict --boxes` runs a crop model (`centered_instance`,
`multi_class_topdown` or `centered_instance_segmentation`) on such boxes instead of
a centroid model or a detector. Pass only the crop model, and a `.slp` input.

```bash
# Detected boxes (COCO results JSON, matched to frames by image id)
sleap-nn predict -m models/vitpose_s -i coco_val2017.slp \
    --boxes COCO_val2017_detections_AP_H_56_person.json \
    --peak_threshold 0 --oks_nms 0.9 -o pred_det.slp

# GT boxes
sleap-nn predict -m models/vitpose_s -i coco_val2017.slp \
    --boxes gt --peak_threshold 0 -o pred_gt.slp

sleap-nn eval -g coco_val2017.slp -p pred_det.slp --coco
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--boxes` | `gt`, a COCO detection results `.json`, or a `.slp` whose frames carry `PredictedBoundingBox`es (a `bbox_*` detector's output, or the input itself) | off |
| `--box_score_threshold` | Drop boxes scoring below this | `0` |
| `--rescore` / `--no-rescore` | Instance score = box score × mean of keypoint scores above `--rescore_threshold` (mmpose) | on with `--boxes` |
| `--rescore_threshold` | Keypoint-score threshold for `--rescore` | `0.2` |
| `--oks_nms` | OKS-NMS threshold (mmpose `oks_nms`); papers use `0.9` with detected boxes | `0` (off) |
| `--oks_sigmas` | Sigmas for `--oks_nms` | by node name |

Details:

- A `.json` file is matched to frames by COCO `image_id`, read from the file's
  `/human_meta` or from the trailing digits of each frame's image file name
  (`000000000139.jpg` → 139). Only `category_id` 1 is used.
- Each box seeds one crop by the crop model's own rule: box-scaled crops
  (`crop_mode: bbox`) resize the box as in training; fixed crops center on it.
- The boxes are kept in the output, linked to the instance each produced.
- With `--boxes gt`, boxes are the GT user boxes linked to each instance (else the
  instance's keypoint box) and score 1, so a rescored instance's score is its mean
  keypoint score.
- Keep `--peak_threshold 0`: COCOeval counts a dropped keypoint as a miss, and the
  default 0.2 costs several AP.
- OKS-NMS uses the mean of two instances' areas, taken from the crop window (as
  mmpose's padded box), else the box, else the keypoint extent.

From Python, `sleap_nn.inference.run.predict(..., boxes=..., box_score_threshold=...)`
and `Predictor.from_model_paths(..., boxes="gt" | "given")`; the rescoring and
OKS-NMS live on `FilterConfig` (`rescore_keypoint_threshold`, `oks_nms_threshold`,
`oks_sigmas`). `sleap_nn.evaluation_coco.attach_coco_detections` and
`transfer_predicted_boxes` attach boxes to a `sio.Labels` yourself.

---

## Post-training evaluation

Set `trainer_config.eval.coco_metrics: true` to add `coco.*` to the metrics computed
on each split after training (and to the wandb summary). For top-down models it
also predicts those splits under the COCO protocol (`peak_threshold` 0 and
rescoring), on GT boxes or GT centroids as before. `trainer_config.eval.oks_sigmas`
takes the same values as `--oks_sigmas`.

```yaml
trainer_config:
  eval:
    coco_metrics: true
    oks_sigmas: coco
```

---

## Known differences in the legacy metrics

`oks_voc.*` (the `Evaluator` in `sleap_nn.evaluation`) differs from COCOeval in
ways that move AP by several points:

- Unmatched predictions are not false positives (fixed behind
  `--count_unmatched_as_fp`).
- Matching runs once at OKS > 0, so a poor detection can hold a GT at every
  threshold.
- GT on frames with no predicted frame drop out of the recall denominator.
- The area is the visible-keypoint box, and there are no ignore regions, maxDets or
  area ranges.
