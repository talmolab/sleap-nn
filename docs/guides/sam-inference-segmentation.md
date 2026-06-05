# SAM-prompted instance segmentation (inference)

Predict a **per-instance mask** for every animal in an existing pose `.slp` by
prompting [Segment Anything](https://github.com/facebookresearch/segment-anything)
(SAM) with the predicted keypoints / centroids. The masks are written as
`sio.PredictedSegmentationMask` so they import into the SLEAP GUI for review and
correction; you do **not** train a segmentation model.

This is the inference complement to
[top-down (trained) segmentation](topdown-segmentation.md): there the mask comes
from a trained `centered_instance_segmentation` head; here it comes from SAM,
prompted by poses you already have. Use it to bootstrap masks from a pose model
without training a mask model.

> **v1 is prompted-only.** Three prompt modes — pose, centroid, and box.
> Unprompted ("everything") mask generation is deferred. The GUI correction
> round-trip is gated on upstream frontend work; v1 ships the masks plus a
> review-overlay PNG.

## Install

SAM1 (ViT-H, Apache-2.0, ungated) is an **optional, lazily imported** extra so
the default install never needs it:

```bash
pip install "sleap-nn[sam]"
```

Download a SAM checkpoint (e.g.
[`sam_vit_h_4b8939.pth`](https://github.com/facebookresearch/segment-anything#model-checkpoints))
and pass its path via `sam_checkpoint=`.

## The backend is explicit — there is no default

`mask_backend` is **required** when you ask for SAM masks; nothing is selected
silently. `"sam"` is SAM1 (this PR); `"sam3"` (the gated SAM3 image path) lands
in a follow-up behind the `sleap_nn[sam3]` extra. Each backend stores its own
**raw** per-model score in `PredictedSegmentationMask.score`.

## Predict masks for a pose `.slp`

```python
from sleap_nn.inference.run import predict

labels = predict(
    "poses.pkg.slp",            # a .slp with predicted poses + image data
    mask_backend="sam",         # explicit, required
    sam_checkpoint="/path/to/sam_vit_h_4b8939.pth",
    sam_prompt_mode="pose",     # "pose" | "centroid" | "box"
    device="cuda",
    output_path="poses_with_masks.pkg.slp",
    overlay_path="review_frame0.png",   # optional review overlay (PLAN L4)
)
```

When `mask_backend` is set, `source` is treated as a pose `.slp` and masks are
produced from its existing instances — **no trained seg model**, so you must not
also pass `model_paths` / `export_dir`.

Or call the producer directly:

```python
from sleap_nn.inference.sam import run_sam_segmentation

labels = run_sam_segmentation(
    "poses.pkg.slp",
    mask_backend="sam",
    prompt_mode="pose",
    sam_checkpoint="/path/to/sam_vit_h_4b8939.pth",
    device="cuda",
)
```

## Prompt modes

| Mode | Prompt | Needs | Notes |
|------|--------|-------|-------|
| **pose** | every visible keypoint as a positive point **+** the padded keypoint box | a pose | cleanest; the product rule below |
| **centroid** | a single positive point (anchor node or the mean keypoint) | a centroid | under-covers; box kept only for candidate rejection |
| **box** | the padded pose box, no points | a pose | leaks between adjacent animals; secondary |

**Product rule:** in `"pose"` mode, an instance with no visible keypoints falls
back to its centroid as a single point. No negative points are used for
automatic prompting.

## What gets written

Each instance produces one `sio.PredictedSegmentationMask` on
`LabeledFrame.masks`:

- `score` — the raw SAM predicted-IoU of the chosen candidate (no drop-gate; low
  scores are surfaced, not dropped).
- `instance` / `track` / `tracking_score` — populated from the source pose when
  present, so a correction is referential (which instance) not positional.
- full-frame masks use identity `scale`/`offset`.

The original pose instances are retained alongside the masks (correction needs
the pose).

## Limitations

- SAM works best on compact animals (e.g. mice); it degrades on tiny / elongated
  ones (e.g. flies) — choose the mode per dataset, or skip SAM there.
- On extreme-low-light data a lone center point can under-segment; prefer the
  pose prompt when a pose exists.
- The GUI correction round-trip (predicted→user mask transition, mask editing) is
  upstream work; v1 is review-only.
