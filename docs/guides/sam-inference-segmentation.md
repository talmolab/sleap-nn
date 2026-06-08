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

SAM3 (`mask_backend="sam3"`) is a second, **gated** backend behind its own extra:

```bash
pip install "sleap-nn[sam3]"
```

SAM3 lives in the gated [`facebook/sam3`](https://huggingface.co/facebook/sam3)
Hugging Face repo and ships in `transformers>=5`. You must request access, then
authenticate (`huggingface-cli login` or set `HF_TOKEN`) with an approved
account. SAM3 also typically wants torch `cu130`, which **may not co-install**
with the rest of sleap-nn: keep it in its own environment, or run the SAM3 image
path out-of-process (write the raw masks + bookkeeping to an `.npz`) and assemble
the `.slp` back in the sleap-nn venv. Selecting `mask_backend="sam3"` without
`transformers` raises a clear, actionable `ImportError` pointing at this install
+ auth.

## The backend is explicit — there is no default

`mask_backend` is **required** when you ask for SAM masks; nothing is selected
silently. `"sam"` is SAM1 (ungated, the `sleap_nn[sam]` extra); `"sam3"` is the
gated SAM3 image path (`facebook/sam3` via `transformers`, the `sleap_nn[sam3]`
extra). Each backend stores its own **raw** per-model score in
`PredictedSegmentationMask.score` — and SAM3's score is on a lower scale than
SAM1's, so its recalibrated floor (`0.5`) is **never** SAM1's `0.88`.

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

## SAM3 backend

`mask_backend="sam3"` swaps SAM1 for Meta SAM 3's image visual-prompt path
(`Sam3TrackerModel`) behind the same interface and the same prompt modes:

```python
labels = run_sam_segmentation(
    "poses.pkg.slp",
    mask_backend="sam3",        # gated facebook/sam3, needs the [sam3] extra
    prompt_mode="pose",
    device="cuda",
    # sam3_model_id="facebook/sam3",  # the default
)
```

Two SAM3 specifics are handled automatically and are **never** shared with SAM1:

- **Recalibrated score floor (`0.5`).** SAM3's predicted-IoU is on a lower scale,
  so its per-model floor (`Sam3Backend.pred_iou_min`) defaults to `0.5` (SAM1's
  `0.88` would reject ~100% of SAM3 masks as a pure calibration artifact). As with
  SAM1 the raw score is reported, not gated on.
- **Speckle cleanup.** Raw SAM3 masks are fragmented (~14 connected components);
  each chosen mask is passed through a morphological open + close + keep-the-
  keypoint-connected-component cleanup (~1 component, ~97% area retained).

SAM3 also runs **all instances of a frame in a single batched forward pass**
(SAM1 loops per instance). End-to-end SAM3 is GPU-only and needs the gated
weights, so it is exercised manually / nightly; the recipe (floor, cleanup,
batched call) is unit-tested against a mocked SAM3.

## CLI

The same masks are produced from the `sleap-nn predict` command (and its hidden
`infer` alias) by passing `--mask_backend`. When set, the input `.slp` is treated
as a pose file and masks are produced from its existing instances — there is **no
trained segmentation model**, so `--mask_backend` is **mutually exclusive with
`--model_paths`** (do not pass it). The `.slp` output is saved like the regular
prediction path — by default images are **not** re-embedded; the output
backreferences the input's source media via provenance (small output; a
`.pkg.slp` input stays matchable to its source videos), and the masks always
serialize into the `.slp`. Only `--output_format slp` is supported (the SLEAP
Analysis HDF5 format stores poses/tracks, not segmentation masks).

### Controlling image embedding (`--embed` / `--restore_source_videos`)

Two independent flags (also `embed` / `restore_source_videos` API kwargs on
`predict` and `run_sam_segmentation`) control how the output `.slp` references
image data. The defaults preserve today's behavior exactly.

- `--embed` (`auto` | `true` | `false`, default `false`): the embedding policy.
  `false` never embeds and backreferences the source media (today's behavior);
  `true` embeds images into a self-contained `.pkg.slp`-style output; `auto`
  embeds only when the input was itself an embedded `.pkg.slp`.
- `--restore_source_videos` / `--no-restore_source_videos` (default
  `--restore_source_videos`): on a non-embedding save, restore references to the
  original source video files (provenance), or use `--no-restore_source_videos`
  to keep references to the input `.pkg.slp` file(s) instead. Ignored when
  embedding. Maps to sleap-io's `restore_original_videos`.

```bash
# Embed images into a self-contained output .pkg.slp:
sleap-nn predict -i poses.pkg.slp --mask_backend sam \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --embed true -o poses_with_masks.pkg.slp
```

SAM1 (ungated):

```bash
sleap-nn predict \
    -i poses.pkg.slp \
    --mask_backend sam \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --sam_prompt_mode pose \
    --overlay_path review_frame0.png \
    -o poses_with_masks.pkg.slp
```

SAM3 (gated `facebook/sam3`, needs the `[sam3]` extra):

```bash
sleap-nn predict \
    -i poses.pkg.slp \
    --mask_backend sam3 \
    --sam3_model_id facebook/sam3 \
    --sam_prompt_mode pose \
    -o poses_with_masks.pkg.slp
```

Every flag also has a dash-spelled alias (`--mask-backend`, `--sam-checkpoint`,
`--sam-model-type`, `--sam-prompt-mode`, `--sam-anchor-ind`,
`--sam-disjointify-masks`, `--sam3-model-id`, `--overlay-path`).

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
