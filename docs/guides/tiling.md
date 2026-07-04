# Tiling for high-resolution, small-object frames

When frames are very large (4K++) and the animals occupy a small fraction of the
image, the default pipeline is stuck between two bad options: **downscale the whole
frame** (which destroys the few pixels each small animal has) or **run at native
resolution** (which blows up GPU memory). **Tiling** takes the third path used in
medical imaging and remote sensing: cut each frame into overlapping square tiles,
run the model per tile at native resolution, and stitch the per-tile predictions
back into one frame.

Tiling is **explicit opt-in** — nothing changes unless you turn it on. The rest of
the pipeline (epoch semantics, memory model, coordinate math) is byte-identical when
it is off.

!!! note "Scope (current release)"
    Tiling supports the **`single_instance`**, **`bottomup_segmentation`**, and
    **`semantic_segmentation`** model types end to end (training **and**
    inference). Top-down (centroid) and bottom-up (PAF) tiling are planned for
    later releases. Tiling requires a **UNet / ConvNeXt / SwinT** backbone;
    pretrained HuggingFace-encoder backbones and `ClassVectorsHead` /
    `multi_class_topdown` models are not supported with tiling (training will
    raise a clear error). Tiled ONNX/TensorRT **export** is not yet implemented —
    exporting a tiled model warns and produces a whole-frame graph; use PyTorch
    inference for tiled prediction.

## When to use it

Reach for tiling when **both** are true:

- Frames are large (roughly min side ≥ ~1500–2000 px).
- The animals are **small relative to the frame** — after the usual whole-frame
  downscale, a keypoint's confidence-map blob would be only a couple of output-stride
  cells wide (blobs merge, peaks collide).

If one big animal fills the frame, you don't need tiling — the normal
`scale`/`max_height`/`max_width` path is fine. Tiling for `single_instance`
specifically targets **one small animal in a large frame** (single-instance keeps
its "one pose per frame" contract; if you have many animals per frame, use a
multi-instance model type instead — training warns if a single-instance model is
given multi-instance labels).

## Scale parity: a precondition, not an add-on

A model trained on downscaled full frames cannot benefit from native-resolution
tiling at inference — it would see objects several times larger than it was trained
on and detect almost nothing. So **tiling geometry is fixed at training time**: the
resolved `tile_size` and `overlap` are written into the trained model config and
**read back (and parity-checked) at inference**. You cannot enable tiling on an
existing whole-frame-trained model; train (or retrain) with tiling on. An inference
geometry override that diverges from the trained values raises an error — retrain to
change the geometry.

## Enabling tiling

Add a `tiling` block under `data_config.preprocessing`:

```yaml
data_config:
  preprocessing:
    tiling:
      enabled: true
      # tile_size / overlap left null => auto-sized from your labels + backbone
      tile_size: null
      overlap: null
```

At training setup, `tile_size` and `overlap` are auto-sized from the largest
labeled instance (plus a backbone-context margin and a few confidence-map sigmas),
rounded to be divisible by both the backbone `max_stride` and the head
`output_stride`, and written back into the config. When labels are too sparse to
estimate object size reliably, a **conservative overlap default** is used and a
warning is emitted — set `overlap` explicitly to override.

You can pin the geometry yourself:

```yaml
      tile_size: 512      # square; auto-rounded up to a multiple of max_stride & output_stride
      overlap: 128        # px; auto-rounded to output_stride, and raised to the min_overlap_fraction floor
```

## Configuration reference

All fields live under `data_config.preprocessing.tiling`:

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Explicit opt-in. Everything below is inert when `false`. |
| `tile_size` | `null` | Square tile side (px). Auto-sized from labels when `null`. |
| `overlap` | `null` | Tile overlap (px). Auto-sized when `null`; conservative default + warning when labels are sparse. |
| `min_overlap_fraction` | `0.25` | Overlap floor as a fraction of `tile_size`. |
| `blend` | `"gaussian"` | Stitch window: `gaussian` (center-weighted, seam-safe), `pyramid`, or `constant` (debug). |
| `sigma_scale` | `0.125` | Gaussian importance-window std as a fraction of the tile. |
| `tile_batch_size` | `null` | Tiles forwarded per backend call at inference (manual knob; conservative default when `null`). |
| `accumulator_device` | `"auto"` | Where the per-frame stitch buffers live: `auto` (predict, spill to CPU under memory pressure, OOM fallback), `cpu`, or `cuda`. |
| `cpu_thresh` | `0.40` | Spill the stitch buffers to CPU when they'd exceed this fraction of free GPU memory. |
| `sampling` | `"foreground"` | Training tile sampling: `foreground` (object-aware) or `grid`. Validation is always full-coverage grid. |
| `tile_fg_fraction` | `0.5` | Fraction of training tiles forced to contain an object (nnU-Net oversampling; never `1.0` — the rest are background/hard-negative tiles). |
| `samples_per_frame` | `null` | Tiles drawn per decoded frame (a conservative one-grid-pass default when `null`). |
| `center_jitter` | `0.5` | Foreground-tile center jitter, as a fraction of half the tile. |
| `min_visible_keypoints` | `1` | Keep an instance in a tile only if at least this many of its keypoints fall inside. |
| `steps_per_epoch` | `null` | Decouples the **training** epoch length from the tile count (validation stays full-coverage). |
| `full_frame_pass` | `false` | Reserved for a later release; inert now. |

## How it works

**Training.** Each frame is decoded once and explodes into `samples_per_frame`
tiles. Most tiles are drawn centered on a random labeled keypoint (with jitter) so
batches always contain foreground; the rest are uniform-random (and often empty —
valuable background/hard-negative tiles). Geometric augmentation is applied per tile
via a √2 halo so rotation never pulls in a black border. Targets are generated per
tile at native resolution. A frame-grouped sampler keeps a frame's tiles together so
the single decode is amortized across them, and epoch length is decoupled from the
(now much larger) tile count.

**Inference.** Each frame is processed at native resolution (the whole-frame
downscale is bypassed), cut into a snapped grid of overlapping tiles, forwarded in
batches, and the per-tile confidence maps are accumulated into one frame-sized canvas
with a Gaussian importance window (edge pixels of each tile are down-weighted because
their receptive field runs off the tile). Peaks are then found **once** on the
stitched map — preserving one pose per frame — and coordinates are mapped back to the
original image. Stitching reproduces whole-frame keypoints essentially exactly
(sub-pixel agreement in practice).

## Choosing `tile_size` / `overlap`

Auto-sizing is a good default. To sanity-check or tune the geometry, visualize the
grid and the blend coverage over one of your frames:

```python
import sleap_io as sio
from sleap_nn.training.utils import plot_tile_grid

labels = sio.load_slp("labels.slp")
img = labels[0].image  # (H, W, C)

fig = plot_tile_grid(
    img,
    tile_size=512,
    overlap=128,
    output_stride=2,       # your head's output stride
    max_stride=16,         # your backbone's max stride
    blend="gaussian",
)
fig.savefig("tiling_preview.png")
```

The red rectangles are the tiles; the heatmap is the summed importance-window
coverage (the stitch denominator). Every pixel should be covered (no dark holes),
and each whole animal should fit inside at least one tile — if an animal is larger
than a tile, increase `tile_size`.

## Notes and limitations

- **Square tiles, constant-zero padding.** Border tiles are zero-padded (reflect
  padding, which can hallucinate phantom keypoints near the frame edge, is not used).
- **One pose per frame** for `single_instance` — the stitched map yields a single
  global peak per node.
- **Memory.** The transient tile-forward batch is bounded by `tile_batch_size`; the
  per-frame stitch buffers auto-spill to CPU under memory pressure. Raise
  `tile_batch_size` for throughput until just below OOM.
- **Multi-GPU.** Whole frames (with all their tiles) are sharded across ranks so the
  decode-once benefit and tile grouping survive DDP.
