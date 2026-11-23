# Frame Models on Several Skeletons

!!! warning "Experimental"
    The settings on this page are experimental and may change. They are off by
    default; a config that does not use them trains, saves and predicts exactly as
    before.

Train one `bottomup`, `centroid` or `single_instance` model on a pool of labels files
whose skeletons differ: for example one file per source, each with its own node set
(`<source>/<node>`, or a shared name such as `nose` where the sources agree).

---

## The Problem

Bottom-up, centroid and single-instance models read whole frames, and by default every
training and validation file must have the same skeleton: training stops with
"The skeleton in the training labels: 2 does not match the skeleton in the first
training label file." There was also no per-file batch balance for these models, and a
centroid model anchored on a node falls back to the mean of the visible nodes for files
that never label that node, so its target moves to a different place on the body.

The top-down [warp data path](warp-data-path.md) already merges skeletons by node name
for centered-instance models. The settings below do the same for the frame models.

---

## Quick Start

```yaml
data_config:
  train_labels_path: [source_a.slp, source_b.slp, source_c.slp]
  merge_skeletons: true          # experimental
  node_loss_mask: true           # default: nodes a file lacks get zero loss

model_config:
  head_configs:
    bottomup:
      confmaps:
        part_names: null         # null = every training file's nodes, first seen first
        sigma: 2.5
        output_stride: 4
      pafs:
        edges: null              # null = every file's edges between the merged nodes
        sigma: 10
        output_stride: 8

trainer_config:
  balanced_sampler:
    counts: [8, 4, 4]            # samples per training file per batch
```

---

## Merging Skeletons

With `data_config.merge_skeletons: true`:

- **The model's skeleton** is the union of the training files' nodes, matched by name:
  the head's `part_names` when set (nodes not listed are dropped), else every node of
  every training file in the order first seen. It carries every edge and every
  left/right symmetry the files have between those nodes. For bottom-up models, an
  explicit `pafs.edges` list replaces the files' edges.
- **Each file's instances are mapped onto it by node name** (training and validation
  files alike), in memory. The nodes a file does not have become unlabeled points
  (sleap-io `labeled=False`) and the file's `provenance["labeled_nodes"]` lists the
  nodes it has (combined with any mask the file already had). The in-memory copy keeps
  what these models read: frames, instances, videos, tracks, identities, suggestions,
  the skeleton and the provenance.
- **A file with no node of the merged skeleton** (for example one whose nodes are all
  left out of `part_names`) is refused at setup with an error naming the file: it
  would add no frame.
- **Loss.** With `node_loss_mask: true` (the default), those nodes get zero loss
  through the per-node weights of [Partially Labeled Skeletons](partial-labels.md). A
  bottom-up part-affinity edge counts only where the file labels both of its ends.
- **Saved model.** `training_config.yaml` holds the merged skeleton, so
  `sleap-nn predict` on the checkpoint predicts every merged node.
- **Validation and visualization** show and score each file on the nodes it labels.

Without `merge_skeletons` the same-skeleton check is unchanged. The setting is
refused for other model types (centered-instance models merge skeletons with
`crop_pipeline: warp`).

---

## Centroid Anchors from Several Nodes

A centroid model's `anchor_part` may be a list of node names, in order of preference:

```yaml
model_config:
  head_configs:
    centroid:
      confmaps:
        anchor_part: [source_a/neck, source_b/neck, neck]
        sigma: 2.5
        output_stride: 2
```

- Only the listed nodes a **file** labels count: those in its
  `provenance["labeled_nodes"]` (set by `merge_skeletons`), or all of them for a file
  without one.
- Each instance is anchored on the first of those it has **located**. An instance
  whose point there is unlabeled, occluded or out of the frame moves to the next one,
  and without any falls back to `centroid_fallback`, exactly as a single anchor does.
- The frames of a file that labels **none** of the listed nodes get no centroids and
  **zero loss** on the centroid map (a frame-level weight), instead of a centroid at a
  fallback location that would mean something else on the body. The files concerned
  are named in a warning at setup.
- Prediction and export place the centroid on the **first** listed node. The
  post-training evaluation scores each file against its first labeled listed node,
  as in training, and skips a file that labels none.
- The list must not be empty. A scalar is read as a node name (`1` is node `"1"`).

A single node name keeps today's behavior exactly (no frame weights).

---

## Balanced Batches

`trainer_config.balanced_sampler` (the same sampler as the warp and box-detector
paths) now also applies to the `bottomup`, `centroid` and `single_instance` datasets.
Tiled single-instance training ignores it, with a warning:

| Option | Meaning |
|--------|---------|
| `counts` | Samples per training file per batch (sum = per-GPU batch size) |
| `weights` | Relative shares per file, rounded to the batch size (when `counts` is not set) |
| `per_sample: true` | Each sample's file drawn at random with probability `weights` |

Files are counted in `train_labels_path` order. Under DDP each rank draws its own stream,
and the stream position is saved in checkpoints.

---

## One Input Scale per File

When the animals differ a lot in pixel size between files, one input scale per file can
bring them closer:

```yaml
data_config:
  preprocessing:
    max_height: 640
    max_width: 640
    file_scales:        # experimental
      source_a: 0.5     # source_a.slp: frames at half size
      source_c: 1.25
```

- A key matches a training or validation file by its path, its file name, or its file
  name up to the first `.` (`source_a` for `source_a.pkg.slp`); a validation split
  drawn from a training file matches that file. Files not listed keep scale 1. A key
  that matches no file, or more than one file, is an error.
- Each frame is resized by its file's scale, then **padded** (not stretched) to
  `max_height` x `max_width`; a frame that is still larger is shrunk to fit. The usual
  `scale` applies afterwards.
- Supported for `bottomup`, `centroid` and `single_instance` without tiling.
- **Training only:** prediction scales every frame by `scale` alone. Choose the input
  scale of new videos yourself (for example with `sleap-nn predict --input_scale`) so the
  animals have the size they had in training.

---

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_config.merge_skeletons` | bool | `false` | Merge the files' skeletons by node name (above) |
| `model_config.head_configs.centroid.confmaps.anchor_part` | str or list | `null` | A list: the first labeled and located node per instance; frames with an instance whose file labels none of them get zero loss |
| `trainer_config.balanced_sampler` | BalancedSamplerConfig | `null` | Per-file batch composition, now also for the frame models |
| `data_config.preprocessing.file_scales` | dict | `null` | One input scale per file, padded to the canvas |

`merge_skeletons` and `file_scales` are written to `training_config.yaml` only when a
run uses them, so a config saved without them stays readable by earlier releases.

---

## Not Covered

- Crops or tiles for bottom-up frames larger than the canvas: such frames are shrunk
  to fit.
- Multi-class (identity) models, centered-instance models outside the warp path, and
  tiled single-instance training with `merge_skeletons` or `file_scales`.
- A test file (`data_config.test_file_path`) is not mapped onto the merged skeleton for
  the post-training evaluation.
- Ground-truth-centroid crops at prediction use the first listed anchor (with the
  fallback), not the per-file rule.
- Multi-GPU (DDP) training with the balanced sampler for these models is untested.
