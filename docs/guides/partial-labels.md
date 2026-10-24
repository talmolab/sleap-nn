# Partially Labeled Skeletons

Train one model on datasets that each label a different subset of the skeleton's nodes,
without teaching it that the missing nodes are absent.

---

## The Problem

A confidence-map model learns two things for every node: where it is (a Gaussian peak)
and where it is not (zeros everywhere else). By default, a node with no coordinates is
trained as **absent**: its target is an all-zero map.

That is right when the annotator saw the node and it was not there. It is wrong when the
node was simply **never labeled**, for example:

- You merge datasets with different keypoint schemas into one union skeleton (e.g. COCO
  + AI Challenger + MPII for human pose). Each file annotates only its own nodes.
- A dataset marks joints as "not labeled" (COCO `v=0`), so the point has no location.
- Augmentation pushes a joint out of the crop.

Training those nodes as absent teaches the model to suppress real keypoints. Per-node
loss weighting gives them **zero loss** instead.

---

## Quick Start

Mark the nodes each file annotates in its provenance, then train as usual:

```python
import sleap_io as sio

labels = sio.load_slp("coco_subset.slp")  # union skeleton, COCO nodes only labeled
labels.provenance["labeled_nodes"] = [
    "nose", "left_eye", "right_eye",  # ... node names (or integer indices)
]
labels.save("coco_subset.slp")
```

```yaml
data_config:
  train_labels_path: [coco_subset.slp, aic_subset.slp, mpii_subset.slp]
  node_loss_mask: true        # default
  unlabeled_points: ignore    # optional: also ignore points with no location
```

Files without `labeled_nodes` behave as if every node is labeled.

---

## Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `data_config.node_loss_mask` | bool | `true` | Per-node loss weighting. Master switch for everything on this page. |
| `data_config.unlabeled_points` | str | `absent` | `absent`: a point with no location trains as absent (weight 1). `ignore`: it gets weight 0, as in mmpose. |
| `data_config.supervise_occluded` | bool | `true` | Train labeled-but-occluded points at their estimated coordinates; `false` gives them weight 0. No-op until sleap-io exposes per-point `labeled` flags. |
| `trainer_config.online_hard_keypoint_mining.per_sample` | bool | `true` | Rank hard keypoints per sample over weighted nodes (see below). |

"No location" means the target point is NaN, or outside the model input (image for
single-instance / bottom-up, crop for centered-instance) after augmentation.

!!! note "Defaults keep existing projects unchanged"
    With every node labeled (no `labeled_nodes`) and `unlabeled_points: absent`, no
    weights are emitted and the losses are exactly the unweighted ones.

---

## How It Works

Each sample carries `node_weights` of shape `(1, n_nodes)`. The weight of a node is the
product of:

1. the file mask from `provenance["labeled_nodes"]` (all ones when absent);
2. per-point `labeled` flags, read automatically once sleap-io provides them (SLP 3.0);
3. with `unlabeled_points: ignore`, whether the point has a location in the model input.

Negative frames always get weight 1 on every node (they are trained as absent). Flip
augmentation swaps the weights together with the left/right node channels.

The confmap loss is normalized by the total weight:

```
loss = Σ_b Σ_c  s_b · w_bc · mse_bc  /  Σ_b Σ_c w_bc
```

where `mse_bc` is the spatial mean squared error of node `c` in sample `b` and `s_b` is
the negative-frame weight (`negative_loss_weight`, train only). With all weights 1 this
equals `nn.MSELoss`. Nodes with weight 0 get no loss and no gradient.

### Online hard keypoint mining

With `per_sample: true`, OHKM ranks nodes **within each sample**, over nodes with
nonzero weight only, and averages the selected (sample, node) losses. The pre-0.4
behavior (`per_sample: false`) ranks nodes on losses summed over the batch: it cannot
see the weights and will mine unlabeled nodes from other datasets in the batch.
`per_sample: true` is the default, so OHKM results change for batch size > 1.

### Logged metrics

- `val/loss` is the weighted loss (and selects checkpoints).
- `val/loss_unmasked` is the plain MSE over all nodes, logged for single-instance and
  centered-instance models when weights are active, for comparison with unmasked runs.
- `train/confmaps/<node>` stays the unweighted per-node MSE.

### Train/val splits

When no `val_labels_path` is given, the automatic split (and the group-aware
`data_config.split`) keeps each file's `labeled_nodes` on both sides. sleap-io's own
splitting drops provenance, which would otherwise turn the mask off silently.

---

## Supported Models

| Model type | File mask | `unlabeled_points: ignore` |
|------------|-----------|----------------------------|
| `single_instance` | yes | per node |
| `centered_instance` | yes | per node (in the crop) |
| `bottomup` (confmaps) | yes | node ignored only if no instance in the frame locates it |

Not covered yet: part affinity fields (bottom-up PAFs stay unweighted), tiled
single-instance training (warns and trains unweighted), multi-class models, and
centroids.
