# Coordinate Losses (experimental)

!!! warning "Experimental"
    `trainer_config.coordinate_losses` is experimental: its keys and behaviour may
    change. It is off by default, and runs that do not set it train, save and
    predict exactly as before.

`trainer_config.coordinate_losses` adds training losses on keypoint
**coordinates** to a top-down `centered_instance` model, through a frozen linear
keypoint autoencoder. The autoencoder follows the affine-combining autoencoder of
Sárándi, Hermans and Leibe (WACV 2023, "Learning 3D Human Pose Estimation from
Dozens of Datasets using a Geometry-Aware Autoencoder to Bridge Between Skeleton
Formats"): an encoder `W_enc` (`L x J`) maps `J` keypoint channels to `L`
latent points, each an affine combination of the channels, and a decoder `W_dec`
(`J x L`) maps the latent points back. Trained on poses labelled in several
skeleton formats, it captures how the channels of the formats relate to each
other.

Two of the paper's uses are available:

- **Consistency** (`consistency`): the model predicts the `J` channels as usual
  and is also asked to predict poses the autoencoder can reproduce:
  `consistency * |P - W_dec W_enc P|_1`.
- **Decoded coordinates** (`decoded`): the model predicts the `L` latent points
  and is trained only through their decoding: `decoded * |W_dec Q - Y|_1` over
  the channels each sample labels. A file that labels five channels and a file
  that labels twenty both supervise the same latent points.

---

## Differentiable coordinates

Confidence maps are trained with a pixel-wise loss and their peaks are found after
the network, without a gradient. The coordinate losses need coordinates with a
gradient, so a **soft-argmax** (integral regression, Sun et al. 2018) turns each
predicted map into the expectation of the grid coordinates under
`softmax(beta * map)`, in crop pixels (grid cell `i` sits at `i * output_stride`,
as in the targets). It is computed only when a coordinate loss is configured.

- `soft_argmax_beta` (default `20`) is the inverse temperature. Over a whole map
  trained towards Gaussians of height 1, a small value lets the flat background
  pull the estimate towards the middle of the crop; a very large value narrows the
  peak below a grid cell and rounds the estimate towards the grid.
- `soft_argmax_window` (default `null`, the whole map) restricts the softmax to the
  cells within that many grid cells of each map's maximum. On Gaussian maps
  (`consistency`), a window of about three times the head's `sigma` with
  `soft_argmax_beta: 10` locates peaks to about 0.2 grid cells even on noisy
  maps. For `decoded`, where the maps are learned only through their
  soft-argmax, the whole map (the default) gives a gradient everywhere.

## The autoencoder file

An `.npz` with:

| Key | Shape | Meaning |
|-----|-------|---------|
| `W_enc` | `L x J` | Encoder; each row sums to 1 |
| `W_dec` | `J x L` | Decoder; each row sums to 1 |
| `names` | `J` strings | Channel names, matched to the model's channels by name |
| `latent_partner` | `L` ints (optional) | Left/right partner of each latent point (itself for a centre point) |

Other keys are ignored. Rows summing to 1 make the losses independent of where
the crop sits in the frame; a file whose rows do not is refused.

## Consistency

```yaml
model_config:
  head_configs:
    centered_instance:
      confmaps:
        part_names: null        # the skeleton's nodes = the autoencoder's names
trainer_config:
  coordinate_losses:
    autoencoder: /models/keypoint_ae.npz
    consistency: 0.001
    soft_argmax_beta: 10.0
    soft_argmax_window: 8
```

- The head's part names must be exactly the autoencoder's channel names (in any
  order); training refuses otherwise and lists the names that differ.
- The confidence-map loss is unchanged; `loss` and `train/loss` log it alone,
  and `val/loss` (checkpoint selection) stays the confidence-map loss.
- `train/coord_consistency` and `val/coord_consistency` log the unweighted term
  (mean `|dx| + |dy|` per channel, in crop pixels). Scale the weight
  accordingly: confidence-map losses are of the order of `1e-3`.
- The term is computed on every sample except negative frames.

## Decoded coordinates

```yaml
trainer_config:
  coordinate_losses:
    autoencoder: /models/keypoint_ae.npz
    decoded: 0.01
```

- The head's channels become the latent points, named `latent_00`,
  `latent_01`, ... The data loader keeps rendering crops of the labelled
  channels (the skeleton's nodes, or `part_names` with the
  [fast top-down data path](warp-data-path.md)); each must be one of the
  autoencoder's channels.
- The loss is `decoded` times the mean `|dx| + |dy|` over the decoded channels
  each sample labels: a channel counts when its point has a location in the crop
  and its per-node weight is not 0 (see
  [Partially labeled skeletons](partial-labels.md)). Negative frames are left out.
  No confidence-map loss is computed for the latent channels.
- `loss`, `train/loss` and `val/loss` log the weighted term;
  `train/coord_decoded` and `val/coord_decoded` the unweighted one (crop
  pixels).
- The saved model is a plain `centered_instance` model whose head predicts the
  latent points: `training_config.yaml` has the latent names as `part_names`, a
  skeleton of the latent points (with `latent_partner` as its symmetries) and no
  anchor part. `sleap-nn predict` returns the latent points; decode them with
  `W_dec` to get the autoencoder's channels. The peaks are found the usual way
  (maximum plus refinement), which matches the soft-argmax only where the learned
  maps have a single peak.
- The labelled channels' head settings are kept in
  `coordinate_losses.data_head` of the saved config. To train again, start from
  the original config, not the saved one.
- Not available with this term: distillation, online hard keypoint mining,
  epoch-end evaluation (`trainer_config.eval.enabled`), prediction plots
  (`visualize_preds_during_training`) and the W&B prediction gallery
  (`trainer_config.wandb.gallery`), which all pair predicted and labelled channels
  one to one. Training refuses these combinations.

## Augmentation

The losses are computed in crop coordinates, after the crop's rotation, scaling,
translation and flips. The autoencoder is equivariant to rotation and scaling
(it acts on the point index, not on x and y), and to translation (rows sum to
1). Flips also swap the skeleton's symmetric channels, so with flip augmentation
on (`flip_p > 0`) training checks that the autoencoder agrees with that swap:

- consistency: `W_dec W_enc` must commute with the swap of the skeleton's
  symmetric channels;
- decoded: the decoder rows of the labelled channels, swapped, must equal the
  decoder with its latent points swapped by `latent_partner`.

An autoencoder built with the same left/right pairs as the skeleton passes; one
that does not is refused (set `flip_p: 0` or rebuild the autoencoder).

## Not built

- The paper's hybrid student-teacher variant (one model with both a channel head
  and a latent-point head, plus a teacher term between them).
- Decoding the latent points at prediction time.
- Other model types (single-instance, bottom-up, multi-class top-down).
