# Upgrading from 0.3

This page lists every default and output of sleap-nn 0.4 that differs from v0.3.5,
with who is affected and how to get the old behaviour where there is a switch. New
features that are off by default are not listed here; see the release notes.

Saved configs from 0.3 train and predict as before, except where an item below says
otherwise.

## Files and dependencies

- **sleap-io 0.10 is required** (`sleap-io>=0.10,<0.11`, was `>=0.9.3,<0.10`).
  sleap-io 0.10 reads both `.slp` file layouts (the one SLEAP 1.x and sleap-io 0.9
  write, and the newer format 3.0 that stores separate *labelled* and *visible* flags
  per point). Which layout `sleap-nn predict` and the other commands write by default,
  and the options to choose it, are set by sleap-io and by `predict`'s save options.
  <!-- verify against sleap-io 0.10 -->
- **Default output name of `sleap-nn predict`.** Without `-o`, `sleap-nn predict -i
  video.mp4` now writes `video.predictions.slp`, as SLEAP's `sleap-track` does (0.3.5
  wrote `video.mp4.slp`, and `labels.slp.slp` for a `labels.slp` input). Who:
  scripts that look for the old name. Old behaviour: pass `-o` explicitly.

## Prediction

- **Top-down keypoints use DARK refinement.** `--integral_refinement` defaults to
  `auto`: DARK sub-pixel refinement for the centered-instance stage of top-down models
  (`centered_instance`, `multi_class_topdown`), integral refinement for every other
  model, as before. Keypoints of top-down models move by a fraction of a pixel. The
  epoch-end and post-training evaluation of centered-instance models decode with DARK
  too. Old behaviour: `--integral_refinement integral` (Python:
  `integral_refinement="integral"`). The legacy `sleap-nn track` command keeps
  `integral`. Exported ONNX / TensorRT models do no sub-pixel refinement, so they now
  differ a little more from checkpoint output than before. See [Sub-pixel refinement
  and flip-test](inference-guide.md#sub-pixel-refinement-and-flip-test).
- **Top-down crops are cut on the GPU** when the centered-instance model runs on
  CUDA. Fixed-size crops are identical; crops that are resized (box-scaled crops) can
  differ by one grey level on about 0.1% of pixels. Who: top-down users on a GPU. No
  switch.
- **Box detectors with a fixed input size** (`max_height` and `max_width` set) read
  `.slp` frames already fitted to that size on the CPU, decoding large JPEGs at reduced
  size. Box coordinates can differ very slightly from 0.3.5. Who: `bbox_centernet` /
  `bbox_ssd` users predicting on `.slp` files. No switch.
- **Centroid-only output.** When centroid-only output keeps a multi-node skeleton, the
  nodes the centroid model does not predict are written as not labelled
  (`labeled=False`) rather than as hidden points. The default single-node `centroid`
  output is unchanged.
- **Exported models.** Predictions from export folders add the export (file, runtime,
  precision) to the `.slp` provenance (`exported_models`); every TensorRT build records
  more metadata (`tensorrt_version`, `typing`, `precision_by_scope`); exporting a ViT
  keypoint model at plain fp16 prints an accuracy warning. Additive only.
- **`--frames_on_gpu` is a deprecated alias of `--upload_batches`**, which moves each
  batch of frames to the GPU for every pipeline. Who: users of pre-release builds that
  had `--frames_on_gpu`. Use `--upload_batches` instead.
- **New `--precision {32,bf16,fp16}` for `sleap-nn predict`**: half-precision
  inference of checkpoint models (`32` / `fp32`, `bf16` and `fp16`, in any case). The
  default, `32`, is the 0.3.5 behaviour. A `bf16` or `fp16` run on CUDA records
  `autocast_precision` in the output file's provenance; fp32 runs have no such key.

## Training

- **Loading `.ckpt` weights.** `model_config.pretrained_backbone_weights` and
  `pretrained_head_weights` behave as in 0.3.5: a tensor whose shape does not match
  raises, and head channels are copied by position. The new `model_config.init_from`
  is lenient by default: shape mismatches are skipped with a warning and head channels
  are matched by node name (see [Init from another model](init-from.md)). To give the
  old keys that behaviour, set `init_from.remap_nodes: true` (match head channels by
  node name) and `init_from.strict: false` (skip mismatched shapes). With the default
  strict loading, a learned positional-embedding table of another size raises, as in
  0.3.5; with `init_from.strict: false` it is resized. When a run is resumed,
  `init_from.path` and `reinit` are skipped, but `strict` and `remap_nodes` still apply
  to the old keys.
- **Early stopping is unchanged.** It stays on by default under every learning-rate
  schedule, as in 0.3.5. Turn it off with
  `trainer_config.early_stopping.stop_training_on_plateau: false`, for example with a
  fixed-length cosine or linear-decay schedule that it could cut short.
- **Sample order on one device.** Training on a single device draws the same split
  and the same per-epoch permutations as 0.3.5, shifted by one epoch, so the first
  epoch's order differs and a run does not reproduce 0.3.5 bit for bit. It is
  statistically the same. (The training data loader now starts its iterator on first
  use, so resumed runs and epoch-aware samplers start at the right epoch.) No switch.
- **Seeding a resumed run.** By default a resumed run is seeded with `seed` again, as
  in earlier releases, so on one device without `balanced_sampler` its first resumed
  epoch repeats the order of the original first epoch. The new
  `trainer_config.reseed_on_resume: true` (default `false`) seeds `random`, numpy and
  torch with `seed + epoch` instead.
- **Data loading.** `prefetch_factor` defaults to `2` (0.3.5 used the batch size,
  which could exhaust host or shared memory with large batches), and
  `num_workers: auto` may pick up to 16 workers (was 4). Who: every training with
  workers. Old behaviour: set `train_data_loader.prefetch_factor` to the batch size
  and `num_workers` to a number.
- **Online hard keypoint mining ranks per sample.** With `online_mining: true`,
  `online_hard_keypoint_mining.per_sample` defaults to `true`: each sample's nodes are
  ranked separately. Old behaviour: `per_sample: false` (ranking over the batch).
- **Post-training evaluation** predicts the splits in the order val, test, train, at
  the training precision (bf16 autocast for `precision: bf16-mixed` models) and with
  the validation batch size, so reported metrics can shift slightly. Old behaviour for
  the precision: `trainer_config.eval.post_train_precision: "32"`. Large training sets
  can be skipped or sampled (`post_train_splits`, `post_train_max_train_frames`).
- **Multi-GPU training is corrected** in five places: the epoch-end evaluation scores
  the whole validation set, `best.ckpt` keeps updating when it tracks an evaluation
  metric, early stopping follows the checkpoint monitor, each GPU draws its own
  augmentation, and each epoch gets a new shuffle. Multi-GPU runs do not reproduce
  0.3.5; single-GPU training is unchanged. No switch.
- **The training rank is read from `RANK`, `LOCAL_RANK`, `SLURM_PROCID` or
  `JSM_NAMESPACE_RANK`**, in that order, as PyTorch Lightning does, so processes
  started by `srun` are no longer all the main process. Independent trainings started
  as the tasks of one `srun` should set `trainer_config.trainer_devices: 1`; each then
  runs as its own main process.
- **Validation with several files.** The pooled `eval/val/*` metrics are now computed
  correctly (they were keyed to the first file's videos), and each file is also scored
  on its own (`val/<name>/loss`, `eval/<name>/*`). Who: runs with several validation
  files and `eval.enabled`.
- **Centroid models hold out the same validation frames** as the other model types
  trained on the same labels, `validation_fraction` and seed, so a top-down pair can be
  scored on frames neither model trained on. `use_negative_frames` now works for
  centroid models (0.3.5 dropped every negative frame from their split). Who: anyone
  comparing centroid validation numbers with 0.3.5. No switch.
- **Config checks at training start.** Explicit `pafs.edges` that differ from the
  skeleton's edges (in order or direction), a PAF head with no edges, and a track name
  used by a training or validation instance with no class in `classes` now raise
  before training starts instead of failing later or training on mismatched targets.
  Inferred ID `classes` are sorted, so the class-to-channel order is the same on every
  run. Who: bottom-up and identity models with hand-written edges or classes.
- **Box models** (`bbox_*`) drop COCO crowd boxes (`metadata["iscrowd"]`) from their
  training targets, and a single-class box model skips boxes of other categories
  (0.3.5 trained them as class 0). Their saved config has `data_config.skeletons: []`,
  and their training files may carry any skeletons.
- **`geometric.mixup_p > 0` warns** that mixup is not implemented (0.3.5 ignored it
  silently); `crop_pipeline: warp` raises on it.

## Logging

- **W&B prediction gallery.** With W&B on, `trainer_config.wandb.gallery: true` (the
  default) logs a fixed panel of validation predictions next to the ground truth at
  every validation, for box detectors and top-down pose models, plus a few augmented
  training samples (new `viz/gallery/*` keys and uploads). Turn it off with
  `gallery: false`. See [WandB logging](../configuration/trainer.md#wandb-logging).
- **Throughput metrics.** Training logs `perf/*` (samples per second, data wait, GPU
  idle share, peak memory, learning rates) every 50 steps; set
  `trainer_config.throughput_log_every_n_steps: 0` to turn it off. Rows of
  `training_log.csv` are no longer one epoch off.
