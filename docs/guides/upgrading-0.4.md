# Upgrading from 0.3

This page lists every default and output of sleap-nn 0.4 that differs from v0.3.5,
with who is affected and how to get the old behaviour where there is a switch. New
features that are off by default are listed only where they replace or extend an
existing option; for the rest, see the release notes.

Saved configs from 0.3 train and predict as before, except where an item below says
otherwise.

## Files and dependencies

- **sleap-io 0.10 is required** (was `>=0.9.3,<0.10`).
  sleap-io 0.10 reads both `.slp` file layouts: the one SLEAP 1.x and sleap-io 0.9
  write, and the newer format 3.0, which stores separate *labelled* and *visible*
  flags per point. With nothing set, a file whose content fits the old layout is
  written in it, so SLEAP 1.x and sleap-io 0.9 open it; a file that needs format 3.0
  is written as 3.0, with a warning. Centroid-only output fits the old layout.
  `sleap-nn detect-boxes` always writes 3.0, because each box keeps its matching
  score.
- **Save options.** `predict`, `track` and `detect-boxes` take `--slp_version
  {auto,v2,v3}`, `--image_format`, `--image_quality`, `--compression` and
  `--unlabeled_as` to choose the layout and how embedded frames are stored. The
  defaults give the behaviour above.
- **Default output name of `sleap-nn predict`.** Without `-o`, `sleap-nn predict -i
  video.mp4` now writes `video.predictions.slp`, as SLEAP's `sleap-track` does (0.3.5
  wrote `video.mp4.slp`, and `labels.slp.slp` for a `labels.slp` input). Who:
  scripts that look for the old name. Old behaviour: pass `-o` explicitly.

## Prediction

- **`--integral_refinement local` works.** It used to do nothing (grid peaks); it now
  applies SLEAP's quarter-pixel step towards the higher neighbour. `local` (and
  `dark`) with a bottom-up model raises, and an unknown refinement name raises a
  `ValueError` instead of silently giving grid peaks. Who: anyone who passed `local`
  or a misspelled name.
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
  keypoint model as a plain fp16 engine prints an accuracy warning. Additive only.
- **New `--precision {32,bf16,fp16}` for `sleap-nn predict`**: half-precision
  inference of checkpoint models (`32` / `fp32`, `bf16` and `fp16`, in any case). The
  default, `32`, is the 0.3.5 behaviour. A `bf16` or `fp16` run on CUDA records
  `autocast_precision` in the output file's provenance; fp32 runs have no such key.
- **Labelled boxes follow the model's box rule on every frame source.** A top-down
  model run on labelled boxes (`--boxes gt`, or a box-scaled crop model given alone)
  crops each labelled instance with the box its training crops used (its linked user
  box, else the padded box of its located keypoints) and crops user instances only,
  also when a `.slp` frame filter (`--only_labeled_frames`, `--only_predicted_frames`,
  `--exclude_user_labeled`, `--only_suggested_frames`) or `--stream-to-file` is given,
  and also for a `LabelsProvider` or `VideoSelection` passed in Python (development
  builds of 0.4 cropped tight boxes of the visible keypoints on those routes, and
  could crop predicted instances). Who: these command lines and callers that build
  their own provider; a caller who wants other boxes can set `instance_box_fn` on the
  provider.
- **`--frames` ignored with a frame filter now warns.** On a `.slp` without
  `--video_index`, `--frames` given together with one of the four frame filters above,
  or with `--stream-to-file`, is ignored; `sleap-nn predict` now says so and points
  to `--video_index N`, where both combine. The frames that run are
  unchanged.
- **Hidden points of old files are not used as boxes.** Boxes from labelled keypoints
  (`--boxes gt`, or a box-scaled crop model given alone) ignore the stored coordinates
  of hidden points in files older than format 3.0, as 0.3.5 did (see Training below).
- **Frame-size messages.** At prediction, frames fitted to a model's `max_height` /
  `max_width` log one info line per size instead of a warning; the box-padding warning
  for a detector and a crop model trained with different `bbox_padding` says what the
  difference does. Predictions are unchanged.

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
  to the old keys. The error a shape mismatch raises is now a `ValueError` (0.3.5
  raised PyTorch's `RuntimeError`); code that catches the old type needs updating.
- **Early stopping is unchanged.** It stays on by default under every learning-rate
  schedule, as in 0.3.5. Turn it off with
  `trainer_config.early_stopping.stop_training_on_plateau: false`, for example with a
  fixed-length cosine or linear-decay schedule that it could cut short.
- **Sample order on one device.** Training on a single device draws the same split
  and the same per-epoch permutations as 0.3.5, shifted by one epoch, so the first
  epoch's order differs and a run does not reproduce 0.3.5 bit for bit. It is
  statistically the same. (The training data loader now starts its iterator on first
  use, so resumed runs and epoch-aware samplers start at the right epoch.) No switch.
- **Data loading.** `prefetch_factor` defaults to `2` (0.3.5 used the batch size,
  which could exhaust host or shared memory with large batches), and
  `num_workers: auto` may pick up to 16 workers (was 4); its memory bound now also
  applies when there are no validation labels (0.3.5's cache estimate counted
  nothing then). Who: every training with workers. Old behaviour: set
  `train_data_loader.prefetch_factor` to the batch size and `num_workers` to a number.
- **Legacy hard keypoint mining** (`per_sample: false`) no longer turns into plain
  MSE when one channel's loss is 0: the easiest-keypoint reference ignores zero-loss
  channels. Results are unchanged when every channel's loss is positive.
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
- **Node order is checked.** Every training and validation labels file must have
  the same nodes in the same order as the first training file; a file with the same
  nodes in another order now raises a `ValueError` (0.3.5 accepted it and trained
  every channel on the wrong node, and did not check validation files). Reorder with
  sleap-io's `Labels.reorder_nodes`. The warp data path (`crop_pipeline: warp`)
  matches nodes by name and is not affected.
- **Per-epoch warmup no longer starts at LR 0.** With `cosine_annealing_warmup` or
  `linear_warmup_linear_decay` warming up per epoch, epoch 0 used to train at
  `warmup_start_lr` (default 0); warmup epoch `e` now trains at `(e + 1) /
  (warmup_epochs + 1)` of the way to the base LR. Who: runs with an epoch-based
  warmup. No switch.
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
- **Hidden points of old files are cleared before training.** In `.slp` files older
  than format 3.0 a hidden point keeps the coordinates it had when it was hidden.
  sleap-io 0.10 loads such files as stored; sleap-nn clears those coordinates (and
  marks visible points without coordinates as hidden) on the training and validation
  labels, on test files in post-training evaluation and on the ground truth of
  `sleap-nn eval`, so they are read as absent, as in 0.3.5. Labels from format 3.0
  files and labels built in memory are left as they are. `Labels` passed to the
  trainer or to `run_evaluation` are changed in place. No switch.
- **A node mask that matches no node stops training.** When the
  `provenance["labeled_nodes"]` entries of every training file name no node of the
  skeleton (for example after the nodes were renamed), training raises a `ValueError`
  that names the entries and the nodes, instead of training every node at zero loss.
  A new warning names nodes that are labelled in every instance of a file but located
  in none.
- **Zoom-out for CenterNet and SSD detectors.** The `zoom_out_*` keys of
  `augmentation_config.detr` (new in 0.4) apply to `bbox_centernet` and `bbox_ssd` as
  to `bbox_detr`, so a config that sets `zoom_out_p` above 0 for them zooms out; with
  `zoom_out_canvas: frame` each training sample holds up to twice as many boxes. Who:
  configs written for development builds of 0.4, which ignored these keys for those
  models. Off by default (`zoom_out_p: 0`).
- **`geometric.mixup_p > 0` warns** that mixup is not implemented (0.3.5 ignored it
  silently); `crop_pipeline: warp` raises on it.

## Logging

- **W&B prediction gallery.** With W&B on, `trainer_config.wandb.gallery: true` (the
  default) logs a fixed panel of validation predictions next to the ground truth at
  every validation, for box detectors, top-down and bottom-up pose models, plus a few augmented
  training samples (new `viz/gallery/*` keys and uploads). Turn it off with
  `gallery: false`. See [WandB logging](../configuration/trainer.md#wandb-logging).
- **Throughput metrics.** Training logs `perf/*` (samples per second, data wait, GPU
  idle share, peak memory, learning rates) every 50 steps; set
  `trainer_config.throughput_log_every_n_steps: 0` to turn it off. Rows of
  `training_log.csv` are no longer one epoch off.

## Export

- **fp16 engines of plain-ViT backbones keep their attention logits in fp32.** With
  the default `--precision fp16`, a model (or a top-down bundle) with a plain-ViT
  backbone (DINOv2 / v3, CLIP, SigLIP, MAE ..., and the windowed DINOv2 backbone of
  `bbox_detr`) keeps each attention block's logits product in fp32, because plain
  fp16 engines of these models move keypoints and lose boxes. On TensorRT 10 such a
  build is strongly typed: the export folder also holds
  `tensorrt_graph/model.fp16.onnx`, and `model.trt.metadata.json` says `typing:
  strong` and gains keys. Below TensorRT 10.13, or on a GPU without fast fp16, the old
  build is kept (with a warning below 10.13). Old behaviour:
  `--no-fp32-attention-logits`. See
  [fp16 for ViT backbones](export.md#fp16-for-vit-backbones-fp32-attention-logits).
- **TensorRT 10 fp16 engines of models that find peaks** (centroid, bottom-up,
  top-down bundles) hold -1e9 in the empty slots of their value outputs, as the ONNX
  graph and TensorRT 11 engines do (0.3.5's engines could return -65504 or -inf). A
  few small layers run in fp32 (`fp16_fp32_layers` in the metadata). Points are
  unchanged.
- **Multi-class bottom-up engines on TensorRT 10 build fp32** when fp16 or bf16 is
  asked for, with a warning, because their reduced-precision engines do not reproduce
  the fp32 graph. Old behaviour: `--force-precision`.
- **Single-instance models with an input scale other than 1** now build TensorRT
  engines. Their ONNX graph resizes each frame to its own scaled size, as checkpoint
  inference does, with an antialiased resize that matches torch's (0.3.5's graph
  resized every frame to the traced size). For scales such as 0.7 the size can be one
  pixel larger than `int(size * scale)`, with a warning at export.
- **`--runtime tensorrt --device cuda:N`** is accepted (it was refused); `cuda` and
  `auto` bind the engine to the device that is current when it is loaded.
