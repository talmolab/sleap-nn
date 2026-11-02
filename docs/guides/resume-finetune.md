# Resuming & Fine-Tuning

Continue training from existing weights — either resuming an interrupted run or
fine-tuning a pre-trained model on new data.

---

## Fine-tuning / Transfer Learning

Initialize with pre-trained weights:

```yaml
model_config:
  pretrained_backbone_weights: /path/to/best.ckpt
  pretrained_head_weights: /path/to/best.ckpt
```

Works with:

- Previous SLEAP-NN checkpoints

- Legacy SLEAP `.h5` files (UNet only)

---

## Resume Training

Point `--resume` at a previous run's folder:

```bash
sleap-nn train --resume models/my_run
```

This restores the model weights, optimizer and scheduler state, and the epoch
count, and continues training:

- It uses the run's own `training_config.yaml`, so you don't need the original
  config file. Overrides still apply, e.g. to train for longer:
  `sleap-nn train --resume models/my_run trainer_config.max_epochs=200`.
- It resumes from `last.ckpt` (the most recent epoch, written when
  `trainer_config.model_ckpt.save_last` is on) if the folder has one, otherwise
  from `best.ckpt`. To pick a checkpoint, pass the file:
  `sleap-nn train --resume models/my_run/best.ckpt`.
- The resumed run is written to a **new folder** next to the original
  (`models/my_run-1`, then `my_run-2`, ...), so the original run is never
  overwritten. Pass `trainer_config.run_name=...` to name it yourself.
- If the checkpoint already reached `max_epochs`, `--resume` stops and tells you
  to raise it, instead of starting a run with nothing to train.

To resume with a different config file, set the checkpoint directly:

```bash
sleap-nn train --config config.yaml \
    trainer_config.resume_ckpt_path=/path/to/checkpoint.ckpt
```

!!! warning "Ensure the same seed when resuming"
    The train/val split is regenerated on resume — it is **not** saved in the checkpoint. If you change `trainer_config.seed` between runs (default: `42`), you will get a different split, which can leak training data into validation. Always use the same seed as the original run. `sleap-nn` will warn you if it detects a mismatch.
