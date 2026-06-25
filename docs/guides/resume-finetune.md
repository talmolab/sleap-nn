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

Resume from a previous checkpoint:

```bash
sleap-nn train --config config.yaml \
    trainer_config.resume_ckpt_path=/path/to/checkpoint.ckpt
```

This restores both model weights and optimizer state.

!!! warning "Ensure the same seed when resuming"
    The train/val split is regenerated on resume — it is **not** saved in the checkpoint. If you change `trainer_config.seed` between runs (default: `42`), you will get a different split, which can leak training data into validation. Always use the same seed as the original run. `sleap-nn` will warn you if it detects a mismatch.
