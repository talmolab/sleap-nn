# FAQ

Frequently asked questions about SLEAP-NN.

---

## General

??? question "What is SLEAP-NN?"
    SLEAP-NN is the PyTorch-based neural network backend for [SLEAP](https://sleap.ai). It handles training and inference for animal pose estimation models.

??? question "How is SLEAP-NN different from SLEAP?"
    - **SLEAP** (<v1.5): TensorFlow backend, GUI labeling tool
    - **SLEAP-NN**: PyTorch backend, CLI-focused, faster training, multi-GPU support. This is currently the neural network backend for SLEAP (>=v1.5)

??? question "Can I use my existing SLEAP data?"
    Yes! SLEAP-NN reads `.slp` and `.pkg.slp` files created by SLEAP.

---

## Migrating from SLEAP <=v1.4 {#migrating-from-sleap-1x}

??? question "Can I use my SLEAP <=v1.4 trained models?"
    Yes, but only **UNet backbone** models. Load them like any other model:
    ```bash
    sleap-nn track -i video.mp4 -m /path/to/sleap_model/
    ```
    The directory should contain `best_model.h5` and `training_config.json`.

??? question "Can I convert SLEAP config files?"
    Yes:
    ```python
    from sleap_nn.config.training_job_config import TrainingJobConfig
    from omegaconf import OmegaConf

    config = TrainingJobConfig.load_sleap_config("config.json")
    OmegaConf.save(config, "config.yaml")
    ```

??? question "What features are new in SLEAP-NN?"
    - Multi-GPU training (DDP)
    - Swin Transformer and ConvNeXt backbones
    - ONNX/TensorRT export
    - Faster augmentation (Skia backend)
    - Better WandB integration

---

## Installation

??? question "Which Python version should I use?"
    Python 3.11, 3.12, or 3.13. Python 3.14 is not yet supported.

??? question "Do I need a GPU?"
    GPUs significantly speed up training and inference but aren't required. CPU-only works but is slower.

??? question "Which CUDA version do I need?"
    SLEAP-NN supports CUDA 11.8, 12.8, and 13.0. Use `--torch-backend auto` to auto-detect:
    ```bash
    uv tool install sleap-nn[torch] --torch-backend auto
    ```

??? question "How do I check if my GPU is working?"
    ```bash
    sleap-nn system
    ```
    This shows GPU details and runs diagnostic tests.

---

## Training

??? question "How much training data do I need?"
    - Minimum: 50-100 labeled frames
    - Good: 200-500 labeled frames
    - Better: 1000+ for complex scenarios

    More diverse poses and scenarios improve generalization.

??? question "How do I know when training is done?"
    - Enable early stopping (default)
    - Watch validation loss plateau
    - Use WandB for detailed monitoring

??? question "Can I resume training?"
    Yes:
    ```bash
    sleap-nn train --config config.yaml \
        trainer_config.resume_ckpt_path=/path/to/checkpoint.ckpt
    ```

??? question "How do I use multiple GPUs?"
    ```yaml
    trainer_config:
      trainer_devices: 4
      trainer_strategy: ddp
    ```
    See the [Multi-GPU Training Guide](../guides/multi-gpu.md) for detailed setup and troubleshooting.

---

## Inference

??? question "How do I speed up inference?"
    1. Increase batch size: `--batch_size 8`
    2. Use GPU: `--device cuda`
    3. Export to TensorRT: `sleap-nn export` + `sleap-nn predict`

??? question "How do I run on specific frames?"
    ```bash
    sleap-nn track -i video.mp4 -m models/ --frames 0-100,500-600
    ```

??? question "How do I limit detected instances?"
    ```bash
    sleap-nn track -i video.mp4 -m models/ --max_instances 5
    ```

---

## Models

??? question "Which model type should I use?"
    | Scenario | Model |
    |----------|-------|
    | One animal | Single Instance |
    | Multiple, not touching | Top-Down |
    | Multiple, overlapping | Bottom-Up |
    | Known identities | Multi-Class |

??? question "Which backbone should I use?"
    - **UNet**: Most flexible, works for any resolution
    - **ConvNeXt**: Good with pretrained weights
    - **Swin Transformer**: Best for large images, highest memory

??? question "What is sigma in the config?"
    Sigma controls the Gaussian spread for confidence maps:
    - Larger (5-10): Easier to learn, less precise
    - Smaller (1-3): More precise, harder to learn

---

## Troubleshooting

??? question "Training is very slow"
    - Enable caching: `data_config.data_pipeline_fw: torch_dataset_cache_img_memory`
    - Reduce image size: `data_config.preprocessing.scale: 0.5`
    - Check GPU is being used: `sleap-nn system`

??? question "Out of GPU memory"
    - Reduce batch size
    - Reduce model size (`filters: 16`)
    - Reduce image resolution

??? question "Poor predictions"
    - Check training loss - did it converge?
    - Verify preprocessing matches training
    - Add more training data
