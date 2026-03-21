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

## GPU & Hardware {#gpu-hardware}

SLEAP-NN uses PyTorch, so it runs on any system where PyTorch is supported. This includes NVIDIA CUDA GPUs, Apple Silicon via MPS, and AMD GPUs via ROCm (with caveats).

### General Questions

??? question "Do I need a GPU?"
    GPUs significantly speed up training and inference but aren't required. CPU-only works but is much slower.

    - **Training**: GPU strongly recommended (10-100x faster)
    - **Inference**: GPU recommended for real-time or large-scale processing
    - **Labeling**: CPU is fine for the labeling GUI

??? question "How much VRAM do I need?"
    | VRAM | Recommendation |
    |------|----------------|
    | **< 6 GB** | Generally insufficient; frequent out-of-memory errors |
    | **8–12 GB** | Minimum recommended for smooth training |
    | **24–48 GB** | Ideal for large models, multi-animal projects, or hyperparameter sweeps |

    Tips for limited VRAM:

    - Reduce batch size: `trainer_config.train_batch_size: 2`
    - Scale down images: `data_config.preprocessing.scale: 0.5`
    - Use smaller backbones: `model_config.backbone_config.unet.filters: 16`

??? question "Which GPU should I buy?"
    | Budget | Recommendation |
    |--------|----------------|
    | **Budget** | RTX 3060 12GB (~$300) - Great entry point |
    | **Mid-range** | RTX 4070 Super 12GB, RTX 4080 16GB |
    | **High-end** | RTX 4090 24GB, RTX 5090 32GB |
    | **Workstation** | RTX A5000 24GB, A6000 48GB |

    The RTX 3060 12GB is often recommended as a cost-effective option that handles most SLEAP workloads well.

### NVIDIA GPUs (Recommended)

??? question "Which NVIDIA GPUs are supported?"
    SLEAP-NN works with most NVIDIA GPUs that support CUDA. Here's our compatibility matrix:

    | GPU Series | VRAM | Status | Notes |
    |------------|------|--------|-------|
    | **RTX 50-series** (5090, 5080, etc.) | 16-32 GB | Tested ✓ | Requires CUDA 12.8+, PyTorch 2.6+ |
    | **RTX 40-series** (4090, 4080, 4070, etc.) | 8-24 GB | Tested ✓ | Excellent performance |
    | **RTX 30-series** (3090, 3080, 3070, 3060) | 8-24 GB | Works | Should work with standard installation |
    | **RTX 20-series** (2080, 2070, 2060) | 6-11 GB | Works | VRAM may limit larger models |
    | **GTX 10-series** (1080, 1070, etc.) | 4-11 GB | Limited | May need older PyTorch; limited VRAM |
    | **Workstation** (A6000, A5000, RTX 6000) | 24-48 GB | Tested ✓ | Excellent for large-scale training |
    | **Data center** (A100, H100) | 40-80 GB | Works | Overkill for most users |

??? question "Which CUDA version do I need?"
    PyTorch 2.10 supports CUDA 12.6, 12.8, and 13.0. Use `--torch-backend auto` to auto-detect:

    ```bash
    uv tool install sleap-nn --torch-backend auto
    ```

    **For RTX 50-series (Blackwell) GPUs**: You need CUDA 12.8 or higher and driver R570+. These GPUs have compute capability 10.0/12.0 which requires the latest PyTorch builds.

    **For older GPUs** (GTX 10-series, Quadro P-series): You may need to use older PyTorch versions or export to ONNX for CPU inference, as newer PyTorch versions have dropped support for Pascal architecture (compute capability 6.x).

??? question "How do I check if my GPU is working?"
    ```bash
    sleap-nn system
    ```
    This shows GPU details and runs diagnostic tests. You should see your GPU listed with CUDA support enabled.

    You can also verify PyTorch GPU access:
    ```python
    import torch
    print(torch.cuda.is_available())  # Should be True
    print(torch.cuda.get_device_name(0))  # Should show your GPU
    ```

### Apple Silicon (M1/M2/M3/M4)

??? question "Does SLEAP-NN work on Apple Silicon Macs?"
    Yes, SLEAP-NN works on M1/M2/M3/M4 Macs using the MPS (Metal Performance Shaders) backend.

    **Performance notes:**

    - MPS is slower than CUDA for training (expect 2-5x slower for large models)
    - Inference works well for moderate workloads
    - Good choice for users who don't need industrial-scale speed
    - Unified memory helps avoid traditional VRAM limits

    To use MPS:
    ```bash
    sleap-nn track -i video.mp4 -m models/ --device mps
    ```

??? question "Why is training slow on my Mac?"
    Apple Silicon MPS is not as optimized as NVIDIA CUDA for deep learning. This is expected.

    Tips for better performance:

    - Use smaller batch sizes (memory is shared with system)
    - Consider using a smaller backbone (UNet instead of Swin Transformer)
    - For production training, consider cloud GPU services (Colab, Lambda Labs, etc.)

### AMD GPUs

??? question "Can I use an AMD GPU?"
    PyTorch supports AMD GPUs via ROCm on Linux, but **SLEAP-NN has not been tested on AMD hardware**.

    If you want to try:

    - ROCm is only available on Linux (not Windows or macOS)
    - You'll need to install PyTorch with ROCm support manually
    - Some CUDA-specific operations may not work
    - We cannot provide support for ROCm-related issues

    We recommend NVIDIA GPUs for the most reliable experience.

### Troubleshooting

??? question "Out of GPU memory (CUDA OOM)"
    Common solutions:

    1. **Reduce batch size**: `trainer_config.train_batch_size: 2`
    2. **Scale down images**: `data_config.preprocessing.scale: 0.5`
    3. **Limit instances during inference**: `--max_instances 5`
    4. **Use disk caching**: `data_config.data_pipeline_fw: torch_dataset`
    5. **Close other GPU applications**

    Check current GPU memory usage:
    ```bash
    nvidia-smi
    ```

??? question "GPU not detected"
    1. Verify NVIDIA drivers are installed: `nvidia-smi`
    2. Check CUDA installation: `nvcc --version`
    3. Verify PyTorch sees the GPU:
       ```python
       import torch
       print(torch.cuda.is_available())
       ```
    4. Reinstall with correct CUDA version:
       ```bash
       uv tool install sleap-nn --torch-backend cu126 --reinstall
       ```

??? question "CUDA kernel not compatible with compute capability"
    This error means your GPU architecture isn't supported by the installed PyTorch/CUDA version.

    **For very new GPUs** (RTX 50-series): Update to PyTorch 2.6+ with CUDA 12.8+

    **For older GPUs** (GTX 10-series, Quadro P-series): These GPUs use Pascal architecture (compute capability 6.x) which has been dropped from newer PyTorch versions. Options:

    - Use an older PyTorch version
    - Export model to ONNX and run inference on CPU
    - Use a newer GPU

---

## Installation

??? question "Which Python version should I use?"
    Python 3.11, 3.12, or 3.13. Python 3.14 is not yet supported.

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
    - See [GPU & Hardware](#gpu-hardware) for more hardware-related troubleshooting

??? question "Poor predictions"
    - Check training loss - did it converge?
    - Verify preprocessing matches training
    - Add more training data
    - See [Inference guide](../guides/inference.md#filtering-instances) for post-processing filters
