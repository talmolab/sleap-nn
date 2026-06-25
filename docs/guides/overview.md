# Guides

Task-oriented guides for common workflows.

| Guide | Description |
|-------|-------------|
| [Training](training/overview.md) | Configure and run model training |
| &nbsp;&nbsp;&nbsp;&nbsp;[Config Generator](config-generator.md) | Generate training configs via TUI or auto mode |
| &nbsp;&nbsp;&nbsp;&nbsp;[Negative Frames](negative-frames.md) | Reduce false positives with background frames |
| &nbsp;&nbsp;&nbsp;&nbsp;[Monitoring](monitoring.md) | WandB, visualizations, epoch-end evaluation |
| &nbsp;&nbsp;&nbsp;&nbsp;[Multi-GPU](multi-gpu.md) | Scale training across multiple GPUs |
| &nbsp;&nbsp;&nbsp;&nbsp;[Resume & Fine-Tune](resume-finetune.md) | Continue from existing weights |
| [Inference](inference/overview.md) | Run predictions on videos and label files |
| &nbsp;&nbsp;&nbsp;&nbsp;[Python API](inference-api.md) | `Predictor` / `predict` / `Outputs` for embedding inference in code |
| &nbsp;&nbsp;&nbsp;&nbsp;[Centroid-Only](centroid-only-inference.md) | Run a centroid model standalone |
| &nbsp;&nbsp;&nbsp;&nbsp;[Top-Down Segmentation](topdown-segmentation.md) | Per-instance masks via centroid + crop-mask |
| &nbsp;&nbsp;&nbsp;&nbsp;[SAM-Prompted Segmentation](sam-inference-segmentation.md) | Per-instance masks from poses via Segment Anything |
| &nbsp;&nbsp;&nbsp;&nbsp;[Performance](inference-performance.md) | Tune inference throughput (FP16, `torch.compile`, workers) |
| [Evaluation](evaluation.md) | Assess model performance with metrics |
| [Tracking](tracking.md) | Assign consistent IDs across frames |
| [Export](export.md) | ONNX/TensorRT for production inference |

---

Looking for step-by-step learning? Check out the [Tutorials](../tutorials/overview.md).
