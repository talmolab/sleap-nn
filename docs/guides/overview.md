# Guides

Task-oriented guides for common workflows.

| Guide | Description |
|-------|-------------|
| [Training](training/overview.md) | Configure and run model training |
| &nbsp;&nbsp;&nbsp;&nbsp;[Training Guide](training-guide.md) | Configure and run model training |
| &nbsp;&nbsp;&nbsp;&nbsp;[Config Generator](config-generator.md) | Generate training configs via TUI or auto mode |
| &nbsp;&nbsp;&nbsp;&nbsp;[Pretrained Backbones](pretrained-backbones.md) | Use a HuggingFace pretrained image model as the backbone |
| &nbsp;&nbsp;&nbsp;&nbsp;[Negative Frames](negative-frames.md) | Reduce false positives with background frames |
| &nbsp;&nbsp;&nbsp;&nbsp;[Centroid Focal Loss](centroid-focal-loss.md) | Opt-in focal loss for the centroid head in dense scenes |
| &nbsp;&nbsp;&nbsp;&nbsp;[Monitoring](monitoring.md) | WandB, visualizations, epoch-end evaluation |
| &nbsp;&nbsp;&nbsp;&nbsp;[Multi-GPU](multi-gpu.md) | Scale training across multiple GPUs |
| &nbsp;&nbsp;&nbsp;&nbsp;[Resume & Fine-Tune](resume-finetune.md) | Continue from existing weights |
| &nbsp;&nbsp;&nbsp;&nbsp;[Supervised ID](supervised-id.md) | Predict pose + persistent identity in one model |
| &nbsp;&nbsp;&nbsp;&nbsp;[Tiling (High-Res)](tiling.md) | Train and predict on overlapping tiles of very large frames |
| [Inference](inference/overview.md) | Run predictions on videos and label files |
| &nbsp;&nbsp;&nbsp;&nbsp;[Running Inference](inference-guide.md) | Run predictions from the CLI |
| &nbsp;&nbsp;&nbsp;&nbsp;[Post-Processing Filters](post-processing-filters.md) | Node-count, confidence, and overlap-NMS filters |
| &nbsp;&nbsp;&nbsp;&nbsp;[Python API](inference-api.md) | `Predictor` / `predict` / `Outputs` for embedding inference in code |
| &nbsp;&nbsp;&nbsp;&nbsp;[Centroid-Only](centroid-only-inference.md) | Run a centroid model standalone |
| &nbsp;&nbsp;&nbsp;&nbsp;[Top-Down Segmentation](topdown-segmentation.md) | Per-instance masks via centroid + crop-mask |
| &nbsp;&nbsp;&nbsp;&nbsp;[Semantic Segmentation](semantic-segmentation.md) | One whole-frame foreground mask per frame |
| &nbsp;&nbsp;&nbsp;&nbsp;[Bounding-Box Detection](bbox-detection.md) | Box detectors (`bbox_centernet`, `bbox_ssd`) |
| &nbsp;&nbsp;&nbsp;&nbsp;[Top-Down with a Box Detector](bbox-topdown.md) | A box detector as the first top-down stage |
| &nbsp;&nbsp;&nbsp;&nbsp;[SAM-Prompted Segmentation](sam-inference-segmentation.md) | Per-instance masks from poses via Segment Anything |
| &nbsp;&nbsp;&nbsp;&nbsp;[Performance](inference-performance.md) | Tune inference throughput (FP16, `torch.compile`, workers) |
| [Evaluation](evaluation.md) | Assess model performance with metrics |
| [Tracking](tracking.md) | Assign consistent IDs across frames |
| [Embedding (re-ID) Tracking](embedding-tracking.md) | Track by appearance with a re-ID embedding model |
| [Export](export.md) | ONNX/TensorRT for production inference |

---

Looking for step-by-step learning? Check out the [Tutorials](../tutorials/overview.md).
