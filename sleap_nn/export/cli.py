"""CLI entry points for export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import math
import shutil

import click
from loguru import logger

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument(
    "model_paths",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory for exported model files.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["onnx", "tensorrt", "both"], case_sensitive=False),
    default="onnx",
    show_default=True,
)
@click.option("--opset-version", type=int, default=17, show_default=True)
@click.option(
    "--input-dtype",
    type=click.Choice(["auto", "uint8", "float32"]),
    default="auto",
    show_default=True,
    help="Dtype of the graph's image input, for box detectors and --network-only "
    "exports (the host prepares their inputs). 'float32' receives the checkpoint "
    "path's own frame / crop, so exported and checkpoint predictions match up to "
    "the runtime's numerics; 'uint8' receives it rounded (a resized frame then "
    "differs by up to one grey level, which can move a bbox_detr's boxes slightly). "
    "'auto': float32 for bbox_detr, uint8 for every other model type.",
)
@click.option(
    "--network-only",
    is_flag=True,
    default=False,
    help="centered_instance only: export the network alone (crop in, raw "
    "confidence maps out; no peak finding baked in). `sleap-nn predict` then "
    "loads the directory in place of the checkpoint (e.g. `-m <detector export> "
    "-m <this>`, or `--part_model <this>`), with the checkpoint path's own crops "
    "(fixed or box-scaled) and peak finding. Box detectors always export this way.",
)
@click.option("--max-instances", type=int, default=20, show_default=True)
@click.option("--max-batch-size", type=int, default=8, show_default=True)
@click.option(
    "--opt-batch-size",
    type=int,
    default=None,
    help="TensorRT: the batch size the engine's kernels are tuned for (its "
    "optimization profile's 'opt' shape; at most --max-batch-size). Default: 1, as "
    "before. A throughput engine (large batches) runs faster tuned for its batch.",
)
@click.option("--input-scale", type=float, default=None)
@click.option("--input-height", type=int, default=None)
@click.option("--input-width", type=int, default=None)
@click.option("--crop-size", type=int, default=None)
@click.option("--max-peaks-per-node", type=int, default=20, show_default=True)
@click.option("--n-line-points", type=int, default=10, show_default=True)
@click.option("--max-edge-length-ratio", type=float, default=0.25, show_default=True)
@click.option("--dist-penalty-weight", type=float, default=1.0, show_default=True)
@click.option("--device", type=str, default="cpu", show_default=True)
@click.option(
    "--precision",
    type=click.Choice(["fp32", "fp16", "tf32", "bf16"], case_sensitive=False),
    default="fp16",
    show_default=True,
    help="TensorRT precision mode. 'bf16' needs a GPU with compute capability >= 8.0 "
    "and keeps fp32's range where fp16 overflows (a DINOv3 bbox_detr detector loses "
    "boxes in fp16); in graphs that compute coordinates (box detectors, peaks found in "
    "the graph) every layer outside the backbone is kept in fp32. TensorRT 11 builds "
    "fp16 / bf16 strongly typed (Cast nodes in the graph; see --strongly-typed). "
    "Reduced precision still moves outputs: check accuracy on your data against fp32.",
)
@click.option(
    "--strongly-typed",
    is_flag=True,
    default=False,
    help="TensorRT 10: build a strongly typed engine, with fp16 / bf16 carried by "
    "Cast nodes in the graph instead of builder flags. TensorRT 11 builds this way "
    "always (it removed the flags); the option lets TensorRT 10 build the same "
    "engines. Default off.",
)
@click.option(
    "--force-precision",
    is_flag=True,
    default=False,
    help="Build fp16 / bf16 even for a model family whose reduced-precision engines do "
    "not reproduce its fp32 graph (multi-class bottom-up). Without it such a family is "
    "built fp32 with a warning on the strongly typed path (TensorRT 11, "
    "--strongly-typed), and built as asked with a warning on TensorRT 10.",
)
@click.option(
    "--fp32-attention-logits",
    is_flag=True,
    default=False,
    help="With --precision fp16 on the strongly typed path (TensorRT 11, or "
    "--strongly-typed on 10): keep each attention block's logits product (Q.K^T, found "
    "by structure) in fp32, so TensorRT accumulates it in fp32; everything else is the "
    "usual fp16 rewrite (in graphs with coordinates, fp16 only inside the backbone). "
    "Measured on DINOv3 ViT-B detector / pose / tail models, whose plain fp16 engines "
    "fail. Default off.",
)
@click.option(
    "--peak-threshold",
    type=float,
    default=0.2,
    show_default=True,
    help="Minimum confidence for a peak. Pose and centroid models: baked into the "
    "exported graph. Box detectors (bbox_*): the score threshold, stored in "
    "export_metadata.json and applied on the host when the export is run.",
)
@click.option(
    "--workspace-size-gb",
    type=float,
    default=None,
    help="TensorRT builder workspace size in GB. Uses exporter default if unset.",
)
@click.option("--verify/--no-verify", default=True, show_default=True)
@click.option(
    "--nodes",
    "nodes",
    type=str,
    multiple=True,
    default=(),
    help=(
        "Export only these confmap nodes (single_instance / centered_instance): "
        "repeat or comma-separate. The head is sliced to their channels."
    ),
)
@click.option(
    "--node-prefix",
    "--node_prefix",
    "node_prefix",
    type=str,
    default=None,
    help="Export only the nodes whose name starts with this prefix.",
)
@click.option(
    "--strip-prefix/--no-strip-prefix",
    "--strip_prefix",
    "strip_prefix",
    default=False,
    show_default=True,
    help="Write the --node-prefix nodes under their names without the prefix.",
)
def export(
    model_paths: tuple[Path, ...],
    output: Optional[Path],
    fmt: str,
    opset_version: int,
    input_dtype: str,
    network_only: bool,
    max_instances: int,
    max_batch_size: int,
    opt_batch_size: Optional[int],
    input_scale: Optional[float],
    input_height: Optional[int],
    input_width: Optional[int],
    crop_size: Optional[int],
    max_peaks_per_node: int,
    n_line_points: int,
    max_edge_length_ratio: float,
    dist_penalty_weight: float,
    device: str,
    precision: str,
    strongly_typed: bool,
    force_precision: bool,
    fp32_attention_logits: bool,
    peak_threshold: float,
    workspace_size_gb: Optional[float],
    verify: bool,
    nodes: tuple = (),
    node_prefix: Optional[str] = None,
    strip_prefix: bool = False,
) -> None:
    """Export trained models to ONNX/TensorRT formats."""
    import torch

    from sleap_nn.export.exporters import export_to_onnx, export_to_tensorrt
    from sleap_nn.export.exporters.tensorrt_exporter import BACKBONE_SCOPE
    from sleap_nn.export.exporters.onnx_exporter import has_rope
    from sleap_nn.export.metadata import (
        build_base_metadata,
        embed_metadata_in_onnx,
        hash_file,
    )
    from sleap_nn.export.utils import (
        load_training_config,
        warn_on_tiled_export,
        resolve_anchor_part,
        resolve_backbone_source,
        resolve_centroid_method,
        resolve_backbone_type,
        resolve_background_fill,
        resolve_burn_in,
        resolve_class_maps_output_stride,
        resolve_class_names,
        resolve_crop_centering,
        resolve_crop_size,
        resolve_edge_inds,
        resolve_embedding_dim,
        resolve_embedding_input_channels,
        resolve_input_channels,
        resolve_input_scale,
        resolve_input_shape,
        resolve_max_hw,
        resolve_model_type,
        resolve_n_classes,
        resolve_node_names,
        resolve_normalize,
        resolve_output_stride,
        resolve_pafs_output_stride,
    )
    from sleap_nn.export.wrappers import (
        BBoxONNXWrapper,
        BottomUpMultiClassONNXWrapper,
        BottomUpONNXWrapper,
        CenteredInstanceONNXWrapper,
        CentroidONNXWrapper,
        EmbeddingONNXWrapper,
        SingleInstanceONNXWrapper,
        TopDownMultiClassCombinedONNXWrapper,
        TopDownMultiClassONNXWrapper,
        TopDownONNXWrapper,
    )

    fmt = fmt.lower()
    if opt_batch_size is not None and fmt == "onnx":
        click.echo(
            "Note: --opt-batch-size only shapes TensorRT engines; it is ignored for "
            "--format onnx (recorded in the export metadata)."
        )
    if opt_batch_size is not None and not 1 <= opt_batch_size <= max_batch_size:
        raise click.ClickException(
            f"--opt-batch-size {opt_batch_size} must be between 1 and "
            f"--max-batch-size ({max_batch_size})."
        )

    if fp32_attention_logits:
        if fmt not in ("tensorrt", "both"):
            raise click.ClickException(
                "--fp32-attention-logits shapes TensorRT engines: use --format "
                "tensorrt or both."
            )
        if precision.lower() != "fp16":
            raise click.ClickException(
                f"--fp32-attention-logits applies to --precision fp16, not "
                f"{precision}: bf16 already accumulates in fp32, and fp32 / tf32 keep "
                "everything in fp32."
            )
        try:
            import tensorrt as _trt
        except ImportError as e:
            raise click.ClickException(
                f"--fp32-attention-logits builds TensorRT engines: TensorRT is not "
                f"installed ({e})."
            )
        from sleap_nn.export.exporters.tensorrt_exporter import _uses_strong_typing

        if not strongly_typed and not _uses_strong_typing(_trt):
            raise click.ClickException(
                "--fp32-attention-logits needs the strongly typed build: add "
                "--strongly-typed on TensorRT 10 (TensorRT 11 builds that way always)."
            )
    fp32_islands = "attention_logits" if fp32_attention_logits else None
    # With the switch, the fp16 warnings below wait for the build: they are dropped
    # only if the rewrite kept logits products and left no softmax unmatched.
    deferred: list = []

    trt_workspace_kwargs = (
        {"workspace_size": int(workspace_size_gb * (1 << 30))}
        if workspace_size_gb is not None
        else {}
    )

    if not model_paths:
        raise click.ClickException("Provide at least one model path to export.")

    model_paths = list(model_paths)
    cfgs = [load_training_config(path) for path in model_paths]
    for cfg in cfgs:
        warn_on_tiled_export(cfg)
    model_types = [resolve_model_type(cfg) for cfg in cfgs]
    backbone_types = [resolve_backbone_type(cfg) for cfg in cfgs]
    # Checkpoints load with their own (unfiltered) configs; `cfgs` describe the
    # exported network, i.e. after the node filter.
    load_cfgs = list(cfgs)
    node_idx = _resolve_export_node_filter(
        cfgs, model_types, nodes, node_prefix, strip_prefix
    )

    # A standalone centroid model is exported from a SINGLE directory; passing two
    # centroid dirs is almost always a mistake (the user likely meant centroid +
    # centered_instance for a top-down bundle). Catch it with a clear message
    # rather than letting it fall through to the generic combination error.
    if len(model_paths) == 2 and all(mt == "centroid" for mt in model_types):
        raise click.ClickException(
            "Received two centroid model directories. A standalone centroid model "
            "is exported from a single directory; a top-down bundle pairs a centroid "
            "directory with a centered_instance (or multi_class_topdown) directory. "
            "Pass one centroid directory for a standalone centroid export."
        )

    if len(model_paths) >= 2 and {"bbox_centernet", "bbox_ssd", "bbox_detr"} & set(
        model_types
    ):
        raise click.ClickException(
            "A bounding-box detector + top-down model pair cannot be exported as one "
            "bundle yet. Export each directory on its own (the detector exports its "
            "network; the centered-instance model exports a network that takes "
            "crops), or run the pair from its checkpoints with `sleap-nn predict "
            "--model_paths <bbox_dir> --model_paths <centered_instance_dir>`."
        )

    from sleap_nn.data.instance_cropping import BBoxCropSpec

    box_scaled = [p for p, c in zip(model_paths, cfgs) if BBoxCropSpec.from_config(c)]
    if box_scaled and len(model_paths) >= 2:
        raise click.ClickException(
            f"{box_scaled[0]} crops each instance's bounding box "
            "(preprocessing.crop_mode='bbox'); combined top-down bundles cut fixed "
            "windows around centroids, so it cannot be exported in one. Export it "
            "alone (the network takes crops) or run it with a box detector via "
            "`sleap-nn predict`."
        )
    if box_scaled:
        logger.warning(
            f"{box_scaled[0]} is a box-scaled centered-instance model "
            "(preprocessing.crop_mode='bbox'): the exported network takes crops "
            "cut by its crop rule, which is recorded in the exported "
            "training_config (data_config.preprocessing.crop_*)."
        )

    if (input_dtype not in ("auto", "uint8") or network_only) and len(model_paths) > 1:
        raise click.ClickException(
            "--input-dtype float32 and --network-only export one model directory "
            "at a time (run `sleap-nn predict` with one -m per exported stage)."
        )

    for c, mt, bt in zip(cfgs, model_types, backbone_types):  # bundles too
        if fp32_attention_logits:
            deferred.append(
                lambda c=c, mt=mt, bt=bt: _warn_fp16_vit_pose(c, mt, bt, fmt, precision)
            )
        else:
            _warn_fp16_vit_pose(c, mt, bt, fmt, precision)

    if len(model_paths) == 1:
        model_path = model_paths[0]
        cfg = cfgs[0]
        model_type = model_types[0]
        backbone_type = backbone_types[0]

        if model_type not in (
            "centroid",
            "centered_instance",
            "bottomup",
            "single_instance",
            "multi_class_topdown",
            "multi_class_bottomup",
            "embedding",
            "bbox_centernet",
            "bbox_ssd",
            "bbox_detr",
        ):
            raise click.ClickException(
                f"Model type '{model_type}' is not supported for export yet."
            )

        ckpt_path = model_path / "best.ckpt"
        if not ckpt_path.exists():
            raise click.ClickException(f"Checkpoint not found: {ckpt_path}")

        lightning_model = _load_lightning_model(
            model_type=model_type,
            backbone_type=backbone_type,
            cfg=load_cfgs[0],
            ckpt_path=ckpt_path,
            device=device,
        )
        _slice_export_head(lightning_model.model, model_type, node_idx.get(0))

        torch_model = lightning_model.model
        torch_model.eval()
        torch_model.to(device)

        export_dir = output or (model_path / "exported")
        export_dir.mkdir(parents=True, exist_ok=True)

        resolved_scale = (
            input_scale if input_scale is not None else resolve_input_scale(cfg)
        )
        output_stride = resolve_output_stride(cfg, model_type)
        resolved_crop_size = (
            (crop_size, crop_size) if crop_size is not None else resolve_crop_size(cfg)
        )
        metadata_max_instances = None
        metadata_max_peaks = None
        metadata_n_classes = None
        metadata_class_names = None
        metadata_embedding_dim = None
        metadata_normalize = None
        metadata_backbone_source = None
        metadata_burn_in = None
        metadata_background_fill = None
        metadata_crop_centering = None
        metadata_max_hw = (None, None)
        metadata_normalization = "0_to_1"

        if model_type == "centroid":
            wrapper = CentroidONNXWrapper(
                torch_model,
                max_instances=max_instances,
                output_stride=output_stride,
                input_scale=resolved_scale,
                peak_threshold=peak_threshold,
            )
            output_names = ["centroids", "centroid_vals", "instance_valid"]
            metadata_max_instances = max_instances
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
        elif model_type == "centered_instance" and network_only:
            from sleap_nn.export.wrappers.network import NetworkONNXWrapper

            wrapper = NetworkONNXWrapper(
                torch_model, head_names=["CenteredInstanceConfmapsHead"]
            )
            output_names = ["CenteredInstanceConfmapsHead"]
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
        elif model_type == "centered_instance":
            wrapper = CenteredInstanceONNXWrapper(
                torch_model,
                output_stride=output_stride,
                input_scale=resolved_scale,
                peak_threshold=peak_threshold,
            )
            output_names = ["peaks", "peak_vals"]
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
        elif model_type == "bottomup":
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
            pafs_output_stride = resolve_pafs_output_stride(cfg)
            wrapper = BottomUpONNXWrapper(
                torch_model,
                skeleton_edges=edge_inds,
                n_nodes=len(node_names),
                max_peaks_per_node=max_peaks_per_node,
                n_line_points=n_line_points,
                cms_output_stride=output_stride,
                pafs_output_stride=pafs_output_stride,
                max_edge_length_ratio=max_edge_length_ratio,
                dist_penalty_weight=dist_penalty_weight,
                input_scale=resolved_scale,
                peak_threshold=peak_threshold,
            )
            output_names = [
                "peaks",
                "peak_vals",
                "peak_mask",
                "line_scores",
                "candidate_mask",
            ]
            metadata_max_peaks = max_peaks_per_node
        elif model_type == "single_instance":
            wrapper = SingleInstanceONNXWrapper(
                torch_model,
                output_stride=output_stride,
                input_scale=resolved_scale,
                peak_threshold=peak_threshold,
            )
            output_names = ["peaks", "peak_vals"]
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
        elif model_type == "multi_class_topdown":
            n_classes = resolve_n_classes(cfg, model_type)
            class_names = resolve_class_names(cfg, model_type)
            wrapper = TopDownMultiClassONNXWrapper(
                torch_model,
                output_stride=output_stride,
                input_scale=resolved_scale,
                n_classes=n_classes,
                peak_threshold=peak_threshold,
            )
            output_names = ["peaks", "peak_vals", "class_logits"]
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
            metadata_n_classes = n_classes
            metadata_class_names = class_names
        elif model_type == "multi_class_bottomup":
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
            n_classes = resolve_n_classes(cfg, model_type)
            class_names = resolve_class_names(cfg, model_type)
            class_maps_output_stride = resolve_class_maps_output_stride(cfg)
            wrapper = BottomUpMultiClassONNXWrapper(
                torch_model,
                n_nodes=len(node_names),
                n_classes=n_classes,
                max_peaks_per_node=max_peaks_per_node,
                cms_output_stride=output_stride,
                class_maps_output_stride=class_maps_output_stride,
                input_scale=resolved_scale,
                peak_threshold=peak_threshold,
            )
            output_names = ["peaks", "peak_vals", "peak_mask", "class_probs"]
            metadata_max_peaks = max_peaks_per_node
            metadata_n_classes = n_classes
            metadata_class_names = class_names
        elif model_type == "embedding":
            # Re-ID head: crop -> appearance vector. Simplest wrapper (single output,
            # no peak finding, no skeleton). Input is the grayscale crop the embedder
            # standardizes per-crop (the wrapper does not /255).
            wrapper = EmbeddingONNXWrapper(
                torch_model, normalize=resolve_normalize(cfg)
            )
            output_names = ["embedding"]
            # No skeleton semantics for an appearance model.
            node_names = []
            edge_inds = []
            metadata_embedding_dim = resolve_embedding_dim(cfg)
            metadata_normalize = resolve_normalize(cfg)
            metadata_backbone_source = resolve_backbone_source(cfg)
            metadata_burn_in = resolve_burn_in(cfg)
            metadata_background_fill = resolve_background_fill(cfg)
            metadata_crop_centering = resolve_crop_centering(cfg)
            metadata_max_hw = resolve_max_hw(cfg)
            metadata_normalization = "per_crop_standardize"
            trained_crop = resolve_crop_size(cfg)
            if crop_size is not None and trained_crop != (crop_size, crop_size):
                # Native inference always crops at the TRAINED size (it rejects
                # overriding it); an export at another size feeds the embedder
                # crops it was never fitted on.
                logger.warning(
                    f"--crop-size {crop_size} does not match the crop size this "
                    f"embedding model was trained on ({trained_crop}). The exported "
                    "graph will embed crops of a different size -- the animal "
                    "appears at a different scale than in training -- and its "
                    "vectors will not match native `sleap-nn predict` inference. "
                    "Omit --crop-size to export at the trained size."
                )
            if metadata_burn_in:
                # The single-input ONNX graph standardizes over the WHOLE crop; a
                # burn_in model's native inference standardizes over the foreground
                # (mask) only and fills the background. The exported embeddings will
                # therefore DIVERGE from native masked inference. Recorded in metadata.
                logger.warning(
                    "Exporting a mask-burn-in embedding model "
                    f"(background_fill='{metadata_background_fill}'): the ONNX graph "
                    "does a MASKLESS whole-crop standardize and cannot reproduce the "
                    "masked (foreground-only) standardize used at training/native "
                    "inference, so exported embeddings will diverge. Use the native "
                    "`sleap-nn predict ... --save_embeddings slp` path for exact "
                    "parity, or train/export a burn_in=False model for a faithful "
                    "single-input ONNX embedder."
                )
        elif model_type in ("bbox_centernet", "bbox_ssd", "bbox_detr"):
            # Detectors export the network only (decoder-ready head outputs);
            # `ExportedBBoxLayer` decodes with the same BBoxDecoder as the
            # checkpoint path, rebuilt from the copied training config.
            from sleap_nn.data.bboxes import bbox_head_categories
            from sleap_nn.export.wrappers.bbox import BBOX_EXPORT_OUTPUTS

            # The graph takes model-input-sized frames; `ExportedBBoxLayer` applies
            # `input_scale` and the stride padding on the host (the metadata keeps
            # the scale for it).
            wrapper = BBoxONNXWrapper(torch_model, model_type=model_type)
            output_names = BBOX_EXPORT_OUTPUTS[model_type]
            node_names = []
            edge_inds = []
            metadata_class_names = bbox_head_categories(
                model_type, cfg.model_config.head_configs
            )
            metadata_n_classes = len(metadata_class_names)
            metadata_max_instances = max_instances
            metadata_max_hw = resolve_max_hw(cfg)
            if input_height is not None or input_width is not None:
                # The graph is traced at --input-height / --input-width (below):
                # the host must fit frames to that size, not the training size.
                metadata_max_hw = (
                    input_height if input_height is not None else metadata_max_hw[0],
                    input_width if input_width is not None else metadata_max_hw[1],
                )
        else:
            raise click.ClickException(
                f"Model type '{model_type}' is not supported for export yet."
            )

        wrapper.eval()
        wrapper.to(device)

        input_shape = resolve_input_shape(
            cfg, input_height=input_height, input_width=input_width
        )
        input_channels = resolve_input_channels(cfg)
        if model_type == "embedding":
            # The embedder consumes a fixed-size grayscale CROP, not a full frame:
            # size the export input from the crop size + the (grayscale) data
            # channels. A 3ch ImageNet backbone repeats gray->3ch internally.
            if resolved_crop_size is None:
                raise click.ClickException(
                    "Embedding export requires a crop size. Provide --crop-size or "
                    "set data_config.preprocessing.crop_size."
                )
            crop_h, crop_w = resolved_crop_size
            input_channels = resolve_embedding_input_channels(cfg)
            input_shape = (1, input_channels, crop_h, crop_w)
        elif (
            model_type in ("centered_instance", "multi_class_topdown")
            and resolved_crop_size is not None
            and input_height is None
            and input_width is None
        ):
            # These wrappers take crops: trace at the crop size, the only size the
            # model sees -- and the size a RoPE backbone's static-H/W export fixes.
            input_shape = (input_shape[0], input_shape[1], *resolved_crop_size)
        if model_type in ("bbox_centernet", "bbox_ssd", "bbox_detr"):
            # Trace on a model-input-sized frame (scaled, stride-padded). A
            # `bbox_detr` model's frames are letterboxed / squashed to
            # max_height x max_width on the host first, so that is the size here;
            # a RoPE encoder (DINOv3) exports at exactly that size (static H/W).
            max_stride = int(
                cfg.model_config.backbone_config[backbone_type]["max_stride"]
            )
            _, c, h, w = input_shape
            h = math.ceil(h * resolved_scale / max_stride) * max_stride
            w = math.ceil(w * resolved_scale / max_stride) * max_stride
            input_shape = (1, c, h, w)
        model_out_path = export_dir / "model.onnx"
        is_detector = model_type in ("bbox_centernet", "bbox_ssd", "bbox_detr")
        if input_dtype == "auto":
            # A new export type, so its default may match checkpoint inference;
            # every other type keeps the uint8 input it always had.
            input_dtype = "float32" if model_type == "bbox_detr" else "uint8"
        if network_only and model_type != "centered_instance":
            raise click.ClickException(
                "--network-only applies to centered_instance models (box detectors "
                "always export their network only)."
            )
        if input_dtype != "uint8" and not (is_detector or network_only):
            raise click.ClickException(
                "--input-dtype float32 applies to box detectors and --network-only "
                "exports only (their inputs are prepared on the host); other models "
                "take uint8."
            )
        graph_dtype = torch.float32 if input_dtype == "float32" else torch.uint8
        if (
            model_type == "bbox_detr"
            and fmt in ("tensorrt", "both")
            and precision.lower() == "fp16"
        ):
            _detr_note = lambda: click.echo(  # noqa: E731
                "Warning: a bbox_detr detector can lose boxes in fp16 (its activations "
                "can overflow fp16's range); export with --precision bf16 or fp32 "
                "unless fp16 is checked on your data."
            )
            if fp32_attention_logits:
                deferred.append(_detr_note)
            else:
                _detr_note()
        if network_only:
            # The host scales each crop by preprocessing.scale and pads it to the
            # backbone's max_stride before the network: trace (and profile) at that
            # size, the only one the graph will see.
            if any(
                v is not None
                for v in (input_scale, crop_size, input_height, input_width)
            ):
                raise click.ClickException(
                    "--network-only exports the network at the size the checkpoint "
                    "pipeline feeds it (the trained crop size x the trained scale, "
                    "padded to the backbone's stride): --input-scale, --crop-size, "
                    "--input-height and --input-width do not apply."
                )
            if resolved_crop_size is None:
                raise click.ClickException(
                    "--network-only needs the model's crop size "
                    "(data_config.preprocessing.crop_size)."
                )
            ns = int(cfg.model_config.backbone_config[backbone_type]["max_stride"])
            nh, nw = _network_only_hw(resolved_crop_size, resolved_scale, ns)
            input_shape = (input_shape[0], input_shape[1], nh, nw)

        export_to_onnx(
            wrapper,
            model_out_path,
            input_shape=input_shape,
            input_dtype=graph_dtype,
            opset_version=opset_version,
            output_names=output_names,
            # A bbox_detr graph's two-stage proposals and decoder grid are fixed at
            # trace time, and the host always fits frames to max_height x max_width:
            # a static height / width is the only correct graph.
            static_hw=True if model_type == "bbox_detr" else None,
            verify=verify,
        )

        cfg_out = cfg if node_idx else None
        if is_detector and (input_height is not None or input_width is not None):
            # The predictors size frames from the copied training config: it must
            # carry the size the graph was traced at, as the metadata does.
            import copy

            cfg_out = copy.deepcopy(cfg)
            cfg_out.data_config.preprocessing.max_height = metadata_max_hw[0]
            cfg_out.data_config.preprocessing.max_width = metadata_max_hw[1]
        training_config_path = _copy_training_config(
            model_path, export_dir, None, cfg=cfg_out
        )
        if training_config_path is not None:
            training_config_hash = hash_file(training_config_path)
            training_config_text = training_config_path.read_text()
        else:
            training_config_hash = ""
            training_config_text = None

        metadata = build_base_metadata(
            export_format="onnx",
            model_type=model_type,
            model_name=model_path.name,
            checkpoint_path=str(ckpt_path),
            backbone=backbone_type,
            n_nodes=len(node_names),
            n_edges=len(edge_inds),
            node_names=node_names,
            edge_inds=edge_inds,
            input_scale=resolved_scale,
            input_channels=input_channels,
            output_stride=output_stride,
            crop_size=resolved_crop_size,
            max_instances=metadata_max_instances,
            max_peaks_per_node=metadata_max_peaks,
            max_batch_size=max_batch_size,
            precision="fp32",
            training_config_hash=training_config_hash,
            training_config_embedded=training_config_text is not None,
            input_dtype=input_dtype,
            normalization=metadata_normalization,
            n_classes=metadata_n_classes,
            class_names=metadata_class_names,
            peak_threshold=peak_threshold,
            anchor_part=resolve_anchor_part(cfg, model_type),
            embedding_dim=metadata_embedding_dim,
            normalize=metadata_normalize,
            backbone_source=metadata_backbone_source,
            burn_in=metadata_burn_in,
            background_fill=metadata_background_fill,
            centroid_method=resolve_centroid_method(cfg, model_type),
            crop_centering=metadata_crop_centering,
            max_height=metadata_max_hw[0],
            max_width=metadata_max_hw[1],
        )

        metadata.save(export_dir / "export_metadata.json")
        if network_only:
            _mark_network_only(export_dir / "export_metadata.json", output_names)
        if opt_batch_size is not None:
            _record_build(
                export_dir / "export_metadata.json",
                {
                    "opt_batch_size": opt_batch_size,
                    "opt_batch_size_applies_to": "tensorrt",
                },
            )

        if training_config_text is not None:
            try:
                embed_metadata_in_onnx(model_out_path, metadata, training_config_text)
            except ImportError:
                pass

        # Export to TensorRT if requested
        if fmt in ("tensorrt", "both"):
            trt_out_path = export_dir / "model.trt"
            B, C, H, W = input_shape

            # For centered_instance, single_instance, and embedding models, use the
            # crop size for TensorRT shape profiles since inference uses cropped inputs
            if network_only:
                # Exactly the size the host sends (traced above).
                trt_input_shape = input_shape
                trt_min_shape = (1, C, H, W)
                trt_opt_shape = (opt_batch_size or 1, C, H, W)
                trt_max_shape = (max_batch_size, C, H, W)
            elif model_type in ("centered_instance", "single_instance", "embedding"):
                if resolved_crop_size is not None:
                    crop_h, crop_w = resolved_crop_size
                    trt_input_shape = (1, C, crop_h, crop_w)
                    # Use crop size for min/opt, allow flexibility for max
                    trt_min_shape = (1, C, crop_h, crop_w)
                    trt_opt_shape = (opt_batch_size or 1, C, crop_h, crop_w)
                    trt_max_shape = (max_batch_size, C, crop_h * 2, crop_w * 2)
                else:
                    trt_input_shape = input_shape
                    trt_min_shape = None
                    trt_opt_shape = None
                    trt_max_shape = (max_batch_size, C, H * 2, W * 2)
            else:
                trt_input_shape = input_shape
                trt_min_shape = None
                trt_opt_shape = None
                trt_max_shape = (max_batch_size, C, H * 2, W * 2)

            if opt_batch_size is not None and trt_opt_shape is None:
                trt_opt_shape = (opt_batch_size,) + tuple(trt_input_shape[1:])
            # Graphs that compute coordinates keep every layer outside the backbone
            # in fp32 under bf16; feature-map graphs (network-only confmaps,
            # embeddings) keep only their transposed convolutions in fp32.
            coords_in_graph = not network_only and model_type != "embedding"
            build_info = {}
            try:
                export_to_tensorrt(
                    wrapper,
                    trt_out_path,
                    input_shape=trt_input_shape,
                    input_dtype=graph_dtype,
                    precision=precision,
                    min_shape=trt_min_shape,
                    opt_shape=trt_opt_shape,
                    max_shape=trt_max_shape,
                    **trt_workspace_kwargs,
                    verbose=True,
                    _bf16_fp32_outside=BACKBONE_SCOPE if coords_in_graph else None,
                    _build_info=build_info,
                    strongly_typed=strongly_typed,
                    force_precision=force_precision,
                    fp32_islands=fp32_islands,
                )
            except BaseException:
                # A failed build still prints the warnings held back above.
                _emit_deferred_warnings(deferred, {})
                raise
            # Update metadata for TensorRT
            trt_metadata = build_base_metadata(
                export_format="tensorrt",
                model_type=model_type,
                model_name=model_path.name,
                checkpoint_path=str(ckpt_path),
                backbone=backbone_type,
                n_nodes=len(node_names),
                n_edges=len(edge_inds),
                node_names=node_names,
                edge_inds=edge_inds,
                input_scale=resolved_scale,
                input_channels=input_channels,
                output_stride=output_stride,
                crop_size=resolved_crop_size,
                max_instances=metadata_max_instances,
                max_peaks_per_node=metadata_max_peaks,
                max_batch_size=max_batch_size,
                precision=precision,
                training_config_hash=training_config_hash,
                training_config_embedded=training_config_text is not None,
                input_dtype=input_dtype,
                normalization=metadata_normalization,
                n_classes=metadata_n_classes,
                class_names=metadata_class_names,
                peak_threshold=peak_threshold,
                anchor_part=resolve_anchor_part(cfg, model_type),
                embedding_dim=metadata_embedding_dim,
                normalize=metadata_normalize,
                backbone_source=metadata_backbone_source,
                burn_in=metadata_burn_in,
                background_fill=metadata_background_fill,
                centroid_method=resolve_centroid_method(cfg, model_type),
                crop_centering=metadata_crop_centering,
                max_height=metadata_max_hw[0],
                max_width=metadata_max_hw[1],
            )
            trt_metadata.save(export_dir / "model.trt.metadata.json")
            if network_only:
                _mark_network_only(export_dir / "model.trt.metadata.json", output_names)
            _record_build(export_dir / "model.trt.metadata.json", build_info)
            _emit_deferred_warnings(deferred, build_info)
        return

    if len(model_paths) == 2 and set(model_types) == {
        "centroid",
        "centered_instance",
    }:
        centroid_idx = model_types.index("centroid")
        instance_idx = model_types.index("centered_instance")

        centroid_path = model_paths[centroid_idx]
        instance_path = model_paths[instance_idx]
        centroid_cfg = cfgs[centroid_idx]
        instance_cfg = cfgs[instance_idx]
        centroid_backbone = backbone_types[centroid_idx]
        instance_backbone = backbone_types[instance_idx]

        centroid_ckpt = centroid_path / "best.ckpt"
        instance_ckpt = instance_path / "best.ckpt"
        if not centroid_ckpt.exists():
            raise click.ClickException(f"Checkpoint not found: {centroid_ckpt}")
        if not instance_ckpt.exists():
            raise click.ClickException(f"Checkpoint not found: {instance_ckpt}")

        centroid_model = _load_lightning_model(
            model_type="centroid",
            backbone_type=centroid_backbone,
            cfg=centroid_cfg,
            ckpt_path=centroid_ckpt,
            device=device,
        ).model
        instance_model = _load_lightning_model(
            model_type="centered_instance",
            backbone_type=instance_backbone,
            cfg=load_cfgs[instance_idx],
            ckpt_path=instance_ckpt,
            device=device,
        ).model
        _slice_export_head(
            instance_model, "centered_instance", node_idx.get(instance_idx)
        )

        centroid_model.eval()
        instance_model.eval()
        centroid_model.to(device)
        instance_model.to(device)

        export_dir = output or (centroid_path / "exported_topdown")
        export_dir.mkdir(parents=True, exist_ok=True)

        centroid_scale = (
            input_scale
            if input_scale is not None
            else resolve_input_scale(centroid_cfg)
        )
        instance_scale = (
            input_scale
            if input_scale is not None
            else resolve_input_scale(instance_cfg)
        )
        centroid_stride = resolve_output_stride(centroid_cfg, "centroid")
        instance_stride = resolve_output_stride(instance_cfg, "centered_instance")

        resolved_crop = resolve_crop_size(instance_cfg)
        if crop_size is not None:
            resolved_crop = (crop_size, crop_size)
        if resolved_crop is None:
            raise click.ClickException(
                "Top-down export requires crop_size. Provide --crop-size or ensure "
                "data_config.preprocessing.crop_size is set."
            )

        node_names = resolve_node_names(instance_cfg, "centered_instance")
        edge_inds = resolve_edge_inds(instance_cfg, node_names)

        wrapper = TopDownONNXWrapper(
            centroid_model=centroid_model,
            instance_model=instance_model,
            max_instances=max_instances,
            crop_size=resolved_crop,
            centroid_output_stride=centroid_stride,
            instance_output_stride=instance_stride,
            centroid_input_scale=centroid_scale,
            instance_input_scale=instance_scale,
            n_nodes=len(node_names),
            centroid_peak_threshold=peak_threshold,
            instance_peak_threshold=peak_threshold,
        )
        wrapper.eval()
        wrapper.to(device)

        input_shape = resolve_input_shape(
            centroid_cfg, input_height=input_height, input_width=input_width
        )
        model_out_path = export_dir / "model.onnx"

        export_to_onnx(
            wrapper,
            model_out_path,
            input_shape=input_shape,
            input_dtype=torch.uint8,
            opset_version=opset_version,
            output_names=[
                "centroids",
                "centroid_vals",
                "peaks",
                "peak_vals",
                "instance_valid",
            ],
            verify=verify,
            # Only the centroid model sees the frame size; the instance model
            # always receives fixed-size crops, so its RoPE grid never changes.
            static_hw=has_rope(centroid_model),
        )

        centroid_cfg_path = _copy_training_config(centroid_path, export_dir, "centroid")
        instance_cfg_path = _copy_training_config(
            instance_path,
            export_dir,
            "centered_instance",
            cfg=instance_cfg if node_idx else None,
        )
        config_payload = {}
        config_hashes = []
        if centroid_cfg_path is not None:
            config_payload["centroid"] = centroid_cfg_path.read_text()
            config_hashes.append(f"centroid:{hash_file(centroid_cfg_path)}")
        if instance_cfg_path is not None:
            config_payload["centered_instance"] = instance_cfg_path.read_text()
            config_hashes.append(f"centered_instance:{hash_file(instance_cfg_path)}")

        training_config_hash = ";".join(config_hashes) if config_hashes else ""
        training_config_text = json.dumps(config_payload) if config_payload else None

        metadata = build_base_metadata(
            export_format="onnx",
            model_type="topdown",
            model_name=f"{centroid_path.name}+{instance_path.name}",
            checkpoint_path=(
                f"centroid:{centroid_ckpt};centered_instance:{instance_ckpt}"
            ),
            backbone=(
                f"centroid:{centroid_backbone};centered_instance:{instance_backbone}"
            ),
            n_nodes=len(node_names),
            n_edges=len(edge_inds),
            node_names=node_names,
            edge_inds=edge_inds,
            input_scale=centroid_scale,
            input_channels=resolve_input_channels(centroid_cfg),
            output_stride=instance_stride,
            crop_size=resolved_crop,
            max_instances=max_instances,
            max_batch_size=max_batch_size,
            precision="fp32",
            training_config_hash=training_config_hash,
            training_config_embedded=training_config_text is not None,
            input_dtype="uint8",
            normalization="0_to_1",
            peak_threshold=peak_threshold,
            anchor_part=resolve_anchor_part(centroid_cfg, "centroid"),
            centroid_method=resolve_centroid_method(centroid_cfg, "centroid"),
        )

        metadata.save(export_dir / "export_metadata.json")

        if training_config_text is not None:
            try:
                embed_metadata_in_onnx(model_out_path, metadata, training_config_text)
            except ImportError:
                pass

        # Export to TensorRT if requested
        if fmt in ("tensorrt", "both"):
            trt_out_path = export_dir / "model.trt"
            B, C, H, W = input_shape
            build_info = {}
            try:
                export_to_tensorrt(
                    wrapper,
                    trt_out_path,
                    input_shape=input_shape,
                    input_dtype=torch.uint8,
                    precision=precision,
                    max_shape=(max_batch_size, C, H * 2, W * 2),
                    opt_shape=(
                        (opt_batch_size, C, H, W)
                        if opt_batch_size is not None
                        else None
                    ),
                    **trt_workspace_kwargs,
                    verbose=True,
                    # As for the ONNX graph: only the centroid model sees the frame.
                    static_hw=has_rope(centroid_model),
                    _bf16_fp32_outside=BACKBONE_SCOPE,
                    _build_info=build_info,
                    strongly_typed=strongly_typed,
                    force_precision=force_precision,
                    fp32_islands=fp32_islands,
                )
            except BaseException:
                # A failed build still prints the warnings held back above.
                _emit_deferred_warnings(deferred, {})
                raise
            # Update metadata for TensorRT
            trt_metadata = build_base_metadata(
                export_format="tensorrt",
                model_type="topdown",
                model_name=f"{centroid_path.name}+{instance_path.name}",
                checkpoint_path=(
                    f"centroid:{centroid_ckpt};centered_instance:{instance_ckpt}"
                ),
                backbone=(
                    f"centroid:{centroid_backbone};centered_instance:{instance_backbone}"
                ),
                n_nodes=len(node_names),
                n_edges=len(edge_inds),
                node_names=node_names,
                edge_inds=edge_inds,
                input_scale=centroid_scale,
                input_channels=resolve_input_channels(centroid_cfg),
                output_stride=instance_stride,
                crop_size=resolved_crop,
                max_instances=max_instances,
                max_batch_size=max_batch_size,
                precision=precision,
                training_config_hash=training_config_hash,
                training_config_embedded=training_config_text is not None,
                input_dtype="uint8",
                normalization="0_to_1",
                peak_threshold=peak_threshold,
                anchor_part=resolve_anchor_part(centroid_cfg, "centroid"),
                centroid_method=resolve_centroid_method(centroid_cfg, "centroid"),
            )
            trt_metadata.save(export_dir / "model.trt.metadata.json")
            _record_build(export_dir / "model.trt.metadata.json", build_info)
            _emit_deferred_warnings(deferred, build_info)
        return

    # Combined multiclass top-down export (centroid + multi_class_topdown)
    if len(model_paths) == 2 and set(model_types) == {
        "centroid",
        "multi_class_topdown",
    }:
        centroid_idx = model_types.index("centroid")
        instance_idx = model_types.index("multi_class_topdown")

        centroid_path = model_paths[centroid_idx]
        instance_path = model_paths[instance_idx]
        centroid_cfg = cfgs[centroid_idx]
        instance_cfg = cfgs[instance_idx]
        centroid_backbone = backbone_types[centroid_idx]
        instance_backbone = backbone_types[instance_idx]

        centroid_ckpt = centroid_path / "best.ckpt"
        instance_ckpt = instance_path / "best.ckpt"
        if not centroid_ckpt.exists():
            raise click.ClickException(f"Checkpoint not found: {centroid_ckpt}")
        if not instance_ckpt.exists():
            raise click.ClickException(f"Checkpoint not found: {instance_ckpt}")

        centroid_model = _load_lightning_model(
            model_type="centroid",
            backbone_type=centroid_backbone,
            cfg=centroid_cfg,
            ckpt_path=centroid_ckpt,
            device=device,
        ).model
        instance_model = _load_lightning_model(
            model_type="multi_class_topdown",
            backbone_type=instance_backbone,
            cfg=instance_cfg,
            ckpt_path=instance_ckpt,
            device=device,
        ).model

        centroid_model.eval()
        instance_model.eval()
        centroid_model.to(device)
        instance_model.to(device)

        export_dir = output or (centroid_path / "exported_multi_class_topdown")
        export_dir.mkdir(parents=True, exist_ok=True)

        centroid_scale = (
            input_scale
            if input_scale is not None
            else resolve_input_scale(centroid_cfg)
        )
        instance_scale = (
            input_scale
            if input_scale is not None
            else resolve_input_scale(instance_cfg)
        )
        centroid_stride = resolve_output_stride(centroid_cfg, "centroid")
        instance_stride = resolve_output_stride(instance_cfg, "multi_class_topdown")

        resolved_crop = resolve_crop_size(instance_cfg)
        if crop_size is not None:
            resolved_crop = (crop_size, crop_size)
        if resolved_crop is None:
            raise click.ClickException(
                "Multiclass top-down export requires crop_size. Provide --crop-size or "
                "ensure data_config.preprocessing.crop_size is set."
            )

        node_names = resolve_node_names(instance_cfg, "multi_class_topdown")
        edge_inds = resolve_edge_inds(instance_cfg, node_names)
        n_classes = resolve_n_classes(instance_cfg, "multi_class_topdown")
        class_names = resolve_class_names(instance_cfg, "multi_class_topdown")

        wrapper = TopDownMultiClassCombinedONNXWrapper(
            centroid_model=centroid_model,
            instance_model=instance_model,
            max_instances=max_instances,
            crop_size=resolved_crop,
            centroid_output_stride=centroid_stride,
            instance_output_stride=instance_stride,
            centroid_input_scale=centroid_scale,
            instance_input_scale=instance_scale,
            n_nodes=len(node_names),
            n_classes=n_classes,
            centroid_peak_threshold=peak_threshold,
            instance_peak_threshold=peak_threshold,
        )
        wrapper.eval()
        wrapper.to(device)

        input_shape = resolve_input_shape(
            centroid_cfg, input_height=input_height, input_width=input_width
        )
        model_out_path = export_dir / "model.onnx"

        export_to_onnx(
            wrapper,
            model_out_path,
            input_shape=input_shape,
            input_dtype=torch.uint8,
            opset_version=opset_version,
            output_names=[
                "centroids",
                "centroid_vals",
                "peaks",
                "peak_vals",
                "class_logits",
                "instance_valid",
            ],
            verify=verify,
            # As for top-down: only the centroid model sees the frame size.
            static_hw=has_rope(centroid_model),
        )

        centroid_cfg_path = _copy_training_config(centroid_path, export_dir, "centroid")
        instance_cfg_path = _copy_training_config(
            instance_path, export_dir, "multi_class_topdown"
        )
        config_payload = {}
        config_hashes = []
        if centroid_cfg_path is not None:
            config_payload["centroid"] = centroid_cfg_path.read_text()
            config_hashes.append(f"centroid:{hash_file(centroid_cfg_path)}")
        if instance_cfg_path is not None:
            config_payload["multi_class_topdown"] = instance_cfg_path.read_text()
            config_hashes.append(f"multi_class_topdown:{hash_file(instance_cfg_path)}")

        training_config_hash = ";".join(config_hashes) if config_hashes else ""
        training_config_text = json.dumps(config_payload) if config_payload else None

        metadata = build_base_metadata(
            export_format="onnx",
            model_type="multi_class_topdown_combined",
            model_name=f"{centroid_path.name}+{instance_path.name}",
            checkpoint_path=(
                f"centroid:{centroid_ckpt};multi_class_topdown:{instance_ckpt}"
            ),
            backbone=(
                f"centroid:{centroid_backbone};multi_class_topdown:{instance_backbone}"
            ),
            n_nodes=len(node_names),
            n_edges=len(edge_inds),
            node_names=node_names,
            edge_inds=edge_inds,
            input_scale=centroid_scale,
            input_channels=resolve_input_channels(centroid_cfg),
            output_stride=instance_stride,
            crop_size=resolved_crop,
            max_instances=max_instances,
            max_batch_size=max_batch_size,
            training_config_hash=training_config_hash,
            training_config_embedded=training_config_text is not None,
            input_dtype="uint8",
            normalization="0_to_1",
            n_classes=n_classes,
            class_names=class_names,
            peak_threshold=peak_threshold,
            anchor_part=resolve_anchor_part(centroid_cfg, "centroid"),
            centroid_method=resolve_centroid_method(centroid_cfg, "centroid"),
        )
        metadata.save(export_dir / "export_metadata.json")
        click.echo(f"ONNX model exported to: {model_out_path}")
        click.echo(f"Metadata saved to: {export_dir / 'export_metadata.json'}")

        # TensorRT export for combined multiclass top-down
        if fmt in ("tensorrt", "both"):
            trt_out_path = export_dir / "model.trt"
            B, C, H, W = input_shape
            build_info = {}
            try:
                export_to_tensorrt(
                    wrapper,
                    trt_out_path,
                    input_shape=input_shape,
                    input_dtype=torch.uint8,
                    precision=precision,
                    max_shape=(max_batch_size, C, H * 2, W * 2),
                    opt_shape=(
                        (opt_batch_size, C, H, W)
                        if opt_batch_size is not None
                        else None
                    ),
                    **trt_workspace_kwargs,
                    verbose=True,
                    # As for the ONNX graph: only the centroid model sees the frame.
                    static_hw=has_rope(centroid_model),
                    _bf16_fp32_outside=BACKBONE_SCOPE,
                    _build_info=build_info,
                    strongly_typed=strongly_typed,
                    force_precision=force_precision,
                    fp32_islands=fp32_islands,
                )
            except BaseException:
                # A failed build still prints the warnings held back above.
                _emit_deferred_warnings(deferred, {})
                raise
            trt_metadata = build_base_metadata(
                export_format="tensorrt",
                model_type="multi_class_topdown_combined",
                model_name=f"{centroid_path.name}+{instance_path.name}",
                checkpoint_path=(
                    f"centroid:{centroid_ckpt};multi_class_topdown:{instance_ckpt}"
                ),
                backbone=(
                    f"centroid:{centroid_backbone};multi_class_topdown:{instance_backbone}"
                ),
                n_nodes=len(node_names),
                n_edges=len(edge_inds),
                node_names=node_names,
                edge_inds=edge_inds,
                input_scale=centroid_scale,
                input_channels=resolve_input_channels(centroid_cfg),
                output_stride=instance_stride,
                crop_size=resolved_crop,
                max_instances=max_instances,
                max_batch_size=max_batch_size,
                precision=precision,
                training_config_hash=training_config_hash,
                training_config_embedded=training_config_text is not None,
                input_dtype="uint8",
                normalization="0_to_1",
                n_classes=n_classes,
                class_names=class_names,
                peak_threshold=peak_threshold,
                anchor_part=resolve_anchor_part(centroid_cfg, "centroid"),
                centroid_method=resolve_centroid_method(centroid_cfg, "centroid"),
            )
            trt_metadata.save(export_dir / "model.trt.metadata.json")
            _record_build(export_dir / "model.trt.metadata.json", build_info)
            _emit_deferred_warnings(deferred, build_info)
        return

    raise click.ClickException(
        "Provide one model path for centroid/centered-instance/bottom-up export, "
        "or two paths (centroid + centered_instance or centroid + multi_class_topdown) "
        "for combined top-down export."
    )


def _emit_deferred_warnings(deferred: list, build_info: dict) -> None:
    """Print the fp16 warnings held back by ``--fp32-attention-logits``.

    They are dropped only when the rewrite kept logits products in fp32, left no
    softmax without one and found none inside control-flow bodies (``fp32_islands``
    in the build info); otherwise (a failed build included) they are printed as they
    would be without the option.
    """
    rec = build_info.get("fp32_islands") or {}
    if (
        rec.get("applied")
        and not rec.get("softmax_unmatched")
        and not rec.get("softmax_in_subgraphs")
    ):
        return
    for warn in deferred:
        warn()


def _record_build(path: Path, info: dict) -> None:
    """Write what the engine build actually did into its metadata file.

    ``precision_built`` replaces ``precision`` (an fp16 request on a GPU without fast
    fp16 builds fp32); other keys (``bf16_scope``, ``bf16_fp32_layers``,
    ``opt_batch_size`` ...) are added. Nothing is written for an empty ``info``.
    """
    import json

    if not info:
        return
    data = json.loads(path.read_text())
    info = dict(info)
    if "precision_built" in info:
        data["precision"] = info.pop("precision_built")
    data.update(info)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _mark_network_only(path: Path, output_names) -> None:
    """Record a ``--network-only`` export in its metadata file.

    The keys are added to the JSON only for such exports (not as
    ``ExportMetadata`` fields), so every other export's metadata is unchanged.
    """
    import json

    data = json.loads(path.read_text())
    data["network_only"] = True
    data["output_names"] = list(output_names)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _network_only_hw(crop_hw, scale: float, stride: int) -> tuple:
    """The crop size the host sends a network-only graph.

    The host resizes a crop to ``int(crop x scale)`` pixels per side
    (:func:`~sleap_nn.data.resizing.resize_image`) and pads that up to the
    backbone's stride (:func:`~sleap_nn.data.resizing.apply_pad_to_stride`).
    """
    return tuple(math.ceil(int(c * scale) / stride) * stride for c in crop_hw)


def _warn_fp16_vit_pose(cfg, model_type, backbone_type, fmt, precision) -> None:
    """Warn on an fp16 TensorRT export of a keypoint model with a ViT backbone.

    Measured on DINOv3 ViT-B pose (21 points) and tail (4 points) models only:
    17-44% of their keypoints within 0.5 px of fp32 in fp16 engines on TensorRT 10
    and 11; bf16 kept ~96% / ~89% and fp32 100%. Other ViTs (DINOv2 ...) are not
    tested. The export default precision is unchanged.
    """
    if fmt not in ("tensorrt", "both") or str(precision).lower() != "fp16":
        return
    if model_type in ("bbox_centernet", "bbox_ssd", "bbox_detr"):
        return  # box detectors have their own note (TensorRT precision guide)
    try:
        bb = cfg.model_config.backbone_config[backbone_type]
        name = str(bb.get("model_name") or "") if backbone_type == "pretrained" else ""
    except (AttributeError, KeyError, TypeError):
        return
    if "vit" not in name.lower():
        return
    click.echo(
        f"Warning: {model_type} model with a ViT backbone ({name}): fp16 TensorRT "
        "engines of DINOv3 ViT-B pose / tail models kept only 17-44% of keypoints "
        "within 0.5 px of fp32 (TensorRT 10 and 11; other ViTs not tested). Export "
        "with --precision bf16 (a GPU with compute capability >= 8.0) or fp32, or fp16 "
        "with --fp32-attention-logits (strongly typed; tested on DINOv3 only), unless "
        "plain fp16 is checked on your data."
    )


def _copy_training_config(
    model_path: Path, export_dir: Path, label: Optional[str], cfg=None
) -> Optional[Path]:
    """Copy a model's training config into the export dir.

    With ``cfg`` (a node-filtered config, :func:`_resolve_export_node_filter`),
    that config is written as YAML instead of copying the file.
    """
    training_config_path = _training_config_path(model_path)
    if training_config_path is None:
        return None

    suffix = ".yaml" if cfg is not None else training_config_path.suffix
    if label:
        dest_name = f"training_config_{label}{suffix}"
    else:
        dest_name = f"training_config{suffix}"

    dest_path = export_dir / dest_name
    if cfg is not None:
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, dest_path)
    else:
        shutil.copy(training_config_path, dest_path)
    return dest_path


def _resolve_export_node_filter(
    cfgs: list, model_types: list, nodes: tuple, node_prefix, strip_prefix: bool
) -> dict:
    """Apply ``--nodes`` / ``--node-prefix`` to the configs being exported.

    Replaces the target model's entry of ``cfgs`` with its node-filtered config
    (:func:`~sleap_nn.inference.node_filter.filter_training_config`) and returns
    ``{model index: confmap channel indices}`` to slice its head with (empty
    without a filter). The target is a lone ``single_instance`` /
    ``centered_instance`` model or the centered-instance model of a
    centroid + centered_instance bundle.
    """
    from sleap_nn.inference.node_filter import filter_training_config

    names = [n.strip() for arg in nodes for n in arg.split(",") if n.strip()]
    if names and node_prefix is not None:
        raise click.ClickException("Pass --nodes or --node-prefix, not both.")
    node_filter = names or node_prefix
    if not node_filter:
        if strip_prefix:
            raise click.ClickException("--strip-prefix needs --node-prefix.")
        return {}
    if strip_prefix and node_prefix is None:
        raise click.ClickException("--strip-prefix needs --node-prefix.")
    targets = [
        i
        for i, mt in enumerate(model_types)
        if mt in ("single_instance", "centered_instance")
    ]
    if len(targets) != 1:
        raise click.ClickException(
            "--nodes/--node-prefix apply to a single_instance or centered_instance "
            f"model (alone or with its centroid model); got {model_types}."
        )
    i = targets[0]
    try:
        cfgs[i], idx = filter_training_config(
            cfgs[i], model_types[i], node_filter, strip_prefix=strip_prefix or None
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from None
    logger.info(f"Node filter: exporting {len(idx)} confmap channels.")
    return {i: idx}


def _slice_export_head(model, model_type: str, idx) -> None:
    """Slice a loaded model's confmap head to the node filter's channels."""
    if idx is None:
        return
    from sleap_nn.inference.node_filter import NODE_FILTER_HEADS, slice_head_conv

    slice_head_conv(model, NODE_FILTER_HEADS[model_type], idx)


def _training_config_path(model_path: Path) -> Optional[Path]:
    yaml_path = model_path / "training_config.yaml"
    json_path = model_path / "training_config.json"
    if yaml_path.exists():
        return yaml_path
    if json_path.exists():
        return json_path
    return None


def _load_lightning_model(
    *,
    model_type: str,
    backbone_type: str,
    cfg,
    ckpt_path: Path,
    device: str,
):
    from sleap_nn.training.lightning_modules import (
        BBoxCenterNetLightningModule,
        BBoxDetrLightningModule,
        BBoxSSDLightningModule,
        BottomUpLightningModule,
        BottomUpMultiClassLightningModule,
        CentroidLightningModule,
        EmbeddingLightningModule,
        SingleInstanceLightningModule,
        TopDownCenteredInstanceLightningModule,
        TopDownCenteredInstanceMultiClassLightningModule,
    )

    lightning_cls = {
        "centroid": CentroidLightningModule,
        "centered_instance": TopDownCenteredInstanceLightningModule,
        "single_instance": SingleInstanceLightningModule,
        "bottomup": BottomUpLightningModule,
        "multi_class_topdown": TopDownCenteredInstanceMultiClassLightningModule,
        "multi_class_bottomup": BottomUpMultiClassLightningModule,
        "embedding": EmbeddingLightningModule,
        "bbox_centernet": BBoxCenterNetLightningModule,
        "bbox_ssd": BBoxSSDLightningModule,
        "bbox_detr": BBoxDetrLightningModule,
    }.get(model_type)

    if lightning_cls is None:
        raise click.ClickException(f"Unsupported model type: {model_type}")

    return lightning_cls.load_from_checkpoint(
        checkpoint_path=str(ckpt_path),
        model_type=model_type,
        backbone_type=backbone_type,
        backbone_config=cfg.model_config.backbone_config,
        head_configs=cfg.model_config.head_configs,
        pretrained_backbone_weights=cfg.model_config.pretrained_backbone_weights,
        pretrained_head_weights=cfg.model_config.pretrained_head_weights,
        init_weights=cfg.model_config.init_weights,
        lr_scheduler=cfg.trainer_config.lr_scheduler,
        online_mining=cfg.trainer_config.online_hard_keypoint_mining.online_mining,
        hard_to_easy_ratio=cfg.trainer_config.online_hard_keypoint_mining.hard_to_easy_ratio,
        min_hard_keypoints=cfg.trainer_config.online_hard_keypoint_mining.min_hard_keypoints,
        max_hard_keypoints=cfg.trainer_config.online_hard_keypoint_mining.max_hard_keypoints,
        loss_scale=cfg.trainer_config.online_hard_keypoint_mining.loss_scale,
        optimizer=cfg.trainer_config.optimizer_name,
        learning_rate=cfg.trainer_config.optimizer.lr,
        amsgrad=cfg.trainer_config.optimizer.amsgrad,
        map_location=device,
        weights_only=False,
    )
