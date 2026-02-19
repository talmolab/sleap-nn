"""CLI entry points for export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import shutil

import click
import torch

from sleap_nn.export.exporters import export_to_onnx, export_to_tensorrt
from sleap_nn.export.metadata import (
    build_base_metadata,
    embed_metadata_in_onnx,
    hash_file,
)
from sleap_nn.export.utils import (
    load_training_config,
    resolve_backbone_type,
    resolve_class_maps_output_stride,
    resolve_class_names,
    resolve_crop_size,
    resolve_edge_inds,
    resolve_input_channels,
    resolve_input_scale,
    resolve_input_shape,
    resolve_model_type,
    resolve_n_classes,
    resolve_node_names,
    resolve_output_stride,
    resolve_pafs_output_stride,
)
from sleap_nn.export.wrappers import (
    BottomUpMultiClassONNXWrapper,
    BottomUpONNXWrapper,
    CenteredInstanceONNXWrapper,
    CentroidONNXWrapper,
    SingleInstanceONNXWrapper,
    TopDownMultiClassCombinedONNXWrapper,
    TopDownMultiClassONNXWrapper,
    TopDownONNXWrapper,
)
from sleap_nn.training.lightning_modules import (
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
    CentroidLightningModule,
    SingleInstanceLightningModule,
    TopDownCenteredInstanceLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
)


@click.command()
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
@click.option("--max-instances", type=int, default=20, show_default=True)
@click.option("--max-batch-size", type=int, default=8, show_default=True)
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
    type=click.Choice(["fp32", "fp16"], case_sensitive=False),
    default="fp16",
    show_default=True,
    help="TensorRT precision mode.",
)
@click.option("--verify/--no-verify", default=True, show_default=True)
def export(
    model_paths: tuple[Path, ...],
    output: Optional[Path],
    fmt: str,
    opset_version: int,
    max_instances: int,
    max_batch_size: int,
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
    verify: bool,
) -> None:
    """Export trained models to ONNX/TensorRT formats."""
    fmt = fmt.lower()

    if not model_paths:
        raise click.ClickException("Provide at least one model path to export.")

    model_paths = list(model_paths)
    cfgs = [load_training_config(path) for path in model_paths]
    model_types = [resolve_model_type(cfg) for cfg in cfgs]
    backbone_types = [resolve_backbone_type(cfg) for cfg in cfgs]

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
            cfg=cfg,
            ckpt_path=ckpt_path,
            device=device,
        )

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

        if model_type == "centroid":
            wrapper = CentroidONNXWrapper(
                torch_model,
                max_instances=max_instances,
                output_stride=output_stride,
                input_scale=resolved_scale,
            )
            output_names = ["centroids", "centroid_vals", "instance_valid"]
            metadata_max_instances = max_instances
            node_names = resolve_node_names(cfg, model_type)
            edge_inds = resolve_edge_inds(cfg, node_names)
        elif model_type == "centered_instance":
            wrapper = CenteredInstanceONNXWrapper(
                torch_model,
                output_stride=output_stride,
                input_scale=resolved_scale,
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
            )
            output_names = ["peaks", "peak_vals", "peak_mask", "class_probs"]
            metadata_max_peaks = max_peaks_per_node
            metadata_n_classes = n_classes
            metadata_class_names = class_names
        else:
            raise click.ClickException(
                f"Model type '{model_type}' is not supported for export yet."
            )

        wrapper.eval()
        wrapper.to(device)

        input_shape = resolve_input_shape(
            cfg, input_height=input_height, input_width=input_width
        )
        model_out_path = export_dir / "model.onnx"

        export_to_onnx(
            wrapper,
            model_out_path,
            input_shape=input_shape,
            input_dtype=torch.uint8,
            opset_version=opset_version,
            output_names=output_names,
            verify=verify,
        )

        training_config_path = _copy_training_config(model_path, export_dir, None)
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
            input_channels=resolve_input_channels(cfg),
            output_stride=output_stride,
            crop_size=resolved_crop_size,
            max_instances=metadata_max_instances,
            max_peaks_per_node=metadata_max_peaks,
            max_batch_size=max_batch_size,
            precision="fp32",
            training_config_hash=training_config_hash,
            training_config_embedded=training_config_text is not None,
            input_dtype="uint8",
            normalization="0_to_1",
            n_classes=metadata_n_classes,
            class_names=metadata_class_names,
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

            # For centered_instance and single_instance models, use crop size
            # for TensorRT shape profiles since inference uses cropped inputs
            if model_type in ("centered_instance", "single_instance"):
                if resolved_crop_size is not None:
                    crop_h, crop_w = resolved_crop_size
                    trt_input_shape = (1, C, crop_h, crop_w)
                    # Use crop size for min/opt, allow flexibility for max
                    trt_min_shape = (1, C, crop_h, crop_w)
                    trt_opt_shape = (1, C, crop_h, crop_w)
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

            export_to_tensorrt(
                wrapper,
                trt_out_path,
                input_shape=trt_input_shape,
                input_dtype=torch.uint8,
                precision=precision,
                min_shape=trt_min_shape,
                opt_shape=trt_opt_shape,
                max_shape=trt_max_shape,
                verbose=True,
            )
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
                input_channels=resolve_input_channels(cfg),
                output_stride=output_stride,
                crop_size=resolved_crop_size,
                max_instances=metadata_max_instances,
                max_peaks_per_node=metadata_max_peaks,
                max_batch_size=max_batch_size,
                precision=precision,
                training_config_hash=training_config_hash,
                training_config_embedded=training_config_text is not None,
                input_dtype="uint8",
                normalization="0_to_1",
                n_classes=metadata_n_classes,
                class_names=metadata_class_names,
            )
            trt_metadata.save(export_dir / "model.trt.metadata.json")
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
            cfg=instance_cfg,
            ckpt_path=instance_ckpt,
            device=device,
        ).model

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
        )

        centroid_cfg_path = _copy_training_config(centroid_path, export_dir, "centroid")
        instance_cfg_path = _copy_training_config(
            instance_path, export_dir, "centered_instance"
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
            export_to_tensorrt(
                wrapper,
                trt_out_path,
                input_shape=input_shape,
                input_dtype=torch.uint8,
                precision=precision,
                max_shape=(max_batch_size, C, H * 2, W * 2),
                verbose=True,
            )
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
            )
            trt_metadata.save(export_dir / "model.trt.metadata.json")
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
        )
        metadata.save(export_dir / "export_metadata.json")
        click.echo(f"ONNX model exported to: {model_out_path}")
        click.echo(f"Metadata saved to: {export_dir / 'export_metadata.json'}")

        # TensorRT export for combined multiclass top-down
        if fmt in ("tensorrt", "both"):
            trt_out_path = export_dir / "model.trt"
            B, C, H, W = input_shape
            export_to_tensorrt(
                wrapper,
                trt_out_path,
                input_shape=input_shape,
                input_dtype=torch.uint8,
                precision=precision,
                max_shape=(max_batch_size, C, H * 2, W * 2),
                verbose=True,
            )
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
            )
            trt_metadata.save(export_dir / "model.trt.metadata.json")
        return

    raise click.ClickException(
        "Provide one model path for centroid/centered-instance/bottom-up export, "
        "or two paths (centroid + centered_instance or centroid + multi_class_topdown) "
        "for combined top-down export."
    )


@click.command()
@click.argument(
    "export_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.argument(
    "video_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output SLP file path. Default: video_name.predictions.slp",
)
@click.option(
    "--runtime",
    type=click.Choice(["auto", "onnx", "tensorrt"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="Runtime to use for inference.",
)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option("--batch-size", type=int, default=4, show_default=True)
@click.option("--n-frames", type=int, default=None, help="Limit to first N frames.")
@click.option(
    "--max-edge-length-ratio",
    type=float,
    default=0.25,
    show_default=True,
    help="Bottom-up: max edge length as ratio of PAF dimensions.",
)
@click.option(
    "--dist-penalty-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Bottom-up: weight for distance penalty in PAF scoring.",
)
@click.option(
    "--n-points",
    type=int,
    default=10,
    show_default=True,
    help="Bottom-up: number of points to sample along PAF.",
)
@click.option(
    "--min-instance-peaks",
    type=float,
    default=0,
    show_default=True,
    help="Bottom-up: minimum peaks required per instance.",
)
@click.option(
    "--min-line-scores",
    type=float,
    default=0.25,
    show_default=True,
    help="Bottom-up: minimum line score threshold.",
)
@click.option(
    "--peak-conf-threshold",
    type=float,
    default=0.2,
    show_default=True,
    help="Bottom-up: peak confidence threshold for filtering candidates.",
)
@click.option(
    "--max-instances",
    type=int,
    default=None,
    help="Maximum instances to output per frame.",
)
@click.option(
    "--cpu-workers",
    type=int,
    default=0,
    show_default=True,
    help="Number of CPU worker processes for parallel post-processing (bottom-up only). "
    "When > 0, inference and PAF grouping run in a producer-consumer pipeline. "
    "0 = sequential (legacy) mode.",
)
def predict(
    export_dir: Path,
    video_path: Path,
    output: Optional[Path],
    runtime: str,
    device: str,
    batch_size: int,
    n_frames: Optional[int],
    max_edge_length_ratio: float,
    dist_penalty_weight: float,
    n_points: int,
    min_instance_peaks: float,
    min_line_scores: float,
    peak_conf_threshold: float,
    max_instances: Optional[int],
    cpu_workers: int,
) -> None:
    """Run inference on exported models and save predictions to SLP.

    EXPORT_DIR is the directory containing the exported model (model.onnx or model.trt)
    along with export_metadata.json and training_config.yaml.

    VIDEO_PATH is the path to the video file to process.
    """
    import sleap_io as sio

    from sleap_nn.export.inference import predict as _predict

    click.echo(f"Processing video: {video_path}")
    click.echo(f"  Export dir: {export_dir}")
    click.echo(f"  Batch size: {batch_size}")
    if cpu_workers > 0:
        click.echo(f"  Using pipelined inference with {cpu_workers} CPU workers")

    def _progress(processed, total):
        click.echo(f"\r  Processed {processed}/{total} frames...", nl=False)

    try:
        labels, stats = _predict(
            export_dir=export_dir,
            video_path=video_path,
            runtime=runtime,
            device=device,
            batch_size=batch_size,
            n_frames=n_frames,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            n_points=n_points,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
            peak_conf_threshold=peak_conf_threshold,
            max_instances=max_instances,
            cpu_workers=cpu_workers,
            progress_callback=_progress,
        )
    except (FileNotFoundError, ValueError) as e:
        raise click.ClickException(str(e))

    click.echo()  # Newline after progress

    # Save predictions
    output_path = output or video_path.with_suffix(".predictions.slp")
    sio.save_file(labels, output_path.as_posix())

    click.echo(f"\nInference complete:")
    click.echo(f"  Total time: {stats['total_time']:.2f}s")
    click.echo(f"  Inference time: {stats['infer_time']:.2f}s")
    click.echo(f"  Post-processing time: {stats['post_time']:.2f}s")
    click.echo(f"  FPS: {stats['fps']:.2f}")
    click.echo(f"  Frames with predictions: {len(labels.labeled_frames)}")
    click.echo(f"  Output saved to: {output_path}")


def _copy_training_config(
    model_path: Path, export_dir: Path, label: Optional[str]
) -> Optional[Path]:
    training_config_path = _training_config_path(model_path)
    if training_config_path is None:
        return None

    if label:
        dest_name = f"training_config_{label}{training_config_path.suffix}"
    else:
        dest_name = training_config_path.name

    dest_path = export_dir / dest_name
    shutil.copy(training_config_path, dest_path)
    return dest_path


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
    lightning_cls = {
        "centroid": CentroidLightningModule,
        "centered_instance": TopDownCenteredInstanceLightningModule,
        "single_instance": SingleInstanceLightningModule,
        "bottomup": BottomUpLightningModule,
        "multi_class_topdown": TopDownCenteredInstanceMultiClassLightningModule,
        "multi_class_bottomup": BottomUpMultiClassLightningModule,
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
