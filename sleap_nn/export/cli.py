"""CLI entry points for export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import shutil

import click
from omegaconf import OmegaConf
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
    default=-0.5,
    show_default=True,
    help="Bottom-up: minimum line score threshold.",
)
@click.option(
    "--peak-conf-threshold",
    type=float,
    default=0.1,
    show_default=True,
    help="Bottom-up: peak confidence threshold for filtering candidates.",
)
@click.option(
    "--max-instances",
    type=int,
    default=None,
    help="Maximum instances to output per frame.",
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
) -> None:
    """Run inference on exported models and save predictions to SLP.

    EXPORT_DIR is the directory containing the exported model (model.onnx or model.trt)
    along with export_metadata.json and training_config.yaml.

    VIDEO_PATH is the path to the video file to process.
    """
    import time
    from datetime import datetime

    import numpy as np
    import sleap_io as sio

    from sleap_nn.export.metadata import ExportMetadata
    from sleap_nn.export.predictors import load_exported_model
    from sleap_nn.export.utils import build_bottomup_candidate_template
    from sleap_nn.inference.paf_grouping import PAFScorer
    from sleap_nn.inference.utils import get_skeleton_from_config

    # Load metadata
    metadata_path = export_dir / "export_metadata.json"
    if not metadata_path.exists():
        raise click.ClickException(f"Metadata not found: {metadata_path}")
    metadata = ExportMetadata.load(metadata_path)

    # Find model file
    onnx_path = export_dir / "model.onnx"
    trt_path = export_dir / "model.trt"

    if runtime == "auto":
        if trt_path.exists():
            model_path = trt_path
            runtime = "tensorrt"
        elif onnx_path.exists():
            model_path = onnx_path
            runtime = "onnx"
        else:
            raise click.ClickException(
                f"No model found in {export_dir}. Expected model.onnx or model.trt."
            )
    elif runtime == "onnx":
        if not onnx_path.exists():
            raise click.ClickException(f"ONNX model not found: {onnx_path}")
        model_path = onnx_path
    elif runtime == "tensorrt":
        if not trt_path.exists():
            raise click.ClickException(f"TensorRT model not found: {trt_path}")
        model_path = trt_path
    else:
        raise click.ClickException(f"Unknown runtime: {runtime}")

    # Load training config for skeleton
    cfg_path = _find_training_config_for_predict(export_dir, metadata.model_type)
    if cfg_path.suffix in {".yaml", ".yml"}:
        cfg = OmegaConf.load(cfg_path.as_posix())
    else:
        from sleap_nn.config.training_job_config import TrainingJobConfig

        cfg = TrainingJobConfig.load_sleap_config(cfg_path.as_posix())
    skeletons = get_skeleton_from_config(cfg.data_config.skeletons)
    skeleton = skeletons[0]

    # Load video
    video = sio.Video.from_filename(video_path.as_posix())
    total_frames = len(video) if n_frames is None else min(n_frames, len(video))
    frame_indices = list(range(total_frames))

    click.echo(f"Loading model from: {model_path}")
    click.echo(f"  Model type: {metadata.model_type}")
    click.echo(f"  Runtime: {runtime}")
    click.echo(f"  Device: {device}")

    predictor = load_exported_model(
        model_path.as_posix(), runtime=runtime, device=device
    )

    click.echo(f"Processing video: {video_path}")
    click.echo(f"  Total frames: {total_frames}")
    click.echo(f"  Batch size: {batch_size}")

    # Set up centroid anchor node if needed
    anchor_node_idx = None
    if metadata.model_type == "centroid":
        anchor_part = cfg.model_config.head_configs.centroid.confmaps.anchor_part
        node_names = [n.name for n in skeleton.nodes]
        if anchor_part in node_names:
            anchor_node_idx = node_names.index(anchor_part)
        else:
            raise click.ClickException(
                f"Anchor part '{anchor_part}' not found in skeleton nodes: {node_names}"
            )

    # Set up bottom-up post-processing if needed
    paf_scorer = None
    candidate_template = None
    if metadata.model_type == "bottomup":
        paf_scorer = PAFScorer.from_config(
            cfg.model_config.head_configs.bottomup,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            n_points=n_points,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
        )
        max_peaks = metadata.max_peaks_per_node
        if max_peaks is None:
            raise click.ClickException(
                "Bottom-up export metadata missing max_peaks_per_node."
            )
        edge_inds_tuples = [(int(e[0]), int(e[1])) for e in paf_scorer.edge_inds]
        peak_channel_inds, edge_inds_tensor, edge_peak_inds = (
            build_bottomup_candidate_template(
                n_nodes=metadata.n_nodes,
                max_peaks_per_node=max_peaks,
                edge_inds=edge_inds_tuples,
            )
        )
        candidate_template = {
            "peak_channel_inds": peak_channel_inds,
            "edge_inds": edge_inds_tensor,
            "edge_peak_inds": edge_peak_inds,
        }

    labeled_frames = []
    total_start = time.perf_counter()
    infer_time = 0.0
    post_time = 0.0

    for start in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[start : start + batch_size]
        batch = _load_video_batch(video, batch_indices)

        infer_start = time.perf_counter()
        outputs = predictor.predict(batch)
        infer_time += time.perf_counter() - infer_start

        post_start = time.perf_counter()
        if metadata.model_type == "topdown":
            labeled_frames.extend(
                _predict_topdown_frames(
                    outputs,
                    batch_indices,
                    video,
                    skeleton,
                    max_instances=max_instances,
                )
            )
        elif metadata.model_type == "bottomup":
            labeled_frames.extend(
                _predict_bottomup_frames(
                    outputs,
                    batch_indices,
                    video,
                    skeleton,
                    paf_scorer,
                    candidate_template,
                    input_scale=metadata.input_scale,
                    peak_conf_threshold=peak_conf_threshold,
                    max_instances=max_instances,
                )
            )
        elif metadata.model_type == "single_instance":
            labeled_frames.extend(
                _predict_single_instance_frames(
                    outputs,
                    batch_indices,
                    video,
                    skeleton,
                )
            )
        elif metadata.model_type == "centroid":
            labeled_frames.extend(
                _predict_centroid_frames(
                    outputs,
                    batch_indices,
                    video,
                    skeleton,
                    anchor_node_idx=anchor_node_idx,
                    max_instances=max_instances,
                )
            )
        elif metadata.model_type == "multi_class_bottomup":
            labeled_frames.extend(
                _predict_multiclass_bottomup_frames(
                    outputs,
                    batch_indices,
                    video,
                    skeleton,
                    class_names=metadata.class_names or [],
                    input_scale=metadata.input_scale,
                    peak_conf_threshold=peak_conf_threshold,
                    max_instances=max_instances,
                )
            )
        elif metadata.model_type == "multi_class_topdown_combined":
            labeled_frames.extend(
                _predict_multiclass_topdown_combined_frames(
                    outputs,
                    batch_indices,
                    video,
                    skeleton,
                    class_names=metadata.class_names or [],
                    max_instances=max_instances,
                )
            )
        else:
            raise click.ClickException(
                f"Unsupported model_type for predict: {metadata.model_type}"
            )
        post_time += time.perf_counter() - post_start

        # Progress update
        processed = min(start + batch_size, len(frame_indices))
        click.echo(
            f"\r  Processed {processed}/{len(frame_indices)} frames...",
            nl=False,
        )

    click.echo()  # Newline after progress

    total_time = time.perf_counter() - total_start
    fps = len(frame_indices) / total_time if total_time > 0 else 0

    # Save predictions
    output_path = output or video_path.with_suffix(".predictions.slp")
    labels = sio.Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=labeled_frames,
    )
    labels.provenance = {
        "sleap_nn_version": metadata.sleap_nn_version,
        "export_format": runtime,
        "model_type": metadata.model_type,
        "inference_timestamp": datetime.now().isoformat(),
    }
    sio.save_file(labels, output_path.as_posix())

    click.echo(f"\nInference complete:")
    click.echo(f"  Total time: {total_time:.2f}s")
    click.echo(f"  Inference time: {infer_time:.2f}s")
    click.echo(f"  Post-processing time: {post_time:.2f}s")
    click.echo(f"  FPS: {fps:.2f}")
    click.echo(f"  Frames with predictions: {len(labeled_frames)}")
    click.echo(f"  Output saved to: {output_path}")


def _find_training_config_for_predict(export_dir: Path, model_type: str) -> Path:
    """Find training config file in export directory."""
    candidates = []
    if model_type == "topdown":
        candidates.extend(
            [
                export_dir / "training_config_centered_instance.yaml",
                export_dir / "training_config_centered_instance.json",
            ]
        )
    elif model_type == "multi_class_topdown_combined":
        candidates.extend(
            [
                export_dir / "training_config_multi_class_topdown.yaml",
                export_dir / "training_config_multi_class_topdown.json",
            ]
        )
    candidates.extend(
        [
            export_dir / "training_config.yaml",
            export_dir / "training_config.json",
            export_dir / f"training_config_{model_type}.yaml",
            export_dir / f"training_config_{model_type}.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise click.ClickException(
        f"No training_config found in {export_dir} for model_type={model_type}."
    )


def _load_video_batch(video, frame_indices):
    """Load a batch of video frames as uint8 NCHW array."""
    import numpy as np

    frames = []
    for idx in frame_indices:
        frame = np.asarray(video[idx])
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        frames.append(frame)
    return np.stack(frames, axis=0)


def _predict_topdown_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    max_instances=None,
):
    """Convert top-down model outputs to LabeledFrames."""
    import sleap_io as sio

    labeled_frames = []
    centroids = outputs["centroids"]
    centroid_vals = outputs["centroid_vals"]
    peaks = outputs["peaks"]
    peak_vals = outputs["peak_vals"]
    instance_valid = outputs["instance_valid"]

    for batch_idx, frame_idx in enumerate(frame_indices):
        instances = []
        valid_mask = instance_valid[batch_idx].astype(bool)
        for inst_idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            pts = peaks[batch_idx, inst_idx]
            scores = peak_vals[batch_idx, inst_idx]
            score = float(centroid_vals[batch_idx, inst_idx])
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=score,
                    skeleton=skeleton,
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_multiclass_topdown_combined_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    class_names: list,
    max_instances=None,
):
    """Convert combined multiclass top-down model outputs to LabeledFrames.

    Args:
        outputs: Model outputs with centroids, centroid_vals, peaks, peak_vals,
                class_logits, instance_valid.
        frame_indices: Frame indices corresponding to batch.
        video: sleap_io.Video object.
        skeleton: sleap_io.Skeleton object.
        class_names: List of class names (e.g., ["female", "male"]).
        max_instances: Maximum instances per frame (None = n_classes).

    Returns:
        List of LabeledFrame objects.
    """
    import numpy as np
    import sleap_io as sio
    from scipy.optimize import linear_sum_assignment

    labeled_frames = []
    centroids = outputs["centroids"]
    centroid_vals = outputs["centroid_vals"]
    peaks = outputs["peaks"]
    peak_vals = outputs["peak_vals"]
    class_logits = outputs["class_logits"]
    instance_valid = outputs["instance_valid"]

    n_classes = len(class_names)

    for batch_idx, frame_idx in enumerate(frame_indices):
        valid_mask = instance_valid[batch_idx].astype(bool)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            continue

        # Gather valid instances
        valid_peaks = peaks[batch_idx, valid_mask]  # (n_valid, n_nodes, 2)
        valid_peak_vals = peak_vals[batch_idx, valid_mask]  # (n_valid, n_nodes)
        valid_centroid_vals = centroid_vals[batch_idx, valid_mask]  # (n_valid,)
        valid_class_logits = class_logits[batch_idx, valid_mask]  # (n_valid, n_classes)

        # Compute softmax probabilities from logits
        logits = valid_class_logits - np.max(valid_class_logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # Use Hungarian matching to assign classes to instances
        # Maximize total probability (minimize negative)
        cost = -probs
        row_inds, col_inds = linear_sum_assignment(cost)

        # Create instances with class assignments
        instances = []
        for row_idx, class_idx in zip(row_inds, col_inds):
            pts = valid_peaks[row_idx]
            scores = valid_peak_vals[row_idx]
            score = float(valid_centroid_vals[row_idx])

            # Get track name from class names
            track_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"class_{class_idx}"
            )

            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=score,
                    skeleton=skeleton,
                    track=sio.Track(name=track_name),
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_bottomup_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    paf_scorer,
    candidate_template,
    input_scale,
    peak_conf_threshold=0.1,
    max_instances=None,
):
    """Convert bottom-up model outputs to LabeledFrames."""
    import numpy as np
    import sleap_io as sio
    import torch

    labeled_frames = []

    peaks = torch.from_numpy(outputs["peaks"]).to(torch.float32)
    peak_vals = torch.from_numpy(outputs["peak_vals"]).to(torch.float32)
    line_scores = torch.from_numpy(outputs["line_scores"]).to(torch.float32)
    candidate_mask = torch.from_numpy(outputs["candidate_mask"]).to(torch.bool)

    batch_size, n_nodes, k, _ = peaks.shape
    peaks_flat = peaks.reshape(batch_size, n_nodes * k, 2)
    peak_vals_flat = peak_vals.reshape(batch_size, n_nodes * k)

    peak_channel_inds_base = candidate_template["peak_channel_inds"]
    edge_inds_base = candidate_template["edge_inds"]
    edge_peak_inds_base = candidate_template["edge_peak_inds"]

    peaks_list = []
    peak_vals_list = []
    peak_channel_inds_list = []
    edge_inds_list = []
    edge_peak_inds_list = []
    line_scores_list = []

    for b in range(batch_size):
        peaks_list.append(peaks_flat[b])
        peak_vals_list.append(peak_vals_flat[b])
        peak_channel_inds_list.append(peak_channel_inds_base)

        candidate_mask_flat = candidate_mask[b].reshape(-1)
        line_scores_flat = line_scores[b].reshape(-1)

        if candidate_mask_flat.numel() == 0:
            edge_inds_list.append(torch.empty((0,), dtype=torch.int32))
            edge_peak_inds_list.append(torch.empty((0, 2), dtype=torch.int32))
            line_scores_list.append(torch.empty((0,), dtype=torch.float32))
            continue

        # Filter candidates by peak confidence threshold
        peak_vals_b = peak_vals_flat[b]
        peak_conf_valid = peak_vals_b > peak_conf_threshold
        src_valid = peak_conf_valid[edge_peak_inds_base[:, 0].long()]
        dst_valid = peak_conf_valid[edge_peak_inds_base[:, 1].long()]
        valid = candidate_mask_flat & src_valid & dst_valid

        edge_inds_list.append(edge_inds_base[valid])
        edge_peak_inds_list.append(edge_peak_inds_base[valid])
        line_scores_list.append(line_scores_flat[valid])

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = paf_scorer.match_candidates(
        edge_inds_list,
        edge_peak_inds_list,
        line_scores_list,
    )

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = paf_scorer.group_instances(
        peaks_list,
        peak_vals_list,
        peak_channel_inds_list,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )

    predicted_instances = [p / input_scale for p in predicted_instances]

    for batch_idx, frame_idx in enumerate(frame_indices):
        instances = []
        for pts, confs, score in zip(
            predicted_instances[batch_idx],
            predicted_peak_scores[batch_idx],
            predicted_instance_scores[batch_idx],
        ):
            pts_np = pts.cpu().numpy()
            if np.isnan(pts_np).all():
                continue
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts_np,
                    point_scores=confs.cpu().numpy(),
                    score=float(score),
                    skeleton=skeleton,
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_single_instance_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
):
    """Convert single-instance model outputs to LabeledFrames."""
    import numpy as np
    import sleap_io as sio

    labeled_frames = []
    peaks = outputs["peaks"]  # (batch, n_nodes, 2)
    peak_vals = outputs["peak_vals"]  # (batch, n_nodes)

    for batch_idx, frame_idx in enumerate(frame_indices):
        pts = peaks[batch_idx]
        scores = peak_vals[batch_idx]

        # Compute instance score as mean of valid peak values
        valid_mask = ~np.isnan(pts[:, 0])
        if valid_mask.any():
            instance_score = float(np.mean(scores[valid_mask]))
        else:
            instance_score = 0.0

        instance = sio.PredictedInstance.from_numpy(
            points_data=pts,
            point_scores=scores,
            score=instance_score,
            skeleton=skeleton,
        )

        labeled_frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=int(frame_idx),
                instances=[instance],
            )
        )

    return labeled_frames


def _predict_centroid_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    anchor_node_idx: int,
    max_instances=None,
):
    """Convert centroid model outputs to LabeledFrames.

    For centroid-only models, creates instances with only the anchor node filled in.
    All other nodes are set to NaN.

    Args:
        outputs: Model outputs with centroids, centroid_vals, instance_valid.
        frame_indices: Frame indices corresponding to batch.
        video: sleap_io.Video object.
        skeleton: sleap_io.Skeleton object.
        anchor_node_idx: Index of the anchor node in the skeleton.
        max_instances: Maximum instances to output per frame.

    Returns:
        List of LabeledFrame objects.
    """
    import numpy as np
    import sleap_io as sio

    labeled_frames = []
    centroids = outputs["centroids"]  # (batch, max_instances, 2)
    centroid_vals = outputs["centroid_vals"]  # (batch, max_instances)
    instance_valid = outputs["instance_valid"]  # (batch, max_instances)

    n_nodes = len(skeleton.nodes)

    for batch_idx, frame_idx in enumerate(frame_indices):
        instances = []
        valid_mask = instance_valid[batch_idx].astype(bool)

        for inst_idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue

            # Create points array with NaN for all nodes except anchor
            pts = np.full((n_nodes, 2), np.nan, dtype=np.float32)
            pts[anchor_node_idx] = centroids[batch_idx, inst_idx]

            # Create scores array - anchor gets centroid score, others get NaN
            scores = np.full((n_nodes,), np.nan, dtype=np.float32)
            scores[anchor_node_idx] = centroid_vals[batch_idx, inst_idx]

            instance_score = float(centroid_vals[batch_idx, inst_idx])

            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=instance_score,
                    skeleton=skeleton,
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_multiclass_bottomup_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    class_names: list,
    input_scale: float = 1.0,
    peak_conf_threshold: float = 0.1,
    max_instances: int = None,
):
    """Convert bottom-up multiclass model outputs to LabeledFrames.

    Uses class probability maps to group peaks by identity rather than PAFs.

    Args:
        outputs: Model outputs with peaks, peak_vals, peak_mask, class_probs.
        frame_indices: Frame indices corresponding to batch.
        video: sleap_io.Video object.
        skeleton: sleap_io.Skeleton object.
        class_names: List of class names (e.g., ["female", "male"]).
        input_scale: Scale factor applied to input.
        peak_conf_threshold: Minimum peak confidence to include.
        max_instances: Maximum instances per frame (None = n_classes).

    Returns:
        List of LabeledFrame objects.
    """
    import numpy as np
    import sleap_io as sio
    from scipy.optimize import linear_sum_assignment

    labeled_frames = []
    n_classes = len(class_names)

    peaks = outputs["peaks"]  # (batch, n_nodes, max_peaks, 2)
    peak_vals = outputs["peak_vals"]  # (batch, n_nodes, max_peaks)
    peak_mask = outputs["peak_mask"]  # (batch, n_nodes, max_peaks)
    class_probs = outputs["class_probs"]  # (batch, n_nodes, max_peaks, n_classes)

    batch_size, n_nodes, max_peaks, _ = peaks.shape
    n_nodes_skel = len(skeleton.nodes)

    for batch_idx, frame_idx in enumerate(frame_indices):
        # Initialize instances for each class
        instance_points = np.full(
            (n_classes, n_nodes_skel, 2), np.nan, dtype=np.float32
        )
        instance_scores = np.full((n_classes, n_nodes_skel), np.nan, dtype=np.float32)
        instance_class_probs = np.full((n_classes,), 0.0, dtype=np.float32)

        # Process each node independently
        for node_idx in range(min(n_nodes, n_nodes_skel)):
            # Get valid peaks for this node
            valid = peak_mask[batch_idx, node_idx].astype(bool)
            valid = valid & (peak_vals[batch_idx, node_idx] > peak_conf_threshold)

            if not valid.any():
                continue

            valid_peaks = peaks[batch_idx, node_idx][valid]  # (n_valid, 2)
            valid_vals = peak_vals[batch_idx, node_idx][valid]  # (n_valid,)
            valid_class_probs = class_probs[batch_idx, node_idx][
                valid
            ]  # (n_valid, n_classes)

            # Use Hungarian matching to assign peaks to classes
            # Maximize class probabilities (minimize negative)
            cost = -valid_class_probs
            row_inds, col_inds = linear_sum_assignment(cost)

            # Assign matched peaks to instances
            for peak_idx, class_idx in zip(row_inds, col_inds):
                if class_idx < n_classes:
                    instance_points[class_idx, node_idx] = (
                        valid_peaks[peak_idx] / input_scale
                    )
                    instance_scores[class_idx, node_idx] = valid_vals[peak_idx]
                    instance_class_probs[class_idx] += valid_class_probs[
                        peak_idx, class_idx
                    ]

        # Create predicted instances
        instances = []
        for class_idx in range(n_classes):
            pts = instance_points[class_idx]
            scores = instance_scores[class_idx]

            # Skip if no valid points
            if np.isnan(pts).all():
                continue

            # Compute instance score as mean of valid peak values
            valid_mask = ~np.isnan(pts[:, 0])
            if valid_mask.any():
                instance_score = float(np.mean(scores[valid_mask]))
            else:
                instance_score = 0.0

            # Get track name from class names
            track_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"class_{class_idx}"
            )

            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=instance_score,
                    skeleton=skeleton,
                    track=sio.Track(name=track_name),
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


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
