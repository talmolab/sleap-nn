"""CLI entry points for export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import shutil

import click
import torch

from sleap_nn.export.exporters import export_to_onnx
from sleap_nn.export.metadata import build_base_metadata, embed_metadata_in_onnx, hash_file
from sleap_nn.export.utils import (
    load_training_config,
    resolve_backbone_type,
    resolve_crop_size,
    resolve_edge_inds,
    resolve_input_channels,
    resolve_input_scale,
    resolve_input_shape,
    resolve_model_type,
    resolve_node_names,
    resolve_output_stride,
    resolve_pafs_output_stride,
)
from sleap_nn.export.wrappers import (
    BottomUpONNXWrapper,
    CenteredInstanceONNXWrapper,
    CentroidONNXWrapper,
    TopDownONNXWrapper,
)
from sleap_nn.training.lightning_modules import (
    BottomUpLightningModule,
    CentroidLightningModule,
    SingleInstanceLightningModule,
    TopDownCenteredInstanceLightningModule,
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
    verify: bool,
) -> None:
    """Export trained models to ONNX/TensorRT formats."""
    if fmt.lower() != "onnx":
        raise click.ClickException("Only ONNX export is implemented for now.")

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
        ):
            raise click.ClickException(
                f"Model type '{model_type}' is not supported for export yet."
            )

        if model_type == "single_instance":
            raise click.ClickException("Single-instance export is not implemented yet.")

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
                "confmaps",
                "pafs",
            ]
            metadata_max_peaks = max_peaks_per_node
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
        )

        metadata.save(export_dir / "export_metadata.json")

        if training_config_text is not None:
            try:
                embed_metadata_in_onnx(model_out_path, metadata, training_config_text)
            except ImportError:
                pass
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

        centroid_cfg_path = _copy_training_config(
            centroid_path, export_dir, "centroid"
        )
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
            config_hashes.append(
                f"centered_instance:{hash_file(instance_cfg_path)}"
            )

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
        return

    raise click.ClickException(
        "Provide one model path for centroid/centered-instance/bottom-up export, "
        "or two paths (centroid + centered_instance) for top-down export."
    )


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
