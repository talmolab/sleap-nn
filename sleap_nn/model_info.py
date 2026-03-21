"""Model information display for trained models and configs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


def _format_param_count(count) -> str:
    """Format parameter count to human-readable string."""
    if count is None:
        return "N/A"
    count = int(count)
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _format_model_type(model_type: Optional[str]) -> str:
    """Format model type to human-readable string."""
    if model_type is None:
        return "Unknown"
    return model_type.replace("_", " ").title()


def _format_file_size(size_bytes: int) -> str:
    """Format file size to human-readable string."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1_000_000_000:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1_000:.1f} KB"
    return f"{size_bytes} B"


def _shorten_path(path: str, max_len: int = 50) -> str:
    """Shorten a path for display, keeping the end."""
    if len(path) <= max_len:
        return path
    return "..." + path[-(max_len - 3) :]


def _format_backbone_summary(cfg, backbone_type: str) -> str:
    """Format backbone config as a one-line summary."""
    from omegaconf import OmegaConf

    if backbone_type is None:
        return "Unknown"

    bb = cfg.model_config.backbone_config[backbone_type]
    if bb is None:
        return backbone_type

    if backbone_type == "unet":
        parts = [f"{bb.filters} filters"]
        if hasattr(bb, "filters_rate") and bb.filters_rate is not None:
            parts.append(f"{bb.filters_rate}x rate")
        parts.append(f"stride {bb.max_stride}")
        return f"UNet ({', '.join(parts)})"
    elif backbone_type == "convnext":
        model_type = getattr(bb, "model_type", "custom")
        return f"ConvNeXt ({model_type}, stride {bb.max_stride})"
    elif backbone_type == "swint":
        model_type = getattr(bb, "model_type", "custom")
        return f"SwinT ({model_type}, stride {bb.max_stride})"
    else:
        return backbone_type


def _format_head_summary(cfg, model_type: str) -> str:
    """Format head config as a one-line summary."""
    if model_type is None:
        return "Unknown"

    head_cfg = cfg.model_config.head_configs[model_type]
    if head_cfg is None:
        return _format_model_type(model_type)

    parts = []

    # Confidence maps info
    confmaps = getattr(head_cfg, "confmaps", None)
    if confmaps is not None:
        parts.append(f"sigma={confmaps.sigma}")
        parts.append(f"output_stride={confmaps.output_stride}")
        part_names = getattr(confmaps, "part_names", None)
        if part_names is not None:
            parts.append(f"{len(part_names)} parts")

    # PAFs info (bottom-up models)
    pafs = getattr(head_cfg, "pafs", None)
    if pafs is not None:
        parts.append(f"PAFs (sigma={pafs.sigma}, stride={pafs.output_stride})")

    if parts:
        return f"ConfMaps ({', '.join(parts)})"
    return _format_model_type(model_type)


def _load_training_log(model_dir: Path) -> Optional[dict]:
    """Load training_log.csv and extract summary stats."""
    log_path = model_dir / "training_log.csv"
    if not log_path.exists():
        return None

    with open(log_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # Parse val_loss values, skipping empty strings
    val_losses = []
    for row in rows:
        vl = row.get("val_loss", "")
        if vl:
            try:
                val_losses.append((int(row["epoch"]), float(vl)))
            except (ValueError, KeyError):
                continue

    last_row = rows[-1]
    best_val = min(val_losses, key=lambda x: x[1]) if val_losses else (None, None)

    def _safe_float(row, key):
        v = row.get(key, "")
        if v:
            try:
                return float(v)
            except ValueError:
                pass
        return None

    return {
        "epochs_trained": int(last_row["epoch"]) + 1 if "epoch" in last_row else None,
        "final_train_loss": _safe_float(last_row, "train_loss"),
        "final_val_loss": _safe_float(last_row, "val_loss"),
        "best_val_loss": best_val[1],
        "best_val_epoch": best_val[0],
        "final_lr": _safe_float(last_row, "learning_rate"),
    }


def _load_available_metrics(model_dir: Path) -> tuple[Optional[dict], str]:
    """Try to load metrics, returning (metrics_dict, split_name) or (None, "")."""
    from sleap_nn.evaluation import load_metrics

    for split in ("val", "train"):
        try:
            metrics = load_metrics(str(model_dir), split=split, dataset_idx=0)
            return metrics, split
        except FileNotFoundError:
            continue
    return None, ""


def print_model_info(path: str) -> None:
    """Display model configuration and evaluation metrics.

    Args:
        path: Path to a trained model directory or a training config YAML file.
    """
    from omegaconf import OmegaConf
    from rich.console import Console
    from rich.table import Table

    from sleap_nn.config.utils import (
        get_backbone_type_from_cfg,
        get_model_type_from_cfg,
    )
    from sleap_nn.export.utils import load_training_config

    console = Console()
    p = Path(path)

    if not p.exists():
        console.print(f"[red]Error:[/red] Path not found: {path}")
        raise SystemExit(1)

    # Load config
    if p.is_dir():
        try:
            cfg = load_training_config(p)
        except FileNotFoundError:
            console.print(
                f"[red]Error:[/red] No training_config.yaml or .json found in {path}"
            )
            raise SystemExit(1)
        is_model_dir = True
        model_dir = p
    elif p.is_file() and p.suffix in (".yaml", ".yml"):
        cfg = OmegaConf.load(str(p))
        is_model_dir = False
        model_dir = None
    else:
        console.print(f"[red]Error:[/red] {path} is not a model directory or YAML file")
        raise SystemExit(1)

    # Extract model/backbone type
    model_type = get_model_type_from_cfg(cfg)
    backbone_type = get_backbone_type_from_cfg(cfg)

    # --- Table 1: Model Info ---
    table = Table(title="Model Info", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    run_name = OmegaConf.select(cfg, "trainer_config.run_name", default=None)
    if run_name:
        table.add_row("Run name", str(run_name))

    description = OmegaConf.select(cfg, "description", default=None)
    if description:
        table.add_row("Description", str(description))

    version = OmegaConf.select(cfg, "sleap_nn_version", default=None)
    if version:
        table.add_row("sleap-nn version", str(version))

    table.add_row("Model type", _format_model_type(model_type))
    table.add_row("Backbone", _format_backbone_summary(cfg, backbone_type))
    table.add_row("Head", _format_head_summary(cfg, model_type))

    total_params = OmegaConf.select(cfg, "model_config.total_params", default=None)
    table.add_row("Total parameters", _format_param_count(total_params))

    # Skeleton info
    skeletons = OmegaConf.select(cfg, "data_config.skeletons", default=None)
    if skeletons and len(skeletons) > 0:
        skel = skeletons[0]
        nodes = OmegaConf.select(skel, "nodes", default=[])
        edges = OmegaConf.select(skel, "edges", default=[])
        node_names = [n.get("name", n) if hasattr(n, "get") else str(n) for n in nodes]
        n_nodes, n_edges = len(nodes), len(edges)
        table.add_row(
            "Skeleton",
            f"{n_nodes} {'node' if n_nodes == 1 else 'nodes'}, "
            f"{n_edges} {'edge' if n_edges == 1 else 'edges'}",
        )
        if node_names:
            table.add_row("Nodes", ", ".join(node_names))

    console.print(table)

    # --- Table 2: Data ---
    console.print()
    data_table = Table(title="Data", show_header=False)
    data_table.add_column("Property", style="cyan")
    data_table.add_column("Value", style="white")

    train_paths = OmegaConf.select(cfg, "data_config.train_labels_path", default=None)
    if train_paths:
        for tp in train_paths:
            data_table.add_row("Training data", _shorten_path(str(tp)))
    else:
        data_table.add_row("Training data", "N/A")

    val_paths = OmegaConf.select(cfg, "data_config.val_labels_path", default=None)
    if val_paths:
        for vp in val_paths:
            data_table.add_row("Validation data", _shorten_path(str(vp)))

    # Preprocessing summary
    pre = OmegaConf.select(cfg, "data_config.preprocessing", default=None)
    if pre is not None:
        scale = OmegaConf.select(pre, "scale", default=1.0)
        max_h = OmegaConf.select(pre, "max_height", default=None)
        max_w = OmegaConf.select(pre, "max_width", default=None)
        crop_size = OmegaConf.select(pre, "crop_size", default=None)

        parts = [f"scale={scale}"]
        if max_h is not None and max_w is not None:
            parts.append(f"{max_h}x{max_w}")
        if crop_size is not None:
            parts.append(f"crop={crop_size}")
        else:
            parts.append("no crop")
        data_table.add_row("Preprocessing", ", ".join(parts))

    aug = OmegaConf.select(cfg, "data_config.use_augmentations_train", default=False)
    data_table.add_row("Augmentations", "Enabled" if aug else "Disabled")

    console.print(data_table)

    # --- Table 3: Training ---
    console.print()
    train_table = Table(title="Training", show_header=False)
    train_table.add_column("Property", style="cyan")
    train_table.add_column("Value", style="white")

    opt_name = OmegaConf.select(cfg, "trainer_config.optimizer_name", default="Adam")
    lr = OmegaConf.select(cfg, "trainer_config.optimizer.lr", default=None)
    if lr is not None:
        train_table.add_row("Optimizer", f"{opt_name} (lr={lr})")
    else:
        train_table.add_row("Optimizer", str(opt_name))

    max_epochs = OmegaConf.select(cfg, "trainer_config.max_epochs", default=None)
    if max_epochs is not None:
        train_table.add_row("Max epochs", str(max_epochs))

    batch_size = OmegaConf.select(
        cfg, "trainer_config.train_data_loader.batch_size", default=None
    )
    if batch_size is not None:
        train_table.add_row("Batch size", str(batch_size))

    # LR scheduler
    step_lr = OmegaConf.select(cfg, "trainer_config.lr_scheduler.step_lr", default=None)
    reduce_lr = OmegaConf.select(
        cfg, "trainer_config.lr_scheduler.reduce_lr_on_plateau", default=None
    )
    if step_lr is not None:
        step_size = OmegaConf.select(step_lr, "step_size", default="?")
        gamma = OmegaConf.select(step_lr, "gamma", default="?")
        train_table.add_row("LR scheduler", f"StepLR (step={step_size}, gamma={gamma})")
    elif reduce_lr is not None:
        factor = OmegaConf.select(reduce_lr, "factor", default="?")
        patience = OmegaConf.select(reduce_lr, "patience", default="?")
        train_table.add_row(
            "LR scheduler", f"ReduceLROnPlateau (factor={factor}, patience={patience})"
        )
    else:
        train_table.add_row("LR scheduler", "None")

    # Early stopping
    es_enabled = OmegaConf.select(
        cfg, "trainer_config.early_stopping.stop_training_on_plateau", default=False
    )
    if es_enabled:
        patience = OmegaConf.select(
            cfg, "trainer_config.early_stopping.patience", default="?"
        )
        min_delta = OmegaConf.select(
            cfg, "trainer_config.early_stopping.min_delta", default="?"
        )
        train_table.add_row(
            "Early stopping", f"patience={patience}, min_delta={min_delta}"
        )
    else:
        train_table.add_row("Early stopping", "Disabled")

    console.print(train_table)

    # Model-dir-only sections
    if not is_model_dir:
        return

    # --- Table 4: Training Results ---
    log_stats = _load_training_log(model_dir)
    if log_stats is not None:
        console.print()
        results_table = Table(title="Training Results", show_header=False)
        results_table.add_column("Property", style="cyan")
        results_table.add_column("Value", style="white")

        if log_stats["epochs_trained"] is not None:
            results_table.add_row("Epochs trained", str(log_stats["epochs_trained"]))
        if log_stats["final_train_loss"] is not None:
            results_table.add_row(
                "Final train loss", f"{log_stats['final_train_loss']:.6f}"
            )
        if log_stats["final_val_loss"] is not None:
            results_table.add_row(
                "Final val loss", f"{log_stats['final_val_loss']:.6f}"
            )
        if log_stats["best_val_loss"] is not None:
            results_table.add_row(
                "Best val loss",
                f"{log_stats['best_val_loss']:.6f} (epoch {log_stats['best_val_epoch']})",
            )
        if log_stats["final_lr"] is not None:
            results_table.add_row("Final LR", f"{log_stats['final_lr']}")

        console.print(results_table)

    # --- Table 5: Evaluation Metrics ---
    metrics, split = _load_available_metrics(model_dir)
    if metrics is not None:
        console.print()
        metrics_table = Table(title=f"Evaluation Metrics ({split})", show_header=False)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")

        def _fmt(val, fmt=".4f"):
            if val is None:
                return "N/A"
            try:
                return f"{float(val):{fmt}}"
            except (TypeError, ValueError):
                return str(val)

        # mOKS
        moks = metrics.get("mOKS", {})
        metrics_table.add_row("mOKS", _fmt(moks.get("mOKS")))

        # VOC metrics
        voc = metrics.get("voc_metrics", {})
        metrics_table.add_row("mAP (OKS)", _fmt(voc.get("oks_voc.mAP")))
        metrics_table.add_row("mAR (OKS)", _fmt(voc.get("oks_voc.mAR")))

        # Distance metrics
        dist = metrics.get("distance_metrics", {})
        metrics_table.add_row("Avg distance (px)", _fmt(dist.get("avg"), ".2f"))
        metrics_table.add_row("Median distance (px)", _fmt(dist.get("p50"), ".2f"))
        metrics_table.add_row("P95 distance (px)", _fmt(dist.get("p95"), ".2f"))

        # PCK metrics
        pck = metrics.get("pck_metrics", {})
        metrics_table.add_row("mPCK", _fmt(pck.get("mPCK")))
        metrics_table.add_row("PCK@5", _fmt(pck.get("PCK@5")))
        metrics_table.add_row("PCK@10", _fmt(pck.get("PCK@10")))

        # Visibility metrics
        vis = metrics.get("visibility_metrics", {})
        metrics_table.add_row("Vis. precision", _fmt(vis.get("precision")))
        metrics_table.add_row("Vis. recall", _fmt(vis.get("recall")))

        console.print(metrics_table)

    # --- Table 6: Files ---
    console.print()
    files_table = Table(title="Files")
    files_table.add_column("File", style="cyan")
    files_table.add_column("Size", style="white", justify="right")

    files = sorted(
        [f for f in model_dir.iterdir() if f.is_file() and not f.name.startswith(".")],
        key=lambda f: f.name,
    )
    for f in files:
        files_table.add_row(f.name, _format_file_size(f.stat().st_size))

    console.print(files_table)
