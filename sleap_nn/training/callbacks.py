"""Custom Callback modules for Lightning Trainer."""

import zmq
import jsonpickle
from typing import Callable, Optional, Union
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from loguru import logger
import matplotlib

matplotlib.use(
    "Agg"
)  # Use non-interactive backend to avoid tkinter issues on Windows CI
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import wandb
import csv
from sleap_nn import RANK


class SleapProgressBar(TQDMProgressBar):
    """Custom progress bar with better formatting for small metric values.

    The default TQDMProgressBar truncates small floats like 1e-5 to "0.000".
    This subclass formats metrics using scientific notation when appropriate.
    """

    def get_metrics(
        self, trainer, pl_module
    ) -> dict[str, Union[int, str, float, dict[str, float]]]:
        """Override to format metrics with scientific notation for small values."""
        items = super().get_metrics(trainer, pl_module)
        formatted = {}
        for k, v in items.items():
            if isinstance(v, float):
                # Use scientific notation for very small values
                if v != 0 and abs(v) < 0.001:
                    formatted[k] = f"{v:.2e}"
                else:
                    # Use 4 decimal places for normal values
                    formatted[k] = f"{v:.4f}"
            else:
                formatted[k] = v
        return formatted


class CSVLoggerCallback(Callback):
    """Callback for logging metrics to csv.

    Attributes:
        filepath: Path to save the csv file.
        keys: List of field names to be logged in the csv.
    """

    def __init__(
        self,
        filepath: Path,
        keys: list = ["epoch", "train_loss", "val_loss", "learning_rate"],
    ):
        """Initialize attributes."""
        super().__init__()
        self.filepath = filepath
        self.keys = keys
        self.initialized = False

    def _init_file(self):
        """Create the .csv file."""
        if RANK in [0, -1]:  # Global rank 0 or -1 (non-distributed)
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.keys)
                writer.writeheader()
        self.initialized = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """Log metrics to csv at the end of validation epoch."""
        # Access callback_metrics BEFORE the is_global_zero guard so all
        # ranks participate in the implicit all_reduce that fires when
        # sync_dist=True metrics are first read.  Only rank 0 does I/O.
        metrics = trainer.callback_metrics
        if trainer.is_global_zero:
            if not self.initialized:
                self._init_file()
            log_data = {}
            for key in self.keys:
                if key == "epoch":
                    log_data["epoch"] = trainer.current_epoch
                elif key == "learning_rate":
                    # Handle multiple formats:
                    # 1. Direct "learning_rate" key
                    # 2. "train/lr" key (current format from lightning modules)
                    # 3. "lr-*" keys from LearningRateMonitor (legacy)
                    value = metrics.get(key, None)
                    if value is None:
                        value = metrics.get("train/lr", None)
                    if value is None:
                        # Look for lr-* keys from LearningRateMonitor (legacy)
                        for metric_key in metrics.keys():
                            if metric_key.startswith("lr-"):
                                value = metrics[metric_key]
                                break
                    log_data[key] = value.item() if value is not None else None
                else:
                    value = metrics.get(key, None)
                    log_data[key] = value.item() if value is not None else None

            with open(self.filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.keys)
                writer.writerow(log_data)

        # Sync all processes after file I/O
        trainer.strategy.barrier()


class WandBPredImageLogger(Callback):
    """Callback for writing image predictions to wandb as a Table.

    .. deprecated::
        This callback logs images to a wandb.Table which doesn't support
        step sliders. Use WandBVizCallback instead for better UX.

    Attributes:
        viz_folder: Path to viz directory.
        wandb_run_name: WandB run name.
        is_bottomup: If the model type is bottomup or not.
    """

    def __init__(
        self,
        viz_folder: str,
        wandb_run_name: str,
        is_bottomup: bool = False,
    ):
        """Initialize attributes."""
        self.viz_folder = viz_folder
        self.wandb_run_name = wandb_run_name
        self.is_bottomup = is_bottomup
        # Callback initialization
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch."""
        if trainer.is_global_zero:
            epoch_num = trainer.current_epoch
            train_img_path = (
                Path(self.viz_folder) / f"train.{epoch_num:04d}.png"
            ).as_posix()
            val_img_path = (
                Path(self.viz_folder) / f"validation.{epoch_num:04d}.png"
            ).as_posix()
            train_img = Image.open(train_img_path)
            val_img = Image.open(val_img_path)

            column_names = [
                "Run name",
                "Epoch",
                "Preds on train",
                "Preds on validation",
            ]
            data = [
                [
                    f"{self.wandb_run_name}",
                    f"{epoch_num}",
                    wandb.Image(train_img),
                    wandb.Image(val_img),
                ]
            ]
            if self.is_bottomup:
                column_names.extend(["Pafs Preds on train", "Pafs Preds on validation"])
                data = [
                    [
                        f"{self.wandb_run_name}",
                        f"{epoch_num}",
                        wandb.Image(train_img),
                        wandb.Image(val_img),
                        wandb.Image(
                            Image.open(
                                (
                                    Path(self.viz_folder)
                                    / f"train.pafs_magnitude.{epoch_num:04d}.png"
                                ).as_posix()
                            )
                        ),
                        wandb.Image(
                            Image.open(
                                (
                                    Path(self.viz_folder)
                                    / f"validation.pafs_magnitude.{epoch_num:04d}.png"
                                ).as_posix()
                            )
                        ),
                    ]
                ]
            table = wandb.Table(columns=column_names, data=data)
            # Use commit=False to accumulate with other metrics in this step
            wandb.log({f"{self.wandb_run_name}": table}, commit=False)

        # Sync all processes after wandb logging
        trainer.strategy.barrier()


class WandBVizCallback(Callback):
    """Callback for logging visualization images directly to wandb with slider support.

    This callback logs images using wandb.log() which enables step slider navigation
    in the wandb UI. Multiple visualization modes can be enabled simultaneously:
    - viz_enabled: Pre-render with matplotlib (same as disk viz)
    - viz_boxes: Interactive keypoint boxes with filtering
    - viz_masks: Confidence map overlay with per-node toggling

    Attributes:
        train_viz_fn: Function that returns VisualizationData for training sample.
        val_viz_fn: Function that returns VisualizationData for validation sample.
        viz_enabled: Whether to log pre-rendered matplotlib images.
        viz_boxes: Whether to log interactive keypoint boxes.
        viz_masks: Whether to log confidence map overlay masks.
        box_size: Size of keypoint boxes in pixels (for viz_boxes).
        confmap_threshold: Threshold for confmap masks (for viz_masks).
        log_table: Whether to also log to a wandb.Table (backwards compat).
    """

    def __init__(
        self,
        train_viz_fn: Callable,
        val_viz_fn: Callable,
        viz_enabled: bool = True,
        viz_boxes: bool = False,
        viz_masks: bool = False,
        box_size: float = 5.0,
        confmap_threshold: float = 0.1,
        log_table: bool = False,
    ):
        """Initialize the callback.

        Args:
            train_viz_fn: Callable that returns VisualizationData for a training sample.
            val_viz_fn: Callable that returns VisualizationData for a validation sample.
            viz_enabled: If True, log pre-rendered matplotlib images.
            viz_boxes: If True, log interactive keypoint boxes.
            viz_masks: If True, log confidence map overlay masks.
            box_size: Size of keypoint boxes in pixels (for viz_boxes).
            confmap_threshold: Threshold for confmap mask generation (for viz_masks).
            log_table: If True, also log images to a wandb.Table (for backwards compat).
        """
        super().__init__()
        self.train_viz_fn = train_viz_fn
        self.val_viz_fn = val_viz_fn
        self.viz_enabled = viz_enabled
        self.viz_boxes = viz_boxes
        self.viz_masks = viz_masks
        self.log_table = log_table

        # Import here to avoid circular imports
        from sleap_nn.training.utils import WandBRenderer

        self.box_size = box_size
        self.confmap_threshold = confmap_threshold

        # Create renderers for each enabled mode
        self.renderers = {}
        if viz_enabled:
            self.renderers["direct"] = WandBRenderer(
                mode="direct", box_size=box_size, confmap_threshold=confmap_threshold
            )
        if viz_boxes:
            self.renderers["boxes"] = WandBRenderer(
                mode="boxes", box_size=box_size, confmap_threshold=confmap_threshold
            )
        if viz_masks:
            self.renderers["masks"] = WandBRenderer(
                mode="masks", box_size=box_size, confmap_threshold=confmap_threshold
            )

    def _get_wandb_logger(self, trainer):
        """Get the WandbLogger from trainer's loggers."""
        from lightning.pytorch.loggers import WandbLogger

        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None

    def on_train_epoch_end(self, trainer, pl_module):
        """Log visualization images at end of each epoch."""
        if trainer.is_global_zero:
            epoch = trainer.current_epoch

            # Get the wandb logger to use its experiment for logging
            wandb_logger = self._get_wandb_logger(trainer)

            # Only do visualization work if wandb logger is available
            if wandb_logger is not None:
                # Get visualization data
                train_data = self.train_viz_fn()
                val_data = self.val_viz_fn()

                # Render and log for each enabled mode
                # Use the logger's experiment to let Lightning manage step tracking
                log_dict = {}
                for mode_name, renderer in self.renderers.items():
                    suffix = "" if mode_name == "direct" else f"_{mode_name}"
                    train_img = renderer.render(
                        train_data, caption=f"Train Epoch {epoch}"
                    )
                    val_img = renderer.render(val_data, caption=f"Val Epoch {epoch}")
                    log_dict[f"viz/train/predictions{suffix}"] = train_img
                    log_dict[f"viz/val/predictions{suffix}"] = val_img

                if log_dict:
                    # Include epoch so wandb can use it as x-axis (via define_metric)
                    log_dict["epoch"] = epoch
                    # Use commit=False to accumulate with other metrics in this step
                    # Lightning will commit when it logs its own metrics
                    wandb_logger.experiment.log(log_dict, commit=False)

                # Optionally also log to table for backwards compat
                if self.log_table and "direct" in self.renderers:
                    train_img = self.renderers["direct"].render(
                        train_data, caption=f"Train Epoch {epoch}"
                    )
                    val_img = self.renderers["direct"].render(
                        val_data, caption=f"Val Epoch {epoch}"
                    )
                    table = wandb.Table(
                        columns=["Epoch", "Train", "Validation"],
                        data=[[epoch, train_img, val_img]],
                    )
                    wandb_logger.experiment.log(
                        {"predictions_table": table}, commit=False
                    )

        # Sync all processes - barrier must be reached by ALL ranks
        trainer.strategy.barrier()


class WandBVizCallbackWithPAFs(WandBVizCallback):
    """Extended WandBVizCallback that also logs PAF visualizations for bottom-up models."""

    def __init__(
        self,
        train_viz_fn: Callable,
        val_viz_fn: Callable,
        train_pafs_viz_fn: Callable,
        val_pafs_viz_fn: Callable,
        viz_enabled: bool = True,
        viz_boxes: bool = False,
        viz_masks: bool = False,
        box_size: float = 5.0,
        confmap_threshold: float = 0.1,
        log_table: bool = False,
    ):
        """Initialize the callback.

        Args:
            train_viz_fn: Callable returning VisualizationData for training sample.
            val_viz_fn: Callable returning VisualizationData for validation sample.
            train_pafs_viz_fn: Callable returning VisualizationData with PAFs for training.
            val_pafs_viz_fn: Callable returning VisualizationData with PAFs for validation.
            viz_enabled: If True, log pre-rendered matplotlib images.
            viz_boxes: If True, log interactive keypoint boxes.
            viz_masks: If True, log confidence map overlay masks.
            box_size: Size of keypoint boxes in pixels.
            confmap_threshold: Threshold for confmap mask generation.
            log_table: If True, also log images to a wandb.Table.
        """
        super().__init__(
            train_viz_fn=train_viz_fn,
            val_viz_fn=val_viz_fn,
            viz_enabled=viz_enabled,
            viz_boxes=viz_boxes,
            viz_masks=viz_masks,
            box_size=box_size,
            confmap_threshold=confmap_threshold,
            log_table=log_table,
        )
        self.train_pafs_viz_fn = train_pafs_viz_fn
        self.val_pafs_viz_fn = val_pafs_viz_fn

        # Import here to avoid circular imports
        from sleap_nn.training.utils import MatplotlibRenderer

        self._mpl_renderer = MatplotlibRenderer()

    def on_train_epoch_end(self, trainer, pl_module):
        """Log visualization images including PAFs at end of each epoch."""
        if trainer.is_global_zero:
            epoch = trainer.current_epoch

            # Get the wandb logger to use its experiment for logging
            wandb_logger = self._get_wandb_logger(trainer)

            # Only do visualization work if wandb logger is available
            if wandb_logger is not None:
                # Get visualization data
                train_data = self.train_viz_fn()
                val_data = self.val_viz_fn()
                train_pafs_data = self.train_pafs_viz_fn()
                val_pafs_data = self.val_pafs_viz_fn()

                # Render and log for each enabled mode
                # Use the logger's experiment to let Lightning manage step tracking
                log_dict = {}
                for mode_name, renderer in self.renderers.items():
                    suffix = "" if mode_name == "direct" else f"_{mode_name}"
                    train_img = renderer.render(
                        train_data, caption=f"Train Epoch {epoch}"
                    )
                    val_img = renderer.render(val_data, caption=f"Val Epoch {epoch}")
                    log_dict[f"viz/train/predictions{suffix}"] = train_img
                    log_dict[f"viz/val/predictions{suffix}"] = val_img

                # Render PAFs (always use matplotlib/direct for PAFs)
                from io import BytesIO
                import matplotlib.pyplot as plt
                from PIL import Image

                train_pafs_fig = self._mpl_renderer.render_pafs(train_pafs_data)
                buf = BytesIO()
                train_pafs_fig.savefig(
                    buf, format="png", bbox_inches="tight", pad_inches=0
                )
                buf.seek(0)
                plt.close(train_pafs_fig)
                train_pafs_pil = Image.open(buf)
                log_dict["viz/train/pafs"] = wandb.Image(
                    train_pafs_pil, caption=f"Train PAFs Epoch {epoch}"
                )

                val_pafs_fig = self._mpl_renderer.render_pafs(val_pafs_data)
                buf = BytesIO()
                val_pafs_fig.savefig(
                    buf, format="png", bbox_inches="tight", pad_inches=0
                )
                buf.seek(0)
                plt.close(val_pafs_fig)
                val_pafs_pil = Image.open(buf)
                log_dict["viz/val/pafs"] = wandb.Image(
                    val_pafs_pil, caption=f"Val PAFs Epoch {epoch}"
                )

                if log_dict:
                    # Include epoch so wandb can use it as x-axis (via define_metric)
                    log_dict["epoch"] = epoch
                    # Use commit=False to accumulate with other metrics in this step
                    # Lightning will commit when it logs its own metrics
                    wandb_logger.experiment.log(log_dict, commit=False)

                # Optionally also log to table
                if self.log_table and "direct" in self.renderers:
                    train_img = self.renderers["direct"].render(
                        train_data, caption=f"Train Epoch {epoch}"
                    )
                    val_img = self.renderers["direct"].render(
                        val_data, caption=f"Val Epoch {epoch}"
                    )
                    table = wandb.Table(
                        columns=[
                            "Epoch",
                            "Train",
                            "Validation",
                            "Train PAFs",
                            "Val PAFs",
                        ],
                        data=[
                            [
                                epoch,
                                train_img,
                                val_img,
                                log_dict["viz/train/pafs"],
                                log_dict["viz/val/pafs"],
                            ]
                        ],
                    )
                    wandb_logger.experiment.log(
                        {"predictions_table": table}, commit=False
                    )

        # Sync all processes - barrier must be reached by ALL ranks
        trainer.strategy.barrier()


class UnifiedVizCallback(Callback):
    """Unified callback for all visualization outputs during training.

    This callback consolidates all visualization functionality into a single callback,
    eliminating redundant dataset copies and inference runs. It handles:
    - Local disk saving (matplotlib figures)
    - WandB logging (multiple modes: direct, boxes, masks)
    - Model-specific visualizations (PAFs for bottomup, class maps for multi_class_bottomup)

    Benefits over separate callbacks:
    - Uses ONE sample per epoch for all visualizations (no dataset deepcopy)
    - Runs inference ONCE per sample (vs 4-8x in previous implementation)
    - Outputs to multiple destinations from the same data
    - Simpler code with less duplication

    Attributes:
        model_trainer: Reference to the ModelTrainer (for lazy access to lightning_model).
        train_pipeline: Iterator over training visualization dataset.
        val_pipeline: Iterator over validation visualization dataset.
        model_type: Type of model (affects which visualizations are enabled).
        save_local: Whether to save matplotlib figures to disk.
        local_save_dir: Directory for local visualization saves.
        log_wandb: Whether to log visualizations to wandb.
        wandb_modes: List of wandb rendering modes ("direct", "boxes", "masks").
        wandb_box_size: Size of keypoint boxes in pixels (for "boxes" mode).
        wandb_confmap_threshold: Threshold for confmap masks (for "masks" mode).
        log_wandb_table: Whether to also log to a wandb.Table.
    """

    def __init__(
        self,
        model_trainer,
        train_dataset,
        val_dataset,
        model_type: str,
        save_local: bool = True,
        local_save_dir: Optional[Path] = None,
        log_wandb: bool = False,
        wandb_modes: Optional[list] = None,
        wandb_box_size: float = 5.0,
        wandb_confmap_threshold: float = 0.1,
        log_wandb_table: bool = False,
    ):
        """Initialize the unified visualization callback.

        Args:
            model_trainer: ModelTrainer instance (lightning_model accessed lazily).
            train_dataset: Training visualization dataset (will be cycled).
            val_dataset: Validation visualization dataset (will be cycled).
            model_type: Model type string (e.g., "bottomup", "multi_class_bottomup").
            save_local: If True, save matplotlib figures to local_save_dir.
            local_save_dir: Path to directory for saving visualization images.
            log_wandb: If True, log visualizations to wandb.
            wandb_modes: List of wandb rendering modes. Defaults to ["direct"].
            wandb_box_size: Size of keypoint boxes in pixels.
            wandb_confmap_threshold: Threshold for confidence map masks.
            log_wandb_table: If True, also log to a wandb.Table.
        """
        super().__init__()
        from itertools import cycle

        self.model_trainer = model_trainer
        self.train_pipeline = cycle(train_dataset)
        self.val_pipeline = cycle(val_dataset)
        # Raw (un-cycled) dataset refs — the embedding scatter samples them by index
        # to spread points across the whole set (the per-sample image viz uses the
        # cycles above).
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_type = model_type

        # Local disk config
        self.save_local = save_local
        self.local_save_dir = local_save_dir

        # WandB config
        self.log_wandb = log_wandb
        self.wandb_modes = wandb_modes or ["direct"]
        self.wandb_box_size = wandb_box_size
        self.wandb_confmap_threshold = wandb_confmap_threshold
        self.log_wandb_table = log_wandb_table

        # Auto-enable model-specific visualizations
        self.viz_pafs = model_type == "bottomup"
        self.viz_class_maps = model_type == "multi_class_bottomup"
        self.viz_center_heatmap = model_type == "bottomup_segmentation"
        self.viz_offsets = model_type == "bottomup_segmentation"
        # Colored grouped per-instance mask overlay (bottom-up seg only; the
        # grouped masks come from the offset-grouping in get_visualization_data).
        self.viz_instance_masks = model_type == "bottomup_segmentation"
        # GT-vs-prediction mask overlay for both segmentation model types.
        self.viz_gt_mask = model_type in (
            "bottomup_segmentation",
            "centered_instance_segmentation",
        )
        # The `embedding` model is skeleton-less (crop -> vector): the per-sample
        # keypoint/confmap viz makes no sense (and would crash on the missing
        # `get_visualization_data`). Instead render a 2D embedding-scatter panel
        # (SPEC §9), handled by a dedicated branch in `on_train_epoch_end`.
        self.viz_embedding = model_type == "embedding"
        # Max crops sampled per split for the scatter (spread across the dataset).
        self.embedding_scatter_n = 256

        # Initialize renderers
        from sleap_nn.training.utils import MatplotlibRenderer, WandBRenderer

        self._mpl_renderer = MatplotlibRenderer()

        # Create wandb renderers for each enabled mode
        self._wandb_renderers = {}
        if log_wandb:
            for mode in self.wandb_modes:
                self._wandb_renderers[mode] = WandBRenderer(
                    mode=mode,
                    box_size=wandb_box_size,
                    confmap_threshold=wandb_confmap_threshold,
                )

    def _get_wandb_logger(self, trainer):
        """Get the WandbLogger from trainer's loggers."""
        from lightning.pytorch.loggers import WandbLogger

        for log in trainer.loggers:
            if isinstance(log, WandbLogger):
                return log
        return None

    def _get_viz_data(self, sample):
        """Get visualization data with all needed fields based on model type.

        Args:
            sample: A sample from the visualization dataset.

        Returns:
            VisualizationData with appropriate fields populated.
        """
        # Build kwargs based on model type
        kwargs = {}
        if self.viz_pafs:
            kwargs["include_pafs"] = True
        if self.viz_class_maps:
            kwargs["include_class_maps"] = True
        if self.viz_center_heatmap:
            kwargs["include_center_heatmap"] = True
        if self.viz_offsets:
            kwargs["include_offsets"] = True
        if self.viz_gt_mask:
            kwargs["include_gt_mask"] = True
        if self.viz_instance_masks:
            kwargs["include_instance_masks"] = True

        # Access lightning_model lazily from model_trainer
        return self.model_trainer.lightning_model.get_visualization_data(
            sample, **kwargs
        )

    def _save_local_viz(self, data, prefix: str, epoch: int):
        """Save visualization to local disk.

        Args:
            data: VisualizationData object.
            prefix: Filename prefix (e.g., "train", "validation").
            epoch: Current epoch number.
        """
        if not self.save_local or self.local_save_dir is None:
            return

        # Confmaps visualization
        fig = self._mpl_renderer.render(data)
        fig_path = self.local_save_dir / f"{prefix}.{epoch:04d}.png"
        fig.savefig(fig_path, format="png")
        plt.close(fig)

        # PAFs visualization (for bottomup models)
        if self.viz_pafs and data.pred_pafs is not None:
            fig = self._mpl_renderer.render_pafs(data)
            fig_path = self.local_save_dir / f"{prefix}.pafs_magnitude.{epoch:04d}.png"
            fig.savefig(fig_path, format="png")
            plt.close(fig)

        # Class maps visualization (for multi_class_bottomup models)
        if self.viz_class_maps and data.pred_class_maps is not None:
            fig = self._render_class_maps(data)
            fig_path = self.local_save_dir / f"{prefix}.class_maps.{epoch:04d}.png"
            fig.savefig(fig_path, format="png")
            plt.close(fig)

        # Center heatmap visualization (for segmentation models)
        if self.viz_center_heatmap and data.pred_center_heatmap is not None:
            fig = self._render_center_heatmap(data)
            fig_path = self.local_save_dir / f"{prefix}.center_heatmap.{epoch:04d}.png"
            fig.savefig(fig_path, format="png")
            plt.close(fig)

        # Center-offset field visualization (for segmentation models)
        if self.viz_offsets and data.pred_offsets is not None:
            fig = self._mpl_renderer.render_offsets(data)
            fig_path = self.local_save_dir / f"{prefix}.offsets.{epoch:04d}.png"
            fig.savefig(fig_path, format="png")
            plt.close(fig)

        # GT-vs-prediction foreground mask overlay (for segmentation models)
        if self.viz_gt_mask and data.gt_mask is not None:
            fig = self._mpl_renderer.render_gt_mask(data)
            fig_path = self.local_save_dir / f"{prefix}.gt_mask.{epoch:04d}.png"
            fig.savefig(fig_path, format="png")
            plt.close(fig)

        # Colored grouped per-instance mask overlay (bottom-up segmentation)
        if self.viz_instance_masks and data.instance_masks is not None:
            fig = self._mpl_renderer.render_instance_masks(data)
            fig_path = self.local_save_dir / f"{prefix}.instance_masks.{epoch:04d}.png"
            fig.savefig(fig_path, format="png")
            plt.close(fig)

    def _render_class_maps(self, data):
        """Render class maps visualization.

        Args:
            data: VisualizationData with pred_class_maps populated.

        Returns:
            A matplotlib Figure object.
        """
        from sleap_nn.training.utils import plot_img, plot_confmaps

        img = data.image
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0

        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(
            data.pred_class_maps,
            output_scale=data.pred_class_maps.shape[0] / img.shape[0],
        )
        return fig

    def _render_center_heatmap(self, data):
        """Render center heatmap visualization for segmentation models.

        Args:
            data: VisualizationData with pred_center_heatmap populated.

        Returns:
            A matplotlib Figure object.
        """
        from sleap_nn.training.utils import plot_img, plot_confmaps

        img = data.image
        scale = 1.0
        if img.shape[0] < 512:
            scale = 2.0
        if img.shape[0] < 256:
            scale = 4.0

        fig = plot_img(img, dpi=72 * scale, scale=scale)
        plot_confmaps(
            data.pred_center_heatmap,
            output_scale=data.pred_center_heatmap.shape[0] / img.shape[0],
        )
        return fig

    def _log_wandb_viz(self, data, prefix: str, epoch: int, wandb_logger):
        """Log visualization to wandb.

        Args:
            data: VisualizationData object.
            prefix: Log prefix (e.g., "train", "val").
            epoch: Current epoch number.
            wandb_logger: WandbLogger instance.
        """
        if not self.log_wandb or wandb_logger is None:
            return

        from io import BytesIO
        from PIL import Image as PILImage

        log_dict = {}

        # Render confmaps for each enabled mode
        for mode_name, renderer in self._wandb_renderers.items():
            suffix = "" if mode_name == "direct" else f"_{mode_name}"
            img = renderer.render(data, caption=f"{prefix.title()} Epoch {epoch}")
            log_dict[f"viz/{prefix}/predictions{suffix}"] = img

        # PAFs visualization (for bottomup models)
        if self.viz_pafs and data.pred_pafs is not None:
            pafs_fig = self._mpl_renderer.render_pafs(data)
            buf = BytesIO()
            pafs_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(pafs_fig)
            pafs_pil = PILImage.open(buf)
            log_dict[f"viz/{prefix}/pafs"] = wandb.Image(
                pafs_pil, caption=f"{prefix.title()} PAFs Epoch {epoch}"
            )

        # Class maps visualization (for multi_class_bottomup models)
        if self.viz_class_maps and data.pred_class_maps is not None:
            class_fig = self._render_class_maps(data)
            buf = BytesIO()
            class_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(class_fig)
            class_pil = PILImage.open(buf)
            log_dict[f"viz/{prefix}/class_maps"] = wandb.Image(
                class_pil, caption=f"{prefix.title()} Class Maps Epoch {epoch}"
            )

        # Center heatmap visualization (for segmentation models)
        if self.viz_center_heatmap and data.pred_center_heatmap is not None:
            center_fig = self._render_center_heatmap(data)
            buf = BytesIO()
            center_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(center_fig)
            center_pil = PILImage.open(buf)
            log_dict[f"viz/{prefix}/center_heatmap"] = wandb.Image(
                center_pil,
                caption=f"{prefix.title()} Center Heatmap Epoch {epoch}",
            )

        # Center-offset field visualization (for segmentation models)
        if self.viz_offsets and data.pred_offsets is not None:
            offsets_fig = self._mpl_renderer.render_offsets(data)
            buf = BytesIO()
            offsets_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(offsets_fig)
            offsets_pil = PILImage.open(buf)
            log_dict[f"viz/{prefix}/offsets"] = wandb.Image(
                offsets_pil,
                caption=f"{prefix.title()} Offset Magnitude Epoch {epoch}",
            )

        # GT-vs-prediction foreground mask overlay (for segmentation models)
        if self.viz_gt_mask and data.gt_mask is not None:
            gt_mask_fig = self._mpl_renderer.render_gt_mask(data)
            buf = BytesIO()
            gt_mask_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(gt_mask_fig)
            gt_mask_pil = PILImage.open(buf)
            log_dict[f"viz/{prefix}/gt_mask"] = wandb.Image(
                gt_mask_pil,
                caption=f"{prefix.title()} GT vs Pred Mask Epoch {epoch}",
            )

        # Colored grouped per-instance mask overlay (bottom-up segmentation)
        if self.viz_instance_masks and data.instance_masks is not None:
            inst_fig = self._mpl_renderer.render_instance_masks(data)
            buf = BytesIO()
            inst_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(inst_fig)
            inst_pil = PILImage.open(buf)
            log_dict[f"viz/{prefix}/instance_masks"] = wandb.Image(
                inst_pil,
                caption=f"{prefix.title()} Instance Masks Epoch {epoch}",
            )

        if log_dict:
            log_dict["epoch"] = epoch
            wandb_logger.experiment.log(log_dict, commit=False)

        # Optionally log to table for backwards compatibility
        if self.log_wandb_table and "direct" in self._wandb_renderers:
            train_img = self._wandb_renderers["direct"].render(
                data, caption=f"{prefix.title()} Epoch {epoch}"
            )
            table_data = [[epoch, train_img]]
            columns = ["Epoch", prefix.title()]

            if self.viz_pafs and data.pred_pafs is not None:
                columns.append(f"{prefix.title()} PAFs")
                table_data[0].append(log_dict.get(f"viz/{prefix}/pafs"))

            if self.viz_class_maps and data.pred_class_maps is not None:
                columns.append(f"{prefix.title()} Class Maps")
                table_data[0].append(log_dict.get(f"viz/{prefix}/class_maps"))

            if self.viz_center_heatmap and data.pred_center_heatmap is not None:
                columns.append(f"{prefix.title()} Center Heatmap")
                table_data[0].append(log_dict.get(f"viz/{prefix}/center_heatmap"))

            if self.viz_offsets and data.pred_offsets is not None:
                columns.append(f"{prefix.title()} Offsets")
                table_data[0].append(log_dict.get(f"viz/{prefix}/offsets"))

            if self.viz_gt_mask and data.gt_mask is not None:
                columns.append(f"{prefix.title()} GT Mask")
                table_data[0].append(log_dict.get(f"viz/{prefix}/gt_mask"))

            if self.viz_instance_masks and data.instance_masks is not None:
                columns.append(f"{prefix.title()} Instance Masks")
                table_data[0].append(log_dict.get(f"viz/{prefix}/instance_masks"))

            table = wandb.Table(columns=columns, data=table_data)
            wandb_logger.experiment.log(
                {f"predictions_table_{prefix}": table}, commit=False
            )

    def on_train_epoch_end(self, trainer, pl_module):
        """Generate and output all visualizations at epoch end.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Lightning module (not used, we use self.lightning_module).
        """
        if trainer.is_global_zero:
            epoch = trainer.current_epoch
            wandb_logger = self._get_wandb_logger(trainer) if self.log_wandb else None

            if self.viz_embedding:
                # Skeleton-less embedder: render a 2D embedding-scatter panel instead
                # of the per-sample keypoint viz (which has no meaning here and would
                # crash on the missing `get_visualization_data`).
                self._embedding_viz_epoch(epoch, wandb_logger)
            else:
                # Get ONE sample for train visualization
                train_sample = next(self.train_pipeline)
                # Run inference ONCE with all needed data
                train_data = self._get_viz_data(train_sample)
                # Output to all destinations
                self._save_local_viz(train_data, "train", epoch)
                self._log_wandb_viz(train_data, "train", epoch, wandb_logger)

                # Same for validation
                val_sample = next(self.val_pipeline)
                val_data = self._get_viz_data(val_sample)
                self._save_local_viz(val_data, "validation", epoch)
                self._log_wandb_viz(val_data, "val", epoch, wandb_logger)

        # Sync all processes - barrier must be reached by ALL ranks
        trainer.strategy.barrier()

    # ── embedding-scatter viz (SPEC §9) ─────────────────────────────────────
    def _embed_dataset_sample(self, module, dataset, n: int):
        """Embed up to ``n`` crops spread evenly across ``dataset``.

        Returns ``(embeddings (M, D) float32, group_ids (M,) int)`` or ``(None,
        None)`` if no crops could be sampled. Mirrors the eval embedding path
        (mask burn-in + standardize -> EmbeddingHead, pre-projection) so the scatter
        matches the retrieval metrics.
        """
        import numpy as np
        import torch

        if dataset is None or not hasattr(dataset, "__len__") or len(dataset) == 0:
            return None, None
        total = len(dataset)
        n = min(int(n), total)
        # Evenly-spaced unique indices so the sample spans the whole set (the index
        # order may cluster by identity).
        idxs = sorted(set(np.linspace(0, total - 1, n).astype(int).tolist()))

        grays, masks, groups = [], [], []
        for i in idxs:
            try:
                s = dataset[i]
            except Exception:  # noqa: BLE001 — skip an unreadable crop, keep going
                continue
            grays.append(s["instance_image"])
            masks.append(s["instance_mask"])
            groups.append(int(s["group_id"]))
        if not grays:
            return None, None

        device = module.device
        gray = torch.stack(grays, 0).squeeze(1).to(device=device, dtype=torch.float32)
        mask = torch.stack(masks, 0).squeeze(1).to(device=device, dtype=torch.float32)
        was_training = module.training
        module.eval()
        try:
            with torch.no_grad():
                x = module._build_input(gray, mask)
                emb = module.model(x)["EmbeddingHead"]
        finally:
            if was_training:
                module.train()
        return emb.detach().cpu().float().numpy(), np.asarray(groups)

    @staticmethod
    def _reduce_to_2d(x):
        """Reduce ``(N, D)`` embeddings to ``(N, 2)`` (UMAP if available, else PCA)."""
        import numpy as np

        if x.shape[1] <= 2:
            out = np.zeros((x.shape[0], 2), dtype=np.float32)
            out[:, : x.shape[1]] = x
            return out, "raw"
        if x.shape[0] >= 10:
            try:
                import umap  # optional dependency

                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, x.shape[0] - 1),
                    random_state=0,
                )
                return reducer.fit_transform(x).astype(np.float32), "umap"
            except Exception:  # noqa: BLE001 — fall back to dependency-free PCA
                pass
        import torch

        t = torch.from_numpy(x.astype("float32"))
        t = t - t.mean(0, keepdim=True)
        q = min(2, t.shape[0], t.shape[1])
        _, _, v = torch.pca_lowrank(t, q=q)
        proj = (t @ v[:, :2]).numpy()
        if proj.shape[1] < 2:  # degenerate (1 sample / 1 dim)
            proj = np.pad(proj, ((0, 0), (0, 2 - proj.shape[1])))
        return proj.astype(np.float32), "pca"

    def _embedding_viz_epoch(self, epoch: int, wandb_logger):
        """Render + save/log the 2D embedding scatter for this epoch."""
        import numpy as np

        module = self.model_trainer.lightning_model
        tr_emb, tr_grp = self._embed_dataset_sample(
            module, self.train_dataset, self.embedding_scatter_n
        )
        va_emb, va_grp = self._embed_dataset_sample(
            module, self.val_dataset, self.embedding_scatter_n
        )
        parts = [
            (e, g) for e, g in ((tr_emb, tr_grp), (va_emb, va_grp)) if e is not None
        ]
        if not parts:
            logger.warning("Embedding viz: no crops to embed; skipping scatter.")
            return

        all_emb = np.concatenate([e for e, _ in parts], axis=0)
        all_grp = np.concatenate([g for _, g in parts], axis=0)
        n_train = tr_emb.shape[0] if tr_emb is not None else 0
        emb2d, method = self._reduce_to_2d(all_emb)

        fig = self._render_embedding_scatter(emb2d, all_grp, n_train, epoch, method)

        if self.save_local and self.local_save_dir is not None:
            fig_path = self.local_save_dir / f"embedding_scatter.{epoch:04d}.png"
            fig.savefig(fig_path, format="png", bbox_inches="tight")

        if self.log_wandb and wandb_logger is not None:
            from io import BytesIO
            from PIL import Image as PILImage

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
            buf.seek(0)
            wandb_logger.experiment.log(
                {
                    "viz/embedding_scatter": wandb.Image(
                        PILImage.open(buf),
                        caption=f"Embedding ({method}) — epoch {epoch}",
                    )
                },
                commit=False,
            )
        plt.close(fig)

    @staticmethod
    def _render_embedding_scatter(emb2d, groups, n_train: int, epoch: int, method: str):
        """Scatter of 2D embeddings colored by identity (train faint, val bold)."""
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        uniq = sorted(set(int(g) for g in groups.tolist()))
        cmap = plt.get_cmap("tab20")
        color_of = {g: cmap(i % 20) for i, g in enumerate(uniq)}
        colors = np.array([color_of[int(g)] for g in groups])

        tr = slice(0, n_train)
        va = slice(n_train, None)
        if n_train > 0:
            ax.scatter(
                emb2d[tr, 0],
                emb2d[tr, 1],
                c=colors[tr],
                s=12,
                alpha=0.35,
                linewidths=0,
            )
        if n_train < len(groups):
            ax.scatter(
                emb2d[va, 0],
                emb2d[va, 1],
                c=colors[va],
                s=44,
                alpha=0.95,
                edgecolors="black",
                linewidths=0.5,
            )
        n_val = len(groups) - n_train
        ax.set_title(
            f"Embedding ({method}) — epoch {epoch}\n"
            f"{len(uniq)} ids · train○ {n_train} · val● {n_val}"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig


class MatplotlibSaver(Callback):
    """Callback for saving images rendered with matplotlib during training.

    This is useful for saving visualizations of the training to disk. It will be called
    at the end of each epoch.

    Attributes:
        plot_fn: Function with no arguments that returns a matplotlib figure handle.
        save_folder: Path to a directory to save images to.
        prefix: String that will be prepended to the filenames. This is useful for
            indicating which dataset the visualization was sampled from.

    Notes:
        This will save images with the naming pattern:
            "{save_folder}/{prefix}.{epoch}.png"
        or:
            "{save_folder}/{epoch}.png"
        if a prefix is not specified.
    """

    def __init__(
        self,
        save_folder: str,
        plot_fn: Callable[[], matplotlib.figure.Figure],
        prefix: Optional[str] = None,
    ):
        """Initialize callback."""
        self.save_folder = save_folder
        self.plot_fn = plot_fn
        self.prefix = prefix
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        """Save figure at the end of each epoch."""
        if trainer.is_global_zero:
            # Call plotting function.
            figure = self.plot_fn()

            # Build filename.
            prefix = ""
            if self.prefix is not None:
                prefix = self.prefix + "."
            figure_path = (
                Path(self.save_folder) / f"{prefix}{trainer.current_epoch:04d}.png"
            ).as_posix()

            # Save rendered figure.
            figure.savefig(figure_path, format="png")
            plt.close(figure)

        # Sync all processes after file I/O
        trainer.strategy.barrier()


class TrainingControllerZMQ(Callback):
    """Lightning callback to receive control commands during training via ZMQ.

    This is typically used to allow SLEAP GUI interface (SLEAP LossViewer)
    to dynamically control the training process (stopping early) by publishing commands to a ZMQ socket.

    Attributes:
        address: ZMQ socket address to subscribe to.
        topic: Topic filter for messages.
        timeout: Poll timeout in milliseconds when checking for new messages.
    """

    def __init__(self, address="tcp://127.0.0.1:9000", topic="", poll_timeout=10):
        """Initialize the controller callback by connecting to the specified ZMQ PUB socket."""
        super().__init__()
        self.address = address
        self.topic = topic
        self.timeout = poll_timeout

        # Initialize ZMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.subscribe(self.topic)
        self.socket.connect(self.address)
        logger.info(
            f"Training controller subscribed to: {self.address} (topic: {self.topic})"
        )

    def __del__(self):
        """Close zmq socket and context when callback is destroyed."""
        logger.info("Closing the training controller socket/context.")
        self.socket.close()
        self.context.term()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        if trainer.is_global_zero:
            if self.socket.poll(self.timeout, zmq.POLLIN):
                msg = jsonpickle.decode(self.socket.recv_string())
                logger.info(f"Received control message: {msg}")

                # Stop training
                if msg.get("command") == "stop":
                    trainer.should_stop = True

        # Propagate should_stop to all ranks via all_reduce so every rank
        # exits the training loop together.  trainer.strategy.barrier()
        # synchronises execution position but does not copy the value of
        # should_stop; reduce_boolean_decision performs an all_reduce so
        # every rank receives the same True/False decision.  For single-GPU
        # training this is a no-op identical to the previous behaviour.
        trainer.should_stop = trainer.strategy.reduce_boolean_decision(
            trainer.should_stop, all=False
        )

    #         # Adjust learning rate # TODO: check if we need lr
    #         elif msg.get("command") == "set_lr":
    #             self.set_lr(trainer, pl_module, msg["lr"])

    # def set_lr(self, trainer, pl_module, new_lr):
    #     """Set learning rate for all parameter groups."""
    #     optimizer = trainer.optimizers[0]  # Assuming single optimizer
    #     if not isinstance(new_lr, (float, np.float32, np.float64)):
    #         new_lr = float(np.array(new_lr).astype(np.float64))

    #     logger.info(f"Setting learning rate to {new_lr}")
    #     for param_group in optimizer.param_groups:
    #         param_group["lr"] = new_lr


class ProgressReporterZMQ(Callback):
    """Callback to publish training progress events to a ZMQ PUB socket.

    This is used to publish training metrics to the given socket.

    Attributes:
        address: The ZMQ address to publish to, e.g., "tcp://127.0.0.1:9001".
        what: Identifier tag for the type of training job (e.g., model name or job type).
    """

    def __init__(self, address="tcp://127.0.0.1:9001", what=""):
        """Initialize the progress reporter callback by connecting to the specified ZMQ PUB socket."""
        super().__init__()
        self.address = address
        self.what = what

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.connect(self.address)

        logger.info(
            f"ProgressReporterZMQ publishing to {self.address} for '{self.what}'"
        )

    def __del__(self):
        """Close zmq socket and context when callback is destroyed."""
        logger.info(f"Closing ZMQ reporter.")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()

    def send(self, event: str, logs=None, **kwargs):
        """Send a message over ZMQ."""
        msg = dict(what=self.what, event=event, logs=logs, **kwargs)
        self.socket.send_string(jsonpickle.encode(msg))

    def on_train_start(self, trainer, pl_module):
        """Called at the beginning of training process."""
        if trainer.is_global_zero:
            # Include WandB URL if available
            wandb_url = None
            if wandb.run is not None:
                wandb_url = wandb.run.url
            self.send("train_begin", wandb_url=wandb_url)
        trainer.strategy.barrier()

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training process."""
        if trainer.is_global_zero:
            self.send("train_end")
        trainer.strategy.barrier()

    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the beginning of each epoch."""
        if trainer.is_global_zero:
            self.send("epoch_begin", epoch=trainer.current_epoch)
        trainer.strategy.barrier()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch."""
        # Access callback_metrics BEFORE the is_global_zero guard so all
        # ranks participate in the implicit all_reduce that fires when
        # sync_dist=True metrics are first read.  Only rank 0 sends ZMQ.
        logs = trainer.callback_metrics
        if trainer.is_global_zero:
            self.send(
                "epoch_end", epoch=trainer.current_epoch, logs=self._sanitize_logs(logs)
            )
        trainer.strategy.barrier()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called at the beginning of each training batch."""
        if trainer.is_global_zero:
            self.send("batch_start", batch=batch_idx)
        trainer.strategy.barrier()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        # Access callback_metrics BEFORE the is_global_zero guard so all
        # ranks participate in the implicit all_reduce that fires when
        # sync_dist=True metrics are first read.  Only rank 0 sends ZMQ.
        logs = trainer.callback_metrics
        if trainer.is_global_zero:
            self.send(
                "batch_end",
                epoch=trainer.current_epoch,
                batch=batch_idx,
                logs=self._sanitize_logs(logs),
            )
        trainer.strategy.barrier()

    def _sanitize_logs(self, logs):
        """Convert any torch tensors to Python floats for serialization."""
        return {
            k: float(v.item()) if hasattr(v, "item") else v for k, v in logs.items()
        }


class EpochEndEvaluationCallback(Callback):
    """Callback to run full evaluation metrics at end of validation epochs.

    This callback collects predictions and ground truth during validation,
    then runs the full evaluation pipeline (OKS, mAP, PCK, etc.) and logs
    metrics to WandB.

    Attributes:
        skeleton: sio.Skeleton for creating instances.
        videos: List of sio.Video objects.
        eval_frequency: Run evaluation every N epochs (default: 1).
        oks_stddev: OKS standard deviation (default: 0.025).
        oks_scale: Optional OKS scale override.
        metrics_to_log: List of metric keys to log.
    """

    def __init__(
        self,
        skeleton: "sio.Skeleton",
        videos: list,
        eval_frequency: int = 1,
        oks_stddev: float = 0.025,
        oks_scale: Optional[float] = None,
        metrics_to_log: Optional[list] = None,
    ):
        """Initialize the callback.

        Args:
            skeleton: sio.Skeleton for creating instances.
            videos: List of sio.Video objects.
            eval_frequency: Run evaluation every N epochs (default: 1).
            oks_stddev: OKS standard deviation (default: 0.025).
            oks_scale: Optional OKS scale override.
            metrics_to_log: List of metric keys to log. If None, logs all available.
        """
        super().__init__()
        self.skeleton = skeleton
        self.videos = videos
        self.eval_frequency = eval_frequency
        self.oks_stddev = oks_stddev
        self.oks_scale = oks_scale
        self.metrics_to_log = metrics_to_log or [
            "mOKS",
            "oks_voc.mAP",
            "oks_voc.mAR",
            "distance/avg",
            "distance/p50",
            "distance/p95",
            "distance/p99",
            "mPCK",
            "PCK@5",
            "PCK@10",
            "visibility_precision",
            "visibility_recall",
        ]

    def on_validation_epoch_start(self, trainer, pl_module):
        """Enable prediction collection at the start of validation.

        Skip during sanity check to avoid inference issues.
        """
        if trainer.sanity_checking:
            return
        pl_module._collect_val_predictions = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run evaluation and log metrics at end of validation epoch."""
        import sleap_io as sio
        import numpy as np
        from lightning.pytorch.loggers import WandbLogger
        from sleap_nn.evaluation import Evaluator

        # Determine if we should run evaluation this epoch (only on rank 0)
        should_evaluate = (
            trainer.current_epoch + 1
        ) % self.eval_frequency == 0 and trainer.is_global_zero

        if should_evaluate:
            # Check if we have predictions
            if not pl_module.val_predictions or not pl_module.val_ground_truth:
                logger.warning("No predictions collected for epoch-end evaluation")
            else:
                try:
                    # Build sio.Labels from accumulated predictions and ground truth
                    pred_labels = self._build_pred_labels(
                        pl_module.val_predictions, sio, np
                    )
                    gt_labels = self._build_gt_labels(
                        pl_module.val_ground_truth, sio, np
                    )

                    # Check if we have valid frames to evaluate
                    if len(pred_labels) == 0:
                        logger.warning(
                            "No valid predictions for epoch-end evaluation "
                            "(all predictions may be empty or NaN)"
                        )
                    else:
                        # Run evaluation
                        evaluator = Evaluator(
                            ground_truth_instances=gt_labels,
                            predicted_instances=pred_labels,
                            oks_stddev=self.oks_stddev,
                            oks_scale=self.oks_scale,
                            user_labels_only=False,  # All validation frames are "user" frames
                        )
                        metrics = evaluator.evaluate()

                        # Log to WandB
                        self._log_metrics(trainer, metrics, trainer.current_epoch)

                        logger.info(
                            f"Epoch {trainer.current_epoch} evaluation: "
                            f"PCK@5={metrics['pck_metrics']['PCK@5']:.4f}, "
                            f"mOKS={metrics['mOKS']['mOKS']:.4f}, "
                            f"mAP={metrics['voc_metrics']['oks_voc.mAP']:.4f}"
                        )

                except Exception as e:
                    logger.warning(f"Epoch-end evaluation failed: {e}")

        # Cleanup - all ranks reset the flag, rank 0 clears the lists
        pl_module._collect_val_predictions = False
        if trainer.is_global_zero:
            pl_module.val_predictions = []
            pl_module.val_ground_truth = []

        # Sync all processes - barrier must be reached by ALL ranks
        trainer.strategy.barrier()

    def _build_pred_labels(self, predictions: list, sio, np) -> "sio.Labels":
        """Convert prediction dicts to sio.Labels."""
        labeled_frames = []
        for pred in predictions:
            pred_peaks = pred["pred_peaks"]
            pred_scores = pred["pred_scores"]

            # Handle NaN/missing predictions
            if pred_peaks is None or (
                isinstance(pred_peaks, np.ndarray) and np.isnan(pred_peaks).all()
            ):
                continue

            # Handle multi-instance predictions (bottomup)
            if len(pred_peaks.shape) == 2:
                # Single instance: (n_nodes, 2) -> (1, n_nodes, 2)
                pred_peaks = pred_peaks.reshape(1, -1, 2)
                pred_scores = pred_scores.reshape(1, -1)

            instances = []
            for inst_idx in range(len(pred_peaks)):
                inst_points = pred_peaks[inst_idx]
                inst_scores = pred_scores[inst_idx] if pred_scores is not None else None

                # Skip if all NaN
                if np.isnan(inst_points).all():
                    continue

                inst = sio.PredictedInstance.from_numpy(
                    points_data=inst_points,
                    skeleton=self.skeleton,
                    point_scores=(
                        inst_scores
                        if inst_scores is not None
                        else np.ones(len(inst_points))
                    ),
                    score=(
                        float(np.nanmean(inst_scores))
                        if inst_scores is not None
                        else 1.0
                    ),
                )
                instances.append(inst)

            if instances:
                lf = sio.LabeledFrame(
                    video=self.videos[pred["video_idx"]],
                    frame_idx=pred["frame_idx"],
                    instances=instances,
                )
                labeled_frames.append(lf)

        return sio.Labels(
            videos=self.videos,
            skeletons=[self.skeleton],
            labeled_frames=labeled_frames,
        )

    def _build_gt_labels(self, ground_truth: list, sio, np) -> "sio.Labels":
        """Convert ground truth dicts to sio.Labels."""
        labeled_frames = []
        for gt in ground_truth:
            instances = []
            gt_instances = gt["gt_instances"]

            # Handle shape variations
            if len(gt_instances.shape) == 2:
                # (n_nodes, 2) -> (1, n_nodes, 2)
                gt_instances = gt_instances.reshape(1, -1, 2)

            for i in range(min(gt["num_instances"], len(gt_instances))):
                inst_data = gt_instances[i]
                if np.isnan(inst_data).all():
                    continue
                inst = sio.Instance.from_numpy(
                    points_data=inst_data,
                    skeleton=self.skeleton,
                )
                instances.append(inst)

            if instances:
                lf = sio.LabeledFrame(
                    video=self.videos[gt["video_idx"]],
                    frame_idx=gt["frame_idx"],
                    instances=instances,
                )
                labeled_frames.append(lf)

        return sio.Labels(
            videos=self.videos,
            skeletons=[self.skeleton],
            labeled_frames=labeled_frames,
        )

    def _log_metrics(self, trainer, metrics: dict, epoch: int):
        """Log evaluation metrics to WandB."""
        import numpy as np
        from lightning.pytorch.loggers import WandbLogger

        # Get WandB logger
        wandb_logger = None
        for log in trainer.loggers:
            if isinstance(log, WandbLogger):
                wandb_logger = log
                break

        if wandb_logger is None:
            return

        log_dict = {"epoch": epoch}

        # Extract key metrics with consistent naming
        # All eval metrics use eval/val/ prefix since they're computed on validation data
        if "mOKS" in self.metrics_to_log:
            log_dict["eval/val/mOKS"] = metrics["mOKS"]["mOKS"]

        if "oks_voc.mAP" in self.metrics_to_log:
            log_dict["eval/val/oks_voc_mAP"] = metrics["voc_metrics"]["oks_voc.mAP"]

        if "oks_voc.mAR" in self.metrics_to_log:
            log_dict["eval/val/oks_voc_mAR"] = metrics["voc_metrics"]["oks_voc.mAR"]

        # Distance metrics grouped under eval/val/distance/
        if "distance/avg" in self.metrics_to_log:
            val = metrics["distance_metrics"]["avg"]
            if not np.isnan(val):
                log_dict["eval/val/distance/avg"] = val

        if "distance/p50" in self.metrics_to_log:
            val = metrics["distance_metrics"]["p50"]
            if not np.isnan(val):
                log_dict["eval/val/distance/p50"] = val

        if "distance/p95" in self.metrics_to_log:
            val = metrics["distance_metrics"]["p95"]
            if not np.isnan(val):
                log_dict["eval/val/distance/p95"] = val

        if "distance/p99" in self.metrics_to_log:
            val = metrics["distance_metrics"]["p99"]
            if not np.isnan(val):
                log_dict["eval/val/distance/p99"] = val

        # PCK metrics
        if "mPCK" in self.metrics_to_log:
            log_dict["eval/val/mPCK"] = metrics["pck_metrics"]["mPCK"]

        # PCK at specific thresholds (precomputed in evaluation.py)
        if "PCK@5" in self.metrics_to_log:
            log_dict["eval/val/PCK_5"] = metrics["pck_metrics"]["PCK@5"]

        if "PCK@10" in self.metrics_to_log:
            log_dict["eval/val/PCK_10"] = metrics["pck_metrics"]["PCK@10"]

        # Visibility metrics
        if "visibility_precision" in self.metrics_to_log:
            val = metrics["visibility_metrics"]["precision"]
            if not np.isnan(val):
                log_dict["eval/val/visibility_precision"] = val

        if "visibility_recall" in self.metrics_to_log:
            val = metrics["visibility_metrics"]["recall"]
            if not np.isnan(val):
                log_dict["eval/val/visibility_recall"] = val

        wandb_logger.experiment.log(log_dict, commit=False)

        # Update best metrics in summary (excluding epoch)
        for key, value in log_dict.items():
            if key == "epoch":
                continue
            # Create summary key like "best/eval/val/mOKS"
            summary_key = f"best/{key}"
            current_best = wandb_logger.experiment.summary.get(summary_key)
            # For distance metrics, lower is better; for others, higher is better
            is_distance = "distance" in key
            if current_best is None:
                wandb_logger.experiment.summary[summary_key] = value
            elif is_distance and value < current_best:
                wandb_logger.experiment.summary[summary_key] = value
            elif not is_distance and value > current_best:
                wandb_logger.experiment.summary[summary_key] = value


# `match_centroids` now lives in `sleap_nn.evaluation` (centroid-only distance
# evaluation subsystem). Re-exported here so existing callback imports/tests
# keep working.
from sleap_nn.evaluation import match_centroids  # noqa: E402


class SegmentationEvaluationCallback(Callback):
    """Per-epoch instance-level mask-IoU evaluation for segmentation models.

    Mirrors :class:`CentroidEvaluationCallback`: it flips
    ``pl_module._collect_val_predictions`` on at validation start so the
    segmentation ``validation_step`` collects per-instance predicted and
    ground-truth masks on a shared preprocessed grid (recovered by grouping the
    predicted/GT heads for bottom-up, or the single centered-crop mask for
    top-down), then matches them with the SAME IoU matcher used by the
    post-training mask evaluator (:func:`sleap_nn.evaluation.match_masks`) and logs
    instance-level mask-IoU + detection metrics to wandb. This complements the
    coarse ``val/fg_iou`` with a metric that is sensitive to instance grouping
    (over-/under-segmentation surfaces as false positives/negatives).

    The callback is a no-op unless ``trainer_config.eval.enabled`` is set (it is
    only attached then), so default runs are unaffected.

    Attributes:
        eval_frequency: Run evaluation every N epochs (default: 1).
        match_threshold: IoU threshold in (0, 1] for a matched mask pair to count
            as a true positive (default: 0.5).
    """

    def __init__(self, eval_frequency: int = 1, match_threshold: float = 0.5):
        """Initialize the callback.

        Args:
            eval_frequency: Run evaluation every N epochs (default: 1).
            match_threshold: IoU threshold in (0, 1] for a matched mask pair to
                count as a true positive. The shared
                ``trainer_config.eval.match_threshold`` defaults to 50.0 (a
                centroid pixel distance), which is never a valid IoU, so any value
                outside (0, 1] falls back to 0.5.
        """
        super().__init__()
        self.eval_frequency = eval_frequency
        if match_threshold is None or not (0.0 < float(match_threshold) <= 1.0):
            match_threshold = 0.5
        self.match_threshold = float(match_threshold)

    def on_validation_epoch_start(self, trainer, pl_module):
        """Enable prediction collection at the start of validation.

        Skip during sanity check to avoid inference issues.
        """
        if trainer.sanity_checking:
            return
        pl_module._collect_val_predictions = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run mask-IoU evaluation and log metrics at end of validation epoch."""
        should_evaluate = (
            trainer.current_epoch + 1
        ) % self.eval_frequency == 0 and trainer.is_global_zero

        if should_evaluate:
            if not pl_module.val_predictions or not pl_module.val_ground_truth:
                logger.warning(
                    "No predictions collected for segmentation epoch-end evaluation"
                )
            else:
                try:
                    metrics = self._compute_metrics(
                        pl_module.val_predictions, pl_module.val_ground_truth
                    )
                    self._log_metrics(trainer, metrics, trainer.current_epoch)
                    logger.info(
                        f"Epoch {trainer.current_epoch} segmentation evaluation: "
                        f"mask_mean_iou={metrics['mask_mean_iou']:.4f}, "
                        f"precision={metrics['precision']:.4f}, "
                        f"recall={metrics['recall']:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Segmentation epoch-end evaluation failed: {e}")

        # Cleanup
        pl_module._collect_val_predictions = False
        if trainer.is_global_zero:
            pl_module.val_predictions = []
            pl_module.val_ground_truth = []

        trainer.strategy.barrier()

    def _compute_metrics(self, predictions: list, ground_truth: list) -> dict:
        """Match collected per-instance masks by IoU and aggregate metrics.

        ``predictions`` and ``ground_truth`` are appended in lockstep by the
        seg ``validation_step``; each entry is ``{"masks": [bool (H, W), ...]}``
        for one image/crop, with predicted and GT masks on the SAME grid.
        """
        import numpy as np
        from sleap_nn.evaluation import match_masks

        ious: list = []
        tp = fp = fn = 0
        n_gt = 0
        for pred, gt in zip(predictions, ground_truth):
            pred_masks = pred.get("masks", [])
            gt_masks = gt.get("masks", [])
            n_gt += len(gt_masks)
            _, _, unmatched_pred, unmatched_gt, pair_ious = match_masks(
                pred_masks, gt_masks, min_iou=self.match_threshold
            )
            ious.extend(float(x) for x in pair_ious)
            tp += len(pair_ious)
            fp += len(unmatched_pred)
            fn += len(unmatched_gt)

        mask_mean_iou = float(np.mean(ious)) if ious else float("nan")
        # Misses (unmatched GT) contribute IoU 0 -> a recall-sensitive mean.
        mask_mean_iou_all_gt = (float(np.sum(ious)) / n_gt) if n_gt else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return {
            "mask_mean_iou": mask_mean_iou,
            "mask_mean_iou_all_gt": mask_mean_iou_all_gt,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_tp": tp,
            "n_fp": fp,
            "n_fn": fn,
        }

    def _log_metrics(self, trainer, metrics: dict, epoch: int):
        """Log mask evaluation metrics to WandB."""
        import numpy as np
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = None
        for log in trainer.loggers:
            if isinstance(log, WandbLogger):
                wandb_logger = log
                break
        if wandb_logger is None:
            return

        log_dict = {"epoch": epoch}
        if not np.isnan(metrics["mask_mean_iou"]):
            log_dict["eval/val/mask_mean_iou"] = metrics["mask_mean_iou"]
        if not np.isnan(metrics["mask_mean_iou_all_gt"]):
            log_dict["eval/val/mask_mean_iou_all_gt"] = metrics["mask_mean_iou_all_gt"]
        log_dict["eval/val/mask_precision"] = metrics["precision"]
        log_dict["eval/val/mask_recall"] = metrics["recall"]
        log_dict["eval/val/mask_f1"] = metrics["f1"]
        log_dict["eval/val/mask_n_tp"] = metrics["n_tp"]
        log_dict["eval/val/mask_n_fp"] = metrics["n_fp"]
        log_dict["eval/val/mask_n_fn"] = metrics["n_fn"]

        wandb_logger.experiment.log(log_dict, commit=False)

        # Update best metrics in summary (higher is better for IoU/precision/etc.).
        for key, value in log_dict.items():
            if key == "epoch" or key.endswith(("n_tp", "n_fp", "n_fn")):
                continue
            summary_key = f"best/{key}"
            current_best = wandb_logger.experiment.summary.get(summary_key)
            if current_best is None or value > current_best:
                wandb_logger.experiment.summary[summary_key] = value


class CentroidEvaluationCallback(Callback):
    """Callback to run centroid-specific evaluation metrics at end of validation epochs.

    This callback is designed specifically for centroid models, which predict a single
    point (centroid) per instance rather than full pose skeletons. It computes
    distance-based metrics and detection metrics that are more appropriate for
    point detection tasks than OKS/PCK metrics.

    Metrics computed:
        - Distance metrics: mean, median, p90, p95, max Euclidean distance
        - Detection metrics: precision, recall, F1 score
        - Counts: true positives, false positives, false negatives

    Attributes:
        videos: List of sio.Video objects.
        eval_frequency: Run evaluation every N epochs (default: 1).
        match_threshold: Maximum distance (pixels) for matching pred to GT (default: 50.0).
    """

    def __init__(
        self,
        videos: list,
        eval_frequency: int = 1,
        match_threshold: float = 50.0,
    ):
        """Initialize the callback.

        Args:
            videos: List of sio.Video objects.
            eval_frequency: Run evaluation every N epochs (default: 1).
            match_threshold: Maximum distance in pixels for a prediction to be
                considered a match to a ground truth centroid (default: 50.0).
        """
        super().__init__()
        self.videos = videos
        self.eval_frequency = eval_frequency
        self.match_threshold = match_threshold

    def on_validation_epoch_start(self, trainer, pl_module):
        """Enable prediction collection at the start of validation.

        Skip during sanity check to avoid inference issues.
        """
        if trainer.sanity_checking:
            return
        pl_module._collect_val_predictions = True

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run centroid evaluation and log metrics at end of validation epoch."""
        import numpy as np
        from lightning.pytorch.loggers import WandbLogger

        # Determine if we should run evaluation this epoch (only on rank 0)
        should_evaluate = (
            trainer.current_epoch + 1
        ) % self.eval_frequency == 0 and trainer.is_global_zero

        if should_evaluate:
            # Check if we have predictions
            if not pl_module.val_predictions or not pl_module.val_ground_truth:
                logger.warning(
                    "No predictions collected for centroid epoch-end evaluation"
                )
            else:
                try:
                    metrics = self._compute_metrics(
                        pl_module.val_predictions, pl_module.val_ground_truth, np
                    )

                    # Log to WandB
                    self._log_metrics(trainer, metrics, trainer.current_epoch)

                    logger.info(
                        f"Epoch {trainer.current_epoch} centroid evaluation: "
                        f"precision={metrics['precision']:.4f}, "
                        f"recall={metrics['recall']:.4f}, "
                        f"dist_avg={metrics['dist_avg']:.2f}px"
                    )

                except Exception as e:
                    logger.warning(f"Centroid epoch-end evaluation failed: {e}")

        # Cleanup - all ranks reset the flag, rank 0 clears the lists
        pl_module._collect_val_predictions = False
        if trainer.is_global_zero:
            pl_module.val_predictions = []
            pl_module.val_ground_truth = []

        # Sync all processes - barrier must be reached by ALL ranks
        trainer.strategy.barrier()

    def _compute_metrics(self, predictions: list, ground_truth: list, np) -> dict:
        """Compute centroid-specific metrics.

        Args:
            predictions: List of prediction dicts with "pred_peaks" key.
            ground_truth: List of ground truth dicts with "gt_instances" key.
            np: NumPy module.

        Returns:
            Dictionary of computed metrics.
        """
        all_distances = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Group predictions and GT by frame
        pred_by_frame = {}
        for pred in predictions:
            key = (pred["video_idx"], pred["frame_idx"])
            if key not in pred_by_frame:
                pred_by_frame[key] = []
            # pred_peaks shape: (n_inst, 1, 2) -> extract centroids as (n_inst, 2)
            centroids = pred["pred_peaks"].reshape(-1, 2)
            # Filter out NaN centroids
            valid_mask = ~np.isnan(centroids).any(axis=1)
            pred_by_frame[key].append(centroids[valid_mask])

        gt_by_frame = {}
        for gt in ground_truth:
            key = (gt["video_idx"], gt["frame_idx"])
            if key not in gt_by_frame:
                gt_by_frame[key] = []
            # gt_instances shape: (n_inst, 1, 2) -> extract centroids as (n_inst, 2)
            centroids = gt["gt_instances"].reshape(-1, 2)
            # Filter out NaN centroids
            valid_mask = ~np.isnan(centroids).any(axis=1)
            gt_by_frame[key].append(centroids[valid_mask])

        # Process each frame
        all_frames = set(pred_by_frame.keys()) | set(gt_by_frame.keys())
        for frame_key in all_frames:
            # Concatenate all predictions for this frame
            if frame_key in pred_by_frame:
                frame_preds = np.concatenate(pred_by_frame[frame_key], axis=0)
            else:
                frame_preds = np.zeros((0, 2))

            # Concatenate all GT for this frame
            if frame_key in gt_by_frame:
                frame_gt = np.concatenate(gt_by_frame[frame_key], axis=0)
            else:
                frame_gt = np.zeros((0, 2))

            # Match predictions to ground truth
            matched_pred, matched_gt, unmatched_pred, unmatched_gt = match_centroids(
                frame_preds, frame_gt, max_distance=self.match_threshold
            )

            # Compute distances for matched pairs
            if len(matched_pred) > 0:
                matched_pred_points = frame_preds[matched_pred]
                matched_gt_points = frame_gt[matched_gt]
                distances = np.linalg.norm(
                    matched_pred_points - matched_gt_points, axis=1
                )
                all_distances.extend(distances.tolist())

            # Update counts
            total_tp += len(matched_pred)
            total_fp += len(unmatched_pred)
            total_fn += len(unmatched_gt)

        # Compute aggregate metrics
        all_distances = np.array(all_distances)

        # Distance metrics (only if we have matches)
        if len(all_distances) > 0:
            dist_avg = float(np.mean(all_distances))
            dist_median = float(np.median(all_distances))
            dist_p90 = float(np.percentile(all_distances, 90))
            dist_p95 = float(np.percentile(all_distances, 95))
            dist_max = float(np.max(all_distances))
        else:
            dist_avg = dist_median = dist_p90 = dist_p95 = dist_max = float("nan")

        # Detection metrics
        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "dist_avg": dist_avg,
            "dist_median": dist_median,
            "dist_p90": dist_p90,
            "dist_p95": dist_p95,
            "dist_max": dist_max,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_true_positives": total_tp,
            "n_false_positives": total_fp,
            "n_false_negatives": total_fn,
            "n_total_predictions": total_tp + total_fp,
            "n_total_ground_truth": total_tp + total_fn,
        }

    def _log_metrics(self, trainer, metrics: dict, epoch: int):
        """Log centroid evaluation metrics to WandB."""
        import numpy as np
        from lightning.pytorch.loggers import WandbLogger

        # Get WandB logger
        wandb_logger = None
        for log in trainer.loggers:
            if isinstance(log, WandbLogger):
                wandb_logger = log
                break

        if wandb_logger is None:
            return

        log_dict = {"epoch": epoch}

        # Distance metrics (with NaN handling)
        if not np.isnan(metrics["dist_avg"]):
            log_dict["eval/val/centroid_dist_avg"] = metrics["dist_avg"]
        if not np.isnan(metrics["dist_median"]):
            log_dict["eval/val/centroid_dist_median"] = metrics["dist_median"]
        if not np.isnan(metrics["dist_p90"]):
            log_dict["eval/val/centroid_dist_p90"] = metrics["dist_p90"]
        if not np.isnan(metrics["dist_p95"]):
            log_dict["eval/val/centroid_dist_p95"] = metrics["dist_p95"]
        if not np.isnan(metrics["dist_max"]):
            log_dict["eval/val/centroid_dist_max"] = metrics["dist_max"]

        # Detection metrics
        log_dict["eval/val/centroid_precision"] = metrics["precision"]
        log_dict["eval/val/centroid_recall"] = metrics["recall"]
        log_dict["eval/val/centroid_f1"] = metrics["f1"]

        # Counts
        log_dict["eval/val/centroid_n_tp"] = metrics["n_true_positives"]
        log_dict["eval/val/centroid_n_fp"] = metrics["n_false_positives"]
        log_dict["eval/val/centroid_n_fn"] = metrics["n_false_negatives"]

        wandb_logger.experiment.log(log_dict, commit=False)

        # Update best metrics in summary
        for key, value in log_dict.items():
            if key == "epoch":
                continue
            summary_key = f"best/{key}"
            current_best = wandb_logger.experiment.summary.get(summary_key)
            # For distance metrics, lower is better; for others, higher is better
            is_distance = "dist" in key
            if current_best is None:
                wandb_logger.experiment.summary[summary_key] = value
            elif is_distance and value < current_best:
                wandb_logger.experiment.summary[summary_key] = value
            elif not is_distance and value > current_best:
                wandb_logger.experiment.summary[summary_key] = value


class EmbeddingEvaluationCallback(Callback):
    """Per-epoch retrieval evaluation for the ``embedding`` model type (SPEC §8).

    Mirrors :class:`SegmentationEvaluationCallback`'s lifecycle: flips
    ``pl_module._collect_val_predictions`` on at validation start so the embedding
    ``validation_step`` collects per-crop ``{"embedding": vec}`` + ``{"label": id}``,
    then computes retrieval (rank-1 / mAP), verification (ROC-AUC / EER) and cosine-kNN
    accuracy over the val set (leave-self-out gallery == query).

    Crucially it logs the selected metric BOTH via ``pl_module.log`` (so it lands in
    ``trainer.callback_metrics`` for ``ModelCheckpoint`` / ``EarlyStopping`` to select
    on a retrieval metric, NOT ``val/loss``) AND to wandb. The selection scalar is
    broadcast from rank 0 so every rank logs the same value (no DDP deadlock).

    Attributes:
        eval_frequency: Run evaluation every N epochs (default: 1).
        select_metric: Metric to log for checkpoint selection (rank1|mAP|auc|knn_acc).
        knn_k: k for the cosine-kNN accuracy.
    """

    def __init__(
        self, eval_frequency: int = 1, select_metric: str = "rank1", knn_k: int = 7
    ):
        """Initialize the callback."""
        super().__init__()
        self.eval_frequency = eval_frequency
        self.select_metric = select_metric
        self.knn_k = knn_k

    def _get_wandb_logger(self, trainer):
        from lightning.pytorch.loggers import WandbLogger

        for log in trainer.loggers:
            if isinstance(log, WandbLogger):
                return log
        return None

    def on_validation_epoch_start(self, trainer, pl_module):
        """Enable per-crop embedding collection (skip sanity check)."""
        if trainer.sanity_checking:
            return
        pl_module._collect_val_predictions = True

    def _compute_metrics(self, predictions: list, ground_truth: list) -> dict:
        """Leave-self-out retrieval/verification/kNN over the collected val embeddings."""
        import numpy as np
        from sleap_nn.evaluation import verification_metrics

        emb = np.stack([p["embedding"].numpy() for p in predictions]).astype(np.float64)
        y = np.asarray([g["label"] for g in ground_truth])
        emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        n = len(emb)
        sim = emb @ emb.T
        np.fill_diagonal(sim, -np.inf)  # leave-self-out (self sorts to the very end)
        order = np.argsort(-sim, axis=1)[:, : n - 1]  # drop the self slot
        ranked = y[order]

        rank1 = float(np.mean(ranked[:, 0] == y))
        aps = []
        for i in range(n):
            rel = (ranked[i] == y[i]).astype(float)
            if rel.sum() == 0:
                continue
            csum = np.cumsum(rel)
            prec = csum / np.arange(1, len(rel) + 1)
            aps.append((prec * rel).sum() / rel.sum())
        mAP = float(np.mean(aps)) if aps else 0.0

        # kNN accuracy (leave-self-out): top-k excluding self.
        k = min(self.knn_k, n - 1)
        idx = order[:, :k]
        nn_y = y[idx]
        nn_s = np.take_along_axis(sim, idx, 1)
        nclass = int(y.max()) + 1
        votes = np.zeros((n, nclass))
        for c in range(nclass):
            votes[:, c] = (nn_s * (nn_y == c)).sum(1)
        knn_acc = float(np.mean(votes.argmax(1) == y))

        ver = verification_metrics(emb, y, emb, y)
        return {
            "rank1": round(rank1, 4),
            "mAP": round(mAP, 4),
            "auc": ver["auc"],
            "eer": ver["eer"],
            "knn_acc": round(knn_acc, 4),
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute retrieval metrics; log to callback_metrics (selection) + wandb."""
        should_evaluate = (
            trainer.current_epoch + 1
        ) % self.eval_frequency == 0 and trainer.is_global_zero

        metrics = None
        if should_evaluate:
            if not pl_module.val_predictions or not pl_module.val_ground_truth:
                logger.warning("No embeddings collected for embedding evaluation.")
            else:
                try:
                    metrics = self._compute_metrics(
                        pl_module.val_predictions, pl_module.val_ground_truth
                    )
                    logger.info(
                        f"Epoch {trainer.current_epoch} embedding eval: "
                        f"rank1={metrics['rank1']:.4f} mAP={metrics['mAP']:.4f} "
                        f"auc={metrics['auc']} knn_acc={metrics['knn_acc']:.4f}"
                    )
                    wandb_logger = self._get_wandb_logger(trainer)
                    if wandb_logger is not None:
                        log_dict = {"epoch": trainer.current_epoch}
                        log_dict.update(
                            {f"eval/val/{k}": v for k, v in metrics.items()}
                        )
                        wandb_logger.experiment.log(log_dict, commit=False)
                except Exception as e:
                    logger.warning(f"Embedding epoch-end evaluation failed: {e}")

        # Broadcast the selection scalar so ALL ranks log the SAME value into
        # callback_metrics (ModelCheckpoint/EarlyStopping monitor reads it). Logging
        # only on rank0 with sync_dist would deadlock.
        sel = (
            float(metrics[self.select_metric])
            if (metrics is not None and self.select_metric in metrics)
            else float("nan")
        )
        sel = trainer.strategy.broadcast(sel, src=0)
        for k in ("rank1", "mAP", "auc", "eer", "knn_acc"):
            v = (
                float(metrics[k])
                if (metrics is not None and k in metrics)
                else float("nan")
            )
            v = trainer.strategy.broadcast(v, src=0)
            pl_module.log(
                f"eval/val/{k}", v, on_epoch=True, sync_dist=False, rank_zero_only=False
            )

        pl_module._collect_val_predictions = False
        if trainer.is_global_zero:
            pl_module.val_predictions = []
            pl_module.val_ground_truth = []
        trainer.strategy.barrier()
