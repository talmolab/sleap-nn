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
        if trainer.is_global_zero:
            if not self.initialized:
                self._init_file()

            metrics = trainer.callback_metrics
            log_data = {}
            for key in self.keys:
                if key == "epoch":
                    log_data["epoch"] = trainer.current_epoch
                elif key == "learning_rate":
                    # Handle both direct logging and LearningRateMonitor format (lr-*)
                    value = metrics.get(key, None)
                    if value is None:
                        # Look for lr-* keys from LearningRateMonitor
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
            if wandb_logger is None:
                return  # No wandb logger, skip visualization logging

            # Get visualization data
            train_data = self.train_viz_fn()
            val_data = self.val_viz_fn()

            # Render and log for each enabled mode
            # Use the logger's experiment to let Lightning manage step tracking
            log_dict = {}
            for mode_name, renderer in self.renderers.items():
                suffix = "" if mode_name == "direct" else f"_{mode_name}"
                train_img = renderer.render(train_data, caption=f"Train Epoch {epoch}")
                val_img = renderer.render(val_data, caption=f"Val Epoch {epoch}")
                log_dict[f"train_predictions{suffix}"] = train_img
                log_dict[f"val_predictions{suffix}"] = val_img

            if log_dict:
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
                wandb_logger.experiment.log({"predictions_table": table}, commit=False)

        # Sync all processes
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
            if wandb_logger is None:
                return  # No wandb logger, skip visualization logging

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
                train_img = renderer.render(train_data, caption=f"Train Epoch {epoch}")
                val_img = renderer.render(val_data, caption=f"Val Epoch {epoch}")
                log_dict[f"train_predictions{suffix}"] = train_img
                log_dict[f"val_predictions{suffix}"] = val_img

            # Render PAFs (always use matplotlib/direct for PAFs)
            from io import BytesIO
            import matplotlib.pyplot as plt
            from PIL import Image

            train_pafs_fig = self._mpl_renderer.render_pafs(train_pafs_data)
            buf = BytesIO()
            train_pafs_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(train_pafs_fig)
            train_pafs_pil = Image.open(buf)
            log_dict["train_pafs"] = wandb.Image(
                train_pafs_pil, caption=f"Train PAFs Epoch {epoch}"
            )

            val_pafs_fig = self._mpl_renderer.render_pafs(val_pafs_data)
            buf = BytesIO()
            val_pafs_fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            plt.close(val_pafs_fig)
            val_pafs_pil = Image.open(buf)
            log_dict["val_pafs"] = wandb.Image(
                val_pafs_pil, caption=f"Val PAFs Epoch {epoch}"
            )

            if log_dict:
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
                    columns=["Epoch", "Train", "Validation", "Train PAFs", "Val PAFs"],
                    data=[
                        [
                            epoch,
                            train_img,
                            val_img,
                            log_dict["train_pafs"],
                            log_dict["val_pafs"],
                        ]
                    ],
                )
                wandb_logger.experiment.log({"predictions_table": table}, commit=False)

        # Sync all processes
        trainer.strategy.barrier()


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
            figure.savefig(figure_path, format="png", pad_inches=0)
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

        # Sync all processes after ZMQ operations
        trainer.strategy.barrier()

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
        if trainer.is_global_zero:
            logs = trainer.callback_metrics
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
        if trainer.is_global_zero:
            logs = trainer.callback_metrics
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
