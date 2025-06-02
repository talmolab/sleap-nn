"""Custom Callback modules for Lightning Trainer."""

import zmq
import jsonpickle
from typing import Callable, Optional
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import wandb


class WandBPredImageLogger(Callback):
    """Callback for writing image predictions to wandb.

    Attributes:
        viz_folder: Path to viz directory.
        wandb_run_name: WandB run name.
    """

    def __init__(
        self,
        viz_folder: str,
        wandb_run_name: str,
        is_bottomup: bool = False,
    ):
        self.viz_folder = viz_folder
        self.wandb_run_name = wandb_run_name
        self.is_bottomup = is_bottomup
        # Callback initialization
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch."""
        epoch_num = trainer.current_epoch
        train_img_path = (
            Path(self.viz_folder) / f"train.{epoch_num:04d}.png"
        ).as_posix()
        val_img_path = (
            Path(self.viz_folder) / f"validation.{epoch_num:04d}.png"
        ).as_posix()
        train_img = Image.open(train_img_path)
        val_img = Image.open(val_img_path)

        column_names = ["Run name", "Epoch", "Preds on train", "Preds on validation"]
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
                                / f"train_pafs_magnitude.{epoch_num:04d}.png"
                            ).as_posix()
                        )
                    ),
                    wandb.Image(
                        Image.open(
                            (
                                Path(self.viz_folder)
                                / f"validation_pafs_magnitude.{epoch_num:04d}.png"
                            ).as_posix()
                        )
                    ),
                ]
            ]
        table = wandb.Table(columns=column_names, data=data)
        wandb.log({f"{self.wandb_run_name}": table})


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

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        if self.socket.poll(self.timeout, zmq.POLLIN):
            msg = jsonpickle.decode(self.socket.recv_string())
            logger.info(f"Received control message: {msg}")

            # Stop training
            if msg.get("command") == "stop":
                trainer.should_stop = True

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
        self.send("train_begin")

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training process."""
        self.send("train_end")

    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the beginning of each epoch."""
        self.send("epoch_begin", epoch=trainer.current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each epoch."""
        logs = trainer.callback_metrics
        self.send(
            "epoch_end", epoch=trainer.current_epoch, logs=self._sanitize_logs(logs)
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Called at the beginning of each training batch."""
        self.send("batch_start", batch=batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training batch."""
        logs = trainer.callback_metrics
        self.send(
            "batch_end",
            epoch=trainer.current_epoch,
            batch=batch_idx,
            logs=self._sanitize_logs(logs),
        )

    def _sanitize_logs(self, logs):
        """Convert any torch tensors to Python floats for serialization."""
        return {
            k: float(v.item()) if hasattr(v, "item") else v for k, v in logs.items()
        }
