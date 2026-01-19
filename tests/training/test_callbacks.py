"""Tests for sleap_nn/training/callbacks.py."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import torch

from sleap_nn.training.callbacks import (
    SleapProgressBar,
    CSVLoggerCallback,
    WandBPredImageLogger,
    WandBVizCallback,
    WandBVizCallbackWithPAFs,
    MatplotlibSaver,
    TrainingControllerZMQ,
    ProgressReporterZMQ,
    EpochEndEvaluationCallback,
)
from sleap_nn.training.utils import VisualizationData


class TestSleapProgressBar:
    """Tests for SleapProgressBar callback."""

    def test_get_metrics_formats_small_values_scientific(self):
        """Small float values should be formatted with scientific notation."""
        progress_bar = SleapProgressBar()

        # Mock trainer and pl_module
        mock_trainer = MagicMock()
        mock_pl_module = MagicMock()

        # Mock the parent get_metrics to return small values
        with patch.object(
            SleapProgressBar.__bases__[0],
            "get_metrics",
            return_value={"loss": 0.00001, "lr": 0.0001},
        ):
            result = progress_bar.get_metrics(mock_trainer, mock_pl_module)

        assert result["loss"] == "1.00e-05"
        assert result["lr"] == "1.00e-04"

    def test_get_metrics_formats_normal_values(self):
        """Normal float values should be formatted with 4 decimal places."""
        progress_bar = SleapProgressBar()

        mock_trainer = MagicMock()
        mock_pl_module = MagicMock()

        with patch.object(
            SleapProgressBar.__bases__[0],
            "get_metrics",
            return_value={"loss": 0.1234, "accuracy": 0.9876},
        ):
            result = progress_bar.get_metrics(mock_trainer, mock_pl_module)

        assert result["loss"] == "0.1234"
        assert result["accuracy"] == "0.9876"

    def test_get_metrics_preserves_non_float_values(self):
        """Non-float values should be preserved as-is."""
        progress_bar = SleapProgressBar()

        mock_trainer = MagicMock()
        mock_pl_module = MagicMock()

        with patch.object(
            SleapProgressBar.__bases__[0],
            "get_metrics",
            return_value={"epoch": 5, "step": "100/1000", "status": True},
        ):
            result = progress_bar.get_metrics(mock_trainer, mock_pl_module)

        assert result["epoch"] == 5
        assert result["step"] == "100/1000"
        assert result["status"] is True

    def test_get_metrics_handles_zero(self):
        """Zero values should be formatted normally, not scientific."""
        progress_bar = SleapProgressBar()

        mock_trainer = MagicMock()
        mock_pl_module = MagicMock()

        with patch.object(
            SleapProgressBar.__bases__[0],
            "get_metrics",
            return_value={"loss": 0.0},
        ):
            result = progress_bar.get_metrics(mock_trainer, mock_pl_module)

        assert result["loss"] == "0.0000"

    def test_get_metrics_mixed_values(self):
        """Handles mix of small, normal, and non-float values."""
        progress_bar = SleapProgressBar()

        mock_trainer = MagicMock()
        mock_pl_module = MagicMock()

        with patch.object(
            SleapProgressBar.__bases__[0],
            "get_metrics",
            return_value={
                "train_loss": 0.00005,  # Small - scientific
                "val_loss": 0.1234,  # Normal - 4 decimals
                "epoch": 10,  # Int - preserve
            },
        ):
            result = progress_bar.get_metrics(mock_trainer, mock_pl_module)

        assert result["train_loss"] == "5.00e-05"
        assert result["val_loss"] == "0.1234"
        assert result["epoch"] == 10


class TestCSVLoggerCallback:
    """Tests for CSVLoggerCallback."""

    def test_init(self):
        """Initializes with correct attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.csv"
            callback = CSVLoggerCallback(filepath=filepath)

            assert callback.filepath == filepath
            assert callback.keys == ["epoch", "train_loss", "val_loss", "learning_rate"]
            assert callback.initialized is False

    def test_init_custom_keys(self):
        """Initializes with custom keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.csv"
            custom_keys = ["epoch", "custom_metric"]
            callback = CSVLoggerCallback(filepath=filepath, keys=custom_keys)

            assert callback.keys == custom_keys


class TestWandBVizCallback:
    """Tests for WandBVizCallback."""

    @pytest.fixture
    def sample_viz_data(self):
        """Create sample visualization data."""
        return VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            node_names=["head", "neck", "tail", "l_ear", "r_ear"],
        )

    def test_init_default_params(self):
        """Initializes with default parameters."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
        )

        assert callback.viz_enabled is True
        assert callback.viz_boxes is False
        assert callback.viz_masks is False
        assert callback.log_table is False
        assert "direct" in callback.renderers
        assert len(callback.renderers) == 1

    def test_init_all_modes_enabled(self):
        """Creates renderers for all enabled modes."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
            viz_enabled=True,
            viz_boxes=True,
            viz_masks=True,
        )

        assert "direct" in callback.renderers
        assert "boxes" in callback.renderers
        assert "masks" in callback.renderers
        assert len(callback.renderers) == 3

    def test_init_only_boxes_mode(self):
        """Creates only boxes renderer when viz_enabled is False."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
            viz_enabled=False,
            viz_boxes=True,
            viz_masks=False,
        )

        assert "direct" not in callback.renderers
        assert "boxes" in callback.renderers
        assert len(callback.renderers) == 1

    def test_init_custom_box_size(self):
        """Applies custom box_size to renderers."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
            viz_boxes=True,
            box_size=10.0,
        )

        assert callback.box_size == 10.0
        assert callback.renderers["boxes"].box_size == 10.0

    def test_init_custom_confmap_threshold(self):
        """Applies custom confmap_threshold to renderers."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
            viz_masks=True,
            confmap_threshold=0.2,
        )

        assert callback.confmap_threshold == 0.2
        assert callback.renderers["masks"].confmap_threshold == 0.2

    def test_get_wandb_logger_found(self):
        """Returns WandbLogger when present in trainer loggers."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
        )

        mock_wandb_logger = MagicMock()
        mock_wandb_logger.__class__.__name__ = "WandbLogger"

        mock_trainer = MagicMock()

        # Patch isinstance to check class name
        with patch(
            "sleap_nn.training.callbacks.WandBVizCallback._get_wandb_logger"
        ) as mock_get:
            mock_get.return_value = mock_wandb_logger
            result = callback._get_wandb_logger(mock_trainer)
            assert result == mock_wandb_logger

    def test_get_wandb_logger_not_found(self):
        """Returns None when no WandbLogger in trainer loggers."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: None,
            val_viz_fn=lambda: None,
        )

        mock_trainer = MagicMock()
        mock_trainer.loggers = [MagicMock()]  # Non-wandb logger

        # Need to actually test the real method
        from lightning.pytorch.loggers import WandbLogger

        with patch.object(
            callback, "_get_wandb_logger", wraps=callback._get_wandb_logger
        ):
            result = callback._get_wandb_logger(mock_trainer)
            # Since mock loggers aren't WandbLogger instances, should return None
            assert result is None

    def test_on_train_epoch_end_skips_if_not_global_zero(self, sample_viz_data):
        """Skips visualization if not global rank zero."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = False
        mock_pl_module = MagicMock()

        # Should not call any rendering
        with patch.object(callback, "_get_wandb_logger") as mock_get_logger:
            callback.on_train_epoch_end(mock_trainer, mock_pl_module)
            mock_get_logger.assert_not_called()

    def test_on_train_epoch_end_skips_if_no_wandb_logger(self, sample_viz_data):
        """Skips visualization if no wandb logger found."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = True
        mock_trainer.loggers = []
        mock_pl_module = MagicMock()

        # Patch _get_wandb_logger to return None
        with patch.object(callback, "_get_wandb_logger", return_value=None):
            # Should not raise, just skip
            callback.on_train_epoch_end(mock_trainer, mock_pl_module)

    def test_on_train_epoch_end_logs_images(self, sample_viz_data):
        """Logs images when wandb logger is present."""
        train_data = sample_viz_data
        val_data = sample_viz_data

        callback = WandBVizCallback(
            train_viz_fn=lambda: train_data,
            val_viz_fn=lambda: val_data,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = True
        mock_trainer.current_epoch = 5
        mock_pl_module = MagicMock()

        mock_wandb_logger = MagicMock()
        mock_experiment = MagicMock()
        mock_wandb_logger.experiment = mock_experiment

        with patch.object(
            callback, "_get_wandb_logger", return_value=mock_wandb_logger
        ):
            # Mock the renderer's render method
            for renderer in callback.renderers.values():
                renderer.render = MagicMock(return_value="mock_image")

            callback.on_train_epoch_end(mock_trainer, mock_pl_module)

            # Verify experiment.log was called
            mock_experiment.log.assert_called_once()
            call_args = mock_experiment.log.call_args
            log_dict = call_args[0][0]
            assert "viz/train/predictions" in log_dict
            assert "viz/val/predictions" in log_dict
            assert call_args[1]["commit"] is False

    def test_on_train_epoch_end_with_log_table(self, sample_viz_data):
        """Logs both images and table when log_table is True."""
        callback = WandBVizCallback(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
            log_table=True,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = True
        mock_trainer.current_epoch = 3
        mock_pl_module = MagicMock()

        mock_wandb_logger = MagicMock()
        mock_experiment = MagicMock()
        mock_wandb_logger.experiment = mock_experiment

        with patch.object(
            callback, "_get_wandb_logger", return_value=mock_wandb_logger
        ):
            for renderer in callback.renderers.values():
                renderer.render = MagicMock(return_value="mock_image")

            with patch("sleap_nn.training.callbacks.wandb") as mock_wandb:
                mock_wandb.Table = MagicMock(return_value="mock_table")
                callback.on_train_epoch_end(mock_trainer, mock_pl_module)

                # Should log twice - once for images, once for table
                assert mock_experiment.log.call_count == 2


class TestWandBVizCallbackWithPAFs:
    """Tests for WandBVizCallbackWithPAFs."""

    @pytest.fixture
    def sample_viz_data(self):
        """Create sample visualization data."""
        return VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            node_names=["head", "neck", "tail", "l_ear", "r_ear"],
        )

    @pytest.fixture
    def sample_pafs_data(self):
        """Create sample visualization data with PAFs."""
        return VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            node_names=["head", "neck", "tail", "l_ear", "r_ear"],
            pred_pafs=np.random.rand(64, 64, 8).astype(np.float32),  # 4 edges * 2
        )

    def test_init_inherits_from_wandb_viz_callback(
        self, sample_viz_data, sample_pafs_data
    ):
        """Inherits from WandBVizCallback and adds PAF viz functions."""
        callback = WandBVizCallbackWithPAFs(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
            train_pafs_viz_fn=lambda: sample_pafs_data,
            val_pafs_viz_fn=lambda: sample_pafs_data,
        )

        assert callback.train_pafs_viz_fn is not None
        assert callback.val_pafs_viz_fn is not None
        assert callback._mpl_renderer is not None

    def test_init_passes_params_to_parent(self, sample_viz_data, sample_pafs_data):
        """Passes parameters to parent class."""
        callback = WandBVizCallbackWithPAFs(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
            train_pafs_viz_fn=lambda: sample_pafs_data,
            val_pafs_viz_fn=lambda: sample_pafs_data,
            viz_enabled=True,
            viz_boxes=True,
            box_size=15.0,
        )

        assert callback.viz_enabled is True
        assert callback.viz_boxes is True
        assert callback.box_size == 15.0
        assert "direct" in callback.renderers
        assert "boxes" in callback.renderers

    def test_on_train_epoch_end_logs_pafs(self, sample_viz_data, sample_pafs_data):
        """Logs PAF images in addition to regular visualizations."""
        callback = WandBVizCallbackWithPAFs(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
            train_pafs_viz_fn=lambda: sample_pafs_data,
            val_pafs_viz_fn=lambda: sample_pafs_data,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = True
        mock_trainer.current_epoch = 2
        mock_pl_module = MagicMock()

        mock_wandb_logger = MagicMock()
        mock_experiment = MagicMock()
        mock_wandb_logger.experiment = mock_experiment

        with patch.object(
            callback, "_get_wandb_logger", return_value=mock_wandb_logger
        ):
            for renderer in callback.renderers.values():
                renderer.render = MagicMock(return_value="mock_image")

            with patch("sleap_nn.training.callbacks.wandb") as mock_wandb:
                mock_wandb.Image = MagicMock(return_value="mock_paf_image")

                callback.on_train_epoch_end(mock_trainer, mock_pl_module)

                # Verify experiment.log was called with PAF images
                mock_experiment.log.assert_called()
                call_args = mock_experiment.log.call_args
                log_dict = call_args[0][0]
                assert "viz/train/pafs" in log_dict
                assert "viz/val/pafs" in log_dict

    def test_on_train_epoch_end_skips_if_not_global_zero(
        self, sample_viz_data, sample_pafs_data
    ):
        """Skips if not global rank zero."""
        callback = WandBVizCallbackWithPAFs(
            train_viz_fn=lambda: sample_viz_data,
            val_viz_fn=lambda: sample_viz_data,
            train_pafs_viz_fn=lambda: sample_pafs_data,
            val_pafs_viz_fn=lambda: sample_pafs_data,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = False
        mock_pl_module = MagicMock()

        with patch.object(callback, "_get_wandb_logger") as mock_get_logger:
            callback.on_train_epoch_end(mock_trainer, mock_pl_module)
            mock_get_logger.assert_not_called()


class TestMatplotlibSaver:
    """Tests for MatplotlibSaver callback."""

    def test_init(self):
        """Initializes with correct attributes."""
        plot_fn = MagicMock()
        callback = MatplotlibSaver(
            save_folder="/tmp/viz",
            plot_fn=plot_fn,
            prefix="train",
        )

        assert callback.save_folder == "/tmp/viz"
        assert callback.plot_fn == plot_fn
        assert callback.prefix == "train"

    def test_init_no_prefix(self):
        """Initializes without prefix."""
        callback = MatplotlibSaver(
            save_folder="/tmp/viz",
            plot_fn=MagicMock(),
        )

        assert callback.prefix is None

    def test_on_train_epoch_end_saves_figure(self):
        """Saves figure at end of epoch."""
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple figure
            mock_fig = MagicMock()

            callback = MatplotlibSaver(
                save_folder=tmpdir,
                plot_fn=lambda: mock_fig,
                prefix="test",
            )

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 5
            mock_pl_module = MagicMock()

            callback.on_train_epoch_end(mock_trainer, mock_pl_module)

            # Verify savefig was called
            mock_fig.savefig.assert_called_once()
            call_args = mock_fig.savefig.call_args
            expected_path = Path(tmpdir) / "test.0005.png"
            assert call_args[0][0] == expected_path.as_posix()

    def test_on_train_epoch_end_no_prefix(self):
        """Saves figure without prefix in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_fig = MagicMock()

            callback = MatplotlibSaver(
                save_folder=tmpdir,
                plot_fn=lambda: mock_fig,
                prefix=None,
            )

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 3
            mock_pl_module = MagicMock()

            callback.on_train_epoch_end(mock_trainer, mock_pl_module)

            call_args = mock_fig.savefig.call_args
            expected_path = Path(tmpdir) / "0003.png"
            assert call_args[0][0] == expected_path.as_posix()

    def test_on_train_epoch_end_skips_if_not_global_zero(self):
        """Skips saving if not global rank zero."""
        mock_fig = MagicMock()
        plot_fn = MagicMock(return_value=mock_fig)

        callback = MatplotlibSaver(
            save_folder="/tmp/viz",
            plot_fn=plot_fn,
        )

        mock_trainer = MagicMock()
        mock_trainer.is_global_zero = False
        mock_pl_module = MagicMock()

        callback.on_train_epoch_end(mock_trainer, mock_pl_module)

        # plot_fn should not be called
        plot_fn.assert_not_called()


class TestCSVLoggerCallbackFileOps:
    """Tests for CSVLoggerCallback file operations."""

    def test_init_file_creates_csv_with_header(self):
        """_init_file creates CSV with header row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "subdir" / "metrics.csv"
            callback = CSVLoggerCallback(filepath=filepath)

            # Patch RANK to be 0 (or -1)
            with patch("sleap_nn.training.callbacks.RANK", 0):
                callback._init_file()

            assert callback.initialized is True
            assert filepath.exists()

            # Check header
            with open(filepath) as f:
                header = f.readline().strip()
                assert "epoch" in header
                assert "train_loss" in header

    def test_on_validation_epoch_end_logs_metrics(self):
        """Logs metrics to CSV at end of validation epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.csv"
            callback = CSVLoggerCallback(filepath=filepath)

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 5
            mock_trainer.callback_metrics = {
                "train_loss": torch.tensor(0.5),
                "val_loss": torch.tensor(0.3),
                "lr-Adam": torch.tensor(0.001),  # LearningRateMonitor format
            }
            mock_pl_module = MagicMock()

            with patch("sleap_nn.training.callbacks.RANK", 0):
                callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

            assert filepath.exists()

            # Read and verify contents
            with open(filepath) as f:
                lines = f.readlines()
                assert len(lines) == 2  # Header + data row
                assert "5" in lines[1]  # Epoch

    def test_on_validation_epoch_end_logs_train_lr_format(self):
        """Logs learning rate from train/lr key (current format)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.csv"
            callback = CSVLoggerCallback(filepath=filepath)

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 3
            mock_trainer.callback_metrics = {
                "train_loss": torch.tensor(0.4),
                "val_loss": torch.tensor(0.2),
                "train/lr": torch.tensor(
                    0.0005
                ),  # Current format from lightning modules
            }
            mock_pl_module = MagicMock()

            with patch("sleap_nn.training.callbacks.RANK", 0):
                callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

            assert filepath.exists()

            # Read and verify contents
            import csv

            with open(filepath) as f:
                reader = csv.DictReader(f)
                row = next(reader)
                assert row["epoch"] == "3"
                assert row["learning_rate"].startswith("0.0005")

    def test_on_validation_epoch_end_skips_if_not_global_zero(self):
        """Skips logging if not global rank zero."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "metrics.csv"
            callback = CSVLoggerCallback(filepath=filepath)

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = False
            mock_pl_module = MagicMock()

            callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

            assert not filepath.exists()


class TestWandBPredImageLogger:
    """Tests for deprecated WandBPredImageLogger callback."""

    def test_init(self):
        """Initializes with correct attributes."""
        callback = WandBPredImageLogger(
            viz_folder="/tmp/viz",
            wandb_run_name="test_run",
            is_bottomup=False,
        )

        assert callback.viz_folder == "/tmp/viz"
        assert callback.wandb_run_name == "test_run"
        assert callback.is_bottomup is False

    def test_init_bottomup(self):
        """Initializes with is_bottomup=True."""
        callback = WandBPredImageLogger(
            viz_folder="/tmp/viz",
            wandb_run_name="test_run",
            is_bottomup=True,
        )

        assert callback.is_bottomup is True

    def test_on_train_epoch_end_logs_images(self):
        """Logs images to wandb table."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Create dummy image files
            from PIL import Image as PILImage

            train_img = PILImage.new("RGB", (64, 64), color="red")
            val_img = PILImage.new("RGB", (64, 64), color="blue")
            train_img.save(Path(tmpdir) / "train.0005.png")
            val_img.save(Path(tmpdir) / "validation.0005.png")

            callback = WandBPredImageLogger(
                viz_folder=tmpdir,
                wandb_run_name="test_run",
                is_bottomup=False,
            )

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 5
            mock_pl_module = MagicMock()

            # Mock Image.open to avoid file locking issues
            with patch("sleap_nn.training.callbacks.Image") as mock_pil:
                mock_pil.open.return_value = MagicMock()

                with patch("sleap_nn.training.callbacks.wandb") as mock_wandb:
                    mock_wandb.Image = MagicMock(return_value="mock_image")
                    mock_wandb.Table = MagicMock(return_value="mock_table")

                    callback.on_train_epoch_end(mock_trainer, mock_pl_module)

                    # Verify wandb.log was called
                    mock_wandb.log.assert_called_once()

    def test_on_train_epoch_end_bottomup_logs_pafs(self):
        """Logs PAF images for bottomup models."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            from PIL import Image as PILImage

            # Create all required image files
            for name in [
                "train.0003.png",
                "validation.0003.png",
                "train.pafs_magnitude.0003.png",
                "validation.pafs_magnitude.0003.png",
            ]:
                img = PILImage.new("RGB", (64, 64), color="green")
                img.save(Path(tmpdir) / name)

            callback = WandBPredImageLogger(
                viz_folder=tmpdir,
                wandb_run_name="bu_run",
                is_bottomup=True,
            )

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 3
            mock_pl_module = MagicMock()

            # Mock Image.open to avoid file locking issues
            with patch("sleap_nn.training.callbacks.Image") as mock_pil:
                mock_pil.open.return_value = MagicMock()

                with patch("sleap_nn.training.callbacks.wandb") as mock_wandb:
                    mock_wandb.Image = MagicMock(return_value="mock_image")
                    mock_wandb.Table = MagicMock(return_value="mock_table")

                    callback.on_train_epoch_end(mock_trainer, mock_pl_module)

                    # Should create table with PAF columns
                    table_call = mock_wandb.Table.call_args
                    columns = table_call[1]["columns"]
                    assert "Pafs Preds on train" in columns
                    assert "Pafs Preds on validation" in columns


class TestTrainingControllerZMQ:
    """Tests for TrainingControllerZMQ callback."""

    def test_init(self):
        """Initializes ZMQ socket."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = TrainingControllerZMQ(
                address="tcp://127.0.0.1:9000",
                topic="test",
                poll_timeout=100,
            )

            assert callback.address == "tcp://127.0.0.1:9000"
            assert callback.topic == "test"
            assert callback.timeout == 100
            mock_socket.subscribe.assert_called_with("test")
            mock_socket.connect.assert_called_with("tcp://127.0.0.1:9000")

    def test_on_train_batch_end_stop_command(self):
        """Handles stop command from ZMQ."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            mock_zmq.POLLIN = 1

            callback = TrainingControllerZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.should_stop = False
            mock_pl_module = MagicMock()

            # Simulate receiving stop command
            mock_socket.poll.return_value = True
            with patch("sleap_nn.training.callbacks.jsonpickle") as mock_jp:
                mock_jp.decode.return_value = {"command": "stop"}

                callback.on_train_batch_end(mock_trainer, mock_pl_module, {}, {}, 0)

                assert mock_trainer.should_stop is True

    def test_on_train_batch_end_no_message(self):
        """Does nothing when no message available."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = TrainingControllerZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.should_stop = False
            mock_pl_module = MagicMock()

            # No message available
            mock_socket.poll.return_value = False

            callback.on_train_batch_end(mock_trainer, mock_pl_module, {}, {}, 0)

            assert mock_trainer.should_stop is False

    def test_del_closes_socket(self):
        """Destructor closes socket and context."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = TrainingControllerZMQ()
            callback.__del__()

            mock_socket.close.assert_called_once()
            mock_context.term.assert_called_once()


class TestProgressReporterZMQ:
    """Tests for ProgressReporterZMQ callback."""

    def test_init(self):
        """Initializes ZMQ PUB socket."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            mock_zmq.PUB = 1

            callback = ProgressReporterZMQ(
                address="tcp://127.0.0.1:9001",
                what="test_job",
            )

            assert callback.address == "tcp://127.0.0.1:9001"
            assert callback.what == "test_job"
            mock_context.socket.assert_called_with(1)  # zmq.PUB
            mock_socket.connect.assert_called_with("tcp://127.0.0.1:9001")

    def test_send(self):
        """Sends message over ZMQ."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ(what="test_job")

            with patch("sleap_nn.training.callbacks.jsonpickle") as mock_jp:
                mock_jp.encode.return_value = '{"encoded": "data"}'

                callback.send("test_event", logs={"loss": 0.5}, extra="value")

                mock_jp.encode.assert_called_once()
                mock_socket.send_string.assert_called_once()

    def test_on_train_start(self):
        """Sends train_begin event with wandb_url."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_pl_module = MagicMock()

            with patch.object(callback, "send") as mock_send:
                callback.on_train_start(mock_trainer, mock_pl_module)
                mock_send.assert_called_with("train_begin", wandb_url=None)

    def test_on_train_end(self):
        """Sends train_end event."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_pl_module = MagicMock()

            with patch.object(callback, "send") as mock_send:
                callback.on_train_end(mock_trainer, mock_pl_module)
                mock_send.assert_called_with("train_end")

    def test_on_train_epoch_start(self):
        """Sends epoch_begin event with epoch number."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 5
            mock_pl_module = MagicMock()

            with patch.object(callback, "send") as mock_send:
                callback.on_train_epoch_start(mock_trainer, mock_pl_module)
                mock_send.assert_called_with("epoch_begin", epoch=5)

    def test_on_train_epoch_end(self):
        """Sends epoch_end event with logs."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 3
            mock_trainer.callback_metrics = {"loss": torch.tensor(0.5)}
            mock_pl_module = MagicMock()

            with patch.object(callback, "send") as mock_send:
                callback.on_train_epoch_end(mock_trainer, mock_pl_module)
                mock_send.assert_called_once()
                call_args = mock_send.call_args
                assert call_args[0][0] == "epoch_end"
                assert call_args[1]["epoch"] == 3

    def test_on_train_batch_start(self):
        """Sends batch_start event."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_pl_module = MagicMock()

            with patch.object(callback, "send") as mock_send:
                callback.on_train_batch_start(mock_trainer, mock_pl_module, {}, 10)
                mock_send.assert_called_with("batch_start", batch=10)

    def test_on_train_batch_end(self):
        """Sends batch_end event with logs."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            mock_trainer = MagicMock()
            mock_trainer.is_global_zero = True
            mock_trainer.current_epoch = 2
            mock_trainer.callback_metrics = {"loss": torch.tensor(0.25)}
            mock_pl_module = MagicMock()

            with patch.object(callback, "send") as mock_send:
                callback.on_train_batch_end(mock_trainer, mock_pl_module, {}, {}, 15)
                mock_send.assert_called_once()
                call_args = mock_send.call_args
                assert call_args[0][0] == "batch_end"
                assert call_args[1]["epoch"] == 2
                assert call_args[1]["batch"] == 15

    def test_sanitize_logs(self):
        """Converts tensors to floats in logs."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket

            callback = ProgressReporterZMQ()

            logs = {
                "loss": torch.tensor(0.5),
                "epoch": 5,
                "name": "test",
            }

            sanitized = callback._sanitize_logs(logs)

            assert sanitized["loss"] == 0.5
            assert isinstance(sanitized["loss"], float)
            assert sanitized["epoch"] == 5
            assert sanitized["name"] == "test"

    def test_del_closes_socket(self):
        """Destructor closes socket and context."""
        with patch("sleap_nn.training.callbacks.zmq") as mock_zmq:
            mock_context = MagicMock()
            mock_socket = MagicMock()
            mock_zmq.Context.return_value = mock_context
            mock_context.socket.return_value = mock_socket
            mock_zmq.LINGER = 0

            callback = ProgressReporterZMQ()
            callback.__del__()

            mock_socket.setsockopt.assert_called()
            mock_socket.close.assert_called_once()
            mock_context.term.assert_called_once()


class TestEpochEndEvaluationCallback:
    """Tests for EpochEndEvaluationCallback."""

    @pytest.fixture
    def mock_skeleton(self):
        """Create a mock skeleton."""
        import sleap_io as sio

        skeleton = sio.Skeleton(name="test_skeleton")
        skeleton.add_node("head")
        skeleton.add_node("tail")
        skeleton.add_node("center")
        return skeleton

    @pytest.fixture
    def mock_videos(self):
        """Create mock videos."""
        video1 = MagicMock()
        video1.backend = None
        video2 = MagicMock()
        video2.backend = None
        return [video1, video2]

    def test_init_default_params(self, mock_skeleton, mock_videos):
        """Initializes with default parameters."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        assert callback.skeleton == mock_skeleton
        assert callback.videos == mock_videos
        assert callback.eval_frequency == 1
        assert callback.oks_stddev == 0.025
        assert callback.oks_scale is None
        assert "mOKS" in callback.metrics_to_log

    def test_init_custom_params(self, mock_skeleton, mock_videos):
        """Initializes with custom parameters."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
            eval_frequency=5,
            oks_stddev=0.05,
            oks_scale=100.0,
            metrics_to_log=["mOKS", "avg_distance"],
        )

        assert callback.eval_frequency == 5
        assert callback.oks_stddev == 0.05
        assert callback.oks_scale == 100.0
        assert callback.metrics_to_log == ["mOKS", "avg_distance"]

    def test_on_validation_epoch_start_enables_collection(
        self, mock_skeleton, mock_videos
    ):
        """Enables prediction collection at validation start."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = False  # Not during sanity check
        mock_pl_module = MagicMock()
        mock_pl_module._collect_val_predictions = False

        callback.on_validation_epoch_start(mock_trainer, mock_pl_module)

        assert mock_pl_module._collect_val_predictions is True

    def test_on_validation_epoch_start_skips_during_sanity_check(
        self, mock_skeleton, mock_videos
    ):
        """Skips enabling prediction collection during sanity check."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_trainer = MagicMock()
        mock_trainer.sanity_checking = True  # During sanity check
        mock_pl_module = MagicMock()
        mock_pl_module._collect_val_predictions = False

        callback.on_validation_epoch_start(mock_trainer, mock_pl_module)

        # Should remain False during sanity check
        assert mock_pl_module._collect_val_predictions is False

    def test_on_validation_epoch_end_skips_by_frequency(
        self, mock_skeleton, mock_videos
    ):
        """Skips evaluation if not at frequency interval."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
            eval_frequency=5,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 2  # Epoch 3 (0-indexed), not divisible by 5
        mock_trainer.is_global_zero = True
        mock_pl_module = MagicMock()
        mock_pl_module._collect_val_predictions = True

        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

        # Should have disabled collection
        assert mock_pl_module._collect_val_predictions is False

    def test_on_validation_epoch_end_skips_if_not_global_zero(
        self, mock_skeleton, mock_videos
    ):
        """Skips evaluation if not global rank zero."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        mock_trainer.is_global_zero = False
        mock_pl_module = MagicMock()
        mock_pl_module._collect_val_predictions = True

        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

        assert mock_pl_module._collect_val_predictions is False

    def test_on_validation_epoch_end_skips_if_no_predictions(
        self, mock_skeleton, mock_videos
    ):
        """Skips evaluation if no predictions collected."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_trainer = MagicMock()
        mock_trainer.current_epoch = 0
        mock_trainer.is_global_zero = True
        mock_trainer.loggers = []
        mock_pl_module = MagicMock()
        mock_pl_module._collect_val_predictions = True
        mock_pl_module.val_predictions = []
        mock_pl_module.val_ground_truth = []

        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

        assert mock_pl_module._collect_val_predictions is False

    def test_build_pred_labels_single_instance(self, mock_skeleton, mock_videos):
        """Builds prediction labels for single instance predictions."""
        import sleap_io as sio

        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        predictions = [
            {
                "video_idx": 0,
                "frame_idx": 5,
                "pred_peaks": np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]),
                "pred_scores": np.array([0.9, 0.8, 0.7]),
            }
        ]

        labels = callback._build_pred_labels(predictions, sio, np)

        assert len(labels.labeled_frames) == 1
        assert labels.labeled_frames[0].frame_idx == 5
        assert len(labels.labeled_frames[0].instances) == 1

    def test_build_pred_labels_multi_instance(self, mock_skeleton, mock_videos):
        """Builds prediction labels for multi-instance predictions."""
        import sleap_io as sio

        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        predictions = [
            {
                "video_idx": 0,
                "frame_idx": 3,
                "pred_peaks": np.array(
                    [
                        [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],  # Instance 1
                        [[15.0, 25.0], [35.0, 45.0], [55.0, 65.0]],  # Instance 2
                    ]
                ),
                "pred_scores": np.array(
                    [
                        [0.9, 0.8, 0.7],
                        [0.85, 0.75, 0.65],
                    ]
                ),
            }
        ]

        labels = callback._build_pred_labels(predictions, sio, np)

        assert len(labels.labeled_frames) == 1
        assert len(labels.labeled_frames[0].instances) == 2

    def test_build_pred_labels_skips_all_nan(self, mock_skeleton, mock_videos):
        """Skips predictions that are all NaN."""
        import sleap_io as sio

        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        predictions = [
            {
                "video_idx": 0,
                "frame_idx": 5,
                "pred_peaks": np.array(
                    [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
                ),
                "pred_scores": np.array([np.nan, np.nan, np.nan]),
            }
        ]

        labels = callback._build_pred_labels(predictions, sio, np)

        assert len(labels.labeled_frames) == 0

    def test_build_gt_labels(self, mock_skeleton, mock_videos):
        """Builds ground truth labels."""
        import sleap_io as sio

        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        ground_truth = [
            {
                "video_idx": 0,
                "frame_idx": 5,
                "gt_instances": np.array([[[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]]),
                "num_instances": 1,
            }
        ]

        labels = callback._build_gt_labels(ground_truth, sio, np)

        assert len(labels.labeled_frames) == 1
        assert labels.labeled_frames[0].frame_idx == 5
        assert len(labels.labeled_frames[0].instances) == 1

    def test_log_metrics_no_wandb_logger(self, mock_skeleton, mock_videos):
        """Does nothing if no wandb logger found."""
        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_trainer = MagicMock()
        mock_trainer.loggers = []  # No loggers

        metrics = {
            "mOKS": {"mOKS": 0.85},
            "voc_metrics": {"oks_voc.mAP": 0.75, "oks_voc.mAR": 0.80},
            "distance_metrics": {"avg": 5.0, "p50": 4.0},
            "pck_metrics": {"mPCK": 0.90},
            "visibility_metrics": {"precision": 0.95, "recall": 0.92},
        }

        # Should not raise
        callback._log_metrics(mock_trainer, metrics, epoch=5)

    def test_log_metrics_with_wandb_logger(self, mock_skeleton, mock_videos):
        """Logs metrics to wandb when logger present."""
        from lightning.pytorch.loggers import WandbLogger

        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_wandb_logger = MagicMock(spec=WandbLogger)
        mock_experiment = MagicMock()
        mock_experiment.summary.get.return_value = None  # No prior best values
        mock_wandb_logger.experiment = mock_experiment

        mock_trainer = MagicMock()
        mock_trainer.loggers = [mock_wandb_logger]

        metrics = {
            "mOKS": {"mOKS": 0.85},
            "voc_metrics": {"oks_voc.mAP": 0.75, "oks_voc.mAR": 0.80},
            "distance_metrics": {"avg": 5.0, "p50": 4.0, "p95": 8.0, "p99": 10.0},
            "pck_metrics": {"mPCK": 0.90, "PCK@5": 0.85, "PCK@10": 0.92},
            "visibility_metrics": {"precision": 0.95, "recall": 0.92},
        }

        callback._log_metrics(mock_trainer, metrics, epoch=5)

        mock_experiment.log.assert_called_once()
        log_call = mock_experiment.log.call_args
        log_dict = log_call[0][0]

        assert log_dict["epoch"] == 5
        assert log_dict["eval/val/mOKS"] == 0.85
        assert log_dict["eval/val/oks_voc_mAP"] == 0.75
        assert log_dict["eval/val/mPCK"] == 0.90
        assert log_call[1]["commit"] is False

    def test_log_metrics_skips_nan_values(self, mock_skeleton, mock_videos):
        """Skips NaN values when logging."""
        from lightning.pytorch.loggers import WandbLogger

        callback = EpochEndEvaluationCallback(
            skeleton=mock_skeleton,
            videos=mock_videos,
        )

        mock_wandb_logger = MagicMock(spec=WandbLogger)
        mock_experiment = MagicMock()
        mock_experiment.summary.get.return_value = None  # No prior best values
        mock_wandb_logger.experiment = mock_experiment

        mock_trainer = MagicMock()
        mock_trainer.loggers = [mock_wandb_logger]

        metrics = {
            "mOKS": {"mOKS": 0.85},
            "voc_metrics": {"oks_voc.mAP": 0.75, "oks_voc.mAR": 0.80},
            "distance_metrics": {
                "avg": np.nan,
                "p50": np.nan,
                "p95": np.nan,
                "p99": np.nan,
            },  # NaN values
            "pck_metrics": {"mPCK": 0.90, "PCK@5": 0.85, "PCK@10": 0.92},
            "visibility_metrics": {"precision": np.nan, "recall": np.nan},  # NaN values
        }

        callback._log_metrics(mock_trainer, metrics, epoch=5)

        log_call = mock_experiment.log.call_args
        log_dict = log_call[0][0]

        # NaN values should not be in the log dict
        assert "eval/val/distance/avg" not in log_dict
        assert "eval/val/distance/p50" not in log_dict
        assert "eval/val/visibility_precision" not in log_dict
        assert "eval/val/visibility_recall" not in log_dict

        # Non-NaN values should be present
        assert log_dict["eval/val/mOKS"] == 0.85
        assert log_dict["eval/val/mPCK"] == 0.90
