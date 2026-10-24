"""Tests for sleap_nn.model_info module."""

import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from sleap_nn.cli import cli
from sleap_nn.model_info import (
    _format_file_size,
    _format_model_type,
    _format_param_count,
    _load_training_log,
    print_model_info,
)

ASSETS = Path(__file__).parent / "assets" / "model_ckpts"
MODEL_WITH_METRICS = ASSETS / "single_instance_with_metrics"
MODEL_NO_METRICS = ASSETS / "minimal_instance_single_instance"


class TestFormatHelpers:
    def test_format_param_count(self):
        assert _format_param_count(1_327_994) == "1.3M"
        assert _format_param_count(18_200) == "18.2K"
        assert _format_param_count(500) == "500"
        assert _format_param_count(None) == "N/A"

    def test_format_model_type(self):
        assert _format_model_type("single_instance") == "Single Instance"
        assert _format_model_type("centered_instance") == "Centered Instance"
        assert _format_model_type("bottomup") == "Bottomup"
        assert _format_model_type(None) == "Unknown"

    def test_format_file_size(self):
        assert _format_file_size(500) == "500 B"
        assert _format_file_size(1_500) == "1.5 KB"
        assert _format_file_size(5_300_000) == "5.3 MB"
        assert _format_file_size(2_100_000_000) == "2.1 GB"


class TestLoadTrainingLog:
    def test_load_existing_log(self):
        stats = _load_training_log(MODEL_WITH_METRICS)
        assert stats is not None
        assert stats["epochs_trained"] == 5
        assert stats["final_train_loss"] is not None
        assert stats["final_val_loss"] is not None
        assert stats["best_val_loss"] is not None
        assert stats["best_val_epoch"] is not None

    def test_load_missing_log(self, tmp_path):
        assert _load_training_log(tmp_path) is None

    def test_load_slash_named_log(self, tmp_path):
        """Current trainer logs use `train/loss` / `val/loss` column names."""
        (tmp_path / "training_log.csv").write_text(
            "epoch,train/loss,val/loss,learning_rate,train/time,val/time\n"
            "0,0.5,0.4,0.001,1.0,0.5\n"
            "1,0.3,0.1,0.001,1.0,0.5\n"
            "2,0.2,0.2,0.0005,1.0,0.5\n"
        )
        stats = _load_training_log(tmp_path)
        assert stats["epochs_trained"] == 3
        assert stats["final_train_loss"] == pytest.approx(0.2)
        assert stats["final_val_loss"] == pytest.approx(0.2)
        assert stats["best_val_loss"] == pytest.approx(0.1)
        assert stats["best_val_epoch"] == 1
        assert stats["final_lr"] == pytest.approx(0.0005)

    def test_slash_names_preferred_over_underscore(self, tmp_path):
        (tmp_path / "training_log.csv").write_text(
            "epoch,train_loss,val_loss,train/loss,val/loss\n" "0,9.0,9.0,0.3,0.2\n"
        )
        stats = _load_training_log(tmp_path)
        assert stats["final_train_loss"] == pytest.approx(0.3)
        assert stats["final_val_loss"] == pytest.approx(0.2)
        assert stats["best_val_loss"] == pytest.approx(0.2)

    def test_slash_named_log_skips_empty_val_rows(self, tmp_path):
        (tmp_path / "training_log.csv").write_text(
            "epoch,train/loss,val/loss,learning_rate\n"
            "0,,0.4,0.001\n"
            "0,0.5,,0.001\n"
            "1,0.3,0.1,0.001\n"
        )
        stats = _load_training_log(tmp_path)
        assert stats["best_val_loss"] == pytest.approx(0.1)
        assert stats["best_val_epoch"] == 1

    def test_unparseable_loss_value_falls_back(self, tmp_path):
        """A non-numeric value is ignored, falling through to the old column name."""
        (tmp_path / "training_log.csv").write_text(
            "epoch,train/loss,val/loss,train_loss,val_loss\n"
            "0,oops,bad,0.3,0.2\n"
            "1,oops,bad,,\n"
        )
        stats = _load_training_log(tmp_path)
        assert stats["best_val_loss"] == pytest.approx(0.2)
        assert stats["best_val_epoch"] == 0
        # Last row has no parseable value under either name.
        assert stats["final_train_loss"] is None
        assert stats["final_val_loss"] is None


class TestPrintModelInfo:
    def test_model_dir_with_metrics(self, capsys):
        print_model_info(str(MODEL_WITH_METRICS))
        output = capsys.readouterr().out
        assert "Single Instance" in output
        assert "UNet" in output
        assert "1.3M" in output
        assert "Training Results" in output
        assert "Evaluation Metrics" in output
        assert "Files" in output

    def test_model_dir_slash_named_log(self, tmp_path, capsys):
        """`info` shows loss rows for logs written with `train/loss`/`val/loss`."""
        shutil.copy(
            MODEL_NO_METRICS / "training_config.yaml", tmp_path / "training_config.yaml"
        )
        (tmp_path / "training_log.csv").write_text(
            "epoch,train/loss,val/loss,learning_rate,train/time,val/time\n"
            "0,0.5,0.4,0.001,1.0,0.5\n"
            "1,0.3,0.1,0.001,1.0,0.5\n"
            "2,0.2,0.2,0.001,1.0,0.5\n"
        )
        print_model_info(str(tmp_path))
        output = capsys.readouterr().out
        assert "Training Results" in output
        assert "Final train loss" in output
        assert "Final val loss" in output
        assert "Best val loss" in output
        assert "0.100000 (epoch 1)" in output

    def test_config_yaml_only(self, capsys):
        print_model_info(str(MODEL_NO_METRICS / "training_config.yaml"))
        output = capsys.readouterr().out
        assert "Single Instance" in output
        assert "UNet" in output
        # Should NOT have model-dir-only sections
        assert "Training Results" not in output
        assert "Files" not in output

    def test_model_dir_no_metrics(self, capsys):
        print_model_info(str(MODEL_NO_METRICS))
        output = capsys.readouterr().out
        assert "Single Instance" in output
        assert "Evaluation Metrics" not in output

    def test_nonexistent_path(self):
        with pytest.raises(SystemExit):
            print_model_info("/nonexistent/path")


class TestInfoCLI:
    def test_info_model_dir(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", str(MODEL_WITH_METRICS)])
        assert result.exit_code == 0
        assert "Single Instance" in result.output
        assert "UNet" in result.output

    def test_info_config_yaml(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["info", str(MODEL_NO_METRICS / "training_config.yaml")]
        )
        assert result.exit_code == 0
        assert "Single Instance" in result.output

    def test_info_nonexistent(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_info_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "info" in result.output


MODEL_CENTERED_INSTANCE = ASSETS / "minimal_instance_centered_instance"


def _data_rows(output: str) -> dict:
    """``{property: value}`` of the rows of the printed "Data" table."""
    lines = output.splitlines()
    titles = [line.strip() for line in lines]
    table = lines[titles.index("Data") + 1 : titles.index("Training")]
    rows = {}
    for line in table:
        cells = [c.strip() for c in line.strip("│ ").split("│")]
        if len(cells) == 2 and cells[0]:
            rows[cells[0]] = cells[1]
    return rows


class TestBBoxCropInfo:
    """`info` on box-scaled (`crop_mode: bbox`) centered-instance configs."""

    def _write_config(self, tmp_path, **preprocessing):
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(MODEL_CENTERED_INSTANCE / "training_config.yaml")
        cfg.data_config.preprocessing.crop_mode = "bbox"
        for key, value in preprocessing.items():
            cfg.data_config.preprocessing[key] = value
        path = tmp_path / "training_config.yaml"
        OmegaConf.save(cfg, path)
        return path

    def test_trained_box_crop(self, tmp_path, capsys):
        """A trained model shows its crop size, aspect, padding and min size."""
        path = self._write_config(
            tmp_path, crop_bbox_aspect="letterbox", crop_bbox_min_size=19.86
        )
        print_model_info(str(path))
        rows = _data_rows(capsys.readouterr().out)
        assert rows["Preprocessing"] == "scale=1.0, 384x384, crop=96"
        assert rows["Crop mode"] == (
            "bbox (aspect=letterbox, padding=0.1, min_size=19.86px)"
        )
        assert rows["Training boxes"] == "bbox_source=auto, bbox_padding=0.0"

    def test_untrained_box_crop_is_auto(self, tmp_path, capsys):
        """A generated config's null sizes are shown as auto, not as "no crop"."""
        path = self._write_config(tmp_path, crop_size=None, crop_bbox_padding=0.25)
        print_model_info(str(path))
        rows = _data_rows(capsys.readouterr().out)
        assert rows["Preprocessing"] == "scale=1.0, 384x384, crop=auto"
        assert rows["Crop mode"] == "bbox (aspect=square, padding=0.25, min_size=auto)"

    def test_fixed_crop_unchanged(self, capsys):
        """A fixed-crop model gets no box rows."""
        print_model_info(str(MODEL_CENTERED_INSTANCE / "training_config.yaml"))
        rows = _data_rows(capsys.readouterr().out)
        assert rows == {
            "Training data": rows["Training data"],
            "Preprocessing": "scale=1.0, 384x384, crop=96",
            "Augmentations": "Enabled",
        }

    def test_detector_shows_training_boxes(self, tmp_path, capsys):
        """A detector shows the box source its paired instance model must share."""
        from omegaconf import OmegaConf

        from sleap_nn.config_generator.generator import ConfigGenerator

        gen = ConfigGenerator.from_slp(
            "tests/assets/datasets/minimal_instance.pkg.slp"
        ).auto()
        cfg = gen.pipeline("topdown_bbox_centernet").build_detector()
        cfg.data_config.bbox_padding = 2.0
        OmegaConf.save(cfg, tmp_path / "detector.yaml")
        print_model_info(str(tmp_path / "detector.yaml"))
        rows = _data_rows(capsys.readouterr().out)
        assert rows["Training boxes"] == "bbox_source=auto, bbox_padding=2.0"
        assert "Crop mode" not in rows
