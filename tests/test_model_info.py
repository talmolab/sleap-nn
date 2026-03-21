"""Tests for sleap_nn.model_info module."""

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
