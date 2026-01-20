"""Tests for the unified CLI functionality."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
import subprocess
from click.testing import CliRunner
from sleap_nn.predict import run_inference
from sleap_nn.cli import cli, print_version, parse_path_map, show_training_help
from sleap_nn import __version__
import sleap_io as sio
import torch


# =============================================================================
# CliRunner-based tests (in-process, tracked by coverage)
# =============================================================================


class TestCliVersion:
    """Tests for --version flag using CliRunner."""

    def test_version_long_flag(self):
        """Test --version flag displays version and exits."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert f"sleap-nn {__version__}" in result.output

    def test_version_short_flag(self):
        """Test -v short flag for version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["-v"])
        assert result.exit_code == 0
        assert f"sleap-nn {__version__}" in result.output


class TestCliHelp:
    """Tests for CLI help output using CliRunner."""

    def test_main_help(self):
        """Test main CLI --help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "SLEAP-NN" in result.output
        assert "train" in result.output
        assert "track" in result.output
        assert "eval" in result.output
        assert "system" in result.output

    def test_train_help(self):
        """Test train command --help shows custom help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "sleap-nn train" in result.output
        assert "Usage" in result.output  # Rich-click renders ## Usage as header
        assert "sleap.ai" in result.output

    def test_train_no_config_shows_help(self):
        """Test train without --config-name shows help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--config-dir", "."])
        assert result.exit_code == 0
        assert "sleap-nn train" in result.output
        assert "Usage" in result.output  # Rich-click renders ## Usage as header


class TestSystemCommand:
    """Tests for system command using CliRunner."""

    def test_system_command_runs(self):
        """Test system command executes and produces output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["system"])
        assert result.exit_code == 0
        # Should contain system info sections
        assert "Python" in result.output or "sleap-nn" in result.output

    def test_system_command_shows_pytorch(self):
        """Test system command shows PyTorch info."""
        runner = CliRunner()
        result = runner.invoke(cli, ["system"])
        assert result.exit_code == 0
        assert "torch" in result.output.lower() or "pytorch" in result.output.lower()


class TestParsePathMap:
    """Tests for parse_path_map callback function."""

    def test_parse_path_map_empty(self):
        """Test parse_path_map with empty value returns None."""
        result = parse_path_map(None, None, None)
        assert result is None

        result = parse_path_map(None, None, ())
        assert result is None

        result = parse_path_map(None, None, [])
        assert result is None

    def test_parse_path_map_single_pair(self):
        """Test parse_path_map with single path pair."""
        result = parse_path_map(None, None, [("/old/path.mp4", "/new/path.mp4")])
        assert result == {"/old/path.mp4": "/new/path.mp4"}

    def test_parse_path_map_multiple_pairs(self):
        """Test parse_path_map with multiple path pairs."""
        pairs = [("/old1.mp4", "/new1.mp4"), ("/old2.mp4", "/new2.mp4")]
        result = parse_path_map(None, None, pairs)
        assert len(result) == 2
        assert result["/old1.mp4"] == "/new1.mp4"
        assert result["/old2.mp4"] == "/new2.mp4"

    def test_parse_path_map_normalizes_paths(self):
        """Test parse_path_map converts paths to posix format."""
        import sys

        # Test with a forward-slash path (works on all platforms)
        result = parse_path_map(None, None, [("/old", "/new/path")])
        assert result["/old"] == "/new/path"

        # On Windows, backslashes should be converted to forward slashes
        # On other platforms, backslashes are literal characters (not path separators)
        if sys.platform == "win32":
            result = parse_path_map(None, None, [("/old", "C:\\new\\path")])
            assert "\\" not in result["/old"]
            assert result["/old"] == "C:/new/path"


class TestShowTrainingHelp:
    """Tests for show_training_help function."""

    def test_show_training_help_output(self, capsys):
        """Test show_training_help outputs correct help text."""
        show_training_help()
        captured = capsys.readouterr()
        assert "sleap-nn train" in captured.out
        assert "Usage" in captured.out  # Rich-click renders ## Usage as header
        assert "config.yaml" in captured.out  # New positional arg usage
        assert "sleap.ai" in captured.out


class TestPrintVersion:
    """Tests for print_version callback function."""

    def test_print_version_with_resilient_parsing(self):
        """Test print_version returns early during resilient parsing."""
        ctx = MagicMock()
        ctx.resilient_parsing = True
        result = print_version(ctx, None, True)
        assert result is None
        ctx.exit.assert_not_called()

    def test_print_version_with_false_value(self):
        """Test print_version returns early when value is False."""
        ctx = MagicMock()
        ctx.resilient_parsing = False
        result = print_version(ctx, None, False)
        assert result is None
        ctx.exit.assert_not_called()


class TestTrackCommand:
    """Tests for track command argument handling using CliRunner."""

    def test_track_without_model_paths(self):
        """Test track command with no model paths sets None."""
        runner = CliRunner()
        # This will fail because data_path is required, but we can check the error
        result = runner.invoke(cli, ["track"])
        assert result.exit_code != 0
        assert (
            "data_path" in result.output.lower() or "required" in result.output.lower()
        )

    def test_track_without_frames(self):
        """Test track command with empty frames string."""
        runner = CliRunner()
        # Mock run_inference to avoid actual inference
        with patch("sleap_nn.cli.run_inference") as mock_inference:
            mock_inference.return_value = None
            result = runner.invoke(
                cli,
                [
                    "track",
                    "--data_path",
                    "/fake/path.mp4",
                    "--model_paths",
                    "/fake/model",
                ],
            )
            # Check that frames was set to None (empty string -> None)
            if mock_inference.called:
                call_kwargs = mock_inference.call_args[1]
                assert call_kwargs.get("frames") is None


class TestEvalCommand:
    """Tests for eval command using CliRunner."""

    def test_eval_missing_required_args(self):
        """Test eval command fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["eval"])
        assert result.exit_code != 0
        assert (
            "ground_truth_path" in result.output.lower()
            or "required" in result.output.lower()
        )


# =============================================================================
# Subprocess-based tests (integration tests, run external process)
# =============================================================================


@pytest.fixture
def sample_config(tmp_path, minimal_instance):
    """Create a sample config file for testing."""
    config_data = OmegaConf.create(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train_labels_path": [f"{minimal_instance}"],
                "val_labels_path": [f"{minimal_instance}"],
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "cache_img_path": None,
                "use_existing_imgs": False,
                "delete_cache_imgs_after_training": True,
                "preprocessing": {
                    "ensure_rgb": False,
                    "ensure_grayscale": False,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "crop_size": 160,
                    "min_crop_size": None,
                },
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": {"contrast_p": 1.0},
                    "geometric": {
                        "rotation_max": 180.0,
                        "rotation_min": -180.0,
                        "scale_min": 1.0,
                        "scale_max": 1.0,
                        "translate_width": 0,
                        "translate_height": 0,
                        "affine_p": 0.5,
                    },
                },
            },
            "model_config": {
                "init_weights": "default",
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 16,
                        "filters_rate": 1.5,
                        "max_stride": 8,
                        "convs_per_block": 2,
                        "stacks": 1,
                        "stem_stride": None,
                        "middle_block": True,
                        "up_interpolate": False,
                        "output_stride": 2,
                    }
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "bottomup": None,
                    "centered_instance": {
                        "confmaps": {
                            "part_names": ["A", "B"],
                            "anchor_part": "B",
                            "sigma": 1.5,
                            "output_stride": 2,
                        }
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "val_data_loader": {
                    "batch_size": 1,
                    "num_workers": 0,
                },
                "model_ckpt": {
                    "save_top_k": 1,
                    "save_last": True,
                },
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "min_delta": 1e-08,
                    "patience": 20,
                },
                "trainer_devices": 1,
                "trainer_device_indices": None,
                "trainer_accelerator": "cpu",
                "max_epochs": 1,
                "min_train_steps_per_epoch": 1,
                "seed": 42,
                "use_wandb": False,
                "save_ckpt": True,
                "ckpt_dir": str(tmp_path),
                "run_name": "test_ckpt",
                "resume_ckpt_path": None,
            },
        }
    )
    return config_data


def _strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes from text.

    This is necessary because some libraries (PyTorch Lightning, Rich, tqdm)
    emit ANSI codes even when NO_COLOR is set, which breaks YAML parsing.
    """
    import re

    # Pattern matches ANSI escape sequences including:
    # - CSI sequences: \x1b[...X (e.g., colors, cursor movement)
    # - Private mode sequences: \x1b[?...X (e.g., cursor show/hide)
    # - OSC sequences: \x1b]...X (e.g., window title)
    ansi_pattern = re.compile(r"\x1b\[[?0-9;]*[a-zA-Z]|\x1b\][^\x07]*\x07")
    return ansi_pattern.sub("", text)


def test_main_cli(sample_config, tmp_path):
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--help",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    # Exit code should be 0
    assert result.returncode == 0
    assert "Usage" in result.stdout  # Should show usage information
    assert "sleap.ai" in result.stdout  # should point user to read the documents

    # Now to test overrides and defaults

    sample_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())

    import os
    import re

    # Environment to disable color output from rich/loguru/lightning
    no_color_env = {**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"}

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=no_color_env,
    )
    # Exit code should be 0
    assert result.returncode == 0
    # Try to parse the output back into the yaml, truncate the beginning (starts with "data_config")
    # Only keep stdout starting from "data_config" and ending at the next log timestamp
    # Strip ANSI codes first as some libraries emit them even with NO_COLOR
    stdout_clean = _strip_ansi_codes(result.stdout)
    stripped_out = stdout_clean[stdout_clean.find("data_config") :].strip()
    # Find the next log line (timestamp pattern: YYYY-MM-DD HH:MM:SS |)
    match = re.search(r"\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|", stripped_out)
    if match:
        stripped_out = stripped_out[: match.start()]
    output = OmegaConf.create(stripped_out)
    assert output == sample_config

    # config override should work
    sample_config.trainer_config.max_epochs = 2
    sample_config.data_config.preprocessing.scale = 1.2
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "trainer_config.max_epochs=2",
        "data_config.preprocessing.scale=1.2",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=no_color_env,
    )
    # Exit code should be 0
    assert result.returncode == 0
    stdout_clean = _strip_ansi_codes(result.stdout)
    stripped_out = stdout_clean[stdout_clean.find("data_config") :].strip()
    match = re.search(r"\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|", stripped_out)
    if match:
        stripped_out = stripped_out[: match.start()]
    output = OmegaConf.create(stripped_out)
    assert output == sample_config

    # Test CLI with '--' to separate config overrides from positional args
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--",
        "trainer_config.max_epochs=1",
        "data_config.preprocessing.scale=1.5",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=no_color_env,
    )
    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()
    # Exit code should be 0
    assert result.returncode == 0
    # Check that overrides are applied
    stdout_clean = _strip_ansi_codes(result.stdout)
    stripped_out = stdout_clean[stdout_clean.find("data_config") :].strip()
    # Find the next log line (timestamp pattern: YYYY-MM-DD HH:MM:SS |)
    match = re.search(r"\n\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \|", stripped_out)
    if match:
        stripped_out = stripped_out[: match.start()]
    output = OmegaConf.create(stripped_out)
    assert output.trainer_config.max_epochs == 1
    assert output.data_config.preprocessing.scale == 1.5


def test_train_cli_with_video_paths(
    sample_config, tmp_path, small_robot_minimal, small_robot_minimal_video
):
    sample_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    sample_config.trainer_config.run_name = "test_small_robot_video_paths"
    sample_config.data_config.train_labels_path = [small_robot_minimal.as_posix()]
    sample_config.data_config.val_labels_path = None
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())

    # with video paths as list
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--video-paths",
        f"{small_robot_minimal_video.as_posix()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()

    # with video paths as dictionary
    sample_config.trainer_config.run_name = "test_small_robot_video_paths_mapper"
    sample_config.data_config.train_labels_path = [small_robot_minimal.as_posix()]
    sample_config.data_config.val_labels_path = None
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())
    labels = sio.load_slp(small_robot_minimal)
    video_path = Path(labels.videos[0].filename).as_posix()

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--video-path-map",
        f"{video_path}",
        f"{small_robot_minimal_video.as_posix()}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0

    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()


def test_train_cli_with_prefix_map(
    sample_config, tmp_path, small_robot_minimal, small_robot_minimal_video
):
    """Test --prefix-map option for replacing path prefixes."""
    sample_config.trainer_config.trainer_accelerator = (
        "cpu" if torch.mps.is_available() else "auto"
    )
    sample_config.trainer_config.run_name = "test_small_robot_prefix_map"
    sample_config.data_config.train_labels_path = [small_robot_minimal.as_posix()]
    sample_config.data_config.val_labels_path = None
    OmegaConf.save(sample_config, (Path(tmp_path) / "test_config.yaml").as_posix())

    # Get the video path from labels and extract the prefix
    labels = sio.load_slp(small_robot_minimal)
    old_video_path = Path(labels.videos[0].filename)
    new_video_path = Path(small_robot_minimal_video)

    # Find the common suffix and determine prefixes
    old_prefix = old_video_path.parent.as_posix()
    new_prefix = new_video_path.parent.as_posix()

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "train",
        "--config-dir",
        f"{tmp_path}",
        "--config-name",
        "test_config",
        "--prefix-map",
        f"{old_prefix}",
        f"{new_prefix}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stderr: {result.stderr}"

    assert (
        Path(f"{sample_config.trainer_config.ckpt_dir}")
        / sample_config.trainer_config.run_name
    ).exists()
    assert Path(
        f"{sample_config.trainer_config.ckpt_dir}/{sample_config.trainer_config.run_name}/best.ckpt"
    ).exists()


def test_track_command(
    centered_instance_video,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
    tmp_path,
):
    import subprocess

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "track",
        "--model_paths",
        minimal_instance_centroid_ckpt,
        "--model_paths",
        minimal_instance_centered_instance_ckpt,
        "--data_path",
        centered_instance_video.as_posix(),
        "--max_instances",
        "2",
        "--output_path",
        f"{tmp_path}/test.slp",
        "--frames",
        "0-99",
        "--device",
        "cpu",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert Path(f"{tmp_path}/test.slp").exists()

    labels = sio.load_slp(f"{tmp_path}/test.slp")
    assert len(labels) == 100


def test_eval_command(
    minimal_instance, tmp_path, minimal_instance_centered_instance_ckpt
):
    """Test eval command execution."""
    output = run_inference(
        model_paths=[minimal_instance_centered_instance_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        max_instances=6,
        output_path=f"{tmp_path}/test.slp",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "eval",
        "--ground_truth_path",
        minimal_instance.as_posix(),
        "--predicted_path",
        f"{tmp_path}/test.slp",
        "--save_metrics",
        f"{tmp_path}/metrics_test.npz",
    ]
    # Run the command and check for errors
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert Path(f"{tmp_path}/metrics_test.npz").exists()
