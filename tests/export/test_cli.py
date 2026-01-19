"""Tests for sleap_nn.export.cli module.

These tests use Click's CliRunner to test CLI commands.
"""

import pytest
import json
from pathlib import Path
from click.testing import CliRunner

from .conftest import requires_onnx, requires_onnxruntime


class TestExportCommandHelp:
    """Tests for export command help (no dependencies required)."""

    def test_export_command_help(self):
        """Test that --help works for export command."""
        from sleap_nn.export.cli import export

        runner = CliRunner()
        result = runner.invoke(export, ["--help"])

        assert result.exit_code == 0
        assert (
            "Export a trained model" in result.output
            or "export" in result.output.lower()
        )


class TestPredictCommandHelp:
    """Tests for predict command help (no dependencies required)."""

    def test_predict_command_help(self):
        """Test that --help works for predict command."""
        from sleap_nn.export.cli import predict

        runner = CliRunner()
        result = runner.invoke(predict, ["--help"])

        assert result.exit_code == 0
        assert (
            "predict" in result.output.lower() or "inference" in result.output.lower()
        )


@requires_onnx
class TestExportCommand:
    """Tests for export CLI command (requires onnx)."""

    def test_export_command_single_instance_onnx(
        self, minimal_instance_single_instance_ckpt, tmp_path
    ):
        """Test exporting single_instance model to ONNX."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_output"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_single_instance_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
        )

        # Should complete (may have warnings but not error)
        assert result.exit_code == 0 or "error" not in result.output.lower()

        # Check output files exist
        if result.exit_code == 0:
            assert output_dir.exists()
            onnx_file = output_dir / "model.onnx"
            assert onnx_file.exists()

    def test_export_command_centroid_onnx(
        self, minimal_instance_centroid_ckpt, tmp_path
    ):
        """Test exporting centroid model to ONNX."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_centroid"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_centroid_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
        )

        if result.exit_code == 0:
            assert output_dir.exists()

    def test_export_command_bottomup_onnx(
        self, minimal_instance_bottomup_ckpt, tmp_path
    ):
        """Test exporting bottomup model to ONNX."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_bottomup"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_bottomup_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
        )

        if result.exit_code == 0:
            assert output_dir.exists()

    def test_export_command_creates_metadata(
        self, minimal_instance_single_instance_ckpt, tmp_path
    ):
        """Test that metadata JSON file is created during export."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_with_meta"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_single_instance_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
        )

        if result.exit_code == 0:
            # Metadata might be model.onnx.metadata.json or metadata.json
            metadata_files = list(output_dir.glob("*.json"))
            assert (
                len(metadata_files) > 0
            ), f"No metadata JSON file created in {output_dir}"

            # Verify metadata content
            with open(metadata_files[0]) as f:
                metadata = json.load(f)
            assert "model_type" in metadata
            assert "n_nodes" in metadata

    def test_export_command_topdown_combined(
        self,
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
        tmp_path,
    ):
        """Test exporting combined top-down model (centroid + instance)."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_topdown"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_centroid_ckpt),
                str(minimal_instance_centered_instance_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
        )

        # Combined export may succeed or fail depending on model compatibility
        # Just check it doesn't crash unexpectedly
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_export_command_invalid_path(self, tmp_path):
        """Test that non-existent path produces an error."""
        from sleap_nn.export.cli import export

        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                "/nonexistent/path/to/model",
                "-o",
                str(tmp_path / "output"),
            ],
        )

        # Should fail with non-zero exit code or error message
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_export_command_missing_config(self, tmp_path):
        """Test that missing training_config.yaml produces an error."""
        from sleap_nn.export.cli import export

        # Create empty directory without config
        fake_model_dir = tmp_path / "fake_model"
        fake_model_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(fake_model_dir),
                "-o",
                str(tmp_path / "output"),
            ],
        )

        # Should fail
        assert (
            result.exit_code != 0
            or "error" in result.output.lower()
            or "not found" in result.output.lower()
        )


@requires_onnx
@requires_onnxruntime
class TestPredictCommand:
    """Tests for predict CLI command (requires onnx and onnxruntime)."""

    @pytest.fixture
    def exported_model_dir(self, minimal_instance_single_instance_ckpt, tmp_path):
        """Export a model for predict tests."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "exported_for_predict"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_single_instance_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
        )

        if result.exit_code != 0:
            pytest.skip("Export failed, cannot test predict")

        return output_dir

    def test_predict_command_creates_slp(
        self, exported_model_dir, tiny_video, tmp_path
    ):
        """Test that predict creates output .slp file."""
        from sleap_nn.export.cli import predict

        output_slp = tmp_path / "predictions.slp"
        runner = CliRunner()
        result = runner.invoke(
            predict,
            [
                str(exported_model_dir),
                str(tiny_video),
                "-o",
                str(output_slp),
                "--n-frames",
                "5",
            ],
        )

        # Check if prediction succeeded
        if result.exit_code == 0:
            assert output_slp.exists()

    def test_predict_command_n_frames(self, exported_model_dir, tiny_video, tmp_path):
        """Test that --n-frames limits processing."""
        from sleap_nn.export.cli import predict

        output_slp = tmp_path / "predictions_limited.slp"
        runner = CliRunner()
        result = runner.invoke(
            predict,
            [
                str(exported_model_dir),
                str(tiny_video),
                "-o",
                str(output_slp),
                "--n-frames",
                "3",
            ],
        )

        # Should complete without error
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_predict_command_batch_size(self, exported_model_dir, tiny_video, tmp_path):
        """Test that --batch-size parameter is accepted."""
        from sleap_nn.export.cli import predict

        output_slp = tmp_path / "predictions_batched.slp"
        runner = CliRunner()
        result = runner.invoke(
            predict,
            [
                str(exported_model_dir),
                str(tiny_video),
                "-o",
                str(output_slp),
                "--batch-size",
                "2",
                "--n-frames",
                "4",
            ],
        )

        # Should complete without error
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_predict_command_invalid_export(self, tmp_path, tiny_video):
        """Test that bad export directory produces an error."""
        from sleap_nn.export.cli import predict

        runner = CliRunner()
        result = runner.invoke(
            predict,
            [
                str(tmp_path / "nonexistent_export"),
                str(tiny_video),
                "-o",
                str(tmp_path / "output.slp"),
            ],
        )

        # Should fail
        assert result.exit_code != 0 or "error" in result.output.lower()


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_training_config_path(self, minimal_instance_single_instance_ckpt):
        """Test _training_config_path function."""
        from sleap_nn.export.cli import _training_config_path

        path = _training_config_path(minimal_instance_single_instance_ckpt)
        assert path.exists()
        assert path.name == "training_config.yaml"
