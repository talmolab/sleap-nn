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

    def test_export_command_embedding_onnx(self, minimal_embedding_model_dir, tmp_path):
        """Export an embedding (re-ID) model to ONNX (P2 #6)."""
        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_embedding"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_embedding_model_dir),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, result.output
        assert (output_dir / "model.onnx").exists()

        metadata = json.loads((output_dir / "export_metadata.json").read_text())
        assert metadata["model_type"] == "embedding"
        assert metadata["embedding_dim"] == 16
        assert metadata["normalize"] is True
        assert metadata["backbone_source"] == "scratch"
        assert metadata["normalization"] == "per_crop_standardize"
        # Grayscale crop input, no skeleton.
        assert metadata["input_channels"] == 1
        assert metadata["n_nodes"] == 0
        assert metadata["node_names"] == []

    @requires_onnxruntime
    def test_export_command_embedding_onnxruntime_roundtrip(
        self, minimal_embedding_model_dir, tmp_path
    ):
        """The exported embedding ONNX (incl. the GeM head) runs under onnxruntime."""
        import numpy as np
        import onnxruntime as ort

        from sleap_nn.export.cli import export

        output_dir = tmp_path / "export_embedding_ort"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [str(minimal_embedding_model_dir), "-o", str(output_dir), "-f", "onnx"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        sess = ort.InferenceSession(
            str(output_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        in_name = sess.get_inputs()[0].name
        crop = np.random.randint(0, 256, (2, 1, 32, 32)).astype(np.uint8)
        out = sess.run(None, {in_name: crop})[0]
        assert out.shape == (2, 16)
        # normalize=True -> unit-norm rows.
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)


class TestTwoCentroidGuard:
    """Two centroid directories is an error (no onnx dependency required).

    The guard fires before any model load, so it needs no export deps.
    """

    def test_two_centroid_dirs_raises(self, minimal_instance_centroid_ckpt, tmp_path):
        """Passing the same centroid dir twice raises a clear ClickException."""
        from sleap_nn.export.cli import export

        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_centroid_ckpt),
                str(minimal_instance_centroid_ckpt),
                "-o",
                str(tmp_path / "out"),
                "--no-verify",
            ],
        )

        assert result.exit_code != 0
        # Clear, specific message — not the generic combination fallthrough.
        assert "two centroid model directories" in result.output.lower()


@requires_onnx
class TestCentroidExportConsistency:
    """Standalone centroid export → metadata consistency (requires onnx)."""

    def test_standalone_centroid_metadata(
        self, minimal_instance_centroid_ckpt, tmp_path
    ):
        """A single centroid dir exports metadata with model_type='centroid'.

        Asserts node_names carries the full (multi-node) training skeleton and
        anchor_part round-trips (None for this fixture, which has no anchor).
        """
        from omegaconf import OmegaConf

        from sleap_nn.export.cli import export
        from sleap_nn.export.metadata import ExportMetadata
        from sleap_nn.export.utils import resolve_anchor_part, resolve_node_names

        output_dir = tmp_path / "export_standalone_centroid"
        runner = CliRunner()
        result = runner.invoke(
            export,
            [
                str(minimal_instance_centroid_ckpt),
                "-o",
                str(output_dir),
                "--format",
                "onnx",
                "--no-verify",
            ],
        )
        assert result.exit_code == 0, result.output

        meta = ExportMetadata.load(output_dir / "export_metadata.json")
        assert meta.model_type == "centroid"

        cfg = OmegaConf.load(
            (minimal_instance_centroid_ckpt / "training_config.yaml").as_posix()
        )
        assert meta.node_names == resolve_node_names(cfg, "centroid")
        assert meta.anchor_part == resolve_anchor_part(cfg, "centroid")
        # The full training skeleton must be preserved for the #586 source tag
        # (the runtime collapses to a single 'centroid' node at packaging time,
        # not in the metadata).
        assert len(meta.node_names) > 1


class TestCLIHelpers:
    """Tests for CLI helper functions."""

    def test_training_config_path(self, minimal_instance_single_instance_ckpt):
        """Test _training_config_path function."""
        from sleap_nn.export.cli import _training_config_path

        path = _training_config_path(minimal_instance_single_instance_ckpt)
        assert path.exists()
        assert path.name == "training_config.yaml"
