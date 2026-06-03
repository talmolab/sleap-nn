"""Tests for the rewired ``sleap-nn predict`` CLI command (PR 22 of #508).

The body now routes through :meth:`Predictor.from_export_dir`
and :class:`sleap_nn.inference.predictor.Predictor`. These tests exercise the
wiring directly via mocks so they don't require a real exported model. The
end-to-end "exports a model, runs predict, asserts SLP output" coverage lives
in :class:`tests.export.test_cli.TestPredictCommand`.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner
from omegaconf import OmegaConf

from .conftest import _make_export_metadata

_SKELETON_CFG = [
    {
        "name": "default",
        "nodes": [{"name": "head"}, {"name": "tail"}],
        "edges": [{"source": {"name": "head"}, "destination": {"name": "tail"}}],
    }
]


def _write_export_dir(tmp_path: Path, model_type: str = "single_instance") -> Path:
    """Create a minimal export directory (metadata + config + fake model)."""
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "model.onnx").write_bytes(b"fake")
    cfg = OmegaConf.create({"data_config": {"skeletons": _SKELETON_CFG}})
    OmegaConf.save(cfg, str(export_dir / "training_config.yaml"))
    meta = _make_export_metadata(model_type=model_type)
    meta.save(export_dir / "export_metadata.json")
    return export_dir


def _fake_video_path(tmp_path: Path) -> Path:
    """Click validates the video path exists; create a placeholder."""
    p = tmp_path / "video.mp4"
    p.write_bytes(b"fake")
    return p


def _patch_predictor_for_predict(model_type: str = "single_instance"):
    """Build a Predictor mock whose ``.predict`` returns a tiny ``Labels``."""
    import sleap_io as sio

    skeleton = sio.Skeleton(["head", "tail"])
    labels = sio.Labels(
        labeled_frames=[],
        videos=[],
        skeletons=[skeleton],
    )
    predictor = MagicMock()
    predictor.predict = MagicMock(return_value=labels)
    return predictor


class TestPredictWiring:
    """The rewired ``predict`` command goes through ``from_export_dir``."""

    @patch("sleap_nn.inference.predictor.Predictor.from_export_dir")
    @patch("sleap_io.Video.from_filename")
    def test_routes_through_from_export_dir(
        self, mock_video_cls, mock_from_export_dir, tmp_path
    ):
        """``--runtime``, ``--device``, and grouping kwargs flow into the factory."""
        from sleap_nn.export.cli import export_predict

        export_dir = _write_export_dir(tmp_path)
        video_path = _fake_video_path(tmp_path)

        sio_video = MagicMock()
        sio_video.__len__ = MagicMock(return_value=8)
        mock_video_cls.return_value = sio_video

        mock_from_export_dir.return_value = _patch_predictor_for_predict()

        runner = CliRunner()
        result = runner.invoke(
            export_predict,
            [
                str(export_dir),
                str(video_path),
                "-o",
                str(tmp_path / "out.slp"),
                "--runtime",
                "onnx",
                "--device",
                "cpu",
                "--batch-size",
                "2",
                "--n-frames",
                "4",
                "--max-instances",
                "5",
                "--min-instance-peaks",
                "1",
                "--min-line-scores",
                "0.3",
                "--cpu-workers",
                "0",
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_from_export_dir.called
        kwargs = mock_from_export_dir.call_args.kwargs
        assert kwargs["runtime"] == "onnx"
        assert kwargs["device"] == "cpu"
        assert kwargs["max_instances"] == 5
        assert kwargs["min_instance_peaks"] == 1.0
        assert kwargs["min_line_scores"] == 0.3
        assert kwargs["paf_workers"] == 0

    @patch("sleap_nn.inference.predictor.Predictor.from_export_dir")
    @patch("sleap_io.Video.from_filename")
    def test_warns_on_baked_in_flags(
        self, mock_video_cls, mock_from_export_dir, tmp_path
    ):
        """Non-default values for graph-baked flags emit a one-time warning."""
        from sleap_nn.export.cli import export_predict

        export_dir = _write_export_dir(tmp_path)
        video_path = _fake_video_path(tmp_path)

        sio_video = MagicMock()
        sio_video.__len__ = MagicMock(return_value=2)
        mock_video_cls.return_value = sio_video
        mock_from_export_dir.return_value = _patch_predictor_for_predict()

        runner = CliRunner()
        result = runner.invoke(
            export_predict,
            [
                str(export_dir),
                str(video_path),
                "-o",
                str(tmp_path / "out.slp"),
                "--peak-conf-threshold",
                "0.7",
                "--max-edge-length-ratio",
                "0.5",
                "--n-frames",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "baked into the exported graph" in result.output
        assert "--peak-conf-threshold=0.7" in result.output
        assert "--max-edge-length-ratio=0.5" in result.output

    @patch("sleap_nn.inference.predictor.Predictor.from_export_dir")
    @patch("sleap_io.Video.from_filename")
    def test_default_baked_in_flags_silent(
        self, mock_video_cls, mock_from_export_dir, tmp_path
    ):
        """Defaults don't trigger the baked-in warning."""
        from sleap_nn.export.cli import export_predict

        export_dir = _write_export_dir(tmp_path)
        video_path = _fake_video_path(tmp_path)

        sio_video = MagicMock()
        sio_video.__len__ = MagicMock(return_value=2)
        mock_video_cls.return_value = sio_video
        mock_from_export_dir.return_value = _patch_predictor_for_predict()

        runner = CliRunner()
        result = runner.invoke(
            export_predict,
            [
                str(export_dir),
                str(video_path),
                "-o",
                str(tmp_path / "out.slp"),
                "--n-frames",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "baked into the exported graph" not in result.output

    def test_missing_metadata_raises_click_exception(self, tmp_path):
        """A bad export dir produces a clean ClickException, not a traceback."""
        from sleap_nn.export.cli import export_predict

        export_dir = tmp_path / "no_metadata"
        export_dir.mkdir()
        video_path = _fake_video_path(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            export_predict,
            [
                str(export_dir),
                str(video_path),
                "-o",
                str(tmp_path / "out.slp"),
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_missing_training_config_raises_click_exception(self, tmp_path):
        """Metadata present but no training_config.yaml → ClickException."""
        from sleap_nn.export.cli import export_predict

        export_dir = tmp_path / "no_cfg"
        export_dir.mkdir()
        (export_dir / "model.onnx").write_bytes(b"fake")
        meta = _make_export_metadata(model_type="single_instance")
        meta.save(export_dir / "export_metadata.json")
        # Deliberately no training_config.yaml.

        video_path = _fake_video_path(tmp_path)

        runner = CliRunner()
        result = runner.invoke(
            export_predict,
            [
                str(export_dir),
                str(video_path),
                "-o",
                str(tmp_path / "out.slp"),
            ],
        )

        assert result.exit_code != 0
        assert "training_config" in result.output.lower()


class TestFindTrainingConfig:
    """``_find_training_config_for_predict`` mirrors the legacy resolution order."""

    def test_finds_training_config_yaml(self, tmp_path):
        from sleap_nn.export.cli import _find_training_config_for_predict

        (tmp_path / "training_config.yaml").write_text("data_config: {}\n")
        path = _find_training_config_for_predict(tmp_path, "single_instance")
        assert path.name == "training_config.yaml"

    def test_topdown_prefers_centered_instance_config(self, tmp_path):
        from sleap_nn.export.cli import _find_training_config_for_predict

        (tmp_path / "training_config_centered_instance.yaml").write_text(
            "data_config: {}\n"
        )
        (tmp_path / "training_config.yaml").write_text("data_config: {}\n")
        path = _find_training_config_for_predict(tmp_path, "topdown")
        assert path.name == "training_config_centered_instance.yaml"

    def test_multiclass_topdown_combined_prefers_multiclass_config(self, tmp_path):
        from sleap_nn.export.cli import _find_training_config_for_predict

        (tmp_path / "training_config_multi_class_topdown.yaml").write_text(
            "data_config: {}\n"
        )
        (tmp_path / "training_config.yaml").write_text("data_config: {}\n")
        path = _find_training_config_for_predict(
            tmp_path, "multi_class_topdown_combined"
        )
        assert path.name == "training_config_multi_class_topdown.yaml"

    def test_falls_back_to_model_type_suffix(self, tmp_path):
        from sleap_nn.export.cli import _find_training_config_for_predict

        (tmp_path / "training_config_centroid.yaml").write_text("data_config: {}\n")
        path = _find_training_config_for_predict(tmp_path, "centroid")
        assert path.name == "training_config_centroid.yaml"

    def test_raises_when_missing(self, tmp_path):
        from sleap_nn.export.cli import _find_training_config_for_predict

        with pytest.raises(FileNotFoundError, match="No training_config"):
            _find_training_config_for_predict(tmp_path, "single_instance")
