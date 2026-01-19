"""Tests for sleap_nn.export.predictors module factory functions."""

import pytest
from pathlib import Path

from .conftest import requires_onnxruntime


class TestDetectRuntime:
    """Tests for detect_runtime function."""

    def test_detect_runtime_onnx_extension(self, tmp_path):
        """Test detection from .onnx extension."""
        from sleap_nn.export.predictors import detect_runtime

        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()

        runtime = detect_runtime(onnx_path)
        assert runtime == "onnx"

    def test_detect_runtime_trt_extension(self, tmp_path):
        """Test detection from .trt extension."""
        from sleap_nn.export.predictors import detect_runtime

        trt_path = tmp_path / "model.trt"
        trt_path.touch()

        runtime = detect_runtime(trt_path)
        assert runtime == "tensorrt"

    def test_detect_runtime_engine_extension(self, tmp_path):
        """Test detection from .engine extension."""
        from sleap_nn.export.predictors import detect_runtime

        engine_path = tmp_path / "model.engine"
        engine_path.touch()

        runtime = detect_runtime(engine_path)
        assert runtime == "tensorrt"

    def test_detect_runtime_unknown_extension(self, tmp_path):
        """Test that unknown extension raises ValueError."""
        from sleap_nn.export.predictors import detect_runtime

        unknown_path = tmp_path / "model.xyz"
        unknown_path.touch()

        with pytest.raises(ValueError, match="Unknown model format"):
            detect_runtime(unknown_path)

    def test_detect_runtime_directory_with_onnx(self, tmp_path):
        """Test detection from directory containing ONNX model."""
        from sleap_nn.export.predictors import detect_runtime

        model_dir = tmp_path / "my_model"
        exported_dir = model_dir / "exported"
        exported_dir.mkdir(parents=True)
        (exported_dir / "model.onnx").touch()

        runtime = detect_runtime(model_dir)
        assert runtime == "onnx"

    def test_detect_runtime_directory_with_trt(self, tmp_path):
        """Test detection from directory containing TensorRT model."""
        from sleap_nn.export.predictors import detect_runtime

        model_dir = tmp_path / "my_model"
        exported_dir = model_dir / "exported"
        exported_dir.mkdir(parents=True)
        (exported_dir / "model.trt").touch()

        runtime = detect_runtime(model_dir)
        assert runtime == "tensorrt"

    def test_detect_runtime_directory_with_both_prefers_trt(self, tmp_path):
        """Test that TensorRT is preferred when both formats exist."""
        from sleap_nn.export.predictors import detect_runtime

        model_dir = tmp_path / "my_model"
        exported_dir = model_dir / "exported"
        exported_dir.mkdir(parents=True)
        (exported_dir / "model.onnx").touch()
        (exported_dir / "model.trt").touch()

        runtime = detect_runtime(model_dir)
        assert runtime == "tensorrt"

    def test_detect_runtime_empty_directory(self, tmp_path):
        """Test that empty directory raises ValueError."""
        from sleap_nn.export.predictors import detect_runtime

        model_dir = tmp_path / "empty_model"
        exported_dir = model_dir / "exported"
        exported_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="No exported model found"):
            detect_runtime(model_dir)

    def test_detect_runtime_string_path(self, tmp_path):
        """Test that string paths work correctly."""
        from sleap_nn.export.predictors import detect_runtime

        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()

        runtime = detect_runtime(str(onnx_path))
        assert runtime == "onnx"


class TestLoadExportedModel:
    """Tests for load_exported_model function."""

    @requires_onnxruntime
    def test_load_exported_model_onnx(self, exported_onnx_model):
        """Test loading an ONNX model."""
        from sleap_nn.export.predictors import load_exported_model, ONNXPredictor

        predictor = load_exported_model(exported_onnx_model, runtime="onnx")
        assert isinstance(predictor, ONNXPredictor)

    @requires_onnxruntime
    def test_load_exported_model_auto_detection(self, exported_onnx_model):
        """Test auto-detection of ONNX model."""
        from sleap_nn.export.predictors import load_exported_model, ONNXPredictor

        predictor = load_exported_model(exported_onnx_model, runtime="auto")
        assert isinstance(predictor, ONNXPredictor)

    @requires_onnxruntime
    def test_load_exported_model_explicit_device(self, exported_onnx_model):
        """Test explicit device specification."""
        from sleap_nn.export.predictors import load_exported_model

        predictor = load_exported_model(
            exported_onnx_model, runtime="onnx", device="cpu"
        )
        assert predictor is not None

    def test_load_exported_model_unknown_runtime(self, tmp_path):
        """Test that unknown runtime raises ValueError."""
        from sleap_nn.export.predictors import load_exported_model

        dummy_path = tmp_path / "model.onnx"
        dummy_path.touch()

        with pytest.raises(ValueError, match="Unknown runtime"):
            load_exported_model(dummy_path, runtime="unknown_runtime")
