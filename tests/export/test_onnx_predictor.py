"""Tests for sleap_nn.export.predictors.onnx module.

These tests require the `onnxruntime` package.
"""

import pytest
import numpy as np
import torch

from .conftest import requires_onnx, requires_onnxruntime


# Helper functions for testing (don't require onnxruntime)
class TestHelperFunctions:
    """Tests for helper functions that don't require onnxruntime."""

    def test_onnx_type_to_numpy_float(self):
        """Test tensor(float) → np.float32 conversion."""
        from sleap_nn.export.predictors.onnx import _onnx_type_to_numpy

        result = _onnx_type_to_numpy("tensor(float)")
        assert result == np.float32

    def test_onnx_type_to_numpy_uint8(self):
        """Test tensor(uint8) → np.uint8 conversion."""
        from sleap_nn.export.predictors.onnx import _onnx_type_to_numpy

        result = _onnx_type_to_numpy("tensor(uint8)")
        assert result == np.uint8

    def test_onnx_type_to_numpy_int32(self):
        """Test tensor(int32) → np.int32 conversion."""
        from sleap_nn.export.predictors.onnx import _onnx_type_to_numpy

        result = _onnx_type_to_numpy("tensor(int32)")
        assert result == np.int32

    def test_onnx_type_to_numpy_float16(self):
        """Test tensor(float16) → np.float16 conversion."""
        from sleap_nn.export.predictors.onnx import _onnx_type_to_numpy

        result = _onnx_type_to_numpy("tensor(float16)")
        assert result == np.float16

    def test_onnx_type_to_numpy_unknown(self):
        """Test that unknown type returns None."""
        from sleap_nn.export.predictors.onnx import _onnx_type_to_numpy

        result = _onnx_type_to_numpy("unknown_type")
        assert result is None

    def test_onnx_type_to_numpy_none(self):
        """Test that None input returns None."""
        from sleap_nn.export.predictors.onnx import _onnx_type_to_numpy

        result = _onnx_type_to_numpy(None)
        assert result is None

    def test_as_numpy_from_numpy(self):
        """Test that numpy array passes through."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        arr = np.random.rand(2, 1, 64, 64).astype(np.float32)
        result = _as_numpy(arr)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_as_numpy_from_tensor(self):
        """Test that torch.Tensor is converted to numpy."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        tensor = torch.rand(2, 1, 64, 64)
        result = _as_numpy(tensor)

        assert isinstance(result, np.ndarray)

    def test_as_numpy_preserves_dtype(self):
        """Test that expected_dtype is respected."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        arr = np.random.rand(2, 1, 64, 64).astype(np.float64)
        result = _as_numpy(arr, expected_dtype=np.float32)

        assert result.dtype == np.float32


class TestSelectProviders:
    """Tests for _select_providers helper function."""

    def test_select_providers_cpu(self):
        """Test that 'cpu' device returns CPUExecutionProvider."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        result = list(_select_providers("cpu", available))

        assert result == ["CPUExecutionProvider"]

    def test_select_providers_auto_no_cuda(self):
        """Test that 'auto' falls back to CPU if no CUDA."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["CPUExecutionProvider"]
        result = list(_select_providers("auto", available))

        assert "CPUExecutionProvider" in result

    def test_select_providers_cuda(self):
        """Test that 'cuda' prefers CUDAExecutionProvider."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        result = list(_select_providers("cuda", available))

        assert result[0] == "CUDAExecutionProvider"

    def test_select_providers_directml(self):
        """Test that 'directml' prefers DmlExecutionProvider."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["DmlExecutionProvider", "CPUExecutionProvider"]
        result = list(_select_providers("directml", available))

        assert result[0] == "DmlExecutionProvider"

    def test_select_providers_dml_alias(self):
        """Test that 'dml' is an alias for directml."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["DmlExecutionProvider", "CPUExecutionProvider"]
        result = list(_select_providers("dml", available))

        assert result[0] == "DmlExecutionProvider"

    def test_select_providers_directml_fallback(self):
        """Test directml falls back to CPU if DML not available."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["CPUExecutionProvider"]
        result = list(_select_providers("directml", available))

        assert result == ["CPUExecutionProvider"]

    def test_select_providers_unknown_device(self):
        """Test that unknown device returns all available providers."""
        from sleap_nn.export.predictors.onnx import _select_providers

        available = ["SomeProvider", "OtherProvider"]
        result = list(_select_providers("unknown", available))

        assert result == available


class TestAsNumpy:
    """Tests for _as_numpy helper function."""

    def test_as_numpy_with_expected_dtype(self):
        """Test dtype conversion when expected_dtype is specified."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = _as_numpy(data, expected_dtype=np.float64)

        assert result.dtype == np.float64

    def test_as_numpy_with_matching_dtype(self):
        """Test no conversion when dtype matches."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        data = np.array([1.0, 2.0], dtype=np.float32)
        result = _as_numpy(data, expected_dtype=np.float32)

        assert result.dtype == np.float32
        # Should be same array if no conversion needed
        assert np.shares_memory(result, data)

    def test_as_numpy_non_float32_default(self):
        """Test conversion to float32 when no expected_dtype and input is not float32."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        data = np.array([1, 2, 3], dtype=np.int32)
        result = _as_numpy(data)

        assert result.dtype == np.float32

    def test_as_numpy_from_list(self):
        """Test conversion from Python list."""
        from sleap_nn.export.predictors.onnx import _as_numpy

        data = [[1.0, 2.0], [3.0, 4.0]]
        result = _as_numpy(data)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


@requires_onnx
@requires_onnxruntime
class TestONNXPredictor:
    """Tests for ONNXPredictor class (requires both onnx and onnxruntime)."""

    @pytest.fixture
    def predictor(self, exported_onnx_model):
        """Create ONNXPredictor from exported model."""
        from sleap_nn.export.predictors import ONNXPredictor

        return ONNXPredictor(str(exported_onnx_model), device="cpu")

    def test_onnx_predictor_init_loads_model(self, exported_onnx_model):
        """Test that session is created successfully."""
        from sleap_nn.export.predictors import ONNXPredictor

        predictor = ONNXPredictor(str(exported_onnx_model), device="cpu")

        assert predictor.session is not None

    def test_onnx_predictor_input_name(self, predictor):
        """Test that correct input name is extracted."""
        assert predictor.input_name == "image"

    def test_onnx_predictor_output_names(self, predictor):
        """Test that output names are extracted."""
        assert isinstance(predictor.output_names, list)
        assert len(predictor.output_names) > 0
        # Should have peaks and peak_vals from SingleInstanceONNXWrapper
        assert "peaks" in predictor.output_names or len(predictor.output_names) >= 2

    def test_onnx_predictor_predict_returns_dict(self, predictor):
        """Test that predict returns dict with expected keys."""
        image = np.random.randint(0, 256, (1, 1, 64, 64), dtype=np.uint8)
        output = predictor.predict(image)

        assert isinstance(output, dict)
        for name in predictor.output_names:
            assert name in output

    def test_onnx_predictor_predict_shapes(self, predictor):
        """Test that output shapes match expected dimensions."""
        batch_size = 2
        image = np.random.randint(0, 256, (batch_size, 1, 64, 64), dtype=np.uint8)
        output = predictor.predict(image)

        # All outputs should have matching batch dimension
        for name, arr in output.items():
            assert arr.shape[0] == batch_size

    def test_onnx_predictor_predict_batch_1(self, predictor):
        """Test single image inference."""
        image = np.random.randint(0, 256, (1, 1, 64, 64), dtype=np.uint8)
        output = predictor.predict(image)

        assert isinstance(output, dict)
        assert all(arr.shape[0] == 1 for arr in output.values())

    def test_onnx_predictor_predict_batch_4(self, predictor):
        """Test batch of 4 images inference."""
        image = np.random.randint(0, 256, (4, 1, 64, 64), dtype=np.uint8)
        output = predictor.predict(image)

        assert isinstance(output, dict)
        assert all(arr.shape[0] == 4 for arr in output.values())

    def test_onnx_predictor_benchmark(self, predictor):
        """Test that benchmark returns timing dict."""
        image = np.random.randint(0, 256, (1, 1, 64, 64), dtype=np.uint8)
        result = predictor.benchmark(image, n_warmup=2, n_runs=5)

        assert isinstance(result, dict)
        assert "latency_ms_mean" in result
        assert "latency_ms_p50" in result
        assert "latency_ms_p95" in result
        assert "fps" in result

        # Values should be positive
        assert result["latency_ms_mean"] > 0
        assert result["fps"] > 0

    def test_onnx_predictor_accepts_torch_tensor(self, predictor):
        """Test that predictor accepts torch.Tensor input."""
        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = predictor.predict(image)

        assert isinstance(output, dict)
        assert all(isinstance(arr, np.ndarray) for arr in output.values())
