"""Tests for sleap_nn.export.exporters.onnx_exporter module.

These tests require the `onnx` package but NOT `onnxruntime`.
"""

import pytest
import torch

from .conftest import requires_onnx


@requires_onnx
class TestExportToONNX:
    """Tests for export_to_onnx function."""

    def test_export_to_onnx_creates_file(self, mock_single_instance_wrapper, tmp_path):
        """Test that export creates an ONNX file."""
        from sleap_nn.export.exporters import export_to_onnx

        onnx_path = tmp_path / "model.onnx"
        result = export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            verify=False,
        )

        assert onnx_path.exists()
        assert result == onnx_path
        assert onnx_path.stat().st_size > 0

    def test_export_to_onnx_valid_model(self, mock_single_instance_wrapper, tmp_path):
        """Test that exported model passes onnx.checker.check_model."""
        from sleap_nn.export.exporters import export_to_onnx
        import onnx

        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            verify=True,  # This calls onnx.checker.check_model
        )

        # Additional verification
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_export_to_onnx_has_inputs(self, mock_single_instance_wrapper, tmp_path):
        """Test that model has expected input name."""
        from sleap_nn.export.exporters import export_to_onnx
        import onnx

        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            input_names=["image"],
            verify=False,
        )

        model = onnx.load(str(onnx_path))
        input_names = [inp.name for inp in model.graph.input]

        assert "image" in input_names

    def test_export_to_onnx_has_outputs(self, mock_single_instance_wrapper, tmp_path):
        """Test that model has expected output names."""
        from sleap_nn.export.exporters import export_to_onnx
        import onnx

        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            output_names=["peaks", "peak_vals"],
            verify=False,
        )

        model = onnx.load(str(onnx_path))
        output_names = [out.name for out in model.graph.output]

        assert "peaks" in output_names
        assert "peak_vals" in output_names

    def test_export_to_onnx_dynamic_batch(self, mock_single_instance_wrapper, tmp_path):
        """Test that batch dimension is dynamic."""
        from sleap_nn.export.exporters import export_to_onnx
        import onnx

        onnx_path = tmp_path / "model.onnx"
        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}},
            verify=False,
        )

        model = onnx.load(str(onnx_path))

        # Check that input has dynamic batch dimension
        input_shape = model.graph.input[0].type.tensor_type.shape
        batch_dim = input_shape.dim[0]

        # Dynamic dimension should have dim_param set (not dim_value)
        assert batch_dim.HasField("dim_param") or batch_dim.dim_value == 0

    def test_export_to_onnx_opset_version(self, mock_single_instance_wrapper, tmp_path):
        """Test that opset_version parameter is respected."""
        from sleap_nn.export.exporters import export_to_onnx
        import onnx

        onnx_path = tmp_path / "model.onnx"
        target_opset = 15

        export_to_onnx(
            mock_single_instance_wrapper,
            onnx_path,
            input_shape=(1, 1, 64, 64),
            opset_version=target_opset,
            verify=False,
        )

        model = onnx.load(str(onnx_path))

        # Find the main opset version
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                assert opset.version == target_opset
                break


class TestInferOutputNames:
    """Tests for _infer_output_names helper function."""

    def test_infer_output_names_dict(self):
        """Test inferring output names from dict output."""
        from sleap_nn.export.exporters.onnx_exporter import _infer_output_names

        output = {
            "peaks": torch.rand(1, 5, 2),
            "peak_vals": torch.rand(1, 5),
        }
        names = _infer_output_names(output)

        assert names == ["peaks", "peak_vals"]

    def test_infer_output_names_tuple(self):
        """Test inferring output names from tuple output."""
        from sleap_nn.export.exporters.onnx_exporter import _infer_output_names

        output = (torch.rand(1, 5, 2), torch.rand(1, 5))
        names = _infer_output_names(output)

        assert names == ["output_0", "output_1"]

    def test_infer_output_names_single(self):
        """Test inferring output name from single tensor."""
        from sleap_nn.export.exporters.onnx_exporter import _infer_output_names

        output = torch.rand(1, 5, 2)
        names = _infer_output_names(output)

        assert names == ["output_0"]


class TestExportModelFactory:
    """Tests for export_model factory function."""

    @requires_onnx
    def test_export_model_onnx(self, mock_single_instance_wrapper, tmp_path):
        """Test export_model with fmt='onnx'."""
        from sleap_nn.export.exporters import export_model

        onnx_path = tmp_path / "model.onnx"
        result = export_model(
            mock_single_instance_wrapper,
            onnx_path,
            fmt="onnx",
            input_shape=(1, 1, 64, 64),
            verify=False,
        )

        assert onnx_path.exists()
        assert result == onnx_path

    def test_export_model_unknown_format(self, mock_single_instance_wrapper, tmp_path):
        """Test that unknown format raises ValueError."""
        from sleap_nn.export.exporters import export_model

        with pytest.raises(ValueError, match="Unknown export format"):
            export_model(
                mock_single_instance_wrapper,
                tmp_path / "model.xyz",
                fmt="unknown_format",
                input_shape=(1, 1, 64, 64),
            )
