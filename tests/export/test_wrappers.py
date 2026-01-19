"""Tests for sleap_nn.export.wrappers module."""

import pytest
import torch

from sleap_nn.export.wrappers import (
    SingleInstanceONNXWrapper,
    CentroidONNXWrapper,
    CenteredInstanceONNXWrapper,
)
from sleap_nn.export.wrappers.base import BaseExportWrapper


class TestBaseExportWrapperNormalize:
    """Tests for _normalize_uint8 method."""

    def test_normalize_uint8_converts_dtype(self, dummy_uint8_image):
        """Test that uint8 input is converted to float32."""
        result = BaseExportWrapper._normalize_uint8(dummy_uint8_image)
        assert result.dtype == torch.float32

    def test_normalize_uint8_scales_values(self, dummy_uint8_image):
        """Test that values are scaled from [0,255] to [0,1]."""
        result = BaseExportWrapper._normalize_uint8(dummy_uint8_image)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Check specific scaling
        test_input = torch.tensor([[[0, 127, 255]]], dtype=torch.uint8)
        result = BaseExportWrapper._normalize_uint8(test_input)
        assert result[0, 0, 0].item() == 0.0
        assert abs(result[0, 0, 1].item() - 127 / 255) < 1e-5
        assert result[0, 0, 2].item() == 1.0

    def test_normalize_uint8_preserves_float(self, dummy_float32_image):
        """Test that float32 input passes through correctly."""
        result = BaseExportWrapper._normalize_uint8(dummy_float32_image * 255)
        assert result.dtype == torch.float32
        # Values should be divided by 255
        expected = dummy_float32_image * 255 / 255
        assert torch.allclose(result, expected)


class TestBaseExportWrapperExtractTensor:
    """Tests for _extract_tensor method."""

    def test_extract_tensor_from_dict(self):
        """Test extracting tensor by key hint from dict."""
        output = {
            "SingleInstanceConfmapsHead": torch.rand(1, 5, 16, 16),
            "other_output": torch.rand(1, 10),
        }
        result = BaseExportWrapper._extract_tensor(output, ["confmap", "head"])

        assert result.shape == (1, 5, 16, 16)

    def test_extract_tensor_fallback(self):
        """Test that extraction falls back to first value if no hint matches."""
        output = {
            "first_key": torch.rand(1, 3, 8, 8),
            "second_key": torch.rand(1, 10),
        }
        result = BaseExportWrapper._extract_tensor(output, ["nonexistent"])

        # Should return first value
        assert result.shape == (1, 3, 8, 8)

    def test_extract_tensor_passthrough(self):
        """Test that non-dict input passes through."""
        tensor = torch.rand(1, 5, 16, 16)
        result = BaseExportWrapper._extract_tensor(tensor, ["confmap"])

        assert result is tensor


class TestBaseExportWrapperPeakFinding:
    """Tests for peak finding methods."""

    def test_find_topk_peaks_shapes(self, synthetic_confmaps):
        """Test that _find_topk_peaks returns correct shapes."""
        # Reduce to single channel for topk (it operates on flattened spatial dims)
        confmaps_single = synthetic_confmaps[:, :1, :, :]  # (2, 1, 64, 64)

        k = 5
        peaks, values, valid = BaseExportWrapper._find_topk_peaks(confmaps_single, k)

        batch_size = confmaps_single.shape[0]
        assert peaks.shape == (batch_size, k, 2)
        assert values.shape == (batch_size, k)
        assert valid.shape == (batch_size, k)

    def test_find_topk_peaks_finds_maximum(self):
        """Test that _find_topk_peaks finds known peak location."""
        confmaps = torch.zeros(1, 1, 32, 32)
        confmaps[0, 0, 16, 20] = 1.0  # Peak at (x=20, y=16)

        peaks, values, valid = BaseExportWrapper._find_topk_peaks(confmaps, k=1)

        assert valid[0, 0].item()
        assert values[0, 0].item() == 1.0
        assert peaks[0, 0, 0].item() == 20.0  # x
        assert peaks[0, 0, 1].item() == 16.0  # y

    def test_find_topk_peaks_nms(self, synthetic_confmaps_with_nms):
        """Test that NMS suppresses adjacent peaks."""
        # Use only first channel
        confmaps = synthetic_confmaps_with_nms[:, :1, :, :]

        peaks, values, valid = BaseExportWrapper._find_topk_peaks(confmaps, k=3)

        # Only the strongest peak (value 1.0) should be detected as valid
        # Adjacent weaker peaks should be suppressed by max pooling NMS
        assert valid[0, 0].item()  # First peak should be valid
        assert values[0, 0].item() == 1.0

    def test_find_topk_peaks_per_node_shapes(self, synthetic_confmaps):
        """Test that _find_topk_peaks_per_node returns correct shapes."""
        batch_size, n_nodes, height, width = synthetic_confmaps.shape
        k = 3

        peaks, values, valid = BaseExportWrapper._find_topk_peaks_per_node(
            synthetic_confmaps, k
        )

        assert peaks.shape == (batch_size, n_nodes, k, 2)
        assert values.shape == (batch_size, n_nodes, k)
        assert valid.shape == (batch_size, n_nodes, k)

    def test_find_global_peaks_shapes(self, synthetic_confmaps):
        """Test that _find_global_peaks returns correct shapes."""
        batch_size, n_nodes, _, _ = synthetic_confmaps.shape

        peaks, values = BaseExportWrapper._find_global_peaks(synthetic_confmaps)

        assert peaks.shape == (batch_size, n_nodes, 2)
        assert values.shape == (batch_size, n_nodes)

    def test_find_global_peaks_finds_maximum(self):
        """Test that _find_global_peaks finds known peak per channel."""
        confmaps = torch.zeros(1, 3, 32, 32)
        # Place peaks at known locations for each channel
        confmaps[0, 0, 5, 10] = 1.0  # Channel 0: (x=10, y=5)
        confmaps[0, 1, 15, 20] = 2.0  # Channel 1: (x=20, y=15)
        confmaps[0, 2, 25, 8] = 0.5  # Channel 2: (x=8, y=25)

        peaks, values = BaseExportWrapper._find_global_peaks(confmaps)

        assert peaks[0, 0, 0].item() == 10.0  # x
        assert peaks[0, 0, 1].item() == 5.0  # y
        assert values[0, 0].item() == 1.0

        assert peaks[0, 1, 0].item() == 20.0
        assert peaks[0, 1, 1].item() == 15.0
        assert values[0, 1].item() == 2.0

        assert peaks[0, 2, 0].item() == 8.0
        assert peaks[0, 2, 1].item() == 25.0
        assert values[0, 2].item() == 0.5


class TestSingleInstanceONNXWrapper:
    """Tests for SingleInstanceONNXWrapper."""

    def test_single_instance_wrapper_init(self, mock_backbone):
        """Test constructor with parameters."""
        wrapper = SingleInstanceONNXWrapper(
            mock_backbone, output_stride=4, input_scale=1.0
        )

        assert wrapper.output_stride == 4
        assert wrapper.input_scale == 1.0

    def test_single_instance_wrapper_forward_shapes(
        self, mock_single_instance_wrapper, dummy_uint8_image
    ):
        """Test that forward returns dict with expected keys and shapes."""
        # Adjust image size to match mock backbone expectations
        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = mock_single_instance_wrapper(image)

        assert isinstance(output, dict)
        assert "peaks" in output
        assert "peak_vals" in output

        batch_size = image.shape[0]
        n_nodes = mock_single_instance_wrapper.model.n_nodes

        assert output["peaks"].shape == (batch_size, n_nodes, 2)
        assert output["peak_vals"].shape == (batch_size, n_nodes)

    def test_single_instance_wrapper_forward_dtypes(
        self, mock_single_instance_wrapper, dummy_uint8_image
    ):
        """Test that output tensors have correct dtypes."""
        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = mock_single_instance_wrapper(image)

        assert output["peaks"].dtype == torch.float32
        assert output["peak_vals"].dtype == torch.float32

    def test_single_instance_wrapper_input_scale(self, mock_backbone):
        """Test that input_scale parameter is stored correctly."""
        wrapper = SingleInstanceONNXWrapper(
            mock_backbone, output_stride=4, input_scale=0.5
        )
        assert wrapper.input_scale == 0.5


class TestCentroidONNXWrapper:
    """Tests for CentroidONNXWrapper."""

    def test_centroid_wrapper_forward_shapes(
        self, mock_centroid_wrapper, dummy_uint8_image
    ):
        """Test that forward returns expected output shapes."""
        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = mock_centroid_wrapper(image)

        assert isinstance(output, dict)
        assert "centroids" in output
        assert "centroid_vals" in output
        assert (
            "instance_valid" in output
        )  # Note: uses instance_valid, not centroid_mask

        batch_size = image.shape[0]
        max_instances = mock_centroid_wrapper.max_instances

        assert output["centroids"].shape == (batch_size, max_instances, 2)
        assert output["centroid_vals"].shape == (batch_size, max_instances)
        assert output["instance_valid"].shape == (batch_size, max_instances)

    def test_centroid_wrapper_max_instances(self, mock_centroid_backbone):
        """Test that max_instances parameter is respected."""
        wrapper = CentroidONNXWrapper(
            mock_centroid_backbone, output_stride=4, max_instances=5
        )
        assert wrapper.max_instances == 5

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["centroids"].shape[1] == 5

    def test_centroid_wrapper_input_scale(self, mock_centroid_backbone):
        """Test centroid wrapper with input_scale."""
        wrapper = CentroidONNXWrapper(
            mock_centroid_backbone,
            output_stride=4,
            max_instances=5,
            input_scale=0.5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["centroids"].shape == (1, 5, 2)


class TestCenteredInstanceONNXWrapper:
    """Tests for CenteredInstanceONNXWrapper."""

    def test_centered_instance_wrapper_forward_shapes(self, mock_backbone):
        """Test that forward returns expected output shapes."""
        wrapper = CenteredInstanceONNXWrapper(
            mock_backbone,
            output_stride=4,
        )

        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert isinstance(output, dict)
        assert "peaks" in output
        assert "peak_vals" in output

        batch_size = image.shape[0]
        n_nodes = mock_backbone.n_nodes

        assert output["peaks"].shape == (batch_size, n_nodes, 2)
        assert output["peak_vals"].shape == (batch_size, n_nodes)

    def test_centered_instance_wrapper_input_scale(self, mock_backbone):
        """Test that input_scale correctly resizes input."""
        wrapper = CenteredInstanceONNXWrapper(
            mock_backbone,
            output_stride=4,
            input_scale=0.5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        # Should still produce correct output shapes
        assert output["peaks"].shape == (1, mock_backbone.n_nodes, 2)


class TestTopDownONNXWrapper:
    """Tests for TopDownONNXWrapper."""

    def test_topdown_wrapper_forward_shapes(
        self, mock_centroid_backbone, mock_backbone
    ):
        """Test that combined TopDownONNXWrapper returns expected shapes."""
        from sleap_nn.export.wrappers import TopDownONNXWrapper

        wrapper = TopDownONNXWrapper(
            centroid_model=mock_centroid_backbone,
            instance_model=mock_backbone,
            centroid_output_stride=4,
            instance_output_stride=4,
            crop_size=(32, 32),
            max_instances=10,
            n_nodes=mock_backbone.n_nodes,
        )

        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert isinstance(output, dict)
        assert "peaks" in output
        assert "peak_vals" in output
        assert "instance_valid" in output  # Note: uses instance_valid, not peak_mask
        assert "centroids" in output
        assert "centroid_vals" in output

        batch_size = image.shape[0]
        max_instances = wrapper.max_instances
        n_nodes = mock_backbone.n_nodes

        assert output["peaks"].shape == (batch_size, max_instances, n_nodes, 2)
        assert output["peak_vals"].shape == (batch_size, max_instances, n_nodes)
        assert output["instance_valid"].shape == (batch_size, max_instances)

    def test_topdown_wrapper_input_scale(self, mock_centroid_backbone, mock_backbone):
        """Test TopDownONNXWrapper with centroid and instance input scaling."""
        from sleap_nn.export.wrappers import TopDownONNXWrapper

        wrapper = TopDownONNXWrapper(
            centroid_model=mock_centroid_backbone,
            instance_model=mock_backbone,
            centroid_output_stride=4,
            instance_output_stride=4,
            crop_size=(32, 32),
            max_instances=5,
            n_nodes=mock_backbone.n_nodes,
            centroid_input_scale=0.5,
            instance_input_scale=0.5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["peaks"].shape == (1, 5, mock_backbone.n_nodes, 2)


class TestBottomUpONNXWrapper:
    """Tests for BottomUpONNXWrapper."""

    def test_bottomup_wrapper_forward_shapes(self, mock_bottomup_backbone):
        """Test that BottomUpONNXWrapper returns expected shapes."""
        from sleap_nn.export.wrappers import BottomUpONNXWrapper

        n_nodes = mock_bottomup_backbone.n_nodes
        n_edges = mock_bottomup_backbone.n_edges
        max_peaks = 10

        wrapper = BottomUpONNXWrapper(
            model=mock_bottomup_backbone,
            cms_output_stride=4,
            pafs_output_stride=8,
            n_nodes=n_nodes,
            skeleton_edges=[(i, i + 1) for i in range(n_edges)],
            max_peaks_per_node=max_peaks,
        )

        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert isinstance(output, dict)
        assert "peaks" in output
        assert "peak_vals" in output
        assert "peak_mask" in output
        assert "line_scores" in output
        assert "candidate_mask" in output

        batch_size = image.shape[0]

        assert output["peaks"].shape == (batch_size, n_nodes, max_peaks, 2)
        assert output["peak_vals"].shape == (batch_size, n_nodes, max_peaks)
        assert output["peak_mask"].shape == (batch_size, n_nodes, max_peaks)
        assert output["line_scores"].shape == (
            batch_size,
            n_edges,
            max_peaks * max_peaks,
        )

    def test_bottomup_wrapper_no_confmaps_pafs_output(self, mock_bottomup_backbone):
        """Test that confmaps/pafs are NOT in output (D2H optimization)."""
        from sleap_nn.export.wrappers import BottomUpONNXWrapper

        wrapper = BottomUpONNXWrapper(
            model=mock_bottomup_backbone,
            cms_output_stride=4,
            pafs_output_stride=8,
            n_nodes=mock_bottomup_backbone.n_nodes,
            skeleton_edges=[(i, i + 1) for i in range(mock_bottomup_backbone.n_edges)],
            max_peaks_per_node=10,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        # These should NOT be in output to minimize D2H transfer
        assert "confmaps" not in output
        assert "pafs" not in output

    def test_bottomup_wrapper_line_scores_shape(self, mock_bottomup_backbone):
        """Test that line_scores has shape (B, n_edges, k*k)."""
        from sleap_nn.export.wrappers import BottomUpONNXWrapper

        n_edges = mock_bottomup_backbone.n_edges
        max_peaks = 5

        wrapper = BottomUpONNXWrapper(
            model=mock_bottomup_backbone,
            cms_output_stride=4,
            pafs_output_stride=8,
            n_nodes=mock_bottomup_backbone.n_nodes,
            skeleton_edges=[(i, i + 1) for i in range(n_edges)],
            max_peaks_per_node=max_peaks,
        )

        image = torch.randint(0, 256, (3, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["line_scores"].shape == (3, n_edges, max_peaks * max_peaks)

    def test_bottomup_wrapper_input_scale(self, mock_bottomup_backbone):
        """Test BottomUpONNXWrapper with input_scale."""
        from sleap_nn.export.wrappers import BottomUpONNXWrapper

        wrapper = BottomUpONNXWrapper(
            model=mock_bottomup_backbone,
            cms_output_stride=4,
            pafs_output_stride=8,
            n_nodes=mock_bottomup_backbone.n_nodes,
            skeleton_edges=[(i, i + 1) for i in range(mock_bottomup_backbone.n_edges)],
            max_peaks_per_node=5,
            input_scale=0.5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["peaks"].shape[0] == 1

    def test_bottomup_wrapper_tuple_output(self):
        """Test BottomUpONNXWrapper when model returns tuple instead of dict."""
        from sleap_nn.export.wrappers import BottomUpONNXWrapper

        # Create a model that returns tuple instead of dict
        class TupleOutputBackbone(torch.nn.Module):
            def __init__(self, n_nodes=5, n_edges=4, output_stride=4, pafs_stride=8):
                super().__init__()
                self.n_nodes = n_nodes
                self.n_edges = n_edges
                self.output_stride = output_stride
                self.pafs_stride = pafs_stride

            def forward(self, x):
                b, c, h, w = x.shape
                cms_h, cms_w = h // self.output_stride, w // self.output_stride
                pafs_h, pafs_w = h // self.pafs_stride, w // self.pafs_stride
                # Return as tuple instead of dict
                confmaps = torch.rand(b, self.n_nodes, cms_h, cms_w)
                pafs = torch.rand(b, self.n_edges * 2, pafs_h, pafs_w)
                return (confmaps, pafs)

        backbone = TupleOutputBackbone()
        wrapper = BottomUpONNXWrapper(
            model=backbone,
            cms_output_stride=4,
            pafs_output_stride=8,
            n_nodes=backbone.n_nodes,
            skeleton_edges=[(i, i + 1) for i in range(backbone.n_edges)],
            max_peaks_per_node=5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert "peaks" in output
        assert "line_scores" in output


class TestMultiClassBottomUpONNXWrapper:
    """Tests for BottomUpMultiClassONNXWrapper."""

    def test_multiclass_bottomup_wrapper_forward_shapes(
        self, mock_multiclass_bottomup_backbone
    ):
        """Test that multiclass bottom-up wrapper includes class_probs."""
        from sleap_nn.export.wrappers import BottomUpMultiClassONNXWrapper

        n_nodes = mock_multiclass_bottomup_backbone.n_nodes
        n_classes = mock_multiclass_bottomup_backbone.n_classes
        max_peaks = 10

        wrapper = BottomUpMultiClassONNXWrapper(
            model=mock_multiclass_bottomup_backbone,
            cms_output_stride=4,
            class_maps_output_stride=8,
            n_nodes=n_nodes,
            n_classes=n_classes,
            max_peaks_per_node=max_peaks,
        )

        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert isinstance(output, dict)
        assert "peaks" in output
        assert "peak_vals" in output
        assert "peak_mask" in output
        assert "class_probs" in output

        batch_size = image.shape[0]

        assert output["peaks"].shape == (batch_size, n_nodes, max_peaks, 2)
        assert output["class_probs"].shape == (
            batch_size,
            n_nodes,
            max_peaks,
            n_classes,
        )

    def test_multiclass_bottomup_wrapper_input_scale(
        self, mock_multiclass_bottomup_backbone
    ):
        """Test multiclass bottom-up wrapper with input_scale."""
        from sleap_nn.export.wrappers import BottomUpMultiClassONNXWrapper

        wrapper = BottomUpMultiClassONNXWrapper(
            model=mock_multiclass_bottomup_backbone,
            cms_output_stride=4,
            class_maps_output_stride=8,
            n_nodes=mock_multiclass_bottomup_backbone.n_nodes,
            n_classes=mock_multiclass_bottomup_backbone.n_classes,
            max_peaks_per_node=5,
            input_scale=0.5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["peaks"].shape[0] == 1
        assert output["class_probs"].shape[0] == 1


class TestMultiClassTopDownONNXWrapper:
    """Tests for TopDownMultiClassONNXWrapper."""

    def test_multiclass_topdown_wrapper_forward_shapes(self, mock_backbone):
        """Test that multiclass top-down wrapper includes class_logits."""
        from sleap_nn.export.wrappers import TopDownMultiClassONNXWrapper

        # Create a mock model that returns confmaps + class_vectors
        class MockMultiClassTopDownModel(torch.nn.Module):
            def __init__(self, n_nodes=5, n_classes=2, output_stride=4):
                super().__init__()
                self.n_nodes = n_nodes
                self.n_classes = n_classes
                self.output_stride = output_stride
                self.conv = torch.nn.Conv2d(1, n_nodes, 1)

            def forward(self, x):
                b, c, h, w = x.shape
                out_h, out_w = h // self.output_stride, w // self.output_stride
                return {
                    "CenteredInstanceConfmapsHead": torch.rand(
                        b, self.n_nodes, out_h, out_w
                    ),
                    "ClassVectorsHead": torch.rand(b, self.n_classes),
                }

        model = MockMultiClassTopDownModel(n_nodes=5, n_classes=2)

        wrapper = TopDownMultiClassONNXWrapper(
            model=model,
            output_stride=4,
            n_classes=2,
        )

        image = torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert isinstance(output, dict)
        assert "peaks" in output
        assert "peak_vals" in output
        assert "class_logits" in output

        batch_size = image.shape[0]
        n_nodes = model.n_nodes
        n_classes = model.n_classes

        assert output["peaks"].shape == (batch_size, n_nodes, 2)
        assert output["class_logits"].shape == (batch_size, n_classes)

    def test_multiclass_topdown_wrapper_input_scale(self, mock_backbone):
        """Test multiclass top-down wrapper with input_scale."""
        from sleap_nn.export.wrappers import TopDownMultiClassONNXWrapper

        class MockMultiClassTopDownModel(torch.nn.Module):
            def __init__(self, n_nodes=5, n_classes=2, output_stride=4):
                super().__init__()
                self.n_nodes = n_nodes
                self.n_classes = n_classes
                self.output_stride = output_stride

            def forward(self, x):
                b, c, h, w = x.shape
                out_h, out_w = h // self.output_stride, w // self.output_stride
                return {
                    "CenteredInstanceConfmapsHead": torch.rand(
                        b, self.n_nodes, out_h, out_w
                    ),
                    "ClassVectorsHead": torch.rand(b, self.n_classes),
                }

        model = MockMultiClassTopDownModel(n_nodes=5, n_classes=2)

        wrapper = TopDownMultiClassONNXWrapper(
            model=model,
            output_stride=4,
            n_classes=2,
            input_scale=0.5,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        assert output["peaks"].shape == (1, 5, 2)
        assert output["class_logits"].shape == (1, 2)


class TestTopDownMultiClassCombinedONNXWrapper:
    """Tests for TopDownMultiClassCombinedONNXWrapper."""

    def test_combined_multiclass_topdown_wrapper_forward_shapes(self):
        """Test that combined multiclass top-down wrapper has correct output shapes."""
        from sleap_nn.export.wrappers.topdown_multiclass import (
            TopDownMultiClassCombinedONNXWrapper,
        )

        # Create mock centroid model
        class MockCentroidModel(torch.nn.Module):
            def __init__(self, output_stride=4):
                super().__init__()
                self.output_stride = output_stride
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                b, c, h, w = x.shape
                out_h, out_w = h // self.output_stride, w // self.output_stride
                return {"CentroidConfmapsHead": torch.rand(b, 1, out_h, out_w)}

        # Create mock instance model with class vectors
        class MockInstanceModel(torch.nn.Module):
            def __init__(self, n_nodes=5, n_classes=2, output_stride=2):
                super().__init__()
                self.n_nodes = n_nodes
                self.n_classes = n_classes
                self.output_stride = output_stride
                self.conv = torch.nn.Conv2d(1, n_nodes, 1)

            def forward(self, x):
                b, c, h, w = x.shape
                out_h, out_w = h // self.output_stride, w // self.output_stride
                return {
                    "CenteredInstanceConfmapsHead": torch.rand(
                        b, self.n_nodes, out_h, out_w
                    ),
                    "ClassVectorsHead": torch.rand(b, self.n_classes),
                }

        centroid_model = MockCentroidModel(output_stride=4)
        instance_model = MockInstanceModel(n_nodes=5, n_classes=2, output_stride=2)

        wrapper = TopDownMultiClassCombinedONNXWrapper(
            centroid_model=centroid_model,
            instance_model=instance_model,
            max_instances=10,
            crop_size=(64, 64),
            centroid_output_stride=4,
            instance_output_stride=2,
            n_nodes=5,
            n_classes=2,
        )

        # Test forward pass
        image = torch.randint(0, 256, (2, 1, 128, 128), dtype=torch.uint8)
        output = wrapper(image)

        assert isinstance(output, dict)
        assert "centroids" in output
        assert "centroid_vals" in output
        assert "peaks" in output
        assert "peak_vals" in output
        assert "class_logits" in output
        assert "instance_valid" in output

        batch_size = 2
        max_instances = 10
        n_nodes = 5
        n_classes = 2

        assert output["centroids"].shape == (batch_size, max_instances, 2)
        assert output["centroid_vals"].shape == (batch_size, max_instances)
        assert output["peaks"].shape == (batch_size, max_instances, n_nodes, 2)
        assert output["peak_vals"].shape == (batch_size, max_instances, n_nodes)
        assert output["class_logits"].shape == (batch_size, max_instances, n_classes)
        assert output["instance_valid"].shape == (batch_size, max_instances)

    def test_combined_multiclass_topdown_wrapper_input_scaling(self):
        """Test combined wrapper with input scaling."""
        from sleap_nn.export.wrappers.topdown_multiclass import (
            TopDownMultiClassCombinedONNXWrapper,
        )

        class MockCentroidModel(torch.nn.Module):
            def __init__(self, output_stride=4):
                super().__init__()
                self.output_stride = output_stride
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, x):
                b, c, h, w = x.shape
                out_h, out_w = h // self.output_stride, w // self.output_stride
                return {"CentroidConfmapsHead": torch.rand(b, 1, out_h, out_w)}

        class MockInstanceModel(torch.nn.Module):
            def __init__(self, n_nodes=3, n_classes=2, output_stride=2):
                super().__init__()
                self.n_nodes = n_nodes
                self.n_classes = n_classes
                self.output_stride = output_stride

            def forward(self, x):
                b, c, h, w = x.shape
                out_h, out_w = h // self.output_stride, w // self.output_stride
                return {
                    "CenteredInstanceConfmapsHead": torch.rand(
                        b, self.n_nodes, out_h, out_w
                    ),
                    "ClassVectorsHead": torch.rand(b, self.n_classes),
                }

        wrapper = TopDownMultiClassCombinedONNXWrapper(
            centroid_model=MockCentroidModel(),
            instance_model=MockInstanceModel(),
            max_instances=5,
            crop_size=(32, 32),
            centroid_input_scale=0.5,
            instance_input_scale=0.5,
            n_nodes=3,
            n_classes=2,
        )

        image = torch.randint(0, 256, (1, 1, 64, 64), dtype=torch.uint8)
        output = wrapper(image)

        # Just verify it runs and has correct output shapes
        assert output["peaks"].shape == (1, 5, 3, 2)
        assert output["class_logits"].shape == (1, 5, 2)
