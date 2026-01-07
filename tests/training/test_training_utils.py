"""Tests for sleap_nn/training/utils.py."""

import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from sleap_nn.training.utils import (
    is_distributed_initialized,
    get_dist_rank,
    get_gpu_memory,
    xavier_init_weights,
    imgfig,
    plot_img,
    plot_confmaps,
    plot_peaks,
    VisualizationData,
    MatplotlibRenderer,
    WandBRenderer,
)


class TestDistributedUtils:
    """Tests for distributed training utilities."""

    def test_is_distributed_initialized_returns_false(self):
        """In a non-distributed environment, should return False."""
        # In normal pytest run, distributed is not initialized
        result = is_distributed_initialized()
        assert result is False

    def test_get_dist_rank_returns_none(self):
        """In a non-distributed environment, should return None."""
        result = get_dist_rank()
        assert result is None


class TestGetGpuMemory:
    """Tests for get_gpu_memory function."""

    def test_no_nvidia_smi(self):
        """Returns empty list when nvidia-smi is not available."""
        with patch("shutil.which", return_value=None):
            result = get_gpu_memory()
            assert result == []

    def test_nvidia_smi_success(self):
        """Returns memory list when nvidia-smi succeeds."""
        mock_output = b"index, memory.free\n0, 8192 MiB\n1, 4096 MiB\n"
        mock_result = MagicMock()
        mock_result.stdout = mock_output

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                with patch.dict("os.environ", {}, clear=True):
                    result = get_gpu_memory()
                    assert result == [8192, 4096]

    def test_nvidia_smi_with_cuda_visible_devices(self):
        """Respects CUDA_VISIBLE_DEVICES environment variable."""
        mock_output = b"index, memory.free\n0, 8192 MiB\n1, 4096 MiB\n2, 2048 MiB\n"
        mock_result = MagicMock()
        mock_result.stdout = mock_output

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", return_value=mock_result):
                with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,2"}):
                    result = get_gpu_memory()
                    assert result == [8192, 2048]

    def test_nvidia_smi_subprocess_error(self):
        """Returns empty list when subprocess fails."""
        import subprocess

        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch(
                "subprocess.run", side_effect=subprocess.SubprocessError("error")
            ):
                result = get_gpu_memory()
                assert result == []

    def test_nvidia_smi_file_not_found(self):
        """Returns empty list when FileNotFoundError is raised."""
        with patch("shutil.which", return_value="/usr/bin/nvidia-smi"):
            with patch("subprocess.run", side_effect=FileNotFoundError("not found")):
                result = get_gpu_memory()
                assert result == []


class TestXavierInitWeights:
    """Tests for xavier_init_weights function."""

    def test_conv2d_initialization(self):
        """Initializes Conv2d weights with Xavier."""
        import torch.nn as nn

        conv = nn.Conv2d(3, 16, kernel_size=3)
        xavier_init_weights(conv)
        # Just verify it doesn't crash - Xavier init is random
        assert conv.weight is not None

    def test_linear_initialization(self):
        """Initializes Linear weights with Xavier."""
        import torch.nn as nn

        linear = nn.Linear(10, 5)
        xavier_init_weights(linear)
        assert linear.weight is not None

    def test_other_module_ignored(self):
        """Non-Conv2d/Linear modules are ignored."""
        import torch.nn as nn

        bn = nn.BatchNorm2d(16)
        xavier_init_weights(bn)  # Should not crash


class TestImgfig:
    """Tests for imgfig function."""

    def test_scalar_size(self):
        """Creates figure with scalar size (square)."""
        fig = imgfig(size=4, dpi=72)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_tuple_size(self):
        """Creates figure with tuple size (width, height)."""
        fig = imgfig(size=(8, 6), dpi=72)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_list_size(self):
        """Creates figure with list size."""
        fig = imgfig(size=[8, 6], dpi=72)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_scale(self):
        """Applies scale factor to figure size."""
        fig = imgfig(size=4, dpi=72, scale=2.0)
        assert isinstance(fig, matplotlib.figure.Figure)
        # Check that figure size is scaled
        figsize = fig.get_size_inches()
        assert figsize[0] == pytest.approx(8.0, rel=0.1)
        assert figsize[1] == pytest.approx(8.0, rel=0.1)
        plt.close(fig)


class TestPlotImg:
    """Tests for plot_img function."""

    def test_rgb_image(self):
        """Plots RGB image."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_img(img)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_grayscale_image(self):
        """Plots grayscale image with single channel."""
        img = np.random.rand(64, 64, 1).astype(np.float32)
        fig = plot_img(img)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_batch_singleton_squeeze(self):
        """Squeezes batch dimension if present."""
        img = np.random.rand(1, 64, 64, 3).astype(np.float32)
        fig = plot_img(img)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_tensor_input(self):
        """Handles torch tensor input."""
        import torch

        img = torch.rand(64, 64, 3)
        fig = plot_img(img)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_normalization(self):
        """Normalizes images with values outside [0, 1]."""
        img = np.random.rand(64, 64, 3).astype(np.float32) * 255
        fig = plot_img(img)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_negative_values_normalization(self):
        """Normalizes images with negative values."""
        img = np.random.rand(64, 64, 3).astype(np.float32) - 0.5
        fig = plot_img(img)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


class TestPlotConfmaps:
    """Tests for plot_confmaps function."""

    def test_basic_confmaps(self):
        """Plots basic confidence maps."""
        # First create a figure with an image
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_img(img)

        confmaps = np.random.rand(64, 64, 5).astype(np.float32)
        result = plot_confmaps(confmaps)
        assert result is not None
        plt.close(fig)

    def test_output_scale(self):
        """Applies output scale to extent."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_img(img)

        confmaps = np.random.rand(32, 32, 5).astype(np.float32)
        result = plot_confmaps(confmaps, output_scale=0.5)
        assert result is not None
        plt.close(fig)


class TestPlotPeaks:
    """Tests for plot_peaks function."""

    def test_gt_only(self):
        """Plots ground truth points only."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_img(img)

        pts_gt = np.random.rand(2, 5, 2) * 64  # 2 instances, 5 nodes
        handles = plot_peaks(pts_gt)
        assert len(handles) > 0
        plt.close(fig)

    def test_gt_and_predictions(self):
        """Plots ground truth and predictions."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_img(img)

        pts_gt = np.random.rand(2, 5, 2) * 64
        pts_pr = np.random.rand(2, 5, 2) * 64
        handles = plot_peaks(pts_gt, pts_pr)
        assert len(handles) > 0
        plt.close(fig)

    def test_paired_error_lines(self):
        """Plots error lines when paired=True."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        fig = plot_img(img)

        pts_gt = np.random.rand(2, 5, 2) * 64
        pts_pr = np.random.rand(2, 5, 2) * 64
        handles = plot_peaks(pts_gt, pts_pr, paired=True)
        assert len(handles) > 0
        plt.close(fig)


class TestVisualizationData:
    """Tests for VisualizationData dataclass."""

    def test_basic_instantiation(self):
        """Creates VisualizationData with required fields."""
        data = VisualizationData(
            image=np.zeros((64, 64, 3)),
            pred_confmaps=np.zeros((64, 64, 5)),
            pred_peaks=np.zeros((2, 5, 2)),
            pred_peak_values=np.zeros((2, 5)),
            gt_instances=np.zeros((2, 5, 2)),
        )
        assert data.image.shape == (64, 64, 3)
        assert data.node_names == []
        assert data.output_scale == 1.0
        assert data.is_paired is True
        assert data.pred_pafs is None
        assert data.pred_class_maps is None

    def test_with_optional_fields(self):
        """Creates VisualizationData with all optional fields."""
        data = VisualizationData(
            image=np.zeros((64, 64, 3)),
            pred_confmaps=np.zeros((64, 64, 5)),
            pred_peaks=np.zeros((2, 5, 2)),
            pred_peak_values=np.zeros((2, 5)),
            gt_instances=np.zeros((2, 5, 2)),
            node_names=["head", "neck", "tail", "l_ear", "r_ear"],
            output_scale=0.5,
            is_paired=False,
            pred_pafs=np.zeros((64, 64, 8)),
            pred_class_maps=np.zeros((64, 64, 3)),
        )
        assert data.node_names == ["head", "neck", "tail", "l_ear", "r_ear"]
        assert data.output_scale == 0.5
        assert data.is_paired is False
        assert data.pred_pafs.shape == (64, 64, 8)


class TestMatplotlibRenderer:
    """Tests for MatplotlibRenderer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample VisualizationData for testing."""
        return VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
        )

    def test_render_basic(self, sample_data):
        """Renders visualization to matplotlib figure."""
        renderer = MatplotlibRenderer()
        fig = renderer.render(sample_data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_small_image_scale_2x(self):
        """Applies 2x scale for images < 512 pixels."""
        data = VisualizationData(
            image=np.random.rand(256, 256, 3).astype(np.float32),
            pred_confmaps=np.random.rand(256, 256, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 256,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 256,
        )
        renderer = MatplotlibRenderer()
        fig = renderer.render(data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_very_small_image_scale_4x(self):
        """Applies 4x scale for images < 256 pixels."""
        data = VisualizationData(
            image=np.random.rand(128, 128, 3).astype(np.float32),
            pred_confmaps=np.random.rand(128, 128, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 128,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 128,
        )
        renderer = MatplotlibRenderer()
        fig = renderer.render(data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_pafs(self, sample_data):
        """Renders PAF magnitude visualization."""
        # Add PAFs to sample data
        sample_data.pred_pafs = np.random.rand(64, 64, 8).astype(np.float32)
        renderer = MatplotlibRenderer()
        fig = renderer.render_pafs(sample_data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_pafs_4d_input(self):
        """Handles 4D PAF array (H, W, edges, 2)."""
        data = VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_pafs=np.random.rand(64, 64, 4, 2).astype(np.float32),
        )
        renderer = MatplotlibRenderer()
        fig = renderer.render_pafs(data)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_render_pafs_raises_without_pafs(self, sample_data):
        """Raises ValueError when pred_pafs is None."""
        renderer = MatplotlibRenderer()
        with pytest.raises(ValueError, match="pred_pafs is None"):
            renderer.render_pafs(sample_data)


class TestWandBRenderer:
    """Tests for WandBRenderer class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample VisualizationData for testing."""
        return VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            node_names=["head", "neck", "tail", "l_ear", "r_ear"],
        )

    def test_init_default_mode(self):
        """Initializes with default 'direct' mode."""
        renderer = WandBRenderer()
        assert renderer.mode == "direct"
        assert renderer.box_size == 5.0
        assert renderer.confmap_threshold == 0.1

    def test_init_custom_params(self):
        """Initializes with custom parameters."""
        renderer = WandBRenderer(mode="boxes", box_size=10.0, confmap_threshold=0.2)
        assert renderer.mode == "boxes"
        assert renderer.box_size == 10.0
        assert renderer.confmap_threshold == 0.2

    def test_render_direct_mode(self, sample_data):
        """Renders in direct mode using matplotlib pre-rendering."""
        mock_wandb = MagicMock()
        mock_wandb.Image = MagicMock(return_value="mock_image")

        renderer = WandBRenderer(mode="direct")
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch.object(renderer, "_render_direct") as mock_render:
                mock_render.return_value = "direct_result"
                result = renderer.render(sample_data, caption="test")
                mock_render.assert_called_once_with(sample_data, "test")

    def test_render_boxes_mode(self, sample_data):
        """Renders in boxes mode."""
        renderer = WandBRenderer(mode="boxes")
        with patch.object(renderer, "_render_with_boxes") as mock_render:
            mock_render.return_value = "boxes_result"
            result = renderer.render(sample_data, caption="test")
            mock_render.assert_called_once_with(sample_data, "test")

    def test_render_masks_mode(self, sample_data):
        """Renders in masks mode."""
        renderer = WandBRenderer(mode="masks")
        with patch.object(renderer, "_render_with_masks") as mock_render:
            mock_render.return_value = "masks_result"
            result = renderer.render(sample_data, caption="test")
            mock_render.assert_called_once_with(sample_data, "test")

    def test_peaks_to_boxes_2d_input(self):
        """Converts 2D peaks array to boxes with percent coordinates."""
        renderer = WandBRenderer()
        peaks = np.array([[10.0, 20.0], [30.0, 40.0]])  # (nodes, 2)
        node_names = ["head", "tail"]
        img_w, img_h = 100, 100  # Image dimensions for percent conversion

        boxes = renderer._peaks_to_boxes(peaks, node_names, img_w, img_h, is_gt=True)

        assert len(boxes) == 2
        # Coordinates are now in percent (0-1 range)
        assert boxes[0]["position"]["middle"] == [0.1, 0.2]  # 10/100, 20/100
        assert boxes[0]["domain"] == "percent"
        assert boxes[0]["box_caption"] == "GT: head"
        assert boxes[0]["class_id"] == 0

    def test_peaks_to_boxes_3d_input(self):
        """Converts 3D peaks array to boxes with percent coordinates."""
        renderer = WandBRenderer()
        peaks = np.array([[[10.0, 20.0], [30.0, 40.0]]])  # (1, nodes, 2)
        node_names = ["head", "tail"]
        img_w, img_h = 100, 100

        boxes = renderer._peaks_to_boxes(peaks, node_names, img_w, img_h, is_gt=False)

        assert len(boxes) == 2
        assert boxes[0]["position"]["middle"] == [0.1, 0.2]  # 10/100, 20/100
        assert boxes[0]["domain"] == "percent"
        assert boxes[0]["box_caption"] == "head"

    def test_peaks_to_boxes_with_confidence(self):
        """Includes confidence scores in boxes."""
        renderer = WandBRenderer()
        peaks = np.array([[10.0, 20.0], [30.0, 40.0]])
        peak_values = np.array([0.95, 0.87])
        node_names = ["head", "tail"]
        img_w, img_h = 100, 100

        boxes = renderer._peaks_to_boxes(
            peaks, node_names, img_w, img_h, peak_values=peak_values, is_gt=False
        )

        assert "scores" in boxes[0]
        assert boxes[0]["scores"]["confidence"] == pytest.approx(0.95, rel=0.01)
        assert "(0.95)" in boxes[0]["box_caption"]

    def test_peaks_to_boxes_skips_nan(self):
        """Skips NaN coordinates."""
        renderer = WandBRenderer()
        peaks = np.array([[10.0, 20.0], [np.nan, np.nan], [30.0, 40.0]])
        node_names = ["head", "neck", "tail"]
        img_w, img_h = 100, 100

        boxes = renderer._peaks_to_boxes(peaks, node_names, img_w, img_h, is_gt=True)

        assert len(boxes) == 2  # Skipped NaN
        # Coordinates are in percent
        assert boxes[0]["position"]["middle"] == [0.1, 0.2]  # 10/100, 20/100
        assert boxes[1]["position"]["middle"] == [0.3, 0.4]  # 30/100, 40/100

    def test_peaks_to_boxes_missing_node_names(self):
        """Uses default names when node_names list is shorter."""
        renderer = WandBRenderer()
        peaks = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        node_names = ["head"]  # Only one name for 3 nodes
        img_w, img_h = 100, 100

        boxes = renderer._peaks_to_boxes(peaks, node_names, img_w, img_h, is_gt=True)

        assert boxes[0]["box_caption"] == "GT: head"
        assert boxes[1]["box_caption"] == "GT: node_1"
        assert boxes[2]["box_caption"] == "GT: node_2"

    def test_render_direct_integration(self, sample_data):
        """Integration test for _render_direct."""
        mock_wandb = MagicMock()
        mock_image_class = MagicMock()
        mock_wandb.Image = mock_image_class

        renderer = WandBRenderer(mode="direct")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            # Need to also patch PIL.Image since it's imported inside the method
            with patch("PIL.Image.open") as mock_pil_open:
                mock_pil_image = MagicMock()
                mock_pil_open.return_value = mock_pil_image

                result = renderer._render_direct(sample_data, caption="test caption")

                # Verify wandb.Image was called
                mock_image_class.assert_called_once()
                call_args = mock_image_class.call_args
                assert call_args[1]["caption"] == "test caption"

    def test_render_with_boxes_integration(self, sample_data):
        """Integration test for _render_with_boxes."""
        mock_wandb = MagicMock()
        mock_image_class = MagicMock()
        mock_wandb.Image = mock_image_class

        renderer = WandBRenderer(mode="boxes")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = renderer._render_with_boxes(sample_data, caption="test")

            # Verify wandb.Image was called with boxes
            mock_image_class.assert_called_once()
            call_args = mock_image_class.call_args
            assert "boxes" in call_args[1]
            assert "ground_truth" in call_args[1]["boxes"]
            assert "predictions" in call_args[1]["boxes"]

    def test_render_with_boxes_no_node_names(self):
        """Uses default node names when none provided."""
        data = VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            node_names=[],  # Empty node names
        )

        mock_wandb = MagicMock()
        mock_wandb.Image = MagicMock()

        renderer = WandBRenderer(mode="boxes")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = renderer._render_with_boxes(data)

            call_args = mock_wandb.Image.call_args
            class_labels = call_args[1]["boxes"]["ground_truth"]["class_labels"]
            assert 0 in class_labels
            assert class_labels[0] == "node_0"

    def test_render_with_masks_integration(self, sample_data):
        """Integration test for _render_with_masks."""
        mock_wandb = MagicMock()
        mock_image_class = MagicMock()
        mock_wandb.Image = mock_image_class

        renderer = WandBRenderer(mode="masks", confmap_threshold=0.1)

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = renderer._render_with_masks(sample_data, caption="test")

            # Verify wandb.Image was called with masks
            mock_image_class.assert_called_once()
            call_args = mock_image_class.call_args
            assert "masks" in call_args[1]
            assert "confidence_maps" in call_args[1]["masks"]

    def test_render_with_masks_no_node_names(self):
        """Uses default node names when none provided."""
        data = VisualizationData(
            image=np.random.rand(64, 64, 3).astype(np.float32),
            pred_confmaps=np.random.rand(64, 64, 5).astype(np.float32),
            pred_peaks=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            pred_peak_values=np.random.rand(2, 5).astype(np.float32),
            gt_instances=np.random.rand(2, 5, 2).astype(np.float32) * 64,
            node_names=[],  # Empty node names
        )

        mock_wandb = MagicMock()
        mock_wandb.Image = MagicMock()

        renderer = WandBRenderer(mode="masks")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            result = renderer._render_with_masks(data)

            call_args = mock_wandb.Image.call_args
            class_labels = call_args[1]["masks"]["confidence_maps"]["class_labels"]
            assert 0 in class_labels
            assert class_labels[0] == "background"
            assert class_labels[1] == "node_0"
