"""Fixtures and markers for sleap_nn.export tests."""

import pytest
import torch
import numpy as np
from pathlib import Path


# =============================================================================
# Dependency Detection Helpers
# =============================================================================


def _has_onnx():
    """Check if onnx package is installed."""
    try:
        import onnx

        return True
    except ImportError:
        return False


def _has_onnxruntime():
    """Check if onnxruntime package is installed."""
    try:
        import onnxruntime

        return True
    except ImportError:
        return False


def _has_tensorrt():
    """Check if tensorrt package is installed."""
    try:
        import tensorrt

        return True
    except ImportError:
        return False


# =============================================================================
# Skip Markers
# =============================================================================

requires_onnx = pytest.mark.skipif(not _has_onnx(), reason="onnx not installed")
requires_onnxruntime = pytest.mark.skipif(
    not _has_onnxruntime(), reason="onnxruntime not installed"
)
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
requires_tensorrt = pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_tensorrt()),
    reason="TensorRT or CUDA not available",
)


# =============================================================================
# Synthetic Data Fixtures
# =============================================================================


@pytest.fixture
def synthetic_confmaps():
    """Synthetic confmaps with known peak locations for testing.

    Creates confmaps with a single peak per channel at known locations.
    Peaks are placed at (y=10+c*10, x=20+c*5) for channel c.
    """
    batch, channels, height, width = 2, 5, 64, 64
    confmaps = torch.zeros(batch, channels, height, width)
    # Place peaks at known locations
    for b in range(batch):
        for c in range(channels):
            y, x = 10 + c * 10, 20 + c * 5
            confmaps[b, c, y, x] = 1.0
    return confmaps


@pytest.fixture
def synthetic_confmaps_with_nms():
    """Confmaps with peaks that require NMS suppression.

    Places a primary peak and adjacent secondary peaks to test NMS.
    """
    batch, channels, height, width = 1, 3, 32, 32
    confmaps = torch.zeros(batch, channels, height, width)
    # Primary peaks
    confmaps[0, 0, 16, 16] = 1.0
    confmaps[0, 1, 10, 20] = 1.0
    confmaps[0, 2, 25, 8] = 1.0
    # Adjacent secondary peaks (should be suppressed by NMS)
    confmaps[0, 0, 16, 17] = 0.5
    confmaps[0, 0, 17, 16] = 0.5
    return confmaps


@pytest.fixture
def dummy_uint8_image():
    """Batch of random uint8 images for testing normalization."""
    return torch.randint(0, 256, (2, 1, 64, 64), dtype=torch.uint8)


@pytest.fixture
def dummy_float32_image():
    """Batch of random float32 images (already normalized)."""
    return torch.rand(2, 1, 64, 64, dtype=torch.float32)


# =============================================================================
# Mock Model Fixtures
# =============================================================================


@pytest.fixture
def mock_backbone():
    """Minimal backbone that returns fixed-shape confmaps.

    This is a minimal PyTorch module that mimics a backbone network.
    It returns synthetic confmaps in a dict format matching real models.
    """

    class MockBackbone(torch.nn.Module):
        def __init__(self, n_nodes=5, output_stride=4):
            super().__init__()
            self.n_nodes = n_nodes
            self.output_stride = output_stride
            # Minimal conv to make it a valid model for tracing
            self.conv = torch.nn.Conv2d(1, n_nodes, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            out_h, out_w = h // self.output_stride, w // self.output_stride
            # Return synthetic confmaps in expected format
            return {
                "SingleInstanceConfmapsHead": torch.rand(b, self.n_nodes, out_h, out_w)
            }

    return MockBackbone()


@pytest.fixture
def mock_centroid_backbone():
    """Mock backbone for centroid models."""

    class MockCentroidBackbone(torch.nn.Module):
        def __init__(self, output_stride=4):
            super().__init__()
            self.output_stride = output_stride
            self.conv = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            out_h, out_w = h // self.output_stride, w // self.output_stride
            return {"CentroidConfmapsHead": torch.rand(b, 1, out_h, out_w)}

    return MockCentroidBackbone()


@pytest.fixture
def mock_bottomup_backbone():
    """Mock backbone for bottom-up models with confmaps and PAFs."""

    class MockBottomUpBackbone(torch.nn.Module):
        def __init__(self, n_nodes=5, n_edges=4, output_stride=4, pafs_output_stride=8):
            super().__init__()
            self.n_nodes = n_nodes
            self.n_edges = n_edges
            self.output_stride = output_stride
            self.pafs_output_stride = pafs_output_stride
            self.conv = torch.nn.Conv2d(1, n_nodes, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            cms_h, cms_w = h // self.output_stride, w // self.output_stride
            pafs_h, pafs_w = h // self.pafs_output_stride, w // self.pafs_output_stride
            return {
                "MultiInstanceConfmapsHead": torch.rand(b, self.n_nodes, cms_h, cms_w),
                "PartAffinityFieldsHead": torch.rand(
                    b, self.n_edges * 2, pafs_h, pafs_w
                ),
            }

    return MockBottomUpBackbone()


@pytest.fixture
def mock_multiclass_bottomup_backbone():
    """Mock backbone for multiclass bottom-up models."""

    class MockMultiClassBottomUpBackbone(torch.nn.Module):
        def __init__(
            self, n_nodes=5, n_classes=2, output_stride=4, class_maps_stride=8
        ):
            super().__init__()
            self.n_nodes = n_nodes
            self.n_classes = n_classes
            self.output_stride = output_stride
            self.class_maps_stride = class_maps_stride
            self.conv = torch.nn.Conv2d(1, n_nodes, 1)

        def forward(self, x):
            b, c, h, w = x.shape
            cms_h, cms_w = h // self.output_stride, w // self.output_stride
            cls_h, cls_w = h // self.class_maps_stride, w // self.class_maps_stride
            return {
                "MultiInstanceConfmapsHead": torch.rand(b, self.n_nodes, cms_h, cms_w),
                "ClassMapsHead": torch.rand(b, self.n_classes, cls_h, cls_w),
            }

    return MockMultiClassBottomUpBackbone()


# =============================================================================
# Wrapper Fixtures
# =============================================================================


@pytest.fixture
def mock_single_instance_wrapper(mock_backbone):
    """SingleInstanceONNXWrapper with mock backbone."""
    from sleap_nn.export.wrappers import SingleInstanceONNXWrapper

    return SingleInstanceONNXWrapper(mock_backbone, output_stride=4)


@pytest.fixture
def mock_centroid_wrapper(mock_centroid_backbone):
    """CentroidONNXWrapper with mock backbone."""
    from sleap_nn.export.wrappers import CentroidONNXWrapper

    return CentroidONNXWrapper(
        mock_centroid_backbone, output_stride=4, max_instances=10
    )


# =============================================================================
# Export Path Fixtures
# =============================================================================


@pytest.fixture
def temp_export_dir(tmp_path):
    """Temporary directory for exports."""
    export_dir = tmp_path / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


# =============================================================================
# ONNX Export Fixtures (requires onnx)
# =============================================================================


@pytest.fixture
def exported_onnx_model(mock_single_instance_wrapper, tmp_path):
    """Export mock wrapper to ONNX and return path.

    Skips if onnx is not installed.
    """
    if not _has_onnx():
        pytest.skip("onnx not installed")

    from sleap_nn.export.exporters import export_to_onnx

    onnx_path = tmp_path / "model.onnx"
    export_to_onnx(
        mock_single_instance_wrapper,
        onnx_path,
        input_shape=(1, 1, 64, 64),
        verify=False,  # Don't verify since mock model may have issues
    )
    return onnx_path


# =============================================================================
# Video Fixtures
# =============================================================================


@pytest.fixture
def tiny_video(tmp_path):
    """Create a tiny synthetic video for testing.

    Creates a 10-frame video of 64x64 random noise.
    Uses imageio if available, otherwise creates a dummy file.
    """
    video_path = tmp_path / "test_video.mp4"

    try:
        import imageio

        writer = imageio.get_writer(str(video_path), fps=10)
        for _ in range(10):
            frame = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            writer.append_data(frame)
        writer.close()
    except ImportError:
        # If imageio not available, create a dummy file
        # Tests using this fixture should handle this case
        video_path.touch()

    return video_path


# =============================================================================
# Config Fixtures
# =============================================================================


@pytest.fixture
def sample_training_config():
    """Sample OmegaConf training config for testing utility functions."""
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {"input_scaling": 1.0, "is_rgb": False},
                "skeletons": {
                    "fly": {
                        "nodes": ["head", "thorax", "abdomen", "wingL", "wingR"],
                        "edges": [
                            {"source": "head", "destination": "thorax"},
                            {"source": "thorax", "destination": "abdomen"},
                            {"source": "thorax", "destination": "wingL"},
                            {"source": "thorax", "destination": "wingR"},
                        ],
                    }
                },
            },
            "model_config": {
                "head_configs": {
                    "single_instance": {
                        "confmaps": {
                            "SingleInstanceConfmapsHead": {
                                "output_stride": 4,
                                "sigma": 2.0,
                            }
                        }
                    }
                }
            },
        }
    )
    return config


@pytest.fixture
def deterministic_single_instance_wrapper():
    """SingleInstanceONNXWrapper with deterministic backbone for accuracy tests.

    Uses a backbone that produces consistent output for the same input,
    enabling meaningful comparison between PyTorch and ONNX inference.
    """
    from sleap_nn.export.wrappers import SingleInstanceONNXWrapper

    class DeterministicBackbone(torch.nn.Module):
        """Backbone that produces deterministic confmaps based on input."""

        def __init__(self, n_nodes=5, output_stride=4):
            super().__init__()
            self.n_nodes = n_nodes
            self.output_stride = output_stride
            # Use a deterministic conv layer
            self.conv = torch.nn.Conv2d(1, n_nodes, kernel_size=1, bias=False)
            # Initialize with fixed weights
            torch.nn.init.constant_(self.conv.weight, 0.01)

        def forward(self, x):
            # Apply actual convolution to produce deterministic output
            x_float = x.float() / 255.0 if x.dtype == torch.uint8 else x
            # Downsample to match output_stride
            x_down = torch.nn.functional.avg_pool2d(
                x_float, kernel_size=self.output_stride
            )
            # Apply conv to get confmaps
            confmaps = self.conv(x_down)
            # Apply sigmoid to get [0, 1] range
            confmaps = torch.sigmoid(confmaps)
            return {"SingleInstanceConfmapsHead": confmaps}

    backbone = DeterministicBackbone()
    return SingleInstanceONNXWrapper(backbone, output_stride=4)


@pytest.fixture
def multiclass_training_config():
    """Sample training config for multiclass models."""
    from omegaconf import OmegaConf

    config = OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {"input_scaling": 1.0, "is_rgb": False},
                "skeletons": {
                    "fly": {
                        "nodes": ["head", "thorax", "abdomen"],
                        "edges": [
                            {"source": "head", "destination": "thorax"},
                            {"source": "thorax", "destination": "abdomen"},
                        ],
                    }
                },
                "class_names": ["female", "male"],
            },
            "model_config": {
                "head_configs": {
                    "multi_class_bottomup": {
                        "confmaps": {"MultiInstanceConfmapsHead": {"output_stride": 4}},
                        "class_maps": {"ClassMapsHead": {"output_stride": 8}},
                    }
                }
            },
        }
    )
    return config
