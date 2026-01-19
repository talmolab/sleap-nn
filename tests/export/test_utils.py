"""Tests for sleap_nn.export.utils module."""

import pytest
import torch

from sleap_nn.export.utils import build_bottomup_candidate_template


class TestBuildBottomupCandidateTemplate:
    """Tests for build_bottomup_candidate_template function."""

    def test_output_shapes(self):
        """Test that output tensors have correct shapes."""
        n_nodes = 5
        k = 3
        edge_inds = [(0, 1), (1, 2), (2, 3)]
        n_edges = len(edge_inds)

        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # peak_channel_inds: (n_nodes * k,)
        assert peak_ch.shape == (n_nodes * k,)

        # edge_inds: (n_edges * k * k,)
        assert edge_idx.shape == (n_edges * k * k,)

        # edge_peak_inds: (n_edges * k * k, 2)
        assert edge_peaks.shape == (n_edges * k * k, 2)

    def test_peak_channel_indices(self):
        """Test that peak_channel_inds maps flat peak index to correct node."""
        n_nodes = 4
        k = 2

        peak_ch, _, _ = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=[(0, 1)]
        )

        # peak_ch should be [0, 0, 1, 1, 2, 2, 3, 3]
        expected = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int32)
        assert torch.equal(peak_ch, expected)

    def test_edge_indices_ordering(self):
        """Test that edge_inds assigns correct edge index for each candidate."""
        n_nodes = 3
        k = 2
        edge_inds = [(0, 1), (1, 2)]

        _, edge_idx, _ = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # For k=2, each edge has k*k=4 candidates
        # Expected: [0,0,0,0, 1,1,1,1]
        expected = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
        assert torch.equal(edge_idx, expected)

    def test_edge_peak_indices_row_major_ordering(self):
        """Test that edge_peak_inds follows row-major (i*k + j) pattern."""
        n_nodes = 3
        k = 2
        edge_inds = [(0, 1)]  # src_node=0, dst_node=1

        _, _, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # For edge (0, 1) with k=2:
        # Position 0: i=0, j=0 -> src=0*2+0=0, dst=1*2+0=2
        # Position 1: i=0, j=1 -> src=0*2+0=0, dst=1*2+1=3
        # Position 2: i=1, j=0 -> src=0*2+1=1, dst=1*2+0=2
        # Position 3: i=1, j=1 -> src=0*2+1=1, dst=1*2+1=3
        expected = torch.tensor([[0, 2], [0, 3], [1, 2], [1, 3]], dtype=torch.int32)
        assert torch.equal(edge_peaks, expected)

    def test_multiple_edges(self):
        """Test correct indexing for multiple edges."""
        n_nodes = 4
        k = 2
        edge_inds = [(0, 1), (2, 3)]

        _, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # Edge 0: (0, 1) -> src_base=0, dst_base=2
        # Edge 1: (2, 3) -> src_base=4, dst_base=6
        expected_peaks = torch.tensor(
            [
                # Edge 0: src from [0,1], dst from [2,3]
                [0, 2],
                [0, 3],
                [1, 2],
                [1, 3],
                # Edge 1: src from [4,5], dst from [6,7]
                [4, 6],
                [4, 7],
                [5, 6],
                [5, 7],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(edge_peaks, expected_peaks)

    def test_realistic_skeleton(self):
        """Test with realistic skeleton dimensions (15 nodes, 14 edges, k=20)."""
        n_nodes = 15
        k = 20
        # Chain skeleton: node 0 -> 1 -> 2 -> ... -> 14
        edge_inds = [(i, i + 1) for i in range(14)]
        n_edges = len(edge_inds)

        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # Verify shapes
        assert peak_ch.shape == (n_nodes * k,)  # 15 * 20 = 300
        assert edge_idx.shape == (n_edges * k * k,)  # 14 * 400 = 5600
        assert edge_peaks.shape == (n_edges * k * k, 2)

        # Spot check: first candidate for edge (5, 6)
        # Edge index 5, position 0
        idx = 5 * k * k
        assert edge_idx[idx] == 5
        assert edge_peaks[idx, 0] == 5 * k  # src_base = 5 * 20 = 100
        assert edge_peaks[idx, 1] == 6 * k  # dst_base = 6 * 20 = 120

    def test_empty_edges(self):
        """Test behavior with no edges."""
        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=5, max_peaks_per_node=3, edge_inds=[]
        )

        assert peak_ch.shape == (15,)  # n_nodes * k
        assert edge_idx.shape == (0,)
        assert edge_peaks.shape == (0, 2)

    def test_single_edge(self):
        """Test with single edge."""
        n_nodes = 10
        k = 5
        edge_inds = [(3, 7)]  # Non-adjacent nodes

        _, _, edge_peaks = build_bottomup_candidate_template(
            n_nodes=n_nodes, max_peaks_per_node=k, edge_inds=edge_inds
        )

        # Verify first and last candidate
        # First: i=0, j=0 -> src=3*5+0=15, dst=7*5+0=35
        assert edge_peaks[0, 0] == 15
        assert edge_peaks[0, 1] == 35

        # Last: i=4, j=4 -> src=3*5+4=19, dst=7*5+4=39
        assert edge_peaks[-1, 0] == 19
        assert edge_peaks[-1, 1] == 39

    def test_dtypes(self):
        """Test that output tensors have correct dtypes."""
        peak_ch, edge_idx, edge_peaks = build_bottomup_candidate_template(
            n_nodes=3, max_peaks_per_node=2, edge_inds=[(0, 1)]
        )

        assert peak_ch.dtype == torch.int32
        assert edge_idx.dtype == torch.int32
        assert edge_peaks.dtype == torch.int32


class TestLoadTrainingConfig:
    """Tests for load_training_config function."""

    def test_load_training_config(self, minimal_instance_single_instance_ckpt):
        """Test loading training config from a checkpoint directory."""
        from sleap_nn.export.utils import load_training_config

        cfg = load_training_config(minimal_instance_single_instance_ckpt)

        # Should return an OmegaConf DictConfig
        from omegaconf import DictConfig

        assert isinstance(cfg, DictConfig)

        # Should have expected top-level keys
        assert hasattr(cfg, "data_config") or "data_config" in cfg
        assert hasattr(cfg, "model_config") or "model_config" in cfg

    def test_load_training_config_missing(self, tmp_path):
        """Test that missing config raises FileNotFoundError."""
        from sleap_nn.export.utils import load_training_config

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No training_config"):
            load_training_config(empty_dir)


class TestResolveModelType:
    """Tests for resolve_model_type function."""

    def test_resolve_model_type_single_instance(
        self, minimal_instance_single_instance_ckpt
    ):
        """Test detecting single_instance model type."""
        from sleap_nn.export.utils import load_training_config, resolve_model_type

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        model_type = resolve_model_type(cfg)
        assert model_type == "single_instance"

    def test_resolve_model_type_centroid(self, minimal_instance_centroid_ckpt):
        """Test detecting centroid model type."""
        from sleap_nn.export.utils import load_training_config, resolve_model_type

        cfg = load_training_config(minimal_instance_centroid_ckpt)
        model_type = resolve_model_type(cfg)
        assert model_type == "centroid"

    def test_resolve_model_type_centered_instance(
        self, minimal_instance_centered_instance_ckpt
    ):
        """Test detecting centered_instance model type."""
        from sleap_nn.export.utils import load_training_config, resolve_model_type

        cfg = load_training_config(minimal_instance_centered_instance_ckpt)
        model_type = resolve_model_type(cfg)
        assert model_type == "centered_instance"

    def test_resolve_model_type_bottomup(self, minimal_instance_bottomup_ckpt):
        """Test detecting bottomup model type."""
        from sleap_nn.export.utils import load_training_config, resolve_model_type

        cfg = load_training_config(minimal_instance_bottomup_ckpt)
        model_type = resolve_model_type(cfg)
        assert model_type == "bottomup"

    def test_resolve_model_type_multiclass_bottomup(
        self, minimal_instance_multi_class_bottomup_ckpt
    ):
        """Test detecting multi_class_bottomup model type."""
        from sleap_nn.export.utils import load_training_config, resolve_model_type

        cfg = load_training_config(minimal_instance_multi_class_bottomup_ckpt)
        model_type = resolve_model_type(cfg)
        assert model_type == "multi_class_bottomup"

    def test_resolve_model_type_multiclass_topdown(
        self, minimal_instance_multi_class_topdown_ckpt
    ):
        """Test detecting multi_class_topdown model type."""
        from sleap_nn.export.utils import load_training_config, resolve_model_type

        cfg = load_training_config(minimal_instance_multi_class_topdown_ckpt)
        model_type = resolve_model_type(cfg)
        assert model_type == "multi_class_topdown"


class TestResolveInputChannels:
    """Tests for resolve_input_channels function."""

    def test_resolve_input_channels(self, minimal_instance_single_instance_ckpt):
        """Test resolving channels from backbone config."""
        from sleap_nn.export.utils import load_training_config, resolve_input_channels

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        channels = resolve_input_channels(cfg)
        # Should be a valid channel count (1 for grayscale or 3 for RGB)
        assert channels in (1, 3)


class TestResolveOutputStride:
    """Tests for resolve_output_stride function."""

    def test_resolve_output_stride(self, minimal_instance_single_instance_ckpt):
        """Test extracting output_stride from confmaps head."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_output_stride,
            resolve_model_type,
        )

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        model_type = resolve_model_type(cfg)
        stride = resolve_output_stride(cfg, model_type)
        # Should be a positive integer
        assert isinstance(stride, int)
        assert stride >= 1


class TestResolveNodeNames:
    """Tests for resolve_node_names function."""

    def test_resolve_node_names(self, minimal_instance_single_instance_ckpt):
        """Test extracting node names from skeleton config."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_node_names,
            resolve_model_type,
        )

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        model_type = resolve_model_type(cfg)
        node_names = resolve_node_names(cfg, model_type)

        # Should return a list of strings
        assert isinstance(node_names, list)
        assert len(node_names) > 0
        assert all(isinstance(name, str) for name in node_names)


class TestResolveEdgeInds:
    """Tests for resolve_edge_inds function."""

    def test_resolve_edge_inds(self, minimal_instance_bottomup_ckpt):
        """Test extracting edge indices from skeleton config."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_edge_inds,
            resolve_node_names,
            resolve_model_type,
        )

        cfg = load_training_config(minimal_instance_bottomup_ckpt)
        model_type = resolve_model_type(cfg)
        node_names = resolve_node_names(cfg, model_type)
        edge_inds = resolve_edge_inds(cfg, node_names)

        # Should return a list of tuples
        assert isinstance(edge_inds, list)
        if edge_inds:
            assert all(isinstance(edge, tuple) for edge in edge_inds)
            assert all(len(edge) == 2 for edge in edge_inds)
            # Indices should be integers
            assert all(isinstance(idx, int) for edge in edge_inds for idx in edge)


class TestResolveCropSize:
    """Tests for resolve_crop_size function."""

    def test_resolve_crop_size(self, minimal_instance_centered_instance_ckpt):
        """Test extracting crop_size for centered_instance models."""
        from sleap_nn.export.utils import load_training_config, resolve_crop_size

        cfg = load_training_config(minimal_instance_centered_instance_ckpt)
        crop_size = resolve_crop_size(cfg)

        # Should return a tuple of (height, width) or None
        if crop_size is not None:
            assert isinstance(crop_size, tuple)
            assert len(crop_size) == 2
            assert all(isinstance(dim, int) for dim in crop_size)
            assert all(dim > 0 for dim in crop_size)


class TestResolveClassInfo:
    """Tests for resolve_n_classes and resolve_class_names functions."""

    def test_resolve_n_classes(self, minimal_instance_multi_class_bottomup_ckpt):
        """Test extracting class count for multiclass models."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_n_classes,
            resolve_model_type,
        )

        cfg = load_training_config(minimal_instance_multi_class_bottomup_ckpt)
        model_type = resolve_model_type(cfg)
        n_classes = resolve_n_classes(cfg, model_type)

        # Should be a non-negative integer
        assert isinstance(n_classes, int)
        assert n_classes >= 0

    def test_resolve_class_names(self, minimal_instance_multi_class_bottomup_ckpt):
        """Test extracting class names for multiclass models."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_class_names,
            resolve_model_type,
        )

        cfg = load_training_config(minimal_instance_multi_class_bottomup_ckpt)
        model_type = resolve_model_type(cfg)
        class_names = resolve_class_names(cfg, model_type)

        # Should be a list (possibly empty)
        assert isinstance(class_names, list)
        if class_names:
            assert all(isinstance(name, str) for name in class_names)


class TestResolveInputShape:
    """Tests for resolve_input_shape function."""

    def test_resolve_input_shape(self, minimal_instance_single_instance_ckpt):
        """Test computing (B, C, H, W) input shape tuple."""
        from sleap_nn.export.utils import load_training_config, resolve_input_shape

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        shape = resolve_input_shape(cfg, input_height=256, input_width=256)

        # Should return (batch, channels, height, width)
        assert isinstance(shape, tuple)
        assert len(shape) == 4
        assert shape[0] == 1  # batch size
        assert shape[1] >= 1  # channels
        assert shape[2] == 256  # height
        assert shape[3] == 256  # width

    def test_resolve_input_shape_defaults(self, minimal_instance_single_instance_ckpt):
        """Test that resolve_input_shape uses defaults when height/width not specified."""
        from sleap_nn.export.utils import load_training_config, resolve_input_shape

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        shape = resolve_input_shape(cfg)

        # Should still return a valid shape tuple
        assert isinstance(shape, tuple)
        assert len(shape) == 4
        assert all(isinstance(dim, int) for dim in shape)
        assert all(dim > 0 for dim in shape)


class TestNormalizeEdges:
    """Tests for _normalize_edges helper function."""

    def test_normalize_edges_with_indices(self):
        """Test normalizing edges that are already integer indices."""
        from sleap_nn.export.utils import _normalize_edges

        edges = [(0, 1), (1, 2), (2, 3)]
        node_names = ["n0", "n1", "n2", "n3"]
        result = _normalize_edges(edges, node_names)

        assert result == [(0, 1), (1, 2), (2, 3)]

    def test_normalize_edges_with_names(self):
        """Test normalizing edges that are node names."""
        from sleap_nn.export.utils import _normalize_edges

        edges = [("head", "thorax"), ("thorax", "abdomen")]
        node_names = ["head", "thorax", "abdomen"]
        result = _normalize_edges(edges, node_names)

        assert result == [(0, 1), (1, 2)]

    def test_normalize_edges_empty(self):
        """Test normalizing empty edge list."""
        from sleap_nn.export.utils import _normalize_edges

        result = _normalize_edges([], ["n0", "n1"])
        assert result == []


class TestResolveInputScale:
    """Tests for resolve_input_scale function."""

    def test_resolve_input_scale_single_value(
        self, minimal_instance_single_instance_ckpt
    ):
        """Test resolving scale when it's a single value."""
        from sleap_nn.export.utils import load_training_config, resolve_input_scale

        cfg = load_training_config(minimal_instance_single_instance_ckpt)
        scale = resolve_input_scale(cfg)

        # Should return a float
        assert isinstance(scale, float)
        assert scale > 0

    def test_resolve_input_scale_tuple(self):
        """Test resolving scale when it's a tuple (native Python)."""
        from omegaconf import DictConfig
        from sleap_nn.export.utils import resolve_input_scale

        # Create a DictConfig with a native Python tuple
        cfg = DictConfig({"data_config": {"preprocessing": {"scale": (0.5, 0.5)}}})
        scale = resolve_input_scale(cfg)
        assert scale == 0.5

    def test_resolve_input_scale_scalar(self):
        """Test resolving scale when it's a scalar value."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_input_scale

        cfg = OmegaConf.create({"data_config": {"preprocessing": {"scale": 0.75}}})
        scale = resolve_input_scale(cfg)
        assert scale == 0.75


class TestResolvePafsOutputStride:
    """Tests for resolve_pafs_output_stride function."""

    def test_resolve_pafs_output_stride(self, minimal_instance_bottomup_ckpt):
        """Test extracting PAFs output stride."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_pafs_output_stride,
        )

        cfg = load_training_config(minimal_instance_bottomup_ckpt)
        stride = resolve_pafs_output_stride(cfg)

        assert isinstance(stride, int)
        assert stride >= 1

    def test_resolve_pafs_output_stride_no_pafs(self):
        """Test default when no PAFs config exists."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_pafs_output_stride

        cfg = OmegaConf.create({"model_config": {"head_configs": {}}})
        stride = resolve_pafs_output_stride(cfg)
        assert stride == 1


class TestResolveClassMapsOutputStride:
    """Tests for resolve_class_maps_output_stride function."""

    def test_resolve_class_maps_output_stride(
        self, minimal_instance_multi_class_bottomup_ckpt
    ):
        """Test extracting class maps output stride."""
        from sleap_nn.export.utils import (
            load_training_config,
            resolve_class_maps_output_stride,
        )

        cfg = load_training_config(minimal_instance_multi_class_bottomup_ckpt)
        stride = resolve_class_maps_output_stride(cfg)

        assert isinstance(stride, int)
        assert stride >= 1

    def test_resolve_class_maps_output_stride_no_config(self):
        """Test default when no class maps config exists."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_class_maps_output_stride

        cfg = OmegaConf.create({"model_config": {"head_configs": {}}})
        stride = resolve_class_maps_output_stride(cfg)
        assert stride == 8


class TestSkeletonParsing:
    """Tests for skeleton parsing helper functions."""

    def test_node_names_from_dict_skeleton(self):
        """Test extracting node names from dict-based skeleton."""
        from sleap_nn.export.utils import _node_names_from_skeletons

        skeletons = [{"nodes": ["head", "thorax", "abdomen"]}]
        names = _node_names_from_skeletons(skeletons)
        assert names == ["head", "thorax", "abdomen"]

    def test_node_names_from_dict_skeleton_with_dicts(self):
        """Test extracting node names from skeleton with node dicts."""
        from sleap_nn.export.utils import _node_names_from_skeletons

        skeletons = [{"nodes": [{"name": "head"}, {"name": "thorax"}]}]
        names = _node_names_from_skeletons(skeletons)
        assert names == ["head", "thorax"]

    def test_node_names_from_dict_with_node_names_key(self):
        """Test extracting node names from node_names key."""
        from sleap_nn.export.utils import _node_names_from_skeletons

        skeletons = [{"node_names": ["head", "thorax"]}]
        names = _node_names_from_skeletons(skeletons)
        assert names == ["head", "thorax"]

    def test_node_names_from_empty_skeleton(self):
        """Test empty skeleton returns empty list."""
        from sleap_nn.export.utils import _node_names_from_skeletons

        assert _node_names_from_skeletons([]) == []
        assert _node_names_from_skeletons(None) == []

    def test_edge_inds_from_dict_skeleton(self):
        """Test extracting edge indices from dict-based skeleton."""
        from sleap_nn.export.utils import _edge_inds_from_skeletons

        skeletons = [{"edges": [(0, 1), (1, 2)]}]
        edges = _edge_inds_from_skeletons(skeletons)
        assert edges == [(0, 1), (1, 2)]

    def test_edge_inds_from_dict_with_edge_inds_key(self):
        """Test extracting edge indices from edge_inds key."""
        from sleap_nn.export.utils import _edge_inds_from_skeletons

        skeletons = [{"edge_inds": [(0, 1)]}]
        edges = _edge_inds_from_skeletons(skeletons)
        assert edges == [(0, 1)]

    def test_edge_inds_from_empty_skeleton(self):
        """Test empty skeleton returns empty list."""
        from sleap_nn.export.utils import _edge_inds_from_skeletons

        assert _edge_inds_from_skeletons([]) == []
        assert _edge_inds_from_skeletons(None) == []


class TestNormalizeEdgesNoNodeNames:
    """Additional tests for _normalize_edges."""

    def test_normalize_edges_no_node_names(self):
        """Test normalizing edges when no node names provided."""
        from sleap_nn.export.utils import _normalize_edges

        edges = [(0, 1), (1, 2)]
        result = _normalize_edges(edges, [])
        assert result == [(0, 1), (1, 2)]


class TestResolveOutputStrideEdgeCases:
    """Edge case tests for resolve_output_stride."""

    def test_resolve_output_stride_none_head_cfg(self):
        """Test that None head_cfg returns 1."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_output_stride

        cfg = OmegaConf.create(
            {"model_config": {"head_configs": {"single_instance": None}}}
        )
        stride = resolve_output_stride(cfg, "single_instance")
        assert stride == 1

    def test_resolve_output_stride_pafs_fallback(self):
        """Test fallback to PAFs output stride when no confmaps."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_output_stride

        cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "bottomup": {"confmaps": None, "pafs": {"output_stride": 8}}
                    }
                }
            }
        )
        stride = resolve_output_stride(cfg, "bottomup")
        assert stride == 8

    def test_resolve_output_stride_no_confmaps_no_pafs(self):
        """Test default when neither confmaps nor pafs."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_output_stride

        cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {"single_instance": {"other_key": "value"}}
                }
            }
        )
        stride = resolve_output_stride(cfg, "single_instance")
        assert stride == 1


class TestResolveClassNamesEdgeCases:
    """Edge case tests for resolve_class_names."""

    def test_resolve_class_names_none_head_cfg(self):
        """Test that None head_cfg returns empty list."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_class_names

        cfg = OmegaConf.create({"model_config": {"head_configs": {}}})
        names = resolve_class_names(cfg, "nonexistent")
        assert names == []

    def test_resolve_class_names_from_class_vectors(self):
        """Test extracting class names from class_vectors."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_class_names

        cfg = OmegaConf.create(
            {
                "model_config": {
                    "head_configs": {
                        "multi_class_topdown": {
                            "class_vectors": {"classes": ["female", "male"]}
                        }
                    }
                }
            }
        )
        names = resolve_class_names(cfg, "multi_class_topdown")
        assert names == ["female", "male"]

    def test_resolve_class_names_no_classes(self):
        """Test when head config has no class info."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_class_names

        cfg = OmegaConf.create(
            {"model_config": {"head_configs": {"single_instance": {"confmaps": {}}}}}
        )
        names = resolve_class_names(cfg, "single_instance")
        assert names == []


class TestResolveCropSizeEdgeCases:
    """Edge case tests for resolve_crop_size."""

    def test_resolve_crop_size_single_value(self):
        """Test crop_size when it's a single integer."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_crop_size

        cfg = OmegaConf.create({"data_config": {"preprocessing": {"crop_size": 128}}})
        result = resolve_crop_size(cfg)
        assert result == (128, 128)

    def test_resolve_crop_size_single_element_list(self):
        """Test crop_size when it's a single-element list."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_crop_size

        cfg = OmegaConf.create({"data_config": {"preprocessing": {"crop_size": [192]}}})
        result = resolve_crop_size(cfg)
        assert result == (192, 192)

    def test_resolve_crop_size_none(self):
        """Test crop_size when it's None."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_crop_size

        cfg = OmegaConf.create({"data_config": {"preprocessing": {"crop_size": None}}})
        result = resolve_crop_size(cfg)
        assert result is None


class TestResolveNodeNamesEdgeCases:
    """Edge case tests for resolve_node_names."""

    def test_resolve_node_names_from_part_names(self):
        """Test extracting node names from confmaps.part_names."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_node_names

        cfg = OmegaConf.create(
            {
                "data_config": {"skeletons": []},
                "model_config": {
                    "head_configs": {
                        "single_instance": {
                            "confmaps": {"part_names": ["head", "thorax", "abdomen"]}
                        }
                    }
                },
            }
        )
        names = resolve_node_names(cfg, "single_instance")
        assert names == ["head", "thorax", "abdomen"]

    def test_resolve_node_names_centroid_anchor(self):
        """Test extracting anchor part name for centroid models."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_node_names

        cfg = OmegaConf.create(
            {
                "data_config": {"skeletons": []},
                "model_config": {
                    "head_configs": {
                        "centroid": {"confmaps": {"anchor_part": "thorax"}}
                    }
                },
            }
        )
        names = resolve_node_names(cfg, "centroid")
        assert names == ["thorax"]

    def test_resolve_node_names_centroid_default(self):
        """Test default 'centroid' name when no anchor_part."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_node_names

        cfg = OmegaConf.create(
            {
                "data_config": {"skeletons": []},
                "model_config": {
                    "head_configs": {"centroid": {"confmaps": {"output_stride": 4}}}
                },
            }
        )
        names = resolve_node_names(cfg, "centroid")
        assert names == ["centroid"]

    def test_resolve_node_names_none_head(self):
        """Test empty list when head config is None."""
        from omegaconf import OmegaConf
        from sleap_nn.export.utils import resolve_node_names

        cfg = OmegaConf.create(
            {"data_config": {"skeletons": []}, "model_config": {"head_configs": {}}}
        )
        names = resolve_node_names(cfg, "nonexistent")
        assert names == []
