"""Tests for the config generator module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import sleap_io as sio
from omegaconf import DictConfig

from sleap_nn.config_generator import (
    ConfigGenerator,
    ConfigRecommendation,
    DatasetStats,
    MemoryEstimate,
    PipelineRecommendation,
    ViewType,
    analyze_slp,
    estimate_memory,
    generate_config,
    recommend_config,
    recommend_pipeline,
)


class TestDatasetStats:
    """Tests for DatasetStats dataclass."""

    def test_properties(self):
        """Test computed properties."""
        stats = DatasetStats(
            slp_path="/test/path.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=50.0,
            avg_bbox_size=40.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=100 * 480 * 640,
        )

        assert stats.frame_area == 480 * 640
        assert stats.is_single_instance is True
        assert stats.is_multi_instance is False
        assert stats.has_identity is False
        assert stats.is_grayscale is True
        assert stats.is_rgb is False
        assert stats.max_dimension == 640

    def test_str_representation(self):
        """Test string representation."""
        stats = DatasetStats(
            slp_path="/test/path.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=50.0,
            avg_bbox_size=40.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=100 * 480 * 640,
        )

        str_repr = str(stats)
        assert "path.slp" in str_repr
        assert "100" in str_repr
        assert "640x480" in str_repr


class TestAnalyzeSlp:
    """Tests for analyze_slp function."""

    def test_analyze_slp(self, minimal_instance):
        """Test basic SLP analysis."""
        stats = analyze_slp(str(minimal_instance))

        assert isinstance(stats, DatasetStats)
        assert stats.num_labeled_frames > 0
        assert stats.num_videos >= 1
        assert stats.num_nodes > 0


class TestRecommendPipeline:
    """Tests for pipeline recommendation."""

    def test_single_instance_recommendation(self):
        """Test recommendation for single instance data."""
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=100.0,
            avg_bbox_size=80.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        rec = recommend_pipeline(stats)

        assert isinstance(rec, PipelineRecommendation)
        assert rec.recommended == "single_instance"
        assert "one animal" in rec.reason.lower()

    def test_topdown_recommendation_small_animals(self):
        """Test recommendation for multiple small animals."""
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=1080,
            max_width=1920,
            num_channels=1,
            max_instances_per_frame=5,
            avg_instances_per_frame=3.0,
            max_bbox_size=50.0,
            avg_bbox_size=40.0,  # Small relative to frame
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        rec = recommend_pipeline(stats)

        assert rec.recommended == "centroid"
        assert rec.requires_second_model is True
        assert rec.second_model_type == "centered_instance"

    def test_bottomup_recommendation_large_animals(self):
        """Test recommendation for multiple large animals."""
        # avg_bbox_size^2 / (max_height * max_width) > 0.15 for large animals
        # Need avg_bbox_size > sqrt(0.15 * 480 * 640) = sqrt(46080) ~= 215
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=3,
            avg_instances_per_frame=2.0,
            max_bbox_size=300.0,
            avg_bbox_size=250.0,  # Large relative to frame (250^2 / 307200 = 0.20)
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        rec = recommend_pipeline(stats)

        assert rec.recommended == "bottomup"

    def test_no_edges_fallback(self):
        """Test recommendation fallback when no edges available."""
        # avg_bbox_size^2 / frame_area >= 0.15 to trigger bottomup path
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=3,
            avg_instances_per_frame=2.0,
            max_bbox_size=300.0,
            avg_bbox_size=250.0,  # Large relative to frame
            num_nodes=5,
            num_edges=0,  # No edges!
            node_names=["a", "b", "c", "d", "e"],
            edges=[],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        rec = recommend_pipeline(stats)

        # Should fall back to centroid (top-down) since bottom-up needs edges
        assert rec.recommended == "centroid"
        assert any("edge" in w.lower() for w in rec.warnings)


class TestRecommendConfig:
    """Tests for full config recommendation."""

    def test_recommend_config(self):
        """Test full config recommendation."""
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=100.0,
            avg_bbox_size=80.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        rec = recommend_config(stats, ViewType.TOP)

        assert isinstance(rec, ConfigRecommendation)
        assert rec.pipeline.recommended == "single_instance"
        assert rec.backbone in ["unet_medium_rf", "unet_large_rf"]
        assert rec.sigma > 0
        assert rec.batch_size > 0
        assert rec.rotation_range == (-180.0, 180.0)  # Top view

    def test_view_type_affects_rotation(self):
        """Test that view type affects rotation recommendation."""
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=100.0,
            avg_bbox_size=80.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        rec_top = recommend_config(stats, ViewType.TOP)
        rec_side = recommend_config(stats, ViewType.SIDE)

        assert rec_top.rotation_range == (-180.0, 180.0)
        assert rec_side.rotation_range == (-15.0, 15.0)


class TestMemoryEstimate:
    """Tests for memory estimation."""

    def test_estimate_memory(self):
        """Test memory estimation."""
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=100.0,
            avg_bbox_size=80.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        mem = estimate_memory(stats, batch_size=4)

        assert isinstance(mem, MemoryEstimate)
        assert mem.total_gpu_mb > 0
        assert mem.cache_memory_mb > 0
        assert mem.gpu_status in ["green", "yellow", "red"]

    def test_larger_batch_increases_memory(self):
        """Test that larger batch size increases memory estimate."""
        stats = DatasetStats(
            slp_path="/test.slp",
            num_labeled_frames=100,
            num_videos=1,
            max_height=480,
            max_width=640,
            num_channels=1,
            max_instances_per_frame=1,
            avg_instances_per_frame=1.0,
            max_bbox_size=100.0,
            avg_bbox_size=80.0,
            num_nodes=5,
            num_edges=4,
            node_names=["a", "b", "c", "d", "e"],
            edges=[("a", "b")],
            has_tracks=False,
            num_tracks=0,
            estimated_total_bytes=1000000,
        )

        mem_small = estimate_memory(stats, batch_size=2)
        mem_large = estimate_memory(stats, batch_size=8)

        assert mem_large.total_gpu_mb > mem_small.total_gpu_mb


class TestConfigGenerator:
    """Tests for ConfigGenerator class."""

    def test_from_slp(self, minimal_instance):
        """Test creating generator from SLP file."""
        gen = ConfigGenerator.from_slp(str(minimal_instance))

        assert gen.slp_path.exists()
        assert gen.stats is not None

    def test_auto_configuration(self, minimal_instance):
        """Test auto configuration."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto()

        assert gen._pipeline is not None
        assert gen._backbone is not None
        assert gen._sigma > 0

    def test_auto_with_view(self, minimal_instance):
        """Test auto configuration with view type."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto(view="top")

        assert gen._rotation_range == (-180.0, 180.0)

    def test_fluent_api(self, minimal_instance):
        """Test fluent API chaining."""
        gen = (
            ConfigGenerator.from_slp(str(minimal_instance))
            .auto()
            .pipeline("single_instance")
            .backbone("unet_medium_rf")
            .batch_size(8)
            .max_epochs(100)
            .learning_rate(0.001)
            .sigma(3.0)
        )

        assert gen._pipeline == "single_instance"
        assert gen._backbone == "unet_medium_rf"
        assert gen._batch_size == 8
        assert gen._max_epochs == 100
        assert gen._learning_rate == 0.001
        assert gen._sigma == 3.0

    def test_build_config(self, minimal_instance):
        """Test building configuration."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto()
        config = gen.build()

        assert isinstance(config, DictConfig)
        assert "data_config" in config
        assert "model_config" in config
        assert "trainer_config" in config

    def test_build_requires_pipeline(self, minimal_instance):
        """Test that build fails without pipeline."""
        gen = ConfigGenerator.from_slp(str(minimal_instance))

        with pytest.raises(ValueError, match="Pipeline not set"):
            gen.build()

    def test_to_yaml(self, minimal_instance):
        """Test YAML output."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto()
        yaml_str = gen.to_yaml()

        assert isinstance(yaml_str, str)
        assert "data_config" in yaml_str
        assert "model_config" in yaml_str

    def test_save(self, minimal_instance):
        """Test saving to file."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            gen.save(f.name)
            assert Path(f.name).exists()
            content = Path(f.name).read_text()
            assert "data_config" in content
        Path(f.name).unlink()

    def test_summary(self, minimal_instance):
        """Test summary output."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto()
        summary = gen.summary()

        assert isinstance(summary, str)
        assert "Dataset" in summary
        assert "Configuration" in summary
        assert "Memory" in summary

    def test_memory_estimate(self, minimal_instance):
        """Test memory estimation method."""
        gen = ConfigGenerator.from_slp(str(minimal_instance)).auto()
        mem = gen.memory_estimate()

        assert isinstance(mem, MemoryEstimate)


class TestGenerateConfig:
    """Tests for generate_config convenience function."""

    def test_generate_config(self, minimal_instance):
        """Test convenience function."""
        config = generate_config(str(minimal_instance))

        assert isinstance(config, DictConfig)
        assert "data_config" in config

    def test_generate_config_with_output(self, minimal_instance):
        """Test generating and saving config."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config = generate_config(str(minimal_instance), f.name)
            assert Path(f.name).exists()
        Path(f.name).unlink()

    def test_generate_config_with_overrides(self, minimal_instance):
        """Test generating config with overrides."""
        config = generate_config(str(minimal_instance), batch_size=16)

        assert config.trainer_config.train_data_loader.batch_size == 16


class TestConfigGeneratorBottomUp:
    """Tests for bottom-up configurations."""

    def test_bottomup_config(self, minimal_instance):
        """Test bottom-up configuration builds correctly."""
        gen = (
            ConfigGenerator.from_slp(str(minimal_instance))
            .pipeline("bottomup")
            .backbone("unet_medium_rf")
        )
        config = gen.build()

        # Bottom-up should have PAF configuration
        assert "bottomup" in config.model_config.head_configs
        assert "pafs" in config.model_config.head_configs.bottomup
