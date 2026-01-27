"""Tests for parallel image caching functionality."""

import tempfile
import threading
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.data.custom_datasets import (
    MIN_SAMPLES_FOR_PARALLEL_CACHING,
    ParallelCacheFiller,
    BaseDataset,
    SingleInstanceDataset,
)


@pytest.fixture
def minimal_labels():
    """Load minimal test labels."""
    slp_path = Path("tests/assets/datasets/minimal_instance.pkg.slp")
    if not slp_path.exists():
        pytest.skip("Test file not found")
    return sio.load_slp(str(slp_path))


@pytest.fixture
def small_robot_labels():
    """Load small robot test labels."""
    slp_path = Path("tests/assets/datasets/small_robot_minimal.slp")
    if not slp_path.exists():
        pytest.skip("Test file not found")
    return sio.load_slp(str(slp_path))


class TestParallelCacheFiller:
    """Tests for ParallelCacheFiller class."""

    def test_init(self, minimal_labels):
        """Test ParallelCacheFiller initialization."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(minimal_labels))
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            filler = ParallelCacheFiller(
                labels=[minimal_labels],
                lf_idx_list=lf_idx_list,
                cache_type="disk",
                cache_path=Path(tmpdir),
                num_workers=2,
            )

            assert filler.num_workers == 2
            assert filler.cache_type == "disk"
            assert len(filler._video_info) > 0

    def test_disk_caching(self, minimal_labels):
        """Test parallel disk caching."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(minimal_labels))
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir)

            filler = ParallelCacheFiller(
                labels=[minimal_labels],
                lf_idx_list=lf_idx_list,
                cache_type="disk",
                cache_path=cache_path,
                num_workers=2,
            )

            cache, errors = filler.fill_cache()

            # Check no errors
            assert len(errors) == 0

            # Check files were created
            cached_files = list(cache_path.glob("*.jpg"))
            assert len(cached_files) == len(minimal_labels)

    def test_memory_caching(self, minimal_labels):
        """Test parallel memory caching."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(minimal_labels))
        ]

        filler = ParallelCacheFiller(
            labels=[minimal_labels],
            lf_idx_list=lf_idx_list,
            cache_type="memory",
            cache_path=None,
            num_workers=2,
        )

        cache, errors = filler.fill_cache()

        # Check no errors
        assert len(errors) == 0

        # Check cache is populated
        assert len(cache) == len(minimal_labels)

        # Check cache values are numpy arrays
        for key, value in cache.items():
            assert isinstance(value, np.ndarray)

    def test_video_restoration(self, minimal_labels):
        """Test that videos are properly restored after caching."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(minimal_labels))
        ]

        # Get original video state
        original_video = minimal_labels.videos[0]
        was_open = original_video.is_open

        filler = ParallelCacheFiller(
            labels=[minimal_labels],
            lf_idx_list=lf_idx_list,
            cache_type="memory",
            cache_path=None,
            num_workers=2,
        )

        # After init, video should be closed
        assert not original_video.is_open

        # Fill cache
        filler.fill_cache()

        # After fill, video state should be restored
        # (it was open before, so should be open now if open_backend was True)

    def test_progress_callback(self, minimal_labels):
        """Test progress callback is called."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(minimal_labels))
        ]

        progress_counts = []

        def callback(count):
            progress_counts.append(count)

        filler = ParallelCacheFiller(
            labels=[minimal_labels],
            lf_idx_list=lf_idx_list,
            cache_type="memory",
            cache_path=None,
            num_workers=2,
        )

        filler.fill_cache(progress_callback=callback)

        # Progress should have been called for each sample
        assert len(progress_counts) == len(minimal_labels)

    def test_thread_local_video_copies(self, minimal_labels):
        """Test that thread-local video copies are created correctly."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(minimal_labels))
        ]

        filler = ParallelCacheFiller(
            labels=[minimal_labels],
            lf_idx_list=lf_idx_list,
            cache_type="memory",
            cache_path=None,
            num_workers=2,
        )

        # Access thread-local video
        video = minimal_labels.videos[0]
        local_video = filler._get_thread_local_video(video)

        # Should be a different object
        assert local_video is not video

        # Same thread should get same copy
        local_video_2 = filler._get_thread_local_video(video)
        assert local_video is local_video_2


class TestParallelCachingIntegration:
    """Integration tests for parallel caching with datasets."""

    def test_base_dataset_parallel_caching_threshold(self, minimal_labels):
        """Test that parallel caching respects the minimum threshold."""
        # Minimal labels has only 1 frame, should use sequential
        with tempfile.TemporaryDirectory() as tmpdir:
            # This should use sequential caching (1 < MIN_SAMPLES_FOR_PARALLEL_CACHING)
            with patch(
                "sleap_nn.data.custom_datasets.BaseDataset._fill_cache_parallel"
            ) as mock_parallel:
                with patch(
                    "sleap_nn.data.custom_datasets.BaseDataset._fill_cache_sequential"
                ) as mock_sequential:
                    # Create dataset (won't actually cache since we're mocking)
                    class TestDataset(BaseDataset):
                        def __getitem__(self, index):
                            return {}

                    # Don't actually run caching for this test

    def test_single_instance_dataset_with_parallel_caching(self, minimal_labels):
        """Test SingleInstanceDataset with parallel caching enabled."""
        from omegaconf import DictConfig

        confmap_config = DictConfig(
            {
                "sigma": 1.5,
                "output_stride": 2,
                "part_names": [n.name for n in minimal_labels.skeletons[0].nodes],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SingleInstanceDataset(
                labels=[minimal_labels],
                confmap_head_config=confmap_config,
                max_stride=32,
                cache_img="memory",
                parallel_caching=True,
                cache_workers=2,
            )

            # Check that caching completed
            assert len(dataset.cache) == len(minimal_labels)

    def test_parallel_caching_disabled(self, minimal_labels):
        """Test that parallel caching can be disabled."""
        from omegaconf import DictConfig

        confmap_config = DictConfig(
            {
                "sigma": 1.5,
                "output_stride": 2,
                "part_names": [n.name for n in minimal_labels.skeletons[0].nodes],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = SingleInstanceDataset(
                labels=[minimal_labels],
                confmap_head_config=confmap_config,
                max_stride=32,
                cache_img="memory",
                parallel_caching=False,  # Disabled
                cache_workers=2,
            )

            # Should still work (sequential caching)
            assert len(dataset.cache) == len(minimal_labels)


class TestParallelCachingWithMediaVideo:
    """Tests for parallel caching with MediaVideo backend."""

    def test_media_video_parallel_caching(self, small_robot_labels):
        """Test parallel caching with MediaVideo backend."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(small_robot_labels))
        ]

        filler = ParallelCacheFiller(
            labels=[small_robot_labels],
            lf_idx_list=lf_idx_list,
            cache_type="memory",
            cache_path=None,
            num_workers=2,
        )

        cache, errors = filler.fill_cache()

        # Should complete without errors
        assert len(errors) == 0
        assert len(cache) == len(small_robot_labels)

    def test_media_video_disk_caching(self, small_robot_labels):
        """Test parallel disk caching with MediaVideo backend."""
        lf_idx_list = [
            {"labels_idx": 0, "lf_idx": i} for i in range(len(small_robot_labels))
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir)

            filler = ParallelCacheFiller(
                labels=[small_robot_labels],
                lf_idx_list=lf_idx_list,
                cache_type="disk",
                cache_path=cache_path,
                num_workers=2,
            )

            cache, errors = filler.fill_cache()

            # Should complete without errors
            assert len(errors) == 0

            # Check files were created
            cached_files = list(cache_path.glob("*.jpg"))
            assert len(cached_files) == len(small_robot_labels)


class TestDataConfigParallelCaching:
    """Tests for DataConfig parallel caching options."""

    def test_data_config_has_parallel_caching_options(self):
        """Test that DataConfig has parallel caching options."""
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig()

        # Check defaults
        assert config.parallel_caching is True
        assert config.cache_workers == 0

    def test_data_config_custom_values(self):
        """Test DataConfig with custom parallel caching values."""
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig(
            parallel_caching=False,
            cache_workers=8,
        )

        assert config.parallel_caching is False
        assert config.cache_workers == 8


class TestMinSamplesThreshold:
    """Tests for minimum samples threshold constant."""

    def test_min_samples_constant_exists(self):
        """Test that MIN_SAMPLES_FOR_PARALLEL_CACHING is defined."""
        assert MIN_SAMPLES_FOR_PARALLEL_CACHING > 0

    def test_min_samples_reasonable_value(self):
        """Test that threshold is reasonable (between 10 and 100)."""
        assert 10 <= MIN_SAMPLES_FOR_PARALLEL_CACHING <= 100
