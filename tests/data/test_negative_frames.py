"""Tests for negative frame support in training datasets."""

import numpy as np
import pytest
import torch
import sleap_io as sio
from omegaconf import DictConfig
from unittest.mock import patch, MagicMock

from sleap_nn.data.providers import process_negative_lf
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    SingleInstanceDataset,
    CentroidDataset,
    BaseDataset,
)


class TestProcessNegativeLf:
    """Tests for the process_negative_lf helper function."""

    def test_basic_output(self):
        """Test that process_negative_lf returns correctly shaped tensors."""
        img = np.random.randint(0, 255, (64, 64, 1), dtype=np.uint8)
        result = process_negative_lf(
            img=img,
            frame_idx=5,
            video_idx=0,
            max_instances=3,
            num_nodes=2,
        )

        assert result["image"].shape == (1, 1, 64, 64)
        assert result["instances"].shape == (1, 3, 2, 2)
        assert torch.all(torch.isnan(result["instances"]))
        assert result["num_instances"] == 0
        assert result["frame_idx"] == 5
        assert result["video_idx"] == 0
        assert result["orig_size"].shape == (1, 2)

    def test_rgb_image(self):
        """Test with 3-channel image."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = process_negative_lf(
            img=img,
            frame_idx=0,
            video_idx=0,
            max_instances=1,
            num_nodes=4,
        )
        assert result["image"].shape == (1, 3, 64, 64)
        assert result["instances"].shape == (1, 1, 4, 2)

    def test_single_max_instance(self):
        """Test with max_instances=0 (edge case, should use at least 1)."""
        img = np.random.randint(0, 255, (32, 32, 1), dtype=np.uint8)
        result = process_negative_lf(
            img=img,
            frame_idx=0,
            video_idx=0,
            max_instances=0,
            num_nodes=2,
        )
        # max(0, 1) = 1, so instances shape should still have at least 1
        assert result["instances"].shape == (1, 1, 2, 2)


class TestNegativeSampleFraction:
    """Tests for negative_sample_fraction in dataset classes."""

    def test_no_negatives_by_default(self, minimal_instance):
        """Test that no negative frames are added when fraction is 0."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        # Count negative samples
        n_neg = sum(1 for s in dataset.lf_idx_list if s.get("is_negative", False))
        assert n_neg == 0

    def test_no_negatives_when_no_unlabeled_frames(self, minimal_instance):
        """Test that no negatives added when all frames are labeled (1-frame video)."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        # The minimal_instance has only 1 frame which is already labeled
        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.5,
        )

        # No unlabeled frames available, so no negatives should be added
        n_neg = sum(1 for s in dataset.lf_idx_list if s.get("is_negative", False))
        assert n_neg == 0

    def test_collect_negative_frames_logic(self, minimal_instance):
        """Test _collect_negative_frames with a multi-frame video mock."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,  # Don't auto-collect
        )

        # Mock the video to have 10 frames
        mock_video = MagicMock()
        mock_video.shape = (10, 384, 384, 1)

        # Replace labels videos with mock
        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.suggestions = []
        # Simulate 1 labeled frame at index 0
        mock_lf = MagicMock()
        mock_lf.frame_idx = 0
        mock_lf.video = mock_video
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))

        neg_samples = dataset._collect_negative_frames([mock_labels], n_negatives=3)

        assert len(neg_samples) == 3
        for s in neg_samples:
            assert s["is_negative"] is True
            assert s["lf_idx"] is None
            assert s["frame_idx"] != 0  # Should not be the labeled frame
            assert s["frame_idx"] < 10

    def test_collect_negative_frames_from_suggestions(self, minimal_instance):
        """Test _collect_negative_frames uses suggestions first."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        # Mock video with 100 frames
        mock_video = MagicMock()
        mock_video.shape = (100, 384, 384, 1)

        # Mock suggestions
        mock_sf1 = MagicMock()
        mock_sf1.video = mock_video
        mock_sf1.frame_idx = 50
        mock_sf2 = MagicMock()
        mock_sf2.video = mock_video
        mock_sf2.frame_idx = 60

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.suggestions = [mock_sf1, mock_sf2]
        mock_lf = MagicMock()
        mock_lf.frame_idx = 0
        mock_lf.video = mock_video
        mock_labels.__iter__ = MagicMock(return_value=iter([mock_lf]))

        neg_samples = dataset._collect_negative_frames([mock_labels], n_negatives=2)

        assert len(neg_samples) == 2
        # Should use suggestion frames first
        suggestion_frames = {s["frame_idx"] for s in neg_samples}
        assert 50 in suggestion_frames or 60 in suggestion_frames

    def test_negative_frame_getitem_produces_zero_confmaps(self, minimal_instance):
        """Test that a manually-injected negative frame produces all-zero confmaps."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        # Manually inject a negative frame at the only available frame (index 0)
        # Since the video has only 1 frame, we reuse frame 0 as a "negative"
        neg_sample = {
            "labels_idx": 0,
            "lf_idx": None,
            "video_idx": 0,
            "frame_idx": 0,
            "is_negative": True,
            "instances": None,
        }
        dataset.lf_idx_list.append(neg_sample)

        neg_idx = len(dataset.lf_idx_list) - 1
        sample = dataset[neg_idx]

        # Confidence maps should be all zeros for negative frames
        assert torch.all(sample["confidence_maps"] == 0)
        assert sample["num_instances"] == 0

    def test_positive_frame_produces_nonzero_confmaps(self, minimal_instance):
        """Test that positive frames still produce non-zero confidence maps."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        sample = dataset[0]
        # Confidence maps should be non-zero for positive frames
        assert torch.any(sample["confidence_maps"] > 0)
        assert sample["num_instances"] > 0

    def test_bottomup_negative_frame(self, minimal_instance):
        """Test negative frames in BottomUpDataset."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
        pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

        dataset = BottomUpDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            pafs_head_config=pafs_head,
            max_stride=32,
            negative_sample_fraction=0.0,
        )

        # Manually inject a negative frame
        neg_sample = {
            "labels_idx": 0,
            "lf_idx": None,
            "video_idx": 0,
            "frame_idx": 0,
            "is_negative": True,
            "instances": None,
        }
        dataset.lf_idx_list.append(neg_sample)

        neg_idx = len(dataset.lf_idx_list) - 1
        sample = dataset[neg_idx]

        assert torch.all(sample["confidence_maps"] == 0)
        assert torch.all(sample["part_affinity_fields"] == 0)
        assert sample["num_instances"] == 0

    def test_centroid_negative_frame(self, minimal_instance):
        """Test negative frames in CentroidDataset."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = CentroidDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        # Manually inject a negative frame
        neg_sample = {
            "labels_idx": 0,
            "lf_idx": None,
            "video_idx": 0,
            "frame_idx": 0,
            "is_negative": True,
            "instances": None,
        }
        dataset.lf_idx_list.append(neg_sample)

        neg_idx = len(dataset.lf_idx_list) - 1
        sample = dataset[neg_idx]

        assert torch.all(sample["centroids_confidence_maps"] == 0)
        assert sample["num_instances"] == 0

    def test_is_negative_flag_set(self, minimal_instance):
        """Test that positive frames have is_negative=False in lf_idx_list."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        # All positive frames should have is_negative=False
        for s in dataset.lf_idx_list:
            assert s.get("is_negative", False) is False
            assert s["lf_idx"] is not None


class TestDataConfigNegativeFraction:
    """Test that the DataConfig accepts negative_sample_fraction."""

    def test_default_value(self):
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig()
        assert config.negative_sample_fraction == 0.0

    def test_custom_value(self):
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig(negative_sample_fraction=0.2)
        assert config.negative_sample_fraction == 0.2
