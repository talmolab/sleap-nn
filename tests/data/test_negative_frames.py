"""Tests for negative frame support in training datasets."""

import numpy as np
import pytest
import torch
import sleap_io as sio
from omegaconf import DictConfig
from unittest.mock import MagicMock

from sleap_nn.data.providers import process_negative_lf
from sleap_nn.data.custom_datasets import (
    BottomUpDataset,
    SingleInstanceDataset,
    CentroidDataset,
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

        n_neg = sum(1 for s in dataset.lf_idx_list if s.get("is_negative", False))
        assert n_neg == 0

    def test_no_negatives_when_none_marked(self, minimal_instance):
        """Test that no negatives added when labels have no negative_frames."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        # Even with fraction > 0, no negatives if none are marked by user
        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.5,
        )

        n_neg = sum(1 for s in dataset.lf_idx_list if s.get("is_negative", False))
        assert n_neg == 0

    def test_collect_uses_only_user_confirmed_negatives(self, minimal_instance):
        """Test _collect_negative_frames only uses labels.negative_frames."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        mock_video = MagicMock()
        mock_video.shape = (100, 384, 384, 1)

        # User-confirmed negative frames
        mock_neg_lf1 = MagicMock()
        mock_neg_lf1.video = mock_video
        mock_neg_lf1.frame_idx = 50
        mock_neg_lf2 = MagicMock()
        mock_neg_lf2.video = mock_video
        mock_neg_lf2.frame_idx = 60

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.negative_frames = [mock_neg_lf1, mock_neg_lf2]

        neg_samples = dataset._collect_negative_frames([mock_labels], n_negatives=5)

        # Should only get 2 (the user-confirmed ones), not 5
        assert len(neg_samples) == 2
        neg_frame_indices = {s["frame_idx"] for s in neg_samples}
        assert neg_frame_indices == {50, 60}
        for s in neg_samples:
            assert s["is_negative"] is True
            assert s["lf_idx"] is None

    def test_collect_does_not_sample_unlabeled_frames(self, minimal_instance):
        """Test that unlabeled frames are never sampled as negatives."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        mock_video = MagicMock()
        mock_video.shape = (1000, 384, 384, 1)  # Many unlabeled frames

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.negative_frames = []  # No user-confirmed negatives

        # Request 100 negatives but none are marked — should get 0
        neg_samples = dataset._collect_negative_frames([mock_labels], n_negatives=100)
        assert len(neg_samples) == 0

    def test_collect_truncates_to_budget(self, minimal_instance):
        """Test that results are truncated when more negatives exist than requested."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        mock_video = MagicMock()
        mock_video.shape = (100, 384, 384, 1)

        # 5 user-confirmed negatives
        neg_lfs = []
        for i in range(5):
            lf = MagicMock()
            lf.video = mock_video
            lf.frame_idx = 10 + i
            neg_lfs.append(lf)

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.negative_frames = neg_lfs

        # Request only 2
        neg_samples = dataset._collect_negative_frames([mock_labels], n_negatives=2)
        assert len(neg_samples) == 2

    def test_collect_from_multiple_label_files(self, minimal_instance):
        """Test negatives collected across multiple label files."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        mock_video1 = MagicMock()
        mock_video1.shape = (50, 384, 384, 1)
        mock_video2 = MagicMock()
        mock_video2.shape = (50, 384, 384, 1)

        neg_lf1 = MagicMock()
        neg_lf1.video = mock_video1
        neg_lf1.frame_idx = 10
        mock_labels1 = MagicMock()
        mock_labels1.videos = [mock_video1]
        mock_labels1.negative_frames = [neg_lf1]

        neg_lf2 = MagicMock()
        neg_lf2.video = mock_video2
        neg_lf2.frame_idx = 20
        neg_lf3 = MagicMock()
        neg_lf3.video = mock_video2
        neg_lf3.frame_idx = 30
        mock_labels2 = MagicMock()
        mock_labels2.videos = [mock_video2]
        mock_labels2.negative_frames = [neg_lf2, neg_lf3]

        neg_samples = dataset._collect_negative_frames(
            [mock_labels1, mock_labels2], n_negatives=10
        )

        # Should get all 3 from both label files
        assert len(neg_samples) == 3
        labels_indices = {s["labels_idx"] for s in neg_samples}
        assert labels_indices == {0, 1}

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

        for s in dataset.lf_idx_list:
            assert s.get("is_negative", False) is False
            assert s["lf_idx"] is not None

    def test_is_negative_in_returned_sample(self, minimal_instance):
        """Test that is_negative key is present in returned sample dict."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
        )

        # Positive sample
        sample = dataset[0]
        assert "is_negative" in sample
        assert sample["is_negative"] is False

        # Inject and test negative sample
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
        assert "is_negative" in sample
        assert sample["is_negative"] is True

    def test_caching_with_negatives(self, minimal_instance):
        """Test that memory caching works with negative_sample_fraction > 0."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            negative_sample_fraction=0.0,
            cache_img="memory",
        )

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
        assert sample["is_negative"] is True


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

    def test_rejects_invalid_values(self):
        """Test that negative_sample_fraction rejects values outside [0, 1]."""
        from sleap_nn.config.data_config import DataConfig

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            DataConfig(negative_sample_fraction=-0.1)

        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            DataConfig(negative_sample_fraction=1.5)

    def test_negative_loss_weight_default(self):
        """Test that negative_loss_weight defaults to 1.0."""
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig()
        assert config.negative_loss_weight == 1.0

    def test_negative_loss_weight_custom(self):
        """Test that negative_loss_weight can be set."""
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig(negative_loss_weight=0.5)
        assert config.negative_loss_weight == 0.5

    def test_negative_loss_weight_rejects_zero(self):
        """Test that negative_loss_weight rejects non-positive values."""
        from sleap_nn.config.data_config import DataConfig

        with pytest.raises((ValueError, Exception)):
            DataConfig(negative_loss_weight=0.0)

        with pytest.raises((ValueError, Exception)):
            DataConfig(negative_loss_weight=-1.0)
