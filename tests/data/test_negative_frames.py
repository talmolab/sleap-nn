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
    """Tests for use_negative_frames in dataset classes."""

    def test_no_negatives_by_default(self, minimal_instance):
        """Test that no negative frames are added when fraction is 0."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=False,
        )

        n_neg = sum(1 for s in dataset.lf_idx_list if s.get("is_negative", False))
        assert n_neg == 0

    def test_no_negatives_when_none_marked(self, minimal_instance):
        """Test that no negatives added when labels have no negative_frames."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        # Even with feature on, no negatives if none are marked by user
        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=True,
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
            use_negative_frames=False,
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

        neg_samples = dataset._collect_negative_frames([mock_labels])

        assert len(neg_samples) == 2
        neg_frame_indices = {s["frame_idx"] for s in neg_samples}
        assert neg_frame_indices == {50, 60}
        for s in neg_samples:
            assert s["is_negative"] is True
            assert s["lf_idx"].startswith("neg_")

    def test_collect_does_not_sample_unlabeled_frames(self, minimal_instance):
        """Test that unlabeled frames are never sampled as negatives."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=False,
        )

        mock_video = MagicMock()
        mock_video.shape = (1000, 384, 384, 1)  # Many unlabeled frames

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.negative_frames = []  # No user-confirmed negatives

        neg_samples = dataset._collect_negative_frames([mock_labels])
        assert len(neg_samples) == 0

    def test_collect_from_multiple_label_files(self, minimal_instance):
        """Test negatives collected across multiple label files."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=False,
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

        neg_samples = dataset._collect_negative_frames([mock_labels1, mock_labels2])

        # Should get all 3 from both label files
        assert len(neg_samples) == 3
        labels_indices = {s["labels_idx"] for s in neg_samples}
        assert labels_indices == {0, 1}

    def test_negatives_included_once_each(self, minimal_instance):
        """Test that negatives are included once each, not oversampled."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=False,
        )

        mock_video = MagicMock()
        mock_video.shape = (100, 384, 384, 1)

        neg_lfs = []
        for i in range(3):
            lf = MagicMock()
            lf.video = mock_video
            lf.frame_idx = 10 + i
            neg_lfs.append(lf)

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]
        mock_labels.negative_frames = neg_lfs

        neg_samples = dataset._collect_negative_frames([mock_labels])
        assert len(neg_samples) == 3
        frame_indices = [s["frame_idx"] for s in neg_samples]
        assert frame_indices == [10, 11, 12]

    def test_negative_frame_getitem_produces_zero_confmaps(self, minimal_instance):
        """Test that a manually-injected negative frame produces all-zero confmaps."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=False,
        )

        neg_sample = {
            "labels_idx": 0,
            "lf_idx": "neg_0_0",
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
            use_negative_frames=False,
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
            use_negative_frames=False,
        )

        neg_sample = {
            "labels_idx": 0,
            "lf_idx": "neg_0_0",
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
            use_negative_frames=False,
        )

        neg_sample = {
            "labels_idx": 0,
            "lf_idx": "neg_0_0",
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
            use_negative_frames=False,
        )

        for s in dataset.lf_idx_list:
            assert s.get("is_negative", False) is False
            assert s["lf_idx"] is not None

    def test_is_negative_not_in_sample_when_feature_off(self, minimal_instance):
        """Test that is_negative key is absent when fraction=0 (feature off)."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=False,
        )

        sample = dataset[0]
        assert "is_negative" not in sample

    def test_is_negative_in_returned_sample(self, minimal_instance):
        """Test that is_negative key is present when fraction > 0."""
        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceDataset(
            labels=[labels],
            confmap_head_config=confmap_head,
            max_stride=8,
            use_negative_frames=True,  # Feature on (even if no negatives found)
        )

        # Positive sample should have is_negative=False
        sample = dataset[0]
        assert "is_negative" in sample
        assert sample["is_negative"] is False

        # Inject and test negative sample
        neg_sample = {
            "labels_idx": 0,
            "lf_idx": "neg_0_0",
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
        """Test that memory caching works with negative frames."""
        from unittest.mock import patch, PropertyMock

        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        # Create a mock negative frame using a valid video/frame
        video = labels.videos[0]
        frame_idx = labels.labeled_frames[0].frame_idx
        neg_lf = MagicMock()
        neg_lf.video = video
        neg_lf.frame_idx = frame_idx

        with patch.object(
            type(labels),
            "negative_frames",
            new_callable=PropertyMock,
            return_value=[neg_lf],
        ):
            dataset = SingleInstanceDataset(
                labels=[labels],
                confmap_head_config=confmap_head,
                max_stride=8,
                use_negative_frames=True,
                cache_img="memory",
            )

        n_neg = sum(1 for s in dataset.lf_idx_list if s.get("is_negative", False))
        assert n_neg == 1

        neg_idx = next(
            i
            for i, s in enumerate(dataset.lf_idx_list)
            if s.get("is_negative", False)
        )
        sample = dataset[neg_idx]

        assert torch.all(sample["confidence_maps"] == 0)
        assert sample["is_negative"] is True

    def test_disk_caching_with_negatives(self, minimal_instance, tmp_path):
        """Test that disk caching works with negative frames (regression #504)."""
        from unittest.mock import patch, PropertyMock

        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        video = labels.videos[0]
        frame_idx = labels.labeled_frames[0].frame_idx
        neg_lf = MagicMock()
        neg_lf.video = video
        neg_lf.frame_idx = frame_idx

        with patch.object(
            type(labels),
            "negative_frames",
            new_callable=PropertyMock,
            return_value=[neg_lf],
        ):
            dataset = SingleInstanceDataset(
                labels=[labels],
                confmap_head_config=confmap_head,
                max_stride=8,
                use_negative_frames=True,
                cache_img="disk",
                cache_img_path=str(tmp_path),
            )

        # Verify negative frame image was cached to disk
        neg_sample = next(
            s for s in dataset.lf_idx_list if s.get("is_negative", False)
        )
        cache_file = tmp_path / f"sample_{neg_sample['labels_idx']}_{neg_sample['lf_idx']}.jpg"
        assert cache_file.exists()

        # Verify it can be loaded from cache
        neg_idx = next(
            i
            for i, s in enumerate(dataset.lf_idx_list)
            if s.get("is_negative", False)
        )
        sample = dataset[neg_idx]

        assert torch.all(sample["confidence_maps"] == 0)
        assert sample["num_instances"] == 0
        assert sample["is_negative"] is True

    def test_parallel_caching_with_negatives(self, minimal_instance, tmp_path):
        """Test that parallel caching works with negative frames."""
        from unittest.mock import patch, PropertyMock

        labels = sio.load_slp(minimal_instance)
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        video = labels.videos[0]
        frame_idx = labels.labeled_frames[0].frame_idx
        neg_lf = MagicMock()
        neg_lf.video = video
        neg_lf.frame_idx = frame_idx

        with patch.object(
            type(labels),
            "negative_frames",
            new_callable=PropertyMock,
            return_value=[neg_lf],
        ), patch(
            "sleap_nn.data.custom_datasets.MIN_SAMPLES_FOR_PARALLEL_CACHING", 0
        ):
            dataset = SingleInstanceDataset(
                labels=[labels],
                confmap_head_config=confmap_head,
                max_stride=8,
                use_negative_frames=True,
                cache_img="disk",
                cache_img_path=str(tmp_path),
                parallel_caching=True,
            )

        # Verify negative was cached and can be loaded
        neg_idx = next(
            i
            for i, s in enumerate(dataset.lf_idx_list)
            if s.get("is_negative", False)
        )
        sample = dataset[neg_idx]

        assert torch.all(sample["confidence_maps"] == 0)
        assert sample["num_instances"] == 0
        assert sample["is_negative"] is True


class TestNegativeLossWeighting:
    """Tests for _compute_negative_weighted_loss in lightning modules."""

    def _make_model(self, weight=1.0):
        """Create a minimal mock for testing _compute_negative_weighted_loss."""
        from sleap_nn.training.lightning_modules import LightningModel
        from unittest.mock import MagicMock

        model = object.__new__(LightningModel)
        model.negative_loss_weight = weight
        model.log = MagicMock()
        return model

    def test_weighted_loss_no_negatives_in_batch(self):
        """When is_negative is present but all False, loss equals MSELoss."""
        model = self._make_model(weight=2.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, False, False, False])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch)
        expected = torch.nn.MSELoss()(y_preds, y)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_weighted_loss_with_negatives(self):
        """When negatives are present, loss is weighted."""
        model = self._make_model(weight=5.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, False, True, False])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch)

        per_sample = (y_preds - y).pow(2).mean(dim=[1, 2, 3])
        weights = torch.tensor([1.0, 1.0, 5.0, 1.0])
        expected = (per_sample * weights).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_weighted_loss_weight_1_skips_weighting(self):
        """When weight=1.0 and is_negative present, still equals MSELoss."""
        model = self._make_model(weight=1.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, True, False, True])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch)
        expected = torch.nn.MSELoss()(y_preds, y)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_weighted_loss_no_is_negative_key(self):
        """When is_negative absent from batch, returns plain MSELoss."""
        model = self._make_model(weight=5.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch)
        expected = torch.nn.MSELoss()(y_preds, y)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_weighted_loss_all_negative_batch(self):
        """When all samples are negative, all get the negative weight."""
        model = self._make_model(weight=0.5)

        y_preds = torch.zeros(2, 1, 4, 4)
        y = torch.zeros(2, 1, 4, 4)
        batch = {"is_negative": torch.tensor([True, True])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch)
        assert loss.item() == 0.0

    def test_split_metrics_logged(self):
        """Verify split loss and count metrics are logged."""
        model = self._make_model(weight=2.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, False, True, False])}

        model._compute_negative_weighted_loss(y_preds, y, batch)

        logged_keys = {call.args[0] for call in model.log.call_args_list}
        assert "train/loss_positive" in logged_keys
        assert "train/loss_negative" in logged_keys
        assert "train/n_positive" in logged_keys
        assert "train/n_negative" in logged_keys

        # Check counts
        for call in model.log.call_args_list:
            if call.args[0] == "train/n_positive":
                assert call.args[1] == 3.0
            if call.args[0] == "train/n_negative":
                assert call.args[1] == 1.0

    def test_no_metrics_when_is_negative_absent(self):
        """No split metrics logged when is_negative is not in batch."""
        model = self._make_model(weight=2.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {}

        model._compute_negative_weighted_loss(y_preds, y, batch)
        model.log.assert_not_called()


class TestDataConfigNegativeFrames:
    """Test that the DataConfig accepts use_negative_frames."""

    def test_default_value(self):
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig()
        assert config.use_negative_frames is False

    def test_enabled(self):
        from sleap_nn.config.data_config import DataConfig

        config = DataConfig(use_negative_frames=True)
        assert config.use_negative_frames is True

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
