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
            i for i, s in enumerate(dataset.lf_idx_list) if s.get("is_negative", False)
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
        neg_sample = next(s for s in dataset.lf_idx_list if s.get("is_negative", False))
        cache_file = (
            tmp_path / f"sample_{neg_sample['labels_idx']}_{neg_sample['lf_idx']}.jpg"
        )
        assert cache_file.exists()

        # Verify it can be loaded from cache
        neg_idx = next(
            i for i, s in enumerate(dataset.lf_idx_list) if s.get("is_negative", False)
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

        with (
            patch.object(
                type(labels),
                "negative_frames",
                new_callable=PropertyMock,
                return_value=[neg_lf],
            ),
            patch("sleap_nn.data.custom_datasets.MIN_SAMPLES_FOR_PARALLEL_CACHING", 0),
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
            i for i, s in enumerate(dataset.lf_idx_list) if s.get("is_negative", False)
        )
        sample = dataset[neg_idx]

        assert torch.all(sample["confidence_maps"] == 0)
        assert sample["num_instances"] == 0
        assert sample["is_negative"] is True


class TestNegativeLossWeighting:
    """Tests for _compute_negative_weighted_loss and _log_negative_split_metrics."""

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

    def test_helper_does_not_log(self):
        """The pure loss helper no longer logs anything (train or val)."""
        model = self._make_model(weight=2.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, False, True, False])}

        model._compute_negative_weighted_loss(y_preds, y, batch)
        model._compute_negative_weighted_loss(y_preds, y, batch, stage="val")
        model.log.assert_not_called()

    def test_val_stage_never_weighted(self):
        """stage='val' returns UNWEIGHTED MSE even when weight != 1.0."""
        model = self._make_model(weight=5.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, False, True, False])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch, stage="val")
        # Equals plain MSELoss == per_sample.mean(); weight=5.0 is ignored.
        expected = torch.nn.MSELoss()(y_preds, y)
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_val_stage_no_is_negative_key(self):
        """stage='val' with is_negative absent => plain MSELoss, nothing logged."""
        model = self._make_model(weight=5.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch, stage="val")
        expected = torch.nn.MSELoss()(y_preds, y)
        assert torch.allclose(loss, expected, atol=1e-6)
        model.log.assert_not_called()

    def test_val_stage_ndim_agnostic_paf_like(self):
        """Per-sample mean over dims 1..ndim works for PAF-like 4D tensors."""
        model = self._make_model(weight=3.0)

        # PAF-like: more channels than confmaps; still 4D (B, 2*n_edges, H, W).
        y_preds = torch.randn(3, 6, 5, 5)
        y = torch.randn(3, 6, 5, 5)
        batch = {"is_negative": torch.tensor([False, True, False])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch, stage="val")
        expected = torch.nn.MSELoss()(y_preds, y)  # unweighted invariant
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_default_stage_is_train(self):
        """Omitting stage keeps train behavior: weighting applied, train/* keys."""
        model = self._make_model(weight=5.0)

        y_preds = torch.randn(4, 2, 8, 8)
        y = torch.randn(4, 2, 8, 8)
        batch = {"is_negative": torch.tensor([False, False, True, False])}

        loss = model._compute_negative_weighted_loss(y_preds, y, batch)

        per_sample = (y_preds - y).pow(2).mean(dim=[1, 2, 3])
        weights = torch.tensor([1.0, 1.0, 5.0, 1.0])
        expected = (per_sample * weights).mean()
        assert torch.allclose(loss, expected, atol=1e-6)

    # ---- _log_negative_split_metrics ----

    @staticmethod
    def _logged(model):
        """Map of logged key -> value from the mock (last write per key)."""
        out = {}
        for call in model.log.call_args_list:
            out[call.args[0]] = call.args[1]
        return out

    def test_log_counts_once_for_two_heads(self):
        """Counts are logged exactly once per call (not 2x) for two heads."""
        model = self._make_model(weight=3.0)

        b = torch.tensor([False, True, True, False])  # n_pos=2, n_neg=2
        cm_p, cm_y = torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8)
        paf_p, paf_y = torch.randn(4, 6, 8, 8), torch.randn(4, 6, 8, 8)
        heads = [("confmaps", cm_p, cm_y, 1.0), ("paf", paf_p, paf_y, 1.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="train")

        n_pos_calls = [
            c for c in model.log.call_args_list if c.args[0] == "train/n_positive"
        ]
        n_neg_calls = [
            c for c in model.log.call_args_list if c.args[0] == "train/n_negative"
        ]
        assert len(n_pos_calls) == 1
        assert len(n_neg_calls) == 1
        assert n_pos_calls[0].args[1] == 2.0  # true count, NOT 4.0 (2x)
        assert n_neg_calls[0].args[1] == 2.0
        # reduce_fx="sum" preserved on the count keys.
        assert n_pos_calls[0].kwargs.get("reduce_fx") == "sum"
        assert n_neg_calls[0].kwargs.get("reduce_fx") == "sum"

    def test_log_per_head_keys_present_for_two_heads(self):
        """Per-head split keys emitted only when there is >1 head."""
        model = self._make_model(weight=2.0)

        b = torch.tensor([False, True, True, False])
        cm = (torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8))
        paf = (torch.randn(4, 6, 8, 8), torch.randn(4, 6, 8, 8))
        heads = [("confmaps", *cm, 1.0), ("paf", *paf, 2.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="train")
        keys = set(self._logged(model))
        assert "train/confmaps_loss_positive" in keys
        assert "train/confmaps_loss_negative" in keys
        assert "train/paf_loss_positive" in keys
        assert "train/paf_loss_negative" in keys

    def test_log_no_per_head_keys_for_single_head(self):
        """Single head => no per-head keys; aggregate == the head value."""
        model = self._make_model(weight=2.0)

        b = torch.tensor([False, True, True, False])
        y_p, y = torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8)
        heads = [("confmaps", y_p, y, 1.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="train")
        logged = self._logged(model)
        assert "train/confmaps_loss_positive" not in logged
        assert "train/confmaps_loss_negative" not in logged

        per_sample = (y_p - y).pow(2).mean(dim=[1, 2, 3])
        exp_pos = per_sample[~b].mean()
        exp_neg = per_sample[b].mean()
        # Single head, weight 1.0: weighted == unweighted == the head value.
        assert torch.allclose(logged["train/loss_positive"], exp_pos, atol=1e-6)
        assert torch.allclose(logged["train/loss_negative"], exp_neg, atol=1e-6)
        assert torch.allclose(
            logged["train/loss_positive_unweighted"], exp_pos, atol=1e-6
        )
        assert torch.allclose(
            logged["train/loss_negative_unweighted"], exp_neg, atol=1e-6
        )

    def test_log_weighted_and_unweighted_aggregates(self):
        """Weighted == sum(weight*head); unweighted == mean(heads); weight=99 ignored."""
        model = self._make_model(weight=99.0)  # must NOT affect logged values

        b = torch.tensor([False, True])  # n_pos=1 (idx0), n_neg=1 (idx1)
        cm_p = torch.tensor([[[[1.0]]], [[[3.0]]]])  # (2,1,1,1); per_sample=[1, 9]
        cm_y = torch.zeros(2, 1, 1, 1)
        paf_p = torch.tensor([[[[2.0]]], [[[4.0]]]])  # per_sample=[4, 16]
        paf_y = torch.zeros(2, 1, 1, 1)
        heads = [("confmaps", cm_p, cm_y, 2.0), ("paf", paf_p, paf_y, 3.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="train")
        logged = self._logged(model)

        # positive (idx 0): cm_pos=1, paf_pos=4
        assert torch.allclose(
            logged["train/loss_positive"], torch.tensor(2.0 * 1.0 + 3.0 * 4.0)
        )  # 14.0
        assert torch.allclose(
            logged["train/loss_positive_unweighted"], torch.tensor((1.0 + 4.0) / 2)
        )  # 2.5
        # negative (idx 1): cm_neg=9, paf_neg=16
        assert torch.allclose(
            logged["train/loss_negative"], torch.tensor(2.0 * 9.0 + 3.0 * 16.0)
        )  # 66.0
        assert torch.allclose(
            logged["train/loss_negative_unweighted"], torch.tensor((9.0 + 16.0) / 2)
        )  # 12.5
        # Per-head values.
        assert torch.allclose(logged["train/confmaps_loss_positive"], torch.tensor(1.0))
        assert torch.allclose(logged["train/paf_loss_negative"], torch.tensor(16.0))

    def test_log_val_prefix_no_cross_leak(self):
        """stage='val' logs only val/* keys, never train/*."""
        model = self._make_model(weight=2.0)

        b = torch.tensor([False, True, True, False])
        y_p, y = torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8)
        heads = [("confmaps", y_p, y, 1.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="val")
        keys = set(self._logged(model))
        assert "val/n_positive" in keys
        assert "val/loss_positive" in keys
        assert not any(k.startswith("train/") for k in keys)

    def test_log_nothing_when_is_negative_absent(self):
        """No is_negative key => logs nothing at all."""
        model = self._make_model(weight=2.0)

        y_p, y = torch.randn(4, 2, 8, 8), torch.randn(4, 2, 8, 8)
        heads = [("confmaps", y_p, y, 1.0)]

        model._log_negative_split_metrics(heads, {}, stage="train")
        model.log.assert_not_called()

    def test_log_all_positive_omits_negative_keys(self):
        """When n_neg==0, negative loss keys are not logged (guard preserved)."""
        model = self._make_model(weight=2.0)

        b = torch.tensor([False, False])  # all positive
        y_p, y = torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4)
        heads = [("confmaps", y_p, y, 1.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="train")
        logged = self._logged(model)
        assert "train/loss_negative" not in logged
        assert "train/loss_negative_unweighted" not in logged
        assert "train/loss_positive" in logged
        # Counts always logged (n_negative == 0.0).
        assert logged["train/n_negative"] == 0.0

    def test_log_all_negative_omits_positive_keys(self):
        """When n_pos==0, positive loss keys are not logged (guard preserved)."""
        model = self._make_model(weight=2.0)

        b = torch.tensor([True, True])  # all negative
        y_p, y = torch.randn(2, 2, 4, 4), torch.randn(2, 2, 4, 4)
        heads = [("confmaps", y_p, y, 1.0)]

        model._log_negative_split_metrics(heads, {"is_negative": b}, stage="train")
        logged = self._logged(model)
        assert "train/loss_positive" not in logged
        assert "train/loss_positive_unweighted" not in logged
        assert "train/loss_negative" in logged
        # Counts always logged (n_positive == 0.0).
        assert logged["train/n_positive"] == 0.0


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
