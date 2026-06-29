"""Tests for ``VideoProvider`` + ``LabelsProvider`` + ``IncrementalLabelsWriter``.

The ``NumpyProvider`` + ``Provider`` protocol live in
``test_predictor_new.py``. This file covers the file-backed providers
and the streaming writer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sleap_nn.inference.outputs import Outputs
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.providers import LabelsProvider, VideoProvider
from sleap_nn.inference.writer import IncrementalLabelsWriter

DATA_ROOT = Path(__file__).resolve().parents[1] / "assets" / "datasets"
VIDEO = DATA_ROOT / "centered_pair_small.mp4"
LABELS = DATA_ROOT / "minimal_instance.pkg.slp"


# ─────────────────────────────────────────────────────────────────────────
# VideoProvider
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_yields_frames_in_batches():
    """A 8-frame slice + batch_size=4 → 2 batches with frame indices 0..7."""
    provider = VideoProvider(video=str(VIDEO), batch_size=4, frames=list(range(8)))
    assert len(provider) == 2
    assert hasattr(provider, "__iter__") and hasattr(provider, "__len__")
    batches = list(provider)
    assert len(batches) == 2
    assert batches[0].images.shape[0] == 4
    assert batches[1].images.shape[0] == 4
    np.testing.assert_array_equal(batches[0].frame_indices, [0, 1, 2, 3])
    np.testing.assert_array_equal(batches[1].frame_indices, [4, 5, 6, 7])
    # video_idx is constant 0 (single source)
    assert (batches[0].video_indices == 0).all()


@pytest.mark.skipif(not VIDEO.exists(), reason="test video not present")
def test_video_provider_uneven_last_batch():
    """5 frames + batch=3 → batches of sizes 3 + 2."""
    provider = VideoProvider(video=str(VIDEO), batch_size=3, frames=list(range(5)))
    sizes = [b.images.shape[0] for b in provider]
    assert sizes == [3, 2]


# ─────────────────────────────────────────────────────────────────────────
# LabelsProvider
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not LABELS.exists(), reason="test labels file not present")
def test_labels_provider_yields_instances():
    """Labels provider attaches GT instances to each batch."""
    provider = LabelsProvider(labels=str(LABELS), batch_size=4)
    assert len(provider) >= 1
    assert hasattr(provider, "__iter__") and hasattr(provider, "__len__")
    batch = next(iter(provider))
    # GT instances are populated.
    assert batch.instances is not None
    assert batch.instances.ndim == 4  # (B, max_inst, n_nodes, 2)
    # Frame indices are the labeled frames' actual indices.
    assert batch.frame_indices is not None
    # Images are the labeled frames.
    assert batch.images.shape[0] == batch.instances.shape[0]


def test_labels_provider_only_labeled_and_exclude_user_labeled_mutually_exclusive():
    """``only_labeled_frames=True`` + ``exclude_user_labeled=True`` raises."""
    import sleap_io as sio

    labels = sio.Labels(videos=[], skeletons=[], labeled_frames=[])
    with pytest.raises(ValueError, match="mutually exclusive"):
        LabelsProvider(
            labels=labels,
            only_labeled_frames=True,
            exclude_user_labeled=True,
        )


def test_labels_provider_exclude_user_labeled_drops_user_frames():
    """``exclude_user_labeled=True`` drops frames with user instances."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=["a", "b"])
    video = sio.Video(filename="dummy.mp4")
    user_inst = sio.Instance.from_numpy(
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), skeleton=skel
    )
    pred_inst = sio.PredictedInstance.from_numpy(
        points_data=np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32),
        skeleton=skel,
        score=0.9,
    )
    lf_user = sio.LabeledFrame(video=video, frame_idx=0, instances=[user_inst])
    lf_pred = sio.LabeledFrame(video=video, frame_idx=1, instances=[pred_inst])
    labels = sio.Labels(
        videos=[video], skeletons=[skel], labeled_frames=[lf_user, lf_pred]
    )

    provider = LabelsProvider(
        labels=labels, exclude_user_labeled=True, only_labeled_frames=False
    )
    assert len(provider._labeled_frames) == 1
    assert provider._labeled_frames[0].frame_idx == 1


def test_labels_provider_only_predicted_frames_keeps_only_predicted():
    """``only_predicted_frames=True`` keeps only frames with predicted instances."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=["a", "b"])
    video = sio.Video(filename="dummy.mp4")
    pred_inst = sio.PredictedInstance.from_numpy(
        points_data=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        skeleton=skel,
        score=0.9,
    )
    lf_empty = sio.LabeledFrame(video=video, frame_idx=0, instances=[])
    lf_pred = sio.LabeledFrame(video=video, frame_idx=1, instances=[pred_inst])
    labels = sio.Labels(
        videos=[video], skeletons=[skel], labeled_frames=[lf_empty, lf_pred]
    )

    provider = LabelsProvider(
        labels=labels, only_predicted_frames=True, only_labeled_frames=False
    )
    assert [lf.frame_idx for lf in provider._labeled_frames] == [1]


def test_labels_provider_only_suggested_frames_yields_unlabeled_suggestions():
    """``only_suggested_frames=True`` yields unlabeled suggestions only."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=["a", "b"])
    video = sio.Video(filename="dummy.mp4")
    user_inst = sio.Instance.from_numpy(
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), skeleton=skel
    )
    lf_with_user = sio.LabeledFrame(video=video, frame_idx=2, instances=[user_inst])
    labels = sio.Labels(
        videos=[video],
        skeletons=[skel],
        labeled_frames=[lf_with_user],
        suggestions=[
            sio.SuggestionFrame(video=video, frame_idx=2),  # already user-labeled
            sio.SuggestionFrame(video=video, frame_idx=5),  # unlabeled
        ],
    )

    provider = LabelsProvider(
        labels=labels, only_suggested_frames=True, only_labeled_frames=False
    )
    # Only the unlabeled suggestion (frame_idx=5) survives.
    assert [lf.frame_idx for lf in provider._labeled_frames] == [5]
    # The yielded LabeledFrame is fresh + empty.
    assert len(provider._labeled_frames[0].instances) == 0


def test_labels_provider_mixed_resolution_batches_by_shape(tmp_path):
    """Frames from videos of different shapes batch without crashing.

    A single ``.slp`` can span videos at different resolutions, so a fixed
    ``batch_size`` chunk can straddle a video boundary and mix ``.image``
    shapes — which used to crash ``np.stack`` with "all input arrays must
    have the same shape". The provider must instead close a batch at the
    resolution boundary, never drop or reorder a frame.
    """
    import imageio.v3 as iio
    import sleap_io as sio

    # Two single-image videos at different resolutions.
    p_small = tmp_path / "small.png"
    p_large = tmp_path / "large.png"
    iio.imwrite(p_small, np.zeros((10, 12, 3), dtype=np.uint8))
    iio.imwrite(p_large, np.zeros((14, 16, 3), dtype=np.uint8))
    v_small = sio.Video.from_filename(str(p_small))
    v_large = sio.Video.from_filename(str(p_large))

    skel = sio.Skeleton(nodes=["a", "b"])
    lf_small = sio.LabeledFrame(video=v_small, frame_idx=0)
    lf_large = sio.LabeledFrame(video=v_large, frame_idx=0)
    labels = sio.Labels(
        videos=[v_small, v_large],
        skeletons=[skel],
        labeled_frames=[lf_small, lf_large],
    )

    # batch_size=4 would put both frames in one chunk; the shape boundary
    # must split them into two single-frame batches instead of crashing.
    provider = LabelsProvider(labels=labels, batch_size=4, only_labeled_frames=False)
    batches = list(provider)

    # Two distinct shapes → two batches, each internally uniform.
    assert len(batches) == 2
    shapes = sorted(tuple(b.images.shape[1:]) for b in batches)
    assert shapes == [(10, 12, 1), (14, 16, 1)]
    # Every frame yielded exactly once, none reordered or dropped.
    assert sum(b.images.shape[0] for b in batches) == 2


# ─────────────────────────────────────────────────────────────────────────
# IncrementalLabelsWriter
# ─────────────────────────────────────────────────────────────────────────


def test_incremental_writer_atomic_rename(tmp_path):
    """``close()`` renames ``<path>.tmp`` → final path; no half-written file."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=[sio.Node(name=f"n{i}") for i in range(2)])
    out_path = tmp_path / "out.slp"
    writer = IncrementalLabelsWriter(
        path=str(out_path), skeleton=skel, write_interval=2
    )

    # Synthetic Outputs: 2 frames, 1 instance, 2 nodes.
    o1 = Outputs(
        pred_keypoints=torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]),
        pred_peak_values=torch.tensor([[[0.9, 0.9]]]),
        frame_indices=torch.tensor([0]),
        video_indices=torch.tensor([0]),
    )
    o2 = Outputs(
        pred_keypoints=torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]),
        pred_peak_values=torch.tensor([[[0.9, 0.9]]]),
        frame_indices=torch.tensor([1]),
        video_indices=torch.tensor([0]),
    )

    with writer:
        writer.write(o1)
        writer.write(o2)

    assert out_path.exists()
    # Tmp file should NOT exist after close (atomic rename done).
    assert not writer.tmp_path.exists()
    # Loadable .slp.
    labels = sio.load_slp(str(out_path))
    assert len(labels.labeled_frames) == 2


def test_incremental_writer_close_is_idempotent(tmp_path):
    """Calling ``close()`` twice doesn't raise or rewrite."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=[sio.Node(name="n0")])
    writer = IncrementalLabelsWriter(path=str(tmp_path / "out.slp"), skeleton=skel)
    writer.write(
        Outputs(
            pred_keypoints=torch.tensor([[[[0.0, 0.0]]]]),
            pred_peak_values=torch.tensor([[[1.0]]]),
            frame_indices=torch.tensor([0]),
            video_indices=torch.tensor([0]),
        )
    )
    writer.close()
    writer.close()  # second call should be a no-op


def test_incremental_writer_write_after_close_raises(tmp_path):
    """Writing to a closed writer raises ``RuntimeError``."""
    import sleap_io as sio

    skel = sio.Skeleton(nodes=[sio.Node(name="n0")])
    writer = IncrementalLabelsWriter(path=str(tmp_path / "out.slp"), skeleton=skel)
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(Outputs(pred_keypoints=torch.zeros(1, 1, 1, 2)))


# ─────────────────────────────────────────────────────────────────────────
# End-to-end: Predictor.predict_to_file
# ─────────────────────────────────────────────────────────────────────────


class _StubLayer:
    """Layer-shaped stub: returns a constant ``Outputs`` per batch."""

    def predict(self, image, **kwargs) -> Outputs:
        b = image.shape[0]
        return Outputs(
            pred_keypoints=torch.zeros(b, 1, 2, 2),
            pred_peak_values=torch.ones(b, 1, 2),
            instance_scores=torch.ones(b, 1) * 0.9,
        )


def test_predictor_predict_to_file_end_to_end(tmp_path):
    """``predict_to_file`` writes a saveable ``.slp`` from a stub layer."""
    import sleap_io as sio

    from sleap_nn.inference.providers import NumpyProvider

    skel = sio.Skeleton(nodes=[sio.Node(name="n0"), sio.Node(name="n1")])
    images = np.zeros((6, 1, 8, 8), dtype=np.float32)
    provider = NumpyProvider(images=images, batch_size=2)

    predictor = Predictor(layer=_StubLayer())
    out_path = tmp_path / "predictions.slp"
    written = predictor.predict_to_file(provider, path=str(out_path), skeleton=skel)
    assert Path(written).exists()
    labels = sio.load_slp(str(out_path))
    # 6 frames in / 2 per batch / 1 instance per frame → 6 LabeledFrames
    assert len(labels.labeled_frames) == 6
