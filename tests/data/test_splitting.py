"""Tests for the group-aware train/val splitter (no-leakage guarantee + edge cases)."""

from types import SimpleNamespace

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.data.splitting import (
    split_labels_list_train_val,
    split_labels_train_val,
)

_SKEL = sio.Skeleton(nodes=["a"], name="s")


def _make_labels(n_videos=1, n_frames=6, n_tracks=3):
    """Synthetic Labels: ``n_videos`` videos x ``n_frames`` frames x ``n_tracks`` tracked instances."""
    tracks = [sio.Track(name=f"t{i}") for i in range(n_tracks)]
    videos = [sio.Video.from_filename(f"vid{v}.mp4") for v in range(n_videos)]
    lfs = []
    for video in videos:
        for fi in range(n_frames):
            insts = [
                sio.Instance.from_numpy(
                    np.array([[float(i), float(fi)]]), skeleton=_SKEL, track=tracks[i]
                )
                for i in range(n_tracks)
            ]
            lfs.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=insts))
    return sio.Labels(
        videos=videos, labeled_frames=lfs, skeletons=[_SKEL], tracks=tracks
    )


def _track_names(labels):
    return {
        inst.track.name
        for lf in labels
        for inst in lf.instances
        if inst.track is not None
    }


def _video_idxs(labels):
    return {labels.videos.index(lf.video) for lf in labels}


class TestNoLeakage:
    def test_identity_split_is_disjoint(self):
        labels = _make_labels(n_videos=1, n_frames=8, n_tracks=4)
        train, val = split_labels_train_val(
            labels, split_by="identity", n_folds=4, fold=0, seed=0
        )
        assert _track_names(train).isdisjoint(_track_names(val))
        assert _track_names(val)  # val is non-empty

    def test_video_split_is_disjoint(self):
        labels = _make_labels(n_videos=4, n_frames=3, n_tracks=2)
        train, val = split_labels_train_val(
            labels, split_by="video", n_folds=4, fold=0, seed=0
        )
        assert _video_idxs(train).isdisjoint(_video_idxs(val))
        assert _video_idxs(val)

    def test_frame_split_keeps_frames_whole(self):
        labels = _make_labels(n_videos=1, n_frames=10, n_tracks=3)
        train, val = split_labels_train_val(
            labels, split_by="frame", n_folds=5, fold=0, seed=0
        )
        train_frames = {(lf.video, lf.frame_idx) for lf in train}
        val_frames = {(lf.video, lf.frame_idx) for lf in val}
        assert train_frames.isdisjoint(val_frames)


class TestGuards:
    def test_single_video_raises_for_video_split(self):
        labels = _make_labels(n_videos=1, n_frames=6, n_tracks=2)
        with pytest.raises(ValueError, match="video"):
            split_labels_train_val(labels, split_by="video", n_folds=5, fold=0, seed=0)

    def test_single_identity_raises_for_identity_split(self):
        labels = _make_labels(n_videos=1, n_frames=6, n_tracks=1)
        with pytest.raises(ValueError, match="track names"):
            split_labels_train_val(
                labels, split_by="identity", n_folds=5, fold=0, seed=0
            )


class TestNegativeAndEmptyHandling:
    def test_negative_frames_are_carried_to_train(self):
        labels = _make_labels(n_videos=1, n_frames=6, n_tracks=3)
        # Append two detection-less (negative) frames.
        for fi in range(6, 8):
            labels.append(
                sio.LabeledFrame(
                    video=labels.videos[0], frame_idx=fi, instances=[], is_negative=True
                )
            )
        train, val = split_labels_train_val(
            labels, split_by="frame", n_folds=5, fold=0, seed=0
        )
        train_neg = sum(1 for lf in train if not lf.instances)
        val_neg = sum(1 for lf in val if not lf.instances)
        assert train_neg == 2  # both negatives on the train side
        assert val_neg == 0  # none dropped to val

    def test_all_negative_file_does_not_abort(self):
        labels = sio.Labels(
            videos=[sio.Video.from_filename("v.mp4")],
            labeled_frames=[],
            skeletons=[_SKEL],
        )
        for fi in range(3):
            labels.append(
                sio.LabeledFrame(
                    video=labels.videos[0], frame_idx=fi, instances=[], is_negative=True
                )
            )
        train, val = split_labels_train_val(
            labels, split_by="frame", n_folds=5, fold=0, seed=0
        )
        assert len(train) == 3 and len(val) == 0  # no abort; negatives -> train


class TestMaskOnly:
    def test_mask_only_round_trip(self):
        # Mask-only labels (no skeleton/instances): the split must read detections from
        # lf.masks and re-attach them on .masks.
        video = sio.Video.from_filename("v.mp4")
        track = sio.Track(name="t0")
        blob = np.zeros((64, 64), dtype=bool)
        blob[20:40, 20:40] = True
        lfs = []
        for fi in range(4):
            mask = sio.UserSegmentationMask.from_numpy(blob)
            mask.track = track
            lfs.append(sio.LabeledFrame(video=video, frame_idx=fi, masks=[mask]))
        labels = sio.Labels(videos=[video], labeled_frames=lfs)

        train, val = split_labels_train_val(
            labels, split_by="frame", n_folds=2, fold=0, seed=0
        )
        assert sum(len(lf.masks) for lf in [*train, *val]) == 4
        # Frames stay whole and detections come back as masks, not instances.
        assert all(not lf.instances for lf in [*train, *val])


def test_list_alignment_and_disjointness():
    a = _make_labels(n_videos=1, n_frames=8, n_tracks=4)
    b = _make_labels(n_videos=1, n_frames=6, n_tracks=4)
    cfg = SimpleNamespace(split_by="identity", n_folds=4, fold=0, seed=0)
    train_list, val_list = split_labels_list_train_val([a, b], cfg)
    assert len(train_list) == len(val_list) == 2
    for tr, vl in zip(train_list, val_list):
        assert _track_names(tr).isdisjoint(_track_names(vl))
