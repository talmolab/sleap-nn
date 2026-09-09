"""`split_by="identity"` must group by the animal, not by the tracklet.

An animal recognized across videos carries one ``sio.Identity`` and a *different*
``sio.Track`` per video. Grouping the split on track names therefore treats one
animal as several groups, and a group-aware split may put those groups on
opposite sides — leaking the identity into both train and val, which is the one
thing this mode exists to prevent. The training objective already keys positives
on ``Identity`` first (``custom_datasets._global_identity_label``), so the
splitter has to agree with it.
"""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.data.splitting import (
    split_labels_list_train_val,
    split_labels_train_val,
)

_SKEL = sio.Skeleton(nodes=["a"], name="s")


def _cross_video_labels(n_frames=8, n_identities=4, n_videos=2):
    """Each identity appears in every video under a DIFFERENT track name.

    Identity ``id{i}`` is ``track_v{v}_i{i}``: track names are unique per video,
    identities are shared across videos.
    """
    identities = [sio.Identity(name=f"id{i}") for i in range(n_identities)]
    videos = [sio.Video.from_filename(f"vid{v}.mp4") for v in range(n_videos)]
    tracks = {
        (v, i): sio.Track(name=f"track_v{v}_i{i}")
        for v in range(n_videos)
        for i in range(n_identities)
    }
    frames = []
    for v, video in enumerate(videos):
        for fi in range(n_frames):
            frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=fi,
                    instances=[
                        sio.Instance.from_numpy(
                            np.array([[float(i), float(fi)]]),
                            skeleton=_SKEL,
                            track=tracks[(v, i)],
                            identity=identities[i],
                        )
                        for i in range(n_identities)
                    ],
                )
            )
    return sio.Labels(
        videos=videos,
        labeled_frames=frames,
        skeletons=[_SKEL],
        tracks=list(tracks.values()),
    )


def _identity_names(labels):
    return {
        inst.identity.name
        for lf in labels
        for inst in lf.instances
        if inst.identity is not None
    }


def _track_names(labels):
    return {
        inst.track.name
        for lf in labels
        for inst in lf.instances
        if inst.track is not None
    }


def test_identity_split_is_disjoint_by_identity_not_by_track():
    """No identity may appear on both sides, even with per-video track names."""
    labels = _cross_video_labels()
    train, val = split_labels_train_val(
        labels, split_by="identity", n_folds=4, fold=0, seed=0
    )

    assert _identity_names(val), "val side is empty"
    assert _identity_names(train).isdisjoint(_identity_names(val))


@pytest.mark.parametrize("fold", [0, 1, 2, 3])
def test_identity_split_is_disjoint_across_folds(fold):
    """Every fold, not just fold 0."""
    labels = _cross_video_labels()
    train, val = split_labels_train_val(
        labels, split_by="identity", n_folds=4, fold=fold, seed=0
    )

    assert _identity_names(train).isdisjoint(_identity_names(val))


def test_identity_split_keeps_all_of_an_animals_tracklets_together():
    """An identity's per-video tracks travel together, on one side."""
    labels = _cross_video_labels()
    train, val = split_labels_train_val(
        labels, split_by="identity", n_folds=4, fold=0, seed=0
    )

    val_ids = _identity_names(val)
    # Each held-out identity contributes one track per video (2 videos here), and
    # none of those track names may appear in train.
    expected_val_tracks = {
        f"track_v{v}_i{name[2:]}" for name in val_ids for v in range(2)
    }
    assert expected_val_tracks <= _track_names(val)
    assert _track_names(train).isdisjoint(expected_val_tracks)


def test_split_without_identities_still_groups_by_track():
    """No `Identity` anywhere: the track name is the identity, as before."""
    labels = _cross_video_labels()
    for lf in labels:
        for inst in lf.instances:
            inst.identity = None

    train, val = split_labels_train_val(
        labels, split_by="identity", n_folds=4, fold=0, seed=0
    )

    assert _track_names(val)
    assert _track_names(train).isdisjoint(_track_names(val))


def test_multi_file_identity_split_warns_about_cross_file_leakage(caplog):
    """Files are split independently, so cross-file identities can still leak."""
    import logging

    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from loguru import logger as loguru_logger

    messages = []
    handler_id = loguru_logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        split_labels_list_train_val(
            [_cross_video_labels(), _cross_video_labels()],
            type(
                "Cfg",
                (),
                {"split_by": "identity", "n_folds": 4, "fold": 0, "seed": 0},
            )(),
        )
    finally:
        loguru_logger.remove(handler_id)

    joined = "".join(str(m) for m in messages)
    assert "each file is split independently" in joined
    assert "both sides of the split" in joined


def test_single_file_identity_split_does_not_warn():
    """The warning is specific to the multi-file case."""
    from loguru import logger as loguru_logger

    messages = []
    handler_id = loguru_logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        split_labels_list_train_val(
            [_cross_video_labels()],
            type(
                "Cfg",
                (),
                {"split_by": "identity", "n_folds": 4, "fold": 0, "seed": 0},
            )(),
        )
    finally:
        loguru_logger.remove(handler_id)

    assert "each file is split independently" not in "".join(str(m) for m in messages)
