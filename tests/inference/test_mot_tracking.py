"""`apply_tracking` with the modern MOT trackers (`TrackerConfig.tracker != "classic"`).

Covers the integration seam only -- the trackers themselves are tested under
`tests/tracking/mot/`: config validation and properties, carrier resolution, the
per-video loop (naming, never dropping, user-instance frames), mask inheritance
AFTER `finalize`, closed-set linking via `max_tracks`, the appearance-vector rules,
the `provenance["tracking"]` record, and that the classic default is untouched.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import sleap_io as sio
from _pytest.logging import LogCaptureFixture
from loguru import logger

from sleap_nn.inference.provenance import carry_tracking_record
from sleap_nn.inference.tracking import (
    MOT_ONLY_OPTIONS,
    TRACKER_NAMES,
    TrackerConfig,
    apply_tracking,
    check_tracking_plan,
    resolve_mot_params,
    tracker_config_params,
    validate_mot_config,
)
from sleap_nn.tracking.mot import APPEARANCE_TRACKERS
from sleap_nn.tracking.mot.params import MOT_TRACKER_NAMES


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru records into pytest's ``caplog`` (project convention)."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def skeleton():
    return sio.Skeleton(nodes=["head", "tail"])


def _pose(skeleton, x, y, score=0.9, vec=None):
    """A predicted 2-node pose at ``(x, y)`` with confident point scores."""
    inst = sio.PredictedInstance.from_numpy(
        points_data=np.array([[x, y], [x + 30.0, y + 10.0]], dtype=np.float32),
        skeleton=skeleton,
        score=score,
        point_scores=np.full(2, score, dtype=np.float32),
    )
    if vec is not None:
        inst.identity_embedding = sio.Embedding(np.asarray(vec, dtype=np.float32))
    return inst


def _make_labels(skeleton, videos, n_frames=12, xs=(0.0, 200.0), vecs=None, gap=()):
    """Two animals walking right, one video after another (frames out of order).

    ``gap``: frame indices on which the FIRST animal is missing.
    """
    lfs = []
    for video in videos:
        for i in range(n_frames):
            insts = []
            for k, x0 in enumerate(xs):
                if k == 0 and i in gap:
                    continue
                vec = None if vecs is None else vecs[k]
                insts.append(_pose(skeleton, x0 + 2.0 * i, 50.0, vec=vec))
            lfs.append(sio.LabeledFrame(video=video, frame_idx=i, instances=insts))
    # Submission order must not matter: apply_tracking sorts (video, frame_idx).
    return sio.Labels(
        videos=list(videos), skeletons=[skeleton], labeled_frames=lfs[::-1]
    )


def _names(lf, attr="instances"):
    return [None if o.track is None else o.track.name for o in getattr(lf, attr)]


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


# ── TrackerConfig ──────────────────────────────────────────────────────────


def test_tracker_names_pin_the_mot_registry():
    assert TRACKER_NAMES == ("classic",) + MOT_TRACKER_NAMES


def test_classic_default_unchanged():
    """The default config is the classic tracker, with classic explicitness."""
    cfg = TrackerConfig()
    assert cfg.tracker == "classic" and not cfg.is_mot
    assert cfg.scoring_method_explicit and cfg.features_explicit
    assert cfg.candidates_method_explicit
    assert all(getattr(cfg, name) is None for name in MOT_ONLY_OPTIONS)
    # Classic provenance carries none of the MOT options.
    params = tracker_config_params(cfg)
    assert "tracker" not in params and "link_tracklets" not in params
    assert not set(MOT_ONLY_OPTIONS) & set(params)
    assert "link_tracklets" in tracker_config_params(TrackerConfig(link_tracklets=True))


def test_mot_config_explicitness_defaults():
    """A MOT config built directly gets its preset similarity / carrier unless the
    caller changed `scoring_method` / `features` from their defaults."""
    cfg = TrackerConfig(tracker="sort")
    assert not cfg.scoring_method_explicit and not cfg.features_explicit
    assert not cfg.candidates_method_explicit
    assert resolve_mot_params(cfg).similarity == "auto"
    assert TrackerConfig(tracker="sort", scoring_method="iou").scoring_method_explicit
    assert TrackerConfig(tracker="sort", features="masks").features_explicit


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(tracker="nope"),
        dict(tracker="sort", track_high_threshold=1.5),
        dict(tracker="sort", min_match_similarity=-0.1),
        dict(tracker="sort", min_hits=0),
        dict(tracker="sort", track_buffer=-1),
        dict(tracker="botsort", appearance_threshold=2.0),
        dict(tracker="botsort", appearance_threshold="high"),
        dict(tracker="sort", mot_params={"bogus": 1}),
    ],
)
def test_tracker_config_validators(kwargs):
    with pytest.raises(ValueError):
        TrackerConfig(**kwargs)


def test_appearance_threshold_converter():
    assert TrackerConfig(appearance_threshold=None).appearance_threshold is None
    assert TrackerConfig(appearance_threshold="AUTO").appearance_threshold == "auto"
    assert TrackerConfig(appearance_threshold="0.4").appearance_threshold == 0.4


def test_appearance_properties():
    sort, botsort = TrackerConfig(tracker="sort"), TrackerConfig(tracker="botsort")
    assert not sort.uses_appearance and not sort.requires_vectors
    assert TrackerConfig(tracker="sort", link_tracklets=True).uses_appearance
    assert TrackerConfig(tracker="sort", max_tracks=2).uses_appearance
    for name in APPEARANCE_TRACKERS:
        cfg = TrackerConfig(tracker=name)
        assert (
            cfg.uses_appearance and cfg.requires_vectors and cfg.follows_vector_carrier
        )
    assert botsort.is_mot and botsort.follows_vector_carrier
    assert not TrackerConfig(tracker="botsort", features="masks").follows_vector_carrier
    # Classic keeps its own rules.
    emb = TrackerConfig(features="embeddings")
    assert emb.uses_appearance and emb.follows_vector_carrier and emb.requires_vectors
    blend = TrackerConfig(appearance_weight=0.3)
    assert blend.uses_appearance and not blend.follows_vector_carrier
    assert not TrackerConfig(link_tracklets=True).uses_appearance


def test_resolve_mot_params_maps_shared_options():
    p = resolve_mot_params(
        TrackerConfig(
            tracker="sort",
            scoring_method="euclidean_dist",
            oks_stddev=0.05,
            max_tracks=3,
            track_buffer=7,
            min_hits=2,
            mot_params={"preset": "paper", "gallery_size": 12},
        )
    )
    assert (p.similarity, p.oks_stddev, p.max_tracks) == ("centroid", 0.05, 3)
    assert (p.track_buffer, p.min_hits, p.preset, p.gallery_size) == (7, 2, "paper", 12)
    assert p.link  # a known animal count implies closed-set linking
    # An unset oks_stddev takes the MOT default, not the classic 0.025.
    assert resolve_mot_params(TrackerConfig(tracker="sort")).oks_stddev == 0.1
    with pytest.raises(ValueError, match="both"):
        resolve_mot_params(
            TrackerConfig(
                tracker="sort", track_buffer=5, mot_params={"track_buffer": 9}
            )
        )


# ── Option-family validation ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(window_size=10),
        dict(min_new_track_points=1),
        dict(min_match_points=1),
        dict(scoring_reduction="max"),
        dict(robust_best_instance=0.5),
        dict(appearance_weight=0.3),
        dict(euclidean_scale=5.0),
        dict(track_matching_method="greedy"),
        dict(use_flow=True),
        dict(use_kalman=True),
        dict(kf_init_frame_count=3),
        dict(tracking_target_instance_count=2),
        dict(tracking_pre_cull_to_target=2, tracking_target_instance_count=2),
        dict(tracking_clean_instance_count=2),
        dict(post_connect_single_breaks=True, tracking_target_instance_count=2),
        dict(candidates_method="local_queues"),
        dict(candidates_method="fixed_window", candidates_method_explicit=True),
        dict(features="embeddings"),
        dict(features="centroids"),
        dict(scoring_method="mask_iou"),
        dict(scoring_method="cosine_sim"),
    ],
)
def test_mot_rejects_classic_only_options(skeleton, kwargs):
    labels = _make_labels(skeleton, [sio.Video(filename="a.mp4")])
    with pytest.raises(ValueError):
        apply_tracking(labels, TrackerConfig(tracker="sort", **kwargs))


@pytest.mark.parametrize("name", MOT_ONLY_OPTIONS)
def test_classic_rejects_mot_only_options(skeleton, name):
    value = {"appearance_threshold": 0.5, "mot_params": {"gallery_size": 3}}.get(
        name, 1
    )
    labels = _make_labels(skeleton, [sio.Video(filename="a.mp4")])
    with pytest.raises(ValueError, match="MOT trackers"):
        apply_tracking(labels, TrackerConfig(**{name: value}))


# ── The MOT loop ───────────────────────────────────────────────────────────


def test_sort_tracks_every_detection_and_names_continue_across_videos(skeleton):
    videos = [sio.Video(filename="a.mp4"), sio.Video(filename="b.mp4")]
    labels = _make_labels(skeleton, videos)
    n_in = sum(len(lf.instances) for lf in labels.labeled_frames)
    progress = []
    out = apply_tracking(
        labels,
        TrackerConfig(tracker="sort"),
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert out is labels  # in place
    assert sum(len(lf.instances) for lf in out.labeled_frames) == n_in  # no drop
    assert [(lf.video, lf.frame_idx) for lf in out.labeled_frames] == [
        (v, i) for v in videos for i in range(12)
    ]
    assert progress[-1] == (24, 24)
    # SORT confirms after min_hits and labels the earlier frames retroactively.
    for lf in out.labeled_frames:
        expected = ["track_0", "track_1"] if lf.video is videos[0] else None
        if expected is None:
            expected = ["track_2", "track_3"]
        assert _names(lf) == expected
    assert sorted(t.name for t in out.tracks) == [f"track_{i}" for i in range(4)]

    record = out.provenance["tracking"]
    assert record["tracker"] == "sort" and record["carrier"] == "pose"
    assert record["params"]["track_buffer"] == 30
    assert [v["video"] for v in record["videos"]] == [0, 1]
    assert "calibration" in record["videos"][0]
    json.dumps(record, allow_nan=False)  # python types only


def test_sort_clears_stale_tracks(skeleton):
    labels = _make_labels(skeleton, [sio.Video(filename="a.mp4")], n_frames=2)
    stale = sio.Track("stale")
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.track = stale
    out = apply_tracking(labels, TrackerConfig(tracker="sort"))
    # Two frames never confirm a SORT track (min_hits=3): explicitly untracked.
    assert all(i.track is None for lf in out.labeled_frames for i in lf.instances)
    assert out.tracks == []


def test_mot_user_instance_frames_pass_predictions_through_untracked(skeleton):
    video = sio.Video(filename="a.mp4")
    labels = _make_labels(skeleton, [video], n_frames=6)
    stale = sio.Track("stale")
    lf = next(lf for lf in labels.labeled_frames if lf.frame_idx == 5)
    for inst in lf.predicted_instances:
        inst.track = stale
        inst.tracking_score = 0.7
    user = [
        sio.Instance.from_numpy(
            np.array([[x + 10.0, 50.0], [x + 40.0, 60.0]]), skeleton=skeleton
        )
        for x in (0.0, 200.0)
    ]
    lf.instances = list(lf.instances) + user

    out = apply_tracking(labels, TrackerConfig(tracker="sort", min_hits=1))
    lf = next(lf for lf in out.labeled_frames if lf.frame_idx == 5)
    assert all(i.track is None for i in lf.predicted_instances)
    assert all(i.tracking_score == 0.0 for i in lf.predicted_instances)
    assert len(lf.user_instances) == 2  # the tracked ones, all kept
    assert stale not in out.tracks


def test_mask_carrier_poses_inherit_after_finalize(skeleton):
    """Masks confirmed at `min_hits` are labeled retroactively by `finalize`-time
    bookkeeping; the linked poses on those early frames must inherit the track
    too (integration review B1)."""
    video = sio.Video(filename="a.mp4")
    lfs = []
    for i in range(8):
        insts, masks = [], []
        for cx in (20, 60):
            inst = _pose(skeleton, cx + i - 5.0, 30.0)
            mask = sio.PredictedSegmentationMask.from_numpy(
                _disk(80, 100, 30, cx + i, 8), score=0.9
            )
            mask.instance = inst
            insts.append(inst)
            masks.append(mask)
        lfs.append(
            sio.LabeledFrame(video=video, frame_idx=i, instances=insts, masks=masks)
        )
    labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)

    out = apply_tracking(
        labels, TrackerConfig(tracker="sort", features="masks", min_hits=3)
    )
    assert out.provenance["tracking"]["carrier"] == "mask"
    for lf in out.labeled_frames:
        mask_names = _names(lf, "masks")
        assert None not in mask_names, lf.frame_idx
        # Frames 0-2 precede the confirmation: inherited only after finalize.
        assert _names(lf) == mask_names, lf.frame_idx


def test_mask_features_without_masks_raises(skeleton):
    labels = _make_labels(skeleton, [sio.Video(filename="a.mp4")])
    with pytest.raises(ValueError, match="no frame holds a mask"):
        apply_tracking(labels, TrackerConfig(tracker="sort", features="masks"))


def test_max_tracks_caps_identities_by_closed_set_linking(skeleton):
    """A 10-frame absence (> track_buffer) fragments the first animal into two
    tracks; `max_tracks=2` (closed set) brings the video back to two identities."""
    video = sio.Video(filename="a.mp4")
    gap = set(range(10, 20))
    labels = _make_labels(skeleton, [video], n_frames=30, gap=gap)
    open_set = apply_tracking(
        labels.copy(), TrackerConfig(tracker="sort", track_buffer=5)
    )
    assert len(open_set.tracks) == 3

    out = apply_tracking(
        labels, TrackerConfig(tracker="sort", track_buffer=5, max_tracks=2)
    )
    assert len(out.tracks) <= 2
    first = {
        lf.instances[0].track.name
        for lf in out.labeled_frames
        if lf.frame_idx not in gap and lf.instances[0].track is not None
    }
    assert len(first) == 1  # the first animal kept one identity across its absence
    assert "linking" in out.provenance["tracking"]["videos"][0]


def test_classic_link_tracklets_runs_and_records(skeleton):
    video = sio.Video(filename="a.mp4")
    labels = _make_labels(skeleton, [video], n_frames=10)
    out = apply_tracking(labels, TrackerConfig(link_tracklets=True))
    assert len(out.tracks) == 2
    assert all(i.track is not None for lf in out.labeled_frames for i in lf.instances)
    record = out.provenance["tracking"]
    assert record["tracker"] == "classic" and record["link_tracklets"] is True
    assert {"calibration", "linking"} <= set(record["videos"][0])
    json.dumps(record, allow_nan=False)


def test_classic_default_records_no_tracking_provenance(skeleton):
    labels = _make_labels(skeleton, [sio.Video(filename="a.mp4")])
    labels.provenance["kept"] = 1
    out = apply_tracking(labels, TrackerConfig())
    assert out.provenance == {"kept": 1}
    assert len(out.tracks) == 2


# ── Appearance vectors ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["deepsort", "strongsort", "botsort", "deepocsort"])
def test_appearance_tracker_without_vectors_raises(skeleton, name):
    labels = _make_labels(skeleton, [sio.Video(filename="a.mp4")])
    with pytest.raises(ValueError, match="associates by appearance"):
        apply_tracking(labels, TrackerConfig(tracker=name))


def _pose_mask_labels_vectors_on_masks(skeleton):
    video = sio.Video(filename="a.mp4")
    lfs = []
    for i in range(4):
        insts, masks = [], []
        for k, cx in enumerate((20, 60)):
            inst = _pose(skeleton, cx + i - 5.0, 30.0)
            mask = sio.PredictedSegmentationMask.from_numpy(
                _disk(80, 100, 30, cx + i, 8), score=0.9
            )
            mask.instance = inst
            mask.identity_embedding = sio.Embedding(np.eye(4, dtype=np.float32)[k])
            insts.append(inst)
            masks.append(mask)
        lfs.append(
            sio.LabeledFrame(video=video, frame_idx=i, instances=insts, masks=masks)
        )
    return sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=lfs)


def test_vector_required_tracker_refuses_vectors_on_the_other_carrier(skeleton):
    labels = _pose_mask_labels_vectors_on_masks(skeleton)
    cfg = TrackerConfig(
        tracker="deepsort", features="keypoints", features_explicit=True
    )
    with pytest.raises(ValueError, match="vectors are on the mask carrier"):
        apply_tracking(labels, cfg)


def test_check_tracking_plan_mot(skeleton):
    """The embedding routes' early check applies the MOT carrier rules too."""
    explicit = TrackerConfig(
        tracker="deepsort", features="keypoints", features_explicit=True
    )
    with pytest.raises(ValueError, match="mask carrier"):
        check_tracking_plan(
            explicit,
            has_masks=True,
            has_predicted_instances=True,
            vector_carrier="mask",
        )
    # Not explicit: deepsort follows the vectors to the masks.
    check_tracking_plan(
        TrackerConfig(tracker="deepsort"),
        has_masks=True,
        has_predicted_instances=True,
        vector_carrier="mask",
    )
    with pytest.raises(ValueError, match="classic tracker"):
        check_tracking_plan(
            TrackerConfig(tracker="sort", use_flow=True),
            has_masks=False,
            has_predicted_instances=True,
            vector_carrier=None,
        )


def test_describe_association_mot():
    from sleap_nn.inference.embedding import _describe_association

    assert "botsort by geometry and appearance" in _describe_association(
        TrackerConfig(tracker="botsort")
    )
    assert "not used" in _describe_association(TrackerConfig(tracker="sort"))
    assert "identity linking" in _describe_association(
        TrackerConfig(tracker="sort", link_tracklets=True)
    )


# ── Provenance ─────────────────────────────────────────────────────────────


def test_carry_tracking_record():
    record = {"tracker": "sort"}
    tracked = sio.Labels()
    tracked.provenance = {"tracking": record, "filename": "in.slp"}
    built = {"input_provenance": dict(tracked.provenance), "x": 1}

    out = carry_tracking_record(dict(built), tracked, TrackerConfig(tracker="sort"))
    assert out["tracking"] is record
    # Tracking is in place, so the copied input provenance held this run's record.
    assert out["input_provenance"] == {"filename": "in.slp"}

    # A classic run does not claim a (stale) record the input carried.
    assert "tracking" not in carry_tracking_record(
        dict(built), tracked, TrackerConfig()
    )
    assert carry_tracking_record({"x": 1}, tracked, None) == {"x": 1}


# ── Every MOT tracker through apply_tracking (smoke) ───────────────────────


@pytest.mark.parametrize("name", MOT_TRACKER_NAMES)
def test_every_mot_tracker_smoke(skeleton, name):
    """Never drops, names continue across videos, provenance is JSON-safe."""
    videos = [sio.Video(filename="a.mp4"), sio.Video(filename="b.mp4")]
    vecs = None
    if name in APPEARANCE_TRACKERS:
        vecs = [np.eye(8, dtype=np.float32)[0], np.eye(8, dtype=np.float32)[1]]
    labels = _make_labels(skeleton, videos, n_frames=15, vecs=vecs)
    n_in = sum(len(lf.instances) for lf in labels.labeled_frames)

    out = apply_tracking(labels, TrackerConfig(tracker=name))
    assert sum(len(lf.instances) for lf in out.labeled_frames) == n_in
    names_by_video = [
        {
            i.track.name
            for lf in out.labeled_frames
            if lf.video is v
            for i in lf.instances
            if i.track
        }
        for v in videos
    ]
    assert all(len(n) == 2 for n in names_by_video)
    assert not names_by_video[0] & names_by_video[1]
    json.dumps(out.provenance["tracking"], allow_nan=False)


def test_appearance_threshold_needs_an_appearance_tracker():
    with pytest.raises(ValueError, match="only applies to the appearance trackers"):
        validate_mot_config(TrackerConfig(tracker="sort", appearance_threshold=0.5))
    validate_mot_config(TrackerConfig(tracker="botsort", appearance_threshold=0.5))
