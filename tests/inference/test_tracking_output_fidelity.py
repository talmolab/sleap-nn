"""`apply_tracking` output fidelity, carrier resolution, and per-video state.

Every test here goes through the real entry point -- :func:`apply_tracking`, or
``sleap-nn predict ... -t`` via ``CliRunner`` -- and pins a defect found in the
2026-09-24 post-merge review of the embedding / re-ID stack:

* tracking rebuilt every frame and the ``Labels`` from scratch, deleting the
  un-tracked carrier (masks in pose mode), ROIs, centroids, and suggestions;
* ``appearance_weight`` was a silent no-op when the vectors rode on the masks;
* one tracker spanned every video, so video 2 continued video 1's tracks;
* ``cosine_sim`` was accepted as the GEOMETRIC side of a blend;
* ``features="embeddings"`` defaulted to ``fixed_window``, which forgets an
  occluded animal after ``window_size`` frames -- the case the mode exists for.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pytest
import sleap_io as sio
from click.testing import CliRunner
from loguru import logger

from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.segmentation_convert import build_predicted_roi
from sleap_nn.inference.tracking import (
    MASK_CARRIER,
    POSE_CARRIER,
    TrackerConfig,
    apply_tracking,
    embedding_carriers,
)

SKEL = sio.Skeleton(nodes=["a", "b"])


@pytest.fixture
def log_text():
    """Collect loguru output (loguru does not reach pytest's caplog on its own)."""
    lines: list = []
    handler_id = logger.add(lambda m: lines.append(str(m)), level="INFO")
    yield lambda: "".join(lines)
    logger.remove(handler_id)


def _pinst(x, y, emb=None):
    inst = sio.PredictedInstance.from_numpy(
        np.array([[x, y], [x + 5.0, y]], dtype=np.float32),
        skeleton=SKEL,
        point_scores=np.ones(2),
        score=0.9,
    )
    if emb is not None:
        inst.identity_embedding = sio.Embedding(np.asarray(emb, np.float32))
    return inst


def _pmask(x, y, emb=None, shape=(100, 100), r=4):
    grid = np.zeros(shape, bool)
    grid[max(0, int(y) - r) : int(y) + r, max(0, int(x) - r) : int(x) + r] = True
    m = sio.PredictedSegmentationMask.from_numpy(grid, score=0.9)
    if emb is not None:
        m.identity_embedding = sio.Embedding(np.asarray(emb, np.float32))
    return m


def _emb_cfg(**kw):
    """Embedding mode as the CLI builds it: features explicit, the rest left unset."""
    kw.setdefault("features", "embeddings")
    kw.setdefault("scoring_method_explicit", False)
    kw.setdefault("candidates_method_explicit", False)
    return TrackerConfig(**kw)


def _pose_mask_labels(n_frames=4, video=None):
    """Two animals as poses, each with a linked mask, plus a ROI, a centroid, and a
    suggestion -- a top-down-segmentation-shaped file with every annotation type
    tracking must pass through. Vectors on the poses."""
    video = video or sio.Video(filename="fake.mp4")
    lfs = []
    for t in range(n_frames):
        insts = [_pinst(10 + t, 10, emb=[1, 0, 0]), _pinst(60 + t, 60, emb=[0, 1, 0])]
        masks = [_pmask(10 + t, 10), _pmask(60 + t, 60)]
        for m, i in zip(masks, insts):
            m.instance = i
        lfs.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=t,
                instances=insts,
                masks=masks,
                rois=[build_predicted_roi(masks[0], 1.0, 0.01)],
                centroids=[sio.PredictedCentroid(x=30.0 + t, y=30.0)],
            )
        )
    labels = sio.Labels(lfs, videos=[video], skeletons=[SKEL])
    labels.suggestions.append(sio.SuggestionFrame(video=video, frame_idx=2))
    labels.provenance["source"] = "unit-test"
    return labels


def _swap_labels(vectors_on="masks"):
    """Two animals SWAP positions at t=2: geometry says "follow the position",
    appearance says "follow the animal". Poses with linked masks; the vectors on
    one carrier -- the embedding model puts them on the masks of such a file."""
    video = sio.Video(filename="fake.mp4")
    pos = {0: [10, 12, 40, 42], 1: [40, 42, 12, 10]}
    embs = {0: [1, 0, 0], 1: [0, 1, 0]}
    lfs = []
    for t in range(4):
        insts, masks = [], []
        for a in (0, 1):
            emb = embs[a]
            i = _pinst(pos[a][t], 10, emb=emb if vectors_on == "poses" else None)
            m = _pmask(pos[a][t], 10, emb=emb if vectors_on == "masks" else None)
            m.instance = i
            insts.append(i)
            masks.append(m)
        lfs.append(
            sio.LabeledFrame(video=video, frame_idx=t, instances=insts, masks=masks)
        )
    return sio.Labels(lfs, videos=[video], skeletons=[SKEL])


def _track_of_vector(dets):
    """``{embedding vector -> set of track names}`` over detections."""
    out: dict = {}
    for d in dets:
        key = tuple(np.round(d.identity_embedding.vector, 3))
        out.setdefault(key, set()).add(d.track.name if d.track else None)
    return out


# ─────────────────────────────────────────────────────────────────────────
# B1 -- nothing but the tracked carrier's tracks changes
# ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "cfg",
    [TrackerConfig(), _emb_cfg()],
    ids=["keypoints", "embeddings"],
)
def test_pose_tracking_keeps_masks_rois_centroids_suggestions(cfg):
    """Pose mode rebuilt each frame as ``LabeledFrame(instances, masks=[])`` and
    returned a bare ``Labels(frames, videos, skeletons)``: masks, ROIs, centroids,
    suggestions and provenance were all deleted. Masks 2/frame -> 0/frame."""
    labels = _pose_mask_labels()
    out = apply_tracking(labels, cfg)

    assert [len(lf.masks) for lf in out] == [2, 2, 2, 2]
    assert [len(lf.rois) for lf in out] == [1, 1, 1, 1]
    assert [len(lf.centroids) for lf in out] == [1, 1, 1, 1]
    assert [s.frame_idx for s in out.suggestions] == [2]
    assert out.provenance.get("source") == "unit-test"
    # The masks stay linked to the (tracked) poses they belong to.
    for lf in out:
        assert all(m.instance in lf.instances for m in lf.masks)
        assert all(i.track is not None for i in lf.instances)


def test_mask_tracking_keeps_rois():
    """Mask mode kept the masks but still dropped every ROI (repro p8)."""
    video = sio.Video(filename="fake.mp4")
    lfs = []
    for t in range(3):
        masks = [_pmask(20 + t, 20, r=6), _pmask(60 + t, 60, r=6)]
        rois = [build_predicted_roi(m, 1.0, 0.01) for m in masks]
        lfs.append(sio.LabeledFrame(video=video, frame_idx=t, masks=masks, rois=rois))
    labels = sio.Labels(lfs, videos=[video], skeletons=[])
    out = apply_tracking(
        labels, TrackerConfig(scoring_method_explicit=False, features_explicit=False)
    )
    assert [len(lf.rois) for lf in out] == [2, 2, 2]
    assert all(m.track is not None for lf in out for m in lf.masks)


def test_tracks_in_place_and_returns_the_input():
    """The documented contract: the input is tracked and returned, never copied --
    so nothing is lost by reconstruction -- and `labels.copy()` keeps an arm
    independent in a sweep (the old shared-objects-in-a-new-Labels hybrid let a
    second run silently rewrite the first run's output)."""
    labels = _pose_mask_labels()
    # A stale catalog entry from an earlier tracking run must not survive.
    stale = sio.Track("track_0")
    for lf in labels:
        lf.instances[0].track = stale
    labels.update()
    assert stale in labels.tracks

    untouched = labels.copy()
    out = apply_tracking(labels, TrackerConfig())
    assert out is labels
    assert all(i.track is not None for lf in labels for i in lf.instances)
    assert stale not in labels.tracks
    assert set(labels.tracks) == {i.track for lf in labels for i in lf.instances}
    # The copy is untouched: still the stale track, and the second arm is its own.
    assert all(lf.instances[0].track.name == "track_0" for lf in untouched)
    assert all(lf.instances[1].track is None for lf in untouched)

    # Predictor.retrack is the same contract.
    again = Predictor.retrack(labels, TrackerConfig())
    assert again is labels


# ─────────────────────────────────────────────────────────────────────────
# B2 -- one count-based carrier resolver for embedding mode AND the blend
# ─────────────────────────────────────────────────────────────────────────
def test_blend_raises_when_the_vectors_are_on_the_untracked_carrier():
    """`appearance_weight` read the POSES' vectors while the embedding model had put
    them on the masks: w=0.9 was byte-identical to w=0.0, with no warning (p2)."""
    labels = _swap_labels(vectors_on="masks")
    with pytest.raises(ValueError, match="vectors are on the mask carrier"):
        apply_tracking(labels, TrackerConfig(appearance_weight=0.9, oks_stddev=0.5))


def test_blend_on_the_mask_carrier_follows_appearance():
    """`--features masks` now tracks the MASK carrier of a pose+mask file (it used
    to run pose mode and score poses as masks), so the blend reads the vectors
    where they are -- and actually changes the result -- and each pose inherits its
    linked mask's track."""
    geometry = apply_tracking(
        _swap_labels(vectors_on="masks"),
        TrackerConfig(features="masks", scoring_method="mask_iou"),
    )
    blended = apply_tracking(
        _swap_labels(vectors_on="masks"),
        TrackerConfig(
            features="masks", scoring_method="mask_iou", appearance_weight=0.9
        ),
    )
    geo = _track_of_vector(m for lf in geometry for m in lf.masks)
    app = _track_of_vector(m for lf in blended for m in lf.masks)
    # Geometry follows the position through the swap (each animal gets 2 tracks);
    # the blend follows the animal (one track each).
    assert all(len(v) == 2 for v in geo.values())
    assert all(len(v) == 1 for v in app.values())
    for lf in blended:
        for m in lf.masks:
            assert m.instance.track is m.track
            assert m.instance.tracking_score == m.tracking_score


def test_masks_on_a_centroid_file_skip_the_single_node_defaults(log_text):
    """A centroid+segmentation file has a 1-node skeleton. `--features masks` tracks
    its masks by mask IoU; the pose-only single-node defaults (euclidean_dist on
    centroids) must neither apply nor be announced."""
    centroid = sio.Skeleton(nodes=["centroid"])
    video = sio.Video(filename="fake.mp4")
    lfs = []
    for t in range(3):
        insts, masks = [], []
        for x in (10 + t, 60 + t):
            i = sio.PredictedInstance.from_numpy(
                np.array([[x, 10.0]], dtype=np.float32),
                skeleton=centroid,
                point_scores=np.ones(1),
                score=0.9,
            )
            m = _pmask(x, 10)
            m.instance = i
            insts.append(i)
            masks.append(m)
        lfs.append(
            sio.LabeledFrame(video=video, frame_idx=t, instances=insts, masks=masks)
        )
    labels = sio.Labels(lfs, videos=[video], skeletons=[centroid])
    out = apply_tracking(
        labels,
        TrackerConfig(
            features="masks", scoring_method_explicit=False, features_explicit=True
        ),
    )
    text = log_text()
    assert "Single-node skeleton detected" not in text
    assert "scoring_method='mask_iou'" in text
    assert len({m.track for lf in out for m in lf.masks}) == 2
    assert all(m.instance.track is m.track for lf in out for m in lf.masks)


def test_embedding_mode_routes_to_the_carrier_holding_more_vectors():
    """Routing used "do the poses carry ANY vector?": one stray pose vector sent
    appearance tracking to the poses, and the masks' vectors were ignored."""
    labels = _swap_labels(vectors_on="masks")
    labels[0].instances[0].identity_embedding = sio.Embedding(
        np.asarray([1, 0, 0], np.float32)
    )
    out = apply_tracking(labels, _emb_cfg())
    per_vec = _track_of_vector(m for lf in out for m in lf.masks)
    assert len(per_vec) == 2 and all(len(v) == 1 for v in per_vec.values())

    # The resolver PR C's `eval-tracking --carrier auto` reuses.
    counts = embedding_carriers(labels)
    assert (counts.n_mask_with, counts.n_pose_with) == (8, 1)
    assert counts.total(MASK_CARRIER) == counts.total(POSE_CARRIER) == 8
    assert counts.dominant == MASK_CARRIER
    assert embedding_carriers(_swap_labels(vectors_on="poses")).dominant == POSE_CARRIER


def test_mask_tracks_are_copied_onto_linked_poses_only_when_unambiguous():
    """Mask-carrier tracking left every linked pose UNTRACKED. A pose inherits its
    mask's track only when exactly one tracked mask on its frame links to it."""
    labels = _swap_labels(vectors_on="masks")
    # Frame 3: both masks claim the first pose -> ambiguous, the second is unlinked.
    lf3 = labels[3]
    lf3.masks[1].instance = lf3.instances[0]
    out = apply_tracking(labels, _emb_cfg())

    for lf in out[:3]:
        for m in lf.masks:
            assert m.instance.track is m.track
    assert all(i.track is None for i in out[3].instances)
    assert all(m.track is not None for m in out[3].masks)


@pytest.mark.parametrize("appearance_only", [True, False])
def test_warns_when_much_of_the_tracked_carrier_lacks_vectors(
    appearance_only, log_text
):
    """A partially embedded file (WF1 on an embed-only output) silently spawned a
    track per vector-less detection: 7 tracks for 2 animals. Now it says so."""
    video = sio.Video(filename="fake.mp4")
    lfs = []
    for t in range(4):
        emb_b = [0, 1, 0] if t % 2 == 0 else None  # animal B embedded half the time
        insts = [_pinst(10 + t, 10, emb=[1, 0, 0]), _pinst(60 + t, 60, emb=emb_b)]
        lfs.append(sio.LabeledFrame(video=video, frame_idx=t, instances=insts))
    labels = sio.Labels(lfs, videos=[video], skeletons=[SKEL])
    cfg = _emb_cfg() if appearance_only else TrackerConfig(appearance_weight=0.3)
    apply_tracking(labels, cfg)
    text = log_text()
    assert "2 of 8 pose detection(s) (25%) carry no appearance vector" in text
    expected = "spawns a fresh track" if appearance_only else "geometry alone"
    assert expected in text


# ─────────────────────────────────────────────────────────────────────────
# B4 -- a fresh tracker per video
# ─────────────────────────────────────────────────────────────────────────
def test_each_video_gets_its_own_tracks():
    """One tracker spanned all videos: video B's first frame was MATCHED to video
    A's last (repro p5 case 3). Now every video spawns its own tracks, named after
    the previous video's so no two videos share a name."""
    va, vb = sio.Video(filename="a.mp4"), sio.Video(filename="b.mp4")
    lfs = []
    for v, xs in ((va, (10, 60)), (vb, (60, 10))):
        for t in range(3):
            lfs.append(
                sio.LabeledFrame(
                    video=v,
                    frame_idx=t,
                    instances=[
                        _pinst(x, 10, emb=np.eye(3)[k]) for k, x in enumerate(xs)
                    ],
                )
            )
    labels = sio.Labels(lfs, videos=[va, vb], skeletons=[SKEL])
    out = apply_tracking(labels, _emb_cfg())

    tracks_in = {
        v.filename: {i.track for lf in out if lf.video is v for i in lf.instances}
        for v in (va, vb)
    }
    assert len(tracks_in["a.mp4"]) == len(tracks_in["b.mp4"]) == 2
    assert not tracks_in["a.mp4"] & tracks_in["b.mp4"]
    names = sorted(t.name for t in out.tracks)
    assert names == ["track_0", "track_1", "track_2", "track_3"]
    # ...and within each video, identity still holds.
    for v in (va, vb):
        dets = [i for lf in out if lf.video is v for i in lf.instances]
        assert all(len(s) == 1 for s in _track_of_vector(dets).values())


# ─────────────────────────────────────────────────────────────────────────
# B5 -- `cosine_sim` is not a geometric score
# ─────────────────────────────────────────────────────────────────────────
def test_cosine_sim_is_rejected_as_the_geometric_side_of_a_blend():
    """`features=keypoints, scoring_method=cosine_sim, appearance_weight>0` scored
    geometry as the cosine of raveled coordinates (poses ~190 px apart -> 0.98)."""
    labels = _pose_mask_labels()
    with pytest.raises(ValueError, match="cosine_sim is the APPEARANCE metric"):
        apply_tracking(
            labels,
            TrackerConfig(scoring_method="cosine_sim", appearance_weight=0.3),
        )


# ─────────────────────────────────────────────────────────────────────────
# B6 -- appearance-only scoring is vectorized
# ─────────────────────────────────────────────────────────────────────────
def test_embedding_tracking_emits_no_empty_slice_warnings():
    """The per-pair loop reduced all-NaN score lists (a detection without a vector)
    with `np.nanmean`, emitting "Mean of empty slice" on every such pair."""
    video = sio.Video(filename="fake.mp4")
    lfs = []
    for t in range(6):
        insts = [_pinst(10 + t, 10, emb=[1, 0, 0]), _pinst(60 + t, 60, emb=[0, 1, 0])]
        insts.append(_pinst(90, 90))  # a third detection with no vector, every frame
        lfs.append(sio.LabeledFrame(video=video, frame_idx=t, instances=insts))
    labels = sio.Labels(lfs, videos=[video], skeletons=[SKEL])
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = apply_tracking(labels, _emb_cfg())
    per_vec = _track_of_vector(
        i for lf in out for i in lf.instances if i.identity_embedding is not None
    )
    assert len(per_vec) == 2 and all(len(v) == 1 for v in per_vec.values())


# ─────────────────────────────────────────────────────────────────────────
# Default: `features=embeddings` uses `local_queues`
# ─────────────────────────────────────────────────────────────────────────
def _occlusion_labels():
    """Animal B disappears for 10 frames (5..14), then returns (repro p9)."""
    video = sio.Video(filename="fake.mp4")
    lfs = []
    for t in range(20):
        insts = [_pinst(10, 10, emb=[1, 0, 0])]
        if not 5 <= t < 15:
            insts.append(_pinst(60, 60, emb=[0, 1, 0]))
        lfs.append(sio.LabeledFrame(video=video, frame_idx=t, instances=insts))
    return sio.Labels(lfs, videos=[video], skeletons=[SKEL])


def test_embedding_tracking_recovers_identity_after_occlusion(log_text):
    """`fixed_window` (window 5) forgot B during its 10-frame absence and minted a
    new track on return -- the post-occlusion re-ID this mode is sold for."""
    out = apply_tracking(_occlusion_labels(), _emb_cfg())
    assert out[4].instances[1].track is out[15].instances[1].track
    assert "candidates_method='local_queues'" in log_text()


def test_explicit_fixed_window_is_still_honored():
    out = apply_tracking(
        _occlusion_labels(),
        _emb_cfg(candidates_method="fixed_window", candidates_method_explicit=True),
    )
    assert out[4].instances[1].track is not out[15].instances[1].track


# ─────────────────────────────────────────────────────────────────────────
# Through the CLI: `sleap-nn predict -i <file>.slp -t` (track-only route)
# ─────────────────────────────────────────────────────────────────────────
def _plain(text: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output (rich-click wraps by width)."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*m", "", text).split())


def _cli_track(in_path, out_path, *extra):
    from sleap_nn.cli import cli

    return CliRunner().invoke(
        cli,
        ["predict", "-i", str(in_path), "-t", "-o", str(out_path), *extra],
    )


def test_cli_track_only_keeps_every_annotation(tmp_path):
    """The written `.slp` keeps the masks, ROIs, centroids and suggestions of a
    pose-tracked file (they used to be deleted on the way to disk)."""
    src = tmp_path / "in.slp"
    sio.save_slp(_pose_mask_labels(), src.as_posix(), embed=False)
    out = tmp_path / "out.slp"
    result = _cli_track(src, out)
    assert result.exit_code == 0, _plain(result.output)

    tracked = sio.load_slp(out.as_posix())
    assert [len(lf.masks) for lf in tracked] == [2, 2, 2, 2]
    assert [len(lf.rois) for lf in tracked] == [1, 1, 1, 1]
    assert [len(lf.centroids) for lf in tracked] == [1, 1, 1, 1]
    assert [s.frame_idx for s in tracked.suggestions] == [2]
    assert all(i.track is not None for lf in tracked for i in lf.instances)
    assert all(m.instance is not None for lf in tracked for m in lf.masks)


def test_cli_embedding_retrack_recovers_identity_after_occlusion(tmp_path):
    """WF1 (`--features embeddings` on a saved-embeddings `.slp`) with
    `--candidates_method` left unset now defaults to `local_queues`."""
    src = tmp_path / "embedded.slp"
    sio.save_slp(
        _occlusion_labels(), src.as_posix(), embed=False, save_embedding_vectors=True
    )
    out = tmp_path / "tracked.slp"
    result = _cli_track(src, out, "--features", "embeddings")
    assert result.exit_code == 0, _plain(result.output)

    tracked = sio.load_slp(out.as_posix())
    b_tracks = {
        lf.instances[1].track.name
        for lf in tracked
        if len(lf.instances) == 2 and lf.frame_idx in (4, 15)
    }
    assert len(b_tracks) == 1, b_tracks


def test_cli_blend_on_vectors_it_cannot_read_fails(tmp_path):
    """The blend misconfiguration surfaces from the CLI as an error, not exit 0 with
    a geometry-only result."""
    src = tmp_path / "swap.slp"
    sio.save_slp(
        _swap_labels(vectors_on="masks"),
        src.as_posix(),
        embed=False,
        save_embedding_vectors=True,
    )
    result = _cli_track(src, tmp_path / "out.slp", "--appearance_weight", "0.9")
    assert result.exit_code != 0
    assert "vectors are on the mask carrier" in _plain(
        result.output + str(result.exception)
    )
