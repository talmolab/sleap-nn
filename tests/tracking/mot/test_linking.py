"""Tests for `sleap_nn.tracking.mot.linking` (offline identity linking)."""

import time

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.tracking.mot.calibration import VideoCalibration
from sleap_nn.tracking.mot.linking import LinkingResult, _s_link, link_tracklets

_SKEL = sio.Skeleton(nodes=["a", "b", "c", "d"], name="s")
_DIM = 16


def _basis(seed: int = 0) -> np.ndarray:
    """Orthonormal rows: one well-separated appearance per synthetic animal."""
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(_DIM, _DIM)))
    return q.T


_BASE = _basis()


def _vec(animal, rng, noise=0.05):
    """A noisy unit vector around basis row ``animal`` (or an explicit vector)."""
    base = _BASE[animal] if np.isscalar(animal) else np.asarray(animal)
    v = base + rng.normal(scale=noise, size=_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _pose(x, y, track=None, vec=None, size=20.0, score=0.9):
    """A square 4-node pose centered at (x, y) with an optional track / vector."""
    h = size / 2
    pts = np.array([[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]])
    inst = sio.PredictedInstance.from_numpy(
        pts, skeleton=_SKEL, score=score, point_scores=np.full(4, score)
    )
    inst.track = track
    inst.tracking_score = 1.0 if track is not None else 0.0
    if vec is not None:
        inst.identity_embedding = sio.Embedding(vec)
    return inst


def _mask(x, y, track=None, vec=None, size=20, shape=(160, 320), score=0.9):
    """A square predicted mask centered at (x, y) with an optional track / vector."""
    data = np.zeros(shape, dtype=bool)
    h = size // 2
    data[int(y) - h : int(y) + h, int(x) - h : int(x) + h] = True
    mask = sio.PredictedSegmentationMask.from_numpy(data, score=score)
    mask.track = track
    mask.tracking_score = 1.0 if track is not None else 0.0
    if vec is not None:
        mask.identity_embedding = sio.Embedding(vec)
    return mask


def _video(frame_indices, animals, vectors=True, seed=0, make=_pose):
    """Build ``[(frame_idx, objects)]``.

    ``animals`` is a list of ``(animal_id, fn)`` where ``fn(frame) -> (x, y,
    track) | None`` (None: absent from that frame). ``animal_id`` picks the
    appearance basis vector (``None``: no vector).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for f in frame_indices:
        objs = []
        for animal, fn in animals:
            out = fn(f)
            if out is None:
                continue
            x, y, track = out
            vec = _vec(animal, rng) if (vectors and animal is not None) else None
            objs.append(make(x, y, track, vec))
        frames.append((f, objs))
    return frames


def _cal(appearance=True, **kwargs):
    """A hand-built calibration (calibrate() itself is tested elsewhere)."""
    fields = dict(
        status="ok" if appearance else "no_vectors",
        frame_step=1,
        regular_stride=True,
        body_size=28.0,
        nn_distance=100.0,
        step_p95=2.0,
        step_p99=3.0,
    )
    if appearance:
        fields.update(
            link_thresh=0.8,
            link_margin=0.05,
            link_recall=0.9,
            appearance_for_linking=True,
        )
    fields.update(kwargs)
    return VideoCalibration(**fields)


def _names(frames):
    """Per-frame list of track names (None for untracked)."""
    return [
        [None if o.track is None else o.track.name for o in objs] for _, objs in frames
    ]


def _tracks_where(frames, predicate):
    """Set of track names of the objects matching ``predicate(frame, obj)``."""
    return {
        None if o.track is None else o.track.name
        for f, objs in frames
        for o in objs
        if predicate(f, o)
    }


def _x_y(o):
    """Center of a test pose."""
    return tuple(float(v) for v in o.numpy().mean(axis=0))


def _x(o):
    return float(o.numpy()[:, 0].mean()) if hasattr(o, "numpy") else None


# ------------------------------------------------------------------- helpers
def test_s_link_is_mean_of_top5_means():
    """s_link = mean over queries of the mean top-5 cosine (all when < 5)."""
    e = np.eye(8)
    gallery = np.stack(
        [e[0] * c + e[1] * np.sqrt(1 - c * c) for c in np.linspace(0, 0.9, 7)]
    )
    q = e[0][None]
    cos = np.sort(gallery @ e[0])[::-1]
    assert _s_link(q, gallery) == pytest.approx(cos[:5].mean())
    assert _s_link(q, gallery[:3]) == pytest.approx((gallery[:3] @ e[0]).mean())
    two = np.stack([e[0], e[1]])
    assert _s_link(two, gallery) == pytest.approx(
        np.mean([np.sort(gallery @ v)[::-1][:5].mean() for v in two])
    )
    assert np.isnan(_s_link(q, gallery[:0]))
    assert np.isnan(_s_link(q[:0], gallery))


# ------------------------------------------------------------- gap linking
def _reid_scene(track_a, track_b, track_c):
    """Animal 0 leaves for 200 frames and comes back as track_c; animal 1 stays."""
    return _video(
        range(400),
        [
            (
                0,
                lambda f: (
                    (100 + f, 100, track_a)
                    if f < 100
                    else (220 + (f - 300), 120, track_c) if f >= 300 else None
                ),
            ),
            (1, lambda f: (400, 300 + 0.1 * f, track_b)),
        ],
    )


def test_appearance_gap_reid_merges_back_to_original_name():
    """An animal absent for 200 frames that returns with a new track is re-identified."""
    a, b, c = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    frames = _reid_scene(a, b, c)
    track_objects = {0: a, 1: b, 2: c}

    res = link_tracklets(frames, calibration=_cal(), track_objects=track_objects)

    assert isinstance(res, LinkingResult)
    assert res.n_identities_before == 3 and res.n_identities == 2
    assert res.n_merges == 1 and res.n_forced_merges == 0 and res.n_untracked == 0
    # Every animal-0 detection now carries the ORIGINAL (earlier) Track object.
    for f, objs in frames:
        for o in objs:
            assert o.track is (b if _x(o) == 400 else a)
    assert track_objects == {0: a, 1: b}
    assert any("track_2 -> track_0" in n for n in res.notes)
    assert res.as_dict()["n_merges"] == 1


def test_cooccurring_identities_are_never_merged():
    """Identical appearance does not merge identities that share frames."""
    a, b = sio.Track("track_0"), sio.Track("track_1")
    animals = [
        (0, lambda f: (100, 100, a)),
        (0, lambda f: (300, 100, b) if 10 <= f < 90 else None),
    ]
    frames = _video(range(100), animals)
    res = link_tracklets(frames, calibration=_cal())
    assert res.n_merges == 0 and res.n_identities == 2
    assert _tracks_where(frames, lambda f, o: _x(o) == 300) == {"track_1"}

    # Closed set N=1: still never merged; the smaller identity is untracked.
    frames = _video(range(100), animals)
    res = link_tracklets(frames, calibration=_cal(), max_tracks=1)
    assert res.n_forced_merges == 0 and res.n_identities == 1
    assert res.n_untracked == 80
    for _, objs in frames:
        for o in objs:
            if _x(o) == 300:
                assert o.track is None and o.tracking_score == 0.0
            else:
                assert o.track is a


def _new_animal_scene(t0, t1, t2):
    return [
        (0, lambda f: (100 + f, 100, t0) if f < 100 else None),
        # Spatially it could be animal 0 (short gap, close by), but its
        # appearance differs.
        (2, lambda f: (210 + (f - 150), 105, t1) if f >= 150 else None),
        (1, lambda f: (400, 300, t2)),
    ]


def test_new_animal_with_different_appearance_is_not_merged():
    """Open set: a truly new animal is not merged even when spatially plausible."""
    t0, t1, t2 = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    frames = _video(range(250), _new_animal_scene(t0, t1, t2))
    res = link_tracklets(frames, calibration=_cal())
    assert res.n_merges == 0 and res.n_identities == 3
    assert _tracks_where(frames, lambda f, o: f >= 150 and _x(o) != 400) == {"track_1"}


def test_spatial_only_unique_short_gap_merges_without_vectors():
    """No vectors: a short gap with a unique feasible pair is linked spatially."""
    t0, t1, t2 = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    frames = _video(range(250), _new_animal_scene(t0, t1, t2), vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False))
    assert res.n_merges == 1 and res.n_identities == 2
    assert _tracks_where(frames, lambda f, o: _x(o) != 400) == {"track_0"}
    assert any("spatial" in n for n in res.notes)


def test_spatial_only_no_merge_for_long_gap():
    """No vectors: a gap longer than 4 track buffers is never linked spatially."""
    t0, t1, t2 = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    animals = [
        (0, lambda f: (100, 100, t0) if f < 100 else None),
        (0, lambda f: (105, 100, t1) if f >= 300 else None),
        (1, lambda f: (400, 300, t2)),
    ]
    frames = _video(range(400), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False), track_buffer=30)
    assert res.n_merges == 0 and res.n_identities == 3


def test_spatial_only_counts_steps_not_frames():
    """With a regular stride, the gap is measured in steps of ``frame_step``."""
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    animals = [
        (None, lambda f: (100 + 0.1 * f, 100, t0) if f < 250 else None),
        (None, lambda f: (110 + 0.1 * f, 100, t1) if f >= 450 else None),
    ]
    # 200-frame gap = 40 steps at stride 5 (<= 4 * 30): linked.
    frames = _video(range(0, 700, 5), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False, frame_step=5))
    assert res.n_merges == 1
    # The same input calibrated at stride 1 is a 200-step gap: not linked.
    frames = _video(range(0, 700, 5), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False, frame_step=1))
    assert res.n_merges == 0
    # The "steps" time base (irregular input under "auto", or forced): one step
    # per processed frame (40 steps): linked.
    frames = _video(range(0, 700, 5), animals, vectors=False)
    cal = _cal(appearance=False, frame_step=1, regular_stride=False, time_base="steps")
    res = link_tracklets(frames, calibration=cal)
    assert res.n_merges == 1


def test_ambiguous_appearance_between_cooccurring_candidates_is_rejected():
    """Two co-occurring (provably different) predecessors that look alike: no link."""
    t0, t1, t2 = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    animals = [
        (0, lambda f: (100, 100, t0) if f < 100 else None),
        (0, lambda f: (140, 100, t1) if f < 100 else None),
        (0, lambda f: (120, 100, t2) if f >= 150 else None),
    ]
    frames = _video(range(250), animals)
    res = link_tracklets(frames, calibration=_cal())
    assert res.n_merges == 0 and res.n_identities == 3


def test_fragment_chain_of_one_animal_links_fully():
    """Three fragments of one animal all link (no skip-link nests the middle one)."""
    t0, t1, t2, t3 = (sio.Track(f"track_{i}") for i in range(4))
    animals = [
        (
            0,
            lambda f: (
                (100 + 0.2 * f, 100, t0)
                if f < 50
                else (
                    (100 + 0.2 * f, 100, t1)
                    if 60 <= f < 100
                    else (100 + 0.2 * f, 100, t2) if f >= 150 else None
                )
            ),
        ),
        (1, lambda f: (400, 300, t3)),
    ]
    # Seeds 0, 3 and 5: noise makes the skip link track_0 -> track_2 the top
    # score (without the deferral those runs end with 3 identities).
    for seed in (0, 1, 3, 5):
        frames = _video(range(200), animals, seed=seed)
        res = link_tracklets(frames, calibration=_cal())
        assert res.n_merges == 2 and res.n_identities == 2, seed
        assert _tracks_where(frames, lambda f, o: _x(o) != 400) == {"track_0"}


def test_ambiguous_fragment_neither_links_nor_blocks():
    """A fragment that looks like both animals stays separate without blocking.

    The middle fragment is equally similar to both animals, so every link to or
    from it fails its margin; each animal's own long-gap link skips over it and
    is still accepted (stage B of the selection).
    """
    t = [sio.Track(f"track_{i}") for i in range(5)]
    mixed = _BASE[0] + _BASE[1]
    animals = [
        (0, lambda f: (100, 100, t[0]) if f < 50 else None),
        (1, lambda f: (100, 160, t[1]) if f < 50 else None),
        (mixed, lambda f: (100, 130, t[2]) if 60 <= f < 100 else None),
        (0, lambda f: (100, 100, t[3]) if f >= 150 else None),
        (1, lambda f: (100, 160, t[4]) if f >= 150 else None),
    ]
    frames = _video(range(200), animals)
    res = link_tracklets(frames, calibration=_cal(link_thresh=0.6))
    assert res.n_merges == 2 and res.n_identities == 3
    assert _tracks_where(frames, lambda f, o: o.numpy()[0, 1] < 110) == {"track_0"}
    assert _tracks_where(frames, lambda f, o: o.numpy()[0, 1] > 140) == {"track_1"}
    assert _tracks_where(frames, lambda f, o: 60 <= f < 100) == {"track_2"}


def test_low_score_vectors_do_not_enter_galleries():
    """Vectors below ``high_threshold`` are not used: no appearance, no re-ID."""
    a, b, c = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    frames = [
        (
            f,
            [
                _pose(*_x_y(o), o.track, o.identity_embedding.vector, score=0.3)
                for o in objs
            ],
        )
        for f, objs in _reid_scene(a, b, c)
    ]
    # 200-step gap, no usable vectors: the spatial-only rule does not apply.
    res = link_tracklets(frames, calibration=_cal(), high_threshold=0.5)
    assert res.n_merges == 0 and res.n_identities == 3
    res = link_tracklets(frames, calibration=_cal(), high_threshold=0.2)
    assert res.n_merges == 1 and res.n_identities == 2


# --------------------------------------------------------------- closed set
def _four_fragments():
    tracks = [sio.Track(f"track_{i}") for i in range(4)]
    animals = [
        (
            None,
            lambda f: (
                (100 + 0.5 * f, 100, tracks[0 if f < 50 else 2])
                if f < 50 or f >= 60
                else None
            ),
        ),
        (
            None,
            lambda f: (
                (100 + 0.5 * f, 160, tracks[1 if f < 50 else 3])
                if f < 50 or f >= 60
                else None
            ),
        ),
    ]
    return _video(range(100), animals, vectors=False), tracks


def test_closed_set_four_fragments_to_two_identities():
    """N=2 with 4 fragments (spatially ambiguous): forced merges keep earliest names."""
    frames, tracks = _four_fragments()
    track_objects = {i: t for i, t in enumerate(tracks)}
    # Open set: the cross pairs are also feasible, so nothing is linked.
    res = link_tracklets(frames, calibration=_cal(appearance=False))
    assert res.n_merges == 0 and res.n_identities == 4

    res = link_tracklets(
        frames,
        calibration=_cal(appearance=False),
        max_tracks=2,
        track_objects=track_objects,
    )
    assert res.n_forced_merges == 2 and res.n_identities == 2
    assert res.n_untracked == 0
    assert _tracks_where(frames, lambda f, o: o.numpy()[0, 1] < 130) == {"track_0"}
    assert _tracks_where(frames, lambda f, o: o.numpy()[0, 1] > 130) == {"track_1"}
    assert track_objects == {0: tracks[0], 1: tracks[1]}


def test_closed_set_inserts_fragment_nested_by_a_forced_merge():
    """A fragment left inside a merged identity's gap is inserted, not untracked.

    The open set links the first and last fragments by appearance; the middle
    fragment (a blurrier look of the same animal) is below the threshold, so it
    ends up nested inside the merged identity's gap, where no ordered link can
    reach it. The closed set inserts it (both junctions spatially gated).
    """
    tracks = [sio.Track(f"track_{i}") for i in range(3)]
    blurry = _BASE[0] + 0.75 * _BASE[1]

    def animal(f):
        if f < 50:
            return 100, 100, tracks[0]
        if 60 <= f < 100:
            return 110, 100, tracks[1]
        if f >= 150:
            return 103, 100, tracks[2]
        return None

    frames = _video(
        range(200),
        [(0, lambda f: animal(f) if not 60 <= f < 100 else None)]
        + [(blurry, lambda f: animal(f) if 60 <= f < 100 else None)],
    )
    res = link_tracklets(frames, calibration=_cal(link_thresh=0.9))
    assert res.n_merges == 1 and res.n_identities == 2  # open set: track_1 stranded
    assert "linked track_2 -> track_0" in res.notes[-1]

    frames = _video(
        range(200),
        [(0, lambda f: animal(f) if not 60 <= f < 100 else None)]
        + [(blurry, lambda f: animal(f) if 60 <= f < 100 else None)],
    )
    res = link_tracklets(frames, calibration=_cal(link_thresh=0.9), max_tracks=1)
    assert res.n_merges == 1 and res.n_forced_merges == 1
    assert res.n_untracked == 0 and res.n_identities == 1
    assert _tracks_where(frames, lambda f, o: True) == {"track_0"}
    assert "track_1 -> track_0 (insertion" in res.notes[-1]


def test_closed_set_geometry_prefers_short_gap_continuation():
    """Geometry-only forced merges weigh the gap, not just the gated distance.

    track_0 ends at frame 49. track_1 starts 3 steps later 6 px away; track_2
    starts 111 steps later 1 px away (then walks off). Both are in the
    spatial-only window, so the open set abstains. Normalized by the gate alone
    the long-gap coincidence would look better (1/228 vs 6/46); the closed set
    takes the short-gap continuation.
    """
    tracks = [sio.Track(f"track_{i}") for i in range(3)]
    animals = [
        (None, lambda f: (100, 100, tracks[0]) if f < 50 else None),
        (None, lambda f: (106, 100, tracks[1]) if f >= 52 else None),
        (None, lambda f: (101, 100 + 2 * (f - 160), tracks[2]) if f >= 160 else None),
    ]
    frames = _video(range(300), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False), max_tracks=2)
    assert res.n_merges == 0
    assert res.n_forced_merges == 1 and res.n_identities == 2
    assert _tracks_where(frames, lambda f, o: _x(o) != 101) == {"track_0"}
    assert _tracks_where(frames, lambda f, o: _x(o) == 101) == {"track_2"}


def test_closed_set_untracks_duplicate_identity():
    """A co-occurring overlapping copy of an animal is untracked (duplicate)."""
    a, b, d = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    animals = [
        (0, lambda f: (100, 100, a)),
        (1, lambda f: (300, 100, b)),
        (0, lambda f: (101, 100, d) if 20 <= f <= 40 else None),
    ]
    frames = _video(range(100), animals)
    track_objects = {0: a, 1: b, 2: d}
    res = link_tracklets(
        frames, calibration=_cal(), max_tracks=2, track_objects=track_objects
    )
    assert res.n_identities == 2 and res.n_untracked == 21
    assert res.n_forced_merges == 0
    for _, objs in frames:
        for o in objs:
            if _x(o) == 101:
                assert o.track is None and o.tracking_score == 0.0
            else:
                assert o.track is (a if _x(o) == 100 else b)
    assert track_objects == {0: a, 1: b}
    assert any("duplicate" in n for n in res.notes)


def test_closed_set_untracks_fewest_detections_not_crossing_animals():
    """Brief overlaps at a crossing do not make a real animal a 'duplicate'."""
    a, b, j = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    animals = [
        (None, lambda f: (50 + 2 * f, 100, a)),
        (None, lambda f: (250 - 2 * f, 100, b)),
        (None, lambda f: (300, 300, j) if 30 <= f < 33 else None),
    ]
    frames = _video(range(100), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False), max_tracks=2)
    assert res.n_identities == 2 and res.n_untracked == 3
    assert _tracks_where(frames, lambda f, o: _x(o) == 300) == {None}
    assert _tracks_where(frames, lambda f, o: _x(o) != 300) == {"track_0", "track_1"}


# ----------------------------------------------------------- swap correction
def _crossing(t0, t1, swap_at=50):
    """Animals 0 and 1 cross at frame 50; the tracker swapped their labels there."""
    return [
        (0, lambda f: (50 + 2 * f, 100, t0 if f < swap_at else t1)),
        (1, lambda f: (250 - 2 * f, 100, t1 if f < swap_at else t0)),
    ]


def _labels_by_animal(frames):
    """{frame: (track of animal moving right, track of animal moving left)}."""
    out = {}
    for f, objs in frames:
        right = [o for o in objs if abs(_x(o) - (50 + 2 * f)) < 1e-6]
        left = [o for o in objs if abs(_x(o) - (250 - 2 * f)) < 1e-6]
        out[f] = (right[0].track.name, left[0].track.name)
    return out


def test_swap_correction_fixes_swap_after_crossing():
    """Discriminative calibrated appearance undoes a swap at a crossing."""
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    frames = _video(range(100), _crossing(t0, t1))
    res = link_tracklets(frames, calibration=_cal())
    assert res.n_swaps == 1 and res.n_merges == 0 and res.n_identities == 2
    labels = _labels_by_animal(frames)
    for f in range(0, 46):
        assert labels[f] == ("track_0", "track_1")
    for f in range(55, 100):
        assert labels[f] == ("track_0", "track_1"), f
    assert any("swap corrected" in n for n in res.notes)


@pytest.mark.parametrize("case", ["uninformative", "uncalibrated", "disabled"])
def test_no_swap_without_discriminative_calibrated_appearance(case):
    """Uninformative vectors, an uncalibrated video or swap_correction=False: no swap."""
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    animals = _crossing(t0, t1)
    if case == "uninformative":
        animals = [(0, fn) for _, fn in animals]  # same appearance for both
    frames = _video(range(100), animals)
    cal = _cal(appearance=case != "uncalibrated")
    res = link_tracklets(frames, calibration=cal, swap_correction=case != "disabled")
    assert res.n_swaps == 0
    labels = _labels_by_animal(frames)
    for f in range(55, 100):
        assert labels[f] == ("track_1", "track_0")


# ------------------------------------------------------------- misc / smoke
def test_deterministic():
    """Same input, same output (decisions and notes)."""
    outs = []
    for _ in range(2):
        frames, _ = _four_fragments()
        res = link_tracklets(frames, calibration=_cal(appearance=False), max_tracks=2)
        t = [sio.Track(f"track_{i}") for i in range(3)]
        reid = _reid_scene(*t)
        res2 = link_tracklets(reid, calibration=_cal())
        outs.append((_names(frames), res.as_dict(), _names(reid), res2.as_dict()))
    assert outs[0] == outs[1]


def test_mask_carrier_gap_reid_smoke():
    """Masks with vectors: gap re-ID works on the mask carrier too."""
    t0, t1, t2 = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    animals = [
        (
            0,
            lambda f: (
                (60 + f, 60, t0)
                if f < 20
                else (90 + (f - 40), 60, t1) if f >= 40 else None
            ),
        ),
        (1, lambda f: (250, 120, t2)),
    ]
    frames = _video(range(60), animals, make=_mask)
    res = link_tracklets(frames, carrier="mask", calibration=_cal())
    assert res.n_merges == 1 and res.n_identities == 2
    tracks = {o.track.name for _, objs in frames for o in objs}
    assert tracks == {"track_0", "track_2"}


def test_degenerate_inputs_never_raise():
    """Empty, untracked-only and single-identity inputs are no-ops."""
    res = link_tracklets([], calibration=_cal())
    assert res.n_identities_before == 0 and res.n_identities == 0

    res = link_tracklets([(0, []), (1, [])], calibration=_cal(), max_tracks=2)
    assert res.n_identities == 0

    untracked = _video(range(5), [(0, lambda f: (10 * f + 50, 50, None))])
    res = link_tracklets(untracked, calibration=_cal(), max_tracks=1)
    assert res.n_identities == 0
    assert all(o.track is None for _, objs in untracked for o in objs)

    a = sio.Track("track_0")
    single = _video(range(20), [(0, lambda f: (50 + f, 50, a))])
    res = link_tracklets(single, calibration=_cal(), max_tracks=1)
    assert res.n_identities_before == 1 and res.n_identities == 1
    assert all(o.track is a for _, objs in single for o in objs)

    # An invalid closed-set size is ignored (noted), never applied.
    frames, _ = _four_fragments()
    res = link_tracklets(frames, calibration=_cal(appearance=False), max_tracks=0)
    assert res.n_identities == 4 and res.n_untracked == 0
    assert any("max_tracks=0 ignored" in n for n in res.notes)


def test_uncalibrated_input_computes_calibration():
    """``calibration=None`` calibrates the video itself (spatial rules if needed)."""
    t0, t1, t2 = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    frames = _video(range(250), _new_animal_scene(t0, t1, t2), vectors=False)
    try:
        res = link_tracklets(frames)
    except NotImplementedError:
        pytest.skip("calibrate() is not implemented yet")
    assert res.n_identities_before == 3
    # No vectors: the short, unique, close-by gap is linked spatially.
    assert res.n_identities == 2 and res.n_merges == 1


# ------------------------------------------------------------- time base
def test_steps_follow_the_calibrations_time_base():
    """Gaps are counted in the calibration's RESOLVED time base (not its stride).

    A regular video with a 280-frame hole: in the "steps" time base (what the
    tracker counted, e.g. ``--mot_param time_base=steps``) the hole is 1 step and
    the two fragments link spatially; in "frames" it is 281 steps, beyond the
    spatial-only window.
    """
    t0, t1 = sio.Track("track_0"), sio.Track("track_1")
    idx = list(range(20)) + list(range(300, 320))
    animals = [
        (None, lambda f: (100, 100, t0) if f < 20 else None),
        (None, lambda f: (102, 100, t1) if f >= 300 else None),
    ]
    frames = _video(idx, animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False, time_base="steps"))
    assert res.n_merges == 1
    assert "gap 1 steps" in res.notes[-1]
    frames = _video(idx, animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False, time_base="frames"))
    assert res.n_merges == 0


def test_half_stride_gaps_round_like_the_trackers():
    """The linker's step positions are the shared `frame_steps` (half to even)."""
    from sleap_nn.tracking.mot.calibration import frame_steps
    from sleap_nn.tracking.mot.linking import _step_positions

    idx = np.array([0, 2, 4, 6, 11, 13, 15, 17])
    cal = _cal(appearance=False, frame_step=2)
    np.testing.assert_array_equal(
        _step_positions(idx, cal), frame_steps(idx, "frames", 2)
    )
    cal = _cal(appearance=False, frame_step=2, time_base="steps")
    np.testing.assert_array_equal(_step_positions(idx, cal), np.arange(8))


def test_uncalibrated_linking_passes_time_base_body_size_and_normalizer(monkeypatch):
    """Without a calibration, `time_base` / `body_size` reach `calibrate()`.

    ``score_normalizer`` rescales the scores before the gallery gate: raw scores
    of 0.45 are below ``high_threshold=0.5`` (no gated vector, no re-ID), but
    divided by the video's normalizer (0.5) they clear it.
    """
    from sleap_nn.tracking.mot import linking

    seen = {}
    real = linking.calibrate

    def spy(dets, **kwargs):
        seen.update(kwargs)
        return real(dets, **kwargs)

    monkeypatch.setattr(linking, "calibrate", spy)
    a, b, c = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    link_tracklets(
        _reid_scene(a, b, c), calibration=None, time_base="steps", body_size=33.0
    )
    assert seen["time_base"] == "steps" and seen["body_size"] == 33.0

    def low(frames):
        return [
            (
                f,
                [
                    _pose(*_x_y(o), o.track, o.identity_embedding.vector, score=0.45)
                    for o in objs
                ],
            )
            for f, objs in frames
        ]

    a, b, c = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    res = link_tracklets(low(_reid_scene(a, b, c)), calibration=_cal())
    assert res.n_merges == 0  # 200-step gap, no gated vector
    a, b, c = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    res = link_tracklets(
        low(_reid_scene(a, b, c)), calibration=_cal(), score_normalizer=0.5
    )
    assert res.n_merges == 1 and res.n_identities == 2


# ---------------------------------------------- spatial-only fragment chains
def _chain(period, dropout, n_frames, tracks, x0=100.0, y=100.0, speed=0.05):
    """One animal re-spawned as a new track after a dropout every ``period``."""

    def fn(f):
        k, r = divmod(f, period)
        if r >= period - dropout:
            return None
        return x0 + speed * f, y, tracks[k]

    return fn


def test_spatial_only_chain_of_short_fragments_links_fully():
    """No vectors: a chain of short fragments of ONE animal links into one.

    The animal's next-but-one fragment is inside the spatial-only window too, but
    it is reachable through the next one, so it is no competitor.
    """
    for period, dropout in ((60, 3), (100, 3), (81, 31)):
        n = 12
        tracks = [sio.Track(f"track_{i}") for i in range(n + 1)]
        animals = [
            (None, _chain(period, dropout, period * n, tracks)),
            (None, lambda f: (900, 700, tracks[n])),
        ]
        frames = _video(range(period * n), animals, vectors=False)
        res = link_tracklets(frames, calibration=_cal(appearance=False))
        assert res.n_identities == 2, (period, dropout)
        assert _tracks_where(frames, lambda f, o: _x(o) != 900) == {"track_0"}


def test_spatial_only_still_abstains_when_two_animals_end_together():
    """Two animals that both vanish and reappear nearby stay unlinked (ambiguous)."""
    t = [sio.Track(f"track_{i}") for i in range(4)]
    animals = [
        (None, lambda f: (100, 100, t[0]) if f < 50 else None),
        (None, lambda f: (100, 130, t[1]) if f < 50 else None),
        (None, lambda f: (100, 100, t[2]) if f >= 60 else None),
        (None, lambda f: (100, 130, t[3]) if f >= 60 else None),
    ]
    frames = _video(range(100), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False))
    assert res.n_merges == 0 and res.n_identities == 4
    assert any("spatial-only junction(s) left unlinked" in n for n in res.notes)


# ------------------------------------------------ per-length link thresholds
def test_short_heads_use_their_own_calibrated_threshold():
    """A 1-vector head is held to the 1-vector threshold, a full one to its own.

    The returning animal's vectors score ~0.7 against its old gallery: above the
    full-length threshold (0.6) but below the 1-vector one (0.8). With a single
    gated head vector it is not linked; with full heads it is.
    """
    a, b, c = sio.Track("track_0"), sio.Track("track_1"), sio.Track("track_2")
    by_len = (0.8,) + (0.6,) * 9
    cal = _cal(
        link_thresh=0.6,
        link_thresh_by_len=by_len,
        link_margin_by_len=(0.05,) * 10,
    )
    mixed = 0.75 * _BASE[0] + np.sqrt(1 - 0.75**2) * _BASE[5]

    def scene(n_gated):
        def ret(f):
            if f < 100:
                return 100 + f, 100, a
            if f >= 300:
                return 220 + (f - 300), 120, c
            return None

        frames = _video(
            range(400),
            [(0, lambda f: ret(f) if f < 100 else None)]
            + [(mixed, lambda f: ret(f) if f >= 300 else None)]
            + [(1, lambda f: (400, 300 + 0.1 * f, b))],
        )
        # Only the first n_gated returning detections are confident.
        k = 0
        for f, objs in frames:
            for i, o in enumerate(objs):
                if f >= 300 and _x(o) != 400:
                    if k >= n_gated:
                        objs[i] = _pose(*_x_y(o), o.track, None, score=0.9)
                    k += 1
        return frames

    res = link_tracklets(scene(1), calibration=cal)
    assert res.n_merges == 0 and res.n_identities == 3
    res = link_tracklets(scene(100), calibration=cal)
    assert res.n_merges == 1 and res.n_identities == 2
    # A length without a calibrated margin cannot be decided by appearance.
    cal = _cal(link_thresh_by_len=(0.6,) * 10, link_margin_by_len=(None,) + (0.05,) * 9)
    res = link_tracklets(scene(1), calibration=cal)
    assert res.n_merges == 0


# ------------------------------------------------------------- closed set v2
def test_closed_set_keeps_a_real_fragment_past_the_spatial_gate():
    """A real animal's fragment that fails the gate is kept, a far FP is not.

    The calibrated motion is tiny (step_p95 0.1 px; the gate radius floors at
    0.05 body per step), animal B walks 300 px while undetected and resumes as
    a new track. It co-occurs with nobody but animal A's identity, so it
    belongs to B: it joins B past the gate (substantial tracklet), nothing of
    B is untracked. A short false-positive track far away is still untracked.
    """
    t = [sio.Track(f"track_{i}") for i in range(4)]
    animals = [
        (None, lambda f: (100, 100, t[0])),
        (None, lambda f: (100, 400, t[1]) if f < 100 else None),
        (None, lambda f: (400, 400, t[2]) if f >= 160 else None),
    ]
    frames = _video(range(300), animals, vectors=False)
    cal = _cal(appearance=False, step_p95=0.1, step_p99=0.2)
    res = link_tracklets(frames, calibration=cal, max_tracks=2)
    assert res.n_untracked == 0 and res.n_identities == 2
    assert _tracks_where(frames, lambda f, o: _x(o) != 100) == {"track_1"}
    assert "past the spatial gate" in res.notes[-1]

    # A short far-away track inside B's gap: untracked, B still linked.
    animals.append((None, lambda f: (700, 50, t[3]) if 120 <= f < 130 else None))
    frames = _video(range(300), animals, vectors=False)
    res = link_tracklets(frames, calibration=cal, max_tracks=2)
    assert res.n_untracked == 10 and res.n_identities == 2
    assert _tracks_where(frames, lambda f, o: _x(o) == 700) == {None}
    assert _tracks_where(frames, lambda f, o: _x(o) == 400) == {"track_1"}


def test_closed_set_huddling_animals_keep_their_tracks():
    """Animals huddling (duplicates by IoU) most of the time are not untracked.

    Three animals sit on top of each other for 70% of the frames, a 30-detection
    false positive appears far away; ``max_tracks=3`` untracks the false
    positive, not a whole animal.
    """
    t = [sio.Track(f"track_{i}") for i in range(4)]

    def animal(k):
        return lambda f: (
            (100 + 3 * k, 100, t[k]) if f % 10 < 7 else (100 + 60 * k, 100, t[k])
        )

    animals = [(None, animal(k)) for k in range(3)]
    animals.append((None, lambda f: (600, 500, t[3]) if 400 <= f < 430 else None))
    frames = _video(range(1000), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False), max_tracks=3)
    assert res.n_identities == 3 and res.n_untracked == 30
    assert _tracks_where(frames, lambda f, o: _x(o) == 600) == {None}
    assert sum(o.track is not None for _, objs in frames for o in objs) == 3000


def test_closed_set_untracks_a_long_ghost_not_a_late_animal():
    """A long double detection of A is untracked, not a real late-entering animal."""
    t = [sio.Track(f"track_{i}") for i in range(4)]
    animals = [
        (None, lambda f: (100 + 0.1 * f, 100, t[0])),
        (None, lambda f: (100 + 0.1 * f, 400, t[1])),
        # A ghost: a second detection 2 px off animal 0 for 510 frames.
        (None, lambda f: (102 + 0.1 * f, 100, t[2]) if 100 <= f < 610 else None),
        # A real animal entering late (200 frames).
        (None, lambda f: (400, 250, t[3]) if f >= 800 else None),
    ]
    frames = _video(range(1000), animals, vectors=False)
    res = link_tracklets(frames, calibration=_cal(appearance=False), max_tracks=3)
    assert res.n_identities == 3 and res.n_untracked == 510
    assert _tracks_where(frames, lambda f, o: _x(o) == 400) == {"track_3"}
    assert any("duplicates track_0" in n for n in res.notes)


def _offset_duplicate_scene(seed=0, n_frames=600):
    """Two animals, misses, a long occlusion, and offset duplicate false positives.

    Returns ``(frames, gt)`` with gt ``0`` / ``1`` per real detection and ``-1``
    per duplicate. Duplicates sit 0.2-0.6 body sizes off a real animal with a
    lower score, as a degraded detector produces them.
    """
    from conftest import make_pose

    rng = np.random.default_rng(seed)
    t = np.arange(n_frames)
    centers = np.stack(
        [
            np.stack([200 + 80 * np.sin(t / 40), 150 + 30 * np.cos(t / 25)], 1),
            np.stack([500 + 80 * np.cos(t / 35), 400 + 30 * np.sin(t / 30)], 1),
        ],
        axis=1,
    )
    body = 60.0
    frames, gt = [], []
    for f in range(n_frames):
        objs, ids = [], []
        for k in range(2):
            if rng.random() < 0.05 or (k == 1 and 250 <= f < 290):
                continue
            objs.append(make_pose(centers[f, k], score=0.9))
            ids.append(k)
            if rng.random() < 0.15:
                ang = rng.uniform(0, 2 * np.pi)
                off = (
                    rng.uniform(0.2, 0.6) * body * np.array([np.cos(ang), np.sin(ang)])
                )
                objs.append(make_pose(centers[f, k] + off, score=0.6))
                ids.append(-1)
        frames.append((f, objs))
        gt.append(ids)
    return frames, gt


@pytest.mark.parametrize("tracker", ["sort", "bytetrack"])
def test_closed_set_with_offset_duplicates_keeps_real_detections(tracker):
    """Duplicate-spawned tracks do not cost the real animals their detections.

    With ``max_tracks=2`` the duplicates' short tracks co-occur with both real
    animals; only duplicate detections may be untracked, and the two animals
    keep one identity each (no swap introduced).
    """
    from conftest import build_tracker, count_id_switches, run_tracker

    frames, gt = _offset_duplicate_scene()
    names_open = run_tracker(build_tracker(tracker, track_buffer=30), frames)
    frames, gt = _offset_duplicate_scene()
    names = run_tracker(build_tracker(tracker, track_buffer=30, max_tracks=2), frames)
    real = [(n, g) for fn, fg in zip(names, gt) for n, g in zip(fn, fg) if g >= 0]
    untracked_real = sum(n is None for n, _ in real)
    assert untracked_real <= 0.01 * len(real), untracked_real
    assert {n for n, _ in real if n is not None} <= {"track_0", "track_1"}
    assert count_id_switches(names, gt) <= count_id_switches(names_open, gt)
    for k in (0, 1):
        own = [n for n, g in real if g == k and n is not None]
        assert len(set(own)) == 1, (k, set(own))


# ---------------------------------------------------------- scaling / events
def _reference_events(overlaps, pos, label):
    """The original (quadratic) ambiguity-event construction, for comparison."""
    frame_pos = pos[overlaps[:, 0]]
    events, active = [], []
    for p in np.unique(frame_pos):
        edges = overlaps[frame_pos == p]
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in edges:
            ra, rb = find(int(label[a])), find(int(label[b]))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        comps = {}
        for a, b in edges:
            for d in (int(a), int(b)):
                lab = int(label[d])
                ev = comps.setdefault(find(lab), [p, p, set(), []])
                ev[2].add(lab)
                if d not in ev[3]:
                    ev[3].append(d)
        active = [e for e in active if e[1] == p - 1]
        current = []
        for root in sorted(comps):
            ev = comps[root]
            while True:
                other = next((e for e in active + current if e[2] & ev[2]), None)
                if other is None:
                    break
                ev[0] = min(ev[0], other[0])
                ev[2] |= other[2]
                ev[3] = other[3] + ev[3]
                pool = active if any(e is other for e in active) else current
                pool[:] = [e for e in pool if e is not other]
                events[:] = [e for e in events if e is not other]
            current.append(ev)
            events.append(ev)
        active = current
    events.sort(key=lambda e: (e[0], e[1], min(e[3])))
    return [(e[0], e[1], e[2], sorted(e[3])) for e in events]


def _overlap_structure(n_frames, n_ids, p_overlap, seed):
    """Random per-frame overlaps between identities (detections in time order)."""
    rng = np.random.default_rng(seed)
    pos, label, overlaps = [], [], []
    for f in range(n_frames):
        first = len(pos)
        for k in range(n_ids):
            pos.append(f)
            label.append(k)
        for a in range(n_ids):
            for b in range(a + 1, n_ids):
                if rng.random() < p_overlap:
                    overlaps.append((first + a, first + b))
    return (
        np.asarray(overlaps, dtype=np.int64).reshape(-1, 2),
        np.asarray(pos),
        np.asarray(label),
    )


def test_ambiguity_events_match_reference_and_scale_linearly():
    """Linear event construction == the original rule; a long huddle stays fast."""
    from sleap_nn.tracking.mot.linking import _Linker

    for seed, (n_frames, n_ids, p) in enumerate(
        [(300, 4, 0.1), (300, 3, 0.6), (200, 6, 0.05), (100, 2, 0.95)]
    ):
        ov, pos, label = _overlap_structure(n_frames, n_ids, p, seed)
        linker = object.__new__(_Linker)
        linker.overlaps, linker.pos, linker.label = ov, pos, label
        got = [
            (e.start, e.end, e.labels, sorted(e.dets))
            for e in linker.ambiguity_events()
        ]
        assert got == _reference_events(ov, pos, label), (n_frames, n_ids, p)

    # A 4-animal huddle over 20k frames (~108k overlaps): well under a second.
    ov, pos, label = _overlap_structure(20000, 4, 0.9, 0)
    linker = object.__new__(_Linker)
    linker.overlaps, linker.pos, linker.label = ov, pos, label
    t0 = time.perf_counter()
    events = linker.ambiguity_events()
    assert time.perf_counter() - t0 < 2.0
    assert len(events) >= 1


def test_linking_perf_smoke_many_identities():
    """~600 identities (10 animals, a new track every 30 frames) link in seconds.

    Exercises the sparse open set (appearance and spatial candidates) and the
    closed set; the dense identity x identity version was cubic.
    """
    n_animals, period, n_frames = 10, 30, 1800
    rng = np.random.default_rng(0)
    tracks = {}
    frames = []
    for f in range(n_frames):
        objs = []
        for k in range(n_animals):
            seg, r = divmod(f, period)
            if r >= period - 2:
                continue
            key = (k, seg)
            if key not in tracks:
                tracks[key] = sio.Track(f"track_{len(tracks)}")
            x = 60 + 120 * k + 10 * np.sin(f / 50 + k)
            objs.append(_pose(x, 300 + 5 * np.cos(f / 30), tracks[key], _vec(k, rng)))
        frames.append((f, objs))
    assert len(tracks) == 600
    t0 = time.perf_counter()
    res = link_tracklets(frames, calibration=_cal(), max_tracks=n_animals)
    elapsed = time.perf_counter() - t0
    assert res.n_identities == n_animals and res.n_untracked == 0
    assert elapsed < 10.0, elapsed
