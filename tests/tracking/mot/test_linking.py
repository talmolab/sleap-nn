"""Tests for `sleap_nn.tracking.mot.linking` (offline identity linking)."""

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
    # Irregular stride: one step per processed frame (40 steps): linked.
    frames = _video(range(0, 700, 5), animals, vectors=False)
    cal = _cal(appearance=False, frame_step=1, regular_stride=False)
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
