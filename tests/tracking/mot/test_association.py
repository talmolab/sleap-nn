"""Tests for `sleap_nn.tracking.mot.association` against brute-force references."""

import itertools
import math

import numpy as np
import pytest

from sleap_nn.evaluation import compute_instance_area, compute_oks
from sleap_nn.tracking.mot.association import (
    CHI2INV95,
    INFEASIBLE_COST,
    adaptive_weight,
    appearance_margins,
    centroid_similarity,
    cosine_matrix,
    fuse_score,
    ioa_min_matrix,
    iou_matrix,
    linear_assignment,
    linear_assignment_masked,
    ocm_matrix,
    oks_matrix,
    one_to_one_pattern,
    speed_direction,
)

# ----------------------------------------------------------------- helpers


def _random_boxes(rng, n, nan_rows=()):
    xy = rng.uniform(0, 50, size=(n, 2))
    wh = rng.uniform(0, 30, size=(n, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    boxes[list(nan_rows)] = np.nan
    return boxes


def _box_ok(b):
    return all(math.isfinite(v) for v in b)


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _brute_iou(A, B):
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if not (_box_ok(a) and _box_ok(b)):
                continue
            union = _area(a) + _area(b) - _inter(a, b)
            out[i, j] = _inter(a, b) / union if union > 0 else 0.0
    return out


def _brute_ioa_min(A, B):
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            if not (_box_ok(a) and _box_ok(b)):
                continue
            den = min(_area(a), _area(b))
            out[i, j] = _inter(a, b) / den if den > 0 else 0.0
    return out


def _brute_oks(kp_a, kp_b, area_b, sigma):
    n, m, k = len(kp_a), len(kp_b), kp_a.shape[1]
    sigma = np.broadcast_to(sigma, (k,))
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            area = area_b[j]
            area = max(area, 1.0) if math.isfinite(area) else 1.0
            vals = []
            for node in range(k):
                pa, pb = kp_a[i, node], kp_b[j, node]
                if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                    continue
                d2 = float(((pa - pb) ** 2).sum())
                e = d2 / (2 * sigma[node]) ** 2 / (area + np.spacing(1)) / 2
                vals.append(math.exp(-e))
            out[i, j] = sum(vals) / len(vals) if vals else 0.0
    return out


def _random_poses(rng, n, k, nan_frac=0.3):
    kp = rng.uniform(0, 100, size=(n, k, 2))
    kp[rng.uniform(size=(n, k)) < nan_frac] = np.nan
    return kp


# -------------------------------------------------------------- similarity


def test_iou_and_ioa_min_match_brute_force():
    """Vectorized IoU / IoMin equal per-pair loops (NaN rows -> 0)."""
    rng = np.random.default_rng(0)
    for trial in range(10):
        A = _random_boxes(rng, 6, nan_rows=(1,))
        B = _random_boxes(rng, 5, nan_rows=(3,) if trial % 2 else ())
        # A zero-area box and an exact duplicate.
        A[2] = [10, 10, 10, 20]
        B[0] = A[0]
        iou = iou_matrix(A, B)
        np.testing.assert_allclose(iou, _brute_iou(A, B), atol=1e-12)
        np.testing.assert_allclose(ioa_min_matrix(A, B), _brute_ioa_min(A, B))
        assert np.isfinite(iou).all() and iou.min() >= 0 and iou.max() <= 1
    assert iou_matrix(A[:1], A[:1])[0, 0] == pytest.approx(1.0)


def test_box_matrices_empty_shapes():
    """(0, m) and (n, 0) inputs give correctly shaped empty outputs."""
    boxes = np.zeros((3, 4))
    for fn in (iou_matrix, ioa_min_matrix):
        assert fn(np.zeros((0, 4)), boxes).shape == (0, 3)
        assert fn(boxes, np.zeros((0, 4))).shape == (3, 0)


def test_ioa_min_contained_box_is_one():
    """A box inside another has IoMin 1 (duplicate/split detection)."""
    out = ioa_min_matrix(np.array([[0, 0, 10, 10]]), np.array([[2, 2, 5, 5]]))
    assert out[0, 0] == pytest.approx(1.0)


def test_oks_matches_brute_force_with_nan_nodes():
    """Vectorized OKS equals a per-node loop with symmetric visibility."""
    rng = np.random.default_rng(1)
    kp_a = _random_poses(rng, 5, 6)
    kp_b = _random_poses(rng, 4, 6)
    kp_a[0] = np.nan  # no shared node at all -> 0
    area_b = rng.uniform(0.0, 2000.0, size=4)
    area_b[1] = np.nan  # floored to 1
    area_b[2] = 0.2  # floored to 1
    for sigma in (0.1, rng.uniform(0.02, 0.2, size=6)):
        out = oks_matrix(kp_a, kp_b, area_b, sigma)
        np.testing.assert_allclose(out, _brute_oks(kp_a, kp_b, area_b, sigma))
        assert np.isfinite(out).all() and out.min() >= 0 and out.max() <= 1
        assert (out[0] == 0).all()


def test_oks_equals_compute_oks_cocoeval_for_visible_poses():
    """Same constants as `evaluation.compute_oks(use_cocoeval=True)`.

    `compute_oks(points_gt, points_pr)` scores against the GT's bounding-box area
    (`compute_instance_area`, the unpadded keypoint extent); here the `b` side
    plays the GT, so `area_b` is that same extent.
    """
    rng = np.random.default_rng(2)
    kp_a = _random_poses(rng, 4, 5, nan_frac=0.0)
    kp_b = kp_a[[2, 0, 1]] + rng.normal(0, 3.0, size=(3, 5, 2))
    area_b = compute_instance_area(kp_b)
    for sigma in (0.025, 0.1):
        expected = compute_oks(kp_b, kp_a, stddev=sigma, use_cocoeval=True).T
        np.testing.assert_allclose(oks_matrix(kp_a, kp_b, area_b, sigma), expected)
    # Explicit scale argument too.
    expected = compute_oks(kp_b, kp_a, scale=area_b, stddev=0.05).T
    np.testing.assert_allclose(oks_matrix(kp_a, kp_b, area_b, 0.05), expected)


def test_oks_empty_and_bad_inputs():
    """Empty sides are fine (even with mismatched K); bad shapes/sigma raise."""
    assert oks_matrix(np.zeros((0, 3, 2)), np.zeros((2, 3, 2)), [1, 1], 0.1).shape == (
        0,
        2,
    )
    assert oks_matrix(np.zeros((2, 3, 2)), np.zeros((0, 1, 2)), [], 0.1).shape == (
        2,
        0,
    )
    with pytest.raises(ValueError):
        oks_matrix(np.zeros((1, 3, 2)), np.zeros((1, 4, 2)), [1.0], 0.1)
    with pytest.raises(ValueError):
        oks_matrix(np.zeros((1, 3, 2)), np.zeros((1, 3, 2)), [1.0], 0.0)


def test_centroid_similarity_matches_brute_force():
    """exp(-d^2 / (2 s^2)); NaN points -> 0."""
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 100, size=(5, 2))
    b = rng.uniform(0, 100, size=(4, 2))
    a[1] = np.nan
    b[2, 0] = np.nan
    scale = 17.5
    out = centroid_similarity(a, b, scale)
    expected = np.zeros((5, 4))
    for i in range(5):
        for j in range(4):
            if np.isfinite(a[i]).all() and np.isfinite(b[j]).all():
                d2 = float(((a[i] - b[j]) ** 2).sum())
                expected[i, j] = math.exp(-d2 / (2 * scale**2))
    np.testing.assert_allclose(out, expected)
    assert centroid_similarity(np.zeros((0, 2)), b, 1.0).shape == (0, 4)
    assert centroid_similarity(a, np.zeros((0, 2)), 1.0).shape == (5, 0)
    with pytest.raises(ValueError):
        centroid_similarity(a, b, 0.0)


def test_cosine_matrix_matches_brute_force():
    """Cosines of available pairs; NaN where either side is missing."""
    rng = np.random.default_rng(4)
    e_a = rng.normal(size=(5, 8))
    e_a /= np.linalg.norm(e_a, axis=1, keepdims=True)
    e_b = rng.normal(size=(3, 8))
    e_b /= np.linalg.norm(e_b, axis=1, keepdims=True)
    ok_a = np.array([True, False, True, True, True])
    ok_b = np.array([True, True, False])
    e_a[1] = np.nan  # ignored (not ok)
    out = cosine_matrix(e_a, ok_a, e_b.astype(np.float32), ok_b)
    for i in range(5):
        for j in range(3):
            if ok_a[i] and ok_b[j]:
                assert out[i, j] == pytest.approx(float(e_a[i] @ e_b[j]), abs=1e-6)
            else:
                assert np.isnan(out[i, j])

    # Missing vector arrays, dimension mismatch and empty shapes.
    assert np.isnan(cosine_matrix(None, ok_a, e_b, ok_b)).all()
    assert np.isnan(cosine_matrix(e_a, ok_a, e_b[:, :4], ok_b)).all()
    assert cosine_matrix(np.zeros((0, 8)), [], e_b, ok_b).shape == (0, 3)


def test_fuse_score():
    """Similarity scaled column-wise by detection scores."""
    sim = np.array([[0.5, 1.0], [0.2, 0.4]])
    np.testing.assert_allclose(
        fuse_score(sim, np.array([0.5, 0.25])), [[0.25, 0.25], [0.1, 0.1]]
    )
    assert fuse_score(np.zeros((0, 2)), np.ones(2)).shape == (0, 2)


def test_chi2_table():
    """The 2-dof 0.95 quantile used for position-only gating."""
    assert CHI2INV95[2] == pytest.approx(5.9915)
    assert CHI2INV95[4] == pytest.approx(9.4877)


# -------------------------------------------------------------- assignment


def _brute_min_gated(cost, max_cost):
    """Min over assignments where an unmatched (row, col) pair costs ~max_cost."""
    n, m = cost.shape
    feasible = np.isfinite(cost) & (cost <= max_cost)
    fill = np.where(feasible, cost, max_cost + 1e-5)
    best = np.inf
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            best = min(best, sum(fill[i, c] for i, c in enumerate(cols)))
    else:
        for rows in itertools.permutations(range(n), m):
            best = min(best, sum(fill[r, j] for j, r in enumerate(rows)))
    return best


def _check_partition(matches, ur, uc, n, m):
    assert matches.ndim == 2 and matches.shape[1] == 2
    assert sorted(set(matches[:, 0]) | set(ur)) == list(range(n))
    assert sorted(set(matches[:, 1]) | set(uc)) == list(range(m))
    assert len(set(matches[:, 0])) == len(matches) == len(set(matches[:, 1]))
    assert not set(matches[:, 0]) & set(ur)
    assert not set(matches[:, 1]) & set(uc)
    assert list(matches[:, 0]) == sorted(matches[:, 0])
    assert list(ur) == sorted(ur) and list(uc) == sorted(uc)


def test_linear_assignment_cap_and_optimality_random():
    """Cap respected, infeasible never matched, optimal under the gated objective."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        n, m = rng.integers(1, 5, size=2)
        cost = rng.uniform(0, 1, size=(n, m))
        cost[rng.uniform(size=(n, m)) < 0.15] = np.nan
        cost[rng.uniform(size=(n, m)) < 0.1] = np.inf
        max_cost = 0.6
        matches, ur, uc = linear_assignment(cost, max_cost)
        _check_partition(matches, ur, uc, n, m)
        if len(matches):
            vals = cost[matches[:, 0], matches[:, 1]]
            assert np.isfinite(vals).all() and (vals <= max_cost).all()
        # The matched set is optimal: its gated objective equals the brute force.
        unmatched_pairs = min(n, m) - len(matches)
        total = sum(cost[r, c] for r, c in matches) + unmatched_pairs * (
            max_cost + 1e-5
        )
        assert total == pytest.approx(_brute_min_gated(cost, max_cost), abs=1e-9)


def test_linear_assignment_keeps_confident_match():
    """A forbidden pair costs ~max_cost: no swap just to add a match.

    Track 0 matches det 0 perfectly; track 1 can only take det 0. Forcing two
    matches (0->1, 1->0) would trade the perfect match for two weak ones.
    """
    cost = np.array([[0.0, 0.6], [0.6, np.inf]])
    matches, ur, uc = linear_assignment(cost, 0.7)
    assert matches.tolist() == [[0, 0]]
    assert ur.tolist() == [1] and uc.tolist() == [1]


def test_linear_assignment_edge_cases():
    """Empty, all-infeasible, inclusive cap, determinism, infinite cap."""
    for shape in ((0, 3), (3, 0), (0, 0)):
        matches, ur, uc = linear_assignment(np.zeros(shape), 0.5)
        assert matches.shape == (0, 2)
        assert ur.tolist() == list(range(shape[0]))
        assert uc.tolist() == list(range(shape[1]))

    matches, ur, uc = linear_assignment(np.full((2, 2), np.nan), 0.5)
    assert matches.shape == (0, 2) and ur.tolist() == [0, 1] and uc.tolist() == [0, 1]

    # Cap is inclusive; above it is dropped.
    matches, _, _ = linear_assignment(np.array([[0.5, 0.50001]]), 0.5)
    assert matches.tolist() == [[0, 0]]
    matches, _, _ = linear_assignment(np.array([[0.7]]), 0.5)
    assert matches.shape == (0, 2)

    # Deterministic on ties.
    cost = np.zeros((3, 3))
    first = linear_assignment(cost, 1.0)
    for _ in range(3):
        again = linear_assignment(cost, 1.0)
        for x, y in zip(first, again):
            np.testing.assert_array_equal(x, y)

    # Infinite cap: only non-finite entries are infeasible.
    cost = np.array([[5.0, np.inf], [np.nan, 1e3]])
    matches, _, _ = linear_assignment(cost, np.inf)
    assert matches.tolist() == [[0, 0], [1, 1]]

    # Gated cells at INFEASIBLE_COST are never matched.
    matches, _, uc = linear_assignment(np.array([[INFEASIBLE_COST, 0.2]]), 0.5)
    assert matches.tolist() == [[0, 1]] and uc.tolist() == [0]

    with pytest.raises(ValueError):
        linear_assignment(np.zeros((2, 2)), np.nan)


def _brute_max_masked(score, valid):
    """Max over full assignments of the score (invalid pairs count their score)."""
    n, m = score.shape
    best = -np.inf
    if n <= m:
        for cols in itertools.permutations(range(m), n):
            best = max(best, sum(score[i, c] for i, c in enumerate(cols)))
    else:
        for rows in itertools.permutations(range(n), m):
            best = max(best, sum(score[r, j] for j, r in enumerate(rows)))
    return best


def test_linear_assignment_masked_never_matches_invalid():
    """Random scores/masks: only valid pairs returned; objective is optimal."""
    rng = np.random.default_rng(6)
    for _ in range(40):
        n, m = rng.integers(1, 5, size=2)
        score = rng.uniform(-0.5, 1.5, size=(n, m))
        valid = rng.uniform(size=(n, m)) < 0.5
        matches, ur, uc = linear_assignment_masked(score, valid)
        _check_partition(matches, ur, uc, n, m)
        if len(matches):
            assert valid[matches[:, 0], matches[:, 1]].all()

    # Objective: full-matrix optimum, then invalid pairs rejected (reference).
    score = rng.uniform(0, 1, size=(3, 3))
    valid = np.ones((3, 3), dtype=bool)
    matches, _, _ = linear_assignment_masked(score, valid)
    assert sum(score[r, c] for r, c in matches) == pytest.approx(
        _brute_max_masked(score, valid)
    )


def test_linear_assignment_masked_semantics():
    """Validity decides acceptance; the score only orders; NaN never matched."""
    # Highest-score pair is invalid -> not returned; the valid pair is.
    score = np.array([[0.9, 0.35], [0.95, 0.1]])
    valid = np.array([[True, True], [False, False]])
    matches, ur, uc = linear_assignment_masked(score, valid)
    assert all(valid[r, c] for r, c in matches)
    # Momentum flips the order between two valid pairs.
    score = np.array([[0.5, 0.45], [0.45, 0.2]]) + np.array([[0, 0.1], [0.1, 0]])
    valid = np.ones((2, 2), dtype=bool)
    matches, _, _ = linear_assignment_masked(score, valid)
    assert matches.tolist() == [[0, 1], [1, 0]]
    # Non-finite scores are never matched; empty shapes are fine.
    matches, _, _ = linear_assignment_masked(
        np.array([[np.nan, 0.5]]), np.array([[True, True]])
    )
    assert matches.tolist() == [[0, 1]]
    for shape in ((0, 2), (2, 0)):
        matches, ur, uc = linear_assignment_masked(
            np.zeros(shape), np.zeros(shape, dtype=bool)
        )
        assert matches.shape == (0, 2)
        assert len(ur) == shape[0] and len(uc) == shape[1]
    with pytest.raises(ValueError):
        linear_assignment_masked(np.zeros((2, 2)), np.zeros((2, 3), dtype=bool))


def test_one_to_one_pattern():
    """Return the pairs only for a non-empty pattern with <= 1 True per row/col."""
    assert one_to_one_pattern(np.zeros((2, 2), dtype=bool)) is None
    assert one_to_one_pattern(np.zeros((0, 3), dtype=bool)) is None
    v = np.array([[False, True, False], [False, False, False], [True, False, False]])
    np.testing.assert_array_equal(one_to_one_pattern(v), [[0, 1], [2, 0]])
    assert one_to_one_pattern(np.array([[True, True], [False, False]])) is None
    assert one_to_one_pattern(np.array([[True, False], [True, False]])) is None
    np.testing.assert_array_equal(one_to_one_pattern(np.array([[True]])), [[0, 0]])


# ---------------------------------------------------------------- OC-SORT


def test_speed_direction():
    """(dy, dx) unit vectors between box centers; zero motion -> (0, 0)."""
    box = np.array([0.0, 0.0, 10.0, 10.0])
    np.testing.assert_allclose(speed_direction(box, box + [3, 0, 3, 0]), [0, 1])
    np.testing.assert_allclose(speed_direction(box, box + [0, 4, 0, 4]), [1, 0])
    np.testing.assert_allclose(
        speed_direction(box, box + [3, 4, 3, 4]), [0.8, 0.6], atol=1e-12
    )
    np.testing.assert_array_equal(speed_direction(box, box), [0.0, 0.0])
    # Row-wise, with a non-finite row.
    prev = np.stack([box, box, box])
    cur = np.stack([box + [-6, 0, -6, 0], box, np.full(4, np.nan)])
    np.testing.assert_allclose(
        speed_direction(prev, cur), [[0, -1], [0, 0], [0, 0]], atol=1e-12
    )
    assert speed_direction(np.zeros((0, 4)), np.zeros((0, 4))).shape == (0, 2)


def _brute_ocm(dirs, prev, has_dir, dets, scores, inertia):
    out = np.zeros((len(dirs), len(dets)))
    for i in range(len(dirs)):
        for j in range(len(dets)):
            pcy, pcx = (prev[i, 1] + prev[i, 3]) / 2, (prev[i, 0] + prev[i, 2]) / 2
            dcy, dcx = (dets[j, 1] + dets[j, 3]) / 2, (dets[j, 0] + dets[j, 2]) / 2
            dy, dx = dcy - pcy, dcx - pcx
            norm = math.hypot(dy, dx)
            if not math.isfinite(norm) or norm == 0:
                dy = dx = 0.0
            else:
                dy, dx = dy / norm, dx / norm
            cos = dirs[i, 0] * dy + dirs[i, 1] * dx
            cos = min(1.0, max(-1.0, cos))
            theta = abs(math.acos(cos))
            out[i, j] = inertia * (math.pi / 2 - theta) / math.pi
            out[i, j] *= float(has_dir[i]) * scores[j]
    return out


def test_ocm_matrix_matches_brute_force():
    """Momentum term equals the OC-SORT formula computed per pair."""
    rng = np.random.default_rng(7)
    n, m = 5, 6
    dirs = rng.normal(size=(n, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    prev = _random_boxes(rng, n)
    has_dir = np.array([True, True, False, True, True])
    dets = _random_boxes(rng, m)
    dets[2] = prev[0]  # detection centered on the earlier observation -> 0
    scores = rng.uniform(0, 1, size=m)
    out = ocm_matrix(dirs, prev, has_dir, dets, scores, 0.2)
    np.testing.assert_allclose(out, _brute_ocm(dirs, prev, has_dir, dets, scores, 0.2))
    assert (out[2] == 0).all() and out[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert np.abs(out).max() <= 0.1 + 1e-12
    assert ocm_matrix(dirs[:0], prev[:0], has_dir[:0], dets, scores, 0.2).shape == (
        0,
        m,
    )


def test_ocm_matrix_sign_convention():
    """Moving along the track's direction is rewarded; reversing is penalized."""
    prev = np.array([[0.0, 0.0, 10.0, 10.0]])
    right = np.array([[0.0, 1.0]])  # (dy, dx): moving +x
    dets = np.array([[20.0, 0.0, 30.0, 10.0], [-20.0, 0.0, -10.0, 10.0]])
    out = ocm_matrix(right, prev, [True], dets, np.array([1.0, 0.5]), 0.2)
    np.testing.assert_allclose(out, [[0.1, -0.05]])


# -------------------------------------------------------------- appearance


def _brute_margins(x):
    vals = [v for v in x if math.isfinite(v)]
    if not vals:
        return np.nan
    if len(vals) == 1:
        return np.inf
    vals.sort(reverse=True)
    return vals[0] - vals[1]


def test_appearance_margins_match_brute_force():
    """top1 - top2 over finite entries; single -> inf; none -> nan."""
    rng = np.random.default_rng(8)
    cos = rng.uniform(-1, 1, size=(5, 4))
    cos[0, 1:] = np.nan  # single finite in row 0
    cos[1] = np.nan  # none in row 1
    cos[3, 2] = np.nan
    rows, cols = appearance_margins(cos)
    np.testing.assert_array_equal(rows, [_brute_margins(r) for r in cos])
    np.testing.assert_array_equal(cols, [_brute_margins(c) for c in cos.T])
    assert rows[0] == np.inf and np.isnan(rows[1])

    r, c = appearance_margins(np.zeros((3, 0)))
    assert r.shape == (3,) and np.isnan(r).all() and c.shape == (0,)
    r, c = appearance_margins(np.array([[0.3, 0.9]]))
    np.testing.assert_allclose(r, [0.6])
    assert (c == np.inf).all()


def _brute_aw(E, aw_param, base):
    n, m = E.shape
    W = np.full((n, m), base)
    for i in range(n):
        s = sorted(E[i], reverse=True)
        W[i] += min(s[0] - s[1], aw_param) / 2
    for j in range(m):
        s = sorted(E[:, j], reverse=True)
        W[:, j] += min(s[0] - s[1], aw_param) / 2
    return W


def test_adaptive_weight():
    """Deep OC-SORT AW per row/col; constant base unless >= 2 rows and cols."""
    rng = np.random.default_rng(9)
    E = rng.uniform(-1, 1, size=(4, 5))
    np.testing.assert_allclose(adaptive_weight(E, 0.5, 0.75), _brute_aw(E, 0.5, 0.75))
    np.testing.assert_allclose(adaptive_weight(E, 0.1, 1.0), _brute_aw(E, 0.1, 1.0))
    for shape in ((1, 4), (4, 1), (1, 1), (0, 3), (3, 0)):
        W = adaptive_weight(rng.uniform(size=shape), 0.5, 0.75)
        assert W.shape == shape and (W == 0.75).all()
    # Bounded by base + aw_param.
    W = adaptive_weight(np.array([[1.0, -1.0], [-1.0, 1.0]]), 0.5, 0.75)
    np.testing.assert_allclose(W, np.full((2, 2), 1.25))
    # Non-finite entries are ignored; < 2 finite -> no bonus for that row/col.
    E = np.array([[0.9, np.nan], [0.1, 0.2]])
    W = adaptive_weight(E, 0.5, 0.75)
    # Row 0: one finite entry -> no row bonus; col 0 margin 0.8 capped at 0.5.
    np.testing.assert_allclose(W[0], 0.75 + np.array([0.25, 0.0]))
    np.testing.assert_allclose(W[1], 0.75 + 0.05 + np.array([0.25, 0.0]))
