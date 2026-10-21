"""Similarity matrices, gates and assignment helpers for the MOT trackers.

Every function here is vectorized and NaN-free on output unless documented.
Similarities are in ``[0, 1]``; costs are ``1 - similarity``. Rows are tracks
(or the first argument), columns are detections (or the second argument). Every
function accepts empty inputs and returns correctly shaped ``(n, 0)`` /
``(0, m)`` results.
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# Chi-square 0.95 quantiles by degrees of freedom (Mahalanobis gating).
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

# Cost of a gated (forbidden) pair, e.g. DeepSORT's Mahalanobis gate. Any cost
# above `max_cost` is infeasible for `linear_assignment`, so this is just a
# conventional "never" value for trackers to write into their cost matrices; it
# is also the margin that keeps non-finite cells out of an uncapped solve.
INFEASIBLE_COST = 1e5

# What an infeasible pair costs in `linear_assignment`, above `max_cost`: just
# enough to never be preferred over a feasible pair at the cap (DeepSORT's
# `max_distance + 1e-5`).
_CAP_EPS = 1e-5

# OKS normalization epsilon, identical to `sleap_nn.evaluation.compute_oks`.
_OKS_EPS = np.spacing(1)


def _as_boxes(x: np.ndarray) -> np.ndarray:
    """``(n, 4)`` float64 view of box input (a single ``(4,)`` box -> ``(1, 4)``)."""
    return np.asarray(x, dtype=np.float64).reshape(-1, 4)


def _intersection_and_areas(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise intersection areas plus per-box areas and validity.

    Returns:
        ``(inter (n, m), area_a (n,), area_b (m,), ok (n, m))`` with zeros where
        either box has a non-finite coordinate (``ok`` False there).
    """
    ok_a = np.isfinite(a).all(axis=1)
    ok_b = np.isfinite(b).all(axis=1)
    a = np.where(ok_a[:, None], a, 0.0)
    b = np.where(ok_b[:, None], b, 0.0)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(
        a[:, None, 0], b[None, :, 0]
    )
    ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(
        a[:, None, 1], b[None, :, 1]
    )
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    ok = ok_a[:, None] & ok_b[None, :]
    inter = np.where(ok, inter, 0.0)
    return inter, area_a, area_b, ok


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` where ``den > 0``, else 0 (no warnings), clipped to ``[0, 1]``."""
    out = np.zeros(np.broadcast(num, den).shape, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return np.clip(out, 0.0, 1.0)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between ``(n, 4)`` and ``(m, 4)`` ``xyxy`` boxes; NaN boxes score 0.

    Args:
        a: ``(n, 4)`` boxes.
        b: ``(m, 4)`` boxes.

    Returns:
        ``(n, m)`` IoU in ``[0, 1]``. Boxes with a non-finite coordinate, and
        pairs whose union is empty (two zero-area boxes), score 0.
    """
    a, b = _as_boxes(a), _as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    inter, area_a, area_b, _ = _intersection_and_areas(a, b)
    union = area_a[:, None] + area_b[None, :] - inter
    return _safe_ratio(inter, union)


def ioa_min_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the smaller area (duplicate detection); NaN boxes score 0.

    Args:
        a: ``(n, 4)`` ``xyxy`` boxes.
        b: ``(m, 4)`` ``xyxy`` boxes.

    Returns:
        ``(n, m)`` values in ``[0, 1]``: 1 when one box contains the other. A
        pair involving a zero-area or non-finite box scores 0.
    """
    a, b = _as_boxes(a), _as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    inter, area_a, area_b, _ = _intersection_and_areas(a, b)
    return _safe_ratio(inter, np.minimum(area_a[:, None], area_b[None, :]))


def oks_matrix(
    kp_a: np.ndarray, kp_b: np.ndarray, area_b: np.ndarray, sigma
) -> np.ndarray:
    """Object keypoint similarity between two sets of poses.

    Uses the COCO/cocoeval form, identical in constants to
    :func:`sleap_nn.evaluation.compute_oks` with ``use_cocoeval=True``:
    ``e = d^2 / (2 sigma)^2 / (area + eps) / 2`` and ``oks = mean(exp(-e))`` over
    the nodes visible in BOTH poses (symmetric visibility: tracking compares two
    observations). Pairs sharing no visible node score 0.

    Args:
        kp_a: ``(n, K, 2)`` keypoints with NaN for missing nodes.
        kp_b: ``(m, K, 2)`` keypoints with NaN for missing nodes.
        area_b: ``(m,)`` scale areas (px^2) of the ``b`` poses, floored at 1.
        sigma: Scalar or ``(K,)`` per-node standard deviation.

    Returns:
        ``(n, m)`` similarities in ``[0, 1]``.

    Raises:
        ValueError: If the two pose sets have different node counts or
            ``sigma`` is not positive and finite.
    """
    kp_a = np.asarray(kp_a, dtype=np.float64)
    kp_b = np.asarray(kp_b, dtype=np.float64)
    n, m = kp_a.shape[0], kp_b.shape[0]
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)
    if kp_a.ndim != 3 or kp_b.ndim != 3 or kp_a.shape[1:] != kp_b.shape[1:]:
        raise ValueError(
            f"Keypoint arrays must be (n, K, 2) and (m, K, 2), got {kp_a.shape} "
            f"and {kp_b.shape}."
        )
    n_nodes = kp_a.shape[1]
    sigma = np.broadcast_to(np.asarray(sigma, dtype=np.float64), (n_nodes,))
    if not (np.all(np.isfinite(sigma)) and np.all(sigma > 0)):
        raise ValueError(f"OKS sigma must be positive and finite, got {sigma}.")

    # Scale: non-finite areas (an invalid pose) and tiny ones are floored at 1.
    area = np.asarray(area_b, dtype=np.float64).reshape(-1)
    area = np.where(np.isfinite(area), np.maximum(area, 1.0), 1.0)

    d2 = ((kp_a[:, None] - kp_b[None]) ** 2).sum(axis=-1)  # (n, m, K)
    shared = np.isfinite(d2)  # node visible in both poses
    spread = (2.0 * sigma) ** 2  # (K,)
    e = np.where(shared, d2, 0.0) / spread / (area[None, :, None] + _OKS_EPS) / 2.0
    ks = np.where(shared, np.exp(-e), 0.0)
    n_shared = shared.sum(axis=-1)
    oks = ks.sum(axis=-1) / np.maximum(n_shared, 1)
    return np.where(n_shared > 0, np.clip(oks, 0.0, 1.0), 0.0)


def centroid_similarity(c_a: np.ndarray, c_b: np.ndarray, scale: float) -> np.ndarray:
    """Gaussian similarity ``exp(-d^2 / (2 scale^2))`` of ``(n, 2)`` vs ``(m, 2)`` points.

    NaN points score 0.

    Args:
        c_a: ``(n, 2)`` points.
        c_b: ``(m, 2)`` points.
        scale: Gaussian scale in px (``body_size / 2`` for the trackers).

    Returns:
        ``(n, m)`` similarities in ``[0, 1]``.

    Raises:
        ValueError: If ``scale`` is not positive and finite.
    """
    scale = float(scale)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"Centroid similarity scale must be > 0, got {scale}.")
    c_a = np.asarray(c_a, dtype=np.float64).reshape(-1, 2)
    c_b = np.asarray(c_b, dtype=np.float64).reshape(-1, 2)
    if len(c_a) == 0 or len(c_b) == 0:
        return np.zeros((len(c_a), len(c_b)), dtype=np.float64)
    d2 = ((c_a[:, None] - c_b[None]) ** 2).sum(axis=-1)
    ok = np.isfinite(d2)
    return np.where(ok, np.exp(-np.where(ok, d2, 0.0) / (2.0 * scale**2)), 0.0)


def cosine_matrix(
    e_a: np.ndarray, ok_a: np.ndarray, e_b: np.ndarray, ok_b: np.ndarray
) -> np.ndarray:
    """Cosine similarity of L2-normalized vectors; NaN where either is missing.

    Args:
        e_a: ``(n, D)`` vectors (rows where ``ok_a`` is False are ignored), or
            ``None`` (all missing).
        ok_a: ``(n,)`` bool availability.
        e_b: ``(m, D)`` vectors, or ``None``.
        ok_b: ``(m,)`` bool availability.

    Returns:
        ``(n, m)`` cosines in ``[-1, 1]``, NaN where unavailable (also everywhere
        when the two vector dimensions differ).
    """
    ok_a = np.asarray(ok_a, dtype=bool).reshape(-1)
    ok_b = np.asarray(ok_b, dtype=bool).reshape(-1)
    out = np.full((len(ok_a), len(ok_b)), np.nan, dtype=np.float64)
    if out.size == 0 or e_a is None or e_b is None:
        return out
    e_a = np.asarray(e_a, dtype=np.float64)
    e_b = np.asarray(e_b, dtype=np.float64)
    if e_a.ndim != 2 or e_b.ndim != 2 or e_a.shape[1] != e_b.shape[1]:
        return out
    ok_a = ok_a & np.isfinite(e_a).all(axis=1)
    ok_b = ok_b & np.isfinite(e_b).all(axis=1)
    ia, ib = np.flatnonzero(ok_a), np.flatnonzero(ok_b)
    if len(ia) and len(ib):
        out[np.ix_(ia, ib)] = np.clip(e_a[ia] @ e_b[ib].T, -1.0, 1.0)
    return out


def fuse_score(sim: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """ByteTrack score fusion: ``sim * det_scores[None, :]``.

    Args:
        sim: ``(n, m)`` similarities.
        det_scores: ``(m,)`` detection scores in ``[0, 1]``.

    Returns:
        ``(n, m)`` fused similarities.
    """
    sim = np.asarray(sim, dtype=np.float64)
    det_scores = np.asarray(det_scores, dtype=np.float64).reshape(-1)
    return sim * det_scores[None, :]


def _assignment_result(
    rows: np.ndarray, cols: np.ndarray, n: int, m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack kept ``(row, col)`` pairs as ``(matches, unmatched_rows, unmatched_cols)``."""
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    matches = np.stack([rows[order], cols[order]], axis=1).reshape(-1, 2)
    unmatched_rows = np.setdiff1d(np.arange(n, dtype=np.int64), rows)
    unmatched_cols = np.setdiff1d(np.arange(m, dtype=np.int64), cols)
    return matches, unmatched_rows, unmatched_cols


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-cost assignment with a cost cap.

    Entries above ``max_cost`` or non-finite are infeasible: they are replaced by
    ``max_cost + 1e-5`` before solving with
    :func:`scipy.optimize.linear_sum_assignment`, and matched pairs that were
    infeasible are then dropped. Pricing a forbidden pair just above the cap
    (DeepSORT's ``max_distance + 1e-5``; equivalently ByteTrack/BoT-SORT's
    ``lapjv(cost_limit=max_cost)``) means leaving a row and a column unmatched
    costs about ``max_cost``: the solver never pulls a track off a confident
    match just to squeeze in more matches. With an infinite ``max_cost`` only
    non-finite entries are infeasible (priced ``INFEASIBLE_COST`` above the
    largest finite cost). Empty matrices are handled; the result is
    deterministic.

    Args:
        cost: ``(n, m)`` costs (lower is better).
        max_cost: Largest admissible cost of a matched pair (inclusive).

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is an
        ``(k, 2)`` int array of ``(row, col)`` sorted by row, and the unmatched
        arrays are sorted int arrays.

    Raises:
        ValueError: If ``max_cost`` is NaN or ``cost`` is not 2-D.
    """
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError(f"cost must be a 2-D matrix, got shape {cost.shape}.")
    max_cost = float(max_cost)
    if np.isnan(max_cost):
        raise ValueError("max_cost must not be NaN.")
    n, m = cost.shape
    if n == 0 or m == 0:
        return _assignment_result([], [], n, m)

    feasible = np.isfinite(cost) & (cost <= max_cost)
    if not feasible.any():
        return _assignment_result([], [], n, m)
    if np.isfinite(max_cost):
        fill = max_cost + _CAP_EPS
    else:
        fill = float(cost[feasible].max()) + INFEASIBLE_COST
    solve = np.where(feasible, cost, fill)
    rows, cols = linear_sum_assignment(solve)
    keep = feasible[rows, cols]
    return _assignment_result(rows[keep], cols[keep], n, m)


def linear_assignment_masked(
    score: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum-score assignment, keeping only ``valid`` pairs.

    OC-SORT semantics: validity is decided separately (on geometry alone) and the
    score (geometry + momentum + appearance) only orders the pairs. As in the
    OC-SORT / Deep OC-SORT reference, the assignment maximizes the score over the
    WHOLE matrix and pairs failing the validity test are then rejected: an
    invalid pair's own (low) score acts as the value of leaving that row and
    column unmatched, so a confident valid match is not traded for two weaker
    ones. A pair with a non-finite score is never matched.

    Args:
        score: ``(n, m)`` scores (higher is better).
        valid: ``(n, m)`` bool admissibility.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)`` as in
        :func:`linear_assignment`; no invalid pair is ever matched.

    Raises:
        ValueError: If ``score`` and ``valid`` shapes differ or are not 2-D.
    """
    score = np.asarray(score, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if score.ndim != 2 or score.shape != valid.shape:
        raise ValueError(
            f"score and valid must be 2-D with equal shapes, got {score.shape} "
            f"and {valid.shape}."
        )
    n, m = score.shape
    if n == 0 or m == 0:
        return _assignment_result([], [], n, m)
    valid = valid & np.isfinite(score)
    if not valid.any():
        return _assignment_result([], [], n, m)
    finite = np.isfinite(score)
    floor = float(score[finite].min()) - INFEASIBLE_COST
    rows, cols = linear_sum_assignment(-np.where(finite, score, floor))
    keep = valid[rows, cols]
    return _assignment_result(rows[keep], cols[keep], n, m)


def one_to_one_pattern(valid: np.ndarray) -> Optional[np.ndarray]:
    """OC-SORT/SORT shortcut: the valid pairs if they form a 1:1 pattern.

    Args:
        valid: ``(n, m)`` bool admissibility.

    Returns:
        ``(k, 2)`` matches (sorted by row) when every row and column of ``valid``
        has at most one True entry (and at least one True overall), else
        ``None``.
    """
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or valid.size == 0 or not valid.any():
        return None
    if valid.sum(axis=1).max() > 1 or valid.sum(axis=0).max() > 1:
        return None
    return np.argwhere(valid).astype(np.int64)


def _unit_directions(dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Stack ``(dy, dx)`` into unit vectors; zero-length or non-finite -> ``(0, 0)``."""
    norm = np.hypot(dy, dx)
    ok = np.isfinite(norm) & (norm > 0)
    safe = np.where(ok, norm, 1.0)
    return np.stack(
        [np.where(ok, dy / safe, 0.0), np.where(ok, dx / safe, 0.0)], axis=-1
    )


def speed_direction(prev_xyxy: np.ndarray, cur_xyxy: np.ndarray) -> np.ndarray:
    """Unit ``(dy, dx)`` direction between box centers, row-wise for ``(n, 4)`` inputs.

    Zero-length motion gives ``(0, 0)``.

    Args:
        prev_xyxy: ``(n, 4)`` (or ``(4,)``) earlier boxes.
        cur_xyxy: ``(n, 4)`` (or ``(4,)``) later boxes.

    Returns:
        ``(n, 2)`` (or ``(2,)`` for single boxes) unit directions from the
        earlier box center to the later one; ``(0, 0)`` for zero-length motion
        or non-finite boxes.
    """
    single = np.ndim(prev_xyxy) == 1 and np.ndim(cur_xyxy) == 1
    prev = _as_boxes(prev_xyxy)
    cur = _as_boxes(cur_xyxy)
    dy = 0.5 * (cur[:, 1] + cur[:, 3]) - 0.5 * (prev[:, 1] + prev[:, 3])
    dx = 0.5 * (cur[:, 0] + cur[:, 2]) - 0.5 * (prev[:, 0] + prev[:, 2])
    out = _unit_directions(dy, dx)
    return out[0] if single else out


def ocm_matrix(
    track_dirs: np.ndarray,
    track_prev_obs: np.ndarray,
    track_has_dir: np.ndarray,
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    inertia: float,
) -> np.ndarray:
    """OC-SORT observation-centric momentum term.

    For track ``i`` with unit velocity direction ``track_dirs[i]`` (``(dy, dx)``,
    from an observation ``delta_t`` steps back to its last observation) and the
    direction from ``track_prev_obs[i]`` (that earlier observation) to each
    detection: ``theta = |arccos(clip(dot, -1, 1))|`` and
    ``ocm = inertia * ((pi/2 - theta) / pi) * has_dir[:, None] * det_scores[None, :]``.

    A detection centered exactly on the earlier observation (no direction) has
    ``dot = 0`` and contributes 0, as in the reference.

    Args:
        track_dirs: ``(n, 2)`` unit ``(dy, dx)`` track directions.
        track_prev_obs: ``(n, 4)`` ``xyxy`` earlier observations.
        track_has_dir: ``(n,)`` bool: the track has a direction (and an earlier
            observation).
        det_boxes: ``(m, 4)`` ``xyxy`` detection boxes.
        det_scores: ``(m,)`` detection scores.
        inertia: Weight of the term (OC-SORT: 0.2).

    Returns:
        ``(n, m)`` additive similarity term (may be negative), in
        ``[-inertia/2, inertia/2]`` for scores in ``[0, 1]``.
    """
    track_dirs = np.asarray(track_dirs, dtype=np.float64).reshape(-1, 2)
    prev = _as_boxes(track_prev_obs)
    dets = _as_boxes(det_boxes)
    n, m = len(track_dirs), len(dets)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float64)
    has_dir = np.asarray(track_has_dir, dtype=bool).reshape(-1)
    has_dir = has_dir & np.isfinite(track_dirs).all(axis=1)
    track_dirs = np.where(has_dir[:, None], track_dirs, 0.0)
    scores = np.asarray(det_scores, dtype=np.float64).reshape(-1)
    scores = np.where(np.isfinite(scores), scores, 0.0)

    # Direction from each track's earlier observation to each detection (dy, dx).
    prev_cy = 0.5 * (prev[:, 1] + prev[:, 3])
    prev_cx = 0.5 * (prev[:, 0] + prev[:, 2])
    det_cy = 0.5 * (dets[:, 1] + dets[:, 3])
    det_cx = 0.5 * (dets[:, 0] + dets[:, 2])
    to_det = _unit_directions(
        det_cy[None, :] - prev_cy[:, None], det_cx[None, :] - prev_cx[:, None]
    )  # (n, m, 2)

    cos = np.clip((track_dirs[:, None, :] * to_det).sum(axis=-1), -1.0, 1.0)
    theta = np.abs(np.arccos(cos))
    term = (np.pi / 2.0 - theta) / np.pi
    return float(inertia) * term * has_dir[:, None] * scores[None, :]


def adaptive_weight(
    E: np.ndarray, aw_param: float = 0.5, base: float = 0.75
) -> np.ndarray:
    """Deep OC-SORT adaptive weighting of the appearance matrix ``E``.

    ``W = base + 0.5 * min(row_top1 - row_top2, aw_param) + 0.5 * min(col_top1 -
    col_top2, aw_param)``, computed only when ``E`` has at least 2 rows and 2
    columns; otherwise every entry is ``base``.

    Non-finite entries of ``E`` are ignored when ranking; a row or column with
    fewer than two finite entries gets no bonus.

    Args:
        E: ``(n, m)`` appearance similarities.
        aw_param: Cap on each margin bonus (Deep OC-SORT ``aw_param``).
        base: Constant weight (Deep OC-SORT ``w_assoc_emb``).

    Returns:
        ``(n, m)`` weights.
    """
    E = np.asarray(E, dtype=np.float64)
    W = np.full(E.shape, float(base), dtype=np.float64)
    if E.ndim != 2 or E.shape[0] < 2 or E.shape[1] < 2:
        return W
    row_margin, col_margin = appearance_margins(E)
    row_bonus = np.where(np.isfinite(row_margin), row_margin, 0.0)
    col_bonus = np.where(np.isfinite(col_margin), col_margin, 0.0)
    row_bonus = 0.5 * np.minimum(row_bonus, aw_param)
    col_bonus = 0.5 * np.minimum(col_bonus, aw_param)
    return W + row_bonus[:, None] + col_bonus[None, :]


def _top2_margin(x: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 over finite entries along the last axis (inf / nan rules)."""
    finite = np.isfinite(x)
    count = finite.sum(axis=-1)
    if x.shape[-1] == 0:
        return np.full(x.shape[:-1], np.nan, dtype=np.float64)
    filled = np.sort(np.where(finite, x, -np.inf), axis=-1)
    top1 = filled[..., -1]
    top2 = filled[..., -2] if x.shape[-1] >= 2 else np.full_like(top1, -np.inf)
    margin = np.full(x.shape[:-1], np.nan, dtype=np.float64)
    margin[count == 1] = np.inf
    two = count >= 2
    margin[two] = top1[two] - top2[two]
    return margin


def appearance_margins(cos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-1 minus top-2 cosine per row and per column over finite entries.

    A row/column with a single finite entry has margin ``inf``; with none, ``nan``.

    Args:
        cos: ``(n, m)`` cosines (NaN where unavailable).

    Returns:
        ``(row_margin (n,), col_margin (m,))``.
    """
    cos = np.asarray(cos, dtype=np.float64)
    if cos.ndim != 2:
        raise ValueError(f"cos must be a 2-D matrix, got shape {cos.shape}.")
    return _top2_margin(cos), _top2_margin(cos.T)
