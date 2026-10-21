"""Similarity matrices, gates and assignment helpers for the MOT trackers.

Every function here is vectorized and NaN-free on output unless documented.
Similarities are in ``[0, 1]``; costs are ``1 - similarity``. Rows are tracks
(or the first argument), columns are detections (or the second argument).
"""

from typing import Optional, Tuple

import numpy as np

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

# Cost assigned to forbidden pairs before solving (added to max_cost).
INFEASIBLE_COST = 1e5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between ``(n, 4)`` and ``(m, 4)`` ``xyxy`` boxes; NaN boxes score 0."""
    raise NotImplementedError


def ioa_min_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over the smaller area (duplicate detection); NaN boxes score 0."""
    raise NotImplementedError


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
    """
    raise NotImplementedError


def centroid_similarity(c_a: np.ndarray, c_b: np.ndarray, scale: float) -> np.ndarray:
    """Gaussian similarity ``exp(-d^2 / (2 scale^2))`` of ``(n, 2)`` vs ``(m, 2)`` points.

    NaN points score 0.
    """
    raise NotImplementedError


def cosine_matrix(
    e_a: np.ndarray, ok_a: np.ndarray, e_b: np.ndarray, ok_b: np.ndarray
) -> np.ndarray:
    """Cosine similarity of L2-normalized vectors; NaN where either is missing.

    Args:
        e_a: ``(n, D)`` vectors (rows where ``ok_a`` is False are ignored).
        ok_a: ``(n,)`` bool availability.
        e_b: ``(m, D)`` vectors.
        ok_b: ``(m,)`` bool availability.

    Returns:
        ``(n, m)`` cosines in ``[-1, 1]``, NaN where unavailable.
    """
    raise NotImplementedError


def fuse_score(sim: np.ndarray, det_scores: np.ndarray) -> np.ndarray:
    """ByteTrack score fusion: ``sim * det_scores[None, :]``."""
    raise NotImplementedError


def linear_assignment(
    cost: np.ndarray, max_cost: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Minimum-cost assignment with a cost cap.

    Entries above ``max_cost`` or non-finite are replaced by
    ``max_cost + INFEASIBLE_COST`` before solving with
    :func:`scipy.optimize.linear_sum_assignment`; matched pairs with cost above
    ``max_cost`` are then dropped. Empty matrices are handled.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)`` where ``matches`` is an
        ``(k, 2)`` int array of ``(row, col)`` sorted by row, and the unmatched
        arrays are sorted int arrays.
    """
    raise NotImplementedError


def linear_assignment_masked(
    score: np.ndarray, valid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Maximum-score assignment restricted to ``valid`` pairs.

    OC-SORT semantics: validity is decided separately (on geometry alone) and the
    score (geometry + momentum + appearance) only orders the valid pairs.

    Returns:
        ``(matches, unmatched_rows, unmatched_cols)`` as in
        :func:`linear_assignment`; no invalid pair is ever matched.
    """
    raise NotImplementedError


def one_to_one_pattern(valid: np.ndarray) -> Optional[np.ndarray]:
    """OC-SORT/SORT shortcut: the valid pairs if they form a 1:1 pattern.

    Returns:
        ``(k, 2)`` matches when every row and column of ``valid`` has at most one
        True entry (and at least one True overall), else ``None``.
    """
    raise NotImplementedError


def speed_direction(prev_xyxy: np.ndarray, cur_xyxy: np.ndarray) -> np.ndarray:
    """Unit ``(dy, dx)`` direction between box centers, row-wise for ``(n, 4)`` inputs.

    Zero-length motion gives ``(0, 0)``.
    """
    raise NotImplementedError


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

    Returns:
        ``(n, m)`` additive similarity term (may be negative).
    """
    raise NotImplementedError


def adaptive_weight(
    E: np.ndarray, aw_param: float = 0.5, base: float = 0.75
) -> np.ndarray:
    """Deep OC-SORT adaptive weighting of the appearance matrix ``E``.

    ``W = base + 0.5 * min(row_top1 - row_top2, aw_param) + 0.5 * min(col_top1 -
    col_top2, aw_param)``, computed only when ``E`` has at least 2 rows and 2
    columns; otherwise every entry is ``base``.
    """
    raise NotImplementedError


def appearance_margins(cos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-1 minus top-2 cosine per row and per column over finite entries.

    A row/column with a single finite entry has margin ``inf``; with none, ``nan``.
    """
    raise NotImplementedError
