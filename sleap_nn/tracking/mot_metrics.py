"""TrackEval-faithful multi-object tracking metrics: CLEAR-MOT, Identity and HOTA.

A pure numpy/scipy core (no sleap-io): it scores sequences of per-timestep
ground-truth ids, predicted ids and a ground-truth x prediction similarity
matrix. Turning ``sio.Labels`` into such sequences -- matching, don't-care ground
truth, frame alignment -- is the job of
:func:`sleap_nn.evaluation.compute_mot_metrics`.

The numbers reproduce TrackEval 1.3.0 (Luiten & Hoffhues; the reference
implementation of HOTA, and the MOTChallenge / KITTI evaluation code) exactly,
including its documented quirks, so they can be quoted next to published MOT
results. Output keys use TrackEval's own field names (``MOTA``, ``IDF1``,
``HOTA``, ``CLR_TP``, ``HOTA(0)``, ...).

Three metric families are computed:

- **CLEAR-MOT** (Bernardin & Stiefelhagen 2008): per-timestep Hungarian
  matching at ``similarity >= threshold`` that first keeps the previous
  timestep's matches (the "continuity bonus"), then maximizes similarity. Gives
  TP / FN / FP, ID switches, MOTA / MOTP / MODA, mostly-tracked / partly-tracked
  / mostly-lost and fragmentations.
- **Identity** (Ristani et al. 2016): one global assignment between
  ground-truth and predicted identities, maximizing the number of timesteps on
  which the pair *could* match (``similarity >= threshold``; every such pair
  counts, not only a one-to-one per-timestep matching). Gives IDTP / IDFN /
  IDFP and IDF1 / IDP / IDR.
- **HOTA** (Luiten et al. 2021): detection, association and localization
  accuracy over 19 localization thresholds ``alpha = 0.05, 0.10, ..., 0.95``
  (:data:`HOTA_ALPHAS`). Like TrackEval -- and unlike the paper's Eq. 15 -- one
  alpha-independent Hungarian matching per timestep maximizes ``global
  alignment score x similarity``, and that single matching is then thresholded
  at each alpha. Array fields hold one value per alpha; the headline number
  quoted in papers is their mean.

TrackEval behaviors reproduced on purpose (each one changes a number):

- **Continuity bonus looks one timestep back; ID switches look any distance
  back.** Matching prefers the prediction matched to the same ground truth at
  the immediately previous timestep, while a switch is scored against the last
  prediction a ground truth was matched to, however long ago.
- **Fragmentation (``Frag``)** is, per ground-truth id, the number of tracked
  segments minus one. A timestep with no prediction at all does NOT end a
  segment (TrackEval skips such timesteps without resetting the continuity
  state), while a timestep with only unrelated predictions does: a one-frame
  gap with nothing predicted gives ``Frag = 0``, the same gap holding a single
  false positive gives ``Frag = 1``.
- **Mostly tracked is strictly above 80 %** of a trajectory's timesteps;
  mostly lost is below 20 %.
- **``LocA`` is 1 where there are no true positives** (TrackEval divides
  ``max(1e-10, sum) / max(1e-10, TP)``).
- **``HOTA(0)`` / ``LocA(0)`` are the values at alpha = 0.05**, the first
  threshold -- not at alpha = 0.
- **Empty sequences short-circuit.** A sequence with no predictions (or no
  ground truth) returns only the counts -- all ground truth missed (or all
  predictions false) -- with every derived field left at 0 except ``MLR = 1``
  and ``LocA = 1``, and ``CLR_Frames = 0``. :func:`combine_sequences`
  recomputes the derived fields from the summed counts, as TrackEval's
  ``COMBINED_SEQ`` row does.

Ids must be unique within a timestep. They need not be contiguous: every
metric relabels them to ``0..n-1`` first (:func:`relabel_contiguous`).
"""

from typing import Any, Dict, List, Sequence, Tuple

import attrs
import numpy as np
from scipy.optimize import linear_sum_assignment

# TrackEval compares similarities against thresholds with one float epsilon of
# slack (`np.finfo("float").eps`), so a similarity of exactly 0.5 matches at 0.5.
EPS = float(np.finfo(float).eps)

# HOTA's localization thresholds -- TrackEval's exact array (`np.arange`, not
# `np.linspace`, whose last digits differ).
HOTA_ALPHAS = np.arange(0.05, 0.99, 0.05)

# Field names, as TrackEval spells them.
CLEAR_INTEGER_FIELDS = (
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "IDSW",
    "MT",
    "PT",
    "ML",
    "Frag",
    "CLR_Frames",
)
CLEAR_FLOAT_FIELDS = (
    "MOTA",
    "MOTP",
    "MODA",
    "CLR_Re",
    "CLR_Pr",
    "MTR",
    "PTR",
    "MLR",
    "sMOTA",
    "CLR_F1",
    "FP_per_frame",
    "MOTAL",
    "MOTP_sum",
)
ID_INTEGER_FIELDS = ("IDTP", "IDFN", "IDFP")
ID_FLOAT_FIELDS = ("IDF1", "IDR", "IDP")
HOTA_COUNT_FIELDS = ("HOTA_TP", "HOTA_FN", "HOTA_FP")
HOTA_ARRAY_FIELDS = (
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "LocA",
    "OWTA",
)
HOTA_SCALAR_FIELDS = ("HOTA(0)", "LocA(0)", "HOTALocA(0)")

# Coverage cuts for mostly-tracked / mostly-lost (TrackEval's fixed values).
_MT_RATIO = 0.8
_ML_RATIO = 0.2

# Weight of the continuity bonus: larger than any similarity, so keeping the
# previous timestep's match always wins over a better-scoring alternative.
_CONTINUITY_BONUS = 1000.0


def _ids_array(ids: Any, side: str, t: int) -> np.ndarray:
    """Coerce one timestep's ids to a 1-D int64 array, rejecting non-integers."""
    arr = np.asarray(ids)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(
            f"timestep {t}: {side} ids must be a 1-D array, got shape {arr.shape}."
        )
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(
            f"timestep {t}: {side} ids must be integers, got dtype {arr.dtype}."
        )
    return arr.astype(np.int64, copy=False)


def _check_unique(ids: np.ndarray, side: str, t: int) -> None:
    """Raise if an id occurs twice in one timestep (TrackEval refuses too)."""
    if len(ids) < 2:
        return
    values, counts = np.unique(ids, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"timestep {t}: duplicate {side} id(s) {values[counts > 1].tolist()} -- "
            "an id may occur at most once per timestep."
        )


@attrs.define
class MOTSequence:
    """One scored sequence: per-timestep ids and similarities.

    A sequence is normally one video. Under global identities several videos
    are concatenated into one sequence, with ``resets`` marking where each new
    video starts.

    Construction normalizes and validates every timestep: ids become 1-D int64
    arrays, similarities ``(n_gt, n_pred)`` float arrays (an empty similarity
    may be given in any shape), and ``resets`` a sorted tuple of unique ints.

    Attributes:
        gt_ids: One integer array of ground-truth ids per timestep. Ids are
            arbitrary integers, unique within a timestep.
        pred_ids: One integer array of predicted ids per timestep, same rules.
        similarity: One ``(len(gt_ids[t]), len(pred_ids[t]))`` array per
            timestep: the similarity of every ground-truth/predicted pair, in
            ``[0, 1]`` (OKS, IoU, ...). Must be finite.
        resets: Timestep indices at which CLEAR-MOT's frame-to-frame state is
            reset -- video boundaries when several videos are scored as one
            sequence. At a reset the continuity bonus and the ID-switch memory
            are cleared (a ground truth's first match in the new video is never
            a switch), and fragmentation is counted per side of the boundary
            (a trajectory's first segment after a reset is not a fragment).
            Identity and HOTA do not depend on timestep order and ignore it.

    Raises:
        ValueError: If the three lists differ in length, a similarity matrix
            has the wrong shape or a non-finite value, ids are not integers, an
            id occurs twice in one timestep, or a reset is out of range.
    """

    gt_ids: List[np.ndarray]
    pred_ids: List[np.ndarray]
    similarity: List[np.ndarray]
    resets: Tuple[int, ...] = ()

    def __attrs_post_init__(self):
        """Normalize and validate every timestep."""
        n = len(self.gt_ids)
        if len(self.pred_ids) != n or len(self.similarity) != n:
            raise ValueError(
                "gt_ids, pred_ids and similarity must have one entry per timestep, "
                f"got {n}, {len(self.pred_ids)} and {len(self.similarity)}."
            )
        gt_ids, pred_ids, similarity = [], [], []
        for t in range(n):
            g = _ids_array(self.gt_ids[t], "ground-truth", t)
            p = _ids_array(self.pred_ids[t], "predicted", t)
            _check_unique(g, "ground-truth", t)
            _check_unique(p, "predicted", t)
            s = np.asarray(self.similarity[t], dtype=float)
            if s.size == 0 and len(g) * len(p) == 0:
                s = np.zeros((len(g), len(p)))
            elif s.shape != (len(g), len(p)):
                raise ValueError(
                    f"timestep {t}: similarity has shape {s.shape}, expected "
                    f"({len(g)}, {len(p)}) for {len(g)} ground-truth and "
                    f"{len(p)} predicted ids."
                )
            if not np.isfinite(s).all():
                raise ValueError(
                    f"timestep {t}: similarity has non-finite values; map NaN to 0 "
                    "before scoring."
                )
            gt_ids.append(g)
            pred_ids.append(p)
            similarity.append(s)
        resets = tuple(sorted({int(r) for r in self.resets}))
        if resets and (resets[0] < 0 or resets[-1] >= max(n, 1)):
            raise ValueError(
                f"resets must be timestep indices in [0, {n}), got {list(resets)}."
            )
        self.gt_ids = gt_ids
        self.pred_ids = pred_ids
        self.similarity = similarity
        self.resets = resets


def relabel_contiguous(seq: MOTSequence) -> MOTSequence:
    """Relabel ground-truth and predicted ids to contiguous ``0..n-1``.

    Each side is relabeled independently, in sorted order of its original ids.
    Every metric needs this (TrackEval's datasets do it while loading): the
    per-id tables are indexed by id, so a gap in the ids would create a phantom
    trajectory -- counted as mostly lost, and a 0/0 in HOTA's alignment score.

    Args:
        seq: The sequence to relabel.

    Returns:
        A new :class:`MOTSequence` with the same similarities and resets (the
        similarity arrays are shared, not copied).
    """
    empty = np.zeros(0, dtype=np.int64)
    gt_unique = np.unique(np.concatenate(list(seq.gt_ids) + [empty]))
    pred_unique = np.unique(np.concatenate(list(seq.pred_ids) + [empty]))
    return MOTSequence(
        gt_ids=[np.searchsorted(gt_unique, g) for g in seq.gt_ids],
        pred_ids=[np.searchsorted(pred_unique, p) for p in seq.pred_ids],
        similarity=list(seq.similarity),
        resets=seq.resets,
    )


def _num_ids(ids: Sequence[np.ndarray]) -> int:
    """Number of ids on one side of a contiguously relabeled sequence."""
    return 1 + max([int(x.max()) for x in ids if len(x)] + [-1])


def _num_dets(ids: Sequence[np.ndarray]) -> int:
    """Total number of detections on one side of a sequence."""
    return int(sum(len(x) for x in ids))


# ---------------------------------------------------------------------------
# CLEAR-MOT
# ---------------------------------------------------------------------------
def _clear_final_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    """Fill CLEAR's derived fields from its counts (per sequence and combined)."""
    tp, fn, fp, idsw = res["CLR_TP"], res["CLR_FN"], res["CLR_FP"], res["IDSW"]
    n_ids = max(1.0, res["MT"] + res["PT"] + res["ML"])
    n_gt_dets = max(1.0, tp + fn)
    res["MTR"] = res["MT"] / n_ids
    res["PTR"] = res["PT"] / n_ids
    res["MLR"] = res["ML"] / n_ids
    res["CLR_Re"] = tp / n_gt_dets
    res["CLR_Pr"] = tp / max(1.0, tp + fp)
    res["MODA"] = (tp - fp) / n_gt_dets
    res["MOTA"] = (tp - fp - idsw) / n_gt_dets
    res["MOTP"] = res["MOTP_sum"] / max(1.0, tp)
    res["sMOTA"] = (res["MOTP_sum"] - fp - idsw) / n_gt_dets
    res["CLR_F1"] = tp / max(1.0, tp + 0.5 * fn + 0.5 * fp)
    res["FP_per_frame"] = fp / max(1.0, res["CLR_Frames"])
    # MOTA with log-damped switches; TrackEval leaves IDSW = 0 as 0.
    log_idsw = float(np.log10(idsw)) if idsw > 0 else 0.0
    res["MOTAL"] = (tp - fp - log_idsw) / n_gt_dets
    return res


def _closed_fragments(segment_starts: np.ndarray) -> int:
    """Fragmentations from per-trajectory segment counts: segments - 1 each."""
    started = segment_starts[segment_starts > 0]
    return int(np.sum(started - 1))


def _clear(seq: MOTSequence, threshold: float) -> Dict[str, Any]:
    """CLEAR-MOT on an already relabeled sequence."""
    res: Dict[str, Any] = {f: 0 for f in CLEAR_INTEGER_FIELDS}
    res.update({f: 0.0 for f in CLEAR_FLOAT_FIELDS})
    n_gt_ids = _num_ids(seq.gt_ids)
    n_gt_dets, n_pred_dets = _num_dets(seq.gt_ids), _num_dets(seq.pred_ids)

    # TrackEval's empty-sequence shortcut: counts only, derived fields left at
    # 0 (MLR at 1), CLR_Frames included.
    if n_pred_dets == 0:
        res.update(CLR_FN=n_gt_dets, ML=n_gt_ids, MLR=1.0)
        return res
    if n_gt_dets == 0:
        res.update(CLR_FP=n_pred_dets, MLR=1.0)
        return res

    tp = fn = fp = idsw = frag = 0
    motp_sum = 0.0
    gt_seen = np.zeros(n_gt_ids)  # timesteps each ground truth is present
    gt_hit = np.zeros(n_gt_ids)  # ...and matched
    segment_starts = np.zeros(n_gt_ids)  # tracked segments begun, per gt id
    # The prediction each ground truth was last matched to, however long ago
    # (scores ID switches), and the one it was matched to at the previous
    # matched timestep (the continuity bonus). NaN = none.
    last_match = np.full(n_gt_ids, np.nan)
    prev_match = np.full(n_gt_ids, np.nan)
    resets = set(seq.resets)

    for t, (g, p, s) in enumerate(zip(seq.gt_ids, seq.pred_ids, seq.similarity)):
        if t in resets:
            # A new video: no continuity bonus or switch across the boundary,
            # and fragments are counted per side of it.
            last_match[:] = np.nan
            prev_match[:] = np.nan
            frag += _closed_fragments(segment_starts)
            segment_starts[:] = 0
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            # TrackEval quirk (kept): continuity is NOT broken by a timestep
            # with no predictions at all -- see the module docstring.
            fn += len(g)
            gt_seen[g] += 1
            continue

        # Keep last timestep's pairs first, then maximize similarity; pairs
        # below the threshold can never match.
        score = s + _CONTINUITY_BONUS * (p[None, :] == prev_match[g][:, None])
        score[s < threshold - EPS] = 0.0
        rows, cols = linear_sum_assignment(-score)
        keep = score[rows, cols] > EPS
        rows, cols = rows[keep], cols[keep]
        matched_gt, matched_pred = g[rows], p[cols]

        before = last_match[matched_gt]
        idsw += int(np.count_nonzero(~np.isnan(before) & (before != matched_pred)))

        gt_seen[g] += 1
        gt_hit[matched_gt] += 1
        was_unmatched = np.isnan(prev_match)
        last_match[matched_gt] = matched_pred
        prev_match[:] = np.nan
        prev_match[matched_gt] = matched_pred
        segment_starts += was_unmatched & ~np.isnan(prev_match)

        tp += len(rows)
        fn += len(g) - len(rows)
        fp += len(p) - len(rows)
        motp_sum += float(s[rows, cols].sum())

    frag += _closed_fragments(segment_starts)
    seen = gt_seen > 0
    coverage = gt_hit[seen] / gt_seen[seen]
    mt = int(np.count_nonzero(coverage > _MT_RATIO))
    pt = int(np.count_nonzero(coverage >= _ML_RATIO)) - mt
    res.update(
        CLR_TP=tp,
        CLR_FN=fn,
        CLR_FP=fp,
        IDSW=idsw,
        MT=mt,
        PT=pt,
        ML=n_gt_ids - mt - pt,
        Frag=frag,
        CLR_Frames=len(seq.gt_ids),
        MOTP_sum=motp_sum,
    )
    return _clear_final_fields(res)


def clear_metrics(seq: MOTSequence, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute the CLEAR-MOT metrics of one sequence, as TrackEval does.

    Args:
        seq: The sequence to score. Ids are relabeled contiguously first.
        threshold: Minimum similarity for a ground-truth/predicted pair to
            match (TrackEval's ``THRESHOLD``, applied with one float epsilon of
            slack).

    Returns:
        Dict keyed by TrackEval's field names: the integer counts
        ``CLR_TP``, ``CLR_FN``, ``CLR_FP``, ``IDSW``, ``MT``, ``PT``, ``ML``,
        ``Frag``, ``CLR_Frames`` and the floats ``MOTA``, ``MOTP``, ``MODA``,
        ``sMOTA``, ``CLR_Re``, ``CLR_Pr``, ``CLR_F1``, ``MTR``, ``PTR``,
        ``MLR``, ``FP_per_frame``, ``MOTAL``, ``MOTP_sum``.
    """
    return _clear(relabel_contiguous(seq), float(threshold))


# ---------------------------------------------------------------------------
# Identity (IDF1)
# ---------------------------------------------------------------------------
def _id_final_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    """Fill IDF1 / IDP / IDR from the identity counts."""
    idtp, idfn, idfp = res["IDTP"], res["IDFN"], res["IDFP"]
    res["IDR"] = idtp / max(1.0, idtp + idfn)
    res["IDP"] = idtp / max(1.0, idtp + idfp)
    res["IDF1"] = idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)
    return res


def _identity(seq: MOTSequence, threshold: float) -> Dict[str, Any]:
    """Identity metrics on an already relabeled sequence."""
    res: Dict[str, Any] = {f: 0 for f in ID_INTEGER_FIELDS}
    res.update({f: 0.0 for f in ID_FLOAT_FIELDS})
    n_gt_dets, n_pred_dets = _num_dets(seq.gt_ids), _num_dets(seq.pred_ids)
    if n_pred_dets == 0:
        res["IDFN"] = n_gt_dets
        return res
    if n_gt_dets == 0:
        res["IDFP"] = n_pred_dets
        return res

    n_gt_ids, n_pred_ids = _num_ids(seq.gt_ids), _num_ids(seq.pred_ids)
    # Timesteps on which each (gt, pred) identity pair COULD be matched: every
    # pair at or above the threshold counts, not a one-to-one matching. Note
    # `>=` with no epsilon here, unlike CLEAR and HOTA (as in TrackEval).
    could_match = np.zeros((n_gt_ids, n_pred_ids))
    gt_len = np.zeros(n_gt_ids)
    pred_len = np.zeros(n_pred_ids)
    for g, p, s in zip(seq.gt_ids, seq.pred_ids, seq.similarity):
        rows, cols = np.nonzero(s >= threshold)
        # Ids are unique per timestep, so these (gt, pred) pairs are distinct
        # and buffered fancy-index addition is exact.
        could_match[g[rows], p[cols]] += 1
        gt_len[g] += 1
        pred_len[p] += 1

    # TrackEval minimizes IDFN + IDFP over an augmented (G+T)^2 matrix that
    # lets any identity stay unassigned. Both totals are fixed minus twice the
    # assigned co-occurrence, so this is the same optimum as a plain
    # max-weight rectangular assignment.
    rows, cols = linear_sum_assignment(-could_match)
    idtp = int(round(float(could_match[rows, cols].sum())))
    res.update(
        IDTP=idtp,
        IDFN=int(gt_len.sum()) - idtp,
        IDFP=int(pred_len.sum()) - idtp,
    )
    return _id_final_fields(res)


def id_metrics(seq: MOTSequence, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute the identity metrics (IDF1, Ristani et al.) of one sequence.

    Named to avoid clashing with :func:`sleap_nn.evaluation.identity_metrics`,
    the older sleap-nn definitions (which differ from TrackEval, see
    :class:`sleap_nn.evaluation.IdentityMetrics`).

    Args:
        seq: The sequence to score. Ids are relabeled contiguously first.
        threshold: Minimum similarity for a pair to count as a potential match.

    Returns:
        Dict with the integer counts ``IDTP``, ``IDFN``, ``IDFP`` and the floats
        ``IDF1``, ``IDP``, ``IDR``.
    """
    return _identity(relabel_contiguous(seq), float(threshold))


# ---------------------------------------------------------------------------
# HOTA
# ---------------------------------------------------------------------------
def _hota_final_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    """Fill HOTA's derived fields from its per-alpha counts and AssA/LocA."""
    tp, fn, fp = res["HOTA_TP"], res["HOTA_FN"], res["HOTA_FP"]
    res["DetRe"] = tp / np.maximum(1.0, tp + fn)
    res["DetPr"] = tp / np.maximum(1.0, tp + fp)
    res["DetA"] = tp / np.maximum(1.0, tp + fn + fp)
    res["HOTA"] = np.sqrt(res["DetA"] * res["AssA"])
    res["OWTA"] = np.sqrt(res["DetRe"] * res["AssA"])
    res["HOTA(0)"] = float(res["HOTA"][0])
    res["LocA(0)"] = float(res["LocA"][0])
    res["HOTALocA(0)"] = res["HOTA(0)"] * res["LocA(0)"]
    return res


def _hota(seq: MOTSequence, alphas: np.ndarray) -> Dict[str, Any]:
    """HOTA on an already relabeled sequence."""
    n_alphas = len(alphas)
    res: Dict[str, Any] = {
        f: np.zeros(n_alphas) for f in HOTA_ARRAY_FIELDS + HOTA_COUNT_FIELDS
    }
    res.update({f: 0.0 for f in HOTA_SCALAR_FIELDS})
    n_gt_dets, n_pred_dets = _num_dets(seq.gt_ids), _num_dets(seq.pred_ids)
    if n_pred_dets == 0 or n_gt_dets == 0:
        if n_pred_dets == 0:
            res["HOTA_FN"] = np.full(n_alphas, float(n_gt_dets))
        else:
            res["HOTA_FP"] = np.full(n_alphas, float(n_pred_dets))
        res["LocA"] = np.ones(n_alphas)
        res["LocA(0)"] = 1.0
        return res

    n_gt_ids, n_pred_ids = _num_ids(seq.gt_ids), _num_ids(seq.pred_ids)
    gt_len = np.zeros((n_gt_ids, 1))
    pred_len = np.zeros((1, n_pred_ids))

    # Pass 1: the global alignment score between identities -- a soft,
    # matching-free Jaccard. Each timestep contributes, per pair, its similarity
    # normalized by the similarity mass of its row and column.
    overlap = np.zeros((n_gt_ids, n_pred_ids))
    for g, p, s in zip(seq.gt_ids, seq.pred_ids, seq.similarity):
        mass = s.sum(axis=0)[None, :] + s.sum(axis=1)[:, None] - s
        share = np.zeros_like(s)
        ok = mass > EPS
        share[ok] = s[ok] / mass[ok]
        overlap[g[:, None], p[None, :]] += share
        gt_len[g, 0] += 1
        pred_len[0, p] += 1
    alignment = overlap / (gt_len + pred_len - overlap)

    # Pass 2: one matching per timestep, thresholded at every alpha.
    tp = np.zeros(n_alphas)
    fn = np.zeros(n_alphas)
    fp = np.zeros(n_alphas)
    loc_sum = np.zeros(n_alphas)
    matches = np.zeros((n_alphas, n_gt_ids, n_pred_ids))
    for g, p, s in zip(seq.gt_ids, seq.pred_ids, seq.similarity):
        if len(g) == 0:
            fp += len(p)
            continue
        if len(p) == 0:
            fn += len(g)
            continue
        rows, cols = linear_sum_assignment(-(alignment[g[:, None], p[None, :]] * s))
        matched_sim = s[rows, cols]
        ok = matched_sim[None, :] >= alphas[:, None] - EPS  # (n_alphas, n_matched)
        n_ok = ok.sum(axis=1)
        tp += n_ok
        fn += len(g) - n_ok
        fp += len(p) - n_ok
        loc_sum += (ok * matched_sim[None, :]).sum(axis=1)
        alpha_idx, k = np.nonzero(ok)
        matches[alpha_idx, g[rows[k]], p[cols[k]]] += 1

    # Association: per matched identity pair, its Jaccard (and recall /
    # precision) over the trajectories, weighted by the pair's matches.
    tp_safe = np.maximum(1.0, tp)
    union = np.maximum(1.0, gt_len[None] + pred_len[None] - matches)
    res["AssA"] = (matches * matches / union).sum(axis=(1, 2)) / tp_safe
    res["AssRe"] = (matches * matches / np.maximum(1.0, gt_len[None])).sum(
        axis=(1, 2)
    ) / tp_safe
    res["AssPr"] = (matches * matches / np.maximum(1.0, pred_len[None])).sum(
        axis=(1, 2)
    ) / tp_safe
    res["LocA"] = np.maximum(1e-10, loc_sum) / np.maximum(1e-10, tp)
    res["HOTA_TP"], res["HOTA_FN"], res["HOTA_FP"] = tp, fn, fp
    return _hota_final_fields(res)


def _check_alphas(alphas: Any) -> np.ndarray:
    """Validate HOTA's localization thresholds."""
    alphas = np.asarray(alphas, dtype=float).reshape(-1)
    if alphas.size == 0:
        raise ValueError("alphas must hold at least one localization threshold.")
    return alphas


def hota_metrics(seq: MOTSequence, alphas: Any = HOTA_ALPHAS) -> Dict[str, Any]:
    """Compute the HOTA metrics of one sequence, as TrackEval does.

    Args:
        seq: The sequence to score. Ids are relabeled contiguously first.
        alphas: Localization thresholds. Default :data:`HOTA_ALPHAS`, the 19
            values ``0.05, 0.10, ..., 0.95`` every published HOTA uses.

    Returns:
        Dict with one value per alpha (float arrays) for ``HOTA``, ``DetA``,
        ``AssA``, ``DetRe``, ``DetPr``, ``AssRe``, ``AssPr``, ``LocA``,
        ``OWTA`` and the counts ``HOTA_TP``, ``HOTA_FN``, ``HOTA_FP``; plus the
        floats ``HOTA(0)``, ``LocA(0)`` (the values at the FIRST alpha, 0.05 by
        default) and ``HOTALocA(0)`` (their product). Quote the mean over
        alphas of an array field as the headline value.
    """
    return _hota(relabel_contiguous(seq), _check_alphas(alphas))


def mot_metrics(seq: MOTSequence, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute CLEAR-MOT, Identity and HOTA for one sequence.

    Args:
        seq: The sequence to score.
        threshold: Match threshold for CLEAR-MOT and Identity. HOTA has its
            own thresholds (:data:`HOTA_ALPHAS`).

    Returns:
        The union of :func:`clear_metrics`, :func:`id_metrics` and
        :func:`hota_metrics`.
    """
    seq = relabel_contiguous(seq)
    threshold = float(threshold)
    res = _clear(seq, threshold)
    res.update(_identity(seq, threshold))
    res.update(_hota(seq, HOTA_ALPHAS))
    return res


# ---------------------------------------------------------------------------
# Combining sequences
# ---------------------------------------------------------------------------
def _has_block(per_seq: Sequence[Dict[str, Any]], key: str) -> bool:
    """Whether every result holds a metric family (keyed by one of its fields)."""
    present = [key in res for res in per_seq]
    if any(present) and not all(present):
        raise ValueError(
            f"Some results have {key!r} and some do not; combine results computed "
            "with the same metric functions."
        )
    return all(present)


def combine_sequences(per_seq: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-sequence results into one, as TrackEval's ``COMBINED_SEQ``.

    Counts are summed and every derived field recomputed from the sums -- so
    the combined MOTA / IDF1 / DetA are detection-weighted, not means of the
    per-sequence values. HOTA's association and localization scores (``AssA``,
    ``AssRe``, ``AssPr``, ``LocA``) are averages weighted by each sequence's
    ``HOTA_TP``. Combining one sequence recomputes its derived fields, which
    differs from the per-sequence result only for an empty sequence (e.g. its
    MOTA becomes ``-FP`` instead of 0).

    Args:
        per_seq: Results of :func:`mot_metrics` (or of any of
            :func:`clear_metrics`, :func:`id_metrics`, :func:`hota_metrics`;
            every result must hold the same families).

    Returns:
        One dict with the same keys.

    Raises:
        ValueError: If ``per_seq`` is empty, the results hold different metric
            families, or their HOTA arrays differ in length.
    """
    per_seq = list(per_seq)
    if not per_seq:
        raise ValueError("combine_sequences needs at least one result.")
    out: Dict[str, Any] = {}

    if _has_block(per_seq, "CLR_TP"):
        res: Dict[str, Any] = {
            f: int(sum(int(r[f]) for r in per_seq)) for f in CLEAR_INTEGER_FIELDS
        }
        res["MOTP_sum"] = float(sum(float(r["MOTP_sum"]) for r in per_seq))
        out.update(_clear_final_fields(res))

    if _has_block(per_seq, "IDTP"):
        res = {f: int(sum(int(r[f]) for r in per_seq)) for f in ID_INTEGER_FIELDS}
        out.update(_id_final_fields(res))

    if _has_block(per_seq, "HOTA_TP"):
        lengths = {len(np.asarray(r["HOTA_TP"])) for r in per_seq}
        if len(lengths) != 1:
            raise ValueError(
                f"HOTA results were computed over different alphas (lengths "
                f"{sorted(lengths)}); they cannot be combined."
            )
        res = {
            f: np.sum([np.asarray(r[f], dtype=float) for r in per_seq], axis=0)
            for f in HOTA_COUNT_FIELDS
        }
        weight = np.maximum(1.0, res["HOTA_TP"])
        for f in ("AssRe", "AssPr", "AssA"):
            res[f] = (
                np.sum(
                    [
                        np.asarray(r[f], dtype=float) * np.asarray(r["HOTA_TP"])
                        for r in per_seq
                    ],
                    axis=0,
                )
                / weight
            )
        loc_weighted = np.sum(
            [
                np.asarray(r["LocA"], dtype=float) * np.asarray(r["HOTA_TP"])
                for r in per_seq
            ],
            axis=0,
        )
        res["LocA"] = np.maximum(1e-10, loc_weighted) / np.maximum(
            1e-10, res["HOTA_TP"]
        )
        out.update(_hota_final_fields(res))

    return out
