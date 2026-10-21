"""Tests for the TrackEval-faithful MOT metric core, ``sleap_nn.tracking.mot_metrics``.

Two kinds of check:

- Hand-checkable toy sequences (A-G), each isolating one definitional detail
  (continuity bonus, non-exclusive IDF1 counting, the MT boundary, the
  fragmentation quirk, LocA = 1 without true positives, HOTA(0) at alpha 0.05),
  asserted exactly.
- A frozen oracle (`tests/assets/mot_metrics/trackeval_oracle.json`): 36 random
  sequences and 6 combined groups scored by TrackEval 1.3.0 itself, so parity
  is tested without TrackEval as a dependency. Every field TrackEval reports
  must agree to 1e-9.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from sleap_nn.tracking.mot_metrics import (
    CLEAR_FLOAT_FIELDS,
    CLEAR_INTEGER_FIELDS,
    HOTA_ALPHAS,
    HOTA_ARRAY_FIELDS,
    HOTA_COUNT_FIELDS,
    HOTA_SCALAR_FIELDS,
    ID_FLOAT_FIELDS,
    ID_INTEGER_FIELDS,
    MOTSequence,
    clear_metrics,
    combine_sequences,
    hota_metrics,
    id_metrics,
    mot_metrics,
    relabel_contiguous,
)

ORACLE = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "mot_metrics"
    / "trackeval_oracle.json"
)
EXACT = 1e-12


def _ids(*x):
    return np.array(x, dtype=int)


def _seq(frames, resets=()):
    """Build a sequence from ``(gt_ids, pred_ids, similarity)`` triples."""
    return MOTSequence(
        gt_ids=[g for g, _p, _s in frames],
        pred_ids=[p for _g, p, _s in frames],
        similarity=[s for _g, _p, s in frames],
        resets=resets,
    )


def _mean(res, key):
    return float(np.mean(res[key]))


# ---------------------------------------------------------------------------
# Toy sequences
# ---------------------------------------------------------------------------
def _toy_a():
    """Two objects, ten frames, perfect tracking."""
    return _seq([(_ids(0, 1), _ids(0, 1), np.eye(2)) for _ in range(10)])


def _toy_b():
    """Two objects, ten frames, predicted identities swap at frame 5."""
    return _seq(
        [
            (_ids(0, 1), _ids(0, 1) if f < 5 else _ids(1, 0), np.eye(2))
            for f in range(10)
        ]
    )


def _toy_gap(gap_frame):
    """One object over five frames; frame 2 is replaced by ``gap_frame``."""
    full = (_ids(0), _ids(0), np.ones((1, 1)))
    return _seq([full if f != 2 else gap_frame for f in range(5)])


def test_toy_a_perfect_tracking():
    r = mot_metrics(_toy_a())
    assert (r["CLR_TP"], r["CLR_FN"], r["CLR_FP"]) == (20, 0, 0)
    assert (r["IDSW"], r["Frag"], r["MT"], r["PT"], r["ML"]) == (0, 0, 2, 0, 0)
    for key in ("MOTA", "MOTP", "MODA", "IDF1", "IDP", "IDR", "HOTA(0)"):
        assert r[key] == pytest.approx(1.0, abs=EXACT), key
    for key in ("HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr"):
        np.testing.assert_allclose(r[key], 1.0, atol=EXACT, err_msg=key)
    assert r["IDTP"] == 20
    assert r["CLR_Frames"] == 10


def test_toy_b_swap_halfway():
    """One swap: a switch on each trajectory, IDF1 0.5, AssA 1/3 at every alpha."""
    r = mot_metrics(_toy_b())
    assert r["IDSW"] == 2
    assert r["Frag"] == 0
    assert (r["MT"], r["PT"], r["ML"]) == (2, 0, 0)
    assert r["MOTA"] == pytest.approx(0.9, abs=EXACT)
    assert r["IDF1"] == pytest.approx(0.5, abs=EXACT)
    assert r["IDTP"] == 10
    np.testing.assert_allclose(r["DetA"], 1.0, atol=EXACT)
    np.testing.assert_allclose(r["AssA"], 1 / 3, atol=EXACT)
    np.testing.assert_allclose(r["HOTA"], np.sqrt(1 / 3), atol=EXACT)
    np.testing.assert_allclose(r["LocA"], 1.0, atol=EXACT)


def test_toy_c1_gap_with_no_predictions_is_not_a_fragment():
    """TrackEval quirk: a timestep with NO predictions does not break a segment."""
    r = mot_metrics(_toy_gap((_ids(0), _ids(), np.zeros((1, 0)))))
    assert r["MOTA"] == pytest.approx(0.8, abs=EXACT)
    assert (r["IDSW"], r["Frag"]) == (0, 0)
    # 4/5 = 0.8 is not > 0.8: partly tracked.
    assert (r["MT"], r["PT"], r["ML"]) == (0, 1, 0)
    assert r["IDF1"] == pytest.approx(4 / 4.5, abs=EXACT)
    assert _mean(r, "HOTA") == pytest.approx(0.8, abs=EXACT)


def test_toy_c2_gap_with_only_a_false_positive_is_a_fragment():
    r = mot_metrics(_toy_gap((_ids(0), _ids(1), np.zeros((1, 1)))))
    assert r["MOTA"] == pytest.approx(0.6, abs=EXACT)
    assert (r["IDSW"], r["Frag"]) == (0, 1)
    assert r["IDF1"] == pytest.approx(0.8, abs=EXACT)
    # DetA = 4 / (4 + 1 + 1); AssA = 4 / (5 + 4 - 4).
    np.testing.assert_allclose(r["HOTA"], np.sqrt(2 / 3 * 0.8), atol=EXACT)


def test_toy_d_localization_and_hota_zero():
    """Constant similarity 0.6: matched at 12 of 19 alphas; LocA = 1 elsewhere."""
    r = mot_metrics(_seq([(_ids(0), _ids(0), np.full((1, 1), 0.6))] * 4))
    assert r["MOTA"] == pytest.approx(1.0, abs=EXACT)
    assert r["MOTP"] == pytest.approx(0.6, abs=EXACT)
    assert _mean(r, "HOTA") == pytest.approx(12 / 19, abs=EXACT)
    assert _mean(r, "LocA") == pytest.approx((12 * 0.6 + 7 * 1.0) / 19, abs=EXACT)
    # HOTA(0) / LocA(0) are at alpha = 0.05, the first threshold.
    assert r["HOTA(0)"] == pytest.approx(1.0, abs=EXACT)
    assert r["LocA(0)"] == pytest.approx(0.6, abs=EXACT)
    assert r["HOTALocA(0)"] == pytest.approx(0.6, abs=EXACT)


def test_toy_e_mostly_tracked_is_strictly_above_80_percent():
    r = mot_metrics(_toy_gap((_ids(0), _ids(0), np.zeros((1, 1)))))
    assert (r["MT"], r["PT"], r["ML"]) == (0, 1, 0)
    assert r["PTR"] == pytest.approx(1.0, abs=EXACT)


def test_toy_f_identity_counts_every_pair_above_threshold():
    """IDTP counts non-exclusive potential matches: 6 here, not 4."""
    frames = [(_ids(0, 1), _ids(0, 1), np.array([[0.9, 0.0], [0.6, 0.9]]))] * 2
    frames += [(_ids(1), _ids(0), np.array([[0.9]]))] * 4
    r = mot_metrics(_seq(frames))
    assert r["IDTP"] == 6
    assert r["IDF1"] == pytest.approx(0.75, abs=EXACT)


def test_toy_g_continuity_bonus_keeps_last_frames_match():
    """A better-scoring new prediction does not steal a continuing match."""
    frames = [
        (_ids(0), _ids(0), np.array([[0.9]])),
        (_ids(0), _ids(0, 1), np.array([[0.6, 0.95]])),
    ]
    r = mot_metrics(_seq(frames))
    assert r["IDSW"] == 0
    assert r["MOTA"] == pytest.approx(0.5, abs=EXACT)
    assert r["MOTP_sum"] == pytest.approx(1.5, abs=EXACT)


def test_mot_metrics_is_the_union_of_the_three_families():
    seq = _toy_b()
    union = {**clear_metrics(seq), **id_metrics(seq), **hota_metrics(seq)}
    combined = mot_metrics(seq)
    assert set(combined) == set(union)
    for key, value in union.items():
        np.testing.assert_array_equal(combined[key], value, err_msg=key)
    assert set(combined) == set(
        CLEAR_INTEGER_FIELDS
        + CLEAR_FLOAT_FIELDS
        + ID_INTEGER_FIELDS
        + ID_FLOAT_FIELDS
        + HOTA_ARRAY_FIELDS
        + HOTA_COUNT_FIELDS
        + HOTA_SCALAR_FIELDS
    )
    assert len(combined["HOTA"]) == len(HOTA_ALPHAS) == 19


def test_empty_sequences_follow_trackeval_shortcuts():
    """No predictions: all GT missed, derived fields left at 0 (MLR, LocA = 1)."""
    no_pred = mot_metrics(
        _seq([(_ids(0, 1), _ids(), np.zeros((2, 0)))] * 3 + [(_ids(2), _ids(), [])])
    )
    assert (no_pred["CLR_FN"], no_pred["ML"], no_pred["MLR"]) == (7, 3, 1.0)
    assert no_pred["MOTA"] == 0.0 and no_pred["CLR_Frames"] == 0
    assert no_pred["IDFN"] == 7 and no_pred["IDF1"] == 0.0
    np.testing.assert_array_equal(no_pred["HOTA_FN"], 7.0)
    np.testing.assert_array_equal(no_pred["LocA"], 1.0)
    assert no_pred["LocA(0)"] == 1.0 and no_pred["HOTA(0)"] == 0.0

    no_gt = mot_metrics(_seq([(_ids(), _ids(0, 1), np.zeros((0, 2)))] * 2))
    assert (no_gt["CLR_FP"], no_gt["ML"], no_gt["MOTA"]) == (4, 0, 0.0)
    assert no_gt["IDFP"] == 4
    np.testing.assert_array_equal(no_gt["HOTA_FP"], 4.0)
    # Combining recomputes the derived fields from the counts.
    assert combine_sequences([no_gt])["MOTA"] == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# TrackEval oracle
# ---------------------------------------------------------------------------
def _load_oracle():
    with open(ORACLE) as f:
        return json.load(f)


def _oracle_seq(case):
    return MOTSequence(
        gt_ids=[f["gt"] for f in case["frames"]],
        pred_ids=[f["pred"] for f in case["frames"]],
        similarity=[f["sim"] for f in case["frames"]],
    )


def _assert_matches(got, expected, label):
    for key, value in expected.items():
        np.testing.assert_allclose(
            np.asarray(got[key], dtype=float),
            np.asarray(value, dtype=float),
            rtol=0,
            atol=1e-9,
            err_msg=f"{label}: {key}",
        )


def test_oracle_every_sequence_matches_trackeval():
    oracle = _load_oracle()
    assert len(oracle["cases"]) == 36
    for case in oracle["cases"]:
        _assert_matches(mot_metrics(_oracle_seq(case)), case["expected"], case["name"])


def test_oracle_every_combined_group_matches_trackeval():
    oracle = _load_oracle()
    per_seq = [mot_metrics(_oracle_seq(case)) for case in oracle["cases"]]
    assert len(oracle["combined"]) == 6
    for group in oracle["combined"]:
        combined = combine_sequences([per_seq[i] for i in group["members"]])
        _assert_matches(combined, group["expected"], f"group {group['members']}")


def test_oracle_is_invariant_to_id_values_and_detection_order():
    """Arbitrary (non-contiguous) ids and shuffled detections change nothing."""
    oracle = _load_oracle()
    rng = np.random.default_rng(0)
    for case in oracle["cases"][:12]:
        frames = []
        for f in case["frames"]:
            g = np.asarray(f["gt"], dtype=int)
            p = np.asarray(f["pred"], dtype=int)
            s = np.asarray(f["sim"], dtype=float).reshape(len(g), len(p))
            gi, pi = rng.permutation(len(g)), rng.permutation(len(p))
            frames.append((g[gi] * 7 + 100, p[pi] * 3 - 50, s[np.ix_(gi, pi)]))
        _assert_matches(mot_metrics(_seq(frames)), case["expected"], case["name"])


def test_combine_sequences_handles_single_families():
    oracle = _load_oracle()
    group = oracle["combined"][0]
    seqs = [_oracle_seq(oracle["cases"][i]) for i in group["members"]]
    clear = combine_sequences([clear_metrics(s) for s in seqs])
    ident = combine_sequences([id_metrics(s) for s in seqs])
    hota = combine_sequences([hota_metrics(s) for s in seqs])
    assert "IDTP" not in clear and "HOTA" not in clear
    _assert_matches(
        {**clear, **ident, **hota}, group["expected"], "single-family combine"
    )


def test_combine_sequences_rejects_bad_input():
    with pytest.raises(ValueError, match="at least one"):
        combine_sequences([])
    with pytest.raises(ValueError, match="same metric functions"):
        combine_sequences([clear_metrics(_toy_a()), id_metrics(_toy_a())])
    with pytest.raises(ValueError, match="different alphas"):
        combine_sequences(
            [hota_metrics(_toy_a()), hota_metrics(_toy_a(), alphas=[0.5])]
        )


# ---------------------------------------------------------------------------
# Validation and relabeling
# ---------------------------------------------------------------------------
def test_duplicate_ids_in_a_timestep_raise_naming_the_timestep():
    frames = [(_ids(0, 1), _ids(0, 1), np.eye(2))] * 3
    frames.append((_ids(0, 1), _ids(4, 4), np.eye(2)))
    with pytest.raises(ValueError, match=r"timestep 3: duplicate predicted id\(s\)"):
        _seq(frames)
    with pytest.raises(ValueError, match=r"timestep 0: duplicate ground-truth"):
        _seq([(_ids(2, 2), _ids(0, 1), np.eye(2))])


def test_sequence_validation():
    with pytest.raises(ValueError, match="one entry per timestep"):
        MOTSequence(gt_ids=[_ids(0)], pred_ids=[], similarity=[])
    with pytest.raises(ValueError, match=r"timestep 0: similarity has shape"):
        _seq([(_ids(0, 1), _ids(0), np.ones((1, 2)))])
    with pytest.raises(ValueError, match=r"timestep 0: similarity has shape"):
        _seq([(_ids(0, 1), _ids(0), [])])
    with pytest.raises(ValueError, match="non-finite"):
        _seq([(_ids(0), _ids(0), np.array([[np.nan]]))])
    with pytest.raises(ValueError, match="must be integers"):
        _seq([(np.array([0.5]), _ids(0), np.ones((1, 1)))])
    with pytest.raises(ValueError, match="resets"):
        _seq([(_ids(0), _ids(0), np.ones((1, 1)))], resets=(1,))
    # Lists and empty similarities in any shape are normalized.
    seq = _seq([([3], [], [[]]), ([], [5, 6], []), ([], [], [])], resets=[2, 1, 2])
    assert seq.similarity[0].shape == (1, 0)
    assert seq.similarity[1].shape == (0, 2)
    assert seq.gt_ids[0].dtype == np.int64
    assert seq.resets == (1, 2)


def test_relabel_contiguous():
    seq = _seq(
        [
            (_ids(10, -3), _ids(7), np.array([[0.9], [0.1]])),
            (_ids(42), _ids(100, 7), np.array([[0.2, 0.8]])),
        ],
        resets=(1,),
    )
    relabeled = relabel_contiguous(seq)
    # Sorted order of the original ids: -3 -> 0, 10 -> 1, 42 -> 2; 7 -> 0, 100 -> 1.
    assert [g.tolist() for g in relabeled.gt_ids] == [[1, 0], [2]]
    assert [p.tolist() for p in relabeled.pred_ids] == [[0], [1, 0]]
    assert relabeled.similarity[0] is seq.similarity[0]
    assert relabeled.resets == (1,)
    # A gap in the ids must not create a phantom (mostly lost) trajectory.
    clear = clear_metrics(seq)
    assert clear["MT"] + clear["PT"] + clear["ML"] == len({10, -3, 42})


# ---------------------------------------------------------------------------
# Resets (video boundaries under global identities)
# ---------------------------------------------------------------------------
def _two_videos(second_pred_id):
    """One animal over two 5-frame videos; video B's prediction id is given."""
    video_a = [(_ids(0), _ids(0), np.ones((1, 1)))] * 5
    video_b = [(_ids(0), _ids(second_pred_id), np.ones((1, 1)))] * 5
    return video_a + video_b


def test_reset_clears_the_switch_memory_at_a_video_boundary():
    frames = _two_videos(second_pred_id=1)
    assert clear_metrics(_seq(frames))["IDSW"] == 1
    reset = clear_metrics(_seq(frames, resets=(5,)))
    assert reset["IDSW"] == 0
    # A new name in video B is still an identity error, for IDF1 and AssA.
    assert id_metrics(_seq(frames, resets=(5,)))["IDF1"] == pytest.approx(0.5)
    assert _mean(hota_metrics(_seq(frames, resets=(5,))), "AssA") < 1.0


def test_reset_counts_fragments_per_side_of_the_boundary():
    """A trajectory tracked through both videos is not fragmented by the cut."""
    frames = _two_videos(second_pred_id=0)
    assert clear_metrics(_seq(frames, resets=(5,)))["Frag"] == 0

    # A real gap inside video B still counts, exactly as for video B alone.
    gap = (_ids(0), _ids(9), np.zeros((1, 1)))
    frames_gap = frames[:7] + [gap] + frames[8:]
    with_reset = clear_metrics(_seq(frames_gap, resets=(5,)))
    per_video = combine_sequences(
        [clear_metrics(_seq(frames_gap[:5])), clear_metrics(_seq(frames_gap[5:]))]
    )
    for key in ("CLR_TP", "CLR_FN", "CLR_FP", "IDSW", "Frag", "MOTP_sum"):
        assert with_reset[key] == per_video[key], key
    assert with_reset["Frag"] == 1


def test_reset_drops_the_continuity_bonus_across_the_boundary():
    """Without a reset, video B's first frame keeps video A's last match."""
    frames = [
        (_ids(0), _ids(0), np.array([[0.9]])),
        (_ids(0), _ids(0, 1), np.array([[0.6, 0.95]])),
    ]
    assert clear_metrics(_seq(frames))["MOTP_sum"] == pytest.approx(1.5)
    assert clear_metrics(_seq(frames, resets=(1,)))["MOTP_sum"] == pytest.approx(1.85)
