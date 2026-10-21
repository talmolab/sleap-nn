"""Tests for `sleap_nn.tracking.mot.appearance` (AppearanceMemory, quality gate)."""

import numpy as np
import pytest

from sleap_nn.tracking.mot.appearance import AppearanceMemory, passes_quality_gate


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


E0 = np.array([1.0, 0.0, 0.0])
E1 = np.array([0.0, 1.0, 0.0])
E2 = np.array([0.0, 0.0, 1.0])


def test_empty_memory():
    """A fresh memory is empty: NaN statistics, (0, 0) gallery."""
    mem = AppearanceMemory()
    assert mem.is_empty and mem.ema is None and mem.n_updates == 0
    assert mem.vectors().shape == (0, 0)
    q = np.stack([E0, E1])
    for stat in (mem.ema_cos, mem.gallery_max_cos, mem.gallery_topk_mean):
        out = stat(q)
        assert out.shape == (2,) and np.isnan(out).all()


def test_seed_normalizes_and_resets():
    """The seed vector becomes the (normalized) EMA and the only gallery entry."""
    mem = AppearanceMemory(gallery_size=5)
    mem.seed([3.0, 4.0, 0.0])
    np.testing.assert_allclose(mem.ema, [0.6, 0.8, 0.0])
    assert len(mem.gallery) == 1 and mem.n_updates == 1 and not mem.is_empty
    np.testing.assert_allclose(mem.vectors(), [[0.6, 0.8, 0.0]])

    mem.update(E1)
    mem.seed(E2)  # re-seeding discards previous content
    np.testing.assert_allclose(mem.ema, E2)
    assert len(mem.gallery) == 1 and mem.n_updates == 1


@pytest.mark.parametrize("bad", [None, np.zeros(3), np.array([np.nan, 1.0, 0.0])])
def test_unusable_vectors_are_ignored(bad):
    """Missing, zero-norm or non-finite vectors never change the memory."""
    mem = AppearanceMemory()
    mem.seed(bad)
    assert mem.is_empty and mem.n_updates == 0
    mem.seed(E0)
    mem.update(bad)
    np.testing.assert_allclose(mem.ema, E0)
    assert len(mem.gallery) == 1 and mem.n_updates == 1


def test_update_ema_math_and_renormalization():
    """EMA = renorm(a * ema + (1 - a) * vec) with a = ema_alpha."""
    mem = AppearanceMemory(ema_alpha=0.8)
    mem.seed(E0)
    mem.update(2.0 * E1)  # scale is irrelevant: vectors are normalized first
    expected = _unit(0.8 * E0 + 0.2 * E1)
    np.testing.assert_allclose(mem.ema, expected)
    np.testing.assert_allclose(np.linalg.norm(mem.ema), 1.0)
    assert mem.n_updates == 2
    np.testing.assert_allclose(mem.vectors(), [E0, E1])

    mem.update(E2)
    np.testing.assert_allclose(mem.ema, _unit(0.8 * expected + 0.2 * E2))


def test_update_on_empty_memory_seeds_even_when_ungated():
    """The first usable vector always seeds (like the spawn vector)."""
    mem = AppearanceMemory()
    mem.update(E1, gated=False)
    np.testing.assert_allclose(mem.ema, E1)
    assert len(mem.gallery) == 1 and mem.n_updates == 1


def test_gate_failure_freezes_ema_and_gallery():
    """gated=False leaves BOTH the EMA and the gallery unchanged (alpha = 1)."""
    mem = AppearanceMemory(ema_alpha=0.5)
    mem.seed(E0)
    mem.update(E1)
    ema_before = mem.ema.copy()
    gallery_before = mem.vectors().copy()
    for _ in range(20):
        mem.update(E2, gated=False)
    np.testing.assert_array_equal(mem.ema, ema_before)
    np.testing.assert_array_equal(mem.vectors(), gallery_before)
    assert mem.n_updates == 2


def test_wrong_dimension_update_is_ignored():
    """A vector of another dimension than the memory's is ignored."""
    mem = AppearanceMemory()
    mem.seed(E0)
    mem.update(np.ones(5))
    np.testing.assert_allclose(mem.ema, E0)
    assert len(mem.gallery) == 1
    assert np.isnan(mem.ema_cos(np.ones((2, 5)))).all()
    assert np.isnan(mem.gallery_max_cos(np.ones((1, 5)))).all()


@pytest.mark.parametrize(
    "score, expected_alpha",
    [
        (1.0, 0.8),  # full trust -> base momentum
        (0.75, 0.9),  # trust 0.5 -> 0.8 + 0.2 * 0.5
        (0.5, 1.0),  # at the floor -> trust 0 -> frozen EMA
        (0.2, 1.0),  # below the floor -> frozen EMA
        (np.nan, 1.0),  # unknown confidence -> no trust
    ],
)
def test_adaptive_ema_momentum(score, expected_alpha):
    """Deep OC-SORT: a = alpha + (1 - alpha) * (1 - clip((s - f) / (1 - f)))."""
    mem = AppearanceMemory(ema_alpha=0.8, adaptive=True, score_floor=0.5)
    mem.seed(E0)
    mem.update(E1, score=score)
    expected = _unit(expected_alpha * E0 + (1 - expected_alpha) * E1)
    np.testing.assert_allclose(mem.ema, expected, atol=1e-12)
    # The gallery takes the gated vector whatever the trust.
    assert len(mem.gallery) == 2


def test_non_adaptive_ignores_score():
    """Without `adaptive`, the score does not change the momentum."""
    mem = AppearanceMemory(ema_alpha=0.8)
    mem.seed(E0)
    mem.update(E1, score=0.0)
    np.testing.assert_allclose(mem.ema, _unit(0.8 * E0 + 0.2 * E1))


def test_antipodal_mix_keeps_previous_ema():
    """a = 0.5 with an antipodal vector cancels out: the EMA is kept, never NaN."""
    mem = AppearanceMemory(ema_alpha=0.5)
    mem.seed(E0)
    mem.update(-E0)
    np.testing.assert_allclose(mem.ema, E0)
    assert len(mem.gallery) == 2


def test_gallery_keeps_most_recent():
    """The gallery is a bounded deque of the last `gallery_size` gated vectors."""
    rng = np.random.default_rng(0)
    vecs = [_unit(v) for v in rng.normal(size=(6, 4))]
    mem = AppearanceMemory(gallery_size=3)
    mem.seed(vecs[0])
    for v in vecs[1:]:
        mem.update(v)
    assert mem.gallery.maxlen == 3
    np.testing.assert_allclose(mem.vectors(), np.stack(vecs[-3:]))
    assert mem.n_updates == 6


def test_cosine_statistics():
    """ema_cos / gallery_max_cos / gallery_topk_mean on a known gallery."""
    mem = AppearanceMemory(ema_alpha=0.8, gallery_size=10)
    mem.seed(E0)
    mem.update(E1)
    mem.update(E2)
    q = np.stack([E0, _unit([1.0, 1.0, 0.0]), np.zeros(3)])

    ema = mem.ema_cos(q)
    np.testing.assert_allclose(ema[:2], q[:2] @ mem.ema)
    assert np.isnan(ema[2])  # missing (zero) query row

    gmax = mem.gallery_max_cos(q)
    np.testing.assert_allclose(gmax[:2], [1.0, np.sqrt(0.5)])
    assert np.isnan(gmax[2])

    # Gallery cosines of q[1]: (0.707, 0.707, 0) -> top-2 mean 0.707, top-3 0.471.
    np.testing.assert_allclose(mem.gallery_topk_mean(q[1:2], k=2), [np.sqrt(0.5)])
    np.testing.assert_allclose(
        mem.gallery_topk_mean(q[1:2], k=3), [2 * np.sqrt(0.5) / 3]
    )
    # k larger than the gallery -> mean over the whole gallery.
    np.testing.assert_allclose(mem.gallery_topk_mean(q[0:1], k=50), [1.0 / 3])
    # A single (D,) query works too.
    assert mem.ema_cos(E0).shape == (1,)


def test_gallery_statistics_fall_back_to_ema_without_gallery():
    """gallery_size=0: gallery statistics use the EMA."""
    mem = AppearanceMemory(gallery_size=0)
    mem.seed(E0)
    mem.update(E1)
    assert len(mem.gallery) == 0 and mem.vectors().shape == (0, 0)
    q = np.stack([E0, E1])
    np.testing.assert_allclose(mem.gallery_max_cos(q), mem.ema_cos(q))
    np.testing.assert_allclose(mem.gallery_topk_mean(q), mem.ema_cos(q))


_GATE_OK = dict(
    det_score=0.9,
    n_visible=4,
    n_nodes=4,
    max_iou_other=0.0,
    margin=0.5,
    high_threshold=0.5,
    min_margin=0.1,
    gallery_len=5,
)


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, True),
        ({"det_score": 0.5}, True),  # score >= high threshold (inclusive)
        ({"det_score": 0.49}, False),
        ({"n_visible": 2}, True),  # 2/4 = 0.5 visible (inclusive)
        ({"n_visible": 1}, False),
        ({"n_visible": 0, "n_nodes": 0}, False),
        ({"n_visible": 1, "n_nodes": 1}, True),  # masks
        ({"max_iou_other": 0.099}, True),
        ({"max_iou_other": 0.1}, False),  # overlap must be strictly below
        ({"max_iou_other": np.nan}, True),  # no overlap evidence
        ({"margin": 0.1}, True),  # margin >= min_margin (inclusive)
        ({"margin": 0.05}, False),
        ({"margin": np.inf}, True),  # no competitor
        ({"margin": np.nan}, False),
        ({"margin": -1.0, "gallery_len": 2}, True),  # young gallery: no margin test
        ({"margin": -1.0, "min_margin": None}, True),  # margin test disabled
        ({"margin": 0.05, "gallery_len": 3}, False),
    ],
)
def test_quality_gate_truth_table(change, expected):
    """Every condition of the quality gate, one at a time."""
    kwargs = {**_GATE_OK, **change}
    assert passes_quality_gate(**kwargs) is expected


def test_quality_gate_custom_fractions():
    """min_visible_fraction / max_overlap_iou are configurable."""
    kwargs = {**_GATE_OK, "n_visible": 3, "max_iou_other": 0.2}
    assert not passes_quality_gate(**kwargs)
    assert passes_quality_gate(**kwargs, min_visible_fraction=0.75, max_overlap_iou=0.3)
    assert not passes_quality_gate(
        **kwargs, min_visible_fraction=0.8, max_overlap_iou=0.3
    )
