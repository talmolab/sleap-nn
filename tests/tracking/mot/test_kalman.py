"""Tests for the MOT box Kalman filters (`sleap_nn.tracking.mot.kalman`)."""

import numpy as np
import pytest

from sleap_nn.tracking.mot.kalman import (
    KalmanFilterXYAH,
    KalmanFilterXYSR,
    KalmanFilterXYWH,
    make_kalman_filter,
    oru_replay,
)

KINDS = ("xywh", "xyah", "xysr")
NDIM = {"xywh": 8, "xyah": 8, "xysr": 7}

# chi-square 0.95 quantiles (2 and 4 degrees of freedom).
CHI2_2DOF = 5.9915
CHI2_4DOF = 9.4877

P_W = 1.0 / 20  # DeepSORT std weight, position
V_W = 1.0 / 160  # DeepSORT std weight, velocity

BOX = np.array([100.0, 50.0, 160.0, 90.0])  # 60 x 40


def _shift(box, dxy):
    """Translate an xyxy box by ``(dx, dy)``."""
    return box + np.array([dxy[0], dxy[1], dxy[0], dxy[1]], dtype=np.float64)


def _cxcywh(box):
    """``[cx, cy, w, h]`` of an xyxy box."""
    box = np.asarray(box, dtype=np.float64)
    return np.r_[(box[:2] + box[2:]) / 2, box[2:] - box[:2]]


def _track(kf, box, velocity, n):
    """Initiate at ``box`` and run ``n`` predict+update steps of constant motion."""
    mean, cov = kf.initiate(box)
    for t in range(1, n + 1):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, _shift(box, np.asarray(velocity) * t))
    return mean, cov


def _assert_finite_psd(cov, tol=-1e-9):
    assert np.all(np.isfinite(cov))
    np.testing.assert_array_equal(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() >= tol


def _recording(kf):
    """A copy of ``kf`` that logs its predict/update calls."""
    calls = []

    class Recording(type(kf)):
        def predict(self, mean, cov):
            calls.append(("predict", None))
            return super().predict(mean, cov)

        def update(self, mean, cov, xyxy, conf=None):
            calls.append(("update", np.array(xyxy, dtype=np.float64)))
            return super().update(mean, cov, xyxy, conf)

    return Recording(nsa=kf.nsa), calls


def test_make_kalman_filter():
    assert isinstance(make_kalman_filter("xywh"), KalmanFilterXYWH)
    assert isinstance(make_kalman_filter("xyah"), KalmanFilterXYAH)
    assert isinstance(make_kalman_filter("xysr"), KalmanFilterXYSR)
    assert isinstance(make_kalman_filter(" XYAH "), KalmanFilterXYAH)
    assert make_kalman_filter("xyah", nsa=True).nsa is True
    assert make_kalman_filter("xywh").nsa is False
    for bad in ("auto", "xyxy", ""):
        with pytest.raises(ValueError, match="Unknown Kalman filter"):
            make_kalman_filter(bad)


@pytest.mark.parametrize("kind", KINDS)
def test_shapes(kind):
    kf = make_kalman_filter(kind)
    n_dim = NDIM[kind]
    assert kf.ndim == n_dim and kf.ndim_z == 4

    mean, cov = kf.initiate(BOX)
    assert mean.shape == (n_dim,) and cov.shape == (n_dim, n_dim)
    mean, cov = kf.predict(mean, cov)
    assert mean.shape == (n_dim,) and cov.shape == (n_dim, n_dim)
    z_mean, s = kf.project(mean, cov)
    assert z_mean.shape == (4,) and s.shape == (4, 4)
    mean, cov = kf.update(mean, cov, BOX)
    assert mean.shape == (n_dim,) and cov.shape == (n_dim, n_dim)
    assert kf.center(mean).shape == (2,)
    assert kf.xyxy_from_mean(mean).shape == (4,)
    assert kf.z_from_xyxy(BOX).shape == (4,)

    boxes = np.stack([BOX, _shift(BOX, (5, 5)), _shift(BOX, (50, 0))])
    assert kf.z_from_xyxy(boxes).shape == (3, 4)
    assert kf.gating_distance(mean, cov, boxes).shape == (3,)
    assert kf.gating_distance(mean, cov, np.zeros((0, 4))).shape == (0,)

    means, covs = kf.multi_predict(np.stack([mean] * 5), np.stack([cov] * 5))
    assert means.shape == (5, n_dim) and covs.shape == (5, n_dim, n_dim)
    assert kf.xyxy_from_mean(means).shape == (5, 4)
    means, covs = kf.multi_predict(np.zeros((0, n_dim)), np.zeros((0, n_dim, n_dim)))
    assert means.shape == (0, n_dim) and covs.shape == (0, n_dim, n_dim)


@pytest.mark.parametrize("kind", KINDS)
def test_noise_constants(kind):
    """P0, Q and R match the published trackers' constants."""
    kf = make_kalman_filter(kind)
    w, h = 60.0, 40.0
    mean, p0 = kf.initiate(BOX)
    # From a zero covariance, one predict gives Q and a projection gives R.
    _, q = kf.predict(mean, np.zeros_like(p0))
    _, r = kf.project(mean, np.zeros_like(p0))
    if kind == "xysr":
        p0_var = [10, 10, 10, 10, 1e4, 1e4, 1e4]
        q_var = [1, 1, 1, 1, 0.01, 0.01, 1e-4]
        r_var = [1, 1, 10, 10]
    elif kind == "xyah":
        p0_var = np.square(
            [2 * P_W * h, 2 * P_W * h, 1e-2, 2 * P_W * h]
            + [10 * V_W * h, 10 * V_W * h, 1e-5, 10 * V_W * h]
        )
        q_var = np.square(
            [P_W * h, P_W * h, 1e-2, P_W * h, V_W * h, V_W * h, 1e-5, V_W * h]
        )
        r_var = np.square([P_W * h, P_W * h, 1e-1, P_W * h])
    else:
        whwh = np.array([w, h, w, h])
        p0_var = np.square(np.r_[2 * P_W * whwh, 10 * V_W * whwh])
        q_var = np.square(np.r_[P_W * whwh, V_W * whwh])
        r_var = np.square(P_W * whwh)
    np.testing.assert_allclose(p0, np.diag(p0_var), rtol=1e-12)
    np.testing.assert_allclose(q, np.diag(q_var), rtol=1e-12)
    np.testing.assert_allclose(r, np.diag(r_var), rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_xyxy_round_trip(kind):
    kf = make_kalman_filter(kind)
    rng = np.random.default_rng(0)
    xy = rng.uniform(-50, 500, size=(20, 2))
    wh = rng.uniform(1, 200, size=(20, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    for box in boxes:
        mean, _ = kf.initiate(box)
        np.testing.assert_allclose(kf.xyxy_from_mean(mean), box, atol=1e-9)
    # Batched conversions agree with the per-box ones.
    z = kf.z_from_xyxy(boxes)
    np.testing.assert_allclose(z, np.stack([kf.z_from_xyxy(b) for b in boxes]))
    means = np.zeros((len(boxes), kf.ndim))
    means[:, :4] = z
    np.testing.assert_allclose(kf.xyxy_from_mean(means), boxes, atol=1e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_constant_velocity_converges(kind):
    kf = make_kalman_filter(kind)
    velocity = np.array([3.0, -2.0])
    mean, cov = _track(kf, BOX, velocity, n=60)
    np.testing.assert_allclose(mean[4:6], velocity, atol=0.01)
    # The one-step prediction lands on the next box.
    pred, _ = kf.predict(mean, cov)
    np.testing.assert_allclose(
        kf.xyxy_from_mean(pred), _shift(BOX, velocity * 61), atol=0.05
    )
    _assert_finite_psd(cov)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("size_velocity", [0.0, -1.0])
def test_covariance_psd_after_long_coasting(kind, size_velocity):
    kf = make_kalman_filter(kind)
    mean, cov = kf.initiate(BOX)
    for t in range(1, 21):
        mean, cov = kf.predict(mean, cov)
        # Constant motion; the box optionally shrinks by 1 px per side per step.
        box = _shift(BOX, (2.0 * t, 1.0 * t)) + size_velocity * t * np.array(
            [-1, -1, 1, 1]
        )
        mean, cov = kf.update(mean, cov, box)
    for _ in range(1000):
        mean, cov = kf.predict(mean, cov)
    assert np.all(np.isfinite(mean))
    _assert_finite_psd(cov)
    # Coasting never produces a box smaller than 1 px.
    box = kf.xyxy_from_mean(mean)
    assert np.all(np.isfinite(box))
    assert box[2] - box[0] >= 1.0 - 1e-9 and box[3] - box[1] >= 1.0 - 1e-9


@pytest.mark.parametrize("kind", KINDS)
def test_size_velocity_guard(kind):
    """A shrinking size never crosses zero in prediction (SORT's ``s + vs <= 0``)."""
    kf = make_kalman_filter(kind)
    mean, cov = kf.initiate(BOX)
    size_idx = 2 if kind != "xyah" else 3
    vel_idx = size_idx + 4
    mean = mean.copy()
    mean[vel_idx] = -2.0 * mean[size_idx]
    pred, _ = kf.predict(mean, cov)
    assert pred[vel_idx] == 0.0
    assert pred[size_idx] == mean[size_idx]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "box",
    [
        [10.0, 10.0, 10.0, 30.0],  # w = 0
        [10.0, 10.0, 40.0, 10.0],  # h = 0
        [10.0, 10.0, 10.0, 10.0],  # a point
        [10.0, 10.0, 5.0, 5.0],  # inverted
    ],
)
def test_degenerate_box_stays_finite(kind, box):
    kf = make_kalman_filter(kind, nsa=True)
    box = np.asarray(box)
    mean, cov = kf.initiate(box)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))
    out = kf.xyxy_from_mean(mean)
    assert out[2] - out[0] >= 1.0 - 1e-9 and out[3] - out[1] >= 1.0 - 1e-9
    for _ in range(20):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, box, conf=1.0)
    assert np.all(np.isfinite(mean))
    _assert_finite_psd(cov)
    d = kf.gating_distance(mean, cov, np.stack([box, _shift(box, (3, 3))]))
    assert np.all(np.isfinite(d))


@pytest.mark.parametrize("kind", KINDS)
def test_non_finite_inputs_are_guarded(kind):
    kf = make_kalman_filter(kind)
    nan_box = np.array([np.nan, 0.0, 10.0, 10.0])
    mean, cov = kf.initiate(nan_box)
    assert np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))

    mean, cov = kf.initiate(BOX)
    pred_mean, pred_cov = kf.predict(mean, cov)
    new_mean, new_cov = kf.update(pred_mean, pred_cov, nan_box)
    np.testing.assert_array_equal(new_mean, pred_mean)
    np.testing.assert_array_equal(new_cov, pred_cov)
    new_mean, _ = kf.update(pred_mean, pred_cov, [0.0, 0.0, np.inf, 10.0])
    np.testing.assert_array_equal(new_mean, pred_mean)

    d = kf.gating_distance(pred_mean, pred_cov, np.stack([BOX, nan_box]))
    assert np.isfinite(d[0]) and d[1] == np.inf

    out_mean, out_cov = oru_replay(kf, pred_mean, pred_cov, BOX, nan_box, n_gap=3)
    assert np.all(np.isfinite(out_mean)) and np.all(np.isfinite(out_cov))


@pytest.mark.parametrize("kind", KINDS)
def test_inputs_not_mutated(kind):
    kf = make_kalman_filter(kind)
    mean, cov = _track(kf, BOX, (2.0, 1.0), n=5)
    mean[6 if kind == "xysr" else 7] = -1e6  # trips the size-velocity guard
    mean0, cov0 = mean.copy(), cov.copy()
    kf.predict(mean, cov)
    kf.multi_predict(mean[None], cov[None])
    kf.update(mean, cov, BOX)
    kf.project(mean, cov)
    kf.gating_distance(mean, cov, BOX[None])
    oru_replay(kf, mean, cov, BOX, _shift(BOX, (5, 5)), n_gap=3)
    np.testing.assert_array_equal(mean, mean0)
    np.testing.assert_array_equal(cov, cov0)


@pytest.mark.parametrize("kind", KINDS)
def test_multi_predict_matches_loop(kind):
    kf = make_kalman_filter(kind)
    rng = np.random.default_rng(1)
    means, covs = [], []
    for i in range(6):
        box = _shift(BOX, rng.uniform(-100, 100, 2)) * rng.uniform(0.5, 2.0)
        mean, cov = _track(kf, box, rng.uniform(-3, 3, 2), n=int(rng.integers(0, 6)))
        means.append(mean)
        covs.append(cov)
    means[0] = means[0].copy()
    means[0][6 if kind == "xysr" else 7] = -1e6  # one row trips the guard
    out_means, out_covs = kf.multi_predict(np.stack(means), np.stack(covs))
    for mean, cov, out_mean, out_cov in zip(means, covs, out_means, out_covs):
        loop_mean, loop_cov = kf.predict(mean, cov)
        np.testing.assert_allclose(out_mean, loop_mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out_cov, loop_cov, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_gating_distance(kind):
    kf = make_kalman_filter(kind)
    velocity = np.array([2.0, 1.0])
    mean, cov = _track(kf, BOX, velocity, n=30)
    mean, cov = kf.predict(mean, cov)
    truth = _shift(BOX, velocity * 31)
    far = _shift(truth, (200.0, -150.0))
    boxes = np.stack([truth, far])

    d_pos = kf.gating_distance(mean, cov, boxes, only_position=True)
    assert d_pos[0] < CHI2_2DOF < d_pos[1]
    d_full = kf.gating_distance(mean, cov, boxes)
    assert d_full[0] < CHI2_4DOF < d_full[1]

    # Squared Mahalanobis distance of the 2-D center block, computed directly.
    z_mean, s = kf.project(mean, cov)
    diff = kf.z_from_xyxy(far)[:2] - z_mean[:2]
    expected = diff @ np.linalg.solve(s[:2, :2], diff)
    np.testing.assert_allclose(d_pos[1], expected, rtol=1e-9)

    # Gating never uses the NSA-scaled noise.
    kf_nsa = make_kalman_filter(kind, nsa=True)
    np.testing.assert_array_equal(kf_nsa.gating_distance(mean, cov, boxes), d_full)


def test_solve_spd_is_cholesky_bit_for_bit():
    """`_solve_spd` calls LAPACK potrf/potrs directly: the same bits as scipy's
    cho_factor/cho_solve on SPD input, and the jitter / pseudo-inverse fallbacks
    still give finite results on singular or indefinite input."""
    from scipy.linalg import cho_factor, cho_solve

    from sleap_nn.tracking.mot.kalman import _solve_spd

    rng = np.random.default_rng(0)
    for k in (2, 4):
        for _ in range(50):
            m = rng.normal(size=(k, k))
            a = 0.5 * ((m @ m.T + 0.1 * np.eye(k)) + (m @ m.T + 0.1 * np.eye(k)).T)
            b = rng.normal(size=(k, 3))
            want = cho_solve(cho_factor(a, lower=True, check_finite=False), b)
            np.testing.assert_array_equal(_solve_spd(a, b), want)
        v = rng.normal(size=k)
        assert np.all(np.isfinite(_solve_spd(np.outer(v, v), np.ones((k, 2)))))
        assert np.all(np.isfinite(_solve_spd(-np.eye(k), np.ones((k, 2)))))
    assert _solve_spd(np.eye(2), np.zeros((2, 0))).shape == (2, 0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("only_position", [True, False])
def test_gating_distances_rows_are_single_state_gates(kind, only_position):
    """Batched gating: row i is the gate of state i alone (bit for bit); a
    non-finite state or box gives inf; no state or no box gives an empty side."""
    kf = make_kalman_filter(kind)
    states = []
    for velocity in ((2.0, 1.0), (-1.0, 3.0), (0.0, 0.0)):
        mean, cov = _track(kf, BOX, velocity, n=10)
        states.append(kf.predict(mean, cov))
    bad = states[0][0].copy()
    bad[0] = np.nan
    states.append((bad, states[0][1]))
    boxes = np.stack([BOX, _shift(BOX, (30, -20)), np.full(4, np.nan)])
    out = kf.gating_distances(
        [m for m, _ in states], [c for _, c in states], boxes, only_position
    )
    assert out.shape == (4, 3)
    assert np.all(np.isinf(out[3])) and np.all(np.isinf(out[:, 2]))
    assert np.all(np.isfinite(out[:3, :2]))
    # A reference gate per state: project, slice, solve on its own.
    for i, (mean, cov) in enumerate(states[:3]):
        z_mean, s = kf.project(mean, cov)
        k = 2 if only_position else 4
        d = kf.z_from_xyxy(boxes[:2])[:, :k] - z_mean[:k]
        want = np.maximum(np.sum(d * np.linalg.solve(s[:k, :k], d.T).T, axis=1), 0)
        np.testing.assert_allclose(out[i, :2], want, rtol=1e-10)
    assert kf.gating_distances([], [], boxes).shape == (0, 3)
    assert kf.gating_distances(
        [m for m, _ in states], [c for _, c in states], np.zeros((0, 4))
    ).shape == (4, 0)


def test_position_only_gate_tolerates_shape_change():
    """A rotating animal's box changes shape: only the 2-dof gate keeps it."""
    kf = make_kalman_filter("xyah")
    elongated = np.array([100.0, 100.0, 190.0, 130.0])  # 90 x 30
    mean, cov = _track(kf, elongated, (0.0, 0.0), n=30)
    mean, cov = kf.predict(mean, cov)
    turned = np.array([110.0, 90.0, 180.0, 140.0])  # 70 x 50, same center
    assert kf.gating_distance(mean, cov, turned[None])[0] > CHI2_4DOF
    assert (
        kf.gating_distance(mean, cov, turned[None], only_position=True)[0] < CHI2_2DOF
    )


@pytest.mark.parametrize("kind", KINDS)
def test_nsa_scales_measurement_noise(kind):
    kf = make_kalman_filter(kind, nsa=True)
    mean, cov = _track(kf, BOX, (1.0, 1.0), n=5)
    zero = np.zeros_like(cov)
    _, r0 = kf.project(mean, zero)
    _, r_none = kf.project(mean, zero, conf=None)
    np.testing.assert_array_equal(r0, r_none)
    assert np.all(np.diag(r0) > 0)

    for c in (0.0, 0.3, 0.8, 0.99):
        _, r_c = kf.project(mean, zero, conf=c)
        np.testing.assert_allclose(r_c, r0 * (1 - c) ** 2, rtol=1e-12)
        # Projection adds the state covariance on top of the scaled noise.
        _, s_c = kf.project(mean, cov, conf=c)
        np.testing.assert_allclose(s_c, cov[:4, :4] + r0 * (1 - c) ** 2, rtol=1e-12)

    # c is clipped to [0, 0.99]: conf=1 keeps R finite and positive.
    _, r_one = kf.project(mean, zero, conf=1.0)
    np.testing.assert_allclose(r_one, r0 * 0.01**2, rtol=1e-12)
    assert np.all(np.isfinite(r_one)) and np.all(np.diag(r_one) > 0)
    _, r_big = kf.project(mean, zero, conf=5.0)
    np.testing.assert_array_equal(r_big, r_one)
    _, r_neg = kf.project(mean, zero, conf=-1.0)
    np.testing.assert_array_equal(r_neg, r0)
    _, r_nan = kf.project(mean, zero, conf=float("nan"))
    np.testing.assert_array_equal(r_nan, r0)

    # Without NSA the confidence is ignored.
    _, r_plain = make_kalman_filter(kind).project(mean, zero, conf=0.9)
    np.testing.assert_array_equal(r_plain, r0)

    # A confident detection pulls the state closer to the measurement.
    pred_mean, pred_cov = kf.predict(mean, cov)
    obs = _shift(BOX, (30.0, 30.0))
    lo, _ = kf.update(pred_mean, pred_cov, obs, conf=0.1)
    hi, hi_cov = kf.update(pred_mean, pred_cov, obs, conf=1.0)
    target = kf.z_from_xyxy(obs)[:2]
    assert np.linalg.norm(hi[:2] - target) < np.linalg.norm(lo[:2] - target)
    assert np.all(np.isfinite(hi))
    _assert_finite_psd(hi_cov)


def _gap_scenario(kf, v_before, v_gap, n_gap):
    """Track 30 steps at ``v_before``, miss, then re-match after ``n_gap`` steps."""
    pos = np.zeros(2)
    mean, cov = kf.initiate(BOX)
    for _ in range(30):
        pos = pos + v_before
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, _shift(BOX, pos))
    last_obs = _shift(BOX, pos)
    # First miss: the reference freezes right after this step's prediction.
    frozen_mean, frozen_cov = kf.predict(mean, cov)
    new_pos = pos + n_gap * np.asarray(v_gap)
    return frozen_mean, frozen_cov, last_obs, _shift(BOX, new_pos), new_pos


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("v_gap", [(0.0, 3.0), (-3.0, 0.0), (0.0, 0.0)])
def test_oru_replay_beats_coasting(kind, v_gap):
    kf = make_kalman_filter(kind)
    v_before, v_gap, n_gap = np.array([3.0, 0.0]), np.array(v_gap), 10
    frozen_mean, frozen_cov, last_obs, new_obs, new_pos = _gap_scenario(
        kf, v_before, v_gap, n_gap
    )

    # Coasting: keep predicting through the gap, then the real update.
    coast_mean, coast_cov = frozen_mean, frozen_cov
    for _ in range(n_gap - 1):
        coast_mean, coast_cov = kf.predict(coast_mean, coast_cov)
    coast_mean, coast_cov = kf.update(coast_mean, coast_cov, new_obs)

    # ORU: replay the virtual trajectory, then the real update.
    oru_mean, oru_cov = oru_replay(
        kf, frozen_mean, frozen_cov, last_obs, new_obs, n_gap
    )
    oru_mean, oru_cov = kf.update(oru_mean, oru_cov, new_obs)

    coast_err = np.linalg.norm(coast_mean[4:6] - v_gap)
    oru_err = np.linalg.norm(oru_mean[4:6] - v_gap)
    assert oru_err < coast_err
    # The next prediction is closer to where the animal goes.
    next_center = kf.z_from_xyxy(_shift(BOX, new_pos + v_gap))[:2]
    coast_next = kf.predict(coast_mean, coast_cov)[0][:2]
    oru_next = kf.predict(oru_mean, oru_cov)[0][:2]
    assert np.linalg.norm(oru_next - next_center) < np.linalg.norm(
        coast_next - next_center
    )
    # ORU also discards the covariance blown up while coasting.
    assert np.trace(oru_cov) < np.trace(coast_cov)
    _assert_finite_psd(oru_cov)


@pytest.mark.parametrize("kind", KINDS)
def test_oru_replay_call_sequence(kind):
    """Reference ORU: update, then (predict, update) per step; last virtual = new obs."""
    base_kf = make_kalman_filter(kind)
    frozen_mean, frozen_cov, last_obs, new_obs, _ = _gap_scenario(
        base_kf, np.array([3.0, 0.0]), np.array([0.0, 2.0]), n_gap=4
    )
    # Change the size over the gap so the interpolation covers w and h too.
    new_obs = new_obs + np.array([-4.0, -2.0, 4.0, 2.0])

    kf, calls = _recording(base_kf)
    oru_replay(kf, frozen_mean, frozen_cov, last_obs, new_obs, n_gap=4)
    assert [name for name, _ in calls] == [
        "update",
        "predict",
        "update",
        "predict",
        "update",
        "predict",
        "update",
    ]
    updates = [box for name, box in calls if name == "update"]
    np.testing.assert_array_equal(updates[-1], new_obs)
    # Virtual boxes interpolate (cx, cy, w, h) linearly.
    start, end = _cxcywh(last_obs), _cxcywh(new_obs)
    for j, box in enumerate(updates[:-1], start=1):
        np.testing.assert_allclose(
            _cxcywh(box), start + j / 4 * (end - start), atol=1e-9
        )

    # frozen_steps=0 (posterior at the last observation) predicts first ...
    calls.clear()
    oru_replay(kf, frozen_mean, frozen_cov, last_obs, new_obs, 4, frozen_steps=0)
    assert [name for name, _ in calls][:2] == ["predict", "update"]
    assert sum(name == "update" for name, _ in calls) == 4
    # ... and a multi-step first miss skips the virtual steps it already passed.
    calls.clear()
    oru_replay(kf, frozen_mean, frozen_cov, last_obs, new_obs, 4, frozen_steps=3)
    assert [name for name, _ in calls] == ["update", "predict", "update"]
    # n_gap=1 is a single update with the new observation.
    calls.clear()
    oru_replay(kf, frozen_mean, frozen_cov, last_obs, new_obs, 1)
    assert [name for name, _ in calls] == ["update"]
    np.testing.assert_array_equal(calls[0][1], new_obs)


@pytest.mark.parametrize("kind", KINDS)
def test_oru_replay_frozen_steps_zero_matches_reference(kind):
    """Freezing the posterior (frozen_steps=0) equals freezing after one predict."""
    kf = make_kalman_filter(kind)
    mean, cov = _track(kf, BOX, (3.0, 1.0), n=10)
    last_obs = _shift(BOX, (30.0, 10.0))
    new_obs = _shift(BOX, (30.0, 40.0))
    a = oru_replay(kf, mean, cov, last_obs, new_obs, 6, frozen_steps=0)
    b = oru_replay(kf, *kf.predict(mean, cov), last_obs, new_obs, 6)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-12)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_oru_replay_with_nsa_confidence(kind):
    kf = make_kalman_filter(kind, nsa=True)
    frozen_mean, frozen_cov, last_obs, new_obs, _ = _gap_scenario(
        kf, np.array([3.0, 0.0]), np.array([0.0, 3.0]), n_gap=5
    )
    mean, cov = oru_replay(kf, frozen_mean, frozen_cov, last_obs, new_obs, 5, conf=1.0)
    assert np.all(np.isfinite(mean))
    _assert_finite_psd(cov)
