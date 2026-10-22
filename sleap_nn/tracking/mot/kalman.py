"""Kalman filters for the MOT trackers (numpy only).

Three constant-velocity box filters, one per parameterization used in the
literature:

- :class:`KalmanFilterXYSR` -- SORT / OC-SORT: ``z = [cx, cy, s=area, r=w/h]``,
  7-D state, fixed noise.
- :class:`KalmanFilterXYAH` -- DeepSORT / StrongSORT / ByteTrack:
  ``z = [cx, cy, a=w/h, h]``, 8-D state, noise scaled by box height.
- :class:`KalmanFilterXYWH` -- BoT-SORT / Deep OC-SORT: ``z = [cx, cy, w, h]``,
  8-D state, noise scaled by width (x, w) and height (y, h).

Filters are stateless configuration objects; each track owns its
``(mean, cov)``. All filters share the interface of :class:`_KalmanBase`, accept
and return boxes as ``xyxy``, clamp degenerate sizes to >= 1 px, keep
covariances symmetric and never produce NaN/inf. The OC-SORT freeze /
observation-centric re-update (ORU) helpers work with every filter.

The measurement model of every filter observes the first four state entries
(``H = [I_4 | 0]``) and the motion model adds each velocity once per step. The
noise constants follow the published trackers (cited per class); the updates
use the Joseph form of the covariance correction, which is algebraically
identical to the references' ``P - K S K^T`` but keeps ``P`` positive
semi-definite under rounding.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
from loguru import logger
from scipy.linalg.lapack import dpotrf, dpotrs

# DeepSORT-style noise weights (XYAH, XYWH): the standard deviations are these
# fractions of the box size.
_STD_WEIGHT_POSITION = 1.0 / 20
_STD_WEIGHT_VELOCITY = 1.0 / 160

# Smallest box side (px) used by the conversions and by size-scaled noise.
_MIN_SIZE = 1.0

# Smallest aspect ratio used when converting an XYSR state back to a box.
_MIN_RATIO = 1e-6

# NSA confidence is clipped to [0, _NSA_MAX_CONF] so the scaled measurement
# noise never reaches 0 (a zero R makes the filter copy the measurement).
_NSA_MAX_CONF = 0.99


def _symmetrize(a: np.ndarray) -> np.ndarray:
    """Return ``(a + a^T) / 2`` over the last two axes."""
    return 0.5 * (a + np.swapaxes(a, -1, -2))


def _cxcywh_from_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` ``xyxy`` boxes to ``[cx, cy, w, h]`` with ``w, h >= 1``.

    Non-finite coordinates propagate (callers test the result with
    ``np.isfinite``).
    """
    b = np.asarray(xyxy, dtype=np.float64)
    if b.shape[-1:] != (4,):
        raise ValueError(f"Expected boxes of shape (..., 4), got {b.shape}.")
    cx = 0.5 * (b[..., 0] + b[..., 2])
    cy = 0.5 * (b[..., 1] + b[..., 3])
    # np.maximum propagates NaN, so invalid boxes stay detectable.
    w = np.maximum(b[..., 2] - b[..., 0], _MIN_SIZE)
    h = np.maximum(b[..., 3] - b[..., 1], _MIN_SIZE)
    return np.stack([cx, cy, w, h], axis=-1)


def _xyxy_from_cxcywh(c: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` ``[cx, cy, w, h]`` to ``xyxy``."""
    half_w = 0.5 * c[..., 2]
    half_h = 0.5 * c[..., 3]
    return np.stack(
        [
            c[..., 0] - half_w,
            c[..., 1] - half_h,
            c[..., 0] + half_w,
            c[..., 1] + half_h,
        ],
        axis=-1,
    )


def _solve_spd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``a x = b`` for a symmetric positive-definite ``a``.

    Uses a Cholesky factorization. If ``a`` is numerically not positive definite
    the solve is retried with a growing diagonal jitter, and the pseudo-inverse is
    the last resort, so finite inputs always give a finite result.

    Calls LAPACK ``dpotrf`` / ``dpotrs`` directly: the routines
    ``scipy.linalg.cho_factor`` / ``cho_solve`` wrap (same arguments, so the same
    bits), without their per-call dispatch and checks -- this runs once per
    Kalman update and per gated track on the hot path. A non-zero LAPACK
    ``info`` is the failure ``cho_factor`` / ``cho_solve`` would raise on.
    """
    a = _symmetrize(np.asarray(a, dtype=np.float64))
    b = np.asarray(b, dtype=np.float64)
    if b.size == 0:
        return np.empty_like(b)
    eye = None  # Built on the first failure only (the common case never jitters).
    jitter = 0.0
    for attempt in range(5):
        # `a + 0.0` is exactly `a + 0 * I` (signed zeros included).
        shifted = a + 0.0 if eye is None else a + jitter * eye
        factor, info = dpotrf(shifted, lower=1, clean=0)
        if info == 0:
            x, info = dpotrs(factor, b, lower=1)
            if info == 0:
                return x
        if eye is None:
            scale = max(float(np.max(np.abs(np.diag(a)))), 1e-12)
            eye = np.eye(len(a))
        # 1e-12, 1e-9, 1e-6, 1e-3 of the largest variance.
        jitter = scale * 10.0 ** (-12 + 3 * attempt)
    return np.linalg.pinv(a) @ b


class _KalmanBase:
    """Interface shared by the box Kalman filters.

    Attributes:
        ndim: State dimension.
        ndim_z: Measurement dimension (4).
        nsa: Scale the measurement noise by detection confidence (StrongSORT NSA):
            ``R_std *= (1 - c)`` with ``c`` clipped to ``[0, 0.99]``. Gating always
            uses the un-scaled ``R``.
    """

    ndim: int
    ndim_z: int = 4

    # (position index, velocity index) pairs of the constant-velocity model.
    _velocity_pairs: Tuple[Tuple[int, int], ...] = ()
    # (size index, velocity index) pairs whose velocity is zeroed before a
    # prediction that would make the size non-positive (SORT's ``s + vs <= 0``).
    _size_velocity_pairs: Tuple[Tuple[int, int], ...] = ()

    def __init__(self, nsa: bool = False):
        """Initialize the filter configuration.

        Args:
            nsa: Enable StrongSORT's NSA measurement-noise scaling.
        """
        self.nsa = bool(nsa)
        motion = np.eye(self.ndim)
        for pos, vel in self._velocity_pairs:
            motion[pos, vel] = 1.0
        self._motion_mat = motion

    # ------------------------------------------------------- per-filter hooks
    def _initial_var(self, z: np.ndarray) -> np.ndarray:
        """Initial state variances ``(ndim,)`` for a measurement ``z``."""
        raise NotImplementedError

    def _process_var(self, means: np.ndarray) -> np.ndarray:
        """Process-noise variances ``(n, ndim)`` for ``(n, ndim)`` pre-step means."""
        raise NotImplementedError

    def _measurement_var(self, mean: np.ndarray) -> np.ndarray:
        """Un-scaled measurement-noise variances ``(4,)`` at a state mean."""
        raise NotImplementedError

    # --------------------------------------------------------------- public
    def z_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray:
        """Convert an ``xyxy`` box to this filter's measurement vector.

        Args:
            xyxy: A ``(4,)`` box or ``(..., 4)`` boxes. Sides are clamped to
                >= 1 px; non-finite coordinates propagate.

        Returns:
            The measurement(s), same leading shape as ``xyxy``.
        """
        raise NotImplementedError

    def xyxy_from_mean(self, mean: np.ndarray) -> np.ndarray:
        """Convert a state mean to an ``xyxy`` box.

        Args:
            mean: A ``(ndim,)`` state mean or ``(..., ndim)`` means.

        Returns:
            The box(es) ``(..., 4)``, sides clamped to >= 1 px.
        """
        raise NotImplementedError

    def center(self, mean: np.ndarray) -> np.ndarray:
        """Box center ``(cx, cy)`` of a state mean."""
        return np.asarray(mean[:2], dtype=np.float64)

    def initiate(self, xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create ``(mean, cov)`` from an initial ``xyxy`` observation.

        Args:
            xyxy: The first observed box. Non-finite coordinates are replaced by 0
                (callers never pass them; the filter must still stay finite).

        Returns:
            ``(mean, cov)``: the measurement with zero velocities and the
            filter's initial covariance.
        """
        box = np.asarray(xyxy, dtype=np.float64).reshape(4)
        z = self.z_from_xyxy(box)
        if not np.all(np.isfinite(z)):
            logger.debug(f"Initiating a Kalman track from a non-finite box {box}.")
            box = np.nan_to_num(box, nan=0.0, posinf=0.0, neginf=0.0)
            z = self.z_from_xyxy(box)
            if not np.all(np.isfinite(z)):
                z = self.z_from_xyxy(np.zeros(4))
        mean = np.zeros(self.ndim)
        mean[: self.ndim_z] = z
        cov = np.diag(self._initial_var(z))
        return mean, cov

    def predict(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One constant-velocity prediction step.

        Args:
            mean: ``(ndim,)`` state mean (not modified).
            cov: ``(ndim, ndim)`` state covariance (not modified).

        Returns:
            The predicted ``(mean, cov)``.
        """
        means, covs = self.multi_predict(
            np.asarray(mean, dtype=np.float64)[None],
            np.asarray(cov, dtype=np.float64)[None],
        )
        return means[0], covs[0]

    def multi_predict(
        self, means: np.ndarray, covs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`predict` over ``(n, ndim)`` means and ``(n, ndim, ndim)`` covs.

        Args:
            means: ``(n, ndim)`` state means (not modified; ``n`` may be 0).
            covs: ``(n, ndim, ndim)`` state covariances (not modified).

        Returns:
            The predicted ``(means, covs)``, same shapes.
        """
        means = np.array(means, dtype=np.float64, copy=True).reshape(-1, self.ndim)
        covs = np.asarray(covs, dtype=np.float64).reshape(-1, self.ndim, self.ndim)
        # Stop a shrinking size from crossing zero (SORT's `s + vs <= 0` rule).
        for size, vel in self._size_velocity_pairs:
            shrinks_out = means[:, size] + means[:, vel] <= 0
            means[shrinks_out, vel] = 0.0
        # Size-scaled process noise is taken from the pre-step state (reference).
        q = self._process_var(means)
        motion = self._motion_mat
        means = means @ motion.T
        covs = np.einsum("ij,njk,lk->nil", motion, covs, motion)
        diag = np.arange(self.ndim)
        covs[:, diag, diag] += q
        return means, _symmetrize(covs)

    def _measurement_noise(
        self, mean: np.ndarray, conf: Optional[float] = None
    ) -> np.ndarray:
        """Measurement-noise variances ``(4,)``, NSA-scaled when enabled and given."""
        r = self._measurement_var(mean)
        if self.nsa and conf is not None:
            c = float(conf)
            if np.isfinite(c):
                c = min(max(c, 0.0), _NSA_MAX_CONF)
                # StrongSORT scales the noise std by (1 - c), i.e. R by (1 - c)^2.
                r = r * (1.0 - c) ** 2
        return r

    def project(
        self, mean: np.ndarray, cov: np.ndarray, conf: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project the state to measurement space, returning ``(z_mean, S)``.

        Args:
            mean: ``(ndim,)`` state mean.
            cov: ``(ndim, ndim)`` state covariance.
            conf: Detection confidence; scales ``R`` only when :attr:`nsa` is set
                (``None`` or non-finite: un-scaled).

        Returns:
            ``(z_mean, S)``: the ``(4,)`` projected mean and the ``(4, 4)``
            innovation covariance ``H P H^T + R``.
        """
        mean = np.asarray(mean, dtype=np.float64).reshape(self.ndim)
        cov = np.asarray(cov, dtype=np.float64).reshape(self.ndim, self.ndim)
        r = self._measurement_noise(mean, conf)
        k = self.ndim_z
        z_mean = mean[:k].copy()
        s = cov[:k, :k] + np.diag(r)
        return z_mean, _symmetrize(s)

    def update(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        xyxy: np.ndarray,
        conf: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the state with an ``xyxy`` observation.

        Args:
            mean: ``(ndim,)`` predicted state mean (not modified).
            cov: ``(ndim, ndim)`` predicted state covariance (not modified).
            xyxy: The observed box. A non-finite box leaves the state unchanged.
            conf: Detection confidence for NSA filters (ignored otherwise).

        Returns:
            The corrected ``(mean, cov)``.
        """
        mean = np.array(mean, dtype=np.float64, copy=True).reshape(self.ndim)
        cov = np.array(cov, dtype=np.float64, copy=True).reshape(self.ndim, self.ndim)
        z = self.z_from_xyxy(np.asarray(xyxy, dtype=np.float64).reshape(4))
        if not np.all(np.isfinite(z)):
            logger.debug("Skipping a Kalman update with a non-finite observation.")
            return mean, cov

        k = self.ndim_z
        r = self._measurement_noise(mean, conf)
        s = _symmetrize(cov[:k, :k] + np.diag(r))
        # Kalman gain K = P H^T S^-1; with H = [I | 0], H P = cov[:k] and, S and P
        # being symmetric, K = (S^-1 H P)^T.
        gain = _solve_spd(s, cov[:k, :]).T
        new_mean = mean + gain @ (z - mean[:k])
        # Joseph form (I - K H) P (I - K H)^T + K R K^T.
        i_kh = np.eye(self.ndim)
        i_kh[:, :k] -= gain
        new_cov = _symmetrize(i_kh @ cov @ i_kh.T + (gain * r) @ gain.T)
        if not (np.all(np.isfinite(new_mean)) and np.all(np.isfinite(new_cov))):
            logger.debug("Kalman update produced non-finite values; keeping prior.")
            return mean, cov
        return new_mean, new_cov

    def gating_distance(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        boxes_xyxy: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """Squared Mahalanobis distance from the prediction to ``(m, 4)`` boxes.

        ``only_position=True`` uses the center ``(cx, cy)`` only (2 dof). NaN boxes
        give ``inf``.

        Args:
            mean: ``(ndim,)`` predicted state mean.
            cov: ``(ndim, ndim)`` predicted state covariance.
            boxes_xyxy: ``(m, 4)`` candidate boxes.
            only_position: Gate on the 2-D center block of the innovation only
                (compare with ``CHI2INV95[2]``); else the full 4-D measurement
                (``CHI2INV95[4]``).

        Returns:
            ``(m,)`` squared distances. The innovation covariance always uses the
            un-scaled measurement noise (no NSA), as in the reference.
        """
        return self.gating_distances([mean], [cov], boxes_xyxy, only_position)[0]

    def gating_distances(
        self,
        means: Sequence[np.ndarray],
        covs: Sequence[np.ndarray],
        boxes_xyxy: np.ndarray,
        only_position: bool = False,
    ) -> np.ndarray:
        """:meth:`gating_distance` of ``n`` states at once: ``(n, m)``.

        Row ``i`` is ``gating_distance(means[i], covs[i], boxes_xyxy,
        only_position)``, bit for bit: the boxes are converted once, and each
        state's innovation covariance is factored and solved on its own (the
        same LAPACK calls), so batching changes no result.

        Args:
            means: ``n`` predicted ``(ndim,)`` state means.
            covs: ``n`` predicted ``(ndim, ndim)`` state covariances.
            boxes_xyxy: ``(m, 4)`` candidate boxes.
            only_position: See :meth:`gating_distance`.

        Returns:
            ``(n, m)`` squared distances (``inf`` for a non-finite box or state).
        """
        boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        out = np.full((len(means), len(boxes)), np.inf)
        if len(means) == 0 or len(boxes) == 0:
            return out
        k = 2 if only_position else self.ndim_z
        z = self.z_from_xyxy(boxes)[:, :k]
        valid = np.all(np.isfinite(z), axis=1)
        if not np.any(valid):
            return out
        z = z[valid]
        for i, (mean, cov) in enumerate(zip(means, covs)):
            mean = np.asarray(mean, dtype=np.float64).reshape(self.ndim)
            cov = np.asarray(cov, dtype=np.float64).reshape(self.ndim, self.ndim)
            if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
                continue
            z_mean, s = self.project(mean, cov)  # conf=None: never NSA-scaled.
            d = z - z_mean[:k]
            # d_j^T S^-1 d_j for every valid box j.
            out[i, valid] = np.maximum(
                np.sum(d * _solve_spd(s[:k, :k], d.T).T, axis=1), 0.0
            )
        return out


class KalmanFilterXYWH(_KalmanBase):
    """``[cx, cy, w, h]`` constant-velocity filter (BoT-SORT, Deep OC-SORT).

    Noise standard deviations: position ``1/20`` and velocity ``1/160`` of the
    box width (x and w terms) or height (y and h terms).
    ``P0`` std ``[2p w, 2p h, 2p w, 2p h, 10v w, 10v h, 10v w, 10v h]`` (sizes of
    the first measurement), ``Q`` std ``[p w, p h, p w, p h, v w, v h, v w, v h]``
    and ``R`` std ``[p w, p h, p w, p h]`` (sizes of the current state), with
    ``p = 1/20``, ``v = 1/160`` and sizes clamped to >= 1 px. Like Deep OC-SORT,
    prediction zeroes ``vw`` (``vh``) when ``w + vw <= 0`` (``h + vh <= 0``).
    """

    ndim = 8
    _velocity_pairs = ((0, 4), (1, 5), (2, 6), (3, 7))
    _size_velocity_pairs = ((2, 6), (3, 7))

    def z_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray:
        """Convert ``xyxy`` box(es) to ``[cx, cy, w, h]`` (sides >= 1 px).

        Args:
            xyxy: A ``(4,)`` box or ``(..., 4)`` boxes.

        Returns:
            The measurement(s), same leading shape as ``xyxy``.
        """
        return _cxcywh_from_xyxy(xyxy)

    def xyxy_from_mean(self, mean: np.ndarray) -> np.ndarray:
        """Convert state mean(s) to ``xyxy`` box(es) (sides >= 1 px).

        Args:
            mean: A ``(8,)`` state mean or ``(..., 8)`` means.

        Returns:
            The box(es) ``(..., 4)``.
        """
        c = np.array(np.asarray(mean, dtype=np.float64)[..., :4], copy=True)
        c[..., 2:4] = np.maximum(c[..., 2:4], _MIN_SIZE)
        return _xyxy_from_cxcywh(c)

    def _initial_var(self, z: np.ndarray) -> np.ndarray:
        """Initial variances from the first measurement's width and height."""
        w, h = max(z[2], _MIN_SIZE), max(z[3], _MIN_SIZE)
        p, v = 2 * _STD_WEIGHT_POSITION, 10 * _STD_WEIGHT_VELOCITY
        return np.square([p * w, p * h, p * w, p * h, v * w, v * h, v * w, v * h])

    def _process_var(self, means: np.ndarray) -> np.ndarray:
        """Process-noise variances from each state's width and height."""
        w = np.maximum(means[:, 2], _MIN_SIZE)
        h = np.maximum(means[:, 3], _MIN_SIZE)
        whwh = np.stack([w, h, w, h], axis=1)
        return np.square(
            np.concatenate(
                [_STD_WEIGHT_POSITION * whwh, _STD_WEIGHT_VELOCITY * whwh], axis=1
            )
        )

    def _measurement_var(self, mean: np.ndarray) -> np.ndarray:
        """Measurement-noise variances from the state's width and height."""
        w, h = max(mean[2], _MIN_SIZE), max(mean[3], _MIN_SIZE)
        return np.square(_STD_WEIGHT_POSITION * np.array([w, h, w, h]))


class KalmanFilterXYAH(_KalmanBase):
    """``[cx, cy, a, h]`` constant-velocity filter (DeepSORT, StrongSORT, ByteTrack).

    DeepSORT constants with ``p = 1/20``, ``v = 1/160``:
    ``P0`` std ``[2p h, 2p h, 1e-2, 2p h, 10v h, 10v h, 1e-5, 10v h]``,
    ``Q`` std ``[p h, p h, 1e-2, p h, v h, v h, 1e-5, v h]``,
    ``R`` std ``[p h, p h, 1e-1, p h]``.

    ``P0`` uses the first measurement's height, ``Q`` and ``R`` the current
    state's (clamped to >= 1 px). Prediction zeroes ``va`` (``vh``) when
    ``a + va <= 0`` (``h + vh <= 0``); this only fires for degenerate boxes.
    """

    ndim = 8
    _velocity_pairs = ((0, 4), (1, 5), (2, 6), (3, 7))
    _size_velocity_pairs = ((2, 6), (3, 7))

    def z_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray:
        """Convert ``xyxy`` box(es) to ``[cx, cy, w/h, h]`` (sides >= 1 px).

        Args:
            xyxy: A ``(4,)`` box or ``(..., 4)`` boxes.

        Returns:
            The measurement(s), same leading shape as ``xyxy``.
        """
        c = _cxcywh_from_xyxy(xyxy)
        c[..., 2] = c[..., 2] / c[..., 3]
        return c

    def xyxy_from_mean(self, mean: np.ndarray) -> np.ndarray:
        """Convert state mean(s) to ``xyxy`` box(es) (``h`` and ``a*h`` >= 1 px).

        Args:
            mean: A ``(8,)`` state mean or ``(..., 8)`` means.

        Returns:
            The box(es) ``(..., 4)``.
        """
        c = np.array(np.asarray(mean, dtype=np.float64)[..., :4], copy=True)
        h = np.maximum(c[..., 3], _MIN_SIZE)
        c[..., 2] = np.maximum(c[..., 2] * h, _MIN_SIZE)
        c[..., 3] = h
        return _xyxy_from_cxcywh(c)

    def _initial_var(self, z: np.ndarray) -> np.ndarray:
        """Initial variances from the first measurement's height."""
        h = max(z[3], _MIN_SIZE)
        p, v = 2 * _STD_WEIGHT_POSITION * h, 10 * _STD_WEIGHT_VELOCITY * h
        return np.square([p, p, 1e-2, p, v, v, 1e-5, v])

    def _process_var(self, means: np.ndarray) -> np.ndarray:
        """Process-noise variances from each state's height."""
        h = np.maximum(means[:, 3], _MIN_SIZE)
        p, v = _STD_WEIGHT_POSITION * h, _STD_WEIGHT_VELOCITY * h
        a_pos, a_vel = np.full_like(h, 1e-2), np.full_like(h, 1e-5)
        return np.square(np.stack([p, p, a_pos, p, v, v, a_vel, v], axis=1))

    def _measurement_var(self, mean: np.ndarray) -> np.ndarray:
        """Measurement-noise variances from the state's height."""
        p = _STD_WEIGHT_POSITION * max(mean[3], _MIN_SIZE)
        return np.square([p, p, 1e-1, p])


class KalmanFilterXYSR(_KalmanBase):
    """``[cx, cy, s, r]`` filter with 7-D state ``[cx, cy, s, r, vx, vy, vs]`` (SORT, OC-SORT).

    SORT constants: ``R = diag(1, 1, 10, 10)``,
    ``P0 = diag(10, 10, 10, 10, 1e4, 1e4, 1e4)``,
    ``Q = diag(1, 1, 1, 1, 0.01, 0.01, 1e-4)``. Prediction zeroes ``vs`` when it
    would make the area non-positive.
    """

    ndim = 7
    _velocity_pairs = ((0, 4), (1, 5), (2, 6))
    _size_velocity_pairs = ((2, 6),)

    _P0_VAR = np.array([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
    _Q_VAR = np.array([1.0, 1.0, 1.0, 1.0, 1e-2, 1e-2, 1e-4])
    _R_VAR = np.array([1.0, 1.0, 10.0, 10.0])

    def z_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray:
        """Convert ``xyxy`` box(es) to ``[cx, cy, w*h, w/h]`` (sides >= 1 px).

        Args:
            xyxy: A ``(4,)`` box or ``(..., 4)`` boxes.

        Returns:
            The measurement(s), same leading shape as ``xyxy``.
        """
        c = _cxcywh_from_xyxy(xyxy)
        w, h = c[..., 2].copy(), c[..., 3].copy()
        c[..., 2] = w * h
        c[..., 3] = w / h
        return c

    def xyxy_from_mean(self, mean: np.ndarray) -> np.ndarray:
        """Convert state mean(s) to ``xyxy`` box(es) (sides >= 1 px).

        Args:
            mean: A ``(7,)`` state mean or ``(..., 7)`` means.

        Returns:
            The box(es) ``(..., 4)``.
        """
        m = np.asarray(mean, dtype=np.float64)
        s = np.maximum(m[..., 2], _MIN_SIZE)
        r = np.maximum(m[..., 3], _MIN_RATIO)
        w = np.maximum(np.sqrt(s * r), _MIN_SIZE)
        h = np.maximum(np.sqrt(s / r), _MIN_SIZE)
        return _xyxy_from_cxcywh(np.stack([m[..., 0], m[..., 1], w, h], axis=-1))

    def _initial_var(self, z: np.ndarray) -> np.ndarray:
        """Fixed SORT initial variances."""
        return self._P0_VAR.copy()

    def _process_var(self, means: np.ndarray) -> np.ndarray:
        """Fixed SORT process-noise variances."""
        return np.broadcast_to(self._Q_VAR, (len(means), self.ndim))

    def _measurement_var(self, mean: np.ndarray) -> np.ndarray:
        """Fixed SORT measurement-noise variances."""
        return self._R_VAR.copy()


# Filter registry for `make_kalman_filter`.
_KALMAN_FILTERS = {
    "xywh": KalmanFilterXYWH,
    "xyah": KalmanFilterXYAH,
    "xysr": KalmanFilterXYSR,
}


def make_kalman_filter(kind: str, nsa: bool = False) -> _KalmanBase:
    """Return a filter by name: ``"xywh"``, ``"xyah"`` or ``"xysr"``.

    Args:
        kind: Filter name (case-insensitive). ``"auto"`` must be resolved by the
            caller (each tracker has its own default).
        nsa: Enable StrongSORT's NSA measurement-noise scaling.

    Returns:
        A new filter instance.

    Raises:
        ValueError: If ``kind`` is not a known filter.
    """
    key = str(kind).strip().lower()
    if key not in _KALMAN_FILTERS:
        raise ValueError(
            f"Unknown Kalman filter {kind!r}; expected one of "
            f"{', '.join(repr(k) for k in _KALMAN_FILTERS)}."
        )
    return _KALMAN_FILTERS[key](nsa=nsa)


def oru_replay(
    kf: _KalmanBase,
    frozen_mean: np.ndarray,
    frozen_cov: np.ndarray,
    last_obs_xyxy: np.ndarray,
    new_obs_xyxy: np.ndarray,
    n_gap: int,
    conf: Optional[float] = None,
    frozen_steps: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """OC-SORT observation-centric re-update (ORU).

    Restore the state frozen at a track's first miss and replay ``n_gap`` steps of
    predict + update with virtual observations linearly interpolated (in
    ``cx, cy, w, h``) from ``last_obs_xyxy`` to ``new_obs_xyxy``; the last virtual
    observation is ``new_obs_xyxy`` itself. The caller then applies the real
    update (the reference applies the real observation twice; kept for fidelity).

    Virtual observation ``j`` (``1 <= j <= n_gap``) sits ``j / n_gap`` of the way
    from the last observation to the new one. The replay updates the state at
    every step from ``frozen_steps`` on and predicts once between consecutive
    steps, so with the reference freeze (``frozen_steps=1``: the state was saved
    right after the first missed step's prediction) it runs ``n_gap`` updates and
    ``n_gap - 1`` predictions and ends at the new observation's step.

    Args:
        kf: The track's filter.
        frozen_mean: State mean saved at the first miss (after that frame's update).
        frozen_cov: State covariance saved at the first miss.
        last_obs_xyxy: Last real observation before the gap.
        new_obs_xyxy: The re-matched observation.
        n_gap: Steps between the last observation and the new one (>= 1).
        conf: Detection confidence for NSA filters.
        frozen_steps: Steps the frozen state had been predicted past
            ``last_obs_xyxy`` when it was saved (clipped to ``[0, n_gap]``):
            1 (default, the reference) when it was saved after the first missed
            step's prediction; ``k > 1`` when that first miss came after a
            ``k``-step frame gap (virtual observations at earlier steps are
            skipped); 0 when it is the posterior of the last update (one extra
            prediction is run first).

    Returns:
        ``(mean, cov)`` after the replay (ready for the real update).
    """
    n_gap = max(int(n_gap), 1)
    frozen_steps = min(max(int(frozen_steps), 0), n_gap)
    mean = np.array(frozen_mean, dtype=np.float64, copy=True)
    cov = np.array(frozen_cov, dtype=np.float64, copy=True)

    new_obs = np.asarray(new_obs_xyxy, dtype=np.float64).reshape(4)
    start = _cxcywh_from_xyxy(np.asarray(last_obs_xyxy, dtype=np.float64).reshape(4))
    end = _cxcywh_from_xyxy(new_obs)
    if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
        # No virtual trajectory without both endpoints: coast to the new step.
        logger.debug("ORU replay without finite endpoints; coasting instead.")
        for _ in range(n_gap - frozen_steps):
            mean, cov = kf.predict(mean, cov)
        return mean, cov

    if frozen_steps == 0:
        mean, cov = kf.predict(mean, cov)
    first = max(frozen_steps, 1)
    for j in range(first, n_gap + 1):
        if j > first:
            mean, cov = kf.predict(mean, cov)
        if j == n_gap:
            # The last virtual observation is the new observation itself.
            virtual_xyxy = new_obs
        else:
            frac = j / n_gap
            virtual_xyxy = _xyxy_from_cxcywh((1.0 - frac) * start + frac * end)
        mean, cov = kf.update(mean, cov, virtual_xyxy, conf)
    return mean, cov
