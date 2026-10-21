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
"""

from typing import Optional, Tuple

import numpy as np


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

    def __init__(self, nsa: bool = False):
        """Initialize the filter configuration."""
        self.nsa = nsa

    def z_from_xyxy(self, xyxy: np.ndarray) -> np.ndarray:
        """Convert an ``xyxy`` box to this filter's measurement vector."""
        raise NotImplementedError

    def xyxy_from_mean(self, mean: np.ndarray) -> np.ndarray:
        """Convert a state mean to an ``xyxy`` box."""
        raise NotImplementedError

    def center(self, mean: np.ndarray) -> np.ndarray:
        """Box center ``(cx, cy)`` of a state mean."""
        return np.asarray(mean[:2], dtype=np.float64)

    def initiate(self, xyxy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create ``(mean, cov)`` from an initial ``xyxy`` observation."""
        raise NotImplementedError

    def predict(
        self, mean: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """One constant-velocity prediction step."""
        raise NotImplementedError

    def multi_predict(
        self, means: np.ndarray, covs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized :meth:`predict` over ``(n, ndim)`` means and ``(n, ndim, ndim)`` covs."""
        raise NotImplementedError

    def project(
        self, mean: np.ndarray, cov: np.ndarray, conf: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project the state to measurement space, returning ``(z_mean, S)``."""
        raise NotImplementedError

    def update(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        xyxy: np.ndarray,
        conf: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Correct the state with an ``xyxy`` observation."""
        raise NotImplementedError

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
        """
        raise NotImplementedError

    def zero_velocity(self, mean: np.ndarray, indices) -> np.ndarray:
        """Return a copy of ``mean`` with the given velocity components set to 0."""
        out = np.array(mean, dtype=np.float64, copy=True)
        out[list(indices)] = 0.0
        return out


class KalmanFilterXYWH(_KalmanBase):
    """``[cx, cy, w, h]`` constant-velocity filter (BoT-SORT, Deep OC-SORT).

    Noise standard deviations: position ``1/20`` and velocity ``1/160`` of the
    box width (x and w terms) or height (y and h terms).
    """

    ndim = 8


class KalmanFilterXYAH(_KalmanBase):
    """``[cx, cy, a, h]`` constant-velocity filter (DeepSORT, StrongSORT, ByteTrack).

    DeepSORT constants with ``p = 1/20``, ``v = 1/160``:
    ``P0`` std ``[2p h, 2p h, 1e-2, 2p h, 10v h, 10v h, 1e-5, 10v h]``,
    ``Q`` std ``[p h, p h, 1e-2, p h, v h, v h, 1e-5, v h]``,
    ``R`` std ``[p h, p h, 1e-1, p h]``.
    """

    ndim = 8


class KalmanFilterXYSR(_KalmanBase):
    """``[cx, cy, s, r]`` filter with 7-D state ``[cx, cy, s, r, vx, vy, vs]`` (SORT, OC-SORT).

    SORT constants: ``R = diag(1, 1, 10, 10)``,
    ``P0 = diag(10, 10, 10, 10, 1e4, 1e4, 1e4)``,
    ``Q = diag(1, 1, 1, 1, 0.01, 0.01, 1e-4)``. Prediction zeroes ``vs`` when it
    would make the area non-positive.
    """

    ndim = 7


def make_kalman_filter(kind: str, nsa: bool = False) -> _KalmanBase:
    """Return a filter by name: ``"xywh"``, ``"xyah"`` or ``"xysr"``."""
    raise NotImplementedError


def oru_replay(
    kf: _KalmanBase,
    frozen_mean: np.ndarray,
    frozen_cov: np.ndarray,
    last_obs_xyxy: np.ndarray,
    new_obs_xyxy: np.ndarray,
    n_gap: int,
    conf: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """OC-SORT observation-centric re-update (ORU).

    Restore the state frozen at a track's first miss and replay ``n_gap`` steps of
    predict + update with virtual observations linearly interpolated (in
    ``cx, cy, w, h``) from ``last_obs_xyxy`` to ``new_obs_xyxy``; the last virtual
    observation is ``new_obs_xyxy`` itself. The caller then applies the real
    update (the reference applies the real observation twice; kept for fidelity).

    Args:
        kf: The track's filter.
        frozen_mean: State mean saved at the first miss (after that frame's update).
        frozen_cov: State covariance saved at the first miss.
        last_obs_xyxy: Last real observation before the gap.
        new_obs_xyxy: The re-matched observation.
        n_gap: Steps between the last observation and the new one (>= 1).
        conf: Detection confidence for NSA filters.

    Returns:
        ``(mean, cov)`` after the replay (ready for the real update).
    """
    raise NotImplementedError
