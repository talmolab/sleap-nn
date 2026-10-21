"""Per-track appearance memory for the MOT trackers.

Each appearance-using track keeps an :class:`AppearanceMemory`: an exponential
moving average (EMA) of its L2-normalized re-ID vectors plus a bounded gallery of
its most recent CONFIDENT vectors. Updates are quality-gated
(:func:`passes_quality_gate`): a vector from an ambiguous frame (a crossing, a
merged or partial crop, a low-confidence detection) changes neither the EMA nor
the gallery, so one contaminated crop cannot pull the memory toward the wrong
animal (the self-reinforcing swap that made ungated appearance tracking lose on
dense video).
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np

# Norm below which a vector (or an EMA mix) is treated as zero.
_EPS = 1e-12


def _unit(vec) -> Optional[np.ndarray]:
    """Return ``vec`` as a flat float64 unit vector, or ``None`` if unusable.

    ``None``, empty, non-finite or zero-norm input is unusable.
    """
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return None
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= _EPS:
        return None
    return v / norm


def _unit_rows(queries) -> Tuple[np.ndarray, np.ndarray]:
    """Row-normalize ``(n, D)`` (or a single ``(D,)``) queries.

    Returns:
        ``(rows, ok)``: float64 unit rows (zeros where unusable) and an ``(n,)``
        bool marking usable rows (finite with a non-zero norm). A zero row is how
        :class:`~sleap_nn.tracking.mot.detections.FrameDetections` marks a
        missing vector.
    """
    q = np.asarray(queries, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    q = q.reshape(q.shape[0], -1)
    finite = np.isfinite(q).all(axis=1)
    q = np.where(finite[:, None], q, 0.0)
    norms = np.linalg.norm(q, axis=1)
    ok = finite & (norms > _EPS)
    rows = np.zeros_like(q)
    rows[ok] = q[ok] / norms[ok, None]
    return rows, ok


class AppearanceMemory:
    """EMA + gallery of one track's appearance vectors.

    Every stored vector is L2-normalized. Vectors that are missing, non-finite,
    zero-norm or of a different dimension than the memory's are ignored, so a
    detection without a usable vector never changes the memory.

    Attributes:
        ema_alpha: Weight on the old EMA (momentum).
        gallery_size: Maximum gallery length (most recent kept).
        adaptive: Deep OC-SORT score-adaptive EMA: ``trust = clip((s - floor) /
            (1 - floor), 0, 1)`` and ``alpha = ema_alpha + (1 - ema_alpha) * (1 -
            trust)``.
        score_floor: ``floor`` of the adaptive EMA.
        ema: Current EMA vector (L2-normalized) or ``None``.
        gallery: ``deque`` of recent gated vectors.
        n_updates: Number of vectors absorbed (seed included).
    """

    def __init__(
        self,
        ema_alpha: float = 0.8,
        gallery_size: int = 30,
        adaptive: bool = False,
        score_floor: float = 0.5,
    ):
        """Initialize an empty memory."""
        self.ema_alpha = ema_alpha
        self.gallery_size = gallery_size
        self.adaptive = adaptive
        self.score_floor = score_floor
        self.ema: Optional[np.ndarray] = None
        self.gallery: deque = deque(maxlen=gallery_size)
        self.n_updates = 0

    def seed(self, vec: np.ndarray) -> None:
        """Initialize from a track's first vector (always accepted).

        The spawn vector seeds both the EMA and the gallery whatever its quality
        (a track needs an appearance to be compared at all). Any previous content
        is discarded. An unusable vector (``None``, zero-norm, non-finite) leaves
        the memory unchanged.

        Args:
            vec: The ``(D,)`` appearance vector (normalized here).
        """
        v = _unit(vec)
        if v is None:
            return
        self.ema = v.copy()
        self.gallery.clear()
        self.gallery.append(v)
        self.n_updates = 1

    def update(self, vec: np.ndarray, score: float = 1.0, gated: bool = True) -> None:
        """Absorb a matched detection's vector.

        ``gated=False`` (the quality gate failed) leaves both the EMA and the
        gallery unchanged. A memory with no EMA yet is seeded instead (whatever
        ``gated``: like the spawn vector, the first usable vector is always
        accepted). Otherwise the EMA becomes ``renorm(a * ema + (1 - a) * vec)``
        with ``a = ema_alpha`` (or the score-adaptive momentum when ``adaptive``)
        and the vector is appended to the gallery.

        Args:
            vec: The ``(D,)`` appearance vector (normalized here). Unusable or
                wrong-dimension vectors are ignored.
            score: Detection confidence (used by the adaptive EMA only).
            gated: Whether the vector passed the quality gate.
        """
        v = _unit(vec)
        if v is None:
            return
        if self.ema is None:
            self.seed(v)
            return
        if v.size != self.ema.size or not gated:
            return
        a = self._momentum(score)
        mixed = a * self.ema + (1.0 - a) * v
        norm = float(np.linalg.norm(mixed))
        # Antipodal old/new vectors at a = 0.5 cancel out: keep the old EMA.
        if np.isfinite(norm) and norm > _EPS:
            self.ema = mixed / norm
        self.gallery.append(v)
        self.n_updates += 1

    def _momentum(self, score: float) -> float:
        """EMA weight on the old vector for a detection of confidence ``score``."""
        alpha = float(self.ema_alpha)
        if not self.adaptive:
            return alpha
        s = float(score) if score is not None else float("nan")
        floor = float(self.score_floor)
        if not np.isfinite(s):
            trust = 0.0
        elif floor >= 1.0:
            trust = 1.0 if s >= floor else 0.0
        else:
            trust = float(np.clip((s - floor) / (1.0 - floor), 0.0, 1.0))
        # Low confidence -> trust 0 -> momentum 1: the EMA does not move.
        return alpha + (1.0 - alpha) * (1.0 - trust)

    @property
    def is_empty(self) -> bool:
        """Whether the memory holds no vector."""
        return self.ema is None

    def ema_cos(self, queries: np.ndarray) -> np.ndarray:
        """Cosine of ``(n, D)`` normalized queries to the EMA; NaN if empty.

        Args:
            queries: ``(n, D)`` vectors (or one ``(D,)``); renormalized here.

        Returns:
            ``(n,)`` cosines; NaN for every query when the memory is empty or the
            dimension differs, and for unusable (zero / non-finite) query rows.
        """
        rows, ok = _unit_rows(queries)
        out = np.full(rows.shape[0], np.nan)
        if self.ema is None or rows.shape[1] != self.ema.size:
            return out
        out[ok] = np.clip(rows[ok] @ self.ema, -1.0, 1.0)
        return out

    def _gallery_cos(self, queries: np.ndarray) -> Optional[Tuple[np.ndarray, ...]]:
        """``(n, G)`` query-to-gallery cosines and usable-row flags.

        Returns ``None`` when the gallery is empty or of another dimension.
        """
        if len(self.gallery) == 0:
            return None
        gallery = self.vectors()
        rows, ok = _unit_rows(queries)
        if rows.shape[1] != gallery.shape[1]:
            return None
        return np.clip(rows @ gallery.T, -1.0, 1.0), ok

    def gallery_max_cos(self, queries: np.ndarray) -> np.ndarray:
        """Max cosine of each query over the gallery (EMA if the gallery is empty).

        This is DeepSORT's nearest-neighbor appearance statistic.

        Args:
            queries: ``(n, D)`` vectors (or one ``(D,)``); renormalized here.

        Returns:
            ``(n,)`` cosines; NaN when the memory is empty and for unusable
            query rows.
        """
        res = self._gallery_cos(queries)
        if res is None:
            return self.ema_cos(queries)
        cos, ok = res
        out = np.full(cos.shape[0], np.nan)
        out[ok] = cos[ok].max(axis=1)
        return out

    def gallery_topk_mean(self, queries: np.ndarray, k: int = 5) -> np.ndarray:
        """Mean of the top-``k`` gallery cosines per query; NaN if empty.

        With fewer than ``k`` gallery vectors, all of them are averaged; with an
        empty gallery (e.g. ``gallery_size=0``) the EMA cosine is used.

        Args:
            queries: ``(n, D)`` vectors (or one ``(D,)``); renormalized here.
            k: Number of best gallery matches averaged per query.

        Returns:
            ``(n,)`` statistics; NaN when the memory is empty and for unusable
            query rows.
        """
        res = self._gallery_cos(queries)
        if res is None:
            return self.ema_cos(queries)
        cos, ok = res
        k_eff = int(min(max(int(k), 1), cos.shape[1]))
        top = np.partition(cos, cos.shape[1] - k_eff, axis=1)[:, -k_eff:]
        out = np.full(cos.shape[0], np.nan)
        out[ok] = top[ok].mean(axis=1)
        return out

    def vectors(self) -> np.ndarray:
        """Gallery as an ``(G, D)`` array (``(0, 0)`` when empty)."""
        if len(self.gallery) == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return np.stack(list(self.gallery), axis=0)


def passes_quality_gate(
    det_score: float,
    n_visible: int,
    n_nodes: int,
    max_iou_other: float,
    margin: float,
    *,
    high_threshold: float,
    min_margin: Optional[float],
    gallery_len: int,
    min_visible_fraction: float = 0.5,
    max_overlap_iou: float = 0.1,
) -> bool:
    """Decide whether a matched detection's vector may update its track's memory.

    All must hold: ``det_score >= high_threshold``; ``n_visible / n_nodes >=
    min_visible_fraction``; ``max_iou_other < max_overlap_iou`` (padded-box IoU to
    every other detection in the frame and every other live track's predicted
    box); and, when ``min_margin`` is not None and ``gallery_len >= 3``, the
    appearance margin (cosine to the matched track minus the best cosine to any
    other in-gate track; ``inf`` when there is no competitor) is ``>= min_margin``.

    Args:
        det_score: Detection confidence.
        n_visible: Visible nodes of the detection (masks: 1).
        n_nodes: Nodes per detection (masks: 1).
        max_iou_other: Largest padded-box IoU to any other box (pass 0.0 when
            there is none; NaN counts as no overlap).
        margin: Appearance margin of the match (NaN fails the margin test).
        high_threshold: Minimum detection confidence.
        min_margin: Minimum appearance margin, or ``None`` to skip the test.
        gallery_len: Current length of the track's gallery; the margin test is
            skipped while it is shorter than 3 (a young memory has no reliable
            margin yet).
        min_visible_fraction: Minimum fraction of visible nodes.
        max_overlap_iou: Overlap (IoU) at or above which the crop may be merged
            with a neighbor's.

    Returns:
        Whether the vector may update the EMA and the gallery.
    """
    if not float(det_score) >= float(high_threshold):
        return False
    if float(n_visible) < float(min_visible_fraction) * max(int(n_nodes), 1):
        return False
    overlap = float(max_iou_other)
    if not np.isnan(overlap) and not overlap < float(max_overlap_iou):
        return False
    if min_margin is not None and int(gallery_len) >= 3:
        # NaN compares False: an undefined margin fails.
        if not float(margin) >= float(min_margin):
            return False
    return True
