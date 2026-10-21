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
from typing import Optional

import numpy as np


class AppearanceMemory:
    """EMA + gallery of one track's appearance vectors.

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
        """Initialize from a track's first vector (always accepted)."""
        raise NotImplementedError

    def update(self, vec: np.ndarray, score: float = 1.0, gated: bool = True) -> None:
        """Absorb a matched detection's vector.

        ``gated=False`` (the quality gate failed) leaves both the EMA and the
        gallery unchanged. A memory with no EMA yet is seeded instead.
        """
        raise NotImplementedError

    @property
    def is_empty(self) -> bool:
        """Whether the memory holds no vector."""
        return self.ema is None

    def ema_cos(self, queries: np.ndarray) -> np.ndarray:
        """Cosine of ``(n, D)`` normalized queries to the EMA; NaN if empty."""
        raise NotImplementedError

    def gallery_max_cos(self, queries: np.ndarray) -> np.ndarray:
        """Max cosine of each query over the gallery (EMA if the gallery is empty)."""
        raise NotImplementedError

    def gallery_topk_mean(self, queries: np.ndarray, k: int = 5) -> np.ndarray:
        """Mean of the top-``k`` gallery cosines per query; NaN if empty."""
        raise NotImplementedError

    def vectors(self) -> np.ndarray:
        """Gallery as an ``(G, D)`` array (``(0, 0)`` when empty)."""
        raise NotImplementedError


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
    """
    raise NotImplementedError
