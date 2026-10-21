"""Shared machinery of the MOT trackers.

:class:`BaseMOTTracker` implements the sleap-nn tracker interface used by
:func:`sleap_nn.inference.tracking.apply_tracking` --
``track(untracked_instances, frame_idx, image) -> list``, ``_track_objects`` and
``track_name_offset`` -- plus an optional per-video ``prepare()`` pre-pass
(detection extraction + :mod:`calibration <sleap_nn.tracking.mot.calibration>`)
and ``finalize()`` post-pass (identity linking). Subclasses implement one method,
:meth:`BaseMOTTracker._associate`, with their algorithm's matching cascade, and
set class attributes for their lifecycle semantics.

Contract (every tracker, tested generically):

1. ``track()`` returns EVERY input object, in input order, each with ``.track``
   (a ``sio.Track`` or explicitly ``None``) and ``.tracking_score`` (in
   ``[0, 1]``; 0.0 when untracked). Stale tracks from a previous run are cleared.
2. Empty frames are fine (time still advances); degenerate input never raises.
3. ``track()`` works without ``prepare()`` / ``finalize()`` (uncalibrated:
   ``"auto"`` appearance is disabled, degenerate boxes use a running body size).
4. Time advances in steps (see :class:`~sleap_nn.tracking.mot.params.MOTParams`
   ``time_base``); Kalman prediction, ``time_since_update``, ``track_buffer`` and
   ``delta_t`` all count steps.
5. A track's detections are labeled when it is confirmed, retroactively for its
   tentative frames; a track that never confirms leaves its detections untracked.
6. ``image`` is ignored (fixed cameras: no camera-motion compensation).
7. Public ids (``sio.Track`` names ``track_{id + track_name_offset}``) are minted
   at confirmation, so names are compact.
8. ``tracking_score`` is the GEOMETRIC similarity of the matched pair (1.0 at
   spawn), comparable across trackers.
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import attrs
import numpy as np
import sleap_io as sio

from sleap_nn.tracking.mot.calibration import VideoCalibration
from sleap_nn.tracking.mot.detections import FrameDetections
from sleap_nn.tracking.mot.params import MOTParams
from sleap_nn.tracking.mot.track import MOTTrack, TrackState


@attrs.define
class AssociationResult:
    """What a tracker's :meth:`BaseMOTTracker._associate` decided for one frame.

    Attributes:
        matches: ``(track, det_index, geometric_similarity)`` per matched pair.
            A LOST track in a match is re-activated (same id).
        no_spawn: Detection indices that must not start a track even if
            unmatched (e.g. low-score detections consumed by a second stage).
        removed: Tracks the algorithm removed this frame (e.g. ByteTrack's
            unmatched tentative tracks, duplicate removal).
    """

    matches: List[Tuple[MOTTrack, int, float]] = attrs.field(factory=list)
    no_spawn: Set[int] = attrs.field(factory=set)
    removed: List[MOTTrack] = attrs.field(factory=list)


class BaseMOTTracker:
    """Base class of the MOT trackers (see module docstring for the contract).

    Class attributes (set by subclasses):
        name: Tracker name (registry key).
        default_kalman: Filter used when ``params.kalman == "auto"``.
        uses_appearance: The algorithm associates by appearance vectors.
        requires_vectors: Appearance is the algorithm's primary cue (raise when
            the tracked detections carry no vectors).
        tentative_dies_on_miss: A tentative track is removed at its first miss
            (DeepSORT/ByteTrack/BoT-SORT) instead of coasting until
            ``track_buffer`` (SORT/OC-SORT/Deep OC-SORT).
        hits_start: ``hits``/``hit_streak`` of a newly spawned track (SORT-family
            reference: 0, so ``min_hits=3`` needs 4 consecutive detections).
        confirm_on_streak: Confirm on ``hit_streak >= min_hits`` (SORT family)
            instead of ``hits >= min_hits``.
        use_oru: Freeze the state at the first miss and apply OC-SORT's
            observation-centric re-update when re-matched.
        lost_zero_velocity: State indices zeroed before predicting a track that is
            not ``CONFIRMED`` (ByteTrack: ``(7,)``; BoT-SORT: ``(6, 7)``).

    Attributes:
        params: Resolved :class:`MOTParams`.
        carrier: ``"pose"`` or ``"mask"``.
        n_nodes: Skeleton size (pose), ``None`` if unknown.
        kf: The Kalman filter in use.
        calibration: Per-video :class:`VideoCalibration` (``None`` before
            ``prepare``).
        tracks: Live (not removed) tracks.
        finished: Removed tracks (kept for identity linking).
        track_name_offset: Added to public ids when naming ``sio.Track``s.
        resolved: JSON-safe record of what ran (resolved params, time base,
            calibration, appearance decisions, linking result) for provenance.
    """

    name: ClassVar[str] = ""
    default_kalman: ClassVar[str] = "xywh"
    uses_appearance: ClassVar[bool] = False
    requires_vectors: ClassVar[bool] = False
    tentative_dies_on_miss: ClassVar[bool] = True
    hits_start: ClassVar[int] = 1
    confirm_on_streak: ClassVar[bool] = False
    use_oru: ClassVar[bool] = False
    lost_zero_velocity: ClassVar[Tuple[int, ...]] = ()

    def __init__(
        self,
        params: MOTParams,
        carrier: str = "pose",
        n_nodes: Optional[int] = None,
    ):
        """Create a tracker for one video."""
        raise NotImplementedError

    # ------------------------------------------------------------------ public
    track_name_offset: int = 0
    _track_objects: Dict[int, sio.Track]

    def prepare(self, frames: Sequence[Tuple[int, Sequence[Any]]]) -> None:
        """Optional per-video pre-pass over every frame that will be tracked.

        Extracts (and caches, keyed by frame index and object identities) each
        frame's :class:`FrameDetections`, calibrates the video
        (:func:`~sleap_nn.tracking.mot.calibration.calibrate`), resolves the
        time base and the pseudo-box size for degenerate geometry, and records
        everything in ``resolved``. ``prepare([])`` is a no-op.

        Args:
            frames: ``(frame_idx, objects)`` per frame, in temporal order, holding
                exactly the objects ``track()`` will receive.
        """
        raise NotImplementedError

    def track(
        self, untracked_instances: Sequence[Any], frame_idx: int, image: Any = None
    ) -> list:
        """Track one frame (see the module contract). Returns every input object."""
        raise NotImplementedError

    def finalize(self) -> None:
        """End-of-video pass: identity linking when enabled. Idempotent."""
        raise NotImplementedError

    # --------------------------------------------------------- subclass hook
    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """Match the live tracks (already predicted to this step) to ``dets``."""
        raise NotImplementedError

    # ------------------------------------------------------ helpers (shared)
    def _live(self, *states: TrackState) -> List[MOTTrack]:
        """Live tracks, optionally restricted to ``states``, in ``tid`` order."""
        raise NotImplementedError

    def _geometric_similarity(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections
    ) -> np.ndarray:
        """``(n_tracks, n_dets)`` similarity of the tracks' PREDICTIONS to ``dets``.

        Kind from ``params.similarity`` (``"auto"`` resolved per carrier and
        skeleton): ``iou`` on padded boxes, ``oks`` against the predicted pose,
        ``centroid`` on box centers with scale ``body_size / 2``. Invalid
        detections score 0.
        """
        raise NotImplementedError

    def _last_observation_similarity(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections
    ) -> np.ndarray:
        """Similarity of each track's LAST OBSERVATION to ``dets`` (OC-SORT OCR).

        Same kinds and conventions as :meth:`_geometric_similarity`.
        """
        raise NotImplementedError

    def _appearance_cos(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections, statistic: str = "ema"
    ) -> np.ndarray:
        """``(n_tracks, n_dets)`` appearance cosines; NaN where unavailable.

        ``statistic`` is ``"ema"`` (cosine to the track's EMA) or
        ``"gallery_max"`` (max over its gallery, EMA if empty). All NaN when
        appearance is disabled for association (see :meth:`_appearance_gate`).
        """
        raise NotImplementedError

    def _appearance_gate(self, statistic: str = "ema") -> Optional[float]:
        """Minimum cosine for appearance to support a match, or ``None`` if disabled.

        ``appearance_threshold="auto"``: the calibrated gate for ``statistic``
        when ``calibration.appearance_for_association``, else ``None``. A float:
        that value (when the video has vectors).
        """
        raise NotImplementedError

    def _appearance_margin(self) -> float:
        """Minimum row/column cosine margin for appearance to decide a pair.

        Calibrated ``m_assoc`` when available, else 0.
        """
        raise NotImplementedError
