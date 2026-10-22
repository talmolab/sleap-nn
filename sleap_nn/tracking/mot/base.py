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
   ``"auto"`` appearance is disabled, scores are not normalized, the stride and
   time base are inferred online from the frame deltas seen so far, and the
   body size is a running estimate with calibration's fallback chain
   (:func:`~sleap_nn.tracking.mot.calibration.resolve_body_size`): box
   diagonals, else per-step motion, else spacing, else its 50 px default).
4. Time advances in steps (see :class:`~sleap_nn.tracking.mot.params.MOTParams`
   ``time_base``); Kalman prediction, ``time_since_update``, ``track_buffer`` and
   ``delta_t`` all count steps, with the shared step rule of
   :func:`~sleap_nn.tracking.mot.calibration.steps_between` (also used by the
   calibration and the identity linker).
5. A track's detections are labeled when it is confirmed, retroactively for its
   tentative frames; a track that never confirms leaves its detections untracked.
6. ``image`` is ignored (fixed cameras: no camera-motion compensation).
7. Public ids (``sio.Track`` names ``track_{id + track_name_offset}``) are minted
   at confirmation, so names are compact.
8. ``tracking_score`` is the GEOMETRIC similarity of the matched pair (1.0 at
   spawn), comparable across trackers.
9. A tracker tracks ONE video, in increasing frame order (``apply_tracking``
   builds one per video and sorts its frames). A repeated or backwards frame
   index counts one step and logs a WARNING (once); ``prepare()`` warns too.

Frame gaps: in the ``"frames"`` time base a gap of ``n`` steps behaves like ``n - 1``
empty processed frames followed by this one (the dense equivalent): a track that
would have been removed during the gap is removed before association, a confirmed
track that missed the gap is ``LOST``, a tentative track of a family whose tentative
tracks die on a miss dies, and the hit streak restarts.
"""

from collections import deque
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import attrs
import numpy as np
import sleap_io as sio
from loguru import logger

from sleap_nn.tracking.mot.appearance import (
    AppearanceMemory,
    passes_quality_gate,
    unit_rows,
)
from sleap_nn.tracking.mot.association import (
    centroid_similarity,
    cosine_matrix,
    iou_matrix,
    oks_matrix,
    speed_direction,
)
from sleap_nn.tracking.mot.calibration import (
    VideoCalibration,
    calibrate,
    pair_geometry,
    resolve_body_size,
    resolve_time_base,
    score_normalizer,
    steps_between,
    stride_from_counts,
)
from sleap_nn.tracking.mot.detections import (
    CARRIERS,
    FrameDetections,
    extract_detections,
    normalize_scores,
    widen_boxes,
)
from sleap_nn.tracking.mot.kalman import make_kalman_filter, oru_replay
from sleap_nn.tracking.mot.linking import link_tracklets
from sleap_nn.tracking.mot.params import MOTParams
from sleap_nn.tracking.mot.track import MOTTrack, TrackState

# Samples (raw-box diagonals, same-frame spacings, per-step link lengths)
# remembered for the running body-size estimate (no calibration).
BODY_SIZE_HISTORY = 2048

# Uncalibrated: frame deltas seen before an "irregular spacing" verdict is worth a
# warning (the first few deltas of a dense stream with drops look irregular).
MIN_DELTAS_TO_WARN = 10

# A LOST track still counts as a neighbor in the appearance quality gate's
# isolation test while it has been missing for at most this many steps.
GATE_LOST_MAX_STEPS = 5

# Padded boxes are floored at this fraction of the body size on each axis, so a
# thin pose (nodes nearly collinear) does not get a sliver box whose IoU collapses
# under a few pixels of motion.
MIN_BOX_FRACTION = 0.25

# Appearance statistics understood by `_appearance_cos` / `_appearance_gate`.
APPEARANCE_STATISTICS = ("ema", "gallery_max")


@attrs.define
class AssociationResult:
    """What a tracker's :meth:`BaseMOTTracker._associate` decided for one frame.

    Attributes:
        matches: ``(track, det_index, geometric_similarity)`` per matched pair.
            A LOST track in a match is re-activated (same id). The similarity
            must be the pair's GEOMETRIC similarity (``_geometric_similarity`` or
            ``_last_observation_similarity``), not a fused / momentum /
            appearance score: it becomes the detection's ``tracking_score``.
            Matches to invalid detections or repeated tracks/detections are
            ignored.
        no_spawn: Detection indices that must not start a track even if
            unmatched (e.g. low-score detections consumed by a second stage).
        removed: Tracks the algorithm removed this frame (e.g. ByteTrack's
            unmatched tentative tracks, duplicate removal). A removed track's
            detection matched this frame stays untracked.
    """

    matches: List[Tuple[MOTTrack, int, float]] = attrs.field(factory=list)
    no_spawn: Set[int] = attrs.field(factory=set)
    removed: List[MOTTrack] = attrs.field(factory=list)


def _set_label(obj: Any, track: Optional[sio.Track], score: float) -> None:
    """Write ``.track`` / ``.tracking_score`` onto a detection (never raises)."""
    try:
        obj.track = track
        obj.tracking_score = float(score)
    except (AttributeError, TypeError):
        logger.debug(f"Could not write a track onto {type(obj).__name__}.")


def _clip_score(value: float) -> float:
    """A similarity as a tracking score: finite, in ``[0, 1]``."""
    value = float(value)
    return min(max(value, 0.0), 1.0) if np.isfinite(value) else 0.0


def _box_area(boxes: np.ndarray) -> np.ndarray:
    """Areas of ``(n, 4)`` ``xyxy`` boxes (NaN for non-finite boxes)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )


def _pad_nodes(kp: np.ndarray, n_nodes: int) -> np.ndarray:
    """NaN-pad ``(n, K, 2)`` keypoints to ``n_nodes`` nodes."""
    if kp.shape[1] >= n_nodes:
        return kp
    out = np.full((kp.shape[0], n_nodes, 2), np.nan, dtype=np.float64)
    out[:, : kp.shape[1]] = kp
    return out


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
        keeps_observation_history: Maintain OC-SORT's observation history on
            every match: ``observations`` (the last ``2 * delta_t + 1``) and the
            ``velocity`` direction its OCM term reads (OC-SORT family). Off for
            the other trackers, which never read them; ``last_observation`` is
            kept by every tracker.
        lost_zero_velocity: State indices zeroed before predicting a track that is
            not ``CONFIRMED`` (ByteTrack: ``(7,)``; BoT-SORT: ``(6, 7)``). Precisely:
            a track that was not matched at the previous step (the references'
            "not Tracked"); a live tentative track of those families always was.

    Attributes:
        params: Resolved :class:`MOTParams`.
        carrier: ``"pose"`` or ``"mask"``.
        n_nodes: Skeleton size (pose), ``None`` if unknown.
        kf: The Kalman filter in use.
        kalman_kind: Its name (``"xywh"``, ``"xyah"`` or ``"xysr"``).
        calibration: Per-video :class:`VideoCalibration` (``None`` before
            ``prepare``).
        tracks: Live (not removed) tracks.
        n_removed: Number of tracks removed so far. Removed tracks are dropped
            (identity linking re-reads the processed frames, not the tracks), so
            memory does not grow with track turnover.
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
    keeps_observation_history: ClassVar[bool] = False
    lost_zero_velocity: ClassVar[Tuple[int, ...]] = ()

    def __init__(
        self,
        params: MOTParams,
        carrier: str = "pose",
        n_nodes: Optional[int] = None,
    ):
        """Create a tracker for one video.

        Args:
            params: Resolved parameters (see :meth:`MOTParams.for_tracker`).
            carrier: ``"pose"`` or ``"mask"``.
            n_nodes: Skeleton size for pose detections (``None`` if unknown; then
                taken from the detections).

        Raises:
            ValueError: On an unknown carrier, or parameters resolved for another
                tracker.
        """
        if carrier not in CARRIERS:
            raise ValueError(f"carrier must be one of {CARRIERS}, got {carrier!r}.")
        if self.name and params.tracker != self.name:
            raise ValueError(
                f"{type(self).__name__} needs parameters resolved for "
                f"{self.name!r}, got parameters for {params.tracker!r}."
            )
        self.params = params
        self.carrier = carrier
        self.n_nodes = None if n_nodes is None else int(n_nodes)
        self.kalman_kind = (
            params.kalman if params.kalman != "auto" else self.default_kalman
        )
        self.kf = make_kalman_filter(self.kalman_kind, nsa=params.nsa)

        self.calibration: Optional[VideoCalibration] = None
        self.tracks: List[MOTTrack] = []
        self.n_removed = 0
        self.track_name_offset = 0
        self._track_objects: Dict[int, sio.Track] = {}

        self._next_tid = 0
        self._next_public_id = 0
        # Step index of the latest processed frame (-1: nothing processed yet).
        self._step = -1
        self._last_frame_idx: Optional[int] = None
        # Frame-index delta to the previous processed frame (None at the first).
        self._last_delta: Optional[int] = None
        # Until `prepare` (or without it), the stride and time base are inferred
        # online from the positive frame deltas seen so far (`_infer_stride`).
        self._time_base = resolve_time_base(params.time_base, True)
        self._frame_step = 1
        self._delta_counts: Dict[int, int] = {}

        self._similarity_kind: Optional[str] = None
        if carrier == "mask" or self.n_nodes is not None:
            self._similarity_kind = self._resolve_similarity(self.n_nodes)

        # Running body-size evidence without calibration: raw-box diagonals,
        # else (extent-less detections) same-frame spacings and per-step lengths
        # of unambiguous frame-to-frame links, with the previous frame for them.
        self._diagonals: deque = deque(maxlen=BODY_SIZE_HISTORY)
        self._spacings: deque = deque(maxlen=BODY_SIZE_HISTORY)
        self._step_lengths: deque = deque(maxlen=BODY_SIZE_HISTORY)
        self._prev_dets: Optional[FrameDetections] = None
        self._embedding_dim: Optional[int] = None
        self._vectors_seen = False
        # Divisor of every detection score (`normalize_scores`; set by `prepare`).
        self._score_normalizer: Optional[float] = None
        self._n_valid_seen = 0
        self._warned: Set[str] = set()

        # `prepare` cache: frame_idx -> FrameDetections (validated by identity);
        # each entry is dropped when its frame is tracked.
        self._cache: Dict[int, FrameDetections] = {}
        # Every processed frame's objects, for identity linking in `finalize`
        # (kept only when linking runs).
        self._seen_frames: List[Tuple[int, list]] = []
        self._n_frames_seen = 0
        self._finalized = False
        self.resolved: Dict[str, Any] = {}
        self._refresh_resolved()

    # ------------------------------------------------------------------ public
    track_name_offset: int = 0
    _track_objects: Dict[int, sio.Track]

    @property
    def next_track_name_offset(self) -> int:
        """``track_name_offset`` for the NEXT video's tracker.

        Every name this tracker minted (``track_{public_id + offset}``) is below
        it, so per-video trackers chained with it never reuse a name.
        """
        return int(self.track_name_offset) + int(self._next_public_id)

    def prepare(self, frames: Sequence[Tuple[int, Sequence[Any]]]) -> None:
        """Optional per-video pre-pass over every frame that will be tracked.

        Extracts (and caches, keyed by frame index and object identities) each
        frame's :class:`FrameDetections`, calibrates the video
        (:func:`~sleap_nn.tracking.mot.calibration.calibrate`), resolves the
        time base, the body size (pseudo-box size of degenerate geometry and the
        ``centroid`` similarity scale) and, with ``normalize_scores``, the score
        normalizer applied to every detection from then on; records everything
        in ``resolved``. ``prepare([])`` is a no-op.

        Args:
            frames: ``(frame_idx, objects)`` per frame of ONE video, in
                increasing frame order, holding exactly the objects ``track()``
                will receive. Repeated or backwards frame indices are logged
                (WARNING): the calibration would measure motion across them.
        """
        frames = [(int(fi), list(objs)) for fi, objs in frames]
        if not frames:
            return
        p = self.params
        frame_idx = np.array([fi for fi, _ in frames], dtype=np.int64)
        n_bad = int(np.count_nonzero(np.diff(frame_idx) <= 0))
        if n_bad:
            logger.warning(
                f"{self.name}: prepare() got {n_bad} repeated or backwards frame "
                "index step(s). A MOT tracker tracks ONE video in increasing frame "
                "order: build one tracker per video and sort its frames (as "
                "apply_tracking does). The calibration measures motion across "
                "these jumps and tracks may continue across them."
            )

        # First pass: raw scores (the calibration measures them) and geometry
        # without a pseudo box (it measures the body size on non-degenerate
        # detections only).
        self._score_normalizer = None
        extracted = [
            self._extract(objs, fi, pseudo_box_size=None) for fi, objs in frames
        ]
        if self._similarity_kind is None and self.carrier == "pose":
            n_nodes = [d.n_nodes for d in extracted if len(d) > 0]
            if n_nodes:
                self._similarity_kind = self._resolve_similarity(max(n_nodes))

        # The g = 1 trials (gallery_cos_gate) use the gallery length the
        # tracker's memory actually keeps (DeepSORT: gallery_budget); the
        # appearance part is skipped when neither association nor linking
        # reads it.
        cal = calibrate(
            extracted,
            track_buffer=p.track_buffer,
            gallery_size=self._make_appearance_memory().gallery_size,
            ema_alpha=p.ema_alpha,
            time_base=p.time_base,
            body_size=p.body_size,
            appearance=self.uses_appearance or p.link,
        )
        self.calibration = cal

        # Scores: normalized per video, or a warning when the raw scores of
        # most detections cannot start a track.
        q95 = cal.score_quantiles[-1] if cal.score_quantiles else None
        if p.normalize_scores:
            self._score_normalizer = score_normalizer(cal)
        if self._score_normalizer is not None and self._score_normalizer != 1.0:
            logger.info(
                f"{self.name}: detection scores divided by the video's score q95 "
                f"({self._score_normalizer:.3f}) and clipped to [0, 1] "
                "(normalize_scores)."
            )
        if not p.normalize_scores and q95 is not None and q95 < p.new_track_threshold:
            logger.warning(
                f"{self.name}: 95% of this video's detections score below "
                f"new_track_threshold={p.new_track_threshold} (score q95 "
                f"{q95:.3f}): most detections can never start a track; lower "
                "--new_track_threshold or use the animal preset "
                "(normalize_scores=True)."
            )

        # Degenerate detections (single node, sub-pixel extent) get the body
        # size as their box; every padded box is floored at MIN_BOX_FRACTION of
        # it. Scores are normalized exactly once (`_extract` does it for the
        # re-extracted frames).
        body = self._body_size()
        for i, dets in enumerate(extracted):
            if self._has_degenerate(dets):
                extracted[i] = self._extract(
                    dets.sources, dets.frame_idx, pseudo_box_size=body
                )
            else:
                widen_boxes(extracted[i], MIN_BOX_FRACTION * body)
                normalize_scores(extracted[i], self._score_normalizer)
        self._cache = {dets.frame_idx: dets for dets in extracted}

        # Time base: the calibration's (it resolved `p.time_base` with the same
        # rule), and its stride in the frames time base.
        self._time_base = cal.time_base
        if p.time_base == "auto" and self._time_base == "steps":
            logger.warning(
                "The frames to track are irregularly spaced (the most common "
                f"frame step, {cal.frame_step}, covers less than half of the "
                "gaps): counting one step per processed frame (time_base="
                "'steps'). Kalman prediction, track_buffer and min_hits then "
                "count processed frames, not video frames."
            )
        self._frame_step = (
            max(1, int(cal.frame_step)) if self._time_base == "frames" else 1
        )

        if self.uses_appearance:
            if not self._vectors_seen:
                logger.warning(
                    f"{self.name}: the tracked detections carry no appearance "
                    "vectors; associating on geometry alone."
                )
            elif p.appearance_threshold == "auto" and self._appearance_gate() is None:
                logger.warning(
                    f"{self.name}: appearance is disabled for association by the "
                    f"video's calibration (status={cal.status!r}, assoc_rank1="
                    f"{cal.assoc_rank1}); associating on geometry alone."
                )
        self._refresh_resolved()

    def track(
        self, untracked_instances: Sequence[Any], frame_idx: int, image: Any = None
    ) -> list:
        """Track one frame (see the module contract). Returns every input object."""
        objs = list(untracked_instances)
        frame_idx = int(frame_idx)
        for obj in objs:
            _set_label(obj, None, 0.0)
        self._n_frames_seen += 1
        if self.params.link:
            self._seen_frames.append((frame_idx, objs))

        dets = self._detections(objs, frame_idx)
        n_steps = self._advance(frame_idx)
        if self.calibration is None:
            # Before association, so this frame already uses the evidence.
            self._observe_uncalibrated(dets, n_steps)
        self._n_valid_seen += int(np.count_nonzero(dets.valid))

        if len(dets) == 0:
            # Nothing to match; the bookkeeping below still ages every track.
            result = AssociationResult()
        else:
            result = self._associate(dets)

        pool = list(self.tracks)
        removed_tids = {t.tid for t in result.removed}
        gate_ctx = self._gate_context(pool, dets, result)

        # Matches: Kalman update, counters, history, members, appearance.
        applied: List[Tuple[MOTTrack, int, float]] = []
        used_tids: Set[int] = set()
        used_dets: Set[int] = set()
        for track, det_idx, sim in result.matches:
            det_idx = int(det_idx)
            if (
                track.tid in used_tids
                or det_idx in used_dets
                or not track.is_live
                or not 0 <= det_idx < len(dets)
                or not dets.valid[det_idx]
            ):
                logger.debug(
                    f"Ignoring an invalid match (track {track.tid}, detection "
                    f"{det_idx}) at frame {frame_idx}."
                )
                continue
            used_tids.add(track.tid)
            used_dets.add(det_idx)
            sim = _clip_score(sim)
            self._apply_match(track, dets, det_idx, sim, gate_ctx)
            applied.append((track, det_idx, sim))

        # Unmatched live tracks: freeze for ORU, CONFIRMED -> LOST, tentative
        # tracks of the DeepSORT/ByteTrack families die.
        for track in pool:
            if track.tid in used_tids or track.tid in removed_tids:
                continue
            self._mark_missed(track)
        for track in result.removed:
            self._remove(track)

        # Spawn tentative tracks from unmatched, valid, confident detections.
        for det_idx in range(len(dets)):
            if (
                det_idx in used_dets
                or det_idx in result.no_spawn
                or not dets.valid[det_idx]
                or dets.scores[det_idx] < self.params.new_track_threshold
            ):
                continue
            self._spawn(dets, det_idx)

        # Confirm tentative tracks that reached `min_hits` (retro-labels members).
        for track in self.tracks:
            if track.state == TrackState.TENTATIVE and track.time_since_update == 0:
                count = track.hit_streak if self.confirm_on_streak else track.hits
                if count >= self.params.min_hits:
                    self._confirm(track)

        # Remove tracks unmatched for longer than the buffer (any state).
        for track in list(self.tracks):
            if track.time_since_update > self.params.track_buffer:
                self._remove(track)

        # Label this frame's detections of confirmed tracks.
        for track, det_idx, sim in applied:
            if track.state == TrackState.CONFIRMED and track.public_id is not None:
                _set_label(
                    dets.sources[det_idx], self._track_objects[track.public_id], sim
                )
        return list(objs)

    def finalize(self) -> None:
        """End-of-video pass: identity linking when enabled. Idempotent.

        Logs a WARNING when valid detections were tracked but no track was ever
        confirmed (every detection came back untracked).
        """
        if self._finalized:
            return
        self._finalized = True
        p = self.params
        if self._n_valid_seen and self._next_public_id == 0:
            logger.warning(
                f"{self.name}: {self._n_valid_seen} valid detection(s) over "
                f"{self._n_frames_seen} frame(s), but no track was ever "
                "confirmed: every detection is untracked. Check the score "
                f"thresholds (new_track_threshold={p.new_track_threshold}, "
                f"track_low_threshold={p.track_low_threshold}; normalize_scores="
                f"{p.normalize_scores}), min_hits={p.min_hits} and the frame "
                "order / time base"
                + (
                    "; without prepare() scores are not normalized and nothing is "
                    "calibrated."
                    if self.calibration is None
                    else "."
                )
            )
        if p.link:
            result = link_tracklets(
                self._seen_frames,
                carrier=self.carrier,
                calibration=self.calibration,
                max_tracks=p.max_tracks,
                track_buffer=p.track_buffer,
                swap_correction=p.link_swap_correction,
                spatial_gate_frames=p.link_spatial_gate_frames,
                box_padding=p.box_padding,
                high_threshold=p.track_high_threshold,
                score_mode=p.score_mode,
                score_normalizer=self._score_normalizer,
                time_base=p.time_base,
                body_size=p.body_size,
                track_objects=self._track_objects,
            )
            self.resolved["linking"] = result.as_dict()
        self._refresh_resolved()

    # --------------------------------------------------------- subclass hook
    def _associate(self, dets: FrameDetections) -> AssociationResult:
        """Match the live tracks (already predicted to this step) to ``dets``."""
        raise NotImplementedError

    # ------------------------------------------------------ helpers (shared)
    def _live(self, *states: TrackState) -> List[MOTTrack]:
        """Live tracks, optionally restricted to ``states``, in ``tid`` order."""
        if not states:
            return list(self.tracks)
        return [t for t in self.tracks if t.state in states]

    def _geometric_similarity(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections
    ) -> np.ndarray:
        """``(n_tracks, n_dets)`` similarity of the tracks' PREDICTIONS to ``dets``.

        Kind from ``params.similarity`` (``"auto"`` resolved per carrier and
        skeleton): ``iou`` on padded boxes, ``oks`` against the predicted pose,
        ``centroid`` on box centers with scale ``body_size / 2``. Invalid
        detections score 0.
        """
        tracks = list(tracks)
        n, m = len(tracks), len(dets)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)
        means = np.stack([t.mean for t in tracks])
        kind = self._similarity(dets)
        if kind == "oks" and dets.keypoints is not None:
            pred_kp, areas = self._predicted_poses(tracks, means)
            sim = self._oks(dets.keypoints, pred_kp, areas)
        elif kind == "centroid":
            centers = np.asarray(means[:, :2], dtype=np.float64)
            sim = centroid_similarity(centers, dets.centroids, 0.5 * self._body_size())
        else:
            sim = iou_matrix(self.kf.xyxy_from_mean(means), dets.boxes)
        return self._finish_similarity(sim, dets)

    def _last_observation_similarity(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections
    ) -> np.ndarray:
        """Similarity of each track's LAST OBSERVATION to ``dets`` (OC-SORT OCR).

        Same kinds and conventions as :meth:`_geometric_similarity`. A track
        never updated since its spawn has no observation (OC-SORT reference: the
        spawn box is not an observation) and scores 0.
        """
        tracks = list(tracks)
        n, m = len(tracks), len(dets)
        if n == 0 or m == 0:
            return np.zeros((n, m), dtype=np.float64)
        observed = np.array([t.last_observation is not None for t in tracks])
        boxes = np.full((n, 4), np.nan, dtype=np.float64)
        for i, t in enumerate(tracks):
            if observed[i]:
                boxes[i] = t.last_observation
        kind = self._similarity(dets)
        if kind == "oks" and dets.keypoints is not None:
            kp = self._stack_keypoints(tracks, dets.keypoints.shape[1])
            kp[~observed] = np.nan
            areas = _box_area(np.stack([t.last_raw_box for t in tracks]))
            sim = self._oks(dets.keypoints, kp, areas)
        elif kind == "centroid":
            centers = 0.5 * (boxes[:, :2] + boxes[:, 2:])
            sim = centroid_similarity(centers, dets.centroids, 0.5 * self._body_size())
        else:
            sim = iou_matrix(boxes, dets.boxes)
        return self._finish_similarity(sim, dets)

    def _appearance_cos(
        self, tracks: Sequence[MOTTrack], dets: FrameDetections, statistic: str = "ema"
    ) -> np.ndarray:
        """``(n_tracks, n_dets)`` appearance cosines; NaN where unavailable.

        ``statistic`` is ``"ema"`` (cosine to the track's EMA) or
        ``"gallery_max"`` (max over its gallery, EMA if empty). All NaN when
        appearance is disabled for association (see :meth:`_appearance_gate`).
        """
        tracks = list(tracks)
        out = np.full((len(tracks), len(dets)), np.nan, dtype=np.float64)
        if (
            out.size == 0
            or dets.embeddings is None
            or self._appearance_gate(statistic) is None
        ):
            return out
        if statistic == "ema":
            return self._ema_cos(tracks, dets)
        q_idx = np.flatnonzero(dets.has_embedding)
        # Normalized once for every track (the memory would redo it per track).
        rows, ok = unit_rows(np.asarray(dets.embeddings[q_idx], dtype=np.float64))
        for i, t in enumerate(tracks):
            mem = t.appearance
            if mem is None or mem.is_empty:
                continue
            ema = np.asarray(mem.ema).reshape(-1)
            if ema.shape[0] != rows.shape[1]:
                continue  # A vector of another model's dimension: no evidence.
            out[i, q_idx] = mem.gallery_max_cos_unit(rows, ok)
        return out

    def _appearance_gate(self, statistic: str = "ema") -> Optional[float]:
        """Minimum cosine for appearance to support a match, or ``None`` if disabled.

        ``appearance_threshold="auto"``: the calibrated gate for ``statistic``
        when ``calibration.appearance_for_association``, else ``None``. A float:
        that value (when the video has vectors).
        """
        if statistic not in APPEARANCE_STATISTICS:
            raise ValueError(
                f"statistic must be one of {APPEARANCE_STATISTICS}, got {statistic!r}."
            )
        if not self.uses_appearance or not self._vectors_seen:
            return None
        threshold = self.params.appearance_threshold
        if threshold != "auto":
            return float(threshold)
        cal = self.calibration
        if cal is None or not cal.appearance_for_association:
            return None
        gate = cal.ema_cos_gate if statistic == "ema" else cal.gallery_cos_gate
        if gate is None or not np.isfinite(gate):
            return None
        return float(gate)

    def _appearance_margin(self) -> float:
        """Minimum row/column cosine margin for appearance to decide a pair.

        Calibrated ``m_assoc`` when available, else 0.
        """
        cal = self.calibration
        if cal is None or cal.m_assoc is None or not np.isfinite(cal.m_assoc):
            return 0.0
        return float(cal.m_assoc)

    # --------------------------------------------------------- private: time
    def _advance(self, frame_idx: int) -> int:
        """Advance the clock to ``frame_idx`` and predict every live track.

        The first processed frame is step 0. Later frames advance by ``n`` steps
        (``1`` in the ``"steps"`` time base; :func:`steps_between` of the frame
        delta and the stride in the ``"frames"`` time base) and the gap's
        intermediate steps are treated as empty frames (see the module
        docstring). Uncalibrated, the stride and time base are first updated
        with this delta (:meth:`_infer_stride`). A repeated or backwards frame
        index counts one step and logs a WARNING once.

        Returns:
            The number of steps advanced (0 at the first processed frame).
        """
        if self._step < 0:
            self._step = 0
            self._last_frame_idx = frame_idx
            self._last_delta = None
            return 0
        delta = int(frame_idx - self._last_frame_idx)
        self._last_delta = delta
        if delta <= 0:
            self._warn_once(
                "order",
                f"{self.name}: frame {frame_idx} does not come after frame "
                f"{self._last_frame_idx} (repeated or backwards frame index). A MOT "
                "tracker tracks ONE video in increasing frame order: build one "
                "tracker per video and sort its frames (as apply_tracking does). "
                "Counting it as one step; live tracks continue across the jump.",
            )
        elif self.calibration is None:
            self._infer_stride(delta)
        n_steps = 1
        if self._time_base == "frames":
            n_steps = steps_between(delta, self._frame_step)
        first_step = self._step + 1
        self._step += n_steps
        self._last_frame_idx = frame_idx
        if not self.tracks:
            return n_steps

        # Tracks that the gap's empty steps would already have removed.
        for track in list(self.tracks):
            expired = track.time_since_update + n_steps - 1 > self.params.track_buffer
            died = (
                n_steps > 1
                and self.tentative_dies_on_miss
                and track.state == TrackState.TENTATIVE
            )
            if expired or died:
                self._remove(track)
        tracks = self.tracks
        if not tracks:
            return n_steps

        means = np.stack([t.mean for t in tracks])
        covs = np.stack([t.cov for t in tracks])
        tsu = np.array([t.time_since_update for t in tracks])
        zero_idx = list(self.lost_zero_velocity)
        for k in range(n_steps):
            if zero_idx:
                # Not matched at the previous step (the references' "not Tracked").
                missed = tsu + k > 0
                if missed.any():
                    means[np.ix_(missed, zero_idx)] = 0.0
            means, covs = self.kf.multi_predict(means, covs)
            if k == 0 and n_steps > 1 and self.use_oru:
                # The gap's first empty step is these tracks' first miss.
                for i, track in enumerate(tracks):
                    if (
                        tsu[i] == 0
                        and track.frozen is None
                        and track.last_observation is not None
                    ):
                        track.frozen = (means[i].copy(), covs[i].copy(), first_step)

        for i, track in enumerate(tracks):
            track.mean = means[i]
            track.cov = covs[i]
            track.time_since_update += n_steps
            track.age += n_steps
            if track.time_since_update > 1:
                # Missed at least one step: the streak restarts (SORT semantics).
                track.hit_streak = 0
            if n_steps > 1 and track.state == TrackState.CONFIRMED:
                track.state = TrackState.LOST
        return n_steps

    def _infer_stride(self, delta: int) -> None:
        """Uncalibrated: update the stride and time base with a positive delta.

        Same rule as the calibration over the deltas seen so far
        (:func:`stride_from_counts`, :func:`resolve_time_base`), applied BEFORE
        this frame's steps are counted, so a strided stream counts one step per
        processed frame from its first gap on, a dense stream with dropped frames
        keeps stride 1 (genuine gaps still count) and, under ``"auto"``,
        irregular input falls back to one step per processed frame (WARNING,
        once). Recorded in ``resolved``.
        """
        self._delta_counts[delta] = self._delta_counts.get(delta, 0) + 1
        frame_step, regular = stride_from_counts(self._delta_counts)
        time_base = resolve_time_base(self.params.time_base, regular)
        n_deltas = sum(self._delta_counts.values())
        if (
            self.params.time_base == "auto"
            and time_base == "steps"
            and n_deltas >= MIN_DELTAS_TO_WARN
        ):
            self._warn_once(
                "irregular",
                f"{self.name}: the frames tracked so far are irregularly spaced "
                f"(the most common frame step, {frame_step}, covers less than half "
                "of the gaps): counting one step per processed frame (time_base="
                "'steps') while that holds. prepare() decides this once over the "
                "whole video.",
            )
        self._time_base = time_base
        self._frame_step = frame_step if time_base == "frames" else 1
        self.resolved["time_base"] = self._time_base
        self.resolved["frame_step"] = int(self._frame_step)

    def _warn_once(self, key: str, message: str) -> None:
        """Log ``message`` as a WARNING the first time ``key`` is seen."""
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    # ---------------------------------------------------- private: lifecycle
    def _apply_match(
        self,
        track: MOTTrack,
        dets: FrameDetections,
        det_idx: int,
        sim: float,
        gate_ctx: Optional[dict],
    ) -> None:
        """Update ``track`` with detection ``det_idx`` (already validated)."""
        box = np.array(dets.boxes[det_idx], dtype=np.float64)
        score = float(dets.scores[det_idx])
        mean, cov = track.mean, track.cov
        if (
            self.use_oru
            and track.frozen is not None
            and track.last_observation is not None
        ):
            frozen_mean, frozen_cov, frozen_step = track.frozen
            mean, cov = oru_replay(
                self.kf,
                frozen_mean,
                frozen_cov,
                track.last_observation,
                box,
                n_gap=self._step - track.step,
                conf=score,
                frozen_steps=frozen_step - track.step,
            )
        track.frozen = None
        track.mean, track.cov = self.kf.update(mean, cov, box, conf=score)

        track.hits += 1
        track.hit_streak += 1
        track.time_since_update = 0
        track.step = self._step
        track.last_frame = dets.frame_idx
        track.last_box = box
        track.last_raw_box = np.array(dets.raw_boxes[det_idx], dtype=np.float64)
        if dets.keypoints is not None:
            track.last_keypoints = np.array(dets.keypoints[det_idx], dtype=np.float64)
        track.last_score = score
        track.kf_center_at_update = np.array(self.kf.center(track.mean), dtype=float)
        if track.state == TrackState.LOST:
            track.state = TrackState.CONFIRMED

        if self.keeps_observation_history:
            # OC-SORT observation history and velocity direction: from the
            # observation `delta_t` steps back (the oldest within that window),
            # else the latest observation, to this one.
            if track.last_observation is not None:
                previous = None
                for dt in range(self.params.delta_t, 0, -1):
                    previous = track.observations.get(self._step - dt)
                    if previous is not None:
                        break
                if previous is None:
                    previous = track.last_observation
                track.velocity = np.asarray(speed_direction(previous, box), dtype=float)
            track.observations[self._step] = box
            while len(track.observations) > 2 * self.params.delta_t + 1:
                del track.observations[min(track.observations)]
        track.last_observation = box

        if track.public_id is None:
            # Labeled when the track confirms; a confirmed track's detections
            # are labeled as they are matched (`track`), so none are kept.
            track.members.append((dets.frame_idx, dets.sources[det_idx], sim))
        self._update_appearance(track, dets, det_idx, gate_ctx)

    def _mark_missed(self, track: MOTTrack) -> None:
        """Handle a live track left unmatched at this step."""
        if (
            self.use_oru
            and track.frozen is None
            and track.last_observation is not None
            and track.time_since_update > 0
        ):
            # First miss since the latest observation: save the predicted state.
            track.frozen = (track.mean.copy(), track.cov.copy(), self._step)
        if track.state == TrackState.CONFIRMED:
            track.state = TrackState.LOST
        elif track.state == TrackState.TENTATIVE and self.tentative_dies_on_miss:
            self._remove(track)

    def _spawn(self, dets: FrameDetections, det_idx: int) -> MOTTrack:
        """Start a tentative track from detection ``det_idx``."""
        box = np.array(dets.boxes[det_idx], dtype=np.float64)
        mean, cov = self.kf.initiate(box)
        track = MOTTrack(
            tid=self._next_tid,
            state=TrackState.TENTATIVE,
            mean=mean,
            cov=cov,
            start_frame=dets.frame_idx,
            last_frame=dets.frame_idx,
            step=self._step,
            last_box=box,
            last_raw_box=np.array(dets.raw_boxes[det_idx], dtype=np.float64),
            kf_center_at_update=np.array(self.kf.center(mean), dtype=np.float64),
            last_keypoints=(
                None
                if dets.keypoints is None
                else np.array(dets.keypoints[det_idx], dtype=np.float64)
            ),
            last_score=float(dets.scores[det_idx]),
            hits=self.hits_start,
            hit_streak=self.hits_start,
            age=0,
            time_since_update=0,
        )
        self._next_tid += 1
        # The spawn detection is a member (labeled if the track confirms) but, as
        # in OC-SORT, not an "observation" for velocity / OCR / ORU.
        track.members.append((dets.frame_idx, dets.sources[det_idx], 1.0))
        if self.uses_appearance:
            track.appearance = self._make_appearance_memory()
            if dets.embeddings is not None and dets.has_embedding[det_idx]:
                track.appearance.seed(np.asarray(dets.embeddings[det_idx]))
        self.tracks.append(track)
        return track

    def _confirm(self, track: MOTTrack) -> None:
        """Confirm ``track``: mint its public id and label all its detections."""
        track.state = TrackState.CONFIRMED
        track.public_id = self._next_public_id
        self._next_public_id += 1
        track_obj = sio.Track(name=f"track_{track.public_id + self.track_name_offset}")
        self._track_objects[track.public_id] = track_obj
        for _, obj, sim in track.members:
            _set_label(obj, track_obj, sim)
        track.members.clear()

    def _remove(self, track: MOTTrack) -> None:
        """Mark ``track`` removed and drop it (counted in ``n_removed``)."""
        if track.state == TrackState.REMOVED:
            return
        track.state = TrackState.REMOVED
        self.tracks = [t for t in self.tracks if t is not track]
        self.n_removed += 1

    # --------------------------------------------------- private: appearance
    def _make_appearance_memory(self) -> AppearanceMemory:
        """A new track's appearance memory (subclasses may override)."""
        p = self.params
        return AppearanceMemory(
            ema_alpha=p.ema_alpha,
            gallery_size=p.gallery_size,
            adaptive=p.adaptive_ema,
            score_floor=p.track_high_threshold,
        )

    def _ema_cos(self, tracks: Sequence[MOTTrack], dets: FrameDetections) -> np.ndarray:
        """Cosine of each track's EMA to each detection vector (NaN if missing)."""
        n, m = len(tracks), len(dets)
        if n == 0 or m == 0 or dets.embeddings is None:
            return np.full((n, m), np.nan, dtype=np.float64)
        dim = dets.embeddings.shape[1]
        emas = np.zeros((n, dim), dtype=np.float64)
        ok = np.zeros(n, dtype=bool)
        for i, t in enumerate(tracks):
            mem = t.appearance
            if mem is None or mem.ema is None:
                continue
            ema = np.asarray(mem.ema, dtype=np.float64).reshape(-1)
            if ema.shape[0] == dim:
                emas[i] = ema
                ok[i] = True
        return cosine_matrix(emas, ok, dets.embeddings, dets.has_embedding)

    def _gate_context(
        self,
        pool: List[MOTTrack],
        dets: FrameDetections,
        result: AssociationResult,
    ) -> Optional[dict]:
        """Everything the appearance quality gate needs, computed BEFORE updates.

        ``None`` when no matched detection carries a vector (or the tracker does
        not use appearance).
        """
        if (
            not self.uses_appearance
            or dets.embeddings is None
            or not result.matches
            or not pool
        ):
            return None
        boxes = dets.boxes
        det_iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(det_iou, 0.0)
        pred_boxes = self.kf.xyxy_from_mean(np.stack([t.mean for t in pool]))
        neighbor = np.array(
            [
                t.state == TrackState.CONFIRMED
                or (
                    t.state == TrackState.LOST
                    and t.time_since_update <= GATE_LOST_MAX_STEPS
                )
                for t in pool
            ]
        )
        return {
            "row": {t.tid: i for i, t in enumerate(pool)},
            "neighbor": neighbor,
            "det_iou": det_iou,
            "trk_iou": iou_matrix(pred_boxes, boxes),
            "geo": self._geometric_similarity(pool, dets),
            "cos": self._ema_cos(pool, dets),
        }

    def _update_appearance(
        self,
        track: MOTTrack,
        dets: FrameDetections,
        det_idx: int,
        gate_ctx: Optional[dict],
    ) -> None:
        """Quality-gated appearance update of a matched track."""
        if (
            not self.uses_appearance
            or dets.embeddings is None
            or not dets.has_embedding[det_idx]
        ):
            return
        vec = np.asarray(dets.embeddings[det_idx])
        if track.appearance is None:
            track.appearance = self._make_appearance_memory()
        if track.appearance.is_empty:
            track.appearance.seed(vec)
            return
        gated = False
        row = None if gate_ctx is None else gate_ctx["row"].get(track.tid)
        if row is not None:
            others = np.arange(len(gate_ctx["neighbor"])) != row
            iou_det = gate_ctx["det_iou"][det_idx]
            iou_trk = gate_ctx["trk_iou"][others & gate_ctx["neighbor"], det_idx]
            max_iou = float(max(iou_det.max(initial=0.0), iou_trk.max(initial=0.0)))
            cos = gate_ctx["cos"][:, det_idx]
            competitors = cos[others & (gate_ctx["geo"][:, det_idx] > 0)]
            competitors = competitors[np.isfinite(competitors)]
            own = cos[row]
            if not np.isfinite(own) or len(competitors) == 0:
                margin = float("inf")
            else:
                margin = float(own - competitors.max())
            gated = passes_quality_gate(
                float(dets.scores[det_idx]),
                int(dets.n_visible[det_idx]),
                int(dets.n_nodes),
                max_iou,
                margin,
                high_threshold=self.params.track_high_threshold,
                min_margin=self._appearance_margin(),
                gallery_len=len(track.appearance.gallery),
            )
        track.appearance.update(vec, score=float(dets.scores[det_idx]), gated=gated)

    # ----------------------------------------------------- private: geometry
    def _resolve_similarity(self, n_nodes: Optional[int]) -> str:
        """Resolve ``params.similarity`` for the carrier and skeleton size."""
        kind = self.params.similarity
        if self.carrier == "mask":
            if kind == "oks":
                logger.warning(
                    "similarity='oks' needs keypoints; using 'iou' on masks."
                )
                return "iou"
            return "iou" if kind == "auto" else kind
        if kind == "auto":
            return "centroid" if n_nodes == 1 else "iou"
        if kind == "oks" and n_nodes == 1:
            logger.warning(
                "similarity='oks' on a single-node skeleton is a centroid distance; "
                "using 'centroid'."
            )
            return "centroid"
        return kind

    def _similarity(self, dets: Optional[FrameDetections] = None) -> str:
        """The resolved similarity kind (resolving it from ``dets`` if needed)."""
        if self._similarity_kind is None and dets is not None and len(dets) > 0:
            self._similarity_kind = self._resolve_similarity(dets.n_nodes)
            self.resolved["similarity"] = self._similarity_kind
        if self._similarity_kind is None:
            # Unknown skeleton yet: the multi-node default.
            return "iou" if self.params.similarity == "auto" else self.params.similarity
        return self._similarity_kind

    def _finish_similarity(self, sim: np.ndarray, dets: FrameDetections) -> np.ndarray:
        """NaN-free, ``[0, 1]``-clipped similarity with invalid detections at 0."""
        sim = np.clip(np.nan_to_num(sim, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
        sim[:, ~np.asarray(dets.valid, dtype=bool)] = 0.0
        return sim

    def _stack_keypoints(self, tracks: Sequence[MOTTrack], n_nodes: int) -> np.ndarray:
        """``(n, K, 2)`` last observed keypoints (NaN where missing), padded to K."""
        k = max(
            [n_nodes]
            + [
                t.last_keypoints.shape[0]
                for t in tracks
                if t.last_keypoints is not None
            ]
        )
        out = np.full((len(tracks), k, 2), np.nan, dtype=np.float64)
        for i, t in enumerate(tracks):
            if t.last_keypoints is not None:
                kp = np.asarray(t.last_keypoints, dtype=np.float64).reshape(-1, 2)
                out[i, : kp.shape[0]] = kp
        return out

    def _predicted_poses(
        self, tracks: Sequence[MOTTrack], means: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted poses and their raw-extent areas for OKS.

        The pose is the last observed pose shifted by the Kalman center's motion
        since the last update (both centers are KF posteriors/predictions of the
        same reference point, the box center). The area is the last raw extent's
        scaled by the predicted / last-observed padded-box area ratio.
        """
        kp = self._stack_keypoints(tracks, 1)
        shift = np.asarray(means[:, :2], dtype=np.float64) - np.stack(
            [t.kf_center_at_update for t in tracks]
        )
        kp = kp + shift[:, None, :]
        last_area = _box_area(np.stack([t.last_box for t in tracks]))
        pred_area = _box_area(self.kf.xyxy_from_mean(means))
        ratio = np.ones(len(tracks), dtype=np.float64)
        np.divide(pred_area, last_area, out=ratio, where=last_area > 0)
        areas = _box_area(np.stack([t.last_raw_box for t in tracks])) * ratio
        return kp, areas

    def _oks(
        self, det_kp: np.ndarray, trk_kp: np.ndarray, areas: np.ndarray
    ) -> np.ndarray:
        """``(n_tracks, n_dets)`` OKS with the tracks' poses as the scale side."""
        k = max(det_kp.shape[1], trk_kp.shape[1])
        sim = oks_matrix(
            _pad_nodes(np.asarray(det_kp, dtype=np.float64), k),
            _pad_nodes(trk_kp, k),
            areas,
            self.params.oks_stddev,
        )
        return sim.T

    def _body_size(self) -> float:
        """Body size (px); see :meth:`_body_size_info`."""
        return self._body_size_info()[0]

    def _body_size_info(self) -> Tuple[float, str]:
        """Body size (px) and its source.

        ``params.body_size`` (``"override"``), else the calibration's, else
        (uncalibrated) calibration's fallback chain
        (:func:`~sleap_nn.tracking.mot.calibration.resolve_body_size`) on the
        frames seen so far: the median raw-box diagonal, else 8x the per-step
        p99 of unambiguous links (single-node detections), else the median
        same-frame spacing, else the chain's default (50 px).
        """
        if self.params.body_size is not None:
            return float(self.params.body_size), "override"
        cal = self.calibration
        if cal is not None and np.isfinite(cal.body_size) and cal.body_size > 0:
            return float(cal.body_size), str(cal.body_size_source)
        measured = (
            float(np.median(np.asarray(self._diagonals)))
            if self._diagonals
            else float("nan")
        )
        step_p99 = (
            float(np.percentile(np.asarray(self._step_lengths), 99))
            if self._step_lengths
            else float("nan")
        )
        spacing = (
            float(np.median(np.asarray(self._spacings)))
            if self._spacings
            else float("nan")
        )
        body, source = resolve_body_size(measured, step_p99, spacing)
        return float(body), source

    # -------------------------------------------------- private: detections
    def _extract(
        self, objs: Sequence[Any], frame_idx: int, pseudo_box_size: Optional[float]
    ) -> FrameDetections:
        """Extract one frame's detections with the tracker's settings."""
        p = self.params
        # Thin poses (nodes nearly collinear) would get sliver boxes whose IoU
        # collapses under tiny shifts; floor the padded box at a fraction of the
        # body size (only known once a body size is: calibrated or observed).
        min_box = (
            MIN_BOX_FRACTION * pseudo_box_size if pseudo_box_size is not None else None
        )
        dets = extract_detections(
            objs,
            frame_idx,
            carrier=self.carrier,
            box_padding=p.box_padding,
            pseudo_box_size=pseudo_box_size,
            score_mode=p.score_mode,
            embedding_dim=self._embedding_dim,
            min_box_size=min_box,
        )
        if dets.embeddings is not None:
            self._vectors_seen = True
            if self._embedding_dim is None:
                self._embedding_dim = int(dets.embeddings.shape[1])
        # Per-video score normalization (set by `prepare`; exactly once here).
        normalize_scores(dets, self._score_normalizer)
        return dets

    def _detections(self, objs: List[Any], frame_idx: int) -> FrameDetections:
        """This frame's detections: from the ``prepare`` cache, else extracted."""
        cached = self._cache.pop(frame_idx, None)
        if (
            cached is not None
            and len(cached.sources) == len(objs)
            and all(a is b for a, b in zip(cached.sources, objs))
        ):
            return cached
        return self._extract(objs, frame_idx, pseudo_box_size=self._body_size())

    def _observe_uncalibrated(self, dets: FrameDetections, n_steps: int) -> None:
        """Uncalibrated: add this frame's body-size evidence (see `_body_size_info`).

        Extents feed the running diagonal median. While no extent has been seen
        (single-node detections), the frame's same-frame spacings and the
        per-step lengths of its unambiguous links to the previous processed
        frame (:func:`~sleap_nn.tracking.mot.calibration.pair_geometry`; not
        across a repeated or backwards frame) are remembered instead.
        """
        self._remember_diagonals(dets)
        if self._diagonals or self.params.body_size is not None:
            self._prev_dets = None
            return
        prev = self._prev_dets
        if self._last_delta is not None and self._last_delta <= 0:
            prev = None
        max_dist = (
            0.5 * float(np.median(np.asarray(self._spacings)))
            if self._spacings
            else None
        )
        spacings, links = pair_geometry(prev, dets, max_dist=max_dist)
        self._spacings.extend(spacings.tolist())
        if n_steps > 0 and len(links):
            self._step_lengths.extend((links / n_steps).tolist())
        self._prev_dets = dets

    def _remember_diagonals(self, dets: FrameDetections) -> None:
        """Add the frame's non-degenerate extents to the running body size."""
        if len(dets) == 0:
            return
        if dets.keypoints is not None:
            # Keypoint extents (a pseudo box would feed the estimate back).
            kp = dets.keypoints[dets.valid & (dets.n_visible >= 2)]
            vis = np.isfinite(kp).all(axis=-1)
            lo = np.where(vis[..., None], kp, np.inf).min(axis=1, initial=np.inf)
            hi = np.where(vis[..., None], kp, -np.inf).max(axis=1, initial=-np.inf)
            wh = hi - lo
        else:
            raw = dets.raw_boxes[dets.valid]
            wh = raw[:, 2:] - raw[:, :2]
        keep = np.isfinite(wh).all(axis=1) & (wh >= 1.0).all(axis=1)
        self._diagonals.extend(np.hypot(wh[keep, 0], wh[keep, 1]).tolist())

    def _has_degenerate(self, dets: FrameDetections) -> bool:
        """Whether any valid detection got a fallback (degenerate) box."""
        if len(dets) == 0 or not dets.valid.any():
            return False
        raw = dets.raw_boxes[dets.valid]
        thin = ((raw[:, 2:] - raw[:, :2]) <= 1.0).any(axis=1)
        single = dets.keypoints is not None and (dets.n_visible[dets.valid] == 1).any()
        return bool(thin.any() or single)

    # ---------------------------------------------------- private: provenance
    def _appearance_summary(self) -> Dict[str, Any]:
        """JSON-safe record of the appearance decisions."""
        if not self.uses_appearance:
            return {"enabled": False, "reason": "the tracker does not use appearance"}
        threshold = self.params.appearance_threshold
        ema_gate = self._appearance_gate("ema")
        gallery_gate = self._appearance_gate("gallery_max")
        cal = self.calibration
        if not self._vectors_seen:
            reason = "no appearance vectors on the tracked detections"
        elif threshold != "auto":
            reason = "explicit appearance_threshold"
        elif cal is None:
            reason = "uncalibrated (no prepare): 'auto' disables appearance"
        elif not cal.appearance_for_association:
            reason = (
                f"calibration status={cal.status!r}, assoc_rank1={cal.assoc_rank1}: "
                "appearance not discriminative enough for association"
            )
        else:
            reason = "calibrated gates"
        return {
            "enabled": ema_gate is not None or gallery_gate is not None,
            "threshold": threshold,
            "ema_cos_gate": ema_gate,
            "gallery_cos_gate": gallery_gate,
            "margin": self._appearance_margin(),
            "reason": reason,
        }

    def _refresh_resolved(self) -> None:
        """Rebuild ``resolved`` (keeps the linking result once recorded)."""
        linking = self.resolved.get("linking") if self.resolved else None
        body, body_source = self._body_size_info()
        self.resolved = {
            "tracker": self.name,
            "params": self.params.as_dict(),
            "kalman": self.kalman_kind,
            "time_base": self._time_base,
            "frame_step": int(self._frame_step),
            "similarity": self._similarity(),
            "body_size": float(body),
            "body_size_source": body_source,
            "score_normalizer": (
                None
                if self._score_normalizer is None
                else float(self._score_normalizer)
            ),
            "calibration": (
                None if self.calibration is None else self.calibration.as_dict()
            ),
            "appearance": self._appearance_summary(),
        }
        if linking is not None:
            self.resolved["linking"] = linking
