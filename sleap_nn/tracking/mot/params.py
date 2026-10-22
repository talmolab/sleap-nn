"""Resolved parameters and presets for the modern MOT trackers.

Every tracker in :mod:`sleap_nn.tracking.mot` is driven by one frozen
:class:`MOTParams`. :meth:`MOTParams.for_tracker` starts from a per-tracker
**preset** and applies user overrides on top:

- ``"paper"``: the constants published with each algorithm (and its reference
  implementation), adapted only where pose/mask geometry forces it (see the
  module docs of each tracker). Use it to compare against the literature.
- ``"animal"`` (default): sleap-nn's defaults for animal pose/mask tracking. They
  start from ``"paper"`` and change what does not transfer to SLEAP data: SLEAP
  detection scores (mean point confidences) sit well below a pedestrian
  detector's (so scores are normalized per video, ``normalize_scores``), the
  reference re-ID thresholds assume a person re-ID embedder, per-track
  appearance is better smoothed with a shorter EMA memory, and the pedestrian
  ``xyah`` Kalman filter (a slowly varying aspect ratio) cannot follow a turning
  animal, so DeepSORT, StrongSORT and ByteTrack use ``xywh``.

All user-facing thresholds are SIMILARITIES in ``[0, 1]`` (higher is stricter),
whatever the geometric similarity in use (``iou``, ``oks`` or ``centroid``); each
tracker converts them to the costs its algorithm minimizes.
"""

import math
from typing import Any, Dict, Optional, Union

import attrs

# Trackers implemented in this package, in the order they are documented. The
# `classic` tracker (`sleap_nn.tracking.tracker.Tracker`) is not part of this
# family and is not listed here.
MOT_TRACKER_NAMES = (
    "sort",
    "deepsort",
    "strongsort",
    "bytetrack",
    "botsort",
    "ocsort",
    "deepocsort",
)

PRESETS = ("animal", "paper")

SIMILARITY_KINDS = ("auto", "iou", "oks", "centroid")
SCORE_MODES = ("auto", "instance", "none")
KALMAN_KINDS = ("auto", "xywh", "xyah", "xysr")
TIME_BASES = ("auto", "frames", "steps")


def _check_unit_interval(instance, attribute, value):
    if value is not None and not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{attribute.name} must be in [0, 1], got {value!r}.")


def _check_positive_or_none(instance, attribute, value):
    if value is not None and not (math.isfinite(value) and value > 0):
        raise ValueError(f"{attribute.name} must be > 0 (or None), got {value!r}.")


def _optional_float(value: Any) -> Optional[float]:
    """``None`` stays ``None``; anything else becomes a float."""
    return None if value is None else float(value)


def _check_choice(choices):
    def _validator(instance, attribute, value):
        if value not in choices:
            raise ValueError(
                f"{attribute.name} must be one of {list(choices)}, got {value!r}."
            )

    return _validator


def _convert_appearance_threshold(value: Union[float, str, None]) -> Union[float, str]:
    """Accept ``"auto"`` or a minimum cosine similarity in ``[-1, 1]``."""
    if value is None:
        return "auto"
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return "auto"
        value = float(value)
    value = float(value)
    if not -1.0 <= value <= 1.0:
        raise ValueError(
            f"appearance_threshold must be 'auto' or a cosine in [-1, 1], got {value!r}."
        )
    return value


@attrs.frozen
class MOTParams:
    """Resolved parameters for one MOT tracker run.

    Attributes:
        tracker: Tracker name, one of :data:`MOT_TRACKER_NAMES`.
        preset: Preset the defaults came from (``"animal"`` or ``"paper"``).
        similarity: Geometric similarity between a track's prediction and a
            detection: ``"iou"`` (padded boxes), ``"oks"`` (keypoints vs the
            track's predicted pose), ``"centroid"`` (Gaussian on the box-center
            distance, scale ``body_size / 2``) or ``"auto"`` (masks -> iou,
            single-node skeletons -> centroid, otherwise iou).
        body_size: Animal size in px (box diagonal), or ``None`` to measure it
            per video. It scales the ``centroid`` similarity (``body_size / 2``),
            sizes the pseudo boxes of single-node detections and floors thin
            boxes. ``None``: the video's calibration (``prepare()``): the median
            box diagonal of multi-node poses / masks; single-node poses have no
            extent, so ``8 x`` the per-step motion p99 (at least 16 px, at most
            the nearest-neighbor spacing), else the spacing, else 50 px; without
            ``prepare()``, the same chain on the frames seen so far. Set it for
            centroid-only models whose calibration cannot size the animal.
        oks_stddev: OKS per-node standard deviation for ``similarity="oks"``
            (a finite number > 0; NaN and inf are refused). The
            MOT default (0.1) is 4x the evaluation default: tracking compares two
            noisy observations a frame apart, and at 0.025 an OKS threshold is
            ~7x stricter than the same IoU threshold.
        box_padding: Padding added to each side of a keypoint/mask box, as a
            fraction of ``max(w, h)``.
        score_mode: Detection confidence: ``"auto"`` (mean visible point score
            for predicted poses, ``score`` for masks, 1.0 for user instances),
            ``"instance"`` (the object's ``score``) or ``"none"`` (all 1.0).
        normalize_scores: Divide every detection score by the video's
            95th-percentile score (``max(q95, 1e-6)`` of the calibration's raw
            ``score_quantiles``) and clip to ``[0, 1]`` before any score
            threshold applies. SLEAP mean point scores depend on the model (q95
            ~0.56 for one mouse model, ~0.97 for a fly model), so fixed
            thresholds tuned for pedestrian detectors (0.5-0.7) can silently
            block every track start; normalized, a typical detection of the
            video scores ~0.85-1.0 and a 1.0-scored (user) detection is
            unchanged. Needs ``prepare()`` (the calibration): without it scores
            are used as they are. ``animal`` preset: on; ``paper``: off (the
            published thresholds on raw scores).
        kalman: Kalman filter state parameterization; ``"auto"`` uses the
            tracker's own (``xysr`` SORT/OC-SORT, ``xyah`` DeepSORT/StrongSORT/
            ByteTrack, ``xywh`` BoT-SORT/Deep OC-SORT). The ``animal`` preset
            sets ``xywh`` for DeepSORT/StrongSORT/ByteTrack: ``xyah`` assumes a
            slowly varying aspect ratio, which a turning animal's box breaks.
        time_base: How time advances between processed frames: ``"frames"``
            (frame-index deltas, normalized by the video's stride: the most
            common delta), ``"steps"`` (one step per processed frame) or
            ``"auto"`` (frames when the processed frames are regularly spaced,
            else steps). ``prepare()`` resolves it over the whole video; without
            it the stride and regularity are inferred online from the deltas
            seen so far (same rule), so a strided stream counts one step per
            processed frame from its first gap on.
        track_buffer: Steps a lost track is kept alive for re-association
            (``max_age``; 1 in SORT's ``paper`` preset, as published).
        min_hits: Matches required before a track is confirmed (``n_init``).
            Detections of a track that is later confirmed are labeled
            retroactively.
        track_high_threshold: Detections with a score above this are
            high-confidence (ByteTrack's ``track_thresh`` / OC-SORT's
            ``det_thresh``); also the minimum score for appearance-memory updates.
        track_low_threshold: Detections with a score at or below this are ignored.
        new_track_threshold: Minimum score for an unmatched detection to start a
            track.
        min_match_similarity: Minimum geometric similarity for a first-stage
            match (cost cap ``1 - min_match_similarity``).
        low_stage_min_similarity: ByteTrack/BoT-SORT second (low-score) stage.
        unconfirmed_min_similarity: ByteTrack/BoT-SORT unconfirmed-track stage.
        duplicate_similarity: ByteTrack/BoT-SORT tracked-vs-lost duplicate removal.
        fuse_score: Multiply the geometric similarity by the detection score
            (ByteTrack ``fuse_score``).
        appearance_threshold: Minimum cosine similarity for appearance to support
            a match, or ``"auto"`` to calibrate per video (see
            :mod:`sleap_nn.tracking.mot.calibration`).
        proximity_threshold: BoT-SORT: minimum geometric similarity for the
            appearance term to apply at all.
        ema_alpha: Appearance EMA momentum (weight on the old feature).
        adaptive_ema: Deep OC-SORT score-adaptive EMA.
        gallery_size: Per-track gallery of recent confident vectors.
        gallery_budget: DeepSORT gallery budget (``nn_budget``).
        mc_lambda: StrongSORT motion-cost weight on appearance (``lambda``).
        nsa: StrongSORT NSA Kalman (measurement noise scaled by detection score).
        delta_t: OC-SORT observation lookback (steps) for velocity direction.
        inertia: OC-SORT weight of the observation-centric momentum term.
        use_byte: OC-SORT optional BYTE second stage.
        w_assoc_emb: Deep OC-SORT base appearance weight.
        aw_param: Deep OC-SORT adaptive-weighting cap.
        aw_off: Disable Deep OC-SORT adaptive weighting (constant ``w_assoc_emb``).
        rescue: Deep OC-SORT: accept a weak-geometry match when appearance is
            decisive (BoostTrack rule).
        rescue_min_similarity: Minimum geometric similarity for a rescue.
        link_tracklets: Run identity linking (long-term memory) at the end of
            each video.
        max_tracks: Known number of animals; closed-set identity linking.
        link_swap_correction: Correct identity swaps at crossings during linking.
        link_spatial_gate_frames: Gap (steps) above which linking applies no
            spatial gate (long absences can end anywhere).
    """

    tracker: str = attrs.field(validator=_check_choice(MOT_TRACKER_NAMES))
    preset: str = attrs.field(default="animal", validator=_check_choice(PRESETS))
    similarity: str = attrs.field(
        default="auto", validator=_check_choice(SIMILARITY_KINDS)
    )
    body_size: Optional[float] = attrs.field(
        default=None, converter=_optional_float, validator=_check_positive_or_none
    )
    oks_stddev: float = 0.1
    box_padding: float = 0.1
    score_mode: str = attrs.field(default="auto", validator=_check_choice(SCORE_MODES))
    normalize_scores: bool = False
    kalman: str = attrs.field(default="auto", validator=_check_choice(KALMAN_KINDS))
    time_base: str = attrs.field(default="auto", validator=_check_choice(TIME_BASES))
    track_buffer: int = 30
    min_hits: int = 3
    track_high_threshold: float = attrs.field(
        default=0.5, validator=_check_unit_interval
    )
    track_low_threshold: float = attrs.field(
        default=0.1, validator=_check_unit_interval
    )
    new_track_threshold: float = attrs.field(
        default=0.6, validator=_check_unit_interval
    )
    min_match_similarity: float = attrs.field(
        default=0.3, validator=_check_unit_interval
    )
    low_stage_min_similarity: float = attrs.field(
        default=0.5, validator=_check_unit_interval
    )
    unconfirmed_min_similarity: float = attrs.field(
        default=0.3, validator=_check_unit_interval
    )
    duplicate_similarity: float = attrs.field(
        default=0.85, validator=_check_unit_interval
    )
    fuse_score: bool = False
    appearance_threshold: Union[float, str] = attrs.field(
        default="auto", converter=_convert_appearance_threshold
    )
    proximity_threshold: float = attrs.field(
        default=0.5, validator=_check_unit_interval
    )
    ema_alpha: float = attrs.field(default=0.8, validator=_check_unit_interval)
    adaptive_ema: bool = False
    gallery_size: int = 30
    gallery_budget: int = 100
    mc_lambda: float = attrs.field(default=0.98, validator=_check_unit_interval)
    nsa: bool = False
    delta_t: int = 3
    inertia: float = 0.2
    use_byte: bool = False
    w_assoc_emb: float = 0.75
    aw_param: float = 0.5
    aw_off: bool = False
    rescue: bool = True
    rescue_min_similarity: float = attrs.field(
        default=0.15, validator=_check_unit_interval
    )
    link_tracklets: bool = False
    max_tracks: Optional[int] = None
    link_swap_correction: bool = True
    link_spatial_gate_frames: int = 300

    def __attrs_post_init__(self):
        """Validate cross-field constraints."""
        if self.track_buffer < 0:
            raise ValueError(f"track_buffer must be >= 0, got {self.track_buffer}.")
        if self.min_hits < 1:
            raise ValueError(f"min_hits must be >= 1, got {self.min_hits}.")
        try:
            oks_ok = math.isfinite(self.oks_stddev) and self.oks_stddev > 0
        except TypeError:
            oks_ok = False
        if not oks_ok:
            raise ValueError(
                f"oks_stddev must be a finite number > 0, got {self.oks_stddev!r}."
            )
        if self.gallery_size < 1 or self.gallery_budget < 1:
            raise ValueError("gallery_size and gallery_budget must be >= 1.")
        if self.delta_t < 1:
            raise ValueError(f"delta_t must be >= 1, got {self.delta_t}.")
        if self.max_tracks is not None and self.max_tracks < 1:
            raise ValueError(f"max_tracks must be >= 1, got {self.max_tracks}.")
        if self.track_low_threshold > self.track_high_threshold:
            raise ValueError(
                "track_low_threshold must be <= track_high_threshold, got "
                f"{self.track_low_threshold} > {self.track_high_threshold}."
            )

    @property
    def link(self) -> bool:
        """Whether identity linking runs at the end of each video.

        Linking runs when requested explicitly, and always when the animal count
        is known: ``max_tracks`` is enforced by closed-set linking, never by
        refusing to start tracks online.
        """
        return self.link_tracklets or self.max_tracks is not None

    @classmethod
    def field_names(cls) -> tuple:
        """Names of every overridable field."""
        return tuple(a.name for a in attrs.fields(cls))

    @classmethod
    def for_tracker(
        cls, tracker: str, preset: str = "animal", **overrides: Any
    ) -> "MOTParams":
        """Resolve a tracker's preset and apply user overrides.

        Args:
            tracker: One of :data:`MOT_TRACKER_NAMES`.
            preset: ``"animal"`` (default) or ``"paper"``.
            **overrides: Field values that replace the preset's. ``None`` values
                are ignored (they mean "use the preset"), so callers can pass
                optional config fields straight through.

        Returns:
            The resolved, validated :class:`MOTParams`.

        Raises:
            ValueError: Unknown tracker, preset or override key, or an invalid
                value.
        """
        if tracker not in MOT_TRACKER_NAMES:
            raise ValueError(
                f"Unknown MOT tracker {tracker!r}; expected one of "
                f"{list(MOT_TRACKER_NAMES)}."
            )
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}; expected one of {list(PRESETS)}."
            )
        valid = set(cls.field_names()) - {"tracker", "preset"}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ValueError(
                f"Unknown MOT parameter(s) {unknown}; valid parameters are "
                f"{sorted(valid)}."
            )
        values: Dict[str, Any] = dict(_PAPER_COMMON)
        values.update(_PAPER[tracker])
        if preset == "animal":
            values.update(_ANIMAL_COMMON)
            values.update(_ANIMAL.get(tracker, {}))
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(tracker=tracker, preset=preset, **values)

    def as_dict(self) -> Dict[str, Any]:
        """Return the parameters as a JSON-serializable dict."""
        return attrs.asdict(self)


# Published constants shared by every tracker: thresholds apply to raw scores.
_PAPER_COMMON: Dict[str, Any] = dict(normalize_scores=False)

# Published constants per tracker (reference implementations' defaults). Fields
# not listed take `_PAPER_COMMON`, then the class defaults above.
_PAPER: Dict[str, Dict[str, Any]] = {
    "sort": dict(
        kalman="xysr",
        min_hits=3,
        # Reference `Sort(max_age=1)` (T_lost = 1 in Bewley et al. 2016): a track
        # is removed after more than one missed step.
        track_buffer=1,
        # SORT has no score logic: the caller filters detections. Keep everything.
        track_high_threshold=0.0,
        track_low_threshold=0.0,
        new_track_threshold=0.0,
        min_match_similarity=0.3,
        fuse_score=False,
    ),
    "deepsort": dict(
        kalman="xyah",
        min_hits=3,
        # deep_sort_app --min_confidence 0.8: lower detections are dropped and every
        # kept detection may start a track.
        track_high_threshold=0.8,
        track_low_threshold=0.8,
        new_track_threshold=0.8,
        min_match_similarity=0.3,  # max_iou_distance 0.7 (geometric stage)
        appearance_threshold=0.8,  # max_cosine_distance 0.2
        gallery_budget=100,
        fuse_score=False,
    ),
    "strongsort": dict(
        kalman="xyah",
        nsa=True,
        min_hits=3,
        track_high_threshold=0.6,
        track_low_threshold=0.6,
        new_track_threshold=0.6,
        min_match_similarity=0.3,
        appearance_threshold=0.7,  # max_cosine_distance 0.3 (+0.05 under MC)
        ema_alpha=0.9,
        mc_lambda=0.98,
        fuse_score=False,
    ),
    "bytetrack": dict(
        kalman="xyah",
        min_hits=2,
        track_high_threshold=0.5,
        track_low_threshold=0.1,
        new_track_threshold=0.6,
        min_match_similarity=0.2,  # match_thresh 0.8
        low_stage_min_similarity=0.5,
        unconfirmed_min_similarity=0.3,  # 0.7 cost cap
        duplicate_similarity=0.85,
        fuse_score=True,
    ),
    "botsort": dict(
        kalman="xywh",
        min_hits=2,
        track_high_threshold=0.6,
        track_low_threshold=0.1,
        new_track_threshold=0.7,
        min_match_similarity=0.2,
        low_stage_min_similarity=0.5,
        unconfirmed_min_similarity=0.3,
        duplicate_similarity=0.85,
        fuse_score=True,
        proximity_threshold=0.5,
        appearance_threshold=0.5,  # halved cosine distance 0.25
        ema_alpha=0.9,
    ),
    "ocsort": dict(
        kalman="xysr",
        min_hits=3,
        track_high_threshold=0.6,
        track_low_threshold=0.1,
        new_track_threshold=0.6,
        min_match_similarity=0.3,
        delta_t=3,
        inertia=0.2,
        use_byte=False,
        fuse_score=False,
    ),
    "deepocsort": dict(
        kalman="xywh",
        min_hits=3,
        track_high_threshold=0.6,
        track_low_threshold=0.1,
        new_track_threshold=0.6,
        min_match_similarity=0.3,
        delta_t=3,
        inertia=0.2,
        adaptive_ema=True,
        ema_alpha=0.95,
        w_assoc_emb=0.75,
        aw_param=0.5,
        rescue=False,
        fuse_score=False,
    ),
}

# sleap-nn animal defaults shared by every tracker: per-video score
# normalization (SLEAP mean point scores sit far below pedestrian-detector
# scores, and their scale depends on the model).
_ANIMAL_COMMON: Dict[str, Any] = dict(normalize_scores=True)

# sleap-nn animal defaults: what differs from `_PAPER` (after `_ANIMAL_COMMON`).
# Tuned on the DEV clips of the benchmark (see the tracking guide); everything
# else is the paper value. `xyah` (DeepSORT/StrongSORT/ByteTrack) carries the
# aspect ratio with pedestrian process noise: a turning animal's predicted box
# inflates and loses IoU, so the animal preset tracks `xywh` instead.
_ANIMAL: Dict[str, Dict[str, Any]] = {
    "sort": dict(
        track_buffer=30,
        track_high_threshold=0.1,
        track_low_threshold=0.1,
        new_track_threshold=0.1,
    ),
    "deepsort": dict(
        kalman="xywh",
        track_high_threshold=0.5,
        track_low_threshold=0.1,
        new_track_threshold=0.1,
        appearance_threshold="auto",
        gallery_budget=30,
    ),
    "strongsort": dict(
        kalman="xywh",
        track_high_threshold=0.5,
        track_low_threshold=0.1,
        new_track_threshold=0.1,
        appearance_threshold="auto",
        ema_alpha=0.8,
    ),
    "bytetrack": dict(kalman="xywh"),
    "botsort": dict(appearance_threshold="auto", ema_alpha=0.8),
    "ocsort": dict(),
    "deepocsort": dict(appearance_threshold="auto", rescue=True),
}


# Optional (None-default) fields, by value type, for `parse_mot_param`.
_OPTIONAL_INTS = ("max_tracks",)
_OPTIONAL_FLOATS = ("body_size",)


def parse_mot_param(text: str) -> tuple:
    """Parse one ``KEY=VALUE`` override for :class:`MOTParams`.

    The value is coerced to the field's type: ``true``/``false`` for booleans,
    ``none`` for optional fields (``max_tracks``, ``body_size``), ``auto`` or a
    number for ``appearance_threshold``, int/float/str otherwise. ``preset`` is
    accepted too.

    Args:
        text: ``"key=value"``.

    Returns:
        ``(key, value)`` with the value coerced.

    Raises:
        ValueError: Malformed text, unknown key, or a value of the wrong type.
    """
    if "=" not in text:
        raise ValueError(f"Expected KEY=VALUE, got {text!r}.")
    key, raw = (part.strip() for part in text.split("=", 1))
    if key == "preset":
        if raw not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS)}, got {raw!r}.")
        return key, raw
    fields = {a.name: a for a in attrs.fields(MOTParams)}
    if key not in fields or key == "tracker":
        valid = sorted(set(fields) - {"tracker"} | {"preset"})
        raise ValueError(f"Unknown MOT parameter {key!r}; valid parameters: {valid}.")
    default = fields[key].default
    lowered = raw.lower()
    if key == "appearance_threshold":
        return key, _convert_appearance_threshold(raw)
    if lowered == "none":
        if key not in _OPTIONAL_INTS + _OPTIONAL_FLOATS:
            raise ValueError(f"{key} cannot be None.")
        return key, None
    if key in _OPTIONAL_FLOATS:
        try:
            return key, float(raw)
        except ValueError:
            raise ValueError(f"{key} expects a number, got {raw!r}.") from None
    if isinstance(default, bool):
        if lowered in ("true", "1", "yes", "on"):
            return key, True
        if lowered in ("false", "0", "no", "off"):
            return key, False
        raise ValueError(f"{key} expects a boolean, got {raw!r}.")
    if key in _OPTIONAL_INTS or isinstance(default, int):
        try:
            return key, int(raw)
        except ValueError:
            raise ValueError(f"{key} expects an integer, got {raw!r}.") from None
    if isinstance(default, float):
        try:
            return key, float(raw)
        except ValueError:
            raise ValueError(f"{key} expects a number, got {raw!r}.") from None
    return key, raw
