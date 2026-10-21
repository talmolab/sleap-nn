"""Tracking integration for the new ``Predictor`` flow.

Wraps :class:`sleap_nn.tracking.tracker.Tracker` (and its post-processing
helpers ``cull_instances`` / ``connect_single_breaks``) as a value-typed
config + a labels-in / labels-out function. ``TrackerConfig.tracker`` selects the
tracker: ``"classic"`` (the default, the tracker above) or one of the modern MOT
trackers of :mod:`sleap_nn.tracking.mot` (SORT, DeepSORT, StrongSORT, ByteTrack,
BoT-SORT, OC-SORT, Deep OC-SORT), which share the same labels-in / labels-out
contract and the same per-video track naming.

Why a config: keeps :class:`~sleap_nn.inference.predictor.Predictor`
picklable and lets the CLI / factory layer build the tracking
configuration once and forward it as plain data, the same shape as
``FilterConfig`` from PR 8.

Why labels-in / labels-out: the tracker is stateful across frames and
operates on ``sio.PredictedInstance`` objects, so the natural seam is
*after* :meth:`Predictor.to_labels` converts ``Outputs`` to
``LabeledFrame``s. ``apply_tracking`` builds a fresh ``Tracker`` per
video (no shared state across ``predict()`` invocations, or across the
videos of one ``Labels``), runs it in ``(video, frame_idx)`` order, and
tracks the given ``Labels`` in place, so every field it does not track
(the other carrier, ROIs, suggestions, ...) passes through untouched.

What this does NOT cover:

* ``--stream-to-file`` + ``--tracking`` — still a UsageError. End-of-
  stream post-processing (``cull_instances`` / ``connect_single_breaks``)
  needs the full LabeledFrame list, which defeats streaming.
* Pre-tracking filter knobs (``filter_max_overlap_*`` /
  ``filter_min_node_count`` / ``filter_min_*_score``) — these run as a
  separate stage in legacy ``run_inference`` and aren't routed through
  the new flow yet. The CLI predicate keeps falling through to legacy
  when those flags are set.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import attrs
from loguru import logger

import numpy as np
import sleap_io as sio

from sleap_nn.tracking.mot import APPEARANCE_TRACKERS, VECTOR_REQUIRED_TRACKERS
from sleap_nn.tracking.mot.params import MOT_TRACKER_NAMES, MOTParams

# Default candidate window. Mask tracking uses a larger default than the
# pose/centroid default because bottom-up segmentation is over-segmented (a
# momentarily over-split or missed instance must survive more frames to keep its
# identity); the cropped mask-IoU (`MaskFeature`) makes the larger window cheap.
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MASK_WINDOW_SIZE = 25

# The tracker families `TrackerConfig.tracker` selects: the classic
# (candidate-window) tracker, then the modern MOT trackers.
CLASSIC_TRACKER = "classic"
TRACKER_NAMES = (CLASSIC_TRACKER,) + MOT_TRACKER_NAMES

# `labels.provenance` key holding what a MOT / identity-linking run resolved (see
# `apply_tracking`).
TRACKING_PROVENANCE_KEY = "tracking"

# TrackerConfig options that only the MOT trackers read (`link_tracklets` is
# shared: it also runs identity linking after the classic tracker).
MOT_ONLY_OPTIONS = (
    "track_buffer",
    "min_hits",
    "track_high_threshold",
    "track_low_threshold",
    "new_track_threshold",
    "min_match_similarity",
    "appearance_threshold",
    "mot_params",
)

# TrackerConfig options that only the classic tracker reads; a MOT tracker refuses
# any of them away from its default (they would be silently ignored).
# `features` / `scoring_method` / `candidates_method` are checked separately.
CLASSIC_ONLY_OPTIONS = (
    "window_size",
    "min_new_track_points",
    "min_match_points",
    "scoring_reduction",
    "robust_best_instance",
    "appearance_weight",
    "euclidean_scale",
    "track_matching_method",
    "use_flow",
    "of_img_scale",
    "of_window_size",
    "of_max_levels",
    "use_kalman",
    "kf_track_features",
    "kf_init_frame_count",
    "kf_node_indices",
    "kf_reset_gap_size",
    "tracking_target_instance_count",
    "tracking_pre_cull_to_target",
    "tracking_pre_cull_iou_threshold",
    "tracking_clean_instance_count",
    "tracking_clean_iou_threshold",
    "post_connect_single_breaks",
)

# CLI flag of a TrackerConfig option, where the names differ (for messages).
_OPTION_FLAGS = {"window_size": "tracking_window_size", "mot_params": "mot_param"}

# `--scoring_method` -> MOT geometric similarity (`MOTParams.similarity`).
_MOT_SIMILARITY = {"oks": "oks", "iou": "iou", "euclidean_dist": "centroid"}

# `--features` values a MOT tracker honors: they pick the carrier.
_MOT_FEATURES = ("keypoints", "masks")


def _flag(name: str) -> str:
    """``--<cli flag>`` of a TrackerConfig option."""
    return "--" + _OPTION_FLAGS.get(name, name)


def _check_tracker(instance, attribute, value) -> None:
    if value not in TRACKER_NAMES:
        raise ValueError(
            f"tracker must be one of {list(TRACKER_NAMES)}, got {value!r}."
        )


def _check_optional_unit(instance, attribute, value) -> None:
    if value is not None and not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{attribute.name} must be in [0, 1], got {value!r}.")


def _check_optional_min(minimum: int):
    def _validator(instance, attribute, value) -> None:
        if value is not None and int(value) < minimum:
            raise ValueError(f"{attribute.name} must be >= {minimum}, got {value!r}.")

    return _validator


def _convert_appearance_threshold(value: Any) -> Optional[Union[float, str]]:
    """``None`` (preset), ``"auto"`` or a minimum cosine similarity in ``[-1, 1]``."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return "auto"
        try:
            value = float(value)
        except ValueError:
            raise ValueError(
                "appearance_threshold must be 'auto' or a cosine similarity in "
                f"[-1, 1], got {value!r}."
            ) from None
    value = float(value)
    if not -1.0 <= value <= 1.0:
        raise ValueError(
            "appearance_threshold must be 'auto' or a cosine similarity in [-1, 1], "
            f"got {value!r}."
        )
    return value


def _convert_mot_params(value: Any) -> Optional[dict]:
    """Copy a ``{MOTParams field: value}`` override dict, checking its keys."""
    if value is None:
        return None
    value = dict(value)
    if not value:
        return None
    valid = (set(MOTParams.field_names()) - {"tracker"}) | {"preset"}
    unknown = sorted(set(value) - valid)
    if unknown:
        raise ValueError(
            f"Unknown MOT parameter(s) {unknown} in mot_params; valid parameters "
            f"are {sorted(valid)}."
        )
    return value


def _explicit_unless_default(name: str, default: Any):
    """Default of a ``*_explicit`` flag, depending on the tracker family.

    Always explicit for the classic tracker (direct constructors keep their
    behavior); for a MOT tracker, explicit only when ``name`` differs from its
    ``TrackerConfig`` default, so that ``TrackerConfig(tracker="sort")`` gets the
    tracker's preset similarity / carrier rather than the classic tracker's
    ``oks`` / ``keypoints`` defaults.
    """

    def _default(self) -> bool:
        return self.tracker == CLASSIC_TRACKER or getattr(self, name) != default

    return attrs.Factory(_default, takes_self=True)


@attrs.frozen(eq=False)
class TrackerConfig:
    """Frozen value type capturing every knob ``run_tracker`` exposes.

    Mirrors :func:`sleap_nn.tracking.tracker.run_tracker`'s signature.
    Picklable so it can sit on :class:`Predictor` without compromising
    the picklability contract from PR 2.

    ``tracker`` selects the tracker family. ``"classic"`` (the default) reads the
    candidate-window options; a modern MOT tracker (:data:`MOT_TRACKER_NAMES`)
    reads the MOT options (``None`` = the tracker's preset, see
    :meth:`~sleap_nn.tracking.mot.params.MOTParams.for_tracker`) plus the shared
    ``max_tracks`` (the known animal count: closed-set identity linking),
    ``oks_stddev`` (``None`` = the MOT default), ``scoring_method`` (when
    explicit: ``oks`` / ``iou`` / ``euclidean_dist`` -> ``centroid`` similarity)
    and ``features`` (``keypoints`` / ``masks``: the carrier). Every option of the
    other family must stay at its default (see :func:`validate_mot_config`).
    ``link_tracklets`` runs identity linking after either family.
    """

    # Tracker.from_config kwargs ────────────────────────────────────────
    window_size: int = DEFAULT_WINDOW_SIZE
    min_new_track_points: int = 0
    candidates_method: str = "fixed_window"
    min_match_points: int = 0
    features: str = "keypoints"
    scoring_method: str = "oks"
    scoring_reduction: str = "mean"
    robust_best_instance: float = 1.0
    oks_stddev: Optional[float] = None
    appearance_weight: float = 0.0
    euclidean_scale: Optional[float] = None
    track_matching_method: str = "hungarian"
    max_tracks: Optional[int] = None
    use_flow: bool = False
    of_img_scale: float = 1.0
    of_window_size: int = 21
    of_max_levels: int = 3
    use_kalman: bool = False
    kf_track_features: str = "centroid"
    kf_init_frame_count: int = 10
    kf_node_indices: Optional[list] = None
    kf_reset_gap_size: int = 5

    # Pre-tracking cull (consumed by Tracker.from_config) ───────────────
    tracking_target_instance_count: Optional[int] = None
    tracking_pre_cull_to_target: int = 0
    tracking_pre_cull_iou_threshold: float = 0.0

    # Post-tracking cleanup (handled in apply_tracking) ─────────────────
    tracking_clean_instance_count: int = 0
    tracking_clean_iou_threshold: float = 0.0
    post_connect_single_breaks: bool = False

    # Tracker family ─────────────────────────────────────────────────────
    # "classic" or a MOT tracker name. Declared before the `*_explicit` flags,
    # whose defaults depend on it.
    tracker: str = attrs.field(default=CLASSIC_TRACKER, validator=_check_tracker)

    # Single-node (centroid) / segmentation (mask) default resolution ───
    # True (the default for the classic tracker) means the user explicitly chose
    # ``scoring_method`` / ``features`` — direct constructors keep current
    # behavior. The CLI sets these False when the corresponding flag was left at
    # its sentinel, which lets ``apply_tracking`` substitute task-appropriate
    # defaults: ``euclidean_dist``/``centroids`` for a 1-node skeleton, and
    # ``mask_iou``/``masks`` for a bottom-up segmentation (mask-only) model.
    # ``candidates_method_explicit`` lets mask mode default the candidate maker to
    # ``local_queues`` (far better identity on over-segmented masks than the
    # ``fixed_window`` default) unless the user explicitly chose a method. For a
    # MOT tracker they default to "differs from the default value" (see
    # `_explicit_unless_default`).
    scoring_method_explicit: bool = attrs.field(
        default=_explicit_unless_default("scoring_method", "oks")
    )
    features_explicit: bool = attrs.field(
        default=_explicit_unless_default("features", "keypoints")
    )
    candidates_method_explicit: bool = attrs.field(
        default=_explicit_unless_default("candidates_method", "fixed_window")
    )

    # Modern MOT trackers (tracker != "classic") ──────────────────────────
    # `None` = the tracker's preset. Thresholds are scores / similarities in
    # [0, 1]; `appearance_threshold` is a minimum cosine or "auto" (calibrated per
    # video). `mot_params` holds `MOTParams` overrides (`--mot_param KEY=VALUE`,
    # `preset` included).
    track_buffer: Optional[int] = attrs.field(
        default=None, validator=_check_optional_min(0)
    )
    min_hits: Optional[int] = attrs.field(
        default=None, validator=_check_optional_min(1)
    )
    track_high_threshold: Optional[float] = attrs.field(
        default=None, validator=_check_optional_unit
    )
    track_low_threshold: Optional[float] = attrs.field(
        default=None, validator=_check_optional_unit
    )
    new_track_threshold: Optional[float] = attrs.field(
        default=None, validator=_check_optional_unit
    )
    min_match_similarity: Optional[float] = attrs.field(
        default=None, validator=_check_optional_unit
    )
    appearance_threshold: Optional[Union[float, str]] = attrs.field(
        default=None, converter=_convert_appearance_threshold
    )
    # Identity linking (long-term identity memory) at the end of each video;
    # works after the classic tracker too.
    link_tracklets: bool = False
    mot_params: Optional[dict] = attrs.field(
        default=None, converter=_convert_mot_params
    )

    @property
    def is_mot(self) -> bool:
        """Whether a modern MOT tracker (not the classic tracker) runs."""
        return self.tracker != CLASSIC_TRACKER

    @property
    def uses_appearance(self) -> bool:
        """Whether the run reads the detections' appearance (re-ID) vectors.

        Classic: ``features="embeddings"`` or an ``appearance_weight`` blend. MOT:
        an appearance tracker, or identity linking (``link_tracklets`` or a known
        ``max_tracks``), which uses vectors where they are present.
        """
        if self.is_mot:
            return (
                self.tracker in APPEARANCE_TRACKERS
                or self.link_tracklets
                or self.max_tracks is not None
            )
        return self.features == "embeddings" or self.appearance_weight > 0.0

    @property
    def follows_vector_carrier(self) -> bool:
        """Whether the tracked carrier is the one holding the appearance vectors.

        Classic: ``features="embeddings"``. MOT: an appearance tracker whose
        ``features`` were not chosen explicitly. Otherwise the carrier follows
        the geometry (:func:`resolve_tracked_carrier`).
        """
        if self.is_mot:
            return self.tracker in APPEARANCE_TRACKERS and not self.features_explicit
        return self.features == "embeddings"

    @property
    def requires_vectors(self) -> bool:
        """Whether a run without any appearance vector is refused.

        Classic: ``features="embeddings"`` or ``appearance_weight > 0``. MOT: the
        appearance trackers (deepsort, strongsort, botsort, deepocsort); for
        deepsort / strongsort the TRACKED carrier must hold them, botsort /
        deepocsort only warn when the vectors ride on the other carrier.
        """
        if self.is_mot:
            return self.tracker in APPEARANCE_TRACKERS
        return self.features == "embeddings" or self.appearance_weight > 0.0


# The two detection carriers of a ``LabeledFrame``: pose instances (``lf.instances``)
# and segmentation masks (``lf.masks``).
POSE_CARRIER = "pose"
MASK_CARRIER = "mask"

# Warn when at least this fraction of the tracked carrier's detections carries no
# appearance vector: under appearance-only tracking each such detection can never
# match and spawns a fresh track; under a blend it silently falls back to geometry.
MISSING_VECTOR_WARN_FRACTION = 0.1


@attrs.frozen
class CarrierCounts:
    """How many detections each carrier holds, and how many of them qualify.

    A ``.slp`` can hold both carriers at once -- top-down segmentation and SAM output
    are poses with linked masks -- so "which carrier?" questions (where are the
    appearance vectors? which carrier was tracked?) are answered by COUNTING, never
    by "any detection anywhere": a single stray mask or vector must not flip a whole
    file. Built by :func:`count_carriers` with the qualifying predicate of the
    question being asked (:func:`embedding_carriers` for appearance vectors).
    """

    n_pose: int = 0
    n_mask: int = 0
    n_pose_with: int = 0
    n_mask_with: int = 0

    def total(self, carrier: str) -> int:
        """Detections on ``carrier``."""
        return self.n_mask if carrier == MASK_CARRIER else self.n_pose

    def n_with(self, carrier: str) -> int:
        """Qualifying detections on ``carrier``."""
        return self.n_mask_with if carrier == MASK_CARRIER else self.n_pose_with

    @property
    def dominant(self) -> Optional[str]:
        """The carrier holding MORE qualifying detections; ``None`` if neither holds any.

        Ties go to the pose carrier, the one every tracker option supports (the
        cull/clean/connect and motion-model options are pose-only).
        """
        if not (self.n_pose_with or self.n_mask_with):
            return None
        return MASK_CARRIER if self.n_mask_with > self.n_pose_with else POSE_CARRIER


def count_carriers(
    labels: sio.Labels, predicate: Callable[[object], bool]
) -> CarrierCounts:
    """Count each carrier's detections, and those satisfying ``predicate``.

    Args:
        labels: Labels to scan (every frame's ``instances`` and ``masks``).
        predicate: ``detection -> bool``, e.g. "carries an appearance vector"
            (:func:`embedding_carriers`) or "carries a track".
    """
    n_pose = n_mask = n_pose_with = n_mask_with = 0
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            n_pose += 1
            n_pose_with += bool(predicate(inst))
        for m in getattr(lf, "masks", None) or []:
            n_mask += 1
            n_mask_with += bool(predicate(m))
    return CarrierCounts(
        n_pose=n_pose, n_mask=n_mask, n_pose_with=n_pose_with, n_mask_with=n_mask_with
    )


def _has_embedding(detection) -> bool:
    return getattr(detection, "identity_embedding", None) is not None


def embedding_carriers(labels: sio.Labels) -> CarrierCounts:
    """Where the appearance (re-ID) vectors are, per carrier.

    Both carriers -- pose ``Instance`` / ``PredictedInstance`` and
    ``PredictedSegmentationMask`` -- hold a vector in the single
    ``identity_embedding`` slot (sleap-io #535). :func:`apply_tracking` routes
    ``features="embeddings"`` to :attr:`CarrierCounts.dominant`, and checks that the
    carrier an ``appearance_weight`` blend tracks actually holds vectors.
    """
    return count_carriers(labels, _has_embedding)


def resolve_tracked_carrier(
    features: Optional[str], has_masks: bool, has_predicted_instances: bool
) -> str:
    """The carrier :func:`apply_tracking` tracks by geometry.

    The masks (``lf.masks``) when the labels hold masks and either no predicted pose
    or an explicit ``features="masks"``; the poses otherwise. It is also the carrier
    an ``appearance_weight`` blend reads its vectors from. (``features="embeddings"``
    instead follows the vectors to the carrier holding more of them.)

    Args:
        features: The tracker's ``features``.
        has_masks: Whether any frame holds a mask.
        has_predicted_instances: Whether any frame holds a predicted pose.
    """
    if has_masks and (not has_predicted_instances or features == "masks"):
        return MASK_CARRIER
    return POSE_CARRIER


def _uses_pose_cleanup(config: TrackerConfig) -> bool:
    """Whether ``config`` asks for the pose-only cull/clean/connect options."""
    return bool(
        config.tracking_pre_cull_to_target
        or config.tracking_clean_instance_count
        or config.post_connect_single_breaks
    )


def resolve_mask_tracking_options(config: TrackerConfig) -> "tuple[str, str]":
    """``(features, scoring_method)`` for tracking the mask carrier by geometry.

    Unset (non-explicit) options resolve to ``masks`` / ``mask_iou``.

    Raises:
        ValueError: An explicit option mask tracking cannot honor: another
            ``features`` / ``scoring_method``, a motion model, or a pose
            cull/clean/connect option.
    """
    scoring_method = (
        config.scoring_method if config.scoring_method_explicit else "mask_iou"
    )
    features = config.features if config.features_explicit else "masks"
    if features != "masks" or scoring_method != "mask_iou":
        raise ValueError(
            "Tracking a bottom-up segmentation (mask-only) model requires "
            "features='masks' and scoring_method='mask_iou' (got features="
            f"{features!r}, scoring_method={scoring_method!r}). "
            "Leave --features/--scoring_method unset to auto-select them."
        )
    # Motion models and pose-shaped cull/clean ops are out of MVP scope for
    # masks (they call .numpy()/same_pose_as on keypoint instances). Fail
    # fast with a clear message rather than crash mid-stream.
    if config.use_flow or config.use_kalman:
        raise ValueError(
            "Mask tracking does not support motion models "
            "(--use_flow/--use_kalman); they are out of scope for the "
            "segmentation tracker MVP."
        )
    if _uses_pose_cleanup(config):
        raise ValueError(
            "Mask tracking does not support the instance cull/clean/connect "
            "options (tracking_pre_cull_to_target / "
            "tracking_clean_instance_count / post_connect_single_breaks); "
            "these operate on keypoint poses, not masks."
        )
    return features, scoring_method


def check_embedding_mask_tracking_options(config: TrackerConfig) -> None:
    """Refuse the pose-only options for appearance-only tracking of masks.

    Raises:
        ValueError: ``config`` sets a pose cull/clean/connect option.
    """
    if _uses_pose_cleanup(config):
        raise ValueError(
            "Embedding tracking on segmentation masks does not support the pose "
            "cull/clean/connect options (tracking_pre_cull_to_target / "
            "tracking_clean_instance_count / post_connect_single_breaks)."
        )


def appearance_blend_way_out(vector_carrier: str) -> str:
    """How to make an ``appearance_weight`` blend read vectors on ``vector_carrier``.

    For a blend that tracks the other carrier, whose detections carry no vector.
    """
    if vector_carrier == MASK_CARRIER:
        return (
            "Track the masks instead (`--features masks`, blending appearance "
            "into mask IoU), or track by appearance alone (`--features "
            "embeddings`, which follows the vectors to their carrier)."
        )
    return (
        "Track the poses instead (a pose `--features`, e.g. "
        "`keypoints`), or track by appearance alone (`--features "
        "embeddings`, which follows the vectors to their carrier)."
    )


def check_tracking_plan(
    config: TrackerConfig,
    *,
    has_masks: bool,
    has_predicted_instances: bool,
    vector_carrier: Optional[str],
    vectors_from: str = "",
) -> None:
    """Raise what :func:`apply_tracking` will raise, before the labels are tracked.

    For the embedding routes, which run a detection stack and/or an embedding pass
    before they track: the carriers the labels will hold and the carrier the vectors
    will be on are known (or predictable) beforehand, so the carrier rules of
    :func:`apply_tracking` can fail first instead of after minutes of inference.
    Checks, in :func:`apply_tracking`'s order: the options that tracking the mask
    carrier cannot honor, then an ``appearance_weight`` blend whose tracked carrier
    will hold no vector (a silent no-op: it reads only the tracked detections' own
    vectors).

    Args:
        config: The tracker configuration.
        has_masks: Whether the labels will hold masks.
        has_predicted_instances: Whether they will hold predicted poses.
        vector_carrier: The carrier the appearance vectors will be on, or ``None``
            when that is not known yet (the carrier checks that need it are then
            left to :func:`apply_tracking`).
        vectors_from: Why the vectors land on ``vector_carrier``, for the message.

    Raises:
        ValueError: The plan cannot be tracked.
    """
    if config.is_mot:
        _check_mot_tracking_plan(
            config,
            has_masks=has_masks,
            has_predicted_instances=has_predicted_instances,
            vector_carrier=vector_carrier,
            vectors_from=vectors_from,
        )
        return
    if config.follows_vector_carrier:
        # Appearance alone follows the vectors to their carrier.
        if vector_carrier == MASK_CARRIER:
            check_embedding_mask_tracking_options(config)
        return
    tracked = resolve_tracked_carrier(
        config.features, has_masks, has_predicted_instances
    )
    if tracked == MASK_CARRIER:
        resolve_mask_tracking_options(config)
    if (
        config.appearance_weight > 0.0
        and vector_carrier is not None
        and vector_carrier != tracked
    ):
        why = f": {vectors_from}" if vectors_from else ""
        raise ValueError(
            f"appearance_weight={config.appearance_weight} would be a silent no-op. "
            f"Tracking follows the {tracked} carrier, but the appearance vectors "
            f"will be on the {vector_carrier} carrier{why}. The blend reads only the "
            f"tracked detections' own vectors. "
            f"{appearance_blend_way_out(vector_carrier)}"
        )


def _other_carrier(carrier: str) -> str:
    return POSE_CARRIER if carrier == MASK_CARRIER else MASK_CARRIER


def _vectors_elsewhere_message(
    config: TrackerConfig, tracked: str, vector_carrier: str, detail: str = ""
) -> str:
    """Why an appearance tracker cannot read vectors that ride on another carrier."""
    way_out = (
        "Drop --features so the tracker follows the vectors to their carrier, or "
        f"pass `--features {'masks' if vector_carrier == MASK_CARRIER else 'keypoints'}`."
    )
    return (
        f"--tracker {config.tracker} associates by appearance, but it tracks the "
        f"{tracked} carrier (--features {config.features}) while the appearance "
        f"vectors are on the {vector_carrier} carrier{detail}. {way_out}"
    )


def _check_mot_tracking_plan(
    config: TrackerConfig,
    *,
    has_masks: bool,
    has_predicted_instances: bool,
    vector_carrier: Optional[str],
    vectors_from: str = "",
) -> None:
    """The MOT half of :func:`check_tracking_plan` (see there for the arguments).

    Raises:
        ValueError: An option of the classic tracker, an invalid MOT parameter, or
            (deepsort / strongsort) vectors that will not be on the tracked
            carrier.
    """
    validate_mot_config(config)
    tracked = resolve_tracked_carrier(
        config.features, has_masks, has_predicted_instances
    )
    if config.follows_vector_carrier and vector_carrier is not None:
        tracked = vector_carrier
    if (
        config.tracker in VECTOR_REQUIRED_TRACKERS
        and vector_carrier is not None
        and vector_carrier != tracked
    ):
        why = f" ({vectors_from})" if vectors_from else ""
        raise ValueError(
            _vectors_elsewhere_message(config, tracked, vector_carrier, why)
        )


def resolve_mot_params(config: TrackerConfig) -> MOTParams:
    """The :class:`~sleap_nn.tracking.mot.params.MOTParams` a MOT config runs with.

    Starts from the tracker's preset (``mot_params["preset"]``, default
    ``"animal"``) and applies, on top, every MOT option that is set (``None`` =
    preset), the shared options (``max_tracks``; ``oks_stddev`` when set; an
    explicit ``scoring_method`` as the geometric similarity: ``oks`` -> ``oks``,
    ``iou`` -> ``iou``, ``euclidean_dist`` -> ``centroid``) and the
    ``mot_params`` overrides.

    Args:
        config: A configuration with ``tracker != "classic"``.

    Returns:
        The resolved, validated parameters.

    Raises:
        ValueError: The classic tracker, an option given both as a flag and in
            ``mot_params``, an unsupported ``scoring_method``, or an invalid
            value.
    """
    if not config.is_mot:
        raise ValueError("resolve_mot_params needs a MOT tracker, got 'classic'.")
    overrides = dict(config.mot_params or {})
    preset = overrides.pop("preset", "animal")
    similarity = None
    if config.scoring_method_explicit:
        if config.scoring_method not in _MOT_SIMILARITY:
            raise ValueError(
                f"--tracker {config.tracker} scores geometry by oks, iou or "
                f"euclidean_dist (--scoring_method), got {config.scoring_method!r}."
            )
        similarity = _MOT_SIMILARITY[config.scoring_method]
    named = {
        "track_buffer": config.track_buffer,
        "min_hits": config.min_hits,
        "track_high_threshold": config.track_high_threshold,
        "track_low_threshold": config.track_low_threshold,
        "new_track_threshold": config.new_track_threshold,
        "min_match_similarity": config.min_match_similarity,
        "appearance_threshold": config.appearance_threshold,
        "similarity": similarity,
        "oks_stddev": config.oks_stddev,
        "link_tracklets": True if config.link_tracklets else None,
        "max_tracks": config.max_tracks,
    }
    named = {k: v for k, v in named.items() if v is not None}
    twice = sorted(set(named) & set(overrides))
    if twice:
        raise ValueError(
            f"MOT parameter(s) {twice} given both as an option and in mot_params "
            "(--mot_param); give each once."
        )
    return MOTParams.for_tracker(config.tracker, preset=preset, **named, **overrides)


def validate_mot_config(config: TrackerConfig) -> None:
    """Refuse options the selected tracker family would silently ignore.

    For the classic tracker: any MOT-only option (``track_buffer``, ...,
    ``mot_params``; ``link_tracklets`` is shared). For a MOT tracker: every
    classic-only option away from its default (the candidate window, the scoring
    knobs, motion models, ``appearance_weight``, the pre-cull / clean / connect
    cleanups -- identity linking covers what those do), ``features`` other than
    ``keypoints`` / ``masks``, an explicit ``candidates_method``, and the MOT
    parameters themselves (resolved with :func:`resolve_mot_params`).

    Args:
        config: The tracker configuration.

    Raises:
        ValueError: Naming every offending option and the way out.
    """
    if not config.is_mot:
        set_mot = [
            name for name in MOT_ONLY_OPTIONS if getattr(config, name) is not None
        ]
        if set_mot:
            raise ValueError(
                f"{', '.join(_flag(n) for n in set_mot)} only appl"
                f"{'ies' if len(set_mot) == 1 else 'y'} to the MOT trackers "
                f"(--tracker {' / '.join(MOT_TRACKER_NAMES)}); the classic "
                "tracker would ignore them. (--link_tracklets works with both.)"
            )
        return

    fields = attrs.fields_dict(TrackerConfig)
    offenders = [
        name
        for name in CLASSIC_ONLY_OPTIONS
        if getattr(config, name) != fields[name].default
    ]
    if config.candidates_method_explicit or config.candidates_method != (
        fields["candidates_method"].default
    ):
        offenders.insert(0, "candidates_method")
    if offenders:
        hints = []
        if {"window_size", "candidates_method"} & set(offenders):
            hints.append("--track_buffer sets how long a lost track is kept")
        if {"use_flow", "use_kalman"} & set(offenders):
            hints.append("the MOT trackers have their own Kalman motion model")
        if "appearance_weight" in offenders:
            hints.append(
                "the appearance trackers (--tracker "
                f"{' / '.join(APPEARANCE_TRACKERS)}) use appearance"
            )
        if {
            "post_connect_single_breaks",
            "tracking_clean_instance_count",
            "tracking_pre_cull_to_target",
            "tracking_target_instance_count",
        } & set(offenders):
            hints.append(
                "identity linking (--link_tracklets, or --max_tracks N for a "
                "known animal count) replaces the cull / clean / connect cleanups"
            )
        raise ValueError(
            f"{', '.join(_flag(n) for n in offenders)} "
            f"{'is an option' if len(offenders) == 1 else 'are options'} of the "
            f"classic tracker; --tracker {config.tracker} would ignore "
            f"{'it' if len(offenders) == 1 else 'them'}."
            + (f" ({'; '.join(hints)}.)" if hints else "")
        )
    if config.features == "embeddings":
        raise ValueError(
            f"--features embeddings is the classic tracker's appearance-only mode; "
            f"with --tracker {config.tracker} use an appearance tracker instead "
            f"(--tracker {' / '.join(APPEARANCE_TRACKERS)})."
        )
    if (
        config.appearance_threshold is not None
        and config.tracker not in APPEARANCE_TRACKERS
    ):
        raise ValueError(
            f"--appearance_threshold only applies to the appearance trackers "
            f"(--tracker {' / '.join(APPEARANCE_TRACKERS)}); --tracker "
            f"{config.tracker} associates by geometry alone."
        )
    if config.features_explicit and config.features not in _MOT_FEATURES:
        raise ValueError(
            f"--tracker {config.tracker} takes --features keypoints or masks (the "
            f"carrier to track), got {config.features!r}. Choose the geometric "
            "similarity with --scoring_method oks / iou / euclidean_dist."
        )
    # Resolves (and so validates) the MOT parameters, scoring_method included.
    resolve_mot_params(config)


def tracker_config_params(config: TrackerConfig) -> Dict[str, Any]:
    """``config`` as a dict for the ``tracking_config`` provenance record.

    For a classic run the MOT options are left out (all unset by
    :func:`validate_mot_config`), so a classic run's provenance is what it was
    before the MOT trackers existed; ``link_tracklets`` is kept when it ran.

    Args:
        config: The tracker configuration.

    Returns:
        ``attrs.asdict(config)``, minus the MOT options of a classic run.
    """
    params = attrs.asdict(config)
    if not config.is_mot:
        params.pop("tracker", None)
        for name in MOT_ONLY_OPTIONS:
            params.pop(name, None)
        if not config.link_tracklets:
            params.pop("link_tracklets", None)
    return params


def _json_safe(value: Any) -> Any:
    """Python-only, JSON-serializable copy of ``value``.

    sleap-io ``json.dumps`` the provenance with no ``default=``: numpy
    scalars/arrays become Python numbers/lists, tuples lists, non-finite floats
    ``None``, keys strings.
    """
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _objects_to_track(lf: sio.LabeledFrame, carrier: str) -> list:
    """The detections of ``lf`` a tracker of ``carrier`` is given.

    Shared by the MOT pre-pass (``prepare``) and the tracking loop, so both see
    exactly the same objects. Masks: every mask of the frame. Poses: the user
    instances when the frame has any (the predicted ones are passed through
    untracked), else the predicted instances.
    """
    if carrier == MASK_CARRIER:
        return list(getattr(lf, "masks", None) or [])
    if lf.has_user_instances:
        return list(lf.user_instances)
    return list(lf.predicted_instances)


def _order_frames(labels: sio.Labels) -> "tuple[list, list]":
    """Frames in ``(video, frame_idx)`` order, and split into one list per video.

    Returns:
        ``(ordered_lfs, per_video)``.
    """
    # Track in temporal order. The tracker is stateful across frames (sliding
    # candidate window + optional flow), so frames MUST be visited sorted by
    # (video, frame_idx) — legacy sorted frames (in the predictors'
    # _make_labeled_frames_from_generator) before tracking. Iterating
    # in raw ``labeled_frames`` submission order (e.g. for a .slp whose frames
    # are unordered, or multi-video) produces wrong track assignments
    # (#530 audit: tracking parity / track-only retrack ordering). Frames of a
    # video missing from ``labels.videos`` sort after the listed videos, as their
    # own video.
    video_rank = {id(v): i for i, v in enumerate(labels.videos)}
    for lf in labels.labeled_frames:
        video_rank.setdefault(id(lf.video), len(video_rank))
    ordered_lfs = sorted(
        labels.labeled_frames,
        key=lambda lf: (video_rank[id(lf.video)], lf.frame_idx),
    )
    per_video: list = []
    for lf in ordered_lfs:
        if per_video and per_video[-1][0].video is lf.video:
            per_video[-1].append(lf)
        else:
            per_video.append([lf])
    return ordered_lfs, per_video


def _finish_tracked_labels(labels: sio.Labels, ordered_lfs: list) -> None:
    """Frames in tracking order and a track catalog of the tracks they reference."""
    # The output IS the input: frames in tracking order, and a track catalog of
    # exactly the tracks the frames reference (tracks the tracker replaced are
    # dropped, as a freshly built Labels would). `reindex` because sleap-io caches
    # a per-track index keyed on frame count, which retracking leaves unchanged.
    labels.labeled_frames = ordered_lfs
    labels.tracks = []
    labels.update()
    labels.reindex()


def _log_mask_inheritance(n_poses_inherited: int, n_poses_mask_mode: int) -> None:
    """Report how many poses of a mask-carrier run inherited their mask's track."""
    if not n_poses_mask_mode:
        return
    if n_poses_inherited:
        logger.info(
            f"Tracking the MASK carrier: {n_poses_inherited} pose instance(s) "
            "inherited the track of the mask linked to them."
        )
    n_left = n_poses_mask_mode - n_poses_inherited
    if n_left:
        logger.warning(
            f"Tracking the MASK carrier, but {n_left} pose instance(s) are not "
            "linked to exactly one tracked mask (`mask.instance`). They are "
            "carried through to the output UNTRACKED by this run (any track "
            "they already had is left as it was) -- only the masks are "
            "tracked. Link each pose to its mask, or attach the appearance "
            "vectors to the pose instances if the poses are what should carry "
            "identity."
        )


def _link_tracked_video(
    video_lfs: list,
    carrier: str,
    track_objects: dict,
    max_tracks: Optional[int],
) -> Dict[str, Any]:
    """Identity linking over one video tracked by the classic tracker, in place.

    Calibrates the video from its tracked detections and runs
    :func:`~sleap_nn.tracking.mot.linking.link_tracklets` with the MOT defaults
    (:class:`~sleap_nn.tracking.mot.params.MOTParams`).

    Returns:
        ``{"calibration": ..., "linking": ...}`` for the provenance record.
    """
    from sleap_nn.tracking.mot.calibration import calibrate
    from sleap_nn.tracking.mot.detections import extract_detections
    from sleap_nn.tracking.mot.linking import link_tracklets

    defaults = {a.name: a.default for a in attrs.fields(MOTParams)}
    frames = [(lf.frame_idx, _objects_to_track(lf, carrier)) for lf in video_lfs]
    dets = [
        extract_detections(
            objs,
            frame_idx,
            carrier=carrier,
            box_padding=defaults["box_padding"],
            score_mode=defaults["score_mode"],
        )
        for frame_idx, objs in frames
    ]
    calibration = calibrate(
        dets,
        track_buffer=defaults["track_buffer"],
        gallery_size=defaults["gallery_size"],
        ema_alpha=defaults["ema_alpha"],
    )
    result = link_tracklets(
        frames,
        carrier=carrier,
        calibration=calibration,
        max_tracks=max_tracks,
        track_buffer=defaults["track_buffer"],
        swap_correction=defaults["link_swap_correction"],
        spatial_gate_frames=defaults["link_spatial_gate_frames"],
        box_padding=defaults["box_padding"],
        high_threshold=defaults["track_high_threshold"],
        score_mode=defaults["score_mode"],
        track_objects=track_objects,
    )
    return {"calibration": calibration.as_dict(), "linking": result.as_dict()}


def _check_mot_vectors(
    config: TrackerConfig, counts: CarrierCounts, carrier: str
) -> None:
    """The appearance-vector rules of a MOT run (on the labels' actual vectors).

    Raises:
        ValueError: An appearance tracker with no vector anywhere, or deepsort /
            strongsort tracking a carrier that holds none (the vectors are on the
            other carrier). botsort / deepocsort then warn and associate on
            geometry alone.
    """
    if not config.requires_vectors:
        # Identity linking uses vectors where present; spatial rules otherwise.
        if config.uses_appearance and counts.n_with(carrier):
            _warn_missing_vectors(counts, carrier, appearance_only=False)
        return
    if counts.dominant is None:
        raise ValueError(
            f"--tracker {config.tracker} associates by appearance, but no "
            "detection in the labels carries an appearance embedding. Run the "
            "embedding (re-ID) model and persist the vectors first (e.g. "
            "`sleap-nn predict --model_paths <embedding_model> ... "
            "--save_embeddings slp`), or use a geometry-only tracker (--tracker "
            "bytetrack / ocsort / sort)."
        )
    if counts.n_with(carrier) == 0:
        other = _other_carrier(carrier)
        detail = f" ({counts.n_with(other)} of {counts.total(other)})"
        message = _vectors_elsewhere_message(config, carrier, other, detail)
        if config.tracker in VECTOR_REQUIRED_TRACKERS:
            raise ValueError(message)
        logger.warning(message + " Associating on geometry alone.")
        return
    _warn_missing_vectors(counts, carrier, appearance_only=False)


def _apply_mot_tracking(
    labels: sio.Labels,
    config: TrackerConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> sio.Labels:
    """:func:`apply_tracking` for a modern MOT tracker (see there for the contract).

    Per video: a fresh tracker, ``prepare`` (calibration + time base) over exactly
    the objects the loop then tracks, ``track`` per frame in temporal order,
    ``finalize`` (identity linking when enabled) and only THEN, on the mask
    carrier, the poses inherit their mask's track -- so retroactive labels of
    tentative tracks and linking decisions reach the poses too. What each video
    resolved is recorded in ``labels.provenance["tracking"]``.
    """
    from sleap_nn.tracking.mot import build_mot_tracker

    # (`apply_tracking` already ran `validate_mot_config`.)
    params = resolve_mot_params(config)

    has_masks = any(getattr(lf, "masks", None) for lf in labels.labeled_frames)
    has_predicted = any(lf.has_predicted_instances for lf in labels.labeled_frames)
    if config.features == "masks" and config.features_explicit and not has_masks:
        raise ValueError(
            "--features masks asks to track the segmentation masks, but no frame "
            "holds a mask."
        )
    counts = embedding_carriers(labels) if config.uses_appearance else CarrierCounts()
    carrier = resolve_tracked_carrier(config.features, has_masks, has_predicted)
    if config.follows_vector_carrier and counts.dominant is not None:
        # An appearance tracker follows its vectors (a pose+mask file may carry
        # them on either carrier) unless --features chose the carrier.
        carrier = counts.dominant
    _check_mot_vectors(config, counts, carrier)

    # Skeleton size (pose carrier) decides the "auto" similarity: a single-node
    # (centroid) skeleton is tracked by centroid distance.
    n_nodes = None
    if carrier == POSE_CARRIER and len(labels.skeletons) == 1:
        n_nodes = len(labels.skeletons[0].nodes)

    logger.info(
        f"MOT tracking: tracker={params.tracker!r} (preset={params.preset!r}), "
        f"carrier={carrier}, similarity={params.similarity!r}, "
        f"track_buffer={params.track_buffer}, min_hits={params.min_hits}, "
        f"identity linking={'on' if params.link else 'off'}"
        + (f" (max_tracks={params.max_tracks})" if params.max_tracks else "")
        + "."
    )

    ordered_lfs, per_video = _order_frames(labels)
    record: Dict[str, Any] = {
        "tracker": params.tracker,
        "carrier": carrier,
        "params": params.as_dict(),
        "videos": [],
    }
    n_frames = len(ordered_lfs)
    n_done = 0
    n_poses_inherited = 0
    n_poses_mask_mode = 0
    track_name_offset = 0
    tracker = None
    for video_idx, video_lfs in enumerate(per_video):
        if tracker is not None:
            # A fresh tracker per video; names continue past the previous video's.
            track_name_offset += max(tracker._track_objects, default=-1) + 1
        tracker = build_mot_tracker(params, carrier=carrier, n_nodes=n_nodes)
        tracker.track_name_offset = track_name_offset

        frames = [(lf.frame_idx, _objects_to_track(lf, carrier)) for lf in video_lfs]
        tracker.prepare(frames)
        for lf, (frame_idx, objs) in zip(video_lfs, frames):
            tracked = tracker.track(objs, frame_idx, image=None)
            if carrier == MASK_CARRIER:
                if objs:
                    lf.masks = tracked
                n_poses_mask_mode += len(lf.instances)
            else:
                passed_through: list = []
                if lf.has_user_instances:
                    # Predicted instances are carried through UNTRACKED: clear any
                    # stale track so the output never mixes two runs' identities.
                    passed_through = list(lf.predicted_instances)
                    for inst in passed_through:
                        inst.track = None
                        inst.tracking_score = 0.0
                lf.instances = passed_through + list(tracked)
            n_done += 1
            if progress_callback is not None:
                progress_callback(n_done, n_frames)

        # Linking (when enabled) relabels detections of earlier frames, so the
        # poses inherit their mask's track only after it.
        tracker.finalize()
        if carrier == MASK_CARRIER:
            for lf in video_lfs:
                n_poses_inherited += _inherit_mask_tracks(lf)

        resolved = {
            k: v for k, v in tracker.resolved.items() if k not in ("tracker", "params")
        }
        record["videos"].append({"video": video_idx, **resolved})

    if not ordered_lfs:
        logger.info("0 frames to track; skipping tracking post-processing.")
    _log_mask_inheritance(n_poses_inherited, n_poses_mask_mode)
    _finish_tracked_labels(labels, ordered_lfs)
    labels.provenance[TRACKING_PROVENANCE_KEY] = _json_safe(record)
    return labels


def _labels_have_embeddings(labels: sio.Labels) -> bool:
    """``True`` if any detection (pose instance or mask) carries a re-ID vector."""
    return embedding_carriers(labels).dominant is not None


def _warn_missing_vectors(
    counts: CarrierCounts, carrier: str, appearance_only: bool
) -> None:
    """Warn when a sizable fraction of the tracked carrier has no appearance vector."""
    total = counts.total(carrier)
    missing = total - counts.n_with(carrier)
    if not total or missing / total < MISSING_VECTOR_WARN_FRACTION:
        return
    consequence = (
        "can never match by appearance, so each one spawns a fresh track"
        if appearance_only
        else "are scored by geometry alone"
    )
    logger.warning(
        f"{missing} of {total} {carrier} detection(s) ({missing / total:.0%}) carry "
        f"no appearance vector; they {consequence}. Embed every detection before "
        "tracking (`sleap-nn predict -m <embedding_model> -i <file>.slp -t` embeds "
        "tracked and untracked detections alike)."
    )


def _inherit_mask_tracks(lf: sio.LabeledFrame) -> int:
    """Copy each tracked mask's track onto the pose instance linked to it.

    Mask-carrier tracking assigns tracks to ``lf.masks``; the poses those masks are
    linked to (``mask.instance``) would otherwise come out untracked, so the poses of
    a top-down-segmentation or SAM file could not be scored or proofread by track.
    Copied only where the link is unambiguous: the instance is on this frame and
    exactly one of the frame's tracked masks links to it.

    Returns:
        How many of the frame's instances inherited a track.
    """
    linked: dict = {}
    for m in lf.masks:
        inst = getattr(m, "instance", None)
        if inst is not None:
            linked.setdefault(id(inst), []).append(m)
    n_inherited = 0
    for inst in lf.instances:
        masks = linked.get(id(inst), [])
        if len(masks) == 1 and masks[0].track is not None:
            inst.track = masks[0].track
            inst.tracking_score = masks[0].tracking_score
            n_inherited += 1
    return n_inherited


def apply_tracking(
    labels: sio.Labels,
    config: TrackerConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> sio.Labels:
    """Track predicted instances on every frame and run post-cleanup, IN PLACE.

    Mirrors :func:`sleap_nn.tracking.tracker.run_tracker` but accepts a
    ``sio.Labels`` directly (instead of a list of LabeledFrames) and runs a fresh
    ``Tracker`` per VIDEO: tracks never span two videos, and their names continue
    across videos (video 2's first new track follows video 1's last) so no two
    videos share a track name.

    The input is modified and returned -- it is NOT copied. Tracks and tracking
    scores are written onto the input's detections, each frame's tracked carrier
    (``lf.instances`` or ``lf.masks``) is replaced by the tracker's output, frames
    are reordered to ``(video, frame_idx)`` order, and ``labels.tracks`` is rebuilt
    to the tracks the frames reference. Everything else passes through untouched:
    the other carrier, ``lf.rois`` / ``lf.centroids`` / ``lf.bboxes``, suggestions,
    sessions, identities, provenance. To keep the input (e.g. an in-memory sweep
    over tracker settings), pass ``labels.copy()``. A lazy ``Labels`` is
    materialized first, so for one the return value is a new object.

    ``config.tracker`` selects the tracker: the classic tracker (default), or a
    modern MOT tracker (:mod:`sleap_nn.tracking.mot`), which tracks every
    detection it is given without dropping any (unconfirmed ones come out with
    ``track=None``), runs identity linking at the end of each video when
    ``link_tracklets`` / ``max_tracks`` ask for it, and records what each video
    resolved (parameters, calibration, linking) in
    ``labels.provenance["tracking"]``. ``link_tracklets`` with the classic tracker
    links each video's tracks after tracking it (recorded the same way).

    Args:
        labels: Untracked predictions. Each ``LabeledFrame``'s
            ``predicted_instances`` are tracked, or its ``user_instances`` when it
            has any (the predicted ones are then carried through untracked,
            matching ``run_tracker`` semantics; a MOT tracker also clears their
            stale tracks). In mask mode ``lf.masks`` is tracked, and each pose
            linked to exactly one tracked mask (``mask.instance``) inherits that
            mask's track.
        config: Tracking configuration.
        progress_callback: Optional ``(processed_frames, total_frames)``
            callback invoked after each frame is tracked.

    Returns:
        ``labels`` itself, tracked. ``provenance`` is left to the caller to
        attach (the CLI builds its own provenance), except the
        ``provenance["tracking"]`` record of a MOT / identity-linking run, which
        the provenance builders carry over.

    Raises:
        ValueError: ``post_connect_single_breaks=True`` requires
            ``tracking_target_instance_count`` to be set; incoherent
            feature/scoring/appearance combinations; ``features="embeddings"``
            or ``appearance_weight > 0`` when the tracked carrier holds no
            appearance vectors; an option the selected tracker family would
            ignore (:func:`validate_mot_config`); an appearance MOT tracker
            without appearance vectors.
    """
    from sleap_nn.tracking.tracker import (
        Tracker,
        connect_single_breaks,
        validate_appearance_config,
    )
    from sleap_nn.tracking.utils import cull_instances

    start_time = datetime.now()
    logger.info(f"Started tracking at: {start_time}")

    # Options of the other tracker family would be silently ignored: refuse them
    # (for the classic tracker this only looks at the MOT-only options).
    validate_mot_config(config)

    # Both post_connect_single_breaks and a non-zero pre-cull target require an
    # explicit tracking_target_instance_count (legacy parity — max_tracks was
    # NEVER accepted as a substitute; the CLI edge layer derives the target from
    # --max_instances before this point, see cli._build_tracker_config). #582.
    if (
        config.post_connect_single_breaks or config.tracking_pre_cull_to_target
    ) and not config.tracking_target_instance_count:
        raise ValueError(
            "post_connect_single_breaks=True and tracking_pre_cull_to_target "
            "require tracking_target_instance_count to be set."
        )

    if labels.is_lazy:
        # Tracking writes onto the detections; a lazy Labels re-materializes them
        # from disk on every access, so the tracks would be lost.
        labels = labels.materialize()

    if config.is_mot:
        # Before the classic single-node / mask / appearance resolution below: a
        # MOT tracker resolves its own carrier, similarity and vector rules.
        labels = _apply_mot_tracking(labels, config, progress_callback)
        _log_tracking_runtime(start_time)
        return labels

    # max_tracks is only honored by the local_queues candidate maker; the
    # fixed_window default silently ignores it. `Tracker.from_config` (the shared
    # tracker constructor below) auto-switches fixed_window -> local_queues and
    # logs an INFO when a track cap is requested, so library callers that build a
    # TrackerConfig directly get the cap honored too (sleap#2720, #582).

    # Single-node (centroid) default resolution. A centroid model collapses to
    # a 1-node Skeleton(['centroid']) (#586); OKS/keypoints are degenerate on a
    # single point, so unless the caller explicitly chose otherwise, substitute
    # euclidean-distance scoring on centroid features. Compute on locals — the
    # frozen config is never mutated. Multi-node / multi-skeleton: unchanged.
    effective_scoring_method = config.scoring_method
    effective_features = config.features
    effective_window_size = config.window_size
    effective_candidates_method = config.candidates_method
    effective_max_tracks = config.max_tracks

    # Embedding (appearance / re-ID) tracking. Selected by an EXPLICIT
    # ``features="embeddings"`` only — apply_tracking never auto-selects it (nothing
    # in the labels distinguishes "track by appearance" from "track by pose"). It
    # tracks by cosine similarity over the appearance vector the embedding model
    # attached to each detection, and works on BOTH pose (``PredictedInstance``) and
    # mask (``PredictedSegmentationMask``) carriers. Resolved here, BEFORE the
    # single-node / mask default branches, so neither clobbers the explicit choice.
    is_embedding_mode = config.features == "embeddings"
    if is_embedding_mode and (
        not config.scoring_method_explicit or effective_scoring_method == "oks"
    ):
        # Auto-pair embeddings with cosine similarity. The CLI leaves
        # scoring_method unset (-> not explicit) so this fires; a direct
        # ``TrackerConfig(features="embeddings")`` inherits the global ``"oks"``
        # default (which is meaningless for a 1-D vector), so correct that too. An
        # explicit vector metric (``euclidean_dist``) is preserved.
        effective_scoring_method = "cosine_sim"

    # Segmentation (mask carrier) detection. A bottom-up segmentation model emits
    # sio.PredictedSegmentationMask into LabeledFrame.masks and no predicted keypoint
    # instances (no skeleton); track masks by pixel mask-IoU. Detect on the labels
    # content (available here, after prediction). An explicit ``features="masks"``
    # also tracks the mask carrier when there are masks to track -- on a pose+mask
    # file (top-down segmentation, SAM) it is the only way to track the masks by
    # geometry. Decided BEFORE the single-node branch, which resolves POSE defaults
    # and must not run (or log) for a mask-carrier run.
    has_masks = any(getattr(lf, "masks", None) for lf in labels.labeled_frames)
    is_mask_mode = (
        resolve_tracked_carrier(
            config.features,
            has_masks,
            any(lf.has_predicted_instances for lf in labels.labeled_frames),
        )
        == MASK_CARRIER
    )

    if (
        not is_embedding_mode
        and not is_mask_mode
        and len(labels.skeletons) == 1
        and len(labels.skeletons[0].nodes) == 1
    ):
        if not config.scoring_method_explicit:
            effective_scoring_method = "euclidean_dist"
        if not config.features_explicit:
            effective_features = "centroids"
        if (
            effective_scoring_method != config.scoring_method
            or effective_features != config.features
        ):
            logger.info(
                "Single-node skeleton detected; applying centroid tracking "
                f"defaults: scoring_method={effective_scoring_method!r}, "
                f"features={effective_features!r}."
            )

    # Segmentation (mask carrier) default resolution.
    if is_mask_mode and not is_embedding_mode:
        # masks / mask_iou, or a ValueError for an option mask tracking cannot
        # honor (shared with the embedding routes' early `check_tracking_plan`).
        effective_features, effective_scoring_method = resolve_mask_tracking_options(
            config
        )
        # Bottom-up segmentation is over-segmented; a larger candidate window
        # keeps identities across transient over-splits/misses. Bump the default
        # only (a non-default window_size is the user's explicit choice).
        if config.window_size == DEFAULT_WINDOW_SIZE:
            effective_window_size = DEFAULT_MASK_WINDOW_SIZE
        # `fixed_window` fragments identity badly on over-segmented masks (its
        # bounded deque forgets any track absent for >window_size frames and mints
        # a fresh id); `local_queues` keeps `current_tracks` and re-binds across
        # gaps. Validated on the 5-mice OFT clip (GT-identity purity: fixed_window
        # 0.28 -> local_queues+cap 0.91). Default to local_queues unless the user
        # explicitly chose a method; cap at the known target count when available
        # (the cap is what lifts local_queues from ~0.52 to ~0.91).
        if not config.candidates_method_explicit:
            effective_candidates_method = "local_queues"
        if effective_max_tracks is None and config.tracking_target_instance_count:
            effective_max_tracks = config.tracking_target_instance_count
        logger.info(
            "Segmentation model detected; applying mask tracking defaults: "
            "features='masks', scoring_method='mask_iou', "
            f"window_size={effective_window_size}, "
            f"candidates_method={effective_candidates_method!r}, "
            f"max_tracks={effective_max_tracks}. For best identity, pass the "
            "known animal count via --max_tracks/--tracking_target_instance_count."
        )

    # Every appearance rule that does not need the labels -- vector-valued metric
    # for `features='embeddings'`, no motion models, a BOUNDED geometric score to
    # blend into, the weight's range -- is shared with `Tracker.from_config` so the
    # legacy `sleap-nn track` command and direct API callers get the same guards.
    # Validated on the RESOLVED effective values, and BEFORE the labels-dependent
    # checks below so an incoherent config is reported ahead of a missing-vectors
    # one.
    validate_appearance_config(
        features=effective_features,
        scoring_method=effective_scoring_method,
        appearance_weight=config.appearance_weight,
        use_flow=config.use_flow,
        use_kalman=config.use_kalman,
        euclidean_scale=config.euclidean_scale,
    )

    # Where the appearance vectors are, counted per carrier -- only when appearance
    # is used, so a geometry-only run never pays for the scan.
    uses_appearance = config.uses_appearance
    vector_counts = embedding_carriers(labels) if uses_appearance else CarrierCounts()

    if is_embedding_mode:
        # Route to the carrier the embeddings actually ride on, NOT the pose/mask
        # presence heuristic: a .slp may have both pose instances and masks but carry
        # the appearance vectors on only one. The default `is_mask_mode` (masks present &&
        # no predicted instances) would, for masks-with-embeddings + pose-instances,
        # track the embedding-less poses (all-NaN -> no association). The carrier
        # holding MORE vectors wins, so a few stray vectors on the other carrier
        # cannot flip the run.
        carrier = vector_counts.dominant
        # The appearance vectors must already ride on the detections'
        # ``identity_embedding`` slot (attached by the `embedding` model); apply_tracking
        # never computes them.
        # Fail loudly if none are present (the common "forgot to run / persist the
        # embedding model" mistake) rather than silently spawning a fresh track per
        # detection (every cosine is NaN -> inf cost -> no match).
        if carrier is None:
            raise ValueError(
                "features='embeddings' but no detection in the labels carries an "
                "appearance embedding. Run the embedding (re-ID) model and "
                "persist the vectors first (e.g. `sleap-nn predict --model_paths "
                "<embedding_model> ... --save_embeddings slp`), then track the "
                "resulting .slp."
            )
        is_mask_mode = carrier == MASK_CARRIER
        # Mask-carried embeddings reuse the mask routing (track ``lf.masks``); the
        # pose-shaped cull/clean/connect ops crash on masks (same as mask_iou mode).
        if is_mask_mode:
            check_embedding_mask_tracking_options(config)
        # Appearance-only tracking is for re-identification after occlusions and
        # across sparse frames. `fixed_window` forgets any track absent for more than
        # `window_size` frames and mints a fresh id when the animal returns, which is
        # exactly the case this mode exists for; `local_queues` keeps every track's
        # own gallery of recent vectors and re-binds across the gap. Default to it
        # unless the user explicitly chose a method (mask mode does the same).
        if not config.candidates_method_explicit:
            effective_candidates_method = "local_queues"
        logger.info(
            "Embedding (appearance) tracking: features='embeddings', "
            f"scoring_method={effective_scoring_method!r}, carrier={carrier}, "
            f"candidates_method={effective_candidates_method!r}."
        )

    tracked_carrier = MASK_CARRIER if is_mask_mode else POSE_CARRIER
    if config.appearance_weight > 0.0:
        other_carrier = POSE_CARRIER if is_mask_mode else MASK_CARRIER
        if vector_counts.dominant is None:
            # `appearance_weight` says "use appearance as a complementary cue", so
            # with no vectors anywhere the blend is a silent no-op byte-identical to
            # weight 0 -- the same "forgot to run / persist the embedding model"
            # mistake `features='embeddings'` fails loudly on. Fail here too.
            raise ValueError(
                f"appearance_weight={config.appearance_weight} was requested but no "
                "detection in the labels carries an appearance embedding, so the "
                "blend would be a silent no-op. Run the embedding (re-ID) model and "
                "persist the vectors first (e.g. `sleap-nn predict --model_paths "
                "<detection_models> <embedding_model> ... --save_embeddings slp`), "
                "or drop --appearance_weight to track on geometry alone."
            )
        if vector_counts.n_with(tracked_carrier) == 0:
            # The blend reads each TRACKED detection's own vector. Vectors that ride
            # on the other carrier (the embedding model puts them on the masks of a
            # pose+mask file) are never read, so the blend would be just as inert.
            # (The embedding routes refuse this before their inference runs; this
            # catches labels that arrive with their vectors already attached.)
            way_out = appearance_blend_way_out(other_carrier)
            raise ValueError(
                f"appearance_weight={config.appearance_weight} was requested, but "
                f"tracking follows the {tracked_carrier} carrier and none of its "
                f"{vector_counts.total(tracked_carrier)} detection(s) carries an "
                "appearance embedding: the vectors are on the "
                f"{other_carrier} carrier ({vector_counts.n_with(other_carrier)} of "
                f"{vector_counts.total(other_carrier)}), so the blend would be a "
                f"silent no-op. {way_out}"
            )
    if uses_appearance:
        _warn_missing_vectors(
            vector_counts, tracked_carrier, appearance_only=is_embedding_mode
        )

    def _new_tracker(track_name_offset: int) -> "Tracker":
        tracker = Tracker.from_config(
            window_size=effective_window_size,
            min_new_track_points=config.min_new_track_points,
            candidates_method=effective_candidates_method,
            min_match_points=config.min_match_points,
            features=effective_features,
            scoring_method=effective_scoring_method,
            scoring_reduction=config.scoring_reduction,
            robust_best_instance=config.robust_best_instance,
            oks_stddev=config.oks_stddev,
            appearance_weight=config.appearance_weight,
            euclidean_scale=config.euclidean_scale,
            track_matching_method=config.track_matching_method,
            max_tracks=effective_max_tracks,
            use_flow=config.use_flow,
            of_img_scale=config.of_img_scale,
            of_window_size=config.of_window_size,
            of_max_levels=config.of_max_levels,
            use_kalman=config.use_kalman,
            kf_track_features=config.kf_track_features,
            kf_init_frame_count=config.kf_init_frame_count,
            kf_node_indices=config.kf_node_indices,
            kf_reset_gap_size=config.kf_reset_gap_size,
            tracking_target_instance_count=config.tracking_target_instance_count,
            tracking_pre_cull_to_target=config.tracking_pre_cull_to_target,
            tracking_pre_cull_iou_threshold=config.tracking_pre_cull_iou_threshold,
        )
        tracker.track_name_offset = track_name_offset
        return tracker

    # Built before the first frame so a config the tracker rejects fails before
    # any work, even on zero frames.
    tracker = _new_tracker(0)

    needs_image = config.use_flow
    ordered_lfs, per_video = _order_frames(labels)

    n_frames = len(ordered_lfs)
    n_done = 0
    n_poses_inherited = 0
    n_poses_mask_mode = 0
    track_name_offset = 0
    linked_videos: list = []
    for video_idx, video_lfs in enumerate(per_video):
        if video_idx:
            # A fresh tracker per video: a track is an identity WITHIN one video, so
            # video 2's first frame must spawn new tracks rather than be matched to
            # video 1's last frame. Names continue past the previous video's.
            track_name_offset += max(tracker._track_objects, default=-1) + 1
            tracker = _new_tracker(track_name_offset)
        for lf in video_lfs:
            if is_mask_mode:
                # Track segmentation masks: feed lf.masks through the same tracker
                # (duck-typed), get back the same mask objects with track /
                # tracking_score set. Pose instances stay on the frame (they are
                # not tracked here); each one linked to a single tracked mask
                # inherits that mask's track once the video is tracked (below).
                if lf.masks:
                    lf.masks = tracker.track(
                        untracked_instances=list(lf.masks),
                        frame_idx=lf.frame_idx,
                        image=None,
                    )
                n_poses_mask_mode += len(lf.instances)
            else:
                instances: list = []
                if lf.has_user_instances:
                    instances_to_track = lf.user_instances
                    if lf.has_predicted_instances:
                        instances = list(lf.predicted_instances)
                else:
                    instances_to_track = lf.predicted_instances
                instances.extend(
                    tracker.track(
                        untracked_instances=instances_to_track,
                        frame_idx=lf.frame_idx,
                        image=lf.image if needs_image else None,
                    )
                )
                lf.instances = instances
            n_done += 1
            if progress_callback is not None:
                progress_callback(n_done, n_frames)

        # Identity linking (long-term identity memory) over the tracked video,
        # before anything reads its tracks: the mask inheritance and the cleanups.
        if config.link_tracklets:
            linked_videos.append(
                {
                    "video": video_idx,
                    **_link_tracked_video(
                        video_lfs,
                        tracked_carrier,
                        tracker._track_objects,
                        effective_max_tracks,
                    ),
                }
            )
        if is_mask_mode:
            for lf in video_lfs:
                n_poses_inherited += _inherit_mask_tracks(lf)

        # Cull/connect cleanups are pose-only (and rejected above for mask mode),
        # and per video like the tracking itself: connecting a break across two
        # videos would join two unrelated identities.
        # Both edit the frames in place.
        if not is_mask_mode and config.tracking_clean_instance_count > 0:
            cull_instances(
                video_lfs,
                config.tracking_clean_instance_count,
                config.tracking_clean_iou_threshold,
            )
            if not config.post_connect_single_breaks:
                connect_single_breaks(video_lfs, config.tracking_clean_instance_count)
        if not is_mask_mode and config.post_connect_single_breaks:
            connect_single_breaks(
                video_lfs, max_instances=config.tracking_target_instance_count
            )

    if not ordered_lfs:
        logger.info("0 frames to track; skipping tracking post-processing.")

    _log_mask_inheritance(n_poses_inherited, n_poses_mask_mode)
    _finish_tracked_labels(labels, ordered_lfs)
    if config.link_tracklets:
        labels.provenance[TRACKING_PROVENANCE_KEY] = _json_safe(
            {
                "tracker": CLASSIC_TRACKER,
                "carrier": tracked_carrier,
                "link_tracklets": True,
                "videos": linked_videos,
            }
        )

    _log_tracking_runtime(start_time)
    return labels


def _log_tracking_runtime(start_time: datetime) -> None:
    """Log the end of a tracking run and its runtime."""
    finish_time = datetime.now()
    logger.info(f"Finished tracking at: {finish_time}")
    logger.info(f"Total runtime: {(finish_time - start_time).total_seconds()} secs")
