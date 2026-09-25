"""Tracking integration for the new ``Predictor`` flow.

Wraps :class:`sleap_nn.tracking.tracker.Tracker` (and its post-processing
helpers ``cull_instances`` / ``connect_single_breaks``) as a value-typed
config + a labels-in / labels-out function.

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

from datetime import datetime
from typing import Callable, Optional

import attrs
from loguru import logger

import sleap_io as sio

# Default candidate window. Mask tracking uses a larger default than the
# pose/centroid default because bottom-up segmentation is over-segmented (a
# momentarily over-split or missed instance must survive more frames to keep its
# identity); the cropped mask-IoU (`MaskFeature`) makes the larger window cheap.
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MASK_WINDOW_SIZE = 25


@attrs.frozen(eq=False)
class TrackerConfig:
    """Frozen value type capturing every knob ``run_tracker`` exposes.

    Mirrors :func:`sleap_nn.tracking.tracker.run_tracker`'s signature.
    Picklable so it can sit on :class:`Predictor` without compromising
    the picklability contract from PR 2.
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

    # Single-node (centroid) / segmentation (mask) default resolution ───
    # True (the default) means the user explicitly chose ``scoring_method`` /
    # ``features`` — direct constructors keep current behavior. The CLI sets
    # these False when the corresponding flag was left at its sentinel, which
    # lets ``apply_tracking`` substitute task-appropriate defaults:
    # ``euclidean_dist``/``centroids`` for a 1-node skeleton, and
    # ``mask_iou``/``masks`` for a bottom-up segmentation (mask-only) model.
    # ``candidates_method_explicit`` lets mask mode default the candidate maker to
    # ``local_queues`` (far better identity on over-segmented masks than the
    # ``fixed_window`` default) unless the user explicitly chose a method.
    scoring_method_explicit: bool = True
    features_explicit: bool = True
    candidates_method_explicit: bool = True


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

    Args:
        labels: Untracked predictions. Each ``LabeledFrame``'s
            ``predicted_instances`` are tracked, or its ``user_instances`` when it
            has any (the predicted ones are then carried through untracked,
            matching ``run_tracker`` semantics). In mask mode ``lf.masks`` is
            tracked, and each pose linked to exactly one tracked mask
            (``mask.instance``) inherits that mask's track.
        config: Tracking configuration.
        progress_callback: Optional ``(processed_frames, total_frames)``
            callback invoked after each frame is tracked.

    Returns:
        ``labels`` itself, tracked. ``provenance`` is left to the caller to
        attach (the CLI builds its own provenance).

    Raises:
        ValueError: ``post_connect_single_breaks=True`` requires
            ``tracking_target_instance_count`` to be set; incoherent
            feature/scoring/appearance combinations; ``features="embeddings"``
            or ``appearance_weight > 0`` when the tracked carrier holds no
            appearance vectors.
    """
    from sleap_nn.tracking.tracker import (
        Tracker,
        connect_single_breaks,
        validate_appearance_config,
    )
    from sleap_nn.tracking.utils import cull_instances

    start_time = datetime.now()
    logger.info(f"Started tracking at: {start_time}")

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
    is_mask_mode = has_masks and (
        not any(lf.has_predicted_instances for lf in labels.labeled_frames)
        or (not is_embedding_mode and config.features == "masks")
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
        if not config.scoring_method_explicit:
            effective_scoring_method = "mask_iou"
        if not config.features_explicit:
            effective_features = "masks"
        if effective_features != "masks" or effective_scoring_method != "mask_iou":
            raise ValueError(
                "Tracking a bottom-up segmentation (mask-only) model requires "
                "features='masks' and scoring_method='mask_iou' (got features="
                f"{effective_features!r}, scoring_method={effective_scoring_method!r}). "
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
        if (
            config.tracking_pre_cull_to_target
            or config.tracking_clean_instance_count
            or config.post_connect_single_breaks
        ):
            raise ValueError(
                "Mask tracking does not support the instance cull/clean/connect "
                "options (tracking_pre_cull_to_target / "
                "tracking_clean_instance_count / post_connect_single_breaks); "
                "these operate on keypoint poses, not masks."
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
    uses_appearance = is_embedding_mode or config.appearance_weight > 0.0
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
        if is_mask_mode and (
            config.tracking_pre_cull_to_target
            or config.tracking_clean_instance_count
            or config.post_connect_single_breaks
        ):
            raise ValueError(
                "Embedding tracking on segmentation masks does not support the pose "
                "cull/clean/connect options (tracking_pre_cull_to_target / "
                "tracking_clean_instance_count / post_connect_single_breaks)."
            )
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
            way_out = (
                "Track the masks instead (`--features masks`, blending appearance "
                "into mask IoU), or track by appearance alone (`--features "
                "embeddings`, which follows the vectors to their carrier)."
                if other_carrier == MASK_CARRIER
                else "Track the poses instead (a pose `--features`, e.g. "
                "`keypoints`), or track by appearance alone (`--features "
                "embeddings`, which follows the vectors to their carrier)."
            )
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

    n_frames = len(ordered_lfs)
    n_done = 0
    n_poses_inherited = 0
    n_poses_mask_mode = 0
    track_name_offset = 0
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
                # inherits that mask's track.
                if lf.masks:
                    lf.masks = tracker.track(
                        untracked_instances=list(lf.masks),
                        frame_idx=lf.frame_idx,
                        image=None,
                    )
                    n_poses_inherited += _inherit_mask_tracks(lf)
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

    if n_poses_mask_mode:
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

    # The output IS the input: frames in tracking order, and a track catalog of
    # exactly the tracks the frames reference (tracks the tracker replaced are
    # dropped, as a freshly built Labels would). `reindex` because sleap-io caches
    # a per-track index keyed on frame count, which retracking leaves unchanged.
    labels.labeled_frames = ordered_lfs
    labels.tracks = []
    labels.update()
    labels.reindex()

    finish_time = datetime.now()
    logger.info(f"Finished tracking at: {finish_time}")
    logger.info(f"Total runtime: {(finish_time - start_time).total_seconds()} secs")

    return labels
