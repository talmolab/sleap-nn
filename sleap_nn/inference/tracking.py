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
call (no shared state across ``predict()`` invocations) and runs it
in submission order.

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

import logging
from typing import Callable, Optional

import attrs

import sleap_io as sio

logger = logging.getLogger(__name__)

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


def apply_tracking(
    labels: sio.Labels,
    config: TrackerConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> sio.Labels:
    """Track predicted instances on every frame and run post-cleanup.

    Mirrors :func:`sleap_nn.tracking.tracker.run_tracker` but accepts
    a ``sio.Labels`` directly (instead of a list of LabeledFrames),
    builds a fresh ``Tracker`` per call, and returns a new ``Labels``
    sharing the input's ``videos`` / ``skeletons``.

    Args:
        labels: Untracked predictions. Each ``LabeledFrame``'s
            ``predicted_instances`` are tracked; ``user_instances`` (if
            any) are passed through unchanged and are preferred when
            both are present (matching ``run_tracker`` semantics).
        config: Tracking configuration.
        progress_callback: Optional ``(processed_frames, total_frames)``
            callback invoked after each frame is tracked.

    Returns:
        ``sio.Labels`` with tracked instances. ``videos`` and
        ``skeletons`` are reused; ``provenance`` is left to the caller
        to attach (the CLI builds its own provenance).

    Raises:
        ValueError: ``post_connect_single_breaks=True`` requires
            ``tracking_target_instance_count`` to be set.
    """
    from sleap_nn.tracking.tracker import (
        Tracker,
        connect_single_breaks,
    )
    from sleap_nn.tracking.utils import cull_instances

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
    if len(labels.skeletons) == 1 and len(labels.skeletons[0].nodes) == 1:
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
                "defaults: scoring_method=%r, features=%r.",
                effective_scoring_method,
                effective_features,
            )

    # Segmentation (mask-only) default resolution. A bottom-up segmentation
    # model emits sio.PredictedSegmentationMask into LabeledFrame.masks and no
    # predicted keypoint instances (no skeleton); track masks by pixel mask-IoU.
    # Detect on the labels content (available here, after prediction), mirroring
    # the single-node centroid branch.
    is_mask_mode = any(
        getattr(lf, "masks", None) for lf in labels.labeled_frames
    ) and not any(lf.has_predicted_instances for lf in labels.labeled_frames)
    if is_mask_mode:
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
            "features='masks', scoring_method='mask_iou', window_size=%d, "
            "candidates_method=%r, max_tracks=%s. For best identity, pass the "
            "known animal count via --max_tracks/--tracking_target_instance_count.",
            effective_window_size,
            effective_candidates_method,
            effective_max_tracks,
        )

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

    needs_image = config.use_flow
    tracked_lfs: list = []
    # Track in temporal order. The tracker is stateful across frames (sliding
    # candidate window + optional flow), so frames MUST be visited sorted by
    # (video, frame_idx) — legacy sorted frames (in the predictors'
    # _make_labeled_frames_from_generator) before tracking. Iterating
    # in raw ``labeled_frames`` submission order (e.g. for a .slp whose frames
    # are unordered, or multi-video) produces wrong track assignments
    # (#530 audit: tracking parity / track-only retrack ordering).
    video_order = {id(v): i for i, v in enumerate(labels.videos)}
    ordered_lfs = sorted(
        labels.labeled_frames,
        key=lambda lf: (video_order.get(id(lf.video), 0), lf.frame_idx),
    )
    n_frames = len(ordered_lfs)
    for i, lf in enumerate(ordered_lfs):
        instances: list = []
        masks: list = []
        if is_mask_mode:
            # Track segmentation masks: feed lf.masks through the same tracker
            # (duck-typed), get back the same mask objects with track /
            # tracking_score set, and preserve them on the rebuilt frame so
            # they are NOT dropped (the #614 breakage). Instances stay empty.
            if lf.masks:
                masks = tracker.track(
                    untracked_instances=list(lf.masks),
                    frame_idx=lf.frame_idx,
                    image=None,
                )
        else:
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
        tracked_lfs.append(
            sio.LabeledFrame(
                video=lf.video,
                frame_idx=lf.frame_idx,
                instances=instances,
                masks=masks,
            )
        )
        if progress_callback is not None:
            progress_callback(i + 1, n_frames)

    # Cull/connect cleanups are pose-only (and rejected above for mask mode).
    if not tracked_lfs:
        logger.info("0 frames to track; skipping tracking post-processing.")
    else:
        if not is_mask_mode and config.tracking_clean_instance_count > 0:
            tracked_lfs = cull_instances(
                tracked_lfs,
                config.tracking_clean_instance_count,
                config.tracking_clean_iou_threshold,
            )
            if not config.post_connect_single_breaks:
                tracked_lfs = connect_single_breaks(
                    tracked_lfs, config.tracking_clean_instance_count
                )

        if not is_mask_mode and config.post_connect_single_breaks:
            tracked_lfs = connect_single_breaks(
                tracked_lfs, max_instances=config.tracking_target_instance_count
            )

    return sio.Labels(
        labeled_frames=tracked_lfs,
        videos=list(labels.videos),
        skeletons=list(labels.skeletons),
    )
