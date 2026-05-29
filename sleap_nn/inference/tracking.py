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
from typing import Optional

import attrs

import sleap_io as sio

logger = logging.getLogger(__name__)


@attrs.frozen(eq=False)
class TrackerConfig:
    """Frozen value type capturing every knob ``run_tracker`` exposes.

    Mirrors :func:`sleap_nn.tracking.tracker.run_tracker`'s signature.
    Picklable so it can sit on :class:`Predictor` without compromising
    the picklability contract from PR 2.
    """

    # Tracker.from_config kwargs ────────────────────────────────────────
    window_size: int = 5
    min_new_track_points: int = 0
    candidates_method: str = "fixed_window"
    min_match_points: int = 0
    features: str = "keypoints"
    scoring_method: str = "oks"
    scoring_reduction: str = "mean"
    robust_best_instance: float = 1.0
    track_matching_method: str = "hungarian"
    max_tracks: Optional[int] = None
    use_flow: bool = False
    of_img_scale: float = 1.0
    of_window_size: int = 21
    of_max_levels: int = 3

    # Pre-tracking cull (consumed by Tracker.from_config) ───────────────
    tracking_target_instance_count: Optional[int] = None
    tracking_pre_cull_to_target: int = 0
    tracking_pre_cull_iou_threshold: float = 0.0

    # Post-tracking cleanup (handled in apply_tracking) ─────────────────
    tracking_clean_instance_count: int = 0
    tracking_clean_iou_threshold: float = 0.0
    post_connect_single_breaks: bool = False


def apply_tracking(
    labels: sio.Labels,
    config: TrackerConfig,
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
    # fixed_window default silently ignores it. The CLI auto-switches to
    # local_queues, but a library caller can still hit this — warn rather than
    # silently drop the cap (#582).
    if config.max_tracks is not None and config.candidates_method == "fixed_window":
        logger.warning(
            "max_tracks=%s is set but candidates_method='fixed_window' ignores "
            "it. Use candidates_method='local_queues' to honor max_tracks.",
            config.max_tracks,
        )

    tracker = Tracker.from_config(
        window_size=config.window_size,
        min_new_track_points=config.min_new_track_points,
        candidates_method=config.candidates_method,
        min_match_points=config.min_match_points,
        features=config.features,
        scoring_method=config.scoring_method,
        scoring_reduction=config.scoring_reduction,
        robust_best_instance=config.robust_best_instance,
        track_matching_method=config.track_matching_method,
        max_tracks=config.max_tracks,
        use_flow=config.use_flow,
        of_img_scale=config.of_img_scale,
        of_window_size=config.of_window_size,
        of_max_levels=config.of_max_levels,
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
    for lf in ordered_lfs:
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
        tracked_lfs.append(
            sio.LabeledFrame(
                video=lf.video,
                frame_idx=lf.frame_idx,
                instances=instances,
            )
        )

    if config.tracking_clean_instance_count > 0:
        tracked_lfs = cull_instances(
            tracked_lfs,
            config.tracking_clean_instance_count,
            config.tracking_clean_iou_threshold,
        )
        if not config.post_connect_single_breaks:
            tracked_lfs = connect_single_breaks(
                tracked_lfs, config.tracking_clean_instance_count
            )

    if config.post_connect_single_breaks:
        tracked_lfs = connect_single_breaks(
            tracked_lfs, max_instances=config.tracking_target_instance_count
        )

    return sio.Labels(
        labeled_frames=tracked_lfs,
        videos=list(labels.videos),
        skeletons=list(labels.skeletons),
    )
