"""ID reconciliation for matching SAM3 masks to poses or input masks.

Lifted **verbatim** from ``talmolab/sam-track`` (BSD-3-Clause) at commit
``7b2531d92b5035f5f83016b12350f7c394b92522``:

    https://github.com/talmolab/sam-track/blob/7b2531d92b5035f5f83016b12350f7c394b92522/src/sam_track/reconciliation.py

Only this attribution header was added; the implementation below is unchanged.

BSD 3-Clause License

Copyright (c) 2025, Talmo Lab

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

----

This module provides tools for:
- Matching SAM3 segmentation masks to pose instances using Hungarian algorithm
- Matching SAM3 segmentation masks to input masks using IoU
- Detecting identity swaps across frames
- Building frame-to-ID mappings for output reconciliation

The key insight from experimentation is that SAM3 mid-propagation re-prompting
works for adding NEW objects, but NOT for correcting existing tracks. Therefore,
identity correction must be done via post-processing with mask-to-pose matching
or mask-to-mask matching (IoU-based).
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    import sleap_io as sio


@dataclass
class MatchContext:
    """Context for match predicate evaluation.

    Attributes:
        frame_idx: Frame index where the match was made.
        sam3_obj_id: SAM3 object ID of the matched mask.
        cost: Raw cost from the cost matrix (negative keypoints inside).
        keypoints_inside: Number of visible keypoints inside the mask.
        keypoints_visible: Total number of visible keypoints in the pose.
        mask_area: Area of the mask in pixels.
        mask_centroid: Centroid of the mask as (x, y).
    """

    frame_idx: int
    sam3_obj_id: int
    cost: float
    keypoints_inside: int
    keypoints_visible: int
    mask_area: int
    mask_centroid: tuple[float, float]


# Type alias for match predicates
MatchPredicate = Callable[["sio.Instance", np.ndarray, MatchContext], bool]


@dataclass
class TrackAssignment:
    """A single track assignment at a frame.

    Attributes:
        frame_idx: Frame index where assignment was made.
        pose_track_name: Name of the pose's track (None if untracked).
        pose_idx: Index of the pose in the frame's instance list.
        sam3_obj_id: SAM3 object ID that was matched.
        confidence: Match quality score (0-1, higher is better).
        sam3_score: SAM3 mask detection confidence score.
    """

    frame_idx: int
    pose_track_name: str | None
    pose_idx: int
    sam3_obj_id: int
    confidence: float
    sam3_score: float = 1.0


@dataclass
class SwapEvent:
    """Detected identity swap.

    Attributes:
        frame_idx: Frame where the swap was detected.
        track_name: Name of the track that swapped.
        old_sam3_id: Previous SAM3 object ID.
        new_sam3_id: New SAM3 object ID after swap.
    """

    frame_idx: int
    track_name: str
    old_sam3_id: int
    new_sam3_id: int


@dataclass
class MaskAssignment:
    """A single mask-to-mask assignment at a frame.

    Used for matching input (anchor) masks to SAM3 output masks.

    Attributes:
        frame_idx: Frame index where assignment was made.
        input_track_id: Track ID from the input/anchor mask.
        input_track_name: Track name from the input/anchor mask.
        sam3_obj_id: SAM3 object ID that was matched.
        iou: Intersection over Union score for the match.
        sam3_score: SAM3 mask detection confidence score.
    """

    frame_idx: int
    input_track_id: int
    input_track_name: str | None
    sam3_obj_id: int
    iou: float
    sam3_score: float = 1.0


def default_match_predicate(
    pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext
) -> bool:
    """Default match predicate: require at least 1 keypoint inside mask."""
    return ctx.keypoints_inside >= 1


@dataclass
class IDReconciler:
    """Matches SAM3 masks to poses and reconciles track IDs.

    This class implements Hungarian algorithm matching between pose instances
    and SAM3 segmentation masks, using keypoints-inside-mask as the cost metric.

    Attributes:
        skeleton: The SLEAP skeleton for node name lookups.
        exclude_nodes: Set of node names to exclude from matching.
        match_predicates: List of predicates that must all pass for a valid match.

    Example:
        >>> reconciler = IDReconciler(
        ...     skeleton=handler.skeleton,
        ...     exclude_nodes={"tail0", "tail1"},
        ... )
        >>> for frame_idx in gt_frame_indices:
        ...     assignments = reconciler.match_frame(
        ...         frame_idx=frame_idx,
        ...         poses=lf.instances,
        ...         masks=result.masks,
        ...         object_ids=result.object_ids,
        ...     )
        >>> swaps = reconciler.detect_swaps()
        >>> id_map = reconciler.build_id_map()
    """

    skeleton: "sio.Skeleton"
    exclude_nodes: set[str] = field(default_factory=set)
    match_predicates: list[MatchPredicate] = field(default_factory=list)
    ignore_gt_tracks: bool = False
    _assignments: list[TrackAssignment] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Add default predicate if none provided."""
        if not self.match_predicates:
            self.match_predicates = [default_match_predicate]

    def compute_cost_matrix(
        self,
        poses: list["sio.Instance"],
        masks: np.ndarray,
    ) -> np.ndarray:
        """Compute cost matrix for Hungarian matching.

        The cost is the negative number of visible keypoints inside each mask.
        Lower cost = better match (more keypoints inside).

        Args:
            poses: List of pose instances to match.
            masks: Array of masks with shape (N, H, W).

        Returns:
            Cost matrix with shape (n_poses, n_masks).
        """
        n_poses = len(poses)
        n_masks = len(masks)

        if n_poses == 0 or n_masks == 0:
            return np.zeros((n_poses, n_masks))

        cost = np.zeros((n_poses, n_masks))

        # Get node names for filtering
        node_names = [n.name for n in self.skeleton.nodes]

        for i, pose in enumerate(poses):
            coords = pose.numpy()
            visible_mask = ~np.isnan(coords[:, 0])

            # Apply node exclusion filter
            if self.exclude_nodes:
                for j, name in enumerate(node_names):
                    if name in self.exclude_nodes:
                        visible_mask[j] = False

            visible_coords = coords[visible_mask].astype(int)

            for j, mask in enumerate(masks):
                inside_count = 0
                for x, y in visible_coords:
                    if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                        if mask[y, x]:
                            inside_count += 1

                # Negative because Hungarian minimizes cost
                cost[i, j] = -inside_count

        return cost

    def match_frame(
        self,
        frame_idx: int,
        poses: list["sio.Instance"],
        masks: np.ndarray,
        object_ids: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> list[TrackAssignment]:
        """Match poses to masks for a single frame.

        Uses Hungarian algorithm for optimal assignment, then filters
        matches through predicates.

        Args:
            frame_idx: Frame index for this match.
            poses: List of pose instances to match.
            masks: Array of masks with shape (N, H, W) or (N, 1, H, W).
            object_ids: Array of SAM3 object IDs corresponding to masks.
            scores: Optional SAM3 mask detection confidence scores, shape (N,).

        Returns:
            List of valid TrackAssignment objects.
        """
        if len(poses) == 0 or len(masks) == 0:
            return []

        # Default scores to 1.0 if not provided
        if scores is None:
            scores = np.ones(len(object_ids))

        # Handle (N, 1, H, W) mask format from SAM3
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(axis=1)

        # Compute cost matrix and solve assignment
        cost = self.compute_cost_matrix(poses, masks)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Get node names for visibility calculation
        node_names = [n.name for n in self.skeleton.nodes]

        assignments = []
        for pose_idx, mask_idx in zip(row_ind, col_ind):
            pose = poses[pose_idx]
            mask = masks[mask_idx]

            # Calculate visibility (excluding filtered nodes)
            coords = pose.numpy()
            visible_mask = ~np.isnan(coords[:, 0])
            if self.exclude_nodes:
                for j, name in enumerate(node_names):
                    if name in self.exclude_nodes:
                        visible_mask[j] = False
            visible_count = int(visible_mask.sum())

            # Calculate mask statistics
            ys, xs = np.where(mask)
            if len(xs) > 0:
                centroid = (float(xs.mean()), float(ys.mean()))
                mask_area = int(mask.sum())
            else:
                centroid = (0.0, 0.0)
                mask_area = 0

            # Build context for predicate evaluation
            keypoints_inside = int(-cost[pose_idx, mask_idx])
            ctx = MatchContext(
                frame_idx=frame_idx,
                sam3_obj_id=int(object_ids[mask_idx]),
                cost=float(cost[pose_idx, mask_idx]),
                keypoints_inside=keypoints_inside,
                keypoints_visible=visible_count,
                mask_area=mask_area,
                mask_centroid=centroid,
            )

            # Apply match predicates
            if all(pred(pose, mask, ctx) for pred in self.match_predicates):
                # Use None for track_name when ignoring GT tracks
                if self.ignore_gt_tracks:
                    track_name = None
                else:
                    track_name = pose.track.name if pose.track else None
                confidence = (
                    keypoints_inside / visible_count if visible_count > 0 else 0.0
                )
                assignment = TrackAssignment(
                    frame_idx=frame_idx,
                    pose_track_name=track_name,
                    pose_idx=pose_idx,
                    sam3_obj_id=ctx.sam3_obj_id,
                    confidence=confidence,
                    sam3_score=float(scores[mask_idx]),
                )
                assignments.append(assignment)

        self._assignments.extend(assignments)
        return assignments

    def detect_swaps(self) -> list[SwapEvent]:
        """Detect identity swaps from accumulated assignments.

        A swap occurs when a track name is matched to different SAM3 object IDs
        across frames.

        Returns:
            List of SwapEvent objects describing detected swaps.
        """
        swaps = []
        by_track: dict[str, list[TrackAssignment]] = defaultdict(list)

        for a in self._assignments:
            if a.pose_track_name:
                by_track[a.pose_track_name].append(a)

        for track_name, track_assignments in by_track.items():
            track_assignments.sort(key=lambda a: a.frame_idx)

            for i in range(1, len(track_assignments)):
                prev = track_assignments[i - 1]
                curr = track_assignments[i]

                if prev.sam3_obj_id != curr.sam3_obj_id:
                    swaps.append(
                        SwapEvent(
                            frame_idx=curr.frame_idx,
                            track_name=track_name,
                            old_sam3_id=prev.sam3_obj_id,
                            new_sam3_id=curr.sam3_obj_id,
                        )
                    )

        return swaps

    def build_id_map(self) -> dict[int, dict[int, str]]:
        """Build frame -> {sam3_id -> track_name} mapping.

        This can be used to remap SAM3 object IDs to consistent track names
        in output files.

        Returns:
            Dictionary mapping frame_idx to {sam3_obj_id: track_name}.
        """
        by_frame: dict[int, dict[int, str]] = defaultdict(dict)
        for a in self._assignments:
            if a.pose_track_name:
                by_frame[a.frame_idx][a.sam3_obj_id] = a.pose_track_name
        return dict(by_frame)

    def get_assignments(self) -> list[TrackAssignment]:
        """Get all accumulated assignments.

        Returns:
            List of all TrackAssignment objects from match_frame() calls.
        """
        return list(self._assignments)

    def clear(self) -> None:
        """Clear accumulated assignments."""
        self._assignments.clear()


@dataclass
class MaskReconciler:
    """Matches SAM3 masks to input masks using IoU and reconciles track IDs.

    This class implements Hungarian algorithm matching between input/anchor masks
    and SAM3 segmentation masks, using IoU (Intersection over Union) as the cost
    metric. This enables post-hoc identity correction using sparse ground truth
    mask annotations.

    Unlike IDReconciler (which uses keypoints-in-mask for pose matching), this
    reconciler works purely with mask overlap, making it suitable for workflows
    where users have corrected masks at specific frames that should be used as
    identity anchors.

    Attributes:
        min_iou: Minimum IoU threshold for a valid match. Matches below this
            threshold are rejected.
        track_names: Optional mapping of input track_id -> name for naming.

    Example:
        >>> reconciler = MaskReconciler(min_iou=0.3)
        >>> for frame_idx in anchor_frames:
        ...     assignments = reconciler.match_frame(
        ...         frame_idx=frame_idx,
        ...         input_masks=reader.get_masks(frame_idx),
        ...         input_track_ids=reader.get_track_ids(frame_idx),
        ...         sam3_masks=result.masks,
        ...         sam3_obj_ids=result.object_ids,
        ...     )
        >>> swaps = reconciler.detect_swaps()
        >>> id_map = reconciler.build_id_map()
    """

    min_iou: float = 0.3
    track_names: dict[int, str] = field(default_factory=dict)
    _assignments: list[MaskAssignment] = field(default_factory=list, repr=False)

    @staticmethod
    def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute Intersection over Union between two binary masks.

        Args:
            mask1: First binary mask as (H, W) array.
            mask2: Second binary mask as (H, W) array.

        Returns:
            IoU score between 0 and 1.
        """
        # Convert to boolean for logical operations
        m1 = mask1.astype(bool)
        m2 = mask2.astype(bool)

        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)

    def compute_cost_matrix(
        self,
        input_masks: np.ndarray,
        sam3_masks: np.ndarray,
    ) -> np.ndarray:
        """Compute cost matrix for Hungarian matching.

        The cost is the negative IoU (because Hungarian minimizes cost).
        Lower cost = better match (higher IoU).

        Args:
            input_masks: Input/anchor masks with shape (N, H, W).
            sam3_masks: SAM3 output masks with shape (M, H, W) or (M, 1, H, W).

        Returns:
            Cost matrix with shape (n_input, n_sam3).
        """
        # Handle (M, 1, H, W) mask format from SAM3
        if sam3_masks.ndim == 4 and sam3_masks.shape[1] == 1:
            sam3_masks = sam3_masks.squeeze(axis=1)

        n_input = len(input_masks)
        n_sam3 = len(sam3_masks)

        if n_input == 0 or n_sam3 == 0:
            return np.zeros((n_input, n_sam3))

        cost = np.zeros((n_input, n_sam3))

        for i, input_mask in enumerate(input_masks):
            for j, sam3_mask in enumerate(sam3_masks):
                iou = self.compute_iou(input_mask, sam3_mask)
                # Negative because Hungarian minimizes cost
                cost[i, j] = -iou

        return cost

    def match_frame(
        self,
        frame_idx: int,
        input_masks: np.ndarray,
        input_track_ids: np.ndarray,
        sam3_masks: np.ndarray,
        sam3_obj_ids: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> list[MaskAssignment]:
        """Match input masks to SAM3 masks for a single frame.

        Uses Hungarian algorithm for optimal assignment, then filters
        matches by IoU threshold.

        Args:
            frame_idx: Frame index for this match.
            input_masks: Input/anchor masks with shape (N, H, W).
            input_track_ids: Track IDs corresponding to input masks.
            sam3_masks: SAM3 output masks with shape (M, H, W) or (M, 1, H, W).
            sam3_obj_ids: SAM3 object IDs corresponding to SAM3 masks.
            scores: Optional SAM3 mask detection confidence scores, shape (M,).

        Returns:
            List of valid MaskAssignment objects.
        """
        if len(input_masks) == 0 or len(sam3_masks) == 0:
            return []

        # Default scores to 1.0 if not provided
        if scores is None:
            scores = np.ones(len(sam3_obj_ids))

        # Compute cost matrix and solve assignment
        cost = self.compute_cost_matrix(input_masks, sam3_masks)
        row_ind, col_ind = linear_sum_assignment(cost)

        assignments = []
        for input_idx, sam3_idx in zip(row_ind, col_ind):
            iou = -cost[input_idx, sam3_idx]  # Convert back from negative

            # Apply IoU threshold
            if iou < self.min_iou:
                continue

            input_track_id = int(input_track_ids[input_idx])
            track_name = self.track_names.get(input_track_id)

            assignment = MaskAssignment(
                frame_idx=frame_idx,
                input_track_id=input_track_id,
                input_track_name=track_name,
                sam3_obj_id=int(sam3_obj_ids[sam3_idx]),
                iou=iou,
                sam3_score=float(scores[sam3_idx]),
            )
            assignments.append(assignment)

        self._assignments.extend(assignments)
        return assignments

    def detect_swaps(self) -> list[SwapEvent]:
        """Detect identity swaps from accumulated assignments.

        A swap occurs when an input track is matched to different SAM3 object IDs
        across frames.

        Returns:
            List of SwapEvent objects describing detected swaps.
        """
        swaps = []
        by_track: dict[int, list[MaskAssignment]] = defaultdict(list)

        for a in self._assignments:
            by_track[a.input_track_id].append(a)

        for input_track_id, track_assignments in by_track.items():
            track_assignments.sort(key=lambda a: a.frame_idx)

            for i in range(1, len(track_assignments)):
                prev = track_assignments[i - 1]
                curr = track_assignments[i]

                if prev.sam3_obj_id != curr.sam3_obj_id:
                    # Use track name if available, otherwise use "track_{id}"
                    track_name = (
                        curr.input_track_name
                        or self.track_names.get(input_track_id)
                        or f"track_{input_track_id}"
                    )
                    swaps.append(
                        SwapEvent(
                            frame_idx=curr.frame_idx,
                            track_name=track_name,
                            old_sam3_id=prev.sam3_obj_id,
                            new_sam3_id=curr.sam3_obj_id,
                        )
                    )

        return swaps

    def build_id_map(self) -> dict[int, dict[int, str]]:
        """Build frame -> {sam3_id -> track_name} mapping.

        This can be used to remap SAM3 object IDs to consistent track names
        in output files.

        Returns:
            Dictionary mapping frame_idx to {sam3_obj_id: track_name}.
        """
        by_frame: dict[int, dict[int, str]] = defaultdict(dict)
        for a in self._assignments:
            name = (
                a.input_track_name
                or self.track_names.get(a.input_track_id)
                or f"track_{a.input_track_id}"
            )
            by_frame[a.frame_idx][a.sam3_obj_id] = name
        return dict(by_frame)

    def get_assignments(self) -> list[MaskAssignment]:
        """Get all accumulated assignments.

        Returns:
            List of all MaskAssignment objects from match_frame() calls.
        """
        return list(self._assignments)

    def get_iou_stats(self) -> dict[str, float]:
        """Get IoU statistics from accumulated assignments.

        Returns:
            Dictionary with 'min', 'max', 'mean', 'median' IoU values.
        """
        if not self._assignments:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

        ious = [a.iou for a in self._assignments]
        return {
            "min": float(min(ious)),
            "max": float(max(ious)),
            "mean": float(np.mean(ious)),
            "median": float(np.median(ious)),
        }

    def clear(self) -> None:
        """Clear accumulated assignments."""
        self._assignments.clear()


# Predefined match predicates for common use cases


def require_min_keypoints_inside(min_count: int = 3) -> MatchPredicate:
    """Create predicate requiring minimum keypoints inside mask.

    Args:
        min_count: Minimum number of keypoints required inside mask.

    Returns:
        A MatchPredicate function.
    """

    def predicate(pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext) -> bool:
        return ctx.keypoints_inside >= min_count

    return predicate


def require_min_fraction_inside(min_frac: float = 0.5) -> MatchPredicate:
    """Create predicate requiring minimum fraction of keypoints inside mask.

    Args:
        min_frac: Minimum fraction (0-1) of visible keypoints inside mask.

    Returns:
        A MatchPredicate function.
    """

    def predicate(pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext) -> bool:
        if ctx.keypoints_visible == 0:
            return False
        return ctx.keypoints_inside / ctx.keypoints_visible >= min_frac

    return predicate


def require_centroid_proximity(max_dist: float = 100.0) -> MatchPredicate:
    """Create predicate requiring pose centroid near mask centroid.

    Args:
        max_dist: Maximum allowed distance between centroids in pixels.

    Returns:
        A MatchPredicate function.
    """

    def predicate(pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext) -> bool:
        coords = pose.numpy()
        pose_centroid = np.nanmean(coords, axis=0)
        if np.any(np.isnan(pose_centroid)):
            return False
        dist = np.linalg.norm(pose_centroid - np.array(ctx.mask_centroid))
        return float(dist) <= max_dist

    return predicate


def require_reasonable_mask_area(
    min_area: int = 1000, max_area: int = 500000
) -> MatchPredicate:
    """Create predicate requiring mask area within bounds.

    Args:
        min_area: Minimum mask area in pixels.
        max_area: Maximum mask area in pixels.

    Returns:
        A MatchPredicate function.
    """

    def predicate(pose: "sio.Instance", mask: np.ndarray, ctx: MatchContext) -> bool:
        return min_area <= ctx.mask_area <= max_area

    return predicate


@dataclass
class TrackNameResolver:
    """Resolves SAM3 obj_ids to GT track names via nearest-anchor flood fill.

    This class takes the sparse ID mappings from GT anchor frames and propagates
    them to all frames using a nearest-anchor approach. Each frame uses the
    mapping from its closest GT anchor frame.

    Attributes:
        gt_anchors: Mapping of frame_idx -> {sam3_obj_id: track_name} at GT frames.
        fallback_names: Optional mapping of sam3_obj_id -> name for objects without
            GT matches (e.g., from initial prompt).

    Example:
        >>> resolver = TrackNameResolver.from_reconciler(reconciler)
        >>> # Get track name for a specific frame and object
        >>> name = resolver.get_track_name(frame_idx=150, sam3_obj_id=1)
        >>> # Get all mappings for batch processing
        >>> all_mappings = resolver.resolve_all_frames(total_frames=1000)
    """

    gt_anchors: dict[int, dict[int, str]] = field(default_factory=dict)
    fallback_names: dict[int, str] = field(default_factory=dict)
    _anchor_frames: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Cache sorted anchor frames for efficient lookup."""
        self._anchor_frames = sorted(self.gt_anchors.keys())

    @classmethod
    def from_reconciler(
        cls,
        reconciler: IDReconciler,
        fallback_names: dict[int, str] | None = None,
    ) -> "TrackNameResolver":
        """Create resolver from an IDReconciler with accumulated assignments.

        Args:
            reconciler: IDReconciler that has processed GT frames.
            fallback_names: Optional mapping for objects without GT matches.

        Returns:
            TrackNameResolver initialized with the reconciler's ID map.
        """
        return cls(
            gt_anchors=reconciler.build_id_map(),
            fallback_names=fallback_names or {},
        )

    @classmethod
    def from_id_map(
        cls,
        id_map: dict[int, dict[int, str]],
        fallback_names: dict[int, str] | None = None,
    ) -> "TrackNameResolver":
        """Create resolver from an existing ID map.

        Args:
            id_map: Mapping of frame_idx -> {sam3_obj_id: track_name}.
            fallback_names: Optional mapping for objects without GT matches.

        Returns:
            TrackNameResolver initialized with the ID map.
        """
        return cls(
            gt_anchors=id_map,
            fallback_names=fallback_names or {},
        )

    def _find_nearest_anchor(self, frame_idx: int) -> int | None:
        """Find the nearest GT anchor frame to a given frame.

        Args:
            frame_idx: The frame index to find nearest anchor for.

        Returns:
            The frame index of the nearest anchor, or None if no anchors exist.
        """
        if not self._anchor_frames:
            return None

        # Binary search would be faster for large anchor lists,
        # but linear is fine for typical use cases (< 100 anchors)
        return min(self._anchor_frames, key=lambda a: abs(frame_idx - a))

    def get_mapping_at_frame(self, frame_idx: int) -> dict[int, str]:
        """Get the sam3_obj_id -> track_name mapping for a frame.

        Uses the mapping from the nearest GT anchor frame.

        Args:
            frame_idx: The frame index to get mapping for.

        Returns:
            Dictionary mapping sam3_obj_id to track_name.
            Returns empty dict if no GT anchors exist.
        """
        nearest = self._find_nearest_anchor(frame_idx)
        if nearest is None:
            return {}
        return self.gt_anchors[nearest]

    def get_track_name(
        self,
        frame_idx: int,
        sam3_obj_id: int,
        default: str | None = None,
    ) -> str:
        """Get track name for a SAM3 obj_id at a given frame.

        Uses the mapping from the nearest GT anchor frame. Falls back to
        fallback_names, then to a generated name.

        Args:
            frame_idx: The frame index.
            sam3_obj_id: The SAM3 object ID.
            default: Optional default name if not found. If None, generates
                a name like "track_{sam3_obj_id}".

        Returns:
            The resolved track name.
        """
        mapping = self.get_mapping_at_frame(frame_idx)

        if sam3_obj_id in mapping:
            return mapping[sam3_obj_id]

        if sam3_obj_id in self.fallback_names:
            return self.fallback_names[sam3_obj_id]

        if default is not None:
            return default

        return f"track_{sam3_obj_id}"

    def resolve_all_frames(
        self,
        total_frames: int,
    ) -> dict[int, dict[int, str]]:
        """Get resolved mappings for all frames.

        Args:
            total_frames: Total number of frames in the video.

        Returns:
            Dictionary mapping frame_idx -> {sam3_obj_id: track_name}.
            Empty frames (no mapping) are not included in the result.
        """
        if not self._anchor_frames:
            return {}

        result: dict[int, dict[int, str]] = {}
        for frame_idx in range(total_frames):
            nearest = self._find_nearest_anchor(frame_idx)
            if nearest is not None:
                result[frame_idx] = self.gt_anchors[nearest]

        return result

    def get_anchor_frames(self) -> list[int]:
        """Get sorted list of GT anchor frame indices.

        Returns:
            List of frame indices where GT anchors exist.
        """
        return list(self._anchor_frames)

    def get_all_track_names(self) -> set[str]:
        """Get all unique track names from GT anchors.

        Returns:
            Set of all track names found in GT mappings.
        """
        names: set[str] = set()
        for mapping in self.gt_anchors.values():
            names.update(mapping.values())
        return names

    def get_all_sam3_obj_ids(self) -> set[int]:
        """Get all unique SAM3 object IDs from GT anchors.

        Returns:
            Set of all SAM3 object IDs found in GT mappings.
        """
        obj_ids: set[int] = set()
        for mapping in self.gt_anchors.values():
            obj_ids.update(mapping.keys())
        return obj_ids

    def get_canonical_mapping(self) -> dict[int, str]:
        """Get a canonical sam3_obj_id -> track_name mapping.

        Returns a single global mapping from SAM3 object IDs to track names.
        For objects that appear in multiple anchors, uses the name from the
        first anchor frame.

        This is useful for writers that need a single consistent mapping
        (like BBoxWriter and SegmentationWriter) rather than per-frame mappings.

        Returns:
            Dictionary mapping sam3_obj_id to track_name.
        """
        canonical: dict[int, str] = {}

        # Iterate anchors in frame order to get consistent "first seen" names
        for frame_idx in self._anchor_frames:
            mapping = self.gt_anchors[frame_idx]
            for obj_id, name in mapping.items():
                if obj_id not in canonical:
                    canonical[obj_id] = name

        return canonical

    def get_anchor_source(self, frame_idx: int) -> tuple[int | None, str]:
        """Get the anchor frame and propagation direction for a frame.

        Useful for debugging and visualization.

        Args:
            frame_idx: The frame index to check.

        Returns:
            Tuple of (anchor_frame_idx, direction) where direction is one of:
            - "anchor": frame_idx is a GT anchor
            - "forward": propagated forward from an earlier anchor
            - "backward": propagated backward from a later anchor
            - "none": no anchors exist
        """
        if not self._anchor_frames:
            return (None, "none")

        nearest = self._find_nearest_anchor(frame_idx)
        if nearest is None:
            return (None, "none")

        if frame_idx == nearest:
            return (nearest, "anchor")
        elif frame_idx > nearest:
            return (nearest, "forward")
        else:
            return (nearest, "backward")
