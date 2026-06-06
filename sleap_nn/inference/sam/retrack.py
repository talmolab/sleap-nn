"""Mask-based re-tracking: refine existing pose/centroid track identities.

This module provides :func:`retrack`, a thin orchestration over the lifted
:mod:`sleap_nn.inference.sam.reconciliation` primitives. It implements the
**"refine existing tracks"** path of the SAM segmentation stack: given a set of
labeled frames whose instances already carry (possibly *wrong*) track labels,
plus identity-consistent per-frame masks (e.g. from a SAM3 video tracker), it
re-derives a clean, swap-free track assignment for every instance.

The recipe (re-derived from ``talmolab/sam-track``'s CLI glue and validated by
prototype P3) is:

1. **Match.** At each frame, Hungarian-match poses to masks by keypoints-inside
   (:class:`~sleap_nn.inference.sam.reconciliation.IDReconciler`), gated by a
   match predicate (the default requires >=1 keypoint inside; a stricter
   ``require_min_keypoints_inside(3)`` is recommended for re-tracking).
2. **Build anchors.** Treat *trusted* frames (by default the frames whose
   instances carry GT/user track labels, else all frames) as identity anchors,
   yielding a sparse ``frame -> {mask_obj_id: track_name}`` map.
3. **Canonical remap.** Name each ``mask_obj_id`` by majority vote across anchor
   frames (strict majority wins); obj_ids with no clear majority (an exact tie)
   are omitted from the canonical map and resolved per-frame via the nearest
   anchor.
4. **Relabel all frames.** For every frame, reassign each matched instance's
   ``track`` to the resolved name (creating :class:`sio.Track` objects as
   needed). Instances with no mask match keep their original track.

This is the *post-processing* identity-correction path. Per the experimental
finding baked into the reconciliation module's docstring, SAM3 mid-propagation
re-prompting adds *new* objects but does not fix *existing* tracks, so identity
correction must happen here, after tracking, via mask<->pose matching.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from sleap_nn.inference.sam.reconciliation import (
    IDReconciler,
    MatchPredicate,
    TrackAssignment,
    TrackNameResolver,
)

if TYPE_CHECKING:
    import sleap_io as sio


@dataclass
class RetrackResult:
    """Result of a :func:`retrack` run.

    Attributes:
        labeled_frames: The relabeled frames. Same objects as the input when
            ``in_place=True``; the corrected deep copies when ``in_place=False``.
        assignments: All :class:`TrackAssignment` objects produced across frames
            (pose<->mask Hungarian matches, after predicate filtering).
        id_map: Sparse anchor map ``frame_idx -> {mask_obj_id: track_name}``
            built from trusted (anchor) frames only.
        canonical_map: The global ``mask_obj_id -> track_name`` map used to
            relabel instances. Each obj_id is named by majority vote across
            anchor frames; obj_ids with no clear majority (an exact tie) are
            omitted here and resolved per-frame via the nearest anchor.
        resolver: The :class:`TrackNameResolver` used for nearest-anchor
            fallback, exposed for inspection / debugging.
        num_relabeled: Number of instances whose ``track`` was changed.
        num_matched: Number of instances that received a mask match.
        anchor_frames: Sorted frame indices used as identity anchors.
    """

    labeled_frames: list["sio.LabeledFrame"] = field(default_factory=list)
    assignments: list[TrackAssignment] = field(default_factory=list)
    id_map: dict[int, dict[int, str]] = field(default_factory=dict)
    canonical_map: dict[int, str] = field(default_factory=dict)
    resolver: TrackNameResolver | None = None
    num_relabeled: int = 0
    num_matched: int = 0
    anchor_frames: list[int] = field(default_factory=list)


def _is_anchor_instance(inst: "sio.Instance") -> bool:
    """Return ``True`` if an instance is a trusted (GT/user) identity anchor.

    Uses the GT-precedence rule from the SAM stack: a *user* instance is exactly
    ``sio.Instance`` (predicted instances are the ``sio.PredictedInstance``
    subclass), so ``type(inst) is sio.Instance`` distinguishes hand-labeled
    anchors from model predictions. An anchor must additionally carry a track.

    Args:
        inst: The instance to test.

    Returns:
        ``True`` if the instance is a user-labeled, tracked anchor.
    """
    import sleap_io as sio

    return type(inst) is sio.Instance and inst.track is not None


def _frame_has_anchor(frame: "sio.LabeledFrame") -> bool:
    """Return ``True`` if a frame contains at least one anchor instance."""
    return any(_is_anchor_instance(inst) for inst in frame.instances)


def _masks_for_frame(
    masks: np.ndarray,
    object_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop padded entries and squeeze ``(N, 1, H, W)`` to ``(N, H, W)``.

    SAM3 batches commonly pad per-frame mask/obj_id arrays to a fixed ``N`` with
    a sentinel ``object_id < 0``. This strips those padded rows and normalizes
    the mask rank.

    Args:
        masks: Masks for one frame, shape ``(N, H, W)`` or ``(N, 1, H, W)``.
        object_ids: Object IDs for one frame, shape ``(N,)``; entries ``< 0``
            are treated as padding and removed.

    Returns:
        Tuple ``(masks, object_ids)`` with padding removed and masks squeezed.
    """
    object_ids = np.asarray(object_ids)
    masks = np.asarray(masks)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(axis=1)

    if object_ids.size and np.any(object_ids < 0):
        keep = object_ids >= 0
        masks = masks[keep]
        object_ids = object_ids[keep]

    return masks, object_ids


def retrack(
    labeled_frames: Sequence["sio.LabeledFrame"],
    masks: Sequence[np.ndarray],
    object_ids: Sequence[np.ndarray],
    skeleton: "sio.Skeleton",
    *,
    scores: Sequence[np.ndarray] | None = None,
    match_predicates: list[MatchPredicate] | None = None,
    exclude_nodes: set[str] | None = None,
    anchor_frame_indices: Sequence[int] | None = None,
    fallback_names: dict[int, str] | None = None,
    in_place: bool = True,
) -> RetrackResult:
    """Refine instance track identities from per-frame masks.

    Re-derives track assignments for ``labeled_frames`` by matching each frame's
    pose instances to its masks (Hungarian on keypoints-inside), anchoring the
    ``mask_obj_id -> track_name`` mapping on trusted frames, and relabeling every
    instance to the canonical / nearest-anchor name implied by its matched mask.

    The mask sequences are aligned to ``labeled_frames`` **by position**:
    ``masks[i]`` / ``object_ids[i]`` describe the same frame as
    ``labeled_frames[i]``. Frames whose instances carry GT/user tracks act as
    identity anchors by default (override via ``anchor_frame_indices``).

    Args:
        labeled_frames: Frames to re-track, in the same order as ``masks``. Each
            instance's ``.track`` may be wrong; it will be corrected in place
            (unless ``in_place=False``, in which case a deep copy is corrected
            and the originals are left untouched).
        masks: Per-frame masks; ``masks[i]`` has shape ``(N_i, H, W)`` or
            ``(N_i, 1, H, W)``. Entries are paired with ``object_ids[i]``.
        object_ids: Per-frame object IDs; ``object_ids[i]`` has shape ``(N_i,)``.
            Entries ``< 0`` are treated as padding and dropped.
        skeleton: Skeleton for node-name lookups during pose<->mask matching.
        scores: Optional per-frame mask confidence scores, ``scores[i]`` shape
            ``(N_i,)``. Defaults to all-ones when omitted.
        match_predicates: Predicates (all must pass) gating a valid match. When
            ``None``, :class:`IDReconciler`'s default (>=1 keypoint inside) is
            used. ``require_min_keypoints_inside(3)`` is recommended.
        exclude_nodes: Node names to ignore when counting keypoints-inside
            (e.g. unreliable tail nodes).
        anchor_frame_indices: Explicit positions into ``labeled_frames`` to use
            as identity anchors. When ``None``, frames containing a GT/user
            tracked instance are used; if there are none, every frame is an
            anchor.
        fallback_names: Optional ``mask_obj_id -> track_name`` map used for
            obj_ids that never appear in any anchor frame.
        in_place: When ``True`` (default), mutate the instances of
            ``labeled_frames`` directly. When ``False``, deep copy the frames,
            correct the copies, and leave the inputs untouched. Either way the
            (corrected) frames are returned on ``RetrackResult.labeled_frames``.

    Returns:
        A :class:`RetrackResult` with the assignments, the sparse and canonical
        ID maps, the resolver, and relabel/match counts.

    Example:
        >>> from sleap_nn.inference.sam.reconciliation import (
        ...     require_min_keypoints_inside,
        ... )
        >>> result = retrack(
        ...     labeled_frames=lfs,
        ...     masks=[m for m in per_frame_masks],
        ...     object_ids=[o for o in per_frame_obj_ids],
        ...     skeleton=labels.skeleton,
        ...     match_predicates=[require_min_keypoints_inside(3)],
        ... )
        >>> result.num_relabeled
        12
    """
    import sleap_io as sio

    n_frames = len(labeled_frames)
    if len(masks) != n_frames or len(object_ids) != n_frames:
        raise ValueError(
            "labeled_frames, masks, and object_ids must be the same length "
            f"(got {n_frames}, {len(masks)}, {len(object_ids)})."
        )
    if scores is not None and len(scores) != n_frames:
        raise ValueError(
            "scores, when provided, must match labeled_frames in length "
            f"(got {len(scores)} vs {n_frames})."
        )

    frames: list["sio.LabeledFrame"] = list(labeled_frames)
    if not in_place:
        from copy import deepcopy

        # Copy the whole list in one deepcopy so shared references (e.g. the
        # same ``Track`` object across frames) stay shared in the copy.
        frames = deepcopy(frames)

    # --- 1. determine anchor frames ---------------------------------------
    if anchor_frame_indices is not None:
        anchor_set = {int(i) for i in anchor_frame_indices}
    else:
        anchor_set = {i for i, lf in enumerate(frames) if _frame_has_anchor(lf)}
        if not anchor_set:
            # No trusted tracks anywhere: every frame anchors itself.
            anchor_set = set(range(n_frames))

    # --- 2. Hungarian match poses<->masks at every frame ------------------
    reconciler = IDReconciler(
        skeleton=skeleton,
        exclude_nodes=set(exclude_nodes) if exclude_nodes else set(),
        match_predicates=list(match_predicates) if match_predicates else [],
    )

    # frame position -> {pose_idx: mask_obj_id} for relabeling.
    per_frame_pose_to_obj: list[dict[int, int]] = []
    for i, lf in enumerate(frames):
        m_i, o_i = _masks_for_frame(masks[i], object_ids[i])
        s_i = None
        if scores is not None:
            s_i = np.asarray(scores[i])
            if s_i.size and o_i.size and len(s_i) != len(o_i):
                # Re-derive the keep mask to align scores with dropped padding.
                raw_ids = np.asarray(object_ids[i])
                if raw_ids.size and np.any(raw_ids < 0):
                    s_i = s_i[raw_ids >= 0]
        assignments = reconciler.match_frame(
            frame_idx=i,
            poses=list(lf.instances),
            masks=m_i,
            object_ids=o_i,
            scores=s_i,
        )
        per_frame_pose_to_obj.append({a.pose_idx: a.sam3_obj_id for a in assignments})

    # --- 3. build the sparse anchor id map + canonical / fallback resolver -
    # Re-run the matcher's id-map logic but restricted to anchor frames: only
    # trusted frames define the obj_id -> track_name identity.
    id_map: dict[int, dict[int, str]] = {}
    for a in reconciler.get_assignments():
        if a.frame_idx not in anchor_set:
            continue
        if a.pose_track_name:
            id_map.setdefault(a.frame_idx, {})[a.sam3_obj_id] = a.pose_track_name

    resolver = TrackNameResolver.from_id_map(
        id_map, fallback_names=fallback_names or {}
    )

    # The mask obj_id is the trustworthy identity signal; the anchor map only
    # *names* it. Name each obj_id by MAJORITY VOTE across anchor frames, so a
    # minority of wrong/swapped anchor frames cannot flip a stable identity
    # (this is what self-corrects the common "every frame labeled, a few
    # swapped" case). Ties / no-clear-majority obj_ids are genuine cross-anchor
    # reassignments (the sparse-GT SAM3 case) and are routed per-frame via the
    # nearest anchor instead.
    from collections import Counter

    votes_per_obj: dict[int, Counter] = {}
    for mapping in id_map.values():
        for obj_id, name in mapping.items():
            votes_per_obj.setdefault(obj_id, Counter())[name] += 1

    canonical_map: dict[int, str] = {}
    ambiguous_obj_ids: set[int] = set()
    for obj_id, counter in votes_per_obj.items():
        ranked = counter.most_common()
        top_name, top_count = ranked[0]
        # A clear majority (strictly more votes than the runner-up) names the
        # obj_id globally; an exact tie is ambiguous -> nearest-anchor per frame.
        if len(ranked) == 1 or top_count > ranked[1][1]:
            canonical_map[obj_id] = top_name
        else:
            ambiguous_obj_ids.add(obj_id)

    # --- 4. relabel every instance to its resolved track name -------------
    track_by_name: dict[str, "sio.Track"] = {}
    # Seed with existing track objects so we reuse identities where possible.
    for lf in frames:
        for inst in lf.instances:
            if inst.track is not None:
                track_by_name.setdefault(inst.track.name, inst.track)

    def _resolve_name(frame_idx: int, obj_id: int) -> str | None:
        # Genuine cross-anchor identity changes route through the nearest anchor
        # (so a real SAM3 swap between two sparse GT frames flips at the
        # midpoint). Stable obj_ids take the canonical majority-vote name globally,
        # immune to a single swapped/wrong anchor frame. Fallback last.
        if obj_id in ambiguous_obj_ids:
            mapping = resolver.get_mapping_at_frame(frame_idx)
            if obj_id in mapping:
                return mapping[obj_id]
        if obj_id in canonical_map:
            return canonical_map[obj_id]
        if fallback_names and obj_id in fallback_names:
            return fallback_names[obj_id]
        return None

    num_relabeled = 0
    num_matched = 0
    for i, lf in enumerate(frames):
        pose_to_obj = per_frame_pose_to_obj[i]
        for pose_idx, inst in enumerate(lf.instances):
            obj_id = pose_to_obj.get(pose_idx)
            if obj_id is None:
                continue
            num_matched += 1
            name = _resolve_name(i, obj_id)
            if name is None:
                continue
            track = track_by_name.get(name)
            if track is None:
                track = sio.Track(name=name)
                track_by_name[name] = track
            if inst.track is not track:
                inst.track = track
                num_relabeled += 1

    return RetrackResult(
        labeled_frames=frames,
        assignments=reconciler.get_assignments(),
        id_map=id_map,
        canonical_map=canonical_map,
        resolver=resolver,
        num_relabeled=num_relabeled,
        num_matched=num_matched,
        anchor_frames=sorted(anchor_set),
    )


# Public API of this module.
__all__: list[str] = [
    "RetrackResult",
    "retrack",
]
