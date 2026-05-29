"""``Outputs`` — the structured container produced by every ``InferenceLayer``.

Single source of truth for what an inference call yields, how to manipulate
its tensors (device, dtype, autograd), and how to reduce it to a slimmer
form for cross-process transport.

Design constraints baked into the class:

* ``slots=True`` halves per-instance memory; long videos can produce
  millions of these.
* ``eq=False`` skips ``__eq__`` machinery; we never compare two ``Outputs``
  for equality and skipping it speeds up construction.
* Custom ``__repr__`` prints field shapes, not tensor contents — a fat
  ``Outputs`` would otherwise dump megabytes into stack traces.
* ``slim()`` is a hard contract: the returned object MUST be pickleable.
  This guarantees multi-process post-processing and the streaming writer
  can ship ``Outputs`` between processes without surprises. Enforced by
  tests.
* No live references: every field is a value (tensor, ndarray, ints, the
  ``PreprocInfo`` struct). No ``InferenceLayer`` / ``Backend`` /
  ``LightningModule`` / file handle / generator. Enforced by tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import attrs
import numpy as np
import torch

from sleap_nn.inference.preprocess_info import PreprocInfo

if TYPE_CHECKING:
    import sleap_io as sio

# Heavy intermediate tensors that ``slim()`` drops — heavy as in
# ``(B, N, H, W)`` confmaps that can be hundreds of MB per frame.
_HEAVY_FIELDS: Tuple[str, ...] = (
    "original_image",
    "processed_image",
    "crops",
    "pred_confmaps",
    "pred_pafs",
    "pred_class_maps",
    "pred_paf_graph",
)


def _tensor_repr(name: str, value: Any) -> Optional[str]:
    """Compact field representation: shape + dtype, no tensor contents."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return f"{name}=Tensor{tuple(value.shape)}[{value.dtype}]"
    if isinstance(value, np.ndarray):
        return f"{name}=ndarray{tuple(value.shape)}[{value.dtype}]"
    if isinstance(value, tuple) and value and isinstance(value[0], torch.Tensor):
        shapes = ",".join(str(tuple(t.shape)) for t in value)
        return f"{name}=Tuple<Tensor>[{shapes}]"
    return f"{name}={value!r}"


@attrs.define(slots=True, eq=False, repr=False)
class Outputs:
    """Structured container for inference outputs.

    Shape convention:
        ``B`` = batch size, ``I`` = max instances, ``N`` = nodes,
        ``C`` = classes, ``H``/``W`` = spatial dims, ``E`` = number of edges.
        ``NaN`` indicates missing/invalid predictions in keypoint fields.
    """

    # ── Images (optional; None unless explicitly requested) ──────────
    original_image: Optional[torch.Tensor] = None  # (B, C, H, W)
    processed_image: Optional[torch.Tensor] = None  # (B, C, H', W')
    crops: Optional[torch.Tensor] = None  # (B, I, C, cH, cW); top-down only

    # ── Core predictions ─────────────────────────────────────────────
    pred_keypoints: Optional[torch.Tensor] = None  # (B, I, N, 2) in image (x, y)
    pred_crop_keypoints: Optional[torch.Tensor] = None  # (B, I, N, 2) crop-local
    pred_peak_values: Optional[torch.Tensor] = None  # (B, I, N)
    pred_confmaps: Optional[torch.Tensor] = None  # (B, N, H, W) — heavy
    pred_pafs: Optional[torch.Tensor] = None  # (B, 2E, H, W) — heavy
    pred_centroids: Optional[torch.Tensor] = None  # (B, I, 2)
    pred_centroid_values: Optional[torch.Tensor] = None  # (B, I)

    # ── Instance-level metadata ──────────────────────────────────────
    instance_scores: Optional[torch.Tensor] = None  # (B, I)
    instance_valid: Optional[torch.Tensor] = None  # (B, I), bool
    instance_bboxes: Optional[torch.Tensor] = None  # (B, I, 4, 2)
    # Per-instance tracking score, separate from ``score``. Multi-class
    # models carry the class probability here (legacy ``tracking_score``)
    # while ``instance_scores`` holds the legacy base score (centroid /
    # mean-confidence). ``None`` for non-multiclass paths.
    instance_tracking_scores: Optional[torch.Tensor] = None  # (B, I)

    # ── Multi-class predictions ──────────────────────────────────────
    pred_class_vectors: Optional[torch.Tensor] = None  # (B, I, N, C)
    pred_class_maps: Optional[torch.Tensor] = None  # (B, C, H, W) — heavy
    pred_class_inds: Optional[torch.Tensor] = None  # (B, I, N)
    pred_class_probs: Optional[torch.Tensor] = None  # (B, I, C)

    # ── PAF graph (bottom-up intermediate; opt-in) ───────────────────
    # Tuple of (peaks, edge_inds, edge_peak_inds, line_scores).
    pred_paf_graph: Optional[Tuple[torch.Tensor, ...]] = None

    # ── Preprocessing metadata (for coord reversal) ──────────────────
    preprocess_info: Optional[PreprocInfo] = None

    # ── Frame/video metadata ─────────────────────────────────────────
    frame_indices: Optional[torch.Tensor] = None  # (B,), int64
    video_indices: Optional[torch.Tensor] = None  # (B,), int64

    # ═══════════════════════════════════════════════════════════════════
    # Repr (compact; never prints tensor contents)
    # ═══════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        """Compact ``Outputs(...)`` summary listing only populated fields."""
        parts: List[str] = []
        for f in attrs.fields(type(self)):
            r = _tensor_repr(f.name, getattr(self, f.name))
            if r is not None:
                parts.append(r)
        if not parts:
            return "Outputs(empty)"
        return f"Outputs({', '.join(parts)})"

    # ═══════════════════════════════════════════════════════════════════
    # Tensor management — device / dtype / autograd
    # ═══════════════════════════════════════════════════════════════════

    def to(self, device: Union[str, torch.device]) -> "Outputs":
        """Return a new ``Outputs`` with all tensor fields moved to ``device``."""
        return self._map(lambda t: t.to(device))

    def cpu(self) -> "Outputs":
        """Return a new ``Outputs`` with all tensors on CPU."""
        return self.to("cpu")

    def detach(self) -> "Outputs":
        """Return a new ``Outputs`` with autograd detached on every tensor."""
        return self._map(lambda t: t.detach())

    def numpy(self) -> Dict[str, Any]:
        """Return non-``None`` fields as numpy.

        Tensors become ``np.ndarray`` (CPU + detached automatically).
        Tuples-of-tensors become tuples-of-ndarrays. ``None`` fields are
        omitted. Non-tensor fields (``preprocess_info``, integers) pass
        through untouched.
        """
        out: Dict[str, Any] = {}
        for f in attrs.fields(type(self)):
            val = getattr(self, f.name)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                out[f.name] = val.detach().cpu().numpy()
            elif isinstance(val, tuple) and val and isinstance(val[0], torch.Tensor):
                out[f.name] = tuple(t.detach().cpu().numpy() for t in val)
            elif isinstance(val, PreprocInfo):
                # Keep as a PreprocInfo (callers rely on this) but move its
                # nested tensors to CPU (#584).
                out[f.name] = val.cpu()
            else:
                out[f.name] = val
        return out

    def slim(self) -> "Outputs":
        """Drop heavy intermediates and force CPU + detach for transport.

        Hard contract: the returned ``Outputs`` is guaranteed pickle-safe.
        Use this before sending across a queue / process boundary
        (``multiprocessing.Queue``, ``concurrent.futures``).

        Drops: ``original_image``, ``processed_image``, ``crops``,
        ``pred_confmaps``, ``pred_pafs``, ``pred_class_maps``,
        ``pred_paf_graph``. These are opt-in heavies; if you needed them
        downstream, call this after the consumer is done with them.
        """
        kwargs: Dict[str, Any] = {}
        for f in attrs.fields(type(self)):
            if f.name in _HEAVY_FIELDS:
                kwargs[f.name] = None
                continue
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                kwargs[f.name] = val.detach().cpu()
            elif isinstance(val, tuple) and val and isinstance(val[0], torch.Tensor):
                kwargs[f.name] = tuple(t.detach().cpu() for t in val)
            elif isinstance(val, PreprocInfo):
                # Move the nested eff_scale / crop_offsets to CPU so the slimmed
                # Outputs is genuinely pickle-safe for spawn workers (#584).
                kwargs[f.name] = val.cpu()
            else:
                kwargs[f.name] = val
        return Outputs(**kwargs)

    def _map(self, fn: "Callable[[torch.Tensor], torch.Tensor]") -> "Outputs":
        """Apply ``fn`` to every tensor field, returning a new ``Outputs``.

        Tuples-of-tensors are mapped element-wise. Non-tensor fields pass
        through unchanged.
        """
        kwargs: Dict[str, Any] = {}
        for f in attrs.fields(type(self)):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                kwargs[f.name] = fn(val)
            elif isinstance(val, tuple) and val and isinstance(val[0], torch.Tensor):
                kwargs[f.name] = tuple(fn(t) for t in val)
            elif isinstance(val, PreprocInfo):
                # Apply the same map to the nested tensors so .to(device)/.cpu()/
                # .detach() carry PreprocInfo along (#584).
                kwargs[f.name] = attrs.evolve(
                    val,
                    eff_scale=fn(val.eff_scale),
                    crop_offsets=(
                        fn(val.crop_offsets) if val.crop_offsets is not None else None
                    ),
                )
            else:
                kwargs[f.name] = val
        return Outputs(**kwargs)

    # ═══════════════════════════════════════════════════════════════════
    # Shape properties
    # ═══════════════════════════════════════════════════════════════════

    @property
    def batch_size(self) -> int:
        """Batch dimension B, or 0 if no keypoint-bearing field is set."""
        for name in ("pred_keypoints", "pred_centroids", "frame_indices"):
            t = getattr(self, name)
            if t is not None:
                return int(t.shape[0])
        return 0

    @property
    def n_instances(self) -> int:
        """Per-frame instance dimension I, or 0 if no instance field is set."""
        for name in ("pred_keypoints", "pred_centroids", "instance_scores"):
            t = getattr(self, name)
            if t is None:
                continue
            return int(t.shape[1]) if t.ndim >= 2 else 0
        return 0

    @property
    def n_nodes(self) -> int:
        """Skeleton-node dimension N, or 0 if not derivable."""
        if self.pred_keypoints is not None and self.pred_keypoints.ndim >= 3:
            return int(self.pred_keypoints.shape[2])
        if self.pred_peak_values is not None and self.pred_peak_values.ndim >= 3:
            return int(self.pred_peak_values.shape[2])
        return 0

    # ═══════════════════════════════════════════════════════════════════
    # sleap-io conversion
    # ═══════════════════════════════════════════════════════════════════

    def to_instances(
        self,
        skeleton: "sio.Skeleton",
        batch_index: int = 0,
        anchor_ind: Optional[int] = None,
        tracks: Optional[list["sio.Track"]] = None,
    ) -> list["sio.PredictedInstance"]:
        """Convert one batch slot into a list of ``sio.PredictedInstance``.

        Args:
            skeleton: ``sleap_io.Skeleton`` describing nodes/edges.
            batch_index: Which sample in the batch to convert. Defaults to 0
                (the common single-frame call site).
            anchor_ind: Centroid-only packaging — when ``pred_keypoints`` is
                None but ``pred_centroids`` is populated, this index decides
                which skeleton-node slot receives the centroid coordinate
                (all other slots are NaN). ``None`` defaults to node 0.
                Ignored when ``pred_keypoints`` is populated.
            tracks: Multi-class identity packaging — a list of ``sio.Track``
                indexed by class. When provided, each instance is assigned
                ``tracks[class_ind]`` (top-down multi-class, where the class
                index is carried per-instance in ``pred_class_inds``) or
                ``tracks[i]`` (bottom-up multi-class, where the instance slot
                ``i`` *is* the class), and ``tracking_score`` is read from
                ``instance_tracking_scores``. Matches legacy
                ``TopDownMultiClass`` / ``BottomUpMultiClass`` packaging.
                ``None`` for non-multiclass paths.

        Returns:
            One ``sio.PredictedInstance`` per non-NaN instance slot.

        Notes:
            Coordinates are taken verbatim from ``pred_keypoints`` —
            assumed to already be in original-image space. Per-keypoint
            scores come from ``pred_peak_values``; per-instance scores
            from ``instance_scores`` if present, else the SUM of node scores
            (``np.nansum``), matching legacy ``SingleInstancePredictor``.

            **Centroid-only mode** (``pred_keypoints is None`` and
            ``pred_centroids is not None``): packages each predicted
            centroid into a ``PredictedInstance`` with the centroid
            coordinate at ``anchor_ind`` (or node 0 if unset) and NaN at
            every other node. Per-instance score = centroid value.
        """
        import sleap_io as sio

        # Centroid-only branch: synthesize NaN-padded keypoints from centroids.
        if self.pred_keypoints is None and self.pred_centroids is not None:
            return self._to_instances_centroid_only(
                skeleton=skeleton,
                batch_index=batch_index,
                anchor_ind=anchor_ind if anchor_ind is not None else 0,
            )

        if self.pred_keypoints is None:
            return []

        kpts = self.pred_keypoints[batch_index].detach().cpu().numpy()  # (I, N, 2)
        vals = (
            self.pred_peak_values[batch_index].detach().cpu().numpy()
            if self.pred_peak_values is not None
            else np.full(kpts.shape[:2], np.nan, dtype=np.float32)
        )
        instance_scores = (
            self.instance_scores[batch_index].detach().cpu().numpy()
            if self.instance_scores is not None
            else None
        )
        # Multi-class identity packaging metadata (None for plain paths).
        tracking_scores = (
            self.instance_tracking_scores[batch_index].detach().cpu().numpy()
            if self.instance_tracking_scores is not None
            else None
        )
        # Per-instance class index for top-down multi-class. ``pred_class_inds``
        # is ``(B, I, N)`` (the same class for every node of an instance), so
        # node 0 carries the per-instance class. Bottom-up multi-class does not
        # set this — there the instance slot ``i`` IS the class (see below).
        class_inds = (
            self.pred_class_inds[batch_index].detach().cpu().numpy()
            if self.pred_class_inds is not None
            else None
        )

        instances: List[sio.PredictedInstance] = []
        for i in range(kpts.shape[0]):
            if np.all(np.isnan(kpts[i])):
                continue
            # Per-instance score fallback (single-instance models, which don't
            # populate ``instance_scores``) must match legacy
            # ``SingleInstancePredictor``: ``np.nansum(pred_values)`` — the SUM
            # of node confidences, NOT the mean (#530 audit F-SCORE /
            # predictors.py:1937). ``np.nansum`` of an all-NaN row is 0.0.
            inst_score = (
                float(instance_scores[i])
                if instance_scores is not None
                else float(np.nansum(vals[i]))
            )

            # Multi-class track + tracking_score assignment. Legacy parity:
            #   - TopDownMultiClass (predictors.py:3808-3880): track =
            #     tracks[class_ind]; tracking_score = class probability;
            #     score = centroid value.
            #   - BottomUpMultiClass (predictors.py:2987-3010): track =
            #     tracks[i] (by instance order); tracking_score =
            #     mean class score; score = mean confidence.
            track = None
            tracking_score = None
            if tracks is not None:
                # Top-down carries an explicit per-instance class index; bottom-up
                # uses the instance slot ``i`` directly as the class (legacy
                # ``tracks[i]``).
                cls_ind = int(class_inds[i, 0]) if class_inds is not None else i
                if 0 <= cls_ind < len(tracks):
                    track = tracks[cls_ind]
                if tracking_scores is not None:
                    tracking_score = float(tracking_scores[i])

            kwargs: Dict[str, Any] = {}
            if track is not None:
                kwargs["track"] = track
            if tracking_score is not None:
                kwargs["tracking_score"] = tracking_score
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=kpts[i],
                    point_scores=vals[i],
                    score=inst_score,
                    skeleton=skeleton,
                    **kwargs,
                )
            )
        return instances

    def _to_instances_centroid_only(
        self,
        skeleton: "sio.Skeleton",
        batch_index: int,
        anchor_ind: int,
    ) -> list["sio.PredictedInstance"]:
        """Centroid-only packaging: NaN-pad skeleton, centroid at ``anchor_ind``.

        See :meth:`to_instances` for semantics.
        """
        import sleap_io as sio

        centroids = self.pred_centroids[batch_index].detach().cpu().numpy()  # (I, 2)
        cvals = (
            self.pred_centroid_values[batch_index].detach().cpu().numpy()
            if self.pred_centroid_values is not None
            else np.full((centroids.shape[0],), np.nan, dtype=np.float32)
        )

        n_nodes = len(skeleton.nodes)
        if not 0 <= anchor_ind < n_nodes:
            raise ValueError(
                f"anchor_ind={anchor_ind} is out of range for skeleton with "
                f"{n_nodes} nodes."
            )

        instances: List[sio.PredictedInstance] = []
        for i in range(centroids.shape[0]):
            if np.all(np.isnan(centroids[i])):
                continue
            kpts = np.full((n_nodes, 2), np.nan, dtype=np.float32)
            kpts[anchor_ind] = centroids[i]
            point_scores = np.full((n_nodes,), np.nan, dtype=np.float32)
            point_scores[anchor_ind] = float(cvals[i])
            inst_score = float(cvals[i]) if not np.isnan(cvals[i]) else 0.0
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=kpts,
                    point_scores=point_scores,
                    score=inst_score,
                    skeleton=skeleton,
                )
            )
        return instances

    def to_labels(
        self,
        skeleton: "sio.Skeleton",
        videos: Optional[list["sio.Video"]] = None,
        anchor_ind: Optional[int] = None,
        tracks: Optional[list["sio.Track"]] = None,
    ) -> "sio.Labels":
        """Convert this ``Outputs`` to a ``sleap_io.Labels``.

        Args:
            skeleton: ``sleap_io.Skeleton`` describing nodes/edges.
            videos: List of ``sio.Video`` indexed by ``video_indices``.
                Defaults to a single ``None`` placeholder.
            anchor_ind: Forwarded to :meth:`to_instances` for centroid-only
                packaging. Ignored when ``pred_keypoints`` is populated.
            tracks: Multi-class identity ``sio.Track`` registry indexed by
                class. Forwarded to :meth:`to_instances`; the tracks that get
                used are registered on the returned ``sio.Labels.tracks``.
                ``None`` for non-multiclass paths.

        Returns:
            A ``sleap_io.Labels`` containing one ``LabeledFrame`` per
            non-empty batch slot.

        Notes:
            For full multi-video / per-frame metadata handling, use
            :meth:`Predictor.predict` which aggregates per-batch
            ``Outputs`` into a single ``sio.Labels``.
        """
        import sleap_io as sio

        videos = list(videos) if videos else [None]
        labeled_frames: List[sio.LabeledFrame] = []
        used_tracks: List["sio.Track"] = []
        seen_track_ids: set[int] = set()
        for b in range(self.batch_size):
            instances = self.to_instances(
                skeleton=skeleton, batch_index=b, anchor_ind=anchor_ind, tracks=tracks
            )
            if not instances:
                continue
            for inst in instances:
                trk = getattr(inst, "track", None)
                if trk is not None and id(trk) not in seen_track_ids:
                    seen_track_ids.add(id(trk))
                    used_tracks.append(trk)
            frame_idx = (
                int(self.frame_indices[b].item())
                if self.frame_indices is not None
                else b
            )
            video_idx = (
                int(self.video_indices[b].item())
                if self.video_indices is not None
                else 0
            )
            # Map the per-frame video index to its Video. For genuine multi-video
            # output, an out-of-range index is a provider/packaging mismatch and
            # must be loud rather than silently wrapping onto the wrong video
            # (the old `% len(videos)` masked exactly that bug). The single-video
            # / placeholder case stays lenient (#582).
            if video_idx < len(videos):
                video = videos[video_idx]
            elif len(videos) == 1:
                video = videos[0]
            else:
                raise IndexError(
                    f"video_index {video_idx} is out of range for {len(videos)} "
                    "videos; the provider emitted a video index with no matching "
                    "video."
                )
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=frame_idx,
                    instances=instances,
                )
            )
        valid_videos = [v for v in videos if v is not None]
        labels = sio.Labels(
            labeled_frames=labeled_frames,
            videos=valid_videos,
            skeletons=[skeleton],
        )
        if used_tracks:
            labels.tracks = used_tracks
        return labels
