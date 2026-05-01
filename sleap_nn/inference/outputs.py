"""``Outputs`` — the structured container produced by every ``InferenceLayer``.

Replaces the dict-of-arrays that the current ``predictors.py`` returns.
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
  This guarantees the multi-process post-processing path (PR 9 / #517) and
  the streaming writer (PR 8 / #516) can ship ``Outputs`` between processes
  without surprises. Enforced by tests.
* No live references: every field is a value (tensor, ndarray, ints, the
  ``PreprocInfo`` struct). No ``InferenceLayer`` / ``Backend`` /
  ``LightningModule`` / file handle / generator. Enforced by tests.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import attrs
import numpy as np
import torch

from sleap_nn.inference.preprocess_info import PreprocInfo

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
    pred_peak_values: Optional[torch.Tensor] = None  # (B, I, N)
    pred_confmaps: Optional[torch.Tensor] = None  # (B, N, H, W) — heavy
    pred_pafs: Optional[torch.Tensor] = None  # (B, 2E, H, W) — heavy
    pred_centroids: Optional[torch.Tensor] = None  # (B, I, 2)
    pred_centroid_values: Optional[torch.Tensor] = None  # (B, I)

    # ── Instance-level metadata ──────────────────────────────────────
    instance_scores: Optional[torch.Tensor] = None  # (B, I)
    instance_valid: Optional[torch.Tensor] = None  # (B, I), bool
    instance_bboxes: Optional[torch.Tensor] = None  # (B, I, 4, 2)

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
            else:
                kwargs[f.name] = val
        return Outputs(**kwargs)

    def _map(self, fn) -> "Outputs":
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
    # sleap-io conversion (extended in PR 8 — minimal here for PR 4 use)
    # ═══════════════════════════════════════════════════════════════════

    def to_instances(
        self,
        skeleton: "Any",
        batch_index: int = 0,
    ) -> List[Any]:
        """Convert one batch slot into a list of ``sio.PredictedInstance``.

        Args:
            skeleton: ``sleap_io.Skeleton`` describing nodes/edges.
            batch_index: Which sample in the batch to convert. Defaults to 0
                (the common single-frame call site).

        Returns:
            One ``sio.PredictedInstance`` per non-NaN instance slot.

        Notes:
            Coordinates are taken verbatim from ``pred_keypoints`` —
            assumed to already be in original-image space. Per-keypoint
            scores come from ``pred_peak_values``; per-instance scores
            from ``instance_scores`` if present, else mean of node scores.
        """
        import sleap_io as sio

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

        instances: List[sio.PredictedInstance] = []
        for i in range(kpts.shape[0]):
            if np.all(np.isnan(kpts[i])):
                continue
            inst_score = (
                float(instance_scores[i])
                if instance_scores is not None
                else (
                    float(np.nanmean(vals[i])) if not np.all(np.isnan(vals[i])) else 0.0
                )
            )
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=kpts[i],
                    point_scores=vals[i],
                    score=inst_score,
                    skeleton=skeleton,
                )
            )
        return instances

    def to_labels(
        self,
        skeleton: "Any",
        videos: Optional[List[Any]] = None,
    ) -> Any:
        """Convert this ``Outputs`` to a ``sleap_io.Labels``.

        Args:
            skeleton: ``sleap_io.Skeleton`` describing nodes/edges.
            videos: List of ``sio.Video`` indexed by ``video_indices``.
                Defaults to a single ``None`` placeholder.

        Returns:
            A ``sleap_io.Labels`` containing one ``LabeledFrame`` per
            non-empty batch slot.

        Notes:
            Minimal implementation for the PR 4 single-instance proof of
            pattern. PR 8 (``Predictor`` orchestrator) extends this with
            full multi-video / per-frame metadata handling.
        """
        import sleap_io as sio

        videos = list(videos) if videos else [None]
        labeled_frames: List[sio.LabeledFrame] = []
        for b in range(self.batch_size):
            instances = self.to_instances(skeleton=skeleton, batch_index=b)
            if not instances:
                continue
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
            labeled_frames.append(
                sio.LabeledFrame(
                    video=videos[video_idx % len(videos)] if videos else None,
                    frame_idx=frame_idx,
                    instances=instances,
                )
            )
        valid_videos = [v for v in videos if v is not None]
        return sio.Labels(
            labeled_frames=labeled_frames,
            videos=valid_videos,
            skeletons=[skeleton],
        )
