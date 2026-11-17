"""Frame in, pose out: a pose stream for live cameras.

:func:`sleap_nn.inference.run.open_stream` builds the same pipeline as
:func:`sleap_nn.inference.run.predict` (detector -> pose -> filters -> streaming part
stage, from checkpoints or network-only export folders) and returns a
:class:`PoseStream`. Each :meth:`PoseStream.push` runs one frame through it and returns
that frame's instances as plain arrays (:class:`FramePoses`); nothing waits for the
next frame.

A video pushed frame by frame gives the same points as ``sleap-nn predict
--batch_size 1 --part_streaming`` with the same arguments: the stream runs predict's
per-batch path with a batch of one frame.

What is and is not real time: ``push`` returns the frame's poses as soon as they are
computed (no queue, no batching, no look-ahead), so its latency is one frame's
processing time. The stream does not drop frames, run threads or pace itself: if
frames arrive faster than ``push`` returns, the caller's camera loop decides what to
skip. Whether a set-up keeps up with a given camera depends on the models, the
settings and the hardware; time it on the target machine (``last_latency_ms``).

Single caller, single thread: a stream keeps per-video state (the part stage's
running crop floor, the frame counter) and must not be pushed from several threads.
"""

from __future__ import annotations

import copy
import time
from typing import Any, List, Optional, Tuple

import attrs
import numpy as np
import torch


@attrs.define
class FramePoses:
    """One frame's instances as plain arrays (``I`` instances, ``N`` nodes).

    Attributes:
        frame_idx: The frame's index in the stream (or the one given to ``push``).
        node_names: The ``N`` node names: the pose model's nodes, then the part
            model's nodes (with its prefix) when a part stage runs.
        points: ``(I, N, 2)`` float32 ``(x, y)`` in frame pixels; NaN = not found.
        point_scores: ``(I, N)`` float32 peak values.
        instance_scores: ``(I,)`` float32 (as written to ``.slp`` by ``predict``).
        boxes: ``(I, 4)`` float32 ``(x1, y1, x2, y2)`` of the detection each
            instance came from (NaN when the stage 1 reports no boxes).
        box_scores: ``(I,)`` float32 detection scores (NaN without boxes).
    """

    frame_idx: int
    node_names: List[str]
    points: np.ndarray
    point_scores: np.ndarray
    instance_scores: np.ndarray
    boxes: np.ndarray
    box_scores: np.ndarray

    def __len__(self) -> int:
        """Number of instances."""
        return int(self.points.shape[0])


def frame_poses_from_outputs(
    outputs: Any, frame_idx: int, node_names: List[str]
) -> FramePoses:
    """``FramePoses`` of batch slot 0, keeping the instances ``Outputs.to_labels`` keeps."""
    n = len(node_names)
    if outputs.pred_keypoints is None:
        e = np.zeros((0,), dtype=np.float32)
        return FramePoses(
            frame_idx,
            list(node_names),
            np.zeros((0, n, 2), np.float32),
            np.zeros((0, n), np.float32),
            e,
            np.zeros((0, 4), np.float32),
            e.copy(),
        )
    kpts = outputs.pred_keypoints[0].detach().float().cpu().numpy()
    vals = (
        outputs.pred_peak_values[0].detach().float().cpu().numpy()
        if outputs.pred_peak_values is not None
        else np.full(kpts.shape[:2], np.nan, dtype=np.float32)
    )
    keep = [i for i in range(kpts.shape[0]) if not np.all(np.isnan(kpts[i]))]
    if outputs.instance_scores is not None:
        inst = outputs.instance_scores[0].detach().float().cpu().numpy()[keep]
    else:
        inst = np.nansum(vals[keep], axis=-1) if keep else np.zeros((0,), np.float32)
    boxes = np.full((len(keep), 4), np.nan, dtype=np.float32)
    box_scores = np.full((len(keep),), np.nan, dtype=np.float32)
    if (
        outputs.pred_bboxes is not None
        and outputs.pred_bboxes.shape[1] == kpts.shape[0]
    ):
        b = outputs.pred_bboxes[0].detach().float().cpu().numpy()
        s = (
            outputs.pred_bbox_scores[0].detach().float().cpu().numpy()
            if outputs.pred_bbox_scores is not None
            else np.ones((b.shape[0],), np.float32)
        )
        boxes[:] = b[keep]
        box_scores[:] = s[keep]
    return FramePoses(
        frame_idx=int(frame_idx),
        node_names=list(node_names),
        points=kpts[keep].astype(np.float32),
        point_scores=vals[keep].astype(np.float32),
        instance_scores=np.asarray(inst, dtype=np.float32),
        boxes=boxes,
        box_scores=box_scores,
    )


class PoseStream:
    """Push one frame, get its poses. Build it with :func:`sleap_nn.inference.run.open_stream`.

    Args:
        predictor: A built :class:`~sleap_nn.inference.predictor.Predictor` (with
            its ``part_stage`` when a part model runs).
        layer: The predictor's layer with the prediction-time overrides applied
            (``Predictor._scoped_postprocess_layer``), as ``predict`` uses it.
        channel_order: ``"rgb"`` (default; what sleap-io's video readers return)
            or ``"bgr"`` (what OpenCV returns): a 3-channel frame in BGR is turned
            to RGB before the models see it. Grey frames are used as they are and
            handled as ``predict`` handles them (the models' preprocessing converts
            channels as configured).
        keep: Keep every pushed frame's outputs for :meth:`labels` / :meth:`save`.
            Memory grows with every frame pushed; leave ``False`` for a long live run.
        tracker: Reserved for a future per-frame tracker; must be ``None``.

    Frames: ``(H, W)`` or ``(H, W, C)`` with ``C`` 1 or 3, ``uint8``, as a numpy array
    or a torch tensor (a CUDA tensor is used where it is). Anything else raises. All
    frames of a stream must have the same size; call :meth:`reset` to start a new
    video (it clears the part stage's crop floor and the frame counter).
    """

    def __init__(
        self,
        predictor: Any,
        layer: Any,
        *,
        channel_order: str = "rgb",
        keep: bool = False,
        tracker: Any = None,
    ) -> None:
        """Wrap a built predictor; see the class docstring."""
        if channel_order not in ("rgb", "bgr"):
            raise ValueError(
                f"channel_order must be 'rgb' or 'bgr'; got {channel_order!r}."
            )
        if tracker is not None:
            raise NotImplementedError(
                "A tracker inside the stream is not available yet; pass tracker=None."
            )
        self.predictor = predictor
        self.layer = layer
        self.channel_order = channel_order
        self.keep = keep
        self.node_names = (
            [n.name for n in predictor.skeleton.nodes]
            if predictor.skeleton is not None
            else []
        )
        self.last_latency_ms: Optional[float] = None
        self._outputs: List[Any] = []
        self._closed = False
        self.reset()

    # ── state ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Start a new video: frame counter, frame size and the part stage's crop floor."""
        self._next_idx = 0
        self._shape: Optional[Tuple[int, ...]] = None
        if self.predictor.part_stage is not None:
            self.predictor.part_stage.reset()

    def close(self) -> None:
        """Release the stream (the models stay with the predictor)."""
        self._closed = True

    def __enter__(self) -> "PoseStream":
        """Context-manager entry."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context-manager exit: :meth:`close`."""
        self.close()

    # ── frames ───────────────────────────────────────────────────────────
    def _as_batch(self, frame: Any) -> Any:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                raise TypeError(
                    f"push() takes uint8 frames; got a {frame.dtype} array."
                )
        elif isinstance(frame, torch.Tensor):
            if frame.dtype != torch.uint8:
                raise TypeError(
                    f"push() takes uint8 frames; got a {frame.dtype} tensor."
                )
        else:
            raise TypeError(
                f"push() takes a numpy array or a torch tensor; got {type(frame).__name__}."
            )
        if frame.ndim == 2:
            frame = frame[..., None]
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3):
            raise ValueError(
                f"push() takes (H, W) or (H, W, C) frames with C = 1 or 3; got shape {tuple(frame.shape)}."
            )
        if self._shape is not None and tuple(frame.shape) != self._shape:
            raise ValueError(
                f"Frame shape changed from {self._shape} to {tuple(frame.shape)}; call reset() to start a new video."
            )
        self._shape = tuple(frame.shape)
        if self.channel_order == "bgr" and frame.shape[-1] == 3:
            if isinstance(frame, np.ndarray):
                frame = np.ascontiguousarray(frame[..., ::-1])
            else:
                frame = torch.flip(frame, dims=[-1])
        return frame[None]

    def push(self, frame: Any, frame_idx: Optional[int] = None) -> FramePoses:
        """Run one frame through the pipeline and return its instances.

        Args:
            frame: ``(H, W)`` or ``(H, W, C)`` uint8 (numpy, or a torch tensor on any device).
            frame_idx: The frame's index (default: the stream's counter, from 0).
        """
        if self._closed:
            raise RuntimeError("The stream is closed.")
        t0 = time.perf_counter()
        images = self._as_batch(frame)
        idx = self._next_idx if frame_idx is None else int(frame_idx)
        self._next_idx = idx + 1
        outputs = self._run(images, idx)
        if self.keep:
            self._outputs.append(outputs)
        poses = frame_poses_from_outputs(outputs, idx, self.node_names)
        self.last_latency_ms = 1e3 * (time.perf_counter() - t0)
        return poses

    def _run(self, images: Any, idx: int) -> Any:
        """Predict's per-batch path (``Predictor._batch_iter``) for one frame."""
        from sleap_nn.inference.providers import Batch

        p = self.predictor
        outputs = self.layer.predict(images)
        outputs = p.filter_pipeline(outputs)
        video_indices = np.zeros((1,), dtype=np.int64)
        if p.part_stage is not None:
            outputs = p.part_stage(images, outputs, video_indices)
        batch = Batch(
            images=images,
            frame_indices=np.asarray([idx], dtype=np.int64),
            video_indices=video_indices,
        )
        return p._stamp_metadata(outputs, batch)

    def warmup(
        self, frame_shape: Tuple[int, ...], crops: Tuple[int, ...] = (1, 2, 3, 4)
    ) -> None:
        """Run dummy data through every network so the first real frame is not the slow one.

        Two empty frames go through the whole pipeline (the detector), then the pose
        model and the part model each run on 1-4 dummy crops (boxes placed in an
        empty frame). The frame counter, the frame size and the crop floor are reset
        afterwards.
        """
        shape = (
            tuple(frame_shape) if len(frame_shape) == 3 else tuple(frame_shape) + (1,)
        )
        dummy = np.zeros(shape, dtype=np.uint8)
        for _ in range(2):
            self._run(dummy[None], 0)
        h, w = shape[:2]
        side = max(8.0, min(h, w) / 4.0)
        part = self.predictor.part_stage
        given = getattr(getattr(part, "layer", None), "centroid_layer", None)
        for k in crops:
            boxes = np.full((1, k, 4), np.nan, dtype=np.float32)
            for j in range(k):
                x0 = (
                    (w - side) * (j + 0.5) / k
                )  # k squares side by side, centred vertically
                y0 = (h - side) / 2
                boxes[0, j] = [x0, y0, x0 + side, y0 + side]
            fb = torch.from_numpy(boxes)
            fs = torch.ones((1, k))
            if given is not None and getattr(given, "use_given_boxes", False):
                pose_layer = copy.copy(self.layer)  # the pose model on given boxes
                pose_layer.centroid_layer = given
                pose_layer.predict(dummy[None], frame_boxes=fb, frame_box_scores=fs)
                part.layer.predict(dummy[None], frame_boxes=fb, frame_box_scores=fs)
        self.reset()

    # ── collected frames ─────────────────────────────────────────────────
    def labels(self, video: Any = None) -> Any:
        """The frames pushed so far as ``sio.Labels`` (needs ``keep=True``).

        Args:
            video: The ``sio.Video`` to attach the frames to (default: a placeholder
                ``sio.Video`` named ``"stream"``, not backed by a file).
        """
        import sleap_io as sio

        if not self.keep:
            raise ValueError("labels() needs a stream opened with keep=True.")
        if video is None:
            video = sio.Video(filename="stream", open_backend=False)
        return self.predictor.to_labels(
            self._outputs, videos=[video], keep_empty_frames=True
        )

    def save(self, path: str, video: Any = None) -> None:
        """Write the frames pushed so far to a ``.slp`` file (needs ``keep=True``)."""
        self.labels(video).save(str(path), embed=False)
