"""Frame in, pose out: one frame at a time through ``predict``'s pipeline.

:func:`sleap_nn.inference.run.open_stream` builds the same pipeline as
:func:`sleap_nn.inference.run.predict` (first stage -> pose -> filters -> streaming part
stage, from checkpoints or network-only export folders) and returns a
:class:`PoseStream`. Each :meth:`PoseStream.push` runs one frame through it and returns
that frame's instances as plain arrays (:class:`FramePoses`).

A video pushed frame by frame gives the same points as ``sleap-nn predict
--batch_size 1 --part_streaming`` with the same arguments: the stream runs predict's
per-batch path with a batch of one frame.

``push`` returns when the frame's poses are back on the host: no queue, no batching,
no look-ahead. The stream does not drop frames, run threads or pace itself; if frames
arrive faster than ``push`` returns, the caller decides what to skip. How long a push
takes depends on the models, the settings, the number of animals and the hardware:
time it on the target machine (``last_latency_ms``).

One caller, one thread: a stream keeps per-video state (the part stage's running crop
floor, the frame counter, the frame size) and must not be pushed from several threads.
"""

from __future__ import annotations

import copy
import time
from typing import Any, List, Optional, Sequence, Tuple, Union

import attrs
import numpy as np
import torch

_WARMUP_CROPS = (1, 2, 3, 4)


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
            instance came from (NaN when the first stage reports no boxes).
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


def _frame_poses_from_outputs(
    outputs: Any, frame_idx: int, node_names: List[str]
) -> FramePoses:
    """``FramePoses`` of batch slot 0, keeping the instances ``Outputs.to_labels`` keeps."""
    n = len(node_names)
    if outputs.pred_keypoints is None:  # refused at open; kept for safety
        e = np.zeros((0,), dtype=np.float32)
        return FramePoses(
            int(frame_idx),
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


def _check_keypoint_layer(layer: Any) -> None:
    """Raise unless ``layer`` gives keypoints on frames alone (what a stream needs)."""
    from sleap_nn.inference.layers.bottomup import BottomUpLayer
    from sleap_nn.inference.layers.bottomup_multiclass import BottomUpMultiClassLayer
    from sleap_nn.inference.layers.single_instance import SingleInstanceLayer
    from sleap_nn.inference.layers.topdown import TopDownLayer
    from sleap_nn.inference.layers.topdown_segmentation import (
        TopDownSegmentationLayer,
    )

    if isinstance(layer, TopDownLayer) and not isinstance(
        layer, TopDownSegmentationLayer
    ):
        stage_1 = layer.centroid_layer
        if any(
            getattr(stage_1, a, False)
            for a in ("use_gt_centroids", "use_gt_boxes", "use_given_boxes")
        ):
            raise ValueError(
                "open_stream needs a first stage that finds the animals in each frame "
                "(a centroid model or a box detector) with the centered-instance model; "
                "a centered-instance model alone needs ground truth or given boxes."
            )
        return
    if isinstance(layer, (SingleInstanceLayer, BottomUpLayer, BottomUpMultiClassLayer)):
        return
    raise ValueError(
        "open_stream needs models that end in keypoints: a single-instance or "
        "bottom-up model, or a top-down pair (a centroid model or a box detector, "
        f"with a centered-instance model). These models build a {type(layer).__name__}, "
        "which gives no keypoints; use predict() for it."
    )


class PoseStream:
    """Push one frame, get its poses. Build it with :func:`sleap_nn.inference.run.open_stream`.

    Frames: ``(H, W)`` or ``(H, W, C)`` with ``C`` 1 or 3, ``uint8``, as a numpy array
    or a torch tensor (a CUDA tensor is used where it is); any memory layout (views
    such as ``frame[..., ::-1]`` are copied to contiguous memory). Anything else
    raises. All frames of a stream have one size; :meth:`reset` starts a new video.

    Args:
        predictor: A built :class:`~sleap_nn.inference.predictor.Predictor` (with
            its ``part_stage`` when a part model runs).
        layer: The predictor's layer with the prediction-time overrides applied
            (``Predictor._scoped_postprocess_layer``), as ``predict`` uses it.
        channel_order: ``"rgb"`` (default; what sleap-io's video readers return)
            or ``"bgr"`` (what OpenCV returns): a 3-channel frame in BGR is turned
            to RGB before the models see it. Grey frames are used as they are and
            handled as ``predict`` handles them.
        keep: Keep every pushed frame's outputs (on the CPU, without images or
            confidence maps) for :meth:`labels` / :meth:`save`. Memory grows with
            every frame pushed; leave ``False`` for a long live run.

    Attributes:
        node_names: The node names of every :class:`FramePoses` this stream returns.
        last_latency_ms: Time of the last successful :meth:`push`, from the call to
            the poses back as numpy arrays (``None`` before the first).
    """

    def __init__(
        self,
        predictor: Any,
        layer: Any,
        *,
        channel_order: str = "rgb",
        keep: bool = False,
    ) -> None:
        """Wrap a built predictor; see the class docstring."""
        if channel_order not in ("rgb", "bgr"):
            raise ValueError(
                f"channel_order must be 'rgb' or 'bgr'; got {channel_order!r}."
            )
        _check_keypoint_layer(layer)
        self._predictor = predictor
        self._layer = layer
        self.channel_order = channel_order
        self.keep = keep
        self.node_names = (
            [n.name for n in predictor.skeleton.nodes]
            if predictor.skeleton is not None
            else []
        )
        self.last_latency_ms: Optional[float] = None
        self._outputs: List[Any] = []
        self._video = 0  # video number of the frames being pushed (one per reset)
        self._video_frames = 0  # frames kept for the current video
        self._closed = False
        self._clear()

    # ── state ────────────────────────────────────────────────────────────
    def _clear(self) -> None:
        self._next_idx = 0
        self._shape: Optional[Tuple[int, ...]] = None
        if self._predictor.part_stage is not None:
            self._predictor.part_stage.reset()

    def reset(self) -> None:
        """Start a new video: frame counter, frame size and the part stage's crop floor.

        With ``keep=True``, the frames pushed after a reset go to a new video in
        :meth:`labels` (frames already kept stay).
        """
        self._check_open()
        if self._video_frames:
            self._video += 1
            self._video_frames = 0
        self._clear()

    def close(self) -> None:
        """Close the stream (calling it again does nothing).

        Without ``keep``, the stream lets go of the models. With ``keep=True`` it holds
        them until it is deleted, so :meth:`labels` and :meth:`save` still work.
        """
        if self._closed:
            return
        self._closed = True
        if not self.keep:
            self._predictor = None
            self._layer = None

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("The stream is closed.")

    def __enter__(self) -> "PoseStream":
        """Context-manager entry."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context-manager exit: :meth:`close`."""
        self.close()

    # ── frames ───────────────────────────────────────────────────────────
    def _as_batch(self, frame: Any) -> Tuple[Any, Tuple[int, ...]]:
        """Checked ``(1, H, W, C)`` batch of ``frame`` and its shape (state untouched)."""
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
        shape = tuple(frame.shape)
        if self._shape is not None and shape != self._shape:
            raise ValueError(
                f"Frame shape changed from {self._shape} to {shape}; call reset() to start a new video."
            )
        if self.channel_order == "bgr" and shape[-1] == 3:
            frame = (
                frame[..., ::-1]
                if isinstance(frame, np.ndarray)
                else torch.flip(frame, dims=[-1])
            )
        frame = (
            np.ascontiguousarray(frame)
            if isinstance(frame, np.ndarray)
            else frame.contiguous()
        )
        return frame[None], shape

    def push(self, frame: Any, frame_idx: Optional[int] = None) -> FramePoses:
        """Run one frame through the pipeline and return its instances.

        A push that raises leaves the stream as it was before the call (frame
        counter, frame size, the part stage's crop floor, kept frames).

        Args:
            frame: ``(H, W)`` or ``(H, W, C)`` uint8 (numpy, or a torch tensor on any device).
            frame_idx: The frame's index (default: the stream's counter, from 0; the
                counter continues from ``frame_idx + 1``).
        """
        self._check_open()
        t0 = time.perf_counter()
        images, shape = self._as_batch(frame)
        idx = self._next_idx if frame_idx is None else int(frame_idx)
        outputs = self._run(images, idx, self._video)
        poses = _frame_poses_from_outputs(outputs, idx, self.node_names)
        if self.keep:
            self._outputs.append(outputs.slim())
            self._video_frames += 1
        self._shape = shape
        self._next_idx = idx + 1
        self.last_latency_ms = 1e3 * (time.perf_counter() - t0)
        return poses

    def _run(self, images: Any, idx: int, video: int) -> Any:
        """Predict's per-batch path (``Predictor._batch_iter``) for one frame."""
        from sleap_nn.inference.providers import Batch

        p = self._predictor
        outputs = self._layer.predict(images)
        outputs = p.filter_pipeline(outputs)
        video_indices = np.full((1,), video, dtype=np.int64)
        if p.part_stage is not None:
            outputs = p.part_stage(images, outputs, video_indices)
        batch = Batch(
            images=images,
            frame_indices=np.asarray([idx], dtype=np.int64),
            video_indices=video_indices,
        )
        return p._stamp_metadata(outputs, batch)

    def _given_box_pose_layer(self) -> Any:
        """The top-down layer with a given-box first stage (``None`` if not top-down)."""
        from sleap_nn.inference.layers.bbox import BBoxStageLayer
        from sleap_nn.inference.layers.configs import PostprocessConfig
        from sleap_nn.inference.layers.topdown import TopDownLayer

        layer = self._layer
        if not isinstance(layer, TopDownLayer):
            return None
        given = BBoxStageLayer(
            backend=layer.centered_instance_layer.backend,
            decoder=None,
            use_given_boxes=True,
            box_score_threshold=0.0,
            preprocess_config=layer.centroid_layer.preprocess_config,
            postprocess_config=PostprocessConfig(),
        )
        pose = copy.copy(layer)
        pose.centroid_layer = given
        return pose

    def warmup(self, frame_shape: Sequence[int]) -> None:
        """Run dummy data through every network so the first real frame is not the slow one.

        :func:`~sleap_nn.inference.run.open_stream` calls this when given
        ``frame_shape``; without it there is no warm-up. Two empty frames go through
        the whole pipeline (the first stage, and everything for one-stage models);
        then the pose model of a top-down pair and the part model each run on 1, 2,
        3 and 4 dummy crops (boxes placed in an empty frame). The frame counter,
        frame size and crop floor are cleared afterwards; nothing is kept.
        """
        self._check_open()
        shape = (
            tuple(frame_shape) if len(frame_shape) == 3 else tuple(frame_shape) + (1,)
        )
        dummy = np.zeros(shape, dtype=np.uint8)[None]
        for _ in range(2):
            self._run(dummy, 0, self._video)
        pose = self._given_box_pose_layer()
        part = self._predictor.part_stage
        h, w = shape[:2]
        side = max(8.0, min(h, w) / 4.0)
        for k in _WARMUP_CROPS:
            boxes = np.full((1, k, 4), np.nan, dtype=np.float32)
            for j in range(k):  # k squares side by side, centred vertically
                x0 = (w - side) * (j + 0.5) / k
                y0 = (h - side) / 2
                boxes[0, j] = [x0, y0, x0 + side, y0 + side]
            fb = torch.from_numpy(boxes)
            fs = torch.ones((1, k))
            if pose is not None:
                pose.predict(dummy, frame_boxes=fb, frame_box_scores=fs)
            if part is not None:
                part.layer.predict(dummy, frame_boxes=fb, frame_box_scores=fs)
        self._clear()

    # ── collected frames ─────────────────────────────────────────────────
    def labels(self, video: Union[Any, Sequence[Any], None] = None) -> Any:
        """The frames pushed so far as ``sio.Labels`` (needs ``keep=True``).

        Args:
            video: The ``sio.Video`` to attach the frames to, or one per video when
                :meth:`reset` started new videos (default: placeholder videos named
                ``"stream"``, ``"stream-1"``, ..., not backed by a file).
        """
        import sleap_io as sio

        if not self.keep:
            raise ValueError("labels() needs a stream opened with keep=True.")
        n_videos = self._video + (1 if self._video_frames else 0)
        if video is None:
            videos = [
                sio.Video(
                    filename="stream" if i == 0 else f"stream-{i}", open_backend=False
                )
                for i in range(max(n_videos, 1))
            ]
        elif isinstance(video, sio.Video):
            videos = [video]
        else:
            videos = list(video)
        if len(videos) < n_videos:
            raise ValueError(
                f"The stream holds frames of {n_videos} videos (reset() starts a new "
                f"one); pass one sio.Video per video, got {len(videos)}."
            )
        return self._predictor.to_labels(
            self._outputs, videos=videos, keep_empty_frames=True
        )

    def save(self, path: str, video: Union[Any, Sequence[Any], None] = None) -> None:
        """Write the frames pushed so far to a ``.slp`` file (needs ``keep=True``)."""
        self.labels(video).save(str(path), embed=False)
