"""Per-frame detection arrays for the MOT trackers.

:func:`extract_detections` turns one frame's sleap-io detections (pose
``Instance`` / ``PredictedInstance`` or ``PredictedSegmentationMask``) into the
vectorized :class:`FrameDetections` every tracker, the calibration and the
identity linker operate on. The source objects are kept (``sources``) so tracks
can be written back onto them.

Geometry conventions (shared by every consumer -- never mix them):

- ``raw_boxes`` is the tight extent of the visible keypoints (or the mask's
  bounding box), ``xyxy`` in image pixels.
- ``boxes`` is ``raw_boxes`` padded by ``box_padding * max(w, h)`` on each side.
- ``centroids`` is the CENTER OF ``raw_boxes`` -- the same reference point the
  Kalman filters track (box center). Do not use the node median here: mixing the
  two reference points injects a fake per-frame displacement.
"""

from typing import Optional, Sequence

import attrs
import numpy as np


@attrs.define
class FrameDetections:
    """Vectorized detections of one frame.

    Attributes:
        frame_idx: Frame index of the detections.
        sources: The original sleap-io objects, in input order.
        boxes: ``(n, 4)`` float64 padded ``xyxy`` boxes; NaN rows where invalid.
        raw_boxes: ``(n, 4)`` float64 unpadded extents; NaN rows where invalid.
        centroids: ``(n, 2)`` float64 raw-box centers; NaN rows where invalid.
        keypoints: ``(n, K, 2)`` float64 keypoints with NaN for missing nodes, or
            ``None`` for masks.
        scores: ``(n,)`` float64 confidences in ``[0, 1]``.
        embeddings: ``(n, D)`` float32 L2-normalized appearance vectors with zero
            rows where missing, or ``None`` when no detection has one.
        has_embedding: ``(n,)`` bool.
        valid: ``(n,)`` bool: the detection has usable geometry.
        n_visible: ``(n,)`` int visible nodes (masks: 1 if non-empty else 0).
        n_nodes: Nodes per detection ``K`` (masks: 1).
    """

    frame_idx: int
    sources: list
    boxes: np.ndarray
    raw_boxes: np.ndarray
    centroids: np.ndarray
    keypoints: Optional[np.ndarray]
    scores: np.ndarray
    embeddings: Optional[np.ndarray]
    has_embedding: np.ndarray
    valid: np.ndarray
    n_visible: np.ndarray
    n_nodes: int

    def __len__(self) -> int:
        """Number of detections."""
        return len(self.sources)

    def subset(self, idx: np.ndarray) -> "FrameDetections":
        """Return the detections at ``idx`` (index array or bool mask), copied."""
        raise NotImplementedError

    @classmethod
    def empty(cls, frame_idx: int, n_nodes: int = 1) -> "FrameDetections":
        """Return an empty frame (zero detections) with correctly shaped arrays."""
        raise NotImplementedError


def extract_detections(
    objs: Sequence,
    frame_idx: int,
    *,
    carrier: str = "pose",
    box_padding: float = 0.1,
    pseudo_box_size: Optional[float] = None,
    score_mode: str = "auto",
    embedding_dim: Optional[int] = None,
) -> FrameDetections:
    """Build :class:`FrameDetections` from one frame's sleap-io detections.

    Args:
        objs: The frame's detections: pose instances (``carrier="pose"``) or
            segmentation masks (``carrier="mask"``).
        frame_idx: Frame index of the detections.
        carrier: ``"pose"`` or ``"mask"``.
        box_padding: Fraction of ``max(w, h)`` added to each side of the box.
        pseudo_box_size: Side (px) of the square box used for degenerate
            geometry (a single visible node, or a raw box narrower than 1 px).
            ``None`` floors the box at 1 px instead.
        score_mode: ``"auto"``, ``"instance"`` or ``"none"`` (see
            :class:`sleap_nn.tracking.mot.params.MOTParams`).
        embedding_dim: Expected appearance-vector dimension; vectors of another
            dimension are treated as missing. ``None`` accepts the first seen.

    Returns:
        The frame's :class:`FrameDetections`.
    """
    raise NotImplementedError
