"""Dataset analysis utilities for config generation.

This module provides tools for extracting statistics from SLP files
to inform automatic configuration of training parameters.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import sleap_io as sio

from sleap_nn.data.instance_cropping import find_max_instance_bbox_size
from sleap_nn.data.providers import get_max_height_width, get_max_instances


class ViewType(Enum):
    """Camera view orientation for augmentation defaults."""

    SIDE = "side"
    TOP = "top"
    UNKNOWN = "unknown"


@dataclass
class DatasetStats:
    """Statistics extracted from an SLP file for auto-configuration.

    Attributes:
        slp_path: Path to the source SLP file.
        num_labeled_frames: Number of frames with user labels.
        num_videos: Number of video sources.
        max_height: Maximum image height across videos.
        max_width: Maximum image width across videos.
        num_channels: Number of image channels (1=grayscale, 3=RGB).
        max_instances_per_frame: Maximum instances in any single frame.
        avg_instances_per_frame: Average instances per frame.
        max_bbox_size: Maximum bounding box dimension of any instance.
        avg_bbox_size: Average bounding box size.
        num_nodes: Number of skeleton nodes.
        num_edges: Number of skeleton edges.
        node_names: List of node names.
        edges: List of edge tuples (source_name, dest_name).
        has_tracks: Whether track annotations exist.
        num_tracks: Number of unique tracks.
        estimated_total_bytes: Estimated memory for all images.
        overlap_frequency: Fraction of frames with overlapping instances (IoU > 0.2).
    """

    slp_path: str
    num_labeled_frames: int
    num_videos: int
    max_height: int
    max_width: int
    num_channels: int
    max_instances_per_frame: int
    avg_instances_per_frame: float
    max_bbox_size: float
    avg_bbox_size: float
    num_nodes: int
    num_edges: int
    node_names: List[str]
    edges: List[Tuple[str, str]]
    has_tracks: bool
    num_tracks: int
    estimated_total_bytes: int
    overlap_frequency: float = 0.0

    @property
    def frame_area(self) -> int:
        """Total pixel area of a frame."""
        return self.max_height * self.max_width

    @property
    def animal_to_frame_ratio(self) -> float:
        """Ratio of average animal size to frame dimension (linear, not area).

        This gives a more intuitive percentage - e.g., if an animal bbox is 100px
        and the frame is 1000px, the ratio is 10% (not 1% which area would give).
        """
        if self.max_dimension == 0:
            return 0
        return self.avg_bbox_size / self.max_dimension

    @property
    def is_single_instance(self) -> bool:
        """Whether dataset has only single animals per frame."""
        return self.max_instances_per_frame == 1

    @property
    def is_multi_instance(self) -> bool:
        """Whether dataset has multiple animals per frame."""
        return self.max_instances_per_frame > 1

    @property
    def has_identity(self) -> bool:
        """Whether identity tracking is available."""
        return self.has_tracks and self.num_tracks > 1

    @property
    def is_grayscale(self) -> bool:
        """Whether images are grayscale."""
        return self.num_channels == 1

    @property
    def is_rgb(self) -> bool:
        """Whether images are RGB."""
        return self.num_channels == 3

    @property
    def max_dimension(self) -> int:
        """Maximum image dimension."""
        return max(self.max_height, self.max_width)

    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Dataset: {Path(self.slp_path).name}",
            f"  Labeled frames: {self.num_labeled_frames}",
            f"  Videos: {self.num_videos}",
            f"  Image size: {self.max_width}x{self.max_height} "
            f"({'grayscale' if self.is_grayscale else 'RGB'})",
            f"  Max instances/frame: {self.max_instances_per_frame}",
            f"  Avg instances/frame: {self.avg_instances_per_frame:.1f}",
            f"  Max bbox size: {self.max_bbox_size:.1f}px",
            f"  Avg bbox size: {self.avg_bbox_size:.1f}px",
            f"  Animal size: ~{self.animal_to_frame_ratio * 100:.1f}% of frame",
            f"  Overlap frequency: {self.overlap_frequency * 100:.1f}%",
            f"  Skeleton: {self.num_nodes} nodes, {self.num_edges} edges",
            f"  Tracks: {self.num_tracks if self.has_tracks else 'none'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return repr string."""
        return (
            f"DatasetStats("
            f"frames={self.num_labeled_frames}, "
            f"size={self.max_width}x{self.max_height}, "
            f"instances={self.max_instances_per_frame}, "
            f"nodes={self.num_nodes})"
        )


def _detect_channels(labels: sio.Labels) -> int:
    """Detect number of image channels from labels.

    Args:
        labels: sleap_io Labels object.

    Returns:
        Number of channels (1 for grayscale, 3 for RGB).
    """
    # Check video properties first
    for video in labels.videos:
        if video.shape is not None and len(video.shape) >= 4:
            return video.shape[3]
        if video.grayscale is not None:
            return 1 if video.grayscale else 3

    # Try to read a frame to detect channels
    try:
        if labels.labeled_frames:
            frame = labels.labeled_frames[0]
            img = frame.image
            if img is not None:
                if img.ndim == 2:
                    return 1
                elif img.ndim == 3:
                    return img.shape[2]
    except Exception:
        pass

    # Default to grayscale
    return 1


def _compute_bbox_stats(
    labels: sio.Labels, user_instances_only: bool = True
) -> Tuple[float, float]:
    """Compute average and minimum bounding box sizes.

    Args:
        labels: sleap_io Labels object.
        user_instances_only: If True, only consider user-labeled instances.

    Returns:
        Tuple of (avg_bbox_size, min_bbox_size).
    """
    bbox_sizes = []

    for lf in labels.labeled_frames:
        instances = lf.user_instances if user_instances_only else lf.instances
        for instance in instances:
            if instance.is_empty:
                continue

            pts = instance.numpy()
            valid_pts = pts[~np.isnan(pts).any(axis=1)]

            if len(valid_pts) < 2:
                continue

            min_pt = valid_pts.min(axis=0)
            max_pt = valid_pts.max(axis=0)
            bbox_size = max(max_pt[0] - min_pt[0], max_pt[1] - min_pt[1])

            if bbox_size > 0:
                bbox_sizes.append(bbox_size)

    if not bbox_sizes:
        return 100.0, 50.0  # Default values

    return float(np.mean(bbox_sizes)), float(np.min(bbox_sizes))


def _compute_avg_instances(
    labels: sio.Labels, user_instances_only: bool = True
) -> float:
    """Compute average instances per frame.

    Args:
        labels: sleap_io Labels object.
        user_instances_only: If True, only count user-labeled instances.

    Returns:
        Average number of instances per frame.
    """
    if not labels.labeled_frames:
        return 0.0

    counts = []
    for lf in labels.labeled_frames:
        instances = lf.user_instances if user_instances_only else lf.instances
        # Filter out empty instances
        non_empty = [inst for inst in instances if not inst.is_empty]
        counts.append(len(non_empty))

    return float(np.mean(counts)) if counts else 0.0


def _compute_bbox_iou(bbox1: Tuple[float, float, float, float],
                      bbox2: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two bounding boxes.

    Args:
        bbox1: (x_min, y_min, x_max, y_max) for first box.
        bbox2: (x_min, y_min, x_max, y_max) for second box.

    Returns:
        Intersection over Union value (0-1).
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def _get_instance_bbox(instance) -> Optional[Tuple[float, float, float, float]]:
    """Get bounding box for an instance.

    Args:
        instance: sleap_io Instance object.

    Returns:
        (x_min, y_min, x_max, y_max) or None if no valid points.
    """
    if instance.is_empty:
        return None

    pts = instance.numpy()
    valid_pts = pts[~np.isnan(pts).any(axis=1)]

    if len(valid_pts) < 2:
        return None

    x_min, y_min = valid_pts.min(axis=0)
    x_max, y_max = valid_pts.max(axis=0)

    return (float(x_min), float(y_min), float(x_max), float(y_max))


def _compute_overlap_frequency(
    labels: sio.Labels,
    user_instances_only: bool = True,
    iou_threshold: float = 0.2,
) -> float:
    """Compute fraction of frames with overlapping instances.

    Args:
        labels: sleap_io Labels object.
        user_instances_only: If True, only consider user-labeled instances.
        iou_threshold: IoU threshold to consider instances overlapping.

    Returns:
        Fraction of frames with at least one pair of overlapping instances.
    """
    if not labels.labeled_frames:
        return 0.0

    overlap_count = 0

    for lf in labels.labeled_frames:
        instances = lf.user_instances if user_instances_only else lf.instances
        # Get bboxes for all non-empty instances
        bboxes = []
        for inst in instances:
            bbox = _get_instance_bbox(inst)
            if bbox is not None:
                bboxes.append(bbox)

        # Check for overlaps between any pair
        has_overlap = False
        for i in range(len(bboxes)):
            for j in range(i + 1, len(bboxes)):
                iou = _compute_bbox_iou(bboxes[i], bboxes[j])
                if iou >= iou_threshold:
                    has_overlap = True
                    break
            if has_overlap:
                break

        if has_overlap:
            overlap_count += 1

    return overlap_count / len(labels.labeled_frames)


def analyze_slp(
    path: str,
    *,
    user_instances_only: bool = True,
) -> DatasetStats:
    """Analyze an SLP file and extract statistics for auto-configuration.

    Args:
        path: Path to the .slp file.
        user_instances_only: If True, only analyze user-labeled instances.

    Returns:
        DatasetStats object with extracted statistics.

    Example:
        >>> stats = analyze_slp("labels.slp")
        >>> print(f"Max instances: {stats.max_instances_per_frame}")
        >>> print(f"Image size: {stats.max_width}x{stats.max_height}")
    """
    path = str(Path(path).resolve())
    labels = sio.load_slp(path)

    # Basic counts
    num_labeled_frames = len(labels.labeled_frames)
    num_videos = len(labels.videos)

    # Image dimensions
    max_height, max_width = get_max_height_width(labels)
    num_channels = _detect_channels(labels)

    # Instance statistics
    max_instances = get_max_instances(labels)

    # Get bbox statistics
    try:
        max_bbox = find_max_instance_bbox_size(labels)
    except Exception:
        max_bbox = 100.0  # Default fallback

    avg_bbox, min_bbox = _compute_bbox_stats(labels, user_instances_only)
    avg_instances = _compute_avg_instances(labels, user_instances_only)

    # Compute overlap frequency (only for multi-instance datasets)
    if max_instances > 1:
        overlap_freq = _compute_overlap_frequency(labels, user_instances_only)
    else:
        overlap_freq = 0.0

    # Skeleton info
    skeleton = labels.skeletons[0] if labels.skeletons else None
    node_names = [n.name for n in skeleton.nodes] if skeleton else []
    edges = (
        [(e.source.name, e.destination.name) for e in skeleton.edges]
        if skeleton
        else []
    )

    # Track info
    has_tracks = len(labels.tracks) > 0
    num_tracks = len(labels.tracks)

    # Estimate total bytes for caching
    bytes_per_frame = max_height * max_width * num_channels
    estimated_total_bytes = bytes_per_frame * num_labeled_frames

    return DatasetStats(
        slp_path=path,
        num_labeled_frames=num_labeled_frames,
        num_videos=num_videos,
        max_height=max_height,
        max_width=max_width,
        num_channels=num_channels,
        max_instances_per_frame=max_instances,
        avg_instances_per_frame=avg_instances,
        max_bbox_size=max_bbox,
        avg_bbox_size=avg_bbox,
        num_nodes=len(node_names),
        num_edges=len(edges),
        node_names=node_names,
        edges=edges,
        has_tracks=has_tracks,
        num_tracks=num_tracks,
        estimated_total_bytes=estimated_total_bytes,
        overlap_frequency=overlap_freq,
    )
