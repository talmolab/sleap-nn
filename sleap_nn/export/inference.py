"""Reusable Python API for running inference on exported SLEAP-NN models.

Example usage::

    from sleap_nn.export.inference import predict

    labels, stats = predict(
        export_dir="path/to/Leopard_trt",
        video_path="path/to/video.mp4",
        batch_size=4,
        cpu_workers=2,
    )
    labels.save("output.slp")
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.export.metadata import ExportMetadata
from sleap_nn.export.predictors import load_exported_model
from sleap_nn.export.utils import build_bottomup_candidate_template
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.inference.utils import get_skeleton_from_config

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def load_video_batch(video, frame_indices):
    """Load a batch of video frames as a uint8 NCHW numpy array.

    Args:
        video: ``sleap_io.Video`` object.
        frame_indices: Iterable of integer frame indices to load.

    Returns:
        ``np.ndarray`` of shape ``(N, C, H, W)`` with dtype ``uint8``.
    """
    frames = []
    for idx in frame_indices:
        frame = np.asarray(video[idx])
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
        frames.append(frame)
    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# Private helpers (moved from cli.py)
# ---------------------------------------------------------------------------


def _prefetch_video_batches(video, frame_indices, batch_size, prefetch_queue):
    """Decode video batches ahead of the producer in a background thread.

    Reads frames from a separate video handle and puts decoded batches onto
    a bounded queue.  The producer thread pulls from this queue instead of
    calling ``load_video_batch`` directly, overlapping I/O with GPU
    inference.

    Args:
        video: ``sio.Video`` (will be deep-copied for thread safety).
        frame_indices: Full list of frame indices to process.
        batch_size: Number of frames per batch.
        prefetch_queue: ``queue.Queue`` to put ``(batch_array, batch_indices)``
            items onto.  A ``None`` sentinel is put at the end.
    """
    import copy

    video_copy = copy.deepcopy(video)
    for start in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[start : start + batch_size]
        batch = load_video_batch(video_copy, batch_indices)
        prefetch_queue.put((batch, batch_indices))
    prefetch_queue.put(None)  # sentinel


def _find_training_config_for_predict(export_dir: Path, model_type: str) -> Path:
    """Find training config file in export directory.

    Raises:
        FileNotFoundError: If no training config is found.
    """
    candidates = []
    if model_type == "topdown":
        candidates.extend(
            [
                export_dir / "training_config_centered_instance.yaml",
                export_dir / "training_config_centered_instance.json",
            ]
        )
    elif model_type == "multi_class_topdown_combined":
        candidates.extend(
            [
                export_dir / "training_config_multi_class_topdown.yaml",
                export_dir / "training_config_multi_class_topdown.json",
            ]
        )
    candidates.extend(
        [
            export_dir / "training_config.yaml",
            export_dir / "training_config.json",
            export_dir / f"training_config_{model_type}.yaml",
            export_dir / f"training_config_{model_type}.json",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No training_config found in {export_dir} for model_type={model_type}."
    )


def _predict_topdown_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    max_instances=None,
):
    """Convert top-down model outputs to LabeledFrames."""
    labeled_frames = []
    centroids = outputs["centroids"]
    centroid_vals = outputs["centroid_vals"]
    peaks = outputs["peaks"]
    peak_vals = outputs["peak_vals"]
    instance_valid = outputs["instance_valid"]

    for batch_idx, frame_idx in enumerate(frame_indices):
        instances = []
        valid_mask = instance_valid[batch_idx].astype(bool)
        for inst_idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue
            pts = peaks[batch_idx, inst_idx]
            scores = peak_vals[batch_idx, inst_idx]
            score = float(centroid_vals[batch_idx, inst_idx])
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=score,
                    skeleton=skeleton,
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_multiclass_topdown_combined_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    class_names: list,
    max_instances=None,
):
    """Convert combined multiclass top-down model outputs to LabeledFrames.

    Args:
        outputs: Model outputs with centroids, centroid_vals, peaks, peak_vals,
                class_logits, instance_valid.
        frame_indices: Frame indices corresponding to batch.
        video: sleap_io.Video object.
        skeleton: sleap_io.Skeleton object.
        class_names: List of class names (e.g., ["female", "male"]).
        max_instances: Maximum instances per frame (None = n_classes).

    Returns:
        List of LabeledFrame objects.
    """
    from scipy.optimize import linear_sum_assignment

    labeled_frames = []
    centroids = outputs["centroids"]
    centroid_vals = outputs["centroid_vals"]
    peaks = outputs["peaks"]
    peak_vals = outputs["peak_vals"]
    class_logits = outputs["class_logits"]
    instance_valid = outputs["instance_valid"]

    n_classes = len(class_names)

    for batch_idx, frame_idx in enumerate(frame_indices):
        valid_mask = instance_valid[batch_idx].astype(bool)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            continue

        # Gather valid instances
        valid_peaks = peaks[batch_idx, valid_mask]  # (n_valid, n_nodes, 2)
        valid_peak_vals = peak_vals[batch_idx, valid_mask]  # (n_valid, n_nodes)
        valid_centroid_vals = centroid_vals[batch_idx, valid_mask]  # (n_valid,)
        valid_class_logits = class_logits[batch_idx, valid_mask]  # (n_valid, n_classes)

        # Compute softmax probabilities from logits
        logits = valid_class_logits - np.max(valid_class_logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        # Use Hungarian matching to assign classes to instances
        # Maximize total probability (minimize negative)
        cost = -probs
        row_inds, col_inds = linear_sum_assignment(cost)

        # Create instances with class assignments
        instances = []
        for row_idx, class_idx in zip(row_inds, col_inds):
            pts = valid_peaks[row_idx]
            scores = valid_peak_vals[row_idx]
            score = float(valid_centroid_vals[row_idx])

            # Get track name from class names
            track_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"class_{class_idx}"
            )

            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=score,
                    skeleton=skeleton,
                    track=sio.Track(name=track_name),
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_bottomup_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    paf_scorer,
    candidate_template,
    input_scale,
    peak_conf_threshold=0.2,
    max_instances=None,
):
    """Convert bottom-up model outputs to LabeledFrames."""
    labeled_frames = []

    peaks = torch.from_numpy(outputs["peaks"]).to(torch.float32)
    peak_vals = torch.from_numpy(outputs["peak_vals"]).to(torch.float32)
    line_scores = torch.from_numpy(outputs["line_scores"]).to(torch.float32)
    candidate_mask = torch.from_numpy(outputs["candidate_mask"]).to(torch.bool)

    batch_size, n_nodes, k, _ = peaks.shape
    peaks_flat = peaks.reshape(batch_size, n_nodes * k, 2)
    peak_vals_flat = peak_vals.reshape(batch_size, n_nodes * k)

    peak_channel_inds_base = candidate_template["peak_channel_inds"]
    edge_inds_base = candidate_template["edge_inds"]
    edge_peak_inds_base = candidate_template["edge_peak_inds"]

    peaks_list = []
    peak_vals_list = []
    peak_channel_inds_list = []
    edge_inds_list = []
    edge_peak_inds_list = []
    line_scores_list = []

    for b in range(batch_size):
        peaks_list.append(peaks_flat[b])
        peak_vals_list.append(peak_vals_flat[b])
        peak_channel_inds_list.append(peak_channel_inds_base)

        candidate_mask_flat = candidate_mask[b].reshape(-1)
        line_scores_flat = line_scores[b].reshape(-1)

        if candidate_mask_flat.numel() == 0:
            edge_inds_list.append(torch.empty((0,), dtype=torch.int32))
            edge_peak_inds_list.append(torch.empty((0, 2), dtype=torch.int32))
            line_scores_list.append(torch.empty((0,), dtype=torch.float32))
            continue

        # Filter candidates by peak confidence threshold
        peak_vals_b = peak_vals_flat[b]
        peak_conf_valid = peak_vals_b > peak_conf_threshold
        src_valid = peak_conf_valid[edge_peak_inds_base[:, 0].long()]
        dst_valid = peak_conf_valid[edge_peak_inds_base[:, 1].long()]
        valid = candidate_mask_flat & src_valid & dst_valid

        edge_inds_list.append(edge_inds_base[valid])
        edge_peak_inds_list.append(edge_peak_inds_base[valid])
        line_scores_list.append(line_scores_flat[valid])

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = paf_scorer.match_candidates(
        edge_inds_list,
        edge_peak_inds_list,
        line_scores_list,
    )

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = paf_scorer.group_instances(
        peaks_list,
        peak_vals_list,
        peak_channel_inds_list,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )

    predicted_instances = [p / input_scale for p in predicted_instances]

    for batch_idx, frame_idx in enumerate(frame_indices):
        instances = []
        for pts, confs, score in zip(
            predicted_instances[batch_idx],
            predicted_peak_scores[batch_idx],
            predicted_instance_scores[batch_idx],
        ):
            pts_np = pts.cpu().numpy()
            if np.isnan(pts_np).all():
                continue
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts_np,
                    point_scores=confs.cpu().numpy(),
                    score=float(score),
                    skeleton=skeleton,
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_single_instance_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
):
    """Convert single-instance model outputs to LabeledFrames."""
    labeled_frames = []
    peaks = outputs["peaks"]  # (batch, n_nodes, 2)
    peak_vals = outputs["peak_vals"]  # (batch, n_nodes)

    for batch_idx, frame_idx in enumerate(frame_indices):
        pts = peaks[batch_idx]
        scores = peak_vals[batch_idx]

        # Compute instance score as mean of valid peak values
        valid_mask = ~np.isnan(pts[:, 0])
        if valid_mask.any():
            instance_score = float(np.mean(scores[valid_mask]))
        else:
            instance_score = 0.0

        instance = sio.PredictedInstance.from_numpy(
            points_data=pts,
            point_scores=scores,
            score=instance_score,
            skeleton=skeleton,
        )

        labeled_frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=int(frame_idx),
                instances=[instance],
            )
        )

    return labeled_frames


def _predict_centroid_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    anchor_node_idx: int,
    max_instances=None,
):
    """Convert centroid model outputs to LabeledFrames.

    For centroid-only models, creates instances with only the anchor node filled in.
    All other nodes are set to NaN.

    Args:
        outputs: Model outputs with centroids, centroid_vals, instance_valid.
        frame_indices: Frame indices corresponding to batch.
        video: sleap_io.Video object.
        skeleton: sleap_io.Skeleton object.
        anchor_node_idx: Index of the anchor node in the skeleton.
        max_instances: Maximum instances to output per frame.

    Returns:
        List of LabeledFrame objects.
    """
    labeled_frames = []
    centroids = outputs["centroids"]  # (batch, max_instances, 2)
    centroid_vals = outputs["centroid_vals"]  # (batch, max_instances)
    instance_valid = outputs["instance_valid"]  # (batch, max_instances)

    n_nodes = len(skeleton.nodes)

    for batch_idx, frame_idx in enumerate(frame_indices):
        instances = []
        valid_mask = instance_valid[batch_idx].astype(bool)

        for inst_idx, is_valid in enumerate(valid_mask):
            if not is_valid:
                continue

            # Create points array with NaN for all nodes except anchor
            pts = np.full((n_nodes, 2), np.nan, dtype=np.float32)
            pts[anchor_node_idx] = centroids[batch_idx, inst_idx]

            # Create scores array - anchor gets centroid score, others get NaN
            scores = np.full((n_nodes,), np.nan, dtype=np.float32)
            scores[anchor_node_idx] = centroid_vals[batch_idx, inst_idx]

            instance_score = float(centroid_vals[batch_idx, inst_idx])

            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=instance_score,
                    skeleton=skeleton,
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_multiclass_bottomup_frames(
    outputs,
    frame_indices,
    video,
    skeleton,
    class_names: list,
    input_scale: float = 1.0,
    peak_conf_threshold: float = 0.2,
    max_instances: int = None,
):
    """Convert bottom-up multiclass model outputs to LabeledFrames.

    Uses class probability maps to group peaks by identity rather than PAFs.

    Args:
        outputs: Model outputs with peaks, peak_vals, peak_mask, class_probs.
        frame_indices: Frame indices corresponding to batch.
        video: sleap_io.Video object.
        skeleton: sleap_io.Skeleton object.
        class_names: List of class names (e.g., ["female", "male"]).
        input_scale: Scale factor applied to input.
        peak_conf_threshold: Minimum peak confidence to include.
        max_instances: Maximum instances per frame (None = n_classes).

    Returns:
        List of LabeledFrame objects.
    """
    from scipy.optimize import linear_sum_assignment

    labeled_frames = []
    n_classes = len(class_names)

    peaks = outputs["peaks"]  # (batch, n_nodes, max_peaks, 2)
    peak_vals = outputs["peak_vals"]  # (batch, n_nodes, max_peaks)
    peak_mask = outputs["peak_mask"]  # (batch, n_nodes, max_peaks)
    class_probs = outputs["class_probs"]  # (batch, n_nodes, max_peaks, n_classes)

    batch_size, n_nodes, max_peaks, _ = peaks.shape
    n_nodes_skel = len(skeleton.nodes)

    for batch_idx, frame_idx in enumerate(frame_indices):
        # Initialize instances for each class
        instance_points = np.full(
            (n_classes, n_nodes_skel, 2), np.nan, dtype=np.float32
        )
        instance_scores = np.full((n_classes, n_nodes_skel), np.nan, dtype=np.float32)
        instance_class_probs = np.full((n_classes,), 0.0, dtype=np.float32)

        # Process each node independently
        for node_idx in range(min(n_nodes, n_nodes_skel)):
            # Get valid peaks for this node
            valid = peak_mask[batch_idx, node_idx].astype(bool)
            valid = valid & (peak_vals[batch_idx, node_idx] > peak_conf_threshold)

            if not valid.any():
                continue

            valid_peaks = peaks[batch_idx, node_idx][valid]  # (n_valid, 2)
            valid_vals = peak_vals[batch_idx, node_idx][valid]  # (n_valid,)
            valid_class_probs = class_probs[batch_idx, node_idx][
                valid
            ]  # (n_valid, n_classes)

            # Use Hungarian matching to assign peaks to classes
            # Maximize class probabilities (minimize negative)
            cost = -valid_class_probs
            row_inds, col_inds = linear_sum_assignment(cost)

            # Assign matched peaks to instances
            for peak_idx, class_idx in zip(row_inds, col_inds):
                if class_idx < n_classes:
                    instance_points[class_idx, node_idx] = (
                        valid_peaks[peak_idx] / input_scale
                    )
                    instance_scores[class_idx, node_idx] = valid_vals[peak_idx]
                    instance_class_probs[class_idx] += valid_class_probs[
                        peak_idx, class_idx
                    ]

        # Create predicted instances
        instances = []
        for class_idx in range(n_classes):
            pts = instance_points[class_idx]
            scores = instance_scores[class_idx]

            # Skip if no valid points
            if np.isnan(pts).all():
                continue

            # Compute instance score as mean of valid peak values
            valid_mask = ~np.isnan(pts[:, 0])
            if valid_mask.any():
                instance_score = float(np.mean(scores[valid_mask]))
            else:
                instance_score = 0.0

            # Get track name from class names
            track_name = (
                class_names[class_idx]
                if class_idx < len(class_names)
                else f"class_{class_idx}"
            )

            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=scores,
                    score=instance_score,
                    skeleton=skeleton,
                    track=sio.Track(name=track_name),
                )
            )

        if max_instances is not None and instances:
            instances = sorted(instances, key=lambda inst: inst.score, reverse=True)
            instances = instances[:max_instances]

        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=int(frame_idx),
                    instances=instances,
                )
            )

    return labeled_frames


def _predict_bottomup_raw(
    outputs,
    frame_indices,
    paf_scorer,
    candidate_template,
    input_scale,
    peak_conf_threshold=0.2,
    max_instances=None,
):
    """Run bottom-up PAF grouping and return raw numpy arrays.

    This is the CPU-heavy core of bottom-up post-processing, factored out so it
    can run in a worker process without needing unpicklable sio objects.

    Returns:
        List of dicts, one per frame that has predictions:
        {"frame_idx": int,
         "instance_peaks": np.ndarray (n_inst, n_nodes, 2),
         "instance_peak_scores": np.ndarray (n_inst, n_nodes),
         "instance_scores": np.ndarray (n_inst,)}
    """
    peaks = torch.from_numpy(outputs["peaks"]).to(torch.float32)
    peak_vals = torch.from_numpy(outputs["peak_vals"]).to(torch.float32)
    line_scores = torch.from_numpy(outputs["line_scores"]).to(torch.float32)
    candidate_mask = torch.from_numpy(outputs["candidate_mask"]).to(torch.bool)

    batch_size, n_nodes, k, _ = peaks.shape
    peaks_flat = peaks.reshape(batch_size, n_nodes * k, 2)
    peak_vals_flat = peak_vals.reshape(batch_size, n_nodes * k)

    peak_channel_inds_base = candidate_template["peak_channel_inds"]
    edge_inds_base = candidate_template["edge_inds"]
    edge_peak_inds_base = candidate_template["edge_peak_inds"]

    peaks_list = []
    peak_vals_list = []
    peak_channel_inds_list = []
    edge_inds_list = []
    edge_peak_inds_list = []
    line_scores_list = []

    for b in range(batch_size):
        peaks_list.append(peaks_flat[b])
        peak_vals_list.append(peak_vals_flat[b])
        peak_channel_inds_list.append(peak_channel_inds_base)

        candidate_mask_flat = candidate_mask[b].reshape(-1)
        line_scores_flat = line_scores[b].reshape(-1)

        if candidate_mask_flat.numel() == 0:
            edge_inds_list.append(torch.empty((0,), dtype=torch.int32))
            edge_peak_inds_list.append(torch.empty((0, 2), dtype=torch.int32))
            line_scores_list.append(torch.empty((0,), dtype=torch.float32))
            continue

        peak_vals_b = peak_vals_flat[b]
        peak_conf_valid = peak_vals_b > peak_conf_threshold
        src_valid = peak_conf_valid[edge_peak_inds_base[:, 0].long()]
        dst_valid = peak_conf_valid[edge_peak_inds_base[:, 1].long()]
        valid = candidate_mask_flat & src_valid & dst_valid

        edge_inds_list.append(edge_inds_base[valid])
        edge_peak_inds_list.append(edge_peak_inds_base[valid])
        line_scores_list.append(line_scores_flat[valid])

    (
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    ) = paf_scorer.match_candidates(
        edge_inds_list,
        edge_peak_inds_list,
        line_scores_list,
    )

    (
        predicted_instances,
        predicted_peak_scores,
        predicted_instance_scores,
    ) = paf_scorer.group_instances(
        peaks_list,
        peak_vals_list,
        peak_channel_inds_list,
        match_edge_inds,
        match_src_peak_inds,
        match_dst_peak_inds,
        match_line_scores,
    )

    predicted_instances = [p / input_scale for p in predicted_instances]

    results = []
    for batch_idx, frame_idx in enumerate(frame_indices):
        frame_peaks = []
        frame_scores = []
        frame_instance_scores = []
        for pts, confs, score in zip(
            predicted_instances[batch_idx],
            predicted_peak_scores[batch_idx],
            predicted_instance_scores[batch_idx],
        ):
            pts_np = pts.cpu().numpy()
            if np.isnan(pts_np).all():
                continue
            frame_peaks.append(pts_np)
            frame_scores.append(confs.cpu().numpy())
            frame_instance_scores.append(float(score))

        if not frame_peaks:
            continue

        instance_peaks = np.stack(frame_peaks, axis=0)
        instance_peak_scores = np.stack(frame_scores, axis=0)
        instance_scores = np.array(frame_instance_scores, dtype=np.float32)

        if max_instances is not None and len(instance_scores) > max_instances:
            top_k = np.argsort(instance_scores)[::-1][:max_instances]
            instance_peaks = instance_peaks[top_k]
            instance_peak_scores = instance_peak_scores[top_k]
            instance_scores = instance_scores[top_k]

        results.append(
            {
                "frame_idx": int(frame_idx),
                "instance_peaks": instance_peaks,
                "instance_peak_scores": instance_peak_scores,
                "instance_scores": instance_scores,
            }
        )

    return results


def _bottomup_postprocess_worker(
    gpu_output_queue,
    result_queue,
    paf_scorer_kwargs,
    candidate_template_data,
    input_scale,
    peak_conf_threshold,
    max_instances,
):
    """Worker process for CPU-bound bottom-up post-processing.

    Reconstructs PAFScorer from config, then loops pulling inference outputs
    from gpu_output_queue and pushing processed results to result_queue.
    Exits when it receives None (sentinel).

    Args:
        gpu_output_queue: Queue of (seq_id, outputs_dict, batch_indices) or None.
        result_queue: Queue of (seq_id, results_list).
        paf_scorer_kwargs: Dict of kwargs to construct PAFScorer directly.
        candidate_template_data: Dict with numpy arrays for candidate template.
        input_scale: Float scale factor.
        peak_conf_threshold: Float confidence threshold.
        max_instances: Optional int max instances per frame.
    """
    paf_scorer = PAFScorer(**paf_scorer_kwargs)

    # Convert candidate template numpy arrays back to tensors
    candidate_template = {
        "peak_channel_inds": torch.from_numpy(
            candidate_template_data["peak_channel_inds"]
        ),
        "edge_inds": torch.from_numpy(candidate_template_data["edge_inds"]),
        "edge_peak_inds": torch.from_numpy(candidate_template_data["edge_peak_inds"]),
    }

    while True:
        item = gpu_output_queue.get()
        if item is None:
            break

        seq_id, outputs, batch_indices = item
        results = _predict_bottomup_raw(
            outputs,
            batch_indices,
            paf_scorer,
            candidate_template,
            input_scale,
            peak_conf_threshold,
            max_instances,
        )
        result_queue.put((seq_id, results))


def _raw_results_to_labeled_frames(raw_results, video, skeleton):
    """Convert raw numpy result dicts to sio.LabeledFrame objects.

    Args:
        raw_results: List of dicts from _predict_bottomup_raw.
        video: sio.Video object.
        skeleton: sio.Skeleton object.

    Returns:
        List of sio.LabeledFrame objects.
    """
    labeled_frames = []
    for r in raw_results:
        instances = []
        for pts, confs, score in zip(
            r["instance_peaks"],
            r["instance_peak_scores"],
            r["instance_scores"],
        ):
            instances.append(
                sio.PredictedInstance.from_numpy(
                    points_data=pts,
                    point_scores=confs,
                    score=float(score),
                    skeleton=skeleton,
                )
            )
        if instances:
            labeled_frames.append(
                sio.LabeledFrame(
                    video=video,
                    frame_idx=r["frame_idx"],
                    instances=instances,
                )
            )
    return labeled_frames


def _run_bottomup_pipelined(
    predictor,
    video,
    skeleton,
    frame_indices,
    batch_size,
    paf_scorer,
    candidate_template,
    input_scale,
    peak_conf_threshold,
    max_instances,
    cpu_workers,
    progress_callback=None,
):
    """Run bottom-up inference with pipelined GPU producer + CPU consumer workers.

    The main thread acts as the inference producer: it loads batches, runs
    ONNX/TensorRT inference, and puts raw outputs on a queue. CPU worker
    processes pull from that queue, run PAF matching/grouping, and put results
    on a result queue. The main thread collects results in frame order using
    a reorder buffer.

    Args:
        predictor: Loaded ONNX/TRT model predictor.
        video: sio.Video object.
        skeleton: sio.Skeleton object.
        frame_indices: List of frame indices to process.
        batch_size: Number of frames per batch.
        paf_scorer: PAFScorer instance (used to extract config for workers).
        candidate_template: Dict with torch tensors for candidate template.
        input_scale: Float scale factor.
        peak_conf_threshold: Float confidence threshold.
        max_instances: Optional int max instances per frame.
        cpu_workers: Number of CPU worker processes.
        progress_callback: Optional callable(processed, total) for progress.

    Returns:
        Tuple of (labeled_frames, infer_time, post_time).
    """
    import queue
    from threading import Thread

    queue_maxsize = 2 * cpu_workers

    ctx = multiprocessing.get_context("spawn")
    gpu_output_queue = ctx.Queue(maxsize=queue_maxsize)
    result_queue = (
        ctx.Queue()
    )  # unbounded: avoids deadlock since producer+collector are sequential

    # Serialize PAFScorer config for workers (all simple types)
    paf_scorer_kwargs = {
        "part_names": list(paf_scorer.part_names),
        "edges": [tuple(e) for e in paf_scorer.edges],
        "pafs_stride": paf_scorer.pafs_stride,
        "max_edge_length_ratio": paf_scorer.max_edge_length_ratio,
        "dist_penalty_weight": paf_scorer.dist_penalty_weight,
        "n_points": paf_scorer.n_points,
        "min_instance_peaks": paf_scorer.min_instance_peaks,
        "min_line_scores": paf_scorer.min_line_scores,
    }

    # Serialize candidate template as numpy arrays
    candidate_template_data = {
        "peak_channel_inds": candidate_template["peak_channel_inds"].numpy(),
        "edge_inds": candidate_template["edge_inds"].numpy(),
        "edge_peak_inds": candidate_template["edge_peak_inds"].numpy(),
    }

    # Start workers
    workers = []
    for _ in range(cpu_workers):
        p = ctx.Process(
            target=_bottomup_postprocess_worker,
            args=(
                gpu_output_queue,
                result_queue,
                paf_scorer_kwargs,
                candidate_template_data,
                input_scale,
                peak_conf_threshold,
                max_instances,
            ),
        )
        p.start()
        workers.append(p)

    # Producer: prefetch video batches in a background thread, run inference
    prefetch_queue = queue.Queue(maxsize=2)
    prefetch_thread = Thread(
        target=_prefetch_video_batches,
        args=(video, frame_indices, batch_size, prefetch_queue),
        daemon=True,
    )
    prefetch_thread.start()

    infer_time = 0.0
    total_batches = 0
    try:
        while True:
            item = prefetch_queue.get()
            if item is None:
                break
            batch, batch_indices = item

            infer_start = time.perf_counter()
            outputs = predictor.predict(batch)
            infer_time += time.perf_counter() - infer_start

            gpu_output_queue.put((total_batches, outputs, batch_indices))
            total_batches += 1

            if progress_callback is not None:
                processed = min(total_batches * batch_size, len(frame_indices))
                progress_callback(processed, len(frame_indices))
    finally:
        # Send sentinels to shut down workers
        for _ in workers:
            gpu_output_queue.put(None)
        prefetch_thread.join()

    # Collector: gather results in order using reorder buffer
    labeled_frames = []
    reorder_buffer = {}
    next_seq_id = 0
    collected = 0
    post_start = time.perf_counter()

    while collected < total_batches:
        seq_id, raw_results = result_queue.get()
        reorder_buffer[seq_id] = raw_results
        collected += 1

        # Drain buffer in order
        while next_seq_id in reorder_buffer:
            raw = reorder_buffer.pop(next_seq_id)
            labeled_frames.extend(_raw_results_to_labeled_frames(raw, video, skeleton))
            next_seq_id += 1

    post_time = time.perf_counter() - post_start

    # Wait for workers to exit
    for p in workers:
        p.join()

    return labeled_frames, infer_time, post_time


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict(
    export_dir: Union[str, Path],
    video_path: Union[str, Path],
    runtime: str = "auto",
    device: str = "auto",
    batch_size: int = 4,
    n_frames: Optional[int] = None,
    max_edge_length_ratio: float = 0.25,
    dist_penalty_weight: float = 1.0,
    n_points: int = 10,
    min_instance_peaks: float = 0,
    min_line_scores: float = 0.25,
    peak_conf_threshold: float = 0.2,
    max_instances: Optional[int] = None,
    cpu_workers: int = 0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[sio.Labels, dict]:
    """Run inference on an exported model and return predictions.

    Args:
        export_dir: Directory containing the exported model (model.onnx or
            model.trt) along with export_metadata.json and training_config.yaml.
        video_path: Path to the video file to process.
        runtime: Runtime to use — ``"auto"`` (prefer TRT), ``"onnx"``, or
            ``"tensorrt"``.
        device: Device string (e.g. ``"cuda:0"``). ``"auto"`` picks CUDA if
            available.
        batch_size: Number of frames per inference batch.
        n_frames: Limit to first *n_frames* frames (``None`` = all).
        max_edge_length_ratio: Bottom-up: max edge length as ratio of PAF dims.
        dist_penalty_weight: Bottom-up: weight for distance penalty in PAF scoring.
        n_points: Bottom-up: number of points to sample along PAF.
        min_instance_peaks: Bottom-up: minimum peaks required per instance.
        min_line_scores: Bottom-up: minimum line score threshold.
        peak_conf_threshold: Bottom-up: peak confidence threshold for filtering.
        max_instances: Maximum instances to output per frame.
        cpu_workers: Number of CPU worker processes for parallel bottom-up
            post-processing.  ``0`` = sequential mode.
        progress_callback: Optional callable ``(processed, total)`` invoked
            after each batch.

    Returns:
        Tuple of ``(labels, stats)`` where *labels* is a ``sleap_io.Labels``
        object and *stats* is a dict with keys ``"total_time"``,
        ``"infer_time"``, ``"post_time"``, and ``"fps"``.

    Raises:
        FileNotFoundError: If the export directory, model file, or training
            config cannot be found.
        ValueError: If the runtime or model type is unsupported.
    """
    from datetime import datetime

    export_dir = Path(export_dir)
    video_path = Path(video_path)

    # Load metadata
    metadata_path = export_dir / "export_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    metadata = ExportMetadata.load(metadata_path)

    # Find model file
    onnx_path = export_dir / "model.onnx"
    trt_path = export_dir / "model.trt"

    if runtime == "auto":
        if trt_path.exists():
            model_path = trt_path
            runtime = "tensorrt"
        elif onnx_path.exists():
            model_path = onnx_path
            runtime = "onnx"
        else:
            raise FileNotFoundError(
                f"No model found in {export_dir}. Expected model.onnx or model.trt."
            )
    elif runtime == "onnx":
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        model_path = onnx_path
    elif runtime == "tensorrt":
        if not trt_path.exists():
            raise FileNotFoundError(f"TensorRT model not found: {trt_path}")
        model_path = trt_path
    else:
        raise ValueError(f"Unknown runtime: {runtime}")

    # Load training config for skeleton
    cfg_path = _find_training_config_for_predict(export_dir, metadata.model_type)
    if cfg_path.suffix in {".yaml", ".yml"}:
        cfg = OmegaConf.load(cfg_path.as_posix())
    else:
        from sleap_nn.config.training_job_config import TrainingJobConfig

        cfg = TrainingJobConfig.load_sleap_config(cfg_path.as_posix())
    skeletons = get_skeleton_from_config(cfg.data_config.skeletons)
    skeleton = skeletons[0]

    # Load video
    video = sio.Video.from_filename(video_path.as_posix())
    total_frames = len(video) if n_frames is None else min(n_frames, len(video))
    frame_indices = list(range(total_frames))

    predictor = load_exported_model(
        model_path.as_posix(), runtime=runtime, device=device
    )

    # Set up centroid anchor node if needed
    anchor_node_idx = None
    if metadata.model_type == "centroid":
        anchor_part = cfg.model_config.head_configs.centroid.confmaps.anchor_part
        node_names = [n.name for n in skeleton.nodes]
        if anchor_part in node_names:
            anchor_node_idx = node_names.index(anchor_part)
        else:
            raise ValueError(
                f"Anchor part '{anchor_part}' not found in skeleton nodes: {node_names}"
            )

    # Set up bottom-up post-processing if needed
    paf_scorer = None
    candidate_template = None
    if metadata.model_type == "bottomup":
        paf_scorer = PAFScorer.from_config(
            cfg.model_config.head_configs.bottomup,
            max_edge_length_ratio=max_edge_length_ratio,
            dist_penalty_weight=dist_penalty_weight,
            n_points=n_points,
            min_instance_peaks=min_instance_peaks,
            min_line_scores=min_line_scores,
        )
        max_peaks = metadata.max_peaks_per_node
        if max_peaks is None:
            raise ValueError("Bottom-up export metadata missing max_peaks_per_node.")
        edge_inds_tuples = [(int(e[0]), int(e[1])) for e in paf_scorer.edge_inds]
        peak_channel_inds, edge_inds_tensor, edge_peak_inds = (
            build_bottomup_candidate_template(
                n_nodes=metadata.n_nodes,
                max_peaks_per_node=max_peaks,
                edge_inds=edge_inds_tuples,
            )
        )
        candidate_template = {
            "peak_channel_inds": peak_channel_inds,
            "edge_inds": edge_inds_tensor,
            "edge_peak_inds": edge_peak_inds,
        }

    labeled_frames = []
    total_start = time.perf_counter()
    infer_time = 0.0
    post_time = 0.0

    # Use pipelined path for bottom-up when cpu_workers > 0
    if cpu_workers > 0 and metadata.model_type == "bottomup":
        labeled_frames, infer_time, post_time = _run_bottomup_pipelined(
            predictor=predictor,
            video=video,
            skeleton=skeleton,
            frame_indices=frame_indices,
            batch_size=batch_size,
            paf_scorer=paf_scorer,
            candidate_template=candidate_template,
            input_scale=metadata.input_scale,
            peak_conf_threshold=peak_conf_threshold,
            max_instances=max_instances,
            cpu_workers=cpu_workers,
            progress_callback=progress_callback,
        )
    else:
        for start in range(0, len(frame_indices), batch_size):
            batch_indices = frame_indices[start : start + batch_size]
            batch = load_video_batch(video, batch_indices)

            infer_start = time.perf_counter()
            outputs = predictor.predict(batch)
            infer_time += time.perf_counter() - infer_start

            post_start = time.perf_counter()
            if metadata.model_type == "topdown":
                labeled_frames.extend(
                    _predict_topdown_frames(
                        outputs,
                        batch_indices,
                        video,
                        skeleton,
                        max_instances=max_instances,
                    )
                )
            elif metadata.model_type == "bottomup":
                labeled_frames.extend(
                    _predict_bottomup_frames(
                        outputs,
                        batch_indices,
                        video,
                        skeleton,
                        paf_scorer,
                        candidate_template,
                        input_scale=metadata.input_scale,
                        peak_conf_threshold=peak_conf_threshold,
                        max_instances=max_instances,
                    )
                )
            elif metadata.model_type == "single_instance":
                labeled_frames.extend(
                    _predict_single_instance_frames(
                        outputs,
                        batch_indices,
                        video,
                        skeleton,
                    )
                )
            elif metadata.model_type == "centroid":
                labeled_frames.extend(
                    _predict_centroid_frames(
                        outputs,
                        batch_indices,
                        video,
                        skeleton,
                        anchor_node_idx=anchor_node_idx,
                        max_instances=max_instances,
                    )
                )
            elif metadata.model_type == "multi_class_bottomup":
                labeled_frames.extend(
                    _predict_multiclass_bottomup_frames(
                        outputs,
                        batch_indices,
                        video,
                        skeleton,
                        class_names=metadata.class_names or [],
                        input_scale=metadata.input_scale,
                        peak_conf_threshold=peak_conf_threshold,
                        max_instances=max_instances,
                    )
                )
            elif metadata.model_type == "multi_class_topdown_combined":
                labeled_frames.extend(
                    _predict_multiclass_topdown_combined_frames(
                        outputs,
                        batch_indices,
                        video,
                        skeleton,
                        class_names=metadata.class_names or [],
                        max_instances=max_instances,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported model_type for predict: {metadata.model_type}"
                )
            post_time += time.perf_counter() - post_start

            # Progress update
            if progress_callback is not None:
                processed = min(start + batch_size, len(frame_indices))
                progress_callback(processed, len(frame_indices))

    total_time = time.perf_counter() - total_start
    fps = len(frame_indices) / total_time if total_time > 0 else 0

    labels = sio.Labels(
        videos=[video],
        skeletons=[skeleton],
        labeled_frames=labeled_frames,
    )
    labels.provenance = {
        "sleap_nn_version": metadata.sleap_nn_version,
        "export_format": runtime,
        "model_type": metadata.model_type,
        "inference_timestamp": datetime.now().isoformat(),
    }

    stats = {
        "total_time": total_time,
        "infer_time": infer_time,
        "post_time": post_time,
        "fps": fps,
    }

    return labels, stats
