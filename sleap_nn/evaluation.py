"""This module is to compute evaluation metrics for trained models."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import attrs
import sleap_io as sio
from loguru import logger
import click
from pathlib import Path


@attrs.define(auto_attribs=True, slots=True)
class MatchInstance:
    """Class to have a new structure for sio.Instance object."""

    instance: sio.Instance
    frame_idx: int
    video_path: str


def get_instances(labeled_frame: sio.LabeledFrame) -> List[MatchInstance]:
    """Get a list of instances of type MatchInstance from the Labeled Frame.

    Args:
        labeled_frame: Input Labeled frame of type sio.LabeledFrame.

    Returns:
        List of MatchInstance objects for the given labeled frame.
    """
    instance_list = []
    frame_idx = labeled_frame.frame_idx

    # Extract video path with fallbacks for embedded videos
    video = labeled_frame.video
    video_path = None
    if video is not None:
        backend = getattr(video, "backend", None)
        if backend is not None:
            # Try source_filename first (for embedded videos with provenance)
            video_path = getattr(backend, "source_filename", None)
            if video_path is None:
                video_path = getattr(backend, "filename", None)
        # Fallback to video.filename if backend doesn't have it
        if video_path is None:
            video_path = getattr(video, "filename", None)
            # Handle list filenames (image sequences)
            if isinstance(video_path, list) and video_path:
                video_path = video_path[0]
    # Final fallback: use a unique identifier
    if video_path is None:
        video_path = f"video_{id(video)}" if video is not None else "unknown"

    for instance in labeled_frame.instances:
        match_instance = MatchInstance(
            instance=instance, frame_idx=frame_idx, video_path=video_path
        )
        instance_list.append(match_instance)
    return instance_list


def find_frame_pairs(
    labels_gt: sio.Labels, labels_pr: sio.Labels, user_labels_only: bool = True
) -> List[Tuple[sio.LabeledFrame, sio.LabeledFrame]]:
    """Find corresponding frames across two sets of labels.

    This function uses sleap-io's robust video matching API to handle various
    scenarios including embedded videos, cross-platform paths, and videos with
    different metadata.

    Args:
        labels_gt: A `sio.Labels` instance with ground truth instances.
        labels_pr: A `sio.Labels` instance with predicted instances.
        user_labels_only: If False, frames with predicted instances in `labels_gt` will
            also be considered for matching.

    Returns:
        A list of pairs of `sio.LabeledFrame`s in the form `(frame_gt, frame_pr)`.
    """
    # Use sleap-io's robust video matching API (added in 0.6.2)
    # The match() method returns a MatchResult with video_map: {pred_video: gt_video}
    match_result = labels_gt.match(labels_pr)

    frame_pairs = []
    # Iterate over matched video pairs (pred_video -> gt_video mapping)
    for video_pr, video_gt in match_result.video_map.items():
        if video_gt is None:
            # No match found for this prediction video
            continue

        # Find labeled frames in this video.
        labeled_frames_gt = labels_gt.find(video_gt)
        if user_labels_only:
            for lf in labeled_frames_gt:
                lf.instances = lf.user_instances
            labeled_frames_gt = [
                lf for lf in labeled_frames_gt if len(lf.user_instances) > 0
            ]

        # Attempt to match each labeled frame in the ground truth.
        for labeled_frame_gt in labeled_frames_gt:
            labeled_frames_pr = labels_pr.find(
                video_pr, frame_idx=labeled_frame_gt.frame_idx
            )

            if not labeled_frames_pr:
                # No match
                continue
            elif len(labeled_frames_pr) == 1:
                # Match!
                frame_pairs.append((labeled_frame_gt, labeled_frames_pr[0]))

    return frame_pairs


def compute_instance_area(points: np.ndarray) -> np.ndarray:
    """Compute the area of the bounding box of a set of keypoints.

    Args:
        points: A numpy array of coordinates.

    Returns:
        The area of the bounding box of the points.
    """
    if points.ndim == 2:
        points = np.expand_dims(points, axis=0)

    min_pt = np.nanmin(points, axis=-2)
    max_pt = np.nanmax(points, axis=-2)

    return np.prod(max_pt - min_pt, axis=-1)


def compute_oks(
    points_gt: np.ndarray,
    points_pr: np.ndarray,
    scale: Optional[float] = None,
    stddev: float = 0.025,
    use_cocoeval: bool = True,
) -> np.ndarray:
    """Compute the object keypoints similarity between sets of points.

    Args:
        points_gt: Ground truth instances of shape (n_gt, n_nodes, n_ed),
            where n_nodes is the number of body parts/keypoint types, and n_ed
            is the number of Euclidean dimensions (typically 2 or 3). Keypoints
            that are missing/not visible should be represented as NaNs.
        points_pr: Predicted instance of shape (n_pr, n_nodes, n_ed).
        use_cocoeval: Indicates whether the OKS score is calculated like cocoeval
            method or not. True indicating the score is calculated using the
            cocoeval method (widely used and the code can be found here at
            https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L192C5-L233C20)
            and False indicating the score is calculated using the method exactly
            as given in the paper referenced in the Notes below.
        scale: Size scaling factor to use when weighing the scores, typically
            the area of the bounding box of the instance (in pixels). This
            should be of the length n_gt. If a scalar is provided, the same
            number is used for all ground truth instances. If set to None, the
            bounding box area of the ground truth instances will be calculated.
        stddev: The standard deviation associated with the spread in the
            localization accuracy of each node/keypoint type. This should be of
            the length n_nodes. "Easier" keypoint types will have lower values
            to reflect the smaller spread expected in localizing it.

    Returns:
        The object keypoints similarity between every pair of ground truth and
        predicted instance, a numpy array of of shape (n_gt, n_pr) in the range
        of [0, 1.0], with 1.0 denoting a perfect match.

    Notes:
        It's important to set the stddev appropriately when accounting for the
        difficulty of each keypoint type. For reference, the median value for
        all keypoint types in COCO is 0.072. The "easiest" keypoint is the left
        eye, with stddev of 0.025, since it is easy to precisely locate the
        eyes when labeling. The "hardest" keypoint is the left hip, with stddev
        of 0.107, since it's hard to locate the left hip bone without external
        anatomical features and since it is often occluded by clothing.

        The implementation here is based off of the descriptions in:
        Ronch & Perona. "Benchmarking and Error Diagnosis in Multi-Instance Pose
        Estimation." ICCV (2017).
    """
    if points_gt.ndim == 2:
        points_gt = np.expand_dims(points_gt, axis=0)
    if points_pr.ndim == 2:
        points_pr = np.expand_dims(points_pr, axis=0)

    if scale is None:
        scale = compute_instance_area(points_gt)

    n_gt, n_nodes, n_ed = points_gt.shape  # n_ed = 2 or 3 (euclidean dimensions)
    n_pr = points_pr.shape[0]

    # If scalar scale was provided, use the same for each ground truth instance.
    if np.isscalar(scale):
        scale = np.full(n_gt, scale)

    # If scalar standard deviation was provided, use the same for each node.
    if np.isscalar(stddev):
        stddev = np.full(n_nodes, stddev)

    # Compute displacement between each pair.
    displacement = np.reshape(points_gt, (n_gt, 1, n_nodes, n_ed)) - np.reshape(
        points_pr, (1, n_pr, n_nodes, n_ed)
    )
    assert displacement.shape == (n_gt, n_pr, n_nodes, n_ed)

    # Convert to pairwise Euclidean distances.
    distance = (displacement**2).sum(axis=-1)  # (n_gt, n_pr, n_nodes)
    assert distance.shape == (n_gt, n_pr, n_nodes)

    # Compute the normalization factor per keypoint.
    if use_cocoeval:
        # If use_cocoeval is True, then compute normalization factor according to cocoeval.
        spread_factor = (2 * stddev) ** 2
        scale_factor = 2 * (scale + np.spacing(1))
    else:
        # If use_cocoeval is False, then compute normalization factor according to the paper.
        spread_factor = stddev**2
        scale_factor = 2 * ((scale + np.spacing(1)) ** 2)
    normalization_factor = np.reshape(spread_factor, (1, 1, n_nodes)) * np.reshape(
        scale_factor, (n_gt, 1, 1)
    )
    assert normalization_factor.shape == (n_gt, 1, n_nodes)

    # Since a "miss" is considered as KS < 0.5, we'll set the
    # distances for predicted points that are missing to inf.
    missing_pr = np.any(np.isnan(points_pr), axis=-1)  # (n_pr, n_nodes)
    assert missing_pr.shape == (n_pr, n_nodes)
    distance[:, missing_pr] = np.inf

    # Compute the keypoint similarity as per the top of Eq. 1.
    ks = np.exp(-(distance / normalization_factor))  # (n_gt, n_pr, n_nodes)
    assert ks.shape == (n_gt, n_pr, n_nodes)

    # Set the KS for missing ground truth points to 0.
    # This is equivalent to the visibility delta function of the bottom
    # of Eq. 1.
    missing_gt = np.any(np.isnan(points_gt), axis=-1)  # (n_gt, n_nodes)
    assert missing_gt.shape == (n_gt, n_nodes)
    ks[np.expand_dims(missing_gt, axis=1)] = 0

    # Compute the OKS.
    n_visible_gt = np.sum(
        (~missing_gt).astype("float32"), axis=-1, keepdims=True
    )  # (n_gt, 1)
    oks = np.sum(ks, axis=-1) / n_visible_gt
    assert oks.shape == (n_gt, n_pr)

    return oks


def match_instances(
    frame_gt: sio.LabeledFrame,
    frame_pr: sio.LabeledFrame,
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
) -> Tuple[List[Tuple[sio.Instance, sio.PredictedInstance, float]], List[sio.Instance]]:
    """Match pairs of instances between ground truth and predictions in a frame.

    Args:
        frame_gt: A `sio.LabeledFrame` with ground truth instances.
        frame_pr: A `sio.LabeledFrame` with predicted instances.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sleap.Instance`s that could not be
        matched.

    Notes:
        This function uses the approach from the PASCAL VOC scoring procedure. Briefly,
        predictions are sorted descending by their instance-level prediction scores and
        greedily matched to ground truth instances which are then removed from the pool
        of available instances.

        Ground truth instances that remain unmatched are considered false negatives.
    """
    # Sort predicted instances by score.
    frame_pr_match_instances = get_instances(frame_pr)

    scores_pr = np.array(
        [
            m.instance.score
            for m in frame_pr_match_instances
            if hasattr(m.instance, "score")
        ]
    )
    idxs_pr = np.argsort(-scores_pr, kind="mergesort")  # descending
    scores_pr = scores_pr[idxs_pr]

    available_instances_gt = get_instances(frame_gt)
    available_instances_gt_idxs = list(range(len(available_instances_gt)))

    positive_pairs = []
    for idx_pr in idxs_pr:
        # Pull out predicted instance.
        instance_pr = frame_pr_match_instances[idx_pr]

        # Convert instances to point arrays.
        points_pr = np.expand_dims(instance_pr.instance.numpy(), axis=0)
        points_gt = np.stack(
            [
                available_instances_gt[idx].instance.numpy()
                for idx in available_instances_gt_idxs
            ],
            axis=0,
        )

        # Find the best match by computing OKS.
        oks = compute_oks(points_gt, points_pr, stddev=stddev, scale=scale)
        oks = np.squeeze(oks, axis=1)
        assert oks.shape == (len(points_gt),)

        oks[oks <= threshold] = np.nan
        best_match_gt_idx = np.argsort(-oks, kind="mergesort")[0]
        best_match_oks = oks[best_match_gt_idx]
        if np.isnan(best_match_oks):
            continue

        # Remove matched ground truth instance and add as a positive pair.
        instance_gt_idx = available_instances_gt_idxs.pop(best_match_gt_idx)
        instance_gt = available_instances_gt[instance_gt_idx]
        positive_pairs.append((instance_gt, instance_pr, best_match_oks))

        # Stop matching lower scoring instances if we run out of candidates in the
        # ground truth.
        if not available_instances_gt_idxs:
            break

    # Any remaining ground truth instances are considered false negatives.
    false_negatives = [
        available_instances_gt[idx] for idx in available_instances_gt_idxs
    ]

    return positive_pairs, false_negatives


def match_frame_pairs(
    frame_pairs: List[Tuple[sio.LabeledFrame, sio.LabeledFrame]],
    stddev: float = 0.025,
    scale: Optional[float] = None,
    threshold: float = 0,
) -> Tuple[List[Tuple[sio.Instance, sio.PredictedInstance, float]], List[sio.Instance]]:
    """Match all ground truth and predicted instances within each pair of frames.

    This is a wrapper for `match_instances()` but operates on lists of frames.

    Args:
        frame_pairs: A list of pairs of `sleap.LabeledFrame`s in the form
            `(frame_gt, frame_pr)`. These can be obtained with `find_frame_pairs()`.
        stddev: The expected spread of coordinates for OKS computation.
        scale: The scale for normalizing the OKS. If not set, the bounding box area will
            be used.
        threshold: The minimum OKS between a candidate pair of instances to be
            considered a match.

    Returns:
        A tuple of (`positive_pairs`, `false_negatives`).

        `positive_pairs` is a list of 3-tuples of the form
        `(instance_gt, instance_pr, oks)` containing the matched pair of instances and
        their OKS.

        `false_negatives` is a list of ground truth `sio.Instance`s that could not be
        matched.
    """
    positive_pairs = []
    false_negatives = []
    for frame_gt, frame_pr in frame_pairs:
        positive_pairs_frame, false_negatives_frame = match_instances(
            frame_gt,
            frame_pr,
            stddev=stddev,
            scale=scale,
            threshold=threshold,
        )
        positive_pairs.extend(positive_pairs_frame)
        false_negatives.extend(false_negatives_frame)

    return positive_pairs, false_negatives


def compute_dists(
    positive_pairs: List[Tuple[sio.Instance, sio.PredictedInstance, Any]],
) -> Dict[str, Union[np.ndarray, List[int], List[str]]]:
    """Compute Euclidean distances between matched pairs of instances.

    Args:
        positive_pairs: A list of tuples of the form `(instance_gt, instance_pr, _)`
            containing the matched pair of instances.

    Returns:
        A dictionary with the following keys:
            dists: An array of pairwise distances of shape `(n_positive_pairs, n_nodes)`
            frame_idxs: A list of frame indices corresponding to the `dists`
            video_paths: A list of video paths corresponding to the `dists`
    """
    dists = []
    frame_idxs = []
    video_paths = []
    for instance_gt, instance_pr, _ in positive_pairs:
        points_gt = instance_gt.instance.numpy()
        points_pr = instance_pr.instance.numpy()

        dists.append(np.linalg.norm(points_pr - points_gt, axis=-1))
        frame_idxs.append(instance_gt.frame_idx)
        video_paths.append(instance_gt.video_path)

    dists = np.array(dists)

    # Bundle everything into a dictionary
    dists_dict = {
        "dists": dists,
        "frame_idxs": frame_idxs,
        "video_paths": video_paths,
    }

    return dists_dict


class Evaluator:
    """Compute the standard evaluation metrics with the predicted and the ground-truth Labels.

    This class is used to calculate the common metrics for pose estimation models which
    includes voc metrics (with oks and pck), mOKS, distance metrics, pck metrics and
    visibility metrics.

    Args:
        ground_truth_instances: The `sio.Labels` dataset object with ground truth labels.
        predicted_instances: The `sio.Labels` dataset object with predicted labels.
        oks_stddev: The standard deviation to use for calculating object
            keypoint similarity; see `compute_oks` function for details.
        oks_scale: The scale to use for calculating object
            keypoint similarity; see `compute_oks` function for details.
        match_threshold: The threshold to use on oks scores when determining
            which instances match between ground truth and predicted frames.
        user_labels_only: If False, predicted instances in the ground truth frame may be
            considered for matching.

    """

    def __init__(
        self,
        ground_truth_instances: sio.Labels,
        predicted_instances: sio.Labels,
        oks_stddev: float = 0.025,
        oks_scale: Optional[float] = None,
        match_threshold: float = 0,
        user_labels_only: bool = True,
    ):
        """Initialize the Evaluator class with ground-truth and predicted labels."""
        self.ground_truth_instances = ground_truth_instances
        self.predicted_instances = predicted_instances
        self.match_threshold = match_threshold
        self.oks_stddev = oks_stddev
        self.oks_scale = oks_scale
        self.user_labels_only = user_labels_only

        self._process_frames()

    def _process_frames(self):
        self.frame_pairs = find_frame_pairs(
            self.ground_truth_instances, self.predicted_instances, self.user_labels_only
        )
        if not self.frame_pairs:
            message = "Empty Frame Pairs. No match found for the video frames"
            logger.error(message)
            raise Exception(message)

        self.positive_pairs, self.false_negatives = match_frame_pairs(
            self.frame_pairs,
            stddev=self.oks_stddev,
            scale=self.oks_scale,
            threshold=self.match_threshold,
        )

        self.dists_dict = compute_dists(self.positive_pairs)

    def voc_metrics(
        self,
        match_score_by="oks",
        match_score_thresholds: np.ndarray = np.linspace(
            0.5, 0.95, 10
        ),  # 0.5:0.05:0.95
        recall_thresholds: np.ndarray = np.linspace(0, 1, 101),  # 0.0:0.01:1.00
    ):
        """Compute VOC metrics for a matched pairs of instances positive pairs and false negatives.

        Args:
            match_score_by: The score to be used for computing the metrics. "ock" or "pck"
            match_score_thresholds: Score thresholds at which to consider matches as a true
                positive match.
            recall_thresholds: Recall thresholds at which to evaluate Average Precision.

        Returns:
            A dictionary of VOC metrics.
        """
        if match_score_by == "oks":
            match_scores = np.array([oks for _, _, oks in self.positive_pairs])
            name = "oks_voc"
        elif match_score_by == "pck":
            pck_metrics = self.pck_metrics()
            match_scores = pck_metrics["pcks"].mean(axis=-1).mean(axis=-1)
            name = "pck_voc"
        else:
            message = "Invalid Option for match_score_by. Choose either `oks` or `pck`"
            logger.error(message)
            raise Exception(message)

        detection_scores = np.array(
            [pp[1].instance.score for pp in self.positive_pairs]
        )

        inds = np.argsort(-detection_scores, kind="mergesort")
        detection_scores = detection_scores[inds]
        match_scores = match_scores[inds]

        precisions = []
        recalls = []

        npig = len(self.positive_pairs) + len(
            self.false_negatives
        )  # total number of GT instances

        for match_score_threshold in match_score_thresholds:
            tp = np.cumsum(match_scores >= match_score_threshold)
            fp = np.cumsum(match_scores < match_score_threshold)

            if tp.size == 0:
                return {
                    name + ".match_score_thresholds": 0,
                    name + ".recall_thresholds": 0,
                    name + ".match_scores": 0,
                    name + ".precisions": 0,
                    name + ".recalls": 0,
                    name + ".AP": 0,
                    name + ".AR": 0,
                    name + ".mAP": 0,
                    name + ".mAR": 0,
                }

            rc = tp / npig
            pr = tp / (fp + tp + np.spacing(1))

            recall = rc[-1]  # best recall at this OKS threshold

            # Ensure strictly decreasing precisions.
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            # Find best precision at each recall threshold.
            rc_inds = np.searchsorted(rc, recall_thresholds, side="left")
            precision = np.zeros(rc_inds.shape)
            is_valid_rc_ind = rc_inds < len(pr)
            precision[is_valid_rc_ind] = pr[rc_inds[is_valid_rc_ind]]

            precisions.append(precision)
            recalls.append(recall)

        precisions = np.array(precisions)
        recalls = np.array(recalls)

        AP = precisions.mean(
            axis=1
        )  # AP = average precision over fixed set of recall thresholds
        AR = recalls  # AR = max recall given a fixed number of detections per image

        mAP = precisions.mean()  # mAP = mean over all OKS thresholds
        mAR = recalls.mean()  # mAR = mean over all OKS thresholds

        return {
            name + ".match_score_thresholds": match_score_thresholds,
            name + ".recall_thresholds": recall_thresholds,
            name + ".match_scores": match_scores,
            name + ".precisions": precisions,
            name + ".recalls": recalls,
            name + ".AP": AP,
            name + ".AR": AR,
            name + ".mAP": mAP,
            name + ".mAR": mAR,
        }

    def mOKS(self):
        """Return the meanOKS value."""
        pair_oks = np.array([oks for _, _, oks in self.positive_pairs])
        return {"mOKS": pair_oks.mean()}

    def distance_metrics(self):
        """Compute the Euclidean distance error at different percentiles using the pairwise distances.

        Returns:
            A dictionary of distance metrics.
        """
        dists = self.dists_dict["dists"]
        results = {
            "frame_idxs": self.dists_dict["frame_idxs"],
            "video_paths": self.dists_dict["video_paths"],
            "dists": dists,
            "avg": np.nanmean(dists),
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

        is_non_nan = ~np.isnan(dists)
        if np.any(is_non_nan):
            non_nans = dists[is_non_nan]
            for ptile in (50, 75, 90, 95, 99):
                results[f"p{ptile}"] = np.percentile(non_nans, ptile)

        return results

    def pck_metrics(self, thresholds: np.ndarray = np.linspace(1, 10, 10)):
        """Compute PCK across a range of thresholds using the pair-wise distances.

        Args:
            thresholds: A list of distance thresholds in pixels.

        Returns:
            A dictionary of PCK metrics evaluated at each threshold.
        """
        dists = self.dists_dict["dists"]
        dists = np.copy(dists)
        dists[np.isnan(dists)] = np.inf
        pcks = np.expand_dims(dists, -1) < np.reshape(thresholds, (1, 1, -1))
        mPCK_parts = pcks.mean(axis=0).mean(axis=-1)
        mPCK = mPCK_parts.mean()

        # Precompute PCK at common thresholds
        idx_5 = np.argmin(np.abs(thresholds - 5))
        idx_10 = np.argmin(np.abs(thresholds - 10))
        pck5 = pcks[:, :, idx_5].mean()
        pck10 = pcks[:, :, idx_10].mean()

        return {
            "thresholds": thresholds,
            "pcks": pcks,
            "mPCK_parts": mPCK_parts,
            "mPCK": mPCK,
            "PCK@5": pck5,
            "PCK@10": pck10,
        }

    def visibility_metrics(self):
        """Compute node visibility metrics for the matched pair of instances.

        Returns:
            A dictionary of visibility metrics, including the confusion matrix.
        """
        vis_tp = 0
        vis_fn = 0
        vis_fp = 0
        vis_tn = 0

        for instance_gt, instance_pr, _ in self.positive_pairs:
            missing_nodes_gt = np.isnan(instance_gt.instance.numpy()).any(axis=-1)
            missing_nodes_pr = np.isnan(instance_pr.instance.numpy()).any(axis=-1)

            vis_tn += ((missing_nodes_gt) & (missing_nodes_pr)).sum()
            vis_fn += ((~missing_nodes_gt) & (missing_nodes_pr)).sum()
            vis_fp += ((missing_nodes_gt) & (~missing_nodes_pr)).sum()
            vis_tp += ((~missing_nodes_gt) & (~missing_nodes_pr)).sum()

        return {
            "tp": vis_tp,
            "fp": vis_fp,
            "tn": vis_tn,
            "fn": vis_fn,
            "precision": vis_tp / (vis_tp + vis_fp) if (vis_tp + vis_fp) else np.nan,
            "recall": vis_tp / (vis_tp + vis_fn) if (vis_tp + vis_fn) else np.nan,
        }

    def evaluate(self):
        """Return the evaluation metrics."""
        metrics = {}
        metrics["voc_metrics"] = self.voc_metrics(match_score_by="oks")
        metrics["voc_metrics"].update(self.voc_metrics(match_score_by="pck"))
        metrics["mOKS"] = self.mOKS()
        metrics["distance_metrics"] = self.distance_metrics()
        metrics["pck_metrics"] = self.pck_metrics()
        metrics["visibility_metrics"] = self.visibility_metrics()

        return metrics


def _find_metrics_file(model_dir: Path, split: str, dataset_idx: int) -> Path:
    """Find the metrics file in a model directory.

    Tries new naming format first, then falls back to old format.
    If split is "test" and not found, falls back to "val".
    """
    # Try new naming format first: metrics.{split}.{idx}.npz
    metrics_path = model_dir / f"metrics.{split}.{dataset_idx}.npz"
    if metrics_path.exists():
        return metrics_path

    # Fall back to old naming format: {split}_{idx}_pred_metrics.npz
    metrics_path = model_dir / f"{split}_{dataset_idx}_pred_metrics.npz"
    if metrics_path.exists():
        return metrics_path

    # If split is "test" and not found, try "val" fallback
    if split == "test":
        return _find_metrics_file(model_dir, "val", dataset_idx)

    # Return the new format path (will raise FileNotFoundError later)
    return model_dir / f"metrics.{split}.{dataset_idx}.npz"


def _load_npz_metrics(metrics_path: Path) -> dict:
    """Load metrics from an npz file, supporting both old and new formats.

    New format: single "metrics" key containing a dict with all metrics.
    Old format: individual metric keys at top level (voc_metrics, mOKS, etc.).
    """
    with np.load(metrics_path, allow_pickle=True) as data:
        keys = list(data.keys())

        # New format: single "metrics" key containing dict
        if "metrics" in keys:
            return data["metrics"].item()

        # Old format: individual metric keys at top level
        expected_keys = {
            "voc_metrics",
            "mOKS",
            "distance_metrics",
            "pck_metrics",
            "visibility_metrics",
        }
        if expected_keys.issubset(set(keys)):
            return {
                k: data[k].item() if data[k].ndim == 0 else data[k]
                for k in expected_keys
            }

        # Unknown format - return all keys as dict
        return {k: data[k].item() if data[k].ndim == 0 else data[k] for k in keys}


def load_metrics(
    path: str,
    split: str = "test",
    dataset_idx: int = 0,
) -> dict:
    """Load metrics from a model folder or metrics file.

    This function supports both the new format (single "metrics" key) and the old
    format (individual metric keys at top level). It also handles both old and new
    file naming conventions in model folders.

    Args:
        path: Path to a model folder or metrics file (.npz).
        split: Name of the split to load. Must be "train", "val", or "test".
            Default: "test". If "test" is not found, falls back to "val".
            Ignored if path points directly to a .npz file.
        dataset_idx: Index of the dataset (for multi-dataset training).
            Default: 0. Ignored if path points directly to a .npz file.

    Returns:
        Dictionary containing metrics with keys: voc_metrics, mOKS,
        distance_metrics, pck_metrics, visibility_metrics.

    Raises:
        FileNotFoundError: If no metrics file is found.

    Examples:
        >>> # Load from model folder (tries test, falls back to val)
        >>> metrics = load_metrics("/path/to/model")
        >>> print(metrics["mOKS"]["mOKS"])

        >>> # Load specific split and dataset
        >>> metrics = load_metrics("/path/to/model", split="val", dataset_idx=1)

        >>> # Load directly from npz file
        >>> metrics = load_metrics("/path/to/metrics.val.0.npz")
    """
    path = Path(path)

    if path.suffix == ".npz":
        metrics_path = path
    else:
        metrics_path = _find_metrics_file(path, split, dataset_idx)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    return _load_npz_metrics(metrics_path)


def run_evaluation(
    ground_truth_path: str,
    predicted_path: str,
    oks_stddev: float = 0.025,
    oks_scale: Optional[float] = None,
    match_threshold: float = 0,
    user_labels_only: bool = True,
    save_metrics: Optional[str] = None,
):
    """Evaluate SLEAP-NN model predictions against ground truth labels."""
    logger.info("Loading ground truth labels...")
    ground_truth_instances = sio.load_slp(ground_truth_path)
    logger.info(
        f"  Ground truth: {len(ground_truth_instances.videos)} videos, "
        f"{len(ground_truth_instances.labeled_frames)} frames"
    )

    logger.info("Loading predicted labels...")
    predicted_instances = sio.load_slp(predicted_path)
    logger.info(
        f"  Predictions: {len(predicted_instances.videos)} videos, "
        f"{len(predicted_instances.labeled_frames)} frames"
    )

    logger.info("Matching videos and frames...")
    # Get match stats before creating evaluator
    match_result = ground_truth_instances.match(predicted_instances)
    logger.info(
        f"  Videos matched: {match_result.n_videos_matched}/{len(match_result.video_map)}"
    )

    logger.info("Matching instances...")
    evaluator = Evaluator(
        ground_truth_instances=ground_truth_instances,
        predicted_instances=predicted_instances,
        oks_stddev=oks_stddev,
        oks_scale=oks_scale,
        match_threshold=match_threshold,
        user_labels_only=user_labels_only,
    )
    logger.info(
        f"  Frame pairs: {len(evaluator.frame_pairs)}, "
        f"Matched instances: {len(evaluator.positive_pairs)}, "
        f"Unmatched GT: {len(evaluator.false_negatives)}"
    )

    logger.info("Computing evaluation metrics...")
    metrics = evaluator.evaluate()

    # Compute PCK at specific thresholds (5 and 10 pixels)
    dists = metrics["distance_metrics"]["dists"]
    dists_clean = np.copy(dists)
    dists_clean[np.isnan(dists_clean)] = np.inf
    pck_5 = (dists_clean < 5).mean()
    pck_10 = (dists_clean < 10).mean()

    # Print key metrics
    logger.info("Evaluation Results:")
    logger.info(f"  mOKS: {metrics['mOKS']['mOKS']:.4f}")
    logger.info(f"  mAP (OKS VOC): {metrics['voc_metrics']['oks_voc.mAP']:.4f}")
    logger.info(f"  mAR (OKS VOC): {metrics['voc_metrics']['oks_voc.mAR']:.4f}")
    logger.info(f"  Average Distance: {metrics['distance_metrics']['avg']:.2f} px")
    logger.info(f"  dist.p50: {metrics['distance_metrics']['p50']:.2f} px")
    logger.info(f"  dist.p95: {metrics['distance_metrics']['p95']:.2f} px")
    logger.info(f"  dist.p99: {metrics['distance_metrics']['p99']:.2f} px")
    logger.info(f"  mPCK: {metrics['pck_metrics']['mPCK']:.4f}")
    logger.info(f"  PCK@5px: {pck_5:.4f}")
    logger.info(f"  PCK@10px: {pck_10:.4f}")
    logger.info(
        f"  Visibility Precision: {metrics['visibility_metrics']['precision']:.4f}"
    )
    logger.info(f"  Visibility Recall: {metrics['visibility_metrics']['recall']:.4f}")

    # Save metrics if path provided
    if save_metrics:
        logger.info(f"Saving metrics to {save_metrics}...")
        save_path = Path(save_metrics)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metrics in SLEAP 1.4 format (single "metrics" key)
        np.savez_compressed(save_path, **{"metrics": metrics})
        logger.info(f"Metrics saved successfully to {save_path}")

    return metrics
