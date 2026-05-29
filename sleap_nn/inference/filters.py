"""Post-inference filtering: ``FilterConfig`` + ``FilterPipeline``.

Single source of truth for every filter applied between an
``InferenceLayer``'s raw ``Outputs`` and the final ``sio.Labels``. The
legacy code spread these filters across
``Predictor._make_labeled_frames_from_generator`` (the per-frame loop
that called ``filter_overlapping_instances``, ``filter_by_node_count``,
``filter_by_node_confidence`` in different orders depending on the
model type). This module pulls them into one place so the order is
documented and predictable.

Tom's design-review comment (epic #508):

  > given that post processing would probably happen on another process
  > or even multiple process we might need to ensure that it's pickle-able

``FilterConfig`` is ``attrs.frozen`` (a value type — picklable). The
filter ops in ``ops.filters`` operate directly on ``Outputs`` tensors
without holding model / file handles, so the whole pipeline is safe to
hand to a worker pool (PR 9 / #517).

Filter order — fixed, documented, runs cheap → expensive:

1. ``min_peak_value`` — NaN-out individual keypoints below threshold
2. Node-count filters (``min_visible_nodes``, ``min_visible_node_fraction``)
3. Score filters (``min_instance_score``, ``min_mean_node_score``)
4. Overlap NMS (``overlapping`` + ``overlapping_method``) — the
   most expensive; runs on the smallest candidate set
"""

from __future__ import annotations

from typing import Literal

import attrs
import torch

from sleap_nn.inference.outputs import Outputs


@attrs.frozen
class FilterConfig:
    """Post-inference filter configuration (value type, picklable).

    All thresholds default to ``0`` / ``False`` so a default
    ``FilterConfig`` is the no-op identity. Set only the knobs you need.

    Attributes:
        min_peak_value: NaN-out per-keypoint scores below this threshold.
            ``0.0`` disables.
        min_instance_score: Drop instances whose ``instance_scores`` fall
            below this. ``0.0`` disables.
        min_mean_node_score: Drop instances whose mean visible-node score
            is below this. ``0.0`` disables.
        min_visible_nodes: Drop instances with fewer than this many
            non-NaN keypoints. ``0`` disables.
        min_visible_node_fraction: Drop instances whose visible-node
            fraction is below this (0.0 to 1.0). ``0.0`` disables.
        overlapping: When ``True``, run greedy overlap-NMS between
            instances per frame (most expensive filter; runs last).
        overlapping_threshold: Similarity threshold above which the
            lower-scoring overlap is dropped.
        overlapping_method: ``"iou"`` (bbox IoU) or ``"oks"`` (keypoint
            OKS) for the overlap similarity metric.
        min_centroid_distance: Centroid-only de-duplication radius in pixels.
            Greedy NMS drops any predicted centroid within this distance of a
            higher-scored kept centroid. ``0.0`` disables. Use this (not
            ``overlapping``) for centroid-only output, since bbox-IoU / OKS are
            degenerate for single points.
    """

    min_peak_value: float = 0.0
    min_instance_score: float = 0.0
    min_mean_node_score: float = 0.0
    min_visible_nodes: int = 0
    min_visible_node_fraction: float = 0.0
    overlapping: bool = False
    overlapping_threshold: float = 0.8
    overlapping_method: Literal["iou", "oks"] = "iou"
    min_centroid_distance: float = 0.0


@attrs.define
class FilterPipeline:
    """Apply a :class:`FilterConfig` to an :class:`Outputs` tensor.

    Order is fixed (cheap → expensive) so reasoning about the pipeline is
    deterministic. ``__call__`` is sugar for ``apply``.

    Args:
        config: The ``FilterConfig`` driving the pipeline.
    """

    config: FilterConfig

    def __call__(self, outputs: Outputs) -> Outputs:
        """Alias for :meth:`apply`."""
        return self.apply(outputs)

    def apply(self, outputs: Outputs) -> Outputs:
        """Run all configured filters in canonical cheap → expensive order."""
        cfg = self.config
        if cfg.min_peak_value > 0.0:
            outputs = self._filter_min_peak_value(outputs, cfg.min_peak_value)
        if cfg.min_visible_nodes > 0 or cfg.min_visible_node_fraction > 0.0:
            outputs = self._filter_node_count(
                outputs,
                min_visible=cfg.min_visible_nodes,
                min_fraction=cfg.min_visible_node_fraction,
            )
        if cfg.min_instance_score > 0.0 or cfg.min_mean_node_score > 0.0:
            outputs = self._filter_confidence(
                outputs,
                min_instance_score=cfg.min_instance_score,
                min_mean_node_score=cfg.min_mean_node_score,
            )
        if cfg.overlapping:
            method = cfg.overlapping_method
            if outputs.pred_keypoints is None and outputs.pred_centroids is not None:
                import warnings

                warnings.warn(
                    "overlapping NMS (iou/oks) is not meaningful for "
                    "centroid-only outputs (single points have degenerate "
                    "bbox-IoU / OKS); use FilterConfig.min_centroid_distance "
                    "for centroid de-duplication. Skipping overlap NMS.",
                    stacklevel=2,
                )
            else:
                outputs = self._filter_overlapping(
                    outputs,
                    threshold=cfg.overlapping_threshold,
                    method=method,
                )
        if cfg.min_centroid_distance > 0.0:
            outputs = self._filter_centroid_distance(outputs, cfg.min_centroid_distance)
        return outputs

    @classmethod
    def run(cls, outputs: Outputs, config: FilterConfig) -> Outputs:
        """One-off convenience: ``FilterPipeline(config)(outputs)``."""
        return cls(config=config)(outputs)

    # ──────────────────────────────────────────────────────────────────
    # Filter implementations — operate purely on Outputs tensor fields
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _filter_min_peak_value(outputs: Outputs, threshold: float) -> Outputs:
        """NaN-out per-keypoint coordinates whose ``pred_peak_values`` < threshold."""
        if outputs.pred_keypoints is None or outputs.pred_peak_values is None:
            return outputs
        mask = outputs.pred_peak_values < threshold  # (B, I, N)
        new_kpts = outputs.pred_keypoints.clone()
        new_kpts[mask] = float("nan")
        new_vals = outputs.pred_peak_values.clone()
        new_vals[mask] = float("nan")
        return attrs.evolve(outputs, pred_keypoints=new_kpts, pred_peak_values=new_vals)

    @staticmethod
    def _filter_node_count(
        outputs: Outputs,
        min_visible: int,
        min_fraction: float,
    ) -> Outputs:
        """Drop instances with too few visible keypoints (NaN out the slot)."""
        if outputs.pred_keypoints is None:
            return outputs
        n_nodes = outputs.pred_keypoints.shape[-2]
        # Visible iff both x and y are finite.
        visible = ~torch.isnan(outputs.pred_keypoints).any(dim=-1)  # (B, I, N)
        n_visible = visible.sum(dim=-1)  # (B, I)
        keep = torch.ones_like(n_visible, dtype=torch.bool)
        if min_visible > 0:
            keep &= n_visible >= min_visible
        if min_fraction > 0.0:
            keep &= (n_visible / max(n_nodes, 1)) >= min_fraction
        return FilterPipeline._nan_out_where(~keep, outputs)

    @staticmethod
    def _filter_confidence(
        outputs: Outputs,
        min_instance_score: float,
        min_mean_node_score: float,
    ) -> Outputs:
        """Drop instances by score (instance-level + mean-node-score)."""
        if outputs.pred_keypoints is None:
            # Centroid-only: gate on the per-instance centroid value (or
            # ``instance_scores`` when present). Mean-node-score is undefined
            # for a single point and is skipped.
            if outputs.pred_centroids is None or min_instance_score <= 0.0:
                return outputs
            scores = (
                outputs.instance_scores
                if outputs.instance_scores is not None
                else outputs.pred_centroid_values
            )
            if scores is None:
                return outputs
            # NaN scores (empty slots) compare False, so they are not dropped
            # here; they are already NaN centroids and ignored at packaging.
            # NaN scores are not a valid pass of a positive gate; drop them too
            # (a valid centroid always carries a finite value upstream).
            drop = (scores < min_instance_score) | torch.isnan(scores)  # (B, I)
            return FilterPipeline._nan_out_where(drop, outputs)
        keep = torch.ones(
            outputs.pred_keypoints.shape[:2],
            dtype=torch.bool,
            device=outputs.pred_keypoints.device,
        )  # (B, I)
        if min_instance_score > 0.0 and outputs.instance_scores is not None:
            keep &= outputs.instance_scores >= min_instance_score
        if min_mean_node_score > 0.0 and outputs.pred_peak_values is not None:
            mean_score = torch.nanmean(outputs.pred_peak_values, dim=-1)  # (B, I)
            # ``nanmean`` returns NaN for all-NaN slots; treat those as failing.
            mean_score = torch.where(
                torch.isnan(mean_score), torch.zeros_like(mean_score), mean_score
            )
            keep &= mean_score >= min_mean_node_score
        return FilterPipeline._nan_out_where(~keep, outputs)

    @staticmethod
    def _filter_overlapping(
        outputs: Outputs,
        threshold: float,
        method: Literal["iou", "oks"],
    ) -> Outputs:
        """Greedy overlap-NMS between instances per frame.

        For each frame, sort instances by score; greedily keep instances
        whose similarity (IoU on bbox or OKS on keypoints) with any
        already-kept instance is below ``threshold``.
        """
        if outputs.pred_keypoints is None:
            return outputs
        B, I, _N, _ = outputs.pred_keypoints.shape
        dev = outputs.pred_keypoints.device
        scores = (
            outputs.instance_scores
            if outputs.instance_scores is not None
            else torch.zeros(B, I, device=dev)
        )
        keep_mask = torch.ones(B, I, dtype=torch.bool, device=dev)
        for b in range(B):
            valid_b = ~torch.isnan(outputs.pred_keypoints[b]).all(dim=-1).all(dim=-1)
            if valid_b.sum() <= 1:
                continue
            order = scores[b].argsort(descending=True)
            kept_idx: list[int] = []
            for idx in order.tolist():
                if not valid_b[idx]:
                    continue
                inst = outputs.pred_keypoints[b, idx]  # (N, 2)
                drop = False
                for k in kept_idx:
                    sim = (
                        FilterPipeline._oks(inst, outputs.pred_keypoints[b, k])
                        if method == "oks"
                        else FilterPipeline._bbox_iou(
                            inst, outputs.pred_keypoints[b, k]
                        )
                    )
                    if sim > threshold:
                        drop = True
                        break
                if drop:
                    keep_mask[b, idx] = False
                else:
                    kept_idx.append(idx)
        return FilterPipeline._nan_out_where(~keep_mask, outputs)

    @staticmethod
    def _bbox_iou(a: torch.Tensor, b: torch.Tensor) -> float:
        """IoU of the bboxes implied by two keypoint sets (NaN-aware)."""
        a_xy = a[~torch.isnan(a).any(dim=-1)]
        b_xy = b[~torch.isnan(b).any(dim=-1)]
        if a_xy.numel() == 0 or b_xy.numel() == 0:
            return 0.0
        ax1, ay1 = a_xy.min(dim=0).values
        ax2, ay2 = a_xy.max(dim=0).values
        bx1, by1 = b_xy.min(dim=0).values
        bx2, by2 = b_xy.max(dim=0).values
        inter_w = (torch.minimum(ax2, bx2) - torch.maximum(ax1, bx1)).clamp(min=0)
        inter_h = (torch.minimum(ay2, by2) - torch.maximum(ay1, by1)).clamp(min=0)
        inter = (inter_w * inter_h).item()
        area_a = ((ax2 - ax1) * (ay2 - ay1)).item()
        area_b = ((bx2 - bx1) * (by2 - by1)).item()
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _oks(a: torch.Tensor, b: torch.Tensor, kappa: float = 0.1) -> float:
        """Object-Keypoint Similarity between two keypoint sets (NaN-aware).

        Bit-for-bit port of legacy ``_compute_oks`` in
        ``sleap_nn/inference/postprocessing.py`` (on ``main``): the per-keypoint
        scale is the bounding-box *area* of instance ``a`` (its own valid
        keypoints), the falloff constant is ``kappa=0.1``, and the result is the
        mean over keypoints visible in *both* instances of
        ``exp(-d^2 / (2 * scale_area * kappa^2))``.
        """
        valid_a = ~torch.isnan(a).any(dim=-1)
        valid_b = ~torch.isnan(b).any(dim=-1)
        valid = valid_a & valid_b
        if valid.sum() == 0:
            return 0.0
        # Scale from the bbox *area* of instance ``a``'s own valid keypoints.
        a_xy = a[valid_a]
        if a_xy.shape[0] < 2:
            return 0.0
        mins = a_xy.min(dim=0).values
        maxs = a_xy.max(dim=0).values
        bbox_w = maxs[0] - mins[0]
        bbox_h = maxs[1] - mins[1]
        scale_sq = bbox_w * bbox_h
        if scale_sq.item() <= 0:
            return 0.0
        d2 = ((a[valid] - b[valid]) ** 2).sum(dim=-1)
        oks_per_kpt = torch.exp(-d2 / (2 * scale_sq * kappa**2))
        return oks_per_kpt.mean().item()

    @staticmethod
    def _nan_out_where(drop_mask: torch.Tensor, outputs: Outputs) -> Outputs:
        """NaN-out instances where ``drop_mask`` is True. Preserves shape."""
        kwargs: dict = {}
        if outputs.pred_keypoints is not None:
            kpts = outputs.pred_keypoints.clone()
            kpts[drop_mask] = float("nan")
            kwargs["pred_keypoints"] = kpts
        if outputs.pred_peak_values is not None:
            vals = outputs.pred_peak_values.clone()
            vals[drop_mask] = float("nan")
            kwargs["pred_peak_values"] = vals
        if outputs.instance_scores is not None:
            scores = outputs.instance_scores.clone()
            scores[drop_mask] = float("nan")
            kwargs["instance_scores"] = scores
        # Centroid-only fields: dropped slots must vanish here too, or they
        # reappear at packaging (to_instances/to_centroids skip only NaN slots).
        if outputs.pred_centroids is not None:
            cen = outputs.pred_centroids.clone()
            cen[drop_mask] = float("nan")
            kwargs["pred_centroids"] = cen
        if outputs.pred_centroid_values is not None:
            cvals = outputs.pred_centroid_values.clone()
            cvals[drop_mask] = float("nan")
            kwargs["pred_centroid_values"] = cvals
        return attrs.evolve(outputs, **kwargs)

    @staticmethod
    def _filter_centroid_distance(outputs: Outputs, min_distance: float) -> Outputs:
        """Greedy NMS on predicted centroids by Euclidean distance.

        For each frame, sort centroids by score (``pred_centroid_values``, else
        ``instance_scores``) descending and drop any centroid within
        ``min_distance`` pixels of an already-kept (higher-scored) centroid.
        No-op when there are no centroids.
        """
        if outputs.pred_centroids is None:
            return outputs
        B, I, _ = outputs.pred_centroids.shape
        dev = outputs.pred_centroids.device
        if outputs.pred_centroid_values is not None:
            scores = outputs.pred_centroid_values
        elif outputs.instance_scores is not None:
            scores = outputs.instance_scores
        else:
            scores = torch.zeros(B, I, device=dev)
        thresh_sq = float(min_distance) ** 2
        drop_mask = torch.zeros(B, I, dtype=torch.bool, device=dev)
        for b in range(B):
            cen_b = outputs.pred_centroids[b]  # (I, 2)
            valid_b = ~torch.isnan(cen_b).any(dim=-1)  # (I,)
            if valid_b.sum() <= 1:
                continue
            # Sort by score desc; NaN scores sort last (treated as -inf).
            sb = scores[b].clone()
            sb = torch.where(torch.isnan(sb), torch.full_like(sb, float("-inf")), sb)
            order = sb.argsort(descending=True)
            kept: list[int] = []
            for idx in order.tolist():
                if not valid_b[idx]:
                    continue
                p = cen_b[idx]
                too_close = False
                for k in kept:
                    if ((p - cen_b[k]) ** 2).sum().item() < thresh_sq:
                        too_close = True
                        break
                if too_close:
                    drop_mask[b, idx] = True
                else:
                    kept.append(idx)
        return FilterPipeline._nan_out_where(drop_mask, outputs)
