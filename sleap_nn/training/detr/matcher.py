"""Hungarian set matcher for DETR-style detectors, with Group DETR support.

Port of the ``rfdetr`` matcher (``models/matcher.py`` @ d7d7e4c) restricted to box
detection. The cost of assigning query ``q`` to target ``t`` is::

    cost_class * focal(q, label_t) + cost_bbox * L1(box_q, box_t) - cost_giou * GIoU(box_q, box_t)

where ``focal`` is the focal-loss classification cost at the target's class
(``alpha=0.25``, ``gamma=2``), L1 is on normalized ``cxcywh`` boxes and GIoU on the
corresponding ``xyxy`` boxes. The cost tensor is built for the whole batch at once
(targets padded to the largest image) and then solved with scipy's
``linear_sum_assignment`` per image and per query group.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from scipy.optimize import linear_sum_assignment
from torch import nn

from sleap_nn.training.detr.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

# Added on top of ``max + max(|cost|)`` so the sentinel is strictly larger than every
# finite cost, even when all finite costs are negative.
_SENTINEL_MARGIN = 1.0

MatchIndices = List[Tuple[torch.Tensor, torch.Tensor]]


def focal_classification_cost(
    logits: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Focal-loss matching cost of logits already gathered at the target classes.

    This is the positive focal term minus the negative one, written with
    ``logsigmoid`` (as in ``rfdetr``) instead of ``log(p + 1e-8)``.

    Args:
        logits: Logits at each target's class, any shape.
        alpha: Focal-loss alpha.
        gamma: Focal-loss gamma.

    Returns:
        The cost, same shape as ``logits``. Lower is a better match.
    """
    prob = logits.sigmoid()
    neg_cost = (1 - alpha) * prob**gamma * -F.logsigmoid(-logits)
    pos_cost = alpha * (1 - prob) ** gamma * -F.logsigmoid(logits)
    return pos_cost - neg_cost


def sanitize_cost_matrix(cost: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Replace non-finite costs with a finite sentinel larger than every valid cost.

    A query or target with a NaN/inf cost (e.g. a diverged box) is then only
    assigned when nothing finite is left, instead of crashing the solver.

    Args:
        cost: Cost tensor of any shape.
        valid: Bool mask, broadcastable to ``cost``, of the entries that are solved
            (padding is excluded from the sentinel statistics).

    Returns:
        ``cost`` with every non-finite entry replaced (a copy if anything changed).
    """
    finite = torch.isfinite(cost)
    if finite.all():
        return cost
    dtype_max = torch.finfo(cost.dtype).max
    finite_valid = cost[finite & valid]
    if finite_valid.numel() == 0:
        sentinel = cost.new_tensor(dtype_max)
    else:
        sentinel = (
            finite_valid.max() + finite_valid.abs().max() + _SENTINEL_MARGIN
        ).clamp(max=dtype_max)
        if not torch.isfinite(sentinel):
            sentinel = cost.new_tensor(dtype_max)
    return torch.where(finite, cost, sentinel)


class HungarianMatcher(nn.Module):
    """One-to-one assignment between predictions and targets, per query group.

    Attributes:
        cost_class: Weight of the focal classification cost.
        cost_bbox: Weight of the L1 box cost.
        cost_giou: Weight of the negative-GIoU cost.
        focal_alpha: Focal-loss alpha for the classification cost.
        focal_gamma: Focal-loss gamma for the classification cost.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """Initialize the matcher.

        Args:
            cost_class: Weight of the focal classification cost.
            cost_bbox: Weight of the L1 box cost.
            cost_giou: Weight of the negative-GIoU cost.
            focal_alpha: Focal-loss alpha for the classification cost.
            focal_gamma: Focal-loss gamma for the classification cost.

        Raises:
            ValueError: If all three cost weights are 0.
        """
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("At least one matcher cost weight must be non-zero.")
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self._warned_non_finite = False

    def cost_matrix(
        self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the batched matching cost.

        Args:
            outputs: Dict with ``pred_logits`` ``(B, Q, C)`` and ``pred_boxes``
                ``(B, Q, 4)``.
            targets: ``B`` dicts with ``labels`` ``(N_i,)`` and ``boxes``
                ``(N_i, 4)``.

        Returns:
            A tuple ``(cost, valid)``: float32 cost ``(B, Q, T)`` with
            ``T = max_i N_i``, and bool ``(B, 1, T)`` marking real (unpadded)
            targets. Padded columns hold arbitrary values.
        """
        logits = outputs["pred_logits"].float()
        pred_boxes = outputs["pred_boxes"].float()
        batch_size, num_queries = logits.shape[:2]
        device = logits.device
        sizes = [len(t["labels"]) for t in targets]
        max_targets = max(sizes, default=0)

        labels = torch.zeros(batch_size, max_targets, dtype=torch.long, device=device)
        tgt_boxes = torch.zeros(batch_size, max_targets, 4, device=device)
        valid = torch.zeros(batch_size, 1, max_targets, dtype=torch.bool, device=device)
        for i, (t, n) in enumerate(zip(targets, sizes)):
            labels[i, :n] = t["labels"].to(device)
            tgt_boxes[i, :n] = t["boxes"].to(device=device, dtype=torch.float32)
            valid[i, 0, :n] = True

        tgt_logits = logits.gather(
            2, labels[:, None, :].expand(batch_size, num_queries, max_targets)
        )
        class_cost = focal_classification_cost(
            tgt_logits, self.focal_alpha, self.focal_gamma
        )
        bbox_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes)
        )
        cost = (
            self.cost_bbox * bbox_cost
            + self.cost_class * class_cost
            + self.cost_giou * giou_cost
        )
        return cost, valid

    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        group_detr: int = 1,
    ) -> MatchIndices:
        """Match predictions to targets.

        Args:
            outputs: Dict with ``pred_logits`` ``(B, Q, C)`` and ``pred_boxes``
                ``(B, Q, 4)``. Other keys are ignored.
            targets: ``B`` dicts with ``labels`` ``(N_i,)`` and ``boxes``
                ``(N_i, 4)``.
            group_detr: Number of query groups ``G``; ``Q`` must be divisible by it.
                Each group of ``Q / G`` consecutive queries is matched to all
                targets on its own.

        Returns:
            One ``(query_idx, target_idx)`` pair of int64 CPU tensors per image. The
            groups' matches are concatenated in group order, with query indices
            offset into the full ``Q`` range, so each pair has
            ``G * min(Q / G, N_i)`` entries.

        Raises:
            ValueError: If ``Q`` is not divisible by ``group_detr``.
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        if num_queries % group_detr != 0:
            raise ValueError(
                f"num_queries ({num_queries}) must be divisible by group_detr "
                f"({group_detr})."
            )
        if len(targets) != batch_size:
            raise ValueError(
                f"Got {len(targets)} targets for a batch of {batch_size} images."
            )
        empty = torch.zeros(0, dtype=torch.int64)
        sizes = [len(t["labels"]) for t in targets]
        if sum(sizes) == 0:
            return [(empty, empty) for _ in targets]

        cost, valid = self.cost_matrix(outputs, targets)
        if not torch.isfinite(cost.masked_select(valid)).all():
            if not self._warned_non_finite:
                logger.warning(
                    "Non-finite values in the DETR matcher cost matrix; replacing "
                    "them with a finite sentinel. Check for numerical instability."
                )
                self._warned_non_finite = True
            cost = sanitize_cost_matrix(cost, valid)
        cost = cost.cpu().numpy()

        group_size = num_queries // group_detr
        indices = []
        for i, n in enumerate(sizes):
            if n == 0:
                indices.append((empty, empty))
                continue
            query_idx, target_idx = [], []
            for g in range(group_detr):
                start = g * group_size
                rows, cols = linear_sum_assignment(
                    cost[i, start : start + group_size, :n]
                )
                query_idx.append(rows + start)
                target_idx.append(cols)
            indices.append(
                (
                    torch.as_tensor(np.concatenate(query_idx), dtype=torch.int64),
                    torch.as_tensor(np.concatenate(target_idx), dtype=torch.int64),
                )
            )
        return indices
