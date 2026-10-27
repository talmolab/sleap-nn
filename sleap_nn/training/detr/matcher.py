"""Hungarian set matcher for DETR-style detectors, with Group DETR support.

Port of the ``rfdetr`` matcher (``models/matcher.py`` @ d7d7e4c) restricted to box
detection. The cost of assigning query ``q`` to target ``t`` is::

    cost_class * focal(q, label_t) + cost_bbox * L1(box_q, box_t) - cost_giou * GIoU(box_q, box_t)

where ``focal`` is the focal-loss classification cost at the target's class
(``alpha=0.25``, ``gamma=2``), L1 is on normalized ``cxcywh`` boxes and GIoU on the
corresponding ``xyxy`` boxes. The cost tensor is built for the whole batch at once
(targets padded to the largest image) and then solved with scipy's
``linear_sum_assignment`` per image and per query group.

:meth:`HungarianMatcher.match_layers` matches several layers (the decoder layers and
the encoder proposals) against the same targets with one device-to-host copy: the
targets are gathered once, every layer's cost is built on the device for real
(unpadded) targets only, and all costs come to the host together. The matches
are the same as matching each layer on its own.
"""

from typing import Dict, List, Sequence, Tuple

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
        # The clamp also maps an overflow to inf back to the largest finite value.
        sentinel = (
            finite_valid.max() + finite_valid.abs().max() + _SENTINEL_MARGIN
        ).clamp(max=dtype_max)
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
        device = outputs["pred_logits"].device
        sizes = [len(t["labels"]) for t in targets]
        max_targets = max(sizes, default=0)
        batch_size = len(targets)
        labels = torch.zeros(batch_size, max_targets, dtype=torch.long, device=device)
        tgt_boxes = torch.zeros(batch_size, max_targets, 4, device=device)
        valid = torch.zeros(batch_size, 1, max_targets, dtype=torch.bool, device=device)
        for i, (t, n) in enumerate(zip(targets, sizes)):
            labels[i, :n] = t["labels"].to(device)
            tgt_boxes[i, :n] = t["boxes"].to(device=device, dtype=torch.float32)
            valid[i, 0, :n] = True
        return self._cost(outputs, labels, tgt_boxes), valid

    def _cost(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        tgt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """The ``(B, Q, T)`` float32 cost against padded targets (``cost_matrix``)."""
        logits = outputs["pred_logits"].float()
        batch_size, num_queries = logits.shape[:2]
        tgt_logits = logits.gather(
            2, labels[:, None, :].expand(batch_size, num_queries, labels.shape[1])
        )
        return self._pair_cost(tgt_logits, outputs["pred_boxes"].float(), tgt_boxes)

    def _compact_cost(
        self,
        outputs: Dict[str, torch.Tensor],
        image_idx: torch.Tensor,
        labels: torch.Tensor,
        tgt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """The cost of every real target against its image's queries, ``(N, Q)``.

        Row ``r`` is target ``r`` (of image ``image_idx[r]``, class ``labels[r]``,
        box ``tgt_boxes[r]``). Each entry is computed by the same per-pair ops as
        in :meth:`_cost`, so it equals the padded cost's entry bit for bit, but no
        work is spent on padding.
        """
        logits = outputs["pred_logits"].float()
        tgt_logits = logits[image_idx, :, labels][..., None]
        pred_boxes = outputs["pred_boxes"].float()[image_idx]
        return self._pair_cost(tgt_logits, pred_boxes, tgt_boxes[:, None])[..., 0]

    def _pair_cost(
        self,
        tgt_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        tgt_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted matching cost ``(..., Q, T)``.

        Args:
            tgt_logits: Logits at each target's class ``(..., Q, T)``.
            pred_boxes: Predicted ``cxcywh`` boxes ``(..., Q, 4)``.
            tgt_boxes: Target ``cxcywh`` boxes ``(..., T, 4)``.
        """
        class_cost = focal_classification_cost(
            tgt_logits, self.focal_alpha, self.focal_gamma
        )
        bbox_cost = torch.cdist(pred_boxes, tgt_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes)
        )
        return (
            self.cost_bbox * bbox_cost
            + self.cost_class * class_cost
            + self.cost_giou * giou_cost
        )

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
        return self.match_layers([outputs], targets, group_detr=group_detr)[0]

    @torch.no_grad()
    def match_layers(
        self,
        layers: Sequence[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        group_detr: int = 1,
    ) -> List[MatchIndices]:
        """Match several layers' predictions to the same targets.

        Equivalent to ``[self(layer, targets, group_detr) for layer in layers]``,
        but the costs are built for real targets only (no padding), in one pass
        when the layers have the same shapes, and all layers' costs are copied to
        the host in one transfer (one host sync per call instead of three per
        layer).

        Args:
            layers: Output dicts (see :meth:`forward`), all for the same ``B``
                images; ``Q`` may differ between layers.
            targets: ``B`` dicts with ``labels`` ``(N_i,)`` and ``boxes``
                ``(N_i, 4)``.
            group_detr: Number of query groups, as in :meth:`forward`.

        Returns:
            One :meth:`forward` result per layer, in order.

        Raises:
            ValueError: If a layer's ``Q`` is not divisible by ``group_detr`` or its
                batch size differs from ``len(targets)``.
        """
        for layer in layers:
            batch_size, num_queries = layer["pred_logits"].shape[:2]
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
        if sum(sizes) == 0 or not layers:
            return [[(empty, empty) for _ in targets] for _ in layers]

        device = layers[0]["pred_logits"].device
        # Every real target in image order: row `offsets[i] + j` of a layer's
        # compact cost is target `j` of image `i` against every query.
        image_idx = torch.cat(
            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(sizes)]
        ).to(device)
        labels = torch.cat([t["labels"].to(device) for t in targets])
        tgt_boxes = torch.cat(
            [t["boxes"].to(device=device, dtype=torch.float32) for t in targets]
        )
        shapes = {
            (layer["pred_logits"].shape, layer["pred_boxes"].shape) for layer in layers
        }
        if len(shapes) == 1 and len(layers) > 1:
            # One cost pass over all layers stacked on the batch axis (fewer
            # kernel launches; each entry is still the same per-pair value).
            stacked = {
                k: torch.cat([layer[k] for layer in layers])
                for k in ("pred_logits", "pred_boxes")
            }
            batch_size = len(targets)
            costs = self._compact_cost(
                stacked,
                torch.cat([image_idx + i * batch_size for i in range(len(layers))]),
                labels.repeat(len(layers)),
                tgt_boxes.repeat(len(layers), 1),
            ).chunk(len(layers))
        else:
            costs = [
                self._compact_cost(layer, image_idx, labels, tgt_boxes)
                for layer in layers
            ]
        host = _to_host(torch.cat([c.reshape(-1) for c in costs]))
        offsets = np.cumsum([0] + sizes)
        return [
            self._solve(
                self._guard_non_finite(chunk.view(c.shape)).numpy(),
                offsets,
                group_detr,
            )
            for chunk, c in zip(host.split([c.numel() for c in costs]), costs)
        ]

    def _guard_non_finite(self, cost: torch.Tensor) -> torch.Tensor:
        """``cost`` with non-finite entries sanitized (warns once per matcher)."""
        if np.isfinite(cost.numpy()).all():
            return cost
        if not self._warned_non_finite:
            logger.warning(
                "Non-finite values in the DETR matcher cost matrix; replacing "
                "them with a finite sentinel. Check for numerical instability."
            )
            self._warned_non_finite = True
        return sanitize_cost_matrix(cost, torch.ones((), dtype=torch.bool))

    @staticmethod
    def _solve(cost: np.ndarray, offsets: np.ndarray, group_detr: int) -> MatchIndices:
        """Per-image, per-group ``linear_sum_assignment`` on a compact host cost.

        Args:
            cost: ``(sum_i N_i, Q)`` costs, image ``i``'s targets in rows
                ``offsets[i]:offsets[i + 1]``.
            offsets: ``B + 1`` row offsets.
            group_detr: Number of query groups.

        Returns:
            One ``(query_idx, target_idx)`` pair per image (see :meth:`forward`).
        """
        empty = torch.zeros(0, dtype=torch.int64)
        group_size = cost.shape[1] // group_detr
        indices = []
        for lo, hi in zip(offsets[:-1], offsets[1:]):
            if hi == lo:
                indices.append((empty, empty))
                continue
            image_cost = cost[lo:hi]
            query_idx, target_idx = [], []
            for start in range(0, group_detr * group_size, group_size):
                # The (queries, targets) view has the same values (and so the same
                # assignment, ties included) as the padded (B, Q, T) cost's block.
                rows, cols = linear_sum_assignment(
                    image_cost[:, start : start + group_size].T
                )
                query_idx.append(rows + start)
                target_idx.append(cols)
            indices.append(
                (
                    torch.from_numpy(np.concatenate(query_idx).astype(np.int64)),
                    torch.from_numpy(np.concatenate(target_idx).astype(np.int64)),
                )
            )
        return indices


def _to_host(x: torch.Tensor) -> torch.Tensor:
    """``x`` on the CPU; from CUDA through pinned memory (one stream sync)."""
    if not x.is_cuda:
        return x.cpu()
    host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    host.copy_(x, non_blocking=True)
    torch.cuda.current_stream(x.device).synchronize()
    return host
