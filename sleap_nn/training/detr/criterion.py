"""RF-DETR set criterion: IoU-aware BCE, L1 and GIoU over matched queries.

Port of ``rfdetr``'s ``SetCriterion`` (``models/criterion.py`` @ d7d7e4c) restricted
to box detection with ``ia_bce_loss=True``, the RF-DETR default. The final decoder
layer, every ``aux_outputs`` layer (loss keys suffixed ``_<i>``) and the two-stage
``enc_outputs`` (suffixed ``_enc``) are each matched separately and get the same
losses, normalized by one shared ``num_boxes``.

Loss terms live in a small registry (:data:`LOSS_TERMS`). A term is a function
``(criterion, outputs, targets, indices, num_boxes) -> {key: tensor}`` registered
with :func:`register_loss_term`; :class:`SetCriterion` runs the terms named in its
``losses`` on every matched layer. Keys listed in ``weight_dict`` count toward the
total loss; any other key (``cardinality_error``) is for logging only.
"""

import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from sleap_nn.training.detr.box_ops import (
    box_cxcywh_to_xyxy,
    elementwise_box_iou,
    elementwise_generalized_box_iou,
)
from sleap_nn.training.detr.matcher import HungarianMatcher, MatchIndices

LossTerm = Callable[
    ["SetCriterion", Dict, List[Dict[str, torch.Tensor]], MatchIndices, torch.Tensor],
    Dict[str, torch.Tensor],
]

#: Registered loss terms by name. See :func:`register_loss_term`.
LOSS_TERMS: Dict[str, LossTerm] = {}

#: Loss weights of the RF-DETR detection recipe (``cls_loss_coef``,
#: ``bbox_loss_coef``, ``giou_loss_coef``).
DEFAULT_WEIGHT_DICT = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}

# A layer suffix: `_enc` for the encoder proposals, `_<i>` for decoder layer i.
_LAYER_SUFFIX = re.compile(r"_(enc|\d+)$")


def register_loss_term(name: str) -> Callable[[LossTerm], LossTerm]:
    """Register a loss term under ``name`` for use in ``SetCriterion(losses=...)``.

    Args:
        name: Name the term is requested by.

    Returns:
        A decorator that registers the term and returns it unchanged.

    Raises:
        ValueError: If ``name`` is already registered.
    """

    def decorator(fn: LossTerm) -> LossTerm:
        if name in LOSS_TERMS:
            raise ValueError(f"DETR loss term '{name}' is already registered.")
        LOSS_TERMS[name] = fn
        return fn

    return decorator


def _matched_query_index(indices: MatchIndices) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batch and query index of every match, to index ``(B, Q, ...)`` outputs."""
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    query_idx = torch.cat([src for src, _ in indices])
    return batch_idx, query_idx


def _matched_targets(
    targets: List[Dict[str, torch.Tensor]], indices: MatchIndices, key: str
) -> torch.Tensor:
    """Target ``key`` values in match order."""
    return torch.cat([t[key][tgt] for t, (_, tgt) in zip(targets, indices)])


@register_loss_term("labels")
def loss_labels(
    criterion: "SetCriterion",
    outputs: Dict,
    targets: List[Dict[str, torch.Tensor]],
    indices: MatchIndices,
    num_boxes: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """IoU-aware BCE classification loss (``loss_ce``).

    At each matched (query, target class) the soft target is
    ``t = clamp(p^alpha * IoU^(1 - alpha), 0.01)`` (detached), with positive weight
    ``t`` and negative weight ``1 - t``. Every other (query, class) entry has
    positive weight 0 and negative weight ``p^gamma``. The weighted BCE is summed
    over queries and classes and divided by ``num_boxes``.
    """
    logits = outputs["pred_logits"]
    idx = _matched_query_index(indices)
    target_classes = _matched_targets(targets, indices, "labels").to(logits.device)
    target_boxes = _matched_targets(targets, indices, "boxes").to(logits.device)
    pred_boxes = outputs["pred_boxes"][idx]

    iou, _ = elementwise_box_iou(
        box_cxcywh_to_xyxy(pred_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
    )
    prob = logits.sigmoid()
    pos_weights = torch.zeros_like(logits)
    neg_weights = prob**criterion.focal_gamma

    pos_index = idx + (target_classes,)
    alpha = criterion.focal_alpha
    t = prob[pos_index].pow(alpha) * iou.detach().pow(1 - alpha)
    t = t.clamp(min=0.01).detach().to(logits.dtype)
    pos_weights[pos_index] = t
    neg_weights[pos_index] = 1 - t

    # -pos * log(p) - neg * log(1 - p), with log(1 - p) = logsigmoid(x) - x.
    loss = neg_weights * logits - F.logsigmoid(logits) * (pos_weights + neg_weights)
    return {"loss_ce": loss.sum() / num_boxes}


@register_loss_term("boxes")
def loss_boxes(
    criterion: "SetCriterion",
    outputs: Dict,
    targets: List[Dict[str, torch.Tensor]],
    indices: MatchIndices,
    num_boxes: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """L1 (``loss_bbox``) and GIoU (``loss_giou``) losses on matched boxes.

    Both are summed over matches and divided by ``num_boxes``. L1 is on normalized
    ``cxcywh``; GIoU on the corresponding ``xyxy`` boxes.
    """
    pred_boxes = outputs["pred_boxes"][_matched_query_index(indices)]
    target_boxes = _matched_targets(targets, indices, "boxes").to(pred_boxes)
    loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction="none")
    loss_giou = 1 - elementwise_generalized_box_iou(
        box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(target_boxes)
    )
    return {
        "loss_bbox": loss_bbox.sum() / num_boxes,
        "loss_giou": loss_giou.sum() / num_boxes,
    }


@register_loss_term("cardinality")
@torch.no_grad()
def loss_cardinality(
    criterion: "SetCriterion",
    outputs: Dict,
    targets: List[Dict[str, torch.Tensor]],
    indices: MatchIndices,
    num_boxes: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Cardinality error (``cardinality_error``), for logging only.

    Mean absolute difference between the number of queries whose top class
    probability is above 0.5 and the number of targets, per image. Carries no
    gradient.
    """
    logits = outputs["pred_logits"]
    num_targets = torch.as_tensor(
        [len(t["labels"]) for t in targets], device=logits.device
    )
    num_pred = (logits.sigmoid().max(-1).values > 0.5).sum(1)
    return {"cardinality_error": F.l1_loss(num_pred.float(), num_targets.float())}


class SetCriterion(nn.Module):
    """Set-prediction loss for DETR-style detectors (RF-DETR recipe).

    Attributes:
        num_classes: Number of object classes. ``pred_logits`` has
            ``num_classes + 1`` sigmoid logits per query; target labels are in
            ``[0, num_classes)``, so the last logit is never a positive.
        matcher: The Hungarian matcher.
        weight_dict: Weight of each loss key, without layer suffix. Keys not listed
            are logged but excluded from :meth:`total_loss`.
        losses: Names of the registered loss terms to compute.
        focal_alpha: Alpha of the IA-BCE soft target.
        focal_gamma: Gamma of the IA-BCE negative weight ``p^gamma``.
        group_detr: Number of query groups used in training. Evaluation (with
            ``self.training == False``) matches a single group.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: Optional[HungarianMatcher] = None,
        weight_dict: Optional[Dict[str, float]] = None,
        losses: Sequence[str] = ("labels", "boxes", "cardinality"),
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        group_detr: int = 1,
    ):
        """Initialize the criterion.

        Args:
            num_classes: Number of object classes (logits are ``num_classes + 1``).
            matcher: The Hungarian matcher. Defaults to ``HungarianMatcher()``
                (costs 2 / 5 / 2).
            weight_dict: Loss weights by key. Defaults to
                :data:`DEFAULT_WEIGHT_DICT` (``loss_ce`` 1, ``loss_bbox`` 5,
                ``loss_giou`` 2).
            losses: Names of registered loss terms to compute.
            focal_alpha: Alpha of the IA-BCE soft target.
            focal_gamma: Gamma of the IA-BCE negative weight.
            group_detr: Number of query groups used in training.

        Raises:
            ValueError: If a loss name is not registered.
        """
        super().__init__()
        unknown = [name for name in losses if name not in LOSS_TERMS]
        if unknown:
            raise ValueError(
                f"Unknown DETR loss term(s) {unknown}; registered: "
                f"{sorted(LOSS_TERMS)}."
            )
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.weight_dict = dict(
            DEFAULT_WEIGHT_DICT if weight_dict is None else weight_dict
        )
        self.losses = list(losses)
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.group_detr = group_detr

    def num_boxes(
        self, targets: List[Dict[str, torch.Tensor]], device: torch.device
    ) -> torch.Tensor:
        """Loss normalizer: mean number of target boxes per rank, times the groups.

        The batch's target count is multiplied by ``group_detr`` in training (each
        group matches every target), summed over ranks when ``torch.distributed`` is
        initialized, divided by the world size and clamped to at least 1.

        Args:
            targets: The batch's target dicts.
            device: Device of the returned tensor.

        Returns:
            A float scalar tensor (kept on device to avoid a host sync).
        """
        groups = self.group_detr if self.training else 1
        num_boxes = torch.as_tensor(
            float(sum(len(t["labels"]) for t in targets) * groups), device=device
        )
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_boxes)
            world_size = dist.get_world_size()
        return torch.clamp(num_boxes / world_size, min=1.0)

    def _layers(self, outputs: Dict) -> List[Tuple[str, Dict]]:
        """The matched layers as ``(key suffix, outputs)``: final, aux, encoder."""
        final = {
            k: v for k, v in outputs.items() if k not in ("aux_outputs", "enc_outputs")
        }
        layers = [("", final)]
        for i, aux in enumerate(outputs.get("aux_outputs") or []):
            layers.append((f"_{i}", aux))
        if outputs.get("enc_outputs") is not None:
            layers.append(("_enc", outputs["enc_outputs"]))
        return layers

    def forward(
        self, outputs: Dict, targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Compute every loss term on every matched layer.

        Args:
            outputs: Model outputs: ``pred_logits`` ``(B, Q, num_classes + 1)``,
                ``pred_boxes`` ``(B, Q, 4)`` normalized ``cxcywh``, and optionally
                ``aux_outputs`` (list of such dicts) and ``enc_outputs`` (one dict).
            targets: ``B`` dicts with ``labels`` ``(N_i,)`` and ``boxes``
                ``(N_i, 4)`` normalized ``cxcywh``.

        Returns:
            Loss tensors by key: ``loss_ce``, ``loss_bbox``, ``loss_giou`` and
            ``cardinality_error`` for the final layer, the same keys with ``_<i>``
            for each aux layer and with ``_enc`` for the encoder outputs.

        Raises:
            ValueError: If a layer's logits are not ``num_classes + 1`` wide.
        """
        layers = self._layers(outputs)
        for suffix, layer in layers:
            width = layer["pred_logits"].shape[-1]
            if width != self.num_classes + 1:
                raise ValueError(
                    f"pred_logits{suffix} has {width} classes; expected "
                    f"num_classes + 1 = {self.num_classes + 1}."
                )
        groups = self.group_detr if self.training else 1
        num_boxes = self.num_boxes(targets, outputs["pred_logits"].device)

        losses = {}
        for suffix, layer in layers:
            indices = self.matcher(layer, targets, group_detr=groups)
            for name in self.losses:
                terms = LOSS_TERMS[name](self, layer, targets, indices, num_boxes)
                losses.update({key + suffix: value for key, value in terms.items()})
        return losses

    def total_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Weighted sum of the losses whose key (without layer suffix) is weighted.

        Args:
            losses: Output of :meth:`forward`.

        Returns:
            The scalar training loss.
        """
        total = None
        for key, value in losses.items():
            weight = self.weight_dict.get(_LAYER_SUFFIX.sub("", key))
            if weight is not None:
                total = weight * value if total is None else total + weight * value
        if total is None:
            raise ValueError(f"None of the losses {list(losses)} has a weight.")
        return total


def build_set_criterion(detr_config, num_classes: Optional[int] = None) -> SetCriterion:
    """Build the set criterion of a ``bbox_detr`` model from its ``detr`` config leaf.

    Args:
        detr_config: The ``model_config.head_configs.bbox_detr.detr`` leaf
            (``DetrHeadConfig`` or its DictConfig).
        num_classes: Overrides the class count (default: ``len(categories)``, or 1
            when categories are unset).

    Returns:
        A :class:`SetCriterion` with the config's matcher costs, loss weights, focal
        parameters and ``group_detr``.
    """
    if num_classes is None:
        num_classes = len(detr_config.categories or [None])
    matcher = HungarianMatcher(
        cost_class=detr_config.set_cost_class,
        cost_bbox=detr_config.set_cost_bbox,
        cost_giou=detr_config.set_cost_giou,
        focal_alpha=detr_config.focal_alpha,
        focal_gamma=detr_config.focal_gamma,
    )
    return SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": detr_config.cls_loss_coef,
            "loss_bbox": detr_config.bbox_loss_coef,
            "loss_giou": detr_config.giou_loss_coef,
        },
        focal_alpha=detr_config.focal_alpha,
        focal_gamma=detr_config.focal_gamma,
        group_detr=detr_config.group_detr,
    )
