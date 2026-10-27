"""The ``bbox_detr`` detector after the backbone: projector, proposals, decoder, heads.

Port of HF transformers' ``RfDetrModel`` + ``RfDetrForObjectDetection``
(``models/rf_detr``, transformers 5.20.2) minus the backbone, for box detection
only. Given the backbone's tapped token maps:

1. **Projector.** Concatenate the taps, C2f + LayerNorm -> one ``hidden_dim`` map
   ("memory") at the token stride.
2. **Two-stage proposals.** Every memory token is a proposal box (its cell center,
   ``wh = 0.05``). Per query group: Linear + LayerNorm, an encoder class head and an
   encoder box head (deltas on the proposal), then the top ``num_queries`` tokens
   by max class logit. Their boxes, detached, are the group's reference boxes.
3. **Queries.** Learned content queries (``query_feat``) and learned reference
   deltas (``reference_point_embed``, zero-init) applied on the proposal boxes,
   both ``num_queries * group_detr`` long. Training uses all groups; eval only
   group 0.
4. **Decoder.** Every layer attends around the same reference boxes.
5. **Heads.** One class head (``num_classes + 1`` sigmoid logits) and one box MLP
   (deltas on the reference boxes), shared by every decoder layer.

Outputs are the dict :class:`~sleap_nn.training.detr.SetCriterion` consumes:
``pred_logits`` / ``pred_boxes`` of the last layer, ``aux_outputs`` of the earlier
layers and ``enc_outputs`` of the proposals. Boxes are normalized ``cxcywh``
relative to the model input.

Deviation from HF: there is no pixel mask. Boxes are relative to the whole input
tensor (padding included), which is what sleap-nn's data pipeline produces;
HF's mask-free path (all-valid mask) is the same computation.
"""

import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import nn

from sleap_nn.architectures.detr.decoder import MLP, Decoder, refine_boxes
from sleap_nn.architectures.detr.deformable_attention import (
    MultiScaleDeformableAttention,
)
from sleap_nn.architectures.detr.projector import ScaleProjector

#: Proposal width/height (normalized) of the two-stage token proposals.
PROPOSAL_WH = 0.05
#: Proposals with any coordinate outside ``(PROPOSAL_MIN, 1 - PROPOSAL_MIN)`` are
#: invalid (never selected), as in Deformable DETR.
PROPOSAL_MIN = 0.01


def token_proposals(
    spatial_shapes: Sequence[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One proposal box per token: the cell center with ``wh = 0.05 * 2**level``.

    Args:
        spatial_shapes: ``(h, w)`` of each level.
        device: Device of the outputs.
        dtype: Dtype of the proposals.

    Returns:
        ``(proposals, invalid)``: ``(S, 4)`` normalized ``cxcywh`` proposals (0 where
        invalid) and the ``(S, 1)`` bool mask of proposals with a coordinate
        outside ``(0.01, 0.99)``.
    """
    proposals = []
    for level, (h, w) in enumerate(spatial_shapes):
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        wh = torch.full_like(grid_x, PROPOSAL_WH * 2.0**level)
        proposals.append(torch.stack([grid_x, grid_y, wh, wh], dim=-1).view(-1, 4))
    proposals = torch.cat(proposals, dim=0)
    valid = ((proposals > PROPOSAL_MIN) & (proposals < 1 - PROPOSAL_MIN)).all(
        -1, keepdim=True
    )
    return proposals.masked_fill(~valid, 0.0), ~valid


class BBoxDetrHead(nn.Module):
    """Everything of the ``bbox_detr`` detector after the backbone.

    Attributes:
        num_classes: Number of object classes (logits are ``num_classes + 1``).
        hidden_dim: Width of the memory and queries.
        num_queries: Queries per group (and detections per image at eval).
        group_detr: Query groups in training.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        hidden_dim: int = 256,
        num_queries: int = 300,
        group_detr: int = 13,
        dec_layers: int = 3,
        sa_nheads: int = 8,
        ca_nheads: int = 16,
        dec_n_points: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        c2f_num_blocks: int = 3,
        c2f_expansion: float = 0.5,
        projector_activation: str = "silu",
        decoder_activation: str = "relu",
    ):
        """Initialize the detector.

        Args:
            in_channels: Channels of the concatenated backbone taps.
            num_classes: Number of object classes.
            hidden_dim: Width of the memory and queries.
            num_queries: Queries per group.
            group_detr: Query groups in training (Group DETR).
            dec_layers: Decoder layers.
            sa_nheads: Decoder self-attention heads.
            ca_nheads: Decoder deformable cross-attention heads.
            dec_n_points: Sampling points per cross-attention head.
            dim_feedforward: Decoder FFN hidden width.
            dropout: Decoder dropout.
            c2f_num_blocks: Bottlenecks in the projector's C2f block.
            c2f_expansion: C2f hidden width fraction.
            projector_activation: Projector activation.
            decoder_activation: Decoder FFN activation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.group_detr = group_detr

        self.projector = ScaleProjector(
            in_channels,
            hidden_dim,
            num_blocks=c2f_num_blocks,
            expansion=c2f_expansion,
            activation=projector_activation,
        )
        self.decoder = Decoder(
            num_layers=dec_layers,
            d_model=hidden_dim,
            sa_nheads=sa_nheads,
            ca_nheads=ca_nheads,
            n_points=dec_n_points,
            n_levels=1,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=decoder_activation,
        )
        self.query_feat = nn.Embedding(num_queries * group_detr, hidden_dim)
        self.reference_point_embed = nn.Embedding(num_queries * group_detr, 4)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        self.enc_output = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(group_detr)
        )
        self.enc_output_norm = nn.ModuleList(
            nn.LayerNorm(hidden_dim) for _ in range(group_detr)
        )
        self._reset_parameters()
        # The encoder heads start as copies of the (initialized) decoder heads.
        self.enc_out_class_embed = nn.ModuleList(
            copy.deepcopy(self.class_embed) for _ in range(group_detr)
        )
        self.enc_out_bbox_embed = nn.ModuleList(
            copy.deepcopy(self.bbox_embed) for _ in range(group_detr)
        )

    @torch.no_grad()
    def _reset_parameters(self):
        """RF-DETR init.

        Xavier on the decoder's matrices, then the deformable attention init;
        reference deltas at 0; class bias at prior 0.01; last box layer at 0.
        """
        for p in self.decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for module in self.decoder.modules():
            if isinstance(module, MultiScaleDeformableAttention):
                module.reset_parameters()
        nn.init.zeros_(self.reference_point_embed.weight)
        prior_prob = 0.01
        nn.init.constant_(
            self.class_embed.bias, -math.log((1 - prior_prob) / prior_prob)
        )
        nn.init.zeros_(self.bbox_embed.layers[-1].weight)
        nn.init.zeros_(self.bbox_embed.layers[-1].bias)

    def _proposals(
        self,
        memory: torch.Tensor,
        proposals: torch.Tensor,
        invalid: torch.Tensor,
        group: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-``num_queries`` encoder proposals of one query group.

        Args:
            memory: ``(B, S, D)`` memory with invalid tokens zeroed.
            proposals: ``(S, 4)`` token proposals.
            invalid: ``(S, 1)`` invalid-proposal mask.
            group: Query group index.

        Returns:
            ``(logits, boxes, reference)``: the selected tokens' encoder class
            logits ``(B, Q, C + 1)`` and boxes ``(B, Q, 4)`` (with gradients), and
            the boxes detached as reference boxes.
        """
        query = self.enc_output_norm[group](self.enc_output[group](memory))
        logits = self.enc_out_class_embed[group](query)
        boxes = refine_boxes(proposals, self.enc_out_bbox_embed[group](query))
        scores = logits.masked_fill(invalid, float("-inf")).max(-1).values
        topk = scores.topk(self.num_queries, dim=1).indices
        logits = logits.gather(1, topk[..., None].expand(-1, -1, logits.shape[-1]))
        boxes = boxes.gather(1, topk[..., None].expand(-1, -1, 4))
        return logits, boxes, boxes.detach()

    def forward(self, features: Sequence[torch.Tensor]) -> Dict:
        """Detect boxes from the backbone's tapped token maps.

        Args:
            features: The backbone taps, each ``(B, D, h, w)`` at one stride.

        Returns:
            Dict with ``pred_logits`` ``(B, G * Q, num_classes + 1)``,
            ``pred_boxes`` ``(B, G * Q, 4)`` normalized ``cxcywh``,
            ``aux_outputs`` (one such dict per earlier decoder layer) and
            ``enc_outputs`` (the selected encoder proposals), where ``G`` is
            ``group_detr`` in training and 1 in eval.

        Raises:
            ValueError: If the map has fewer tokens than ``num_queries``.
        """
        memory = self.projector(features)
        batch, _, h, w = memory.shape
        if not torch.jit.is_tracing() and h * w < self.num_queries:
            raise ValueError(
                f"The {h}x{w} token map has fewer tokens than num_queries "
                f"({self.num_queries}); use a larger input or fewer queries."
            )
        spatial_shapes = [(h, w)]
        memory = memory.flatten(2).transpose(1, 2)  # (B, S, D)
        proposals, invalid = token_proposals(
            spatial_shapes, memory.device, memory.dtype
        )
        enc_memory = memory.masked_fill(invalid, 0.0)

        groups = self.group_detr if self.training else 1
        enc_logits, enc_boxes, references = [], [], []
        for g in range(groups):
            logits, boxes, reference = self._proposals(
                enc_memory, proposals, invalid, g
            )
            enc_logits.append(logits)
            enc_boxes.append(boxes)
            references.append(reference)
        reference = torch.cat(references, dim=1)  # (B, G * Q, 4)

        num_total = groups * self.num_queries
        reference = refine_boxes(
            reference, self.reference_point_embed.weight[:num_total][None]
        )
        queries = self.query_feat.weight[:num_total][None].expand(batch, -1, -1)

        hidden = self.decoder(queries, reference, memory, spatial_shapes, groups)
        logits = self.class_embed(hidden)  # (L, B, G * Q, C + 1)
        boxes = refine_boxes(reference, self.bbox_embed(hidden))

        return {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "aux_outputs": [
                {"pred_logits": lg, "pred_boxes": bx}
                for lg, bx in zip(logits[:-1], boxes[:-1])
            ],
            "enc_outputs": {
                "pred_logits": torch.cat(enc_logits, dim=1),
                "pred_boxes": torch.cat(enc_boxes, dim=1),
            },
        }


def build_bbox_detr_head(
    detr_config, in_channels: int, num_classes: Optional[int] = None
) -> BBoxDetrHead:
    """Build a :class:`BBoxDetrHead` from a ``DetrHeadConfig`` (or its DictConfig).

    Args:
        detr_config: The ``model_config.head_configs.bbox_detr.detr`` leaf.
        in_channels: Channels of the concatenated backbone taps.
        num_classes: Overrides the class count (default: ``len(categories)``, or
            1 when categories are unset).

    Returns:
        The detector head.
    """
    if num_classes is None:
        num_classes = len(detr_config.categories or [None])
    return BBoxDetrHead(
        in_channels=in_channels,
        num_classes=num_classes,
        hidden_dim=detr_config.hidden_dim,
        num_queries=detr_config.num_queries,
        group_detr=detr_config.group_detr,
        dec_layers=detr_config.dec_layers,
        sa_nheads=detr_config.sa_nheads,
        ca_nheads=detr_config.ca_nheads,
        dec_n_points=detr_config.dec_n_points,
        dim_feedforward=detr_config.dim_feedforward,
        dropout=detr_config.dropout,
        c2f_num_blocks=detr_config.c2f_num_blocks,
        c2f_expansion=detr_config.c2f_expansion,
    )
