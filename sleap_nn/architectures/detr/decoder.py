"""Transformer decoder of the ``bbox_detr`` detector.

Port of HF transformers' ``RfDetrDecoder`` / ``RfDetrDecoderLayer``
(``models/rf_detr``, transformers 5.20.2) with RF-DETR's settings fixed:

- ``lite_refpoint_refine``: every layer attends around the same reference boxes
  (no layer-to-layer refinement), so the query position embedding is computed once.
- Group DETR: in training the ``G * Q`` queries hold ``G`` independent groups;
  self-attention stays within a group by stacking the groups on the batch axis.
- Post-norm layers: self-attention -> deformable cross-attention -> FFN, each with a
  residual and a LayerNorm.

Module and parameter names follow HF's so RF-DETR weights load by renaming
prefixes only.
"""

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sleap_nn.architectures.detr.deformable_attention import (
    MultiScaleDeformableAttention,
)

_ACTIVATIONS = {"relu": F.relu, "gelu": F.gelu, "silu": F.silu}


def sine_position_embedding(
    boxes: torch.Tensor, num_pos_feats: int = 128, temperature: float = 10000.0
) -> torch.Tensor:
    """Sinusoidal embedding of normalized box coordinates (DAB/LW-DETR style).

    Each coordinate is embedded with ``num_pos_feats`` interleaved sin/cos
    features and the per-coordinate embeddings are concatenated in the order
    ``(y, x, w, h)`` (x and y swapped, the DETR convention).

    Args:
        boxes: ``(..., 4)`` normalized ``cxcywh`` (or ``(..., 2)`` centers).
        num_pos_feats: Features per coordinate.
        temperature: Frequency base.

    Returns:
        ``(..., n_coords * num_pos_feats)`` embedding, in ``boxes``' dtype.
    """
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=boxes.device)
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )
    embeddings = []
    for coord in boxes.unbind(-1):
        e = coord[..., None] * (2 * math.pi) / dim_t
        embeddings.append(
            torch.stack((e[..., 0::2].sin(), e[..., 1::2].cos()), dim=-1).flatten(-2)
        )
    if len(embeddings) >= 2:
        embeddings[0], embeddings[1] = embeddings[1], embeddings[0]
    return torch.cat(embeddings, dim=-1).to(boxes.dtype)


def refine_boxes(reference: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply box deltas to reference boxes (RF-DETR's ``bbox_reparam``).

    ``cxcy = delta_xy * wh_ref + cxcy_ref`` and ``wh = exp(delta_wh) * wh_ref``,
    on normalized ``cxcywh``; no sigmoid.

    Args:
        reference: ``(..., 4)`` reference boxes.
        deltas: ``(..., 4)`` predicted deltas (broadcast against ``reference``).

    Returns:
        ``(..., 4)`` refined boxes.
    """
    cxcy = deltas[..., :2] * reference[..., 2:] + reference[..., :2]
    wh = deltas[..., 2:].exp() * reference[..., 2:]
    return torch.cat([cxcy, wh], dim=-1)


class MLP(nn.Module):
    """Linear layers with ReLU between them (HF ``RfDetrMLPPredictionHead``)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        """Initialize ``num_layers`` linears: ``input -> hidden ... -> output``."""
        super().__init__()
        self.num_layers = num_layers
        dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + dims, dims + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layers."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class GroupSelfAttention(nn.Module):
    """Multi-head self-attention restricted to each query group (HF ``RfDetrAttention``).

    Queries and keys see the position embedding; values do not.
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.0):
        """Initialize the attention.

        Args:
            d_model: Query width; must be divisible by ``n_heads``.
            n_heads: Number of heads.
            dropout: Attention-probability dropout in training.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self, hidden_states: torch.Tensor, query_pos: torch.Tensor, num_groups: int = 1
    ) -> torch.Tensor:
        """Attend within each of ``num_groups`` equal, contiguous query groups.

        Args:
            hidden_states: ``(B, G * Q, d_model)`` queries.
            query_pos: ``(B, G * Q, d_model)`` position embeddings.
            num_groups: Number of groups ``G``.

        Returns:
            ``(B, G * Q, d_model)`` attention output.
        """
        batch, seq_len, d_model = hidden_states.shape
        qk_in = hidden_states + query_pos
        v_in = hidden_states
        if num_groups > 1:
            group_len = seq_len // num_groups
            qk_in = torch.cat(qk_in.split(group_len, dim=1), dim=0)
            v_in = torch.cat(v_in.split(group_len, dim=1), dim=0)

        def heads(x: torch.Tensor) -> torch.Tensor:
            return x.view(
                x.shape[0], x.shape[1], self.n_heads, self.head_dim
            ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            heads(self.q_proj(qk_in)),
            heads(self.k_proj(qk_in)),
            heads(self.v_proj(v_in)),
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = self.o_proj(out.transpose(1, 2).reshape(out.shape[0], -1, d_model))
        if num_groups > 1:
            out = torch.cat(out.split(batch, dim=0), dim=1)
        return out


class FeedForward(nn.Module):
    """Residual two-layer FFN (HF ``RfDetrMLP``); the residual is added inside."""

    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize ``fc1`` (``d_model -> dim_feedforward``) and ``fc2`` (back)."""
        super().__init__()
        self.dropout = dropout
        self.activation = _ACTIVATIONS[activation]
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x + fc2(act(fc1(x)))`` (with dropout in training)."""
        y = F.dropout(self.activation(self.fc1(x)), self.dropout, self.training)
        y = F.dropout(self.fc2(y), self.dropout, self.training)
        return x + y


class DecoderLayer(nn.Module):
    """Group self-attention -> deformable cross-attention -> FFN, each post-normed."""

    def __init__(
        self,
        d_model: int = 256,
        sa_nheads: int = 8,
        ca_nheads: int = 16,
        n_points: int = 2,
        n_levels: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize the layer.

        Args:
            d_model: Query width.
            sa_nheads: Self-attention heads.
            ca_nheads: Deformable cross-attention heads.
            n_points: Sampling points per cross-attention head and level.
            n_levels: Feature levels.
            dim_feedforward: FFN hidden width.
            dropout: Dropout after each sub-block (and inside the FFN) in training.
            activation: FFN activation.
        """
        super().__init__()
        self.dropout = dropout
        self.self_attn = GroupSelfAttention(d_model, sa_nheads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.cross_attn = MultiScaleDeformableAttention(
            d_model, n_levels, ca_nheads, n_points
        )
        self.cross_attn_layer_norm = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, dim_feedforward, dropout, activation)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_pos: torch.Tensor,
        reference_boxes: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        num_groups: int = 1,
    ) -> torch.Tensor:
        """Run the layer.

        Args:
            hidden_states: ``(B, G * Q, d_model)`` content queries.
            query_pos: ``(B, G * Q, d_model)`` position embeddings.
            reference_boxes: ``(B, G * Q, 4)`` normalized ``cxcywh``.
            memory: ``(B, S, d_model)`` flattened feature maps.
            spatial_shapes: ``(h, w)`` of each feature level.
            num_groups: Query groups ``G``.

        Returns:
            ``(B, G * Q, d_model)`` updated queries.
        """
        sa = self.self_attn(hidden_states, query_pos, num_groups)
        hidden_states = self.self_attn_layer_norm(
            hidden_states + F.dropout(sa, self.dropout, self.training)
        )
        ca = self.cross_attn(
            hidden_states + query_pos, reference_boxes, memory, spatial_shapes
        )
        hidden_states = self.cross_attn_layer_norm(
            hidden_states + F.dropout(ca, self.dropout, self.training)
        )
        return self.layer_norm(self.mlp(hidden_states))


class Decoder(nn.Module):
    """Stack of decoder layers around fixed reference boxes (HF ``RfDetrDecoder``).

    Attributes:
        d_model: Query width.
        layers: The decoder layers.
        layernorm: Shared norm applied to every layer's output (the input of the
            prediction heads; the next layer gets the un-normed output).
        ref_point_head: MLP from the sine embedding of the reference box to the
            query position embedding.
    """

    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 256,
        sa_nheads: int = 8,
        ca_nheads: int = 16,
        n_points: int = 2,
        n_levels: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        """Initialize ``num_layers`` identical layers (see :class:`DecoderLayer`)."""
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            DecoderLayer(
                d_model,
                sa_nheads,
                ca_nheads,
                n_points,
                n_levels,
                dim_feedforward,
                dropout,
                activation,
            )
            for _ in range(num_layers)
        )
        self.layernorm = nn.LayerNorm(d_model)
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, num_layers=2)

    def forward(
        self,
        queries: torch.Tensor,
        reference_boxes: torch.Tensor,
        memory: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        num_groups: int = 1,
    ) -> torch.Tensor:
        """Decode the queries.

        Args:
            queries: ``(B, G * Q, d_model)`` initial content queries.
            reference_boxes: ``(B, G * Q, 4)`` normalized ``cxcywh``, used by every
                layer.
            memory: ``(B, S, d_model)`` flattened feature maps.
            spatial_shapes: ``(h, w)`` of each feature level.
            num_groups: Query groups ``G``.

        Returns:
            ``(num_layers, B, G * Q, d_model)`` normed output of every layer.
        """
        query_pos = self.ref_point_head(
            sine_position_embedding(reference_boxes, num_pos_feats=self.d_model // 2)
        )
        hidden_states = queries
        intermediate = []
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                query_pos,
                reference_boxes,
                memory,
                spatial_shapes,
                num_groups,
            )
            intermediate.append(self.layernorm(hidden_states))
        return torch.stack(intermediate)
