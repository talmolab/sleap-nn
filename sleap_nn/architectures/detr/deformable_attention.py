"""Multi-scale deformable attention in pure PyTorch (``grid_sample``).

Port of Deformable DETR's attention as used by RF-DETR (HF transformers'
``RfDetrMultiscaleDeformableAttention`` and the pure-PyTorch
``MultiScaleDeformableAttention`` path). There is no CUDA kernel: sampling is one
bilinear ``grid_sample`` per level, which runs on any device and exports to ONNX.

Each query attends to ``n_points`` sampled locations per head and level, placed
around its reference box: ``loc = ref_cxcy + offset / n_points * ref_wh / 2``.
"""

import math
import warnings
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def multi_scale_deformable_attention(
    value: torch.Tensor,
    spatial_shapes: Sequence[Tuple[int, int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """Sample values at fractional locations and sum them with attention weights.

    Args:
        value: ``(B, S, n_heads, head_dim)`` values of all levels, flattened
            row-major and concatenated (``S = sum(h * w)``).
        spatial_shapes: ``(h, w)`` of each level, in ``value`` order.
        sampling_locations: ``(B, Q, n_heads, n_levels, n_points, 2)`` normalized
            ``(x, y)`` in ``[0, 1]`` (0 and 1 are the outer edges of the map).
        attention_weights: ``(B, Q, n_heads, n_levels, n_points)``, summing to 1
            over levels and points.

    Returns:
        ``(B, Q, n_heads * head_dim)`` attended values.
    """
    batch, _, n_heads, head_dim = value.shape
    _, num_queries, _, n_levels, n_points, _ = sampling_locations.shape
    value_list = value.split([h * w for h, w in spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (h, w) in enumerate(spatial_shapes):
        # (B, h*w, heads, d) -> (B*heads, d, h, w)
        value_l = (
            value_list[level]
            .flatten(2)
            .transpose(1, 2)
            .reshape(batch * n_heads, head_dim, h, w)
        )
        # (B, Q, heads, points, 2) -> (B*heads, Q, points, 2)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # (B*heads, d, Q, points)
        sampled.append(
            F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        )
    # (B, Q, heads, levels, points) -> (B*heads, 1, Q, levels*points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch * n_heads, 1, num_queries, n_levels * n_points
    )
    output = (torch.stack(sampled, dim=-2).flatten(-2) * attention_weights).sum(-1)
    return output.view(batch, n_heads * head_dim, num_queries).transpose(1, 2)


class MultiScaleDeformableAttention(nn.Module):
    """Deformable cross-attention from queries to (multi-level) feature maps.

    Attributes:
        d_model: Query and value width.
        n_levels: Number of feature levels.
        n_heads: Number of attention heads.
        n_points: Sampling points per head and level.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 1,
        n_heads: int = 16,
        n_points: int = 2,
    ):
        """Initialize the attention.

        Args:
            d_model: Query and value width; must be divisible by ``n_heads``.
            n_levels: Number of feature levels.
            n_heads: Number of heads.
            n_points: Sampling points per head and level.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_heads``.
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )
        head_dim = d_model // n_heads
        if head_dim & (head_dim - 1):
            warnings.warn(
                f"Deformable attention head_dim {head_dim} is not a power of 2."
            )
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Deformable DETR init: offsets on a ring per head, uniform weights."""
        nn.init.zeros_(self.sampling_offsets.weight)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid = (grid / grid.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2)
        grid = grid.repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.copy_(grid.view(-1))
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        reference_boxes: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Attend from each query to points sampled around its reference box.

        Args:
            query: ``(B, Q, d_model)`` queries (position embedding already added).
            reference_boxes: ``(B, Q, 4)`` normalized ``cxcywh`` reference boxes,
                shared by all levels.
            value: ``(B, S, d_model)`` flattened feature maps of all levels.
            spatial_shapes: ``(h, w)`` of each level.

        Returns:
            ``(B, Q, d_model)`` attention output.
        """
        batch, num_queries, _ = query.shape
        seq_len = value.shape[1]
        value = self.value_proj(value).view(
            batch, seq_len, self.n_heads, self.d_model // self.n_heads
        )
        offsets = self.sampling_offsets(query).view(
            batch, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        weights = self.attention_weights(query).view(
            batch, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        weights = weights.softmax(-1).view(
            batch, num_queries, self.n_heads, self.n_levels, self.n_points
        )
        ref = reference_boxes[:, :, None, None, None, :]
        locations = ref[..., :2] + offsets / self.n_points * ref[..., 2:] * 0.5
        output = multi_scale_deformable_attention(
            value, spatial_shapes, locations, weights
        )
        return self.output_proj(output)
