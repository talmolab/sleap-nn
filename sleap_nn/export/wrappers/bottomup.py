"""Bottom-up ONNX wrapper."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from sleap_nn.export.wrappers.base import BaseExportWrapper


class BottomUpONNXWrapper(BaseExportWrapper):
    """ONNX-exportable wrapper for bottom-up inference up to PAF scoring.

    Expects input images as uint8 tensors in [0, 255].
    """

    def __init__(
        self,
        model: nn.Module,
        skeleton_edges: list,
        n_nodes: int,
        max_peaks_per_node: int = 20,
        n_line_points: int = 10,
        cms_output_stride: int = 4,
        pafs_output_stride: int = 8,
        max_edge_length_ratio: float = 0.25,
        dist_penalty_weight: float = 1.0,
        input_scale: float = 1.0,
    ) -> None:
        """Initialize bottom-up ONNX wrapper.

        Args:
            model: Bottom-up model producing confidence maps and PAFs.
            skeleton_edges: List of (src, dst) edge tuples defining skeleton.
            n_nodes: Number of nodes in the skeleton.
            max_peaks_per_node: Maximum peaks to detect per node type.
            n_line_points: Points to sample along PAF edges.
            cms_output_stride: Confidence map output stride.
            pafs_output_stride: PAF output stride.
            max_edge_length_ratio: Maximum edge length as ratio of image size.
            dist_penalty_weight: Weight for distance penalty in scoring.
            input_scale: Input scaling factor.
        """
        super().__init__(model)
        self.n_nodes = n_nodes
        self.n_edges = len(skeleton_edges)
        self.max_peaks_per_node = max_peaks_per_node
        self.n_line_points = n_line_points
        self.cms_output_stride = cms_output_stride
        self.pafs_output_stride = pafs_output_stride
        self.max_edge_length_ratio = max_edge_length_ratio
        self.dist_penalty_weight = dist_penalty_weight
        self.input_scale = input_scale

        edge_src = torch.tensor([e[0] for e in skeleton_edges], dtype=torch.long)
        edge_dst = torch.tensor([e[1] for e in skeleton_edges], dtype=torch.long)
        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)

        line_samples = torch.linspace(0, 1, n_line_points, dtype=torch.float32)
        self.register_buffer("line_samples", line_samples)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run bottom-up inference and return fixed-size outputs.

        Note: confmaps and pafs are NOT returned to avoid D2H transfer bottleneck.
        Peak detection and PAF scoring are performed on GPU within this wrapper.
        """
        image = self._normalize_uint8(image)
        if self.input_scale != 1.0:
            height = int(image.shape[-2] * self.input_scale)
            width = int(image.shape[-1] * self.input_scale)
            image = F.interpolate(
                image, size=(height, width), mode="bilinear", align_corners=False
            )

        batch_size, _, height, width = image.shape

        out = self.model(image)
        if isinstance(out, dict):
            confmaps = self._extract_tensor(out, ["confmap", "multiinstance"])
            pafs = self._extract_tensor(out, ["paf", "affinity"])
        else:
            confmaps, pafs = out[:2]

        peaks, peak_vals, peak_mask = self._find_topk_peaks_per_node(
            confmaps, self.max_peaks_per_node
        )

        peaks = peaks * self.cms_output_stride

        # Compute max_edge_length to match PyTorch implementation:
        # max_edge_length = ratio * max(paf_dims) * pafs_stride
        # PAFs shape is (batch, 2*edges, H, W)
        _, n_paf_channels, paf_height, paf_width = pafs.shape
        max_paf_dim = max(n_paf_channels, paf_height, paf_width)
        max_edge_length = torch.tensor(
            self.max_edge_length_ratio * max_paf_dim * self.pafs_output_stride,
            dtype=peaks.dtype,
            device=peaks.device,
        )

        line_scores, candidate_mask = self._score_all_candidates(
            pafs, peaks, peak_mask, max_edge_length
        )

        # Only return final outputs needed for CPU-side grouping.
        # Do NOT return confmaps/pafs - they are large (~29 MB/batch) and
        # cause D2H transfer bottleneck. Peak detection and PAF scoring
        # are already done on GPU above.
        return {
            "peaks": peaks,
            "peak_vals": peak_vals,
            "peak_mask": peak_mask,
            "line_scores": line_scores,
            "candidate_mask": candidate_mask,
        }

    def _score_all_candidates(
        self,
        pafs: torch.Tensor,
        peaks: torch.Tensor,
        peak_mask: torch.Tensor,
        max_edge_length: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score all K*K candidate connections for each edge."""
        batch_size = peaks.shape[0]
        k = self.max_peaks_per_node
        n_edges = self.n_edges

        _, _, paf_height, paf_width = pafs.shape

        src_peaks = peaks[:, self.edge_src, :, :]
        dst_peaks = peaks[:, self.edge_dst, :, :]

        src_mask = peak_mask[:, self.edge_src, :]
        dst_mask = peak_mask[:, self.edge_dst, :]

        src_peaks_exp = src_peaks.unsqueeze(3).expand(-1, -1, -1, k, -1)
        dst_peaks_exp = dst_peaks.unsqueeze(2).expand(-1, -1, k, -1, -1)

        src_mask_exp = src_mask.unsqueeze(3).expand(-1, -1, -1, k)
        dst_mask_exp = dst_mask.unsqueeze(2).expand(-1, -1, k, -1)
        candidate_mask = src_mask_exp & dst_mask_exp

        src_peaks_flat = src_peaks_exp.reshape(batch_size, n_edges, k * k, 2)
        dst_peaks_flat = dst_peaks_exp.reshape(batch_size, n_edges, k * k, 2)
        candidate_mask_flat = candidate_mask.reshape(batch_size, n_edges, k * k)

        spatial_vecs = dst_peaks_flat - src_peaks_flat
        spatial_lengths = torch.norm(spatial_vecs, dim=-1, keepdim=True).clamp(min=1e-6)
        spatial_vecs_norm = spatial_vecs / spatial_lengths

        line_samples = self.line_samples.view(1, 1, 1, -1, 1)
        src_exp = src_peaks_flat.unsqueeze(3)
        dst_exp = dst_peaks_flat.unsqueeze(3)
        line_points = src_exp + line_samples * (dst_exp - src_exp)

        line_points_paf = line_points / self.pafs_output_stride
        line_x = line_points_paf[..., 0].clamp(0, paf_width - 1)
        line_y = line_points_paf[..., 1].clamp(0, paf_height - 1)

        line_scores = self._sample_and_score_lines(
            pafs,
            line_x,
            line_y,
            spatial_vecs_norm,
            spatial_lengths.squeeze(-1),
            max_edge_length,
        )

        line_scores = line_scores.masked_fill(~candidate_mask_flat, -2.0)
        return line_scores, candidate_mask_flat

    def _sample_and_score_lines(
        self,
        pafs: torch.Tensor,
        line_x: torch.Tensor,
        line_y: torch.Tensor,
        spatial_vecs_norm: torch.Tensor,
        spatial_lengths: torch.Tensor,
        max_edge_length: torch.Tensor,
    ) -> torch.Tensor:
        """Sample PAF values along lines and compute scores."""
        batch_size, n_edges, k2, n_points = line_x.shape
        _, _, paf_height, paf_width = pafs.shape

        all_scores = []
        for edge_idx in range(n_edges):
            paf_x = pafs[:, 2 * edge_idx, :, :]
            paf_y = pafs[:, 2 * edge_idx + 1, :, :]

            lx = line_x[:, edge_idx, :, :]
            ly = line_y[:, edge_idx, :, :]

            lx_norm = (lx / (paf_width - 1)) * 2 - 1
            ly_norm = (ly / (paf_height - 1)) * 2 - 1

            grid = torch.stack([lx_norm, ly_norm], dim=-1)

            paf_x_samples = F.grid_sample(
                paf_x.unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(1)

            paf_y_samples = F.grid_sample(
                paf_y.unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(1)

            paf_samples = torch.stack([paf_x_samples, paf_y_samples], dim=-1)
            disp_vec = spatial_vecs_norm[:, edge_idx, :, :]

            dot_products = (paf_samples * disp_vec.unsqueeze(2)).sum(dim=-1)
            mean_scores = dot_products.mean(dim=-1)

            edge_lengths = spatial_lengths[:, edge_idx, :]
            dist_penalty = self._compute_distance_penalty(edge_lengths, max_edge_length)

            all_scores.append(mean_scores + dist_penalty)

        return torch.stack(all_scores, dim=1)

    def _compute_distance_penalty(
        self, distances: torch.Tensor, max_edge_length: torch.Tensor
    ) -> torch.Tensor:
        """Compute distance penalty for edge candidates.

        Matches the PyTorch implementation in sleap_nn.inference.paf_grouping.
        Penalty is 0 when distance <= max_edge_length, and negative when longer.
        """
        # Match PyTorch: penalty = clamp((max_edge_length / distance) - 1, max=0) * weight
        penalty = torch.clamp((max_edge_length / distances) - 1, max=0)
        return penalty * self.dist_penalty_weight
