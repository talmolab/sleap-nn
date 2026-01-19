"""Utilities for export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from sleap_nn.config.training_job_config import TrainingJobConfig
from sleap_nn.config.utils import get_backbone_type_from_cfg, get_model_type_from_cfg


def load_training_config(model_dir: str | Path) -> DictConfig:
    """Load training configuration from a model directory."""
    model_dir = Path(model_dir)
    yaml_path = model_dir / "training_config.yaml"
    json_path = model_dir / "training_config.json"

    if yaml_path.exists():
        return OmegaConf.load(yaml_path.as_posix())
    if json_path.exists():
        return TrainingJobConfig.load_sleap_config(json_path.as_posix())

    raise FileNotFoundError(
        f"No training_config.yaml or training_config.json found in {model_dir}"
    )


def resolve_input_scale(cfg: DictConfig) -> float:
    """Resolve preprocessing scale from config."""
    scale = cfg.data_config.preprocessing.scale
    # Check for list/tuple or OmegaConf ListConfig
    if isinstance(scale, (list, tuple)) or (
        hasattr(scale, "__iter__")
        and hasattr(scale, "__len__")
        and not isinstance(scale, str)
    ):
        return float(scale[0]) if len(scale) > 0 else 1.0
    return float(scale)


def resolve_input_channels(cfg: DictConfig) -> int:
    """Resolve input channels from backbone config."""
    backbone_type = get_backbone_type_from_cfg(cfg)
    return int(cfg.model_config.backbone_config[backbone_type].in_channels)


def resolve_output_stride(cfg: DictConfig, model_type: str) -> int:
    """Resolve output stride from head config."""
    head_cfg = cfg.model_config.head_configs[model_type]
    if head_cfg is None:
        return 1
    if hasattr(head_cfg, "confmaps") and head_cfg.confmaps is not None:
        return int(head_cfg.confmaps.output_stride)
    if hasattr(head_cfg, "pafs") and head_cfg.pafs is not None:
        return int(head_cfg.pafs.output_stride)
    return 1


def resolve_pafs_output_stride(cfg: DictConfig) -> int:
    """Resolve PAFs output stride for bottom-up models."""
    bottomup_cfg = getattr(cfg.model_config.head_configs, "bottomup", None)
    if bottomup_cfg is not None and bottomup_cfg.pafs is not None:
        return int(bottomup_cfg.pafs.output_stride)
    return 1


def resolve_class_maps_output_stride(cfg: DictConfig) -> int:
    """Resolve class maps output stride for multiclass bottom-up models."""
    mc_bottomup_cfg = getattr(
        cfg.model_config.head_configs, "multi_class_bottomup", None
    )
    if mc_bottomup_cfg is not None and mc_bottomup_cfg.class_maps is not None:
        return int(mc_bottomup_cfg.class_maps.output_stride)
    return 8


def resolve_class_names(cfg: DictConfig, model_type: str) -> List[str]:
    """Resolve class names for multiclass models."""
    head_cfg = cfg.model_config.head_configs.get(model_type)
    if head_cfg is None:
        return []

    # Top-down multiclass: class_vectors.classes
    if hasattr(head_cfg, "class_vectors") and head_cfg.class_vectors is not None:
        classes = getattr(head_cfg.class_vectors, "classes", None)
        if classes:
            return list(classes)

    # Bottom-up multiclass: class_maps.classes
    if hasattr(head_cfg, "class_maps") and head_cfg.class_maps is not None:
        classes = getattr(head_cfg.class_maps, "classes", None)
        if classes:
            return list(classes)

    return []


def resolve_n_classes(cfg: DictConfig, model_type: str) -> int:
    """Resolve number of classes for multiclass models."""
    class_names = resolve_class_names(cfg, model_type)
    return len(class_names) if class_names else 0


def resolve_crop_size(cfg: DictConfig) -> Optional[Tuple[int, int]]:
    """Resolve crop size from preprocessing config."""
    crop_size = cfg.data_config.preprocessing.crop_size
    if crop_size is None:
        return None
    # Check for list/tuple or OmegaConf ListConfig
    if isinstance(crop_size, (list, tuple)) or (
        hasattr(crop_size, "__iter__")
        and hasattr(crop_size, "__len__")
        and not isinstance(crop_size, (str, int))
    ):
        if len(crop_size) == 2:
            return int(crop_size[0]), int(crop_size[1])
        if len(crop_size) == 1:
            return int(crop_size[0]), int(crop_size[0])
    return int(crop_size), int(crop_size)


def resolve_node_names(cfg: DictConfig, model_type: str) -> List[str]:
    """Resolve node names for metadata."""
    skeleton_nodes = _node_names_from_skeletons(cfg.data_config.skeletons)
    if skeleton_nodes:
        return skeleton_nodes

    head_cfg = cfg.model_config.head_configs.get(model_type)
    if head_cfg is None:
        return []

    if hasattr(head_cfg, "confmaps") and head_cfg.confmaps is not None:
        part_names = getattr(head_cfg.confmaps, "part_names", None)
        if part_names:
            return list(part_names)

    if model_type == "centroid":
        anchor = getattr(head_cfg.confmaps, "anchor_part", None) if head_cfg else None
        return [anchor] if anchor else ["centroid"]

    return []


def resolve_edge_inds(cfg: DictConfig, node_names: List[str]) -> List[Tuple[int, int]]:
    """Resolve edge indices for metadata."""
    edges = _edge_inds_from_skeletons(cfg.data_config.skeletons)
    if edges:
        return _normalize_edges(edges, node_names)

    bottomup_cfg = getattr(cfg.model_config.head_configs, "bottomup", None)
    if bottomup_cfg is not None and bottomup_cfg.pafs is not None:
        edges = bottomup_cfg.pafs.edges
        if edges:
            return _normalize_edges(edges, node_names)

    return []


def resolve_model_type(cfg: DictConfig) -> str:
    """Return model type from config."""
    return get_model_type_from_cfg(cfg)


def resolve_backbone_type(cfg: DictConfig) -> str:
    """Return backbone type from config."""
    return get_backbone_type_from_cfg(cfg)


def resolve_input_shape(
    cfg: DictConfig,
    input_height: Optional[int] = None,
    input_width: Optional[int] = None,
) -> Tuple[int, int, int, int]:
    """Resolve a dummy input shape for export."""
    channels = resolve_input_channels(cfg)
    height = input_height or cfg.data_config.preprocessing.max_height or 512
    width = input_width or cfg.data_config.preprocessing.max_width or 512
    return 1, channels, int(height), int(width)


def _node_names_from_skeletons(skeletons) -> List[str]:
    if not skeletons:
        return []
    skeleton = skeletons[0]
    if hasattr(skeleton, "nodes"):
        try:
            return [node.name for node in skeleton.nodes]
        except Exception:
            pass
    if isinstance(skeleton, dict):
        nodes = skeleton.get("nodes")
        if nodes:
            if isinstance(nodes[0], dict):
                return [node.get("name", "") for node in nodes if node.get("name")]
            return [str(node) for node in nodes]
        node_names = skeleton.get("node_names")
        if node_names:
            return [str(name) for name in node_names]
    return []


def _edge_inds_from_skeletons(skeletons) -> List:
    if not skeletons:
        return []
    skeleton = skeletons[0]
    if hasattr(skeleton, "edge_inds"):
        try:
            return list(skeleton.edge_inds)
        except Exception:
            pass
    if isinstance(skeleton, dict):
        edges = skeleton.get("edges") or skeleton.get("edge_inds")
        if edges:
            return list(edges)
    return []


def _normalize_edges(edges: List, node_names: List[str]) -> List[Tuple[int, int]]:
    if not edges:
        return []
    if not node_names:
        return [(int(src), int(dst)) for src, dst in edges]

    if isinstance(edges[0][0], str):
        name_to_idx = {name: idx for idx, name in enumerate(node_names)}
        normalized = []
        for src, dst in edges:
            if src in name_to_idx and dst in name_to_idx:
                normalized.append((name_to_idx[src], name_to_idx[dst]))
        return normalized

    return [(int(src), int(dst)) for src, dst in edges]


def build_bottomup_candidate_template(
    n_nodes: int, max_peaks_per_node: int, edge_inds: List[Tuple[int, int]]
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Build candidate template matching ONNX wrapper's line_scores ordering.

    The ONNX BottomUpONNXWrapper produces line_scores with shape (n_edges, k*k) where
    for each edge connecting (src_node, dst_node), position i*k + j corresponds to:
    - src peak flat index: src_node * k + i
    - dst peak flat index: dst_node * k + j

    This function builds edge_inds and edge_peak_inds tensors that match this exact
    ordering, so that line_scores_flat[idx] corresponds to edge_peak_inds[idx].

    Args:
        n_nodes: Number of nodes in the skeleton.
        max_peaks_per_node: Maximum peaks per node (k) used during export.
        edge_inds: List of (src_node, dst_node) tuples defining skeleton edges.

    Returns:
        Tuple of (peak_channel_inds, edge_inds_tensor, edge_peak_inds_tensor):
        - peak_channel_inds: (n_nodes * k,) tensor mapping flat peak index to node
        - edge_inds_tensor: (n_edges * k * k,) tensor of edge indices for each candidate
        - edge_peak_inds_tensor: (n_edges * k * k, 2) tensor of (src, dst) peak indices

    Example:
        >>> from sleap_nn.export.utils import build_bottomup_candidate_template
        >>> peak_ch, edge_inds, edge_peaks = build_bottomup_candidate_template(
        ...     n_nodes=15, max_peaks_per_node=20, edge_inds=[(1, 2), (1, 5)]
        ... )
        >>> # Use with ONNX output:
        >>> line_scores_flat = line_scores.reshape(-1)
        >>> valid_scores = line_scores_flat[valid_mask]
        >>> valid_edge_peaks = edge_peaks[valid_mask]

    Note:
        This function is necessary because `get_connection_candidates()` in
        `sleap_nn.inference.paf_grouping` uses unstable argsort, which shuffles
        peak indices within each node and breaks alignment with ONNX output ordering.
    """
    import torch

    k = max_peaks_per_node
    n_edges = len(edge_inds)

    # peak_channel_inds: [0,0,...0, 1,1,...1, ...] (k times each)
    peak_channel_inds = torch.arange(n_nodes, dtype=torch.int32).repeat_interleave(k)

    edge_inds_list = []
    edge_peak_inds_list = []

    for edge_idx, (src_node, dst_node) in enumerate(edge_inds):
        # Build k*k candidate pairs in row-major order (i*k + j)
        # src indices: [src_node*k + 0, src_node*k + 0, ..., src_node*k + 1, ...]
        # dst indices: [dst_node*k + 0, dst_node*k + 1, ..., dst_node*k + 0, ...]
        src_base = src_node * k
        dst_base = dst_node * k

        src_indices = torch.arange(k, dtype=torch.int32).repeat_interleave(k) + src_base
        dst_indices = torch.arange(k, dtype=torch.int32).repeat(k) + dst_base

        edge_inds_list.append(torch.full((k * k,), edge_idx, dtype=torch.int32))
        edge_peak_inds_list.append(torch.stack([src_indices, dst_indices], dim=1))

    if edge_inds_list:
        edge_inds_tensor = torch.cat(edge_inds_list)
        edge_peak_inds_tensor = torch.cat(edge_peak_inds_list)
    else:
        edge_inds_tensor = torch.empty((0,), dtype=torch.int32)
        edge_peak_inds_tensor = torch.empty((0, 2), dtype=torch.int32)

    return peak_channel_inds, edge_inds_tensor, edge_peak_inds_tensor
