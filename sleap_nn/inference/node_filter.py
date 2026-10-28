"""Inference-time node filter: compute only some of a confmap head's channels.

A model trained on a merged skeleton (e.g. many datasets' nodes prefixed
``<dataset>/<node>``, each file supervised on its own channels) carries hundreds of
confmap channels, of which a user wants one dataset's. :func:`apply_node_filter`
slices the confmap head's output convolution (``Conv2d(in -> K)``, 1x1 or 3x3) to
the selected channels after the checkpoint is loaded, so the unused confmaps are
never computed, and restricts the output skeleton (edges, symmetries, the
flip-test channel permutation) to the kept nodes. Sliced channels are the
unfiltered model's channels: a 1x1 / 3x3 conv's output channel ``c`` depends only
on its own filter ``weight[c]`` and ``bias[c]`` (up to float rounding: a conv
backend may sum in another order for another channel count, ~1e-7 on macOS CPU;
bit-identical on Linux CPU).

Scope: ``single_instance`` and top-down ``centered_instance`` models.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from loguru import logger
from torch import nn

#: A node filter: a list of node names, or a string prefix (``"hcm-2025/"``).
NodeFilter = Union[str, Sequence[str]]

#: Model types whose confmap head the filter can slice, and that head's name.
NODE_FILTER_HEADS = {
    "single_instance": "SingleInstanceConfmapsHead",
    "centered_instance": "CenteredInstanceConfmapsHead",
}


def resolve_node_filter(
    node_names: Sequence[str], node_filter: NodeFilter
) -> List[int]:
    """Indices of the nodes a filter keeps, in model (channel) order.

    Args:
        node_names: The model's confmap channel names.
        node_filter: A list of node names, or a string prefix selecting every node
            whose name starts with it.

    Raises:
        ValueError: If a listed node is not a model node, or nothing is selected.
    """
    names = list(node_names)
    if isinstance(node_filter, str):
        idx = [i for i, n in enumerate(names) if n.startswith(node_filter)]
        if not idx:
            raise ValueError(
                f"No model node starts with the prefix {node_filter!r} (model nodes: "
                f"{names[:10]}{' ...' if len(names) > 10 else ''})."
            )
        return idx
    wanted = list(node_filter)
    missing = [n for n in wanted if n not in names]
    if missing:
        raise ValueError(f"Nodes {missing} are not model nodes.")
    if not wanted:
        raise ValueError("The node filter selects no node.")
    keep = set(wanted)
    return [i for i, n in enumerate(names) if n in keep]


def output_names(
    names: Sequence[str],
    node_filter: NodeFilter,
    strip_prefix: Union[bool, str, None],
) -> List[str]:
    """Kept node names as written to the output, with ``strip_prefix`` applied.

    ``strip_prefix=True`` strips the filter's own prefix (``node_filter`` must then
    be a string); a string strips that prefix from every name that starts with it.

    Raises:
        ValueError: If ``strip_prefix=True`` with a list filter, or stripping
            leaves an empty or repeated name.
    """
    if strip_prefix is None or strip_prefix is False:
        return list(names)
    if strip_prefix is True:
        if not isinstance(node_filter, str):
            raise ValueError(
                "strip_prefix=True strips the node filter's prefix; with a list of "
                "nodes, pass the prefix to strip as a string."
            )
        strip_prefix = node_filter
    out = [n[len(strip_prefix) :] if n.startswith(strip_prefix) else n for n in names]
    if any(not n for n in out) or len(set(out)) != len(out):
        raise ValueError(
            f"Stripping {strip_prefix!r} from {list(names)} leaves empty or repeated "
            "node names."
        )
    return out


def restrict_skeleton(
    skeleton, keep: Sequence[str], rename: Optional[Dict[str, str]] = None
):
    """A copy of ``skeleton`` on the ``keep`` nodes (in that order).

    Edges and symmetries whose two nodes are both kept survive; ``rename`` maps
    old to new node names.
    """
    import sleap_io as sio

    rename = rename or {}
    kept = set(keep)

    def new(n):
        return rename.get(n, n)

    edges = [
        (new(e.source.name), new(e.destination.name))
        for e in skeleton.edges
        if e.source.name in kept and e.destination.name in kept
    ]
    symmetries = []
    for sym in skeleton.symmetries:
        a, b = (n.name for n in sym.nodes)
        if a in kept and b in kept:
            symmetries.append((new(a), new(b)))
    return sio.Skeleton(
        nodes=[new(n) for n in keep],
        edges=edges,
        symmetries=symmetries,
        name=skeleton.name,
    )


def check_flip_partners(skeleton, keep: Sequence[str]) -> None:
    """Flip-test needs both nodes of every symmetric pair with a kept node.

    The mirrored pass reads a node's left/right partner channel; a dropped
    partner's channel is not computed.

    Raises:
        ValueError: If a kept node's symmetric partner is not kept.
    """
    kept = set(keep)
    missing = []
    for sym in skeleton.symmetries:
        a, b = (n.name for n in sym.nodes)
        if (a in kept) != (b in kept):
            missing.append((a, b))
    if missing:
        raise ValueError(
            "flip_test needs the symmetric partner of every kept node; add them to "
            f"the node filter or turn flip_test off. Unpaired: {missing}."
        )


def slice_head_conv(model: nn.Module, head_name: str, idx: Sequence[int]) -> nn.Conv2d:
    """Keep only the ``idx`` output channels of a head's output convolution.

    ``model`` is a :class:`~sleap_nn.architectures.model.Model`; the head's layer is
    ``Sequential({head_name: Sequential(Conv2d, activation)})`` (see
    :meth:`~sleap_nn.architectures.heads.Head.make_head`). The conv is replaced in
    place by one with the selected filters (same kernel, padding, dtype, device).

    Returns:
        The new convolution.

    Raises:
        ValueError: If the model has no such head.
    """
    for head, layer in zip(model.heads, model.head_layers):
        if head.name != head_name:
            continue
        block = getattr(layer, head_name)
        conv = block[0]
        if not isinstance(conv, nn.Conv2d):
            raise ValueError(f"{head_name} does not end in a Conv2d: {block}.")
        sel = torch.as_tensor(list(idx), dtype=torch.long, device=conv.weight.device)
        new = nn.Conv2d(
            conv.in_channels,
            len(idx),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        with torch.no_grad():
            new.weight.copy_(conv.weight.index_select(0, sel))
            if conv.bias is not None:
                new.bias.copy_(conv.bias.index_select(0, sel))
        new.train(conv.training)
        block[0] = new
        return new
    raise ValueError(f"The model has no {head_name}.")


def _head_target(assets, model_types: Sequence[str]) -> Tuple[str, nn.Module, object]:
    """``(model_type, lightning module, its confmaps head config)`` to filter."""
    if "centered_instance" in model_types:
        peaks = getattr(assets.inference_model, "instance_peaks", None)
        module = getattr(peaks, "torch_model", None)
        if module is None:
            raise ValueError("The node filter needs a centered_instance model.")
        cfg = assets.confmap_config.model_config.head_configs.centered_instance
        return "centered_instance", module, cfg.confmaps
    if "single_instance" in model_types:
        cfg = assets.confmap_config.model_config.head_configs.single_instance
        return "single_instance", assets.inference_model.torch_model, cfg.confmaps
    raise ValueError(
        "The node filter is implemented for single_instance and centered_instance "
        f"models only; detected types: {list(model_types)}."
    )


def apply_node_filter(
    assets,
    model_types: Sequence[str],
    node_filter: NodeFilter,
    strip_prefix: Union[bool, str, None] = None,
) -> List[str]:
    """Restrict loaded inference assets to some confmap channels, in place.

    Slices the confmap head's output conv (:func:`slice_head_conv`), restricts
    ``assets.skeletons[0]`` (:func:`restrict_skeleton`; renamed by
    ``strip_prefix``), updates the head config's ``part_names`` and the head's own,
    and recomputes the flip-test channel permutation for the kept nodes.

    Args:
        assets: :class:`~sleap_nn.inference.loaders.LoadedAssets`.
        model_types: The detected model types.
        node_filter: A list of node names, or a string prefix.
        strip_prefix: ``True`` strips the filter's prefix from the output node
            names; a string strips that prefix; ``None``/``False`` keeps the
            model's names.

    Returns:
        The kept model node names (before ``strip_prefix``).
    """
    from sleap_nn.inference.loaders import _flip_test_idx

    model_type, module, head_cfg = _head_target(assets, model_types)
    skeleton = assets.skeletons[0]
    channels = list(head_cfg.part_names or skeleton.node_names)
    idx = resolve_node_filter(channels, node_filter)
    keep = [channels[i] for i in idx]
    out_names = output_names(keep, node_filter, strip_prefix)
    peaks = getattr(assets.inference_model, "instance_peaks", None)
    flip = bool(getattr(peaks, "flip_test", False))
    if flip:
        check_flip_partners(skeleton, keep)

    slice_head_conv(module.model, NODE_FILTER_HEADS[model_type], idx)
    for head in module.model.heads:
        if head.name == NODE_FILTER_HEADS[model_type]:
            head.part_names = list(out_names)
    head_cfg.part_names = list(out_names)
    if hasattr(module, "node_names"):
        module.node_names = list(out_names)
    rename = dict(zip(keep, out_names))
    missing = [n for n in keep if n not in skeleton.node_names]
    if missing:
        raise ValueError(f"Confmap channels {missing} are not skeleton nodes.")
    new_skeleton = restrict_skeleton(skeleton, keep, rename)
    assets.skeletons = [new_skeleton, *assets.skeletons[1:]]
    if flip:
        peaks.flip_idx = _flip_test_idx(new_skeleton, out_names)
    logger.info(
        f"Node filter: {len(keep)} of {len(channels)} confmap channels "
        f"({keep[:5]}{' ...' if len(keep) > 5 else ''})"
        + (f", written as {out_names[:5]}" if out_names != keep else "")
    )
    return keep


def filter_training_config(
    cfg, model_type: str, node_filter: NodeFilter, strip_prefix=None
):
    """A copy of a training config restricted to a node filter's nodes.

    For export: the copy's confmap ``part_names`` and ``data_config.skeletons`` (the
    first skeleton, restricted with :func:`restrict_skeleton`) describe the sliced
    head, so the export metadata and the exported ``training_config`` agree with
    the network. Load the checkpoint with the ORIGINAL config (the checkpoint
    holds every channel), then :func:`slice_head_conv` with the returned indices.

    Returns:
        ``(filtered_cfg, channel_indices)``.

    Raises:
        ValueError: If ``model_type`` has no sliceable confmap head, or the filter
            selects nothing.
    """
    import yaml
    from omegaconf import OmegaConf
    from sleap_io.io.skeleton import SkeletonYAMLEncoder

    from sleap_nn.inference.utils import get_skeleton_from_config

    if model_type not in NODE_FILTER_HEADS:
        raise ValueError(
            "The node filter is implemented for single_instance and "
            f"centered_instance models only, not {model_type}."
        )
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    skeleton = get_skeleton_from_config(cfg.data_config.skeletons)[0]
    head_cfg = cfg.model_config.head_configs[model_type].confmaps
    channels = list(head_cfg.part_names or skeleton.node_names)
    idx = resolve_node_filter(channels, node_filter)
    keep = [channels[i] for i in idx]
    out_names = output_names(keep, node_filter, strip_prefix)
    restricted = restrict_skeleton(skeleton, keep, dict(zip(keep, out_names)))
    # Stored as the trainer stores skeletons: the encoder's {name: skeleton}
    # entry with its name inside.
    encoded = yaml.safe_load(SkeletonYAMLEncoder().encode([restricted]))
    ((name, entry),) = encoded.items()
    entry["name"] = name
    skeletons = OmegaConf.to_container(cfg.data_config.skeletons)
    skeletons[0] = entry
    cfg.data_config.skeletons = skeletons
    head_cfg.part_names = list(out_names)
    anchor = head_cfg.get("anchor_part", None)
    if anchor is not None and anchor not in keep:
        # The anchor only places crops (stage 1 / GT centroids), which an export of
        # the crop network does not do; a dropped anchor has no channel to name.
        head_cfg.anchor_part = None
    elif anchor is not None:
        head_cfg.anchor_part = dict(zip(keep, out_names))[anchor]
    return cfg, idx
