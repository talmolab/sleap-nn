"""Train one frame-level model on labels files with different skeletons (experimental).

With ``data_config.merge_skeletons: true``, ``bottomup``, ``centroid`` and
``single_instance`` models accept training files whose skeletons differ (for
example one prefixed node set per source, ``<source>/<node>``). The model's
skeleton is the union of the training files' nodes, matched by name, with every
file's edges and symmetries (:func:`pool_skeleton`, the same union the warp
centered-instance pipeline builds). Each file's instances are then mapped onto that
skeleton by node name (:func:`map_labels_to_skeleton`): the nodes a file does not
have become unlabeled points (sleap-io ``labeled=False``), and the file's
``provenance["labeled_nodes"]`` lists the nodes it does have. The per-node loss
weights of :mod:`sleap_nn.data.node_weights` then give those nodes zero loss, and a
bottom-up part-affinity edge counts only where both of its ends are labeled.

The settings in :data:`OPT_IN_KEYS` are written to ``training_config.yaml`` only when
a run uses them (:func:`config_for_saving`), so configs of runs without them stay
readable by releases that do not know them.
"""

import copy
from typing import List, Optional, Sequence

import sleap_io as sio
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from sleap_nn.data.node_weights import LABELED_NODES_KEY

#: Model types that train on whole frames and support ``merge_skeletons``.
MERGE_MODEL_TYPES = ("bottomup", "centroid", "single_instance")

#: ``(parent key, key, default)`` of the opt-in settings :func:`config_for_saving`
#: leaves out of a saved config while they hold their default.
OPT_IN_KEYS = (
    ("data_config", "merge_skeletons", False),
    ("data_config.preprocessing", "file_scales", None),
)

#: The head-config path whose ``part_names`` orders the union skeleton.
_PART_NAMES_KEYS = {
    "bottomup": "model_config.head_configs.bottomup.confmaps.part_names",
    "single_instance": "model_config.head_configs.single_instance.confmaps.part_names",
    "centered_instance": (
        "model_config.head_configs.centered_instance.confmaps.part_names"
    ),
}


def merge_skeletons_enabled(
    config: DictConfig, model_type: Optional[str] = None
) -> bool:
    """Whether ``data_config.merge_skeletons`` is on.

    Raises:
        ValueError: If it is on and ``model_type`` (when given) is not in
            :data:`MERGE_MODEL_TYPES`.
    """
    merge = bool(OmegaConf.select(config, "data_config.merge_skeletons", default=False))
    if merge and model_type is not None and model_type not in MERGE_MODEL_TYPES:
        raise ValueError(
            "data_config.merge_skeletons is supported for the "
            f"{', '.join(MERGE_MODEL_TYPES)} model types, not {model_type!r} "
            "(centered_instance with preprocessing.crop_pipeline='warp' merges "
            "skeletons on its own)."
        )
    return merge


def pool_node_names(
    config: DictConfig, model_type: str, labels_list: Sequence[sio.Labels]
) -> List[str]:
    """The union skeleton's node names.

    The head's ``part_names`` when set, else every node of the files' first
    skeletons, in the order first seen.
    """
    key = _PART_NAMES_KEYS.get(model_type)
    names = OmegaConf.select(config, key, default=None) if key else None
    if names:
        return list(names)
    return list(
        dict.fromkeys(
            n for lb in labels_list for sk in lb.skeletons[:1] for n in sk.node_names
        )
    )


def pool_skeleton(
    labels_list: Sequence[sio.Labels],
    node_names: Sequence[str],
    edges: Optional[Sequence[Sequence[str]]] = None,
) -> sio.Skeleton:
    """The union skeleton on ``node_names``.

    Every file's edges and symmetries between those nodes
    (:func:`~sleap_nn.data.topdown_warp.merged_skeleton`), or ``edges`` (pairs of
    node names) in place of the files' edges when given.

    Raises:
        ValueError: If an edge in ``edges`` names a node not in ``node_names``.
    """
    from sleap_nn.data.topdown_warp import merged_skeleton

    skeleton = merged_skeleton(labels_list, list(node_names))
    if edges:
        pairs = [tuple(e) for e in edges]
        unknown = sorted({n for e in pairs for n in e} - set(node_names))
        if unknown:
            raise ValueError(
                f"Edges name nodes {unknown} that are not in the merged skeleton "
                f"{list(node_names)}."
            )
        skeleton = sio.Skeleton(
            nodes=list(node_names),
            edges=pairs,
            symmetries=[tuple(n.name for n in s.nodes) for s in skeleton.symmetries],
            name=skeleton.name,
        )
    return skeleton


def file_node_names(labels: sio.Labels, node_names: Sequence[str]) -> List[str]:
    """The nodes of ``node_names`` (in that order) that a labels file labels.

    Those of its skeletons, narrowed by its ``provenance["labeled_nodes"]`` when it
    has one (node names, or indices into its first skeleton).
    """
    own = {n for sk in labels.skeletons for n in sk.node_names}
    mask = (labels.provenance or {}).get(LABELED_NODES_KEY, None)
    if mask is not None:
        first = labels.skeletons[0].node_names if labels.skeletons else []
        listed = set()
        for node in mask:
            if isinstance(node, int) and not isinstance(node, bool):
                if 0 <= node < len(first):
                    listed.add(first[node])
            else:
                listed.add(node)
        own &= listed
    return [n for n in node_names if n in own]


def own_node_set(labels: sio.Labels, config: DictConfig) -> set:
    """The node names a training or validation file is shown and scored on.

    Those of its skeletons; with ``merge_skeletons`` (where every mapped file has
    the union skeleton) only those it labels (:func:`file_node_names`).
    """
    names = {n for sk in labels.skeletons for n in sk.node_names}
    if merge_skeletons_enabled(config):
        return set(file_node_names(labels, sorted(names)))
    return names


def file_anchor_part(labels: sio.Labels, anchor_part) -> Optional[str]:
    """The first node of an ``anchor_part`` list that a labels file labels.

    The anchor a centroid model trained with a list of anchors (experimental)
    uses for that file (:func:`file_node_names`); ``None`` when it labels none.
    """
    from sleap_nn.data.instance_centroids import anchor_part_names

    own = file_node_names(labels, anchor_part_names(anchor_part))
    return own[0] if own else None


def _map_instance(instance, skeleton: sio.Skeleton):
    """A copy of ``instance`` on ``skeleton``; nodes it lacks are unlabeled."""
    mapped = copy.copy(instance)
    mapped.replace_skeleton(skeleton, unmapped="unlabeled")
    return mapped


def map_labels_to_skeleton(labels: sio.Labels, skeleton: sio.Skeleton) -> sio.Labels:
    """A copy of ``labels`` whose instances are on ``skeleton``, matched by name.

    Frames and instances are copied (the input is not changed). The nodes a file
    does not have become unlabeled points, and ``provenance["labeled_nodes"]``
    lists the nodes it has (combined with any mask the file already carried), so
    they get zero loss under ``node_loss_mask``. Points of nodes not in
    ``skeleton`` are dropped.

    The copy keeps only what the frame models read: frames, instances, videos,
    tracks, identities, suggestions, the skeleton and the provenance. Sessions,
    events, categories and ROIs are left out, and a frame's other annotations
    (centroids, boxes, masks) are shared with the input, so any link they hold
    to an instance points at the original, unmapped one.
    """
    own = file_node_names(labels, skeleton.node_names)
    frames = []
    for lf in labels.labeled_frames:
        new = copy.copy(lf)
        new.instances = [_map_instance(inst, skeleton) for inst in lf.instances]
        frames.append(new)
    provenance = dict(labels.provenance or {})
    provenance[LABELED_NODES_KEY] = own
    return sio.Labels(
        labeled_frames=frames,
        videos=labels.videos,
        skeletons=[skeleton],
        tracks=labels.tracks,
        identities=labels.identities,
        suggestions=labels.suggestions,
        provenance=provenance,
    )


def _file_label(labels: sio.Labels, kind: str, index: int) -> str:
    """``"<kind> file <index> (<path>)"`` for messages."""
    prov = labels.provenance or {}
    path = prov.get("filename") or prov.get("source_labels")
    return f"{kind} file {index}" + (f" ({path})" if path else "")


def map_labels_lists(
    labels: Sequence[sio.Labels],
    val_labels: Optional[Sequence[sio.Labels]],
    skeleton: sio.Skeleton,
):
    """:func:`map_labels_to_skeleton` over the training and validation files.

    Logs the nodes each training file labels.

    Raises:
        ValueError: If a file has no node of ``skeleton``: it would add no frame
            (its instances would all be empty).
    """
    out = []
    for kind, lbs in (("training", labels), ("validation", val_labels)):
        if lbs is None:
            out.append(None)
            continue
        mapped = []
        for i, lb in enumerate(lbs):
            m = map_labels_to_skeleton(lb, skeleton)
            own = m.provenance[LABELED_NODES_KEY]
            if not own:
                raise ValueError(
                    f"data_config.merge_skeletons: {_file_label(lb, kind, i)} "
                    f"labels no node of the merged skeleton {skeleton.node_names}, "
                    "so it would add nothing. Remove it, or add its nodes to the "
                    "head's part_names."
                )
            if kind == "training":
                logger.info(
                    f"Merged skeleton: {_file_label(lb, kind, i)} labels "
                    f"{len(own)} of {len(skeleton.node_names)} nodes."
                )
            mapped.append(m)
        out.append(mapped)
    return out[0], out[1]


def config_for_saving(config: DictConfig) -> DictConfig:
    """``config`` without the :data:`OPT_IN_KEYS` that hold their default.

    Returns ``config`` itself when there is nothing to leave out.
    """
    drop = [
        (parent, key)
        for parent, key, default in OPT_IN_KEYS
        if (node := OmegaConf.select(config, parent, default=None)) is not None
        and key in node
        and node[key] == default
    ]
    if not drop:
        return config
    out = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    for parent, key in drop:
        del OmegaConf.select(out, parent)[key]
    return out
