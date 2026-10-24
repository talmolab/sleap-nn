"""Per-node loss weights for training on partially labeled skeletons.

When datasets with different keypoint schemas are merged into one union skeleton,
each source labels only some of the nodes. A node the source does not annotate must
be *unlabeled* (zero loss weight), not trained as "absent" (an all-zero confmap).

Confmap datasets (single-instance, centered-instance, bottom-up) emit a
``node_weights`` sample key of shape ``(1, n_nodes)`` whenever weighting is active
(see :func:`node_weights_active`). The weight of node ``n`` is the product of:

1. the per-file mask from ``Labels.provenance["labeled_nodes"]`` (a list of node
   names or indices the file annotates; all ones when the key is absent). This is
   the stopgap until sleap-io exposes per-point flags.
2. per-point ``labeled`` flags on the instance (sleap-io >= 0.10 / SLP 3.0), read
   automatically when the point array has a ``labeled`` field: ``labeled | visible``.
3. with ``unlabeled_points="ignore"``, whether the point has a location in the
   model input (finite and inside the image/crop after augmentation). Under the
   default ``"absent"`` such points keep weight 1 and are trained as absent
   (all-zero confmap), as before.

Negative frames keep weight 1 on every node. Flip augmentation swaps the weights
with the symmetric node channels on the same random draw.

``supervise_occluded`` decides what happens to *labeled but occluded* points that
carry estimated coordinates (``labeled & ~visible`` with finite xy): ``True`` (the
default) trains them as Gaussians at the estimated location, ``False`` gives them
zero weight. This is a no-op until sleap-io exposes the ``labeled`` flag: today an
invisible point is NaN in ``Instance.numpy()``.

The losses that consume the weights live in :mod:`sleap_nn.training.losses`
(:func:`~sleap_nn.training.losses.compute_weighted_mse_loss`,
:func:`~sleap_nn.training.losses.compute_weighted_ohkm_loss`).
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import sleap_io as sio
import torch
from loguru import logger

LABELED_NODES_KEY = "labeled_nodes"

UNLABELED_POINTS_MODES = ("absent", "ignore")


def get_labeled_nodes_weights(
    labels: sio.Labels, node_names: Optional[Sequence[str]] = None
) -> np.ndarray:
    """Return the per-file ``(n_nodes,)`` node mask from ``provenance``.

    Args:
        labels: Labels whose ``provenance["labeled_nodes"]`` lists the node names
            (or integer indices) the file annotates.
        node_names: Node order of the model. Defaults to the first skeleton's.

    Returns:
        ``float32`` array of ones (labeled) and zeros (unlabeled). All ones when
        the key is absent, so files without it train exactly as before.
    """
    if node_names is None:
        node_names = labels.skeletons[0].node_names if labels.skeletons else []
    node_names = list(node_names)
    weights = np.ones(len(node_names), dtype=np.float32)
    labeled_nodes = (labels.provenance or {}).get(LABELED_NODES_KEY, None)
    if labeled_nodes is None:
        return weights

    weights[:] = 0.0
    unknown = []
    for node in labeled_nodes:
        if isinstance(node, (int, np.integer)) and not isinstance(node, bool):
            if 0 <= int(node) < len(node_names):
                weights[int(node)] = 1.0
            else:
                unknown.append(node)
        elif node in node_names:
            weights[node_names.index(node)] = 1.0
        else:
            unknown.append(node)
    if unknown:
        logger.warning(
            f"provenance['{LABELED_NODES_KEY}'] lists {len(unknown)} node(s) not in "
            f"the skeleton (ignored): {unknown[:5]}"
        )
    if weights.sum() == 0:
        logger.warning(
            f"provenance['{LABELED_NODES_KEY}'] matches no skeleton node; every node "
            "of this file gets zero loss weight."
        )
    return weights


def has_partial_labeling_info(labels_list: Sequence[Optional[sio.Labels]]) -> bool:
    """Whether any file carries per-node labeling info.

    True if some file has ``provenance["labeled_nodes"]`` or its points expose a
    per-point ``labeled`` flag (sleap-io >= 0.10).
    """
    for labels in labels_list:
        if labels is None:
            continue
        if LABELED_NODES_KEY in (labels.provenance or {}):
            return True
        for lf in labels.labeled_frames[:1]:
            for inst in lf.instances[:1]:
                names = getattr(getattr(inst, "points", None), "dtype", None)
                if names is not None and "labeled" in (names.names or ()):
                    return True
    return False


def node_weights_active(
    labels_list: Sequence[Optional[sio.Labels]],
    node_loss_mask: bool = True,
    unlabeled_points: str = "absent",
) -> bool:
    """Whether a dataset over ``labels_list`` should emit ``node_weights``.

    Weights are emitted only when they can differ from 1: some file carries
    partial-labeling info, or ``unlabeled_points="ignore"``. Otherwise the batch
    has no ``node_weights`` and the loss is exactly the unweighted one.

    Args:
        labels_list: The dataset's labels files.
        node_loss_mask: Master switch (``data_config.node_loss_mask``).
        unlabeled_points: ``data_config.unlabeled_points``.

    Raises:
        ValueError: If ``unlabeled_points`` is not one of
            :data:`UNLABELED_POINTS_MODES`.
    """
    if unlabeled_points not in UNLABELED_POINTS_MODES:
        raise ValueError(
            f"unlabeled_points must be one of {UNLABELED_POINTS_MODES}, got "
            f"{unlabeled_points!r}."
        )
    if not node_loss_mask:
        return False
    return unlabeled_points == "ignore" or has_partial_labeling_info(labels_list)


def carry_labeled_nodes(source: sio.Labels, *splits: sio.Labels) -> None:
    """Copy ``provenance["labeled_nodes"]`` from ``source`` onto its splits.

    ``sio.Labels.split`` / ``make_training_splits`` (sleap-io 0.9.x) and the
    group-aware splitter build new ``Labels`` with a fresh provenance, which would
    silently turn the file's node mask off. Splits that already carry the key are
    left alone.

    Args:
        source: The labels the splits were made from.
        *splits: The split labels (modified in place).
    """
    labeled_nodes = (source.provenance or {}).get(LABELED_NODES_KEY, None)
    if labeled_nodes is None:
        return
    for split in splits:
        if LABELED_NODES_KEY not in split.provenance:
            split.provenance[LABELED_NODES_KEY] = list(labeled_nodes)


def instance_node_weights(
    instance: sio.Instance,
    file_weights: np.ndarray,
    points: np.ndarray,
    supervise_occluded: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine file and per-point labeled flags into one instance's node weights.

    Args:
        instance: The instance (its point array is checked for ``labeled`` /
            ``visible`` fields).
        file_weights: ``(n_nodes,)`` per-file mask from
            :func:`get_labeled_nodes_weights`.
        points: ``(n_nodes, 2)`` coordinates used for the targets (from
            ``instance.numpy()``; invisible points are NaN).
        supervise_occluded: See the module docstring.

    Returns:
        ``(weights, points)``: ``(n_nodes,)`` float32 weights and the target
        coordinates, where occluded points with estimated coordinates are
        restored when ``supervise_occluded`` is ``True``.
    """
    weights = np.asarray(file_weights, dtype=np.float32).copy()
    pts_arr = getattr(instance, "points", None)
    names = getattr(getattr(pts_arr, "dtype", None), "names", None) or ()
    if "labeled" not in names:
        # sleap-io < 0.10: no per-point labeled flag; only the file mask applies.
        return weights, points

    labeled = np.asarray(pts_arr["labeled"], dtype=bool)
    visible = np.asarray(pts_arr["visible"], dtype=bool)
    xy = np.asarray(pts_arr["xy"], dtype=np.float64)
    weights = weights * (labeled | visible).astype(np.float32)
    occluded = labeled & ~visible & np.isfinite(xy).all(axis=-1)
    if occluded.any():
        if supervise_occluded:
            points = points.copy()
            points[occluded] = xy[occluded]
        else:
            weights[occluded] = 0.0
    return weights, points


def finite_point_weights(points: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """``(n_nodes,)`` weights: 1 where a node has a finite location, else 0.

    Used by ``unlabeled_points="ignore"`` on the final (augmented, cropped,
    out-of-bounds-filtered) target points. With several instances (leading
    axes), a node counts as located if ANY instance has it.

    Args:
        points: ``(..., n_nodes, 2)`` coordinates.

    Returns:
        ``float32`` tensor of shape ``(n_nodes,)``.
    """
    points = torch.as_tensor(points)
    finite = torch.isfinite(points).all(dim=-1).reshape(-1, points.shape[-2])
    return finite.any(dim=0).to(torch.float32)
