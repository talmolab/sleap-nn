"""Handle calculation of instance centroids.

**Centroid methods (#586).** A centroid can be derived from an instance's points in
several ways, and sleap-nn now exposes the same four-way vocabulary as
``sleap_io``'s ``Instance.to_centroid`` / ``SegmentationMask.to_centroid``:

===================== =========================================================
``center_of_mass``    Mean of the visible nodes (the historical default).
``bbox_center``       Midpoint of the visible nodes' bounding box.
``geometric_median``  Weiszfeld geometric median of the visible nodes; robust to
                      a few outlying nodes, so it tracks the body center better
                      for elongated or curled animals.
``anchor``            A named node, with a reduce-method fallback when that node
                      is not visible.
===================== =========================================================

Two levels compute centroids and **must agree**: this batched torch op (training
targets, top-down crop centers, GT-centroid inference) and ``to_centroid`` at the
object level. :func:`resolve_centroid_method` is the single place that turns the
config pair (``anchor_part``, ``centroid_method``/``centroid_fallback``) into the
``(method, fallback)`` argument pair both levels take, so there is one spelling of
these concepts across sleap-nn and sleap-io. ``tests/data/test_instance_centroids.py``
asserts the two levels agree per method.

**Divergence from ``sleap_io``, deliberate.** ``sleap_io`` decides node visibility
from the x-coordinate alone (``~isnan(pts[:, 0])``); the reductions here count
non-NaN values *per axis* (``find_points_mean``, #584) or per point
(``find_points_geometric_median``). The two agree whenever a node's coordinates are
NaN together — which is what ``sleap_io`` itself writes — and differ only for a
half-NaN point, where this module's answer is the better-defined one.
"""

from typing import Optional

import torch
from loguru import logger

#: Centroid derivation methods, spelled as in ``sleap_io``'s ``to_centroid``.
CENTROID_METHODS = ("center_of_mass", "bbox_center", "geometric_median", "anchor")

#: The subset that reduces a whole point set (i.e. everything but ``"anchor"``).
#: These are the valid values for an anchor *fallback*.
REDUCE_METHODS = ("center_of_mass", "bbox_center", "geometric_median")

#: Fallback applied when a configured anchor node is not visible. Unlike
#: ``sio.Instance.to_centroid``, whose ``fallback=None`` yields a NaN centroid,
#: sleap-nn always falls back — a NaN training target for a partially-visible
#: instance would be a silent data loss. This is the pre-#586 behavior.
DEFAULT_ANCHOR_FALLBACK = "center_of_mass"


def find_points_mean(points: torch.Tensor) -> torch.Tensor:
    """Find the mean position of a set of points, ignoring NaNs.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
            i.e., rank >= 2.

    Returns:
        The NaN-ignoring mean across the ``n_points`` axis. Output shape ``(..., 2)``
        (rank reduced by 1). When *every* point in a slot is NaN, the corresponding
        output row is NaN.
    """
    mask = ~torch.isnan(points)
    # Count non-NaN values PER AXIS so each coordinate's mean divides by the
    # number of non-NaN values on that axis. Counting per-point (mask.any)
    # biased the mean toward 0 when only one coordinate of a point was NaN
    # (#584). For the common all-or-nothing-per-point case the two are equal.
    counts = mask.sum(dim=-2).clamp(min=1).to(points.dtype)  # (..., 2)
    safe = torch.where(mask, points, torch.zeros_like(points))
    means = safe.sum(dim=-2) / counts
    all_nan = (~mask.any(dim=-1)).all(dim=-1, keepdim=True)
    return torch.where(all_nan, torch.full_like(means, float("nan")), means)


def find_points_bbox_midpoint(points: torch.Tensor) -> torch.Tensor:
    """Find the midpoint of the bounding box of a set of points.

    Retained as a utility for callers that explicitly want bbox-midpoint behavior.
    The canonical anchor fallback used by :func:`generate_centroids` is
    :func:`find_points_mean` (mean of visible nodes) — see that function for the
    project-wide convention.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
            i.e., rank >= 2.

    Returns:
        The midpoints between the bounds of each set of points. The output will be of
        shape (..., 2), reducing the rank of the input by 1. NaNs will be ignored in the
        calculation.

    Notes:
        The midpoint is calculated as:
            xy_mid = xy_min + ((xy_max - xy_min) / 2)
                   = ((2 * xy_min) / 2) + ((xy_max - xy_min) / 2)
                   = (2 * xy_min + xy_max - xy_min) / 2
                   = (xy_min + xy_max) / 2
    """
    pts_min = torch.min(
        torch.where(torch.isnan(points), torch.inf, points), dim=-2
    ).values
    pts_max = torch.max(
        torch.where(torch.isnan(points), -torch.inf, points), dim=-2
    ).values

    return (pts_max + pts_min) * 0.5


def find_points_geometric_median(
    points: torch.Tensor,
    max_iter: int = 100,
    tol: float = 1e-6,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Find the geometric median of a set of points via Weiszfeld's algorithm.

    The geometric median minimizes the sum of Euclidean distances to the input
    points, which makes it markedly more robust than the mean to a few badly
    localized or anatomically extreme nodes — a curled tail pulls
    :func:`find_points_mean` off the body, but barely moves this.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_points, 2),
            i.e., rank >= 2.
        max_iter: Maximum Weiszfeld iterations.
        tol: Convergence tolerance on the estimate's movement between iterations.
            Iteration stops once *every* slot has moved less than this.
        eps: Distances below this are treated as coincident with the estimate and
            dropped from the reweighting (their weight would be unbounded).

    Returns:
        The geometric medians. The output will be of shape (..., 2), reducing the
        rank of the input by 1. Slots whose points are all NaN return NaN.

    Notes:
        A point is used only when BOTH of its coordinates are non-NaN — the
        estimate is a joint 2D quantity, so per-axis visibility (which
        :func:`find_points_mean` uses) has no meaning here. Matches the algorithm
        in ``sleap_io.model.centroid._geometric_median`` (initialized at the
        arithmetic mean, inverse-distance reweighting) so the object level and
        this batched op agree.

        The iteration runs in float64 regardless of the input dtype, and the
        result is cast back. Weiszfeld reweights by ``1 / distance``, so near a
        Fermat point that sits on one of the input nodes the distances approach
        zero and float32 loses all relative precision there: measured against the
        numpy reference on random 13-node instances, float32 drifts up to 0.18 px
        while float64 agrees to ~1e-13. The tensors involved are a few hundred
        floats, so the promotion costs nothing worth measuring — and it is what
        lets the object/tensor parity test assert equality rather than a loose
        tolerance.
    """
    out_dtype = points.dtype
    points = points.to(torch.float64)

    visible = ~torch.isnan(points).any(dim=-1)  # (..., n_points)
    vis_f = visible.to(points.dtype).unsqueeze(-1)  # (..., n_points, 1)
    # Zero out invisible points so they contribute nothing to any sum; they are
    # excluded from every weight by `visible`, so the zeros are never read back.
    pts = torch.where(visible.unsqueeze(-1), points, torch.zeros_like(points))

    # Initialize at the arithmetic mean of the visible points.
    counts = vis_f.sum(dim=-2).clamp(min=1)  # (..., 1)
    estimate = pts.sum(dim=-2) / counts  # (..., 2)

    # Convergence is tracked PER SLOT, not for the batch as a whole. Stopping the
    # whole loop once every slot has converged would let a slow-converging slot
    # keep iterating while its neighbours are done -- so an instance's centroid
    # would depend on which other instances shared its batch. Freezing each slot
    # as it converges makes the result batch-invariant and identical to the
    # per-instance numpy reference.
    active = torch.ones_like(estimate[..., 0], dtype=torch.bool)
    for _ in range(max_iter):
        dist = torch.linalg.vector_norm(pts - estimate.unsqueeze(-2), dim=-1)
        # Points coincident with the current estimate would get an unbounded
        # weight; drop them, exactly as the numpy reference does.
        usable = visible & (dist > eps)
        weights = torch.where(usable, 1.0 / dist.clamp(min=eps), torch.zeros_like(dist))
        wsum = weights.sum(dim=-1, keepdim=True)  # (..., 1)
        update = (weights.unsqueeze(-1) * pts).sum(dim=-2) / wsum.clamp(min=eps)
        # wsum == 0 means every visible point coincides with the estimate (or
        # there are none): the estimate is already the answer.
        new_estimate = torch.where(wsum > 0, update, estimate)
        shift = torch.linalg.vector_norm(new_estimate - estimate, dim=-1)
        # The converging step is applied before the slot is frozen, matching the
        # reference's `estimate = new_estimate; break` ordering.
        estimate = torch.where(active.unsqueeze(-1), new_estimate, estimate)
        active = active & (shift >= tol)
        if not bool(active.any()):
            break

    none_visible = ~visible.any(dim=-1, keepdim=True)
    estimate = torch.where(
        none_visible, torch.full_like(estimate, float("nan")), estimate
    )
    return estimate.to(out_dtype)


def reduce_points(points: torch.Tensor, method: str) -> torch.Tensor:
    """Reduce a set of points to one centroid by the named method.

    Args:
        points: A torch.Tensor of shape (..., n_points, 2), i.e., rank >= 2.
        method: One of :data:`REDUCE_METHODS` — ``"center_of_mass"``,
            ``"bbox_center"`` or ``"geometric_median"``. ``"anchor"`` is not a
            reduction (it selects a node) and is rejected here; see
            :func:`generate_centroids`.

    Returns:
        The centroids, of shape (..., 2).

    Raises:
        ValueError: If ``method`` is not a known reduce method.
    """
    if method == "center_of_mass":
        return find_points_mean(points)
    if method == "bbox_center":
        return find_points_bbox_midpoint(points)
    if method == "geometric_median":
        return find_points_geometric_median(points)
    message = (
        f"Unknown centroid reduce method {method!r}. Expected one of "
        f"{', '.join(repr(m) for m in REDUCE_METHODS)}."
    )
    raise ValueError(message)


def resolve_centroid_method(
    anchor_part: Optional[str] = None,
    centroid_method: Optional[str] = None,
    centroid_fallback: Optional[str] = None,
) -> tuple:
    """Resolve the head config's centroid knobs into ``(method, fallback)``.

    The single place that maps sleap-nn's config fields onto ``sleap_io``'s
    ``to_centroid`` vocabulary, so the object level, the batched tensor level
    (:func:`generate_centroids`) and the recorded ``sio.Centroid.source`` tag can
    never disagree about what a model's centroid means.

    Args:
        anchor_part: The configured anchor node name, or ``None``. Setting it
            implies ``method="anchor"`` — that is the pre-#586 behavior and stays
            the meaning of a config that predates ``centroid_method``.
        centroid_method: One of :data:`CENTROID_METHODS`, or ``None`` (default) to
            infer: ``"anchor"`` when ``anchor_part`` is set, else
            ``"center_of_mass"``.
        centroid_fallback: The reduce method used when the anchor node is not
            visible. One of :data:`REDUCE_METHODS`, or ``None`` for
            :data:`DEFAULT_ANCHOR_FALLBACK`. Only meaningful for the anchor method.

    Returns:
        ``(method, fallback)``, where ``method`` is a member of
        :data:`CENTROID_METHODS` and ``fallback`` is a member of
        :data:`REDUCE_METHODS` when ``method == "anchor"`` and ``None`` otherwise.

    Raises:
        ValueError: If a value is not in the vocabulary, if ``centroid_method`` is
            a reduce method while ``anchor_part`` is set (contradictory — the two
            name different centroids), or if ``"anchor"`` is requested without an
            ``anchor_part``.
    """
    if centroid_method is not None and centroid_method not in CENTROID_METHODS:
        message = (
            f"Unknown centroid_method {centroid_method!r}. Expected one of "
            f"{', '.join(repr(m) for m in CENTROID_METHODS)}."
        )
        raise ValueError(message)
    if centroid_fallback is not None and centroid_fallback not in REDUCE_METHODS:
        message = (
            f"Unknown centroid_fallback {centroid_fallback!r}. Expected one of "
            f"{', '.join(repr(m) for m in REDUCE_METHODS)} (an anchor cannot fall "
            f"back to another anchor)."
        )
        raise ValueError(message)

    if anchor_part is not None:
        if centroid_method is not None and centroid_method != "anchor":
            message = (
                f"Contradictory centroid config: anchor_part={anchor_part!r} asks "
                f"for the anchor node, but centroid_method={centroid_method!r} asks "
                f"for a whole-instance reduction. Set centroid_fallback="
                f"{centroid_method!r} to use it when the anchor is occluded, or "
                f"drop anchor_part to use it for every instance."
            )
            raise ValueError(message)
        return "anchor", centroid_fallback or DEFAULT_ANCHOR_FALLBACK

    if centroid_method == "anchor":
        message = (
            "centroid_method='anchor' requires anchor_part to name the node to "
            "anchor on."
        )
        raise ValueError(message)
    return (centroid_method or "center_of_mass"), None


def centroid_method_from_config(head_config) -> tuple:
    """Read and resolve the centroid knobs off a head-config leaf.

    Convenience wrapper over :func:`resolve_centroid_method` for the many callers
    that hold a head-config leaf (``head_configs.centroid.confmaps``,
    ``...centered_instance.confmaps``, ``...embedding.embedding``, ...). Missing
    keys resolve to ``None``, so a config written before #586 — or a plain dict —
    yields the historical behavior.

    Args:
        head_config: A head-config leaf (``DictConfig``, dataclass or mapping), or
            ``None``.

    Returns:
        ``(method, fallback)`` as documented on :func:`resolve_centroid_method`.
    """
    from omegaconf import OmegaConf

    def get(key):
        if head_config is None:
            return None
        if isinstance(head_config, dict):
            return head_config.get(key)
        if OmegaConf.is_config(head_config):
            return OmegaConf.select(head_config, key, default=None)
        return getattr(head_config, key, None)

    return resolve_centroid_method(
        anchor_part=get("anchor_part"),
        centroid_method=get("centroid_method"),
        centroid_fallback=get("centroid_fallback"),
    )


def degrade_anchor_if_unresolved(
    method: str, fallback: Optional[str], anchor_ind: Optional[int]
) -> tuple:
    """Degrade an ``"anchor"`` method to its fallback when the node is unresolvable.

    ``anchor_part`` names a node that may be absent from the skeleton — the
    centroid model deliberately tolerates this (an anchor is only a fallback path
    there), and the embedding dataset resolves the index leniently. Rather than
    raising from deep inside the batched op, degrade to the configured fallback,
    which is what the pre-#586 code did implicitly, and say so once.

    Args:
        method: The resolved method, as returned by :func:`resolve_centroid_method`.
        fallback: The resolved fallback.
        anchor_ind: The anchor node index, or ``None`` if it could not be resolved.

    Returns:
        ``(method, fallback)`` unchanged, unless the anchor is unresolvable, in
        which case ``(fallback, None)``.
    """
    if method != "anchor" or anchor_ind is not None:
        return method, fallback
    effective = fallback or DEFAULT_ANCHOR_FALLBACK
    logger.warning(
        f"Centroid anchor node could not be resolved against the skeleton; "
        f"deriving centroids with {effective!r} for every instance instead. Check "
        f"that `anchor_part` names a node in the skeleton."
    )
    return effective, None


def add_centroids_from_masks(
    labels,
    method: str = "center_of_mass",
    overwrite: bool = False,
) -> int:
    """Derive ``UserCentroid`` annotations from a labels' segmentation masks.

    Mask-only datasets (no pose annotations at all) cannot train a centroid model
    today: the confmap target needs either pose keypoints or first-class centroid
    annotations, and such labels have neither. ``sio.SegmentationMask.to_centroid``
    supplies the missing piece — one call per mask, carrying the mask's ``track`` /
    ``identity`` / ``instance`` linkage — after which the ordinary
    ``centroid_source="user"`` path takes over unchanged. Nothing downstream of
    this function knows the centroids came from masks.

    Args:
        labels: An ``sio.Labels`` to annotate **in place**.
        method: The derivation method, one of :data:`REDUCE_METHODS`. Masks have
            no nodes, so ``"anchor"`` is meaningless here.
        overwrite: If ``False`` (default), frames that already carry user
            centroids are left alone — a real annotation always outranks a
            derived one. If ``True``, derived centroids replace them.

    Returns:
        The number of centroids added.

    Raises:
        ValueError: If ``method`` is not a mask-applicable reduce method.
    """
    if method not in REDUCE_METHODS:
        message = (
            f"centroids_from_masks: unknown method {method!r}. Expected one of "
            f"{', '.join(repr(m) for m in REDUCE_METHODS)} (a mask has no nodes, "
            f"so 'anchor' does not apply)."
        )
        raise ValueError(message)

    n_added = 0
    n_frames = 0
    for lf in labels:
        masks = getattr(lf, "masks", None)
        if not masks:
            continue
        existing = [c for c in getattr(lf, "centroids", []) if not c.is_predicted]
        if existing and not overwrite:
            continue
        if existing and overwrite:
            lf.centroids = [c for c in lf.centroids if c.is_predicted]
        added_here = 0
        for mask in masks:
            if getattr(mask, "is_predicted", False):
                continue
            centroid = mask.to_centroid(method=method)
            # A mask can be empty (every pixel background) after a filter or a
            # bad annotation; `to_centroid` returns NaN rather than raising.
            if centroid.x != centroid.x or centroid.y != centroid.y:
                continue
            lf.centroids.append(centroid)
            added_here += 1
        n_added += added_here
        n_frames += bool(added_here)

    if n_added:
        logger.info(
            f"Derived {n_added} centroid annotation(s) from segmentation masks "
            f"across {n_frames} frame(s) using method={method!r}."
        )
    else:
        logger.warning(
            "centroids_from_masks is enabled but no centroids were derived: the "
            "labels carry no user segmentation masks (or every frame already has "
            "user centroids)."
        )
    return n_added


def generate_centroids(
    points: torch.Tensor,
    anchor_ind: Optional[int] = None,
    method: Optional[str] = None,
    fallback: Optional[str] = None,
) -> torch.Tensor:
    """Return centroids derived from instance points by the configured method.

    Args:
        points: A torch.Tensor of dtype torch.float32 and of shape (..., n_nodes, 2),
            i.e., rank >= 2.
        anchor_ind: The index of the node to use as the anchor for the centroid.
            Required by (and only used by) ``method="anchor"``. If the anchor node
            is NaN (not visible) for a given instance, that instance's centroid
            falls back to ``fallback``.
        method: One of :data:`CENTROID_METHODS`. ``None`` (default) infers it from
            ``anchor_ind`` — ``"anchor"`` when an index is given, else
            ``"center_of_mass"`` — which is exactly the pre-#586 behavior, so
            existing callers are unchanged. Use
            :func:`resolve_centroid_method` to derive this from a head config.
        fallback: The reduce method for a missing anchor, one of
            :data:`REDUCE_METHODS`. ``None`` means :data:`DEFAULT_ANCHOR_FALLBACK`.

    Returns:
        The centroids of the instances. The output will be of shape (..., 2),
        reducing the rank of the input by 1. NaNs will be ignored in the calculation.

    Raises:
        ValueError: If ``method="anchor"`` without an ``anchor_ind``, or if
            ``method``/``fallback`` is not in the vocabulary.

    Note:
        This op defines what a centroid *means* for training targets, top-down crop
        centers and GT-centroid inference alike; it must stay in lockstep with the
        object-level ``to_centroid`` (see the module docstring) and with the
        ``sio.Centroid.source`` tag written by
        ``sleap_nn.inference.centroid_convert``.
    """
    if method is None:
        method = "anchor" if anchor_ind is not None else "center_of_mass"

    if method != "anchor":
        return reduce_points(points, method)

    if anchor_ind is None:
        message = (
            "generate_centroids(method='anchor') requires anchor_ind, the index of "
            "the node to anchor on."
        )
        raise ValueError(message)
    if fallback is None:
        fallback = DEFAULT_ANCHOR_FALLBACK
    elif fallback not in REDUCE_METHODS:
        message = (
            f"Unknown anchor fallback {fallback!r}. Expected one of "
            f"{', '.join(repr(m) for m in REDUCE_METHODS)}."
        )
        raise ValueError(message)

    centroids = points[..., anchor_ind, :].clone()

    missing_anchors = torch.isnan(centroids).any(dim=-1)
    if missing_anchors.any():
        centroids[missing_anchors] = reduce_points(points[missing_anchors], fallback)

    return centroids  # (..., n_instances, 2)
