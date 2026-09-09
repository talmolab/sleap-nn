"""Conversion helpers for centroid-only output representation.

Single place that owns the mapping between sleap-nn's centroid predictions
and ``sleap_io`` objects, so the representation (single-node ``PredictedInstance``
vs ``sio.PredictedCentroid``) and the ``source`` tag stay consistent across
``outputs.py`` / ``predictor.py`` / ``writer.py``.

**#586 consistency.** The centroid's *meaning* is defined by
``sleap_nn/data/instance_centroids.py::generate_centroids`` (shared by training
target generation AND GT-centroid inference), and is selected by the head
config's ``centroid_method`` / ``centroid_fallback`` (resolved by
``resolve_centroid_method``). The ``source`` tag we record mirrors that, in the
``sio.Centroid.from_instance(method=...)`` vocabulary:

* an explicit, configured anchor node  -> ``"anchor:<node>"``
* no anchor configured                 -> the reduce method itself,
  ``"center_of_mass"`` (the default), ``"bbox_center"`` or
  ``"geometric_median"``

so the recorded metadata never contradicts the trained target. Keep this in
lockstep with ``generate_centroids``: whatever ``resolve_centroid_method``
returns for a config is what a model trained on it predicts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import sleap_io as sio

# Default ``source`` when no explicit anchor node is configured: the centroid is
# the mean of visible nodes (``generate_centroids`` fallback == sio
# ``center_of_mass``).
CENTER_OF_MASS = "center_of_mass"


def centroid_source_for_anchor(
    anchor_ind: Optional[int],
    node_names: Optional[List[str]] = None,
    centroid_method: Optional[str] = None,
) -> str:
    """Return the ``sio.Centroid.source`` tag for a centroid model.

    Args:
        anchor_ind: The configured anchor-node index (``None`` when the model
            was trained with no explicit anchor — see :data:`CENTER_OF_MASS`).
        node_names: The training skeleton's node names, used to resolve a
            human-readable ``"anchor:<name>"`` tag. When unavailable, the
            index is used (``"anchor:<ind>"``).
        centroid_method: The resolved method the model was trained with, one of
            ``sleap_nn.data.instance_centroids.CENTROID_METHODS``. ``None``
            (default) infers it from ``anchor_ind``, matching
            ``resolve_centroid_method``.

    Returns:
        ``"anchor:<node-name-or-index>"`` for the anchor method; otherwise the
        reduce method itself (``"center_of_mass"`` by default).

    Note:
        This is a MODEL-level descriptor, not per-instance: a trained centroid
        model with an anchor will have learned to predict that anchor when
        visible and (per #586) its fallback otherwise, but we cannot recover
        per-instance visibility from a bare centroid peak. ``anchor:*`` is the
        closest faithful model-level tag.
    """
    if centroid_method is None:
        centroid_method = "anchor" if anchor_ind is not None else CENTER_OF_MASS
    if centroid_method != "anchor":
        return centroid_method
    if anchor_ind is None:
        return CENTER_OF_MASS
    if node_names is not None and 0 <= anchor_ind < len(node_names):
        return f"anchor:{node_names[anchor_ind]}"
    return f"anchor:{anchor_ind}"


def build_predicted_centroid(
    x: float,
    y: float,
    score: float,
    *,
    track: Optional["sio.Track"] = None,
    tracking_score: Optional[float] = None,
    source: str = CENTER_OF_MASS,
) -> "sio.PredictedCentroid":
    """Construct a ``sio.PredictedCentroid`` from a predicted centroid peak.

    Args:
        x: Centroid x-coordinate in original-image pixels.
        y: Centroid y-coordinate in original-image pixels.
        score: Per-instance confidence (the centroid peak value).
        track: Optional ``sio.Track`` assignment.
        tracking_score: Optional per-instance tracking score.
        source: The ``source`` method tag (see :func:`centroid_source_for_anchor`).

    Returns:
        A ``sio.PredictedCentroid``. ``PredictedCentroid`` carries only an
        instance-level ``score`` (no per-node slot), per the representation
        contract.
    """
    import sleap_io as sio

    kwargs = {}
    if track is not None:
        kwargs["track"] = track
    if tracking_score is not None:
        kwargs["tracking_score"] = tracking_score
    return sio.PredictedCentroid(
        x=float(x),
        y=float(y),
        score=float(score),
        source=source,
        **kwargs,
    )
