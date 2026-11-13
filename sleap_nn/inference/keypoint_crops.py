"""Crops built from an earlier stage's keypoints: a part stage after top-down pose.

Some body parts leave the crop a top-down pose model sees -- a mouse's tail lies
outside the body box the detector and the body model use. A *part model* is a
second centered-instance model trained on crops around such a part (e.g. a square
centered on the tail base). At inference its crops have to come from the body
model's predictions, so detector, body and part run as one pipeline::

    detector -> body boxes -> body model -> keypoints
             -> KeypointCropRule -> part boxes -> part model -> part keypoints
             -> merged instances (body nodes + part nodes)

A :class:`KeypointCropRule` turns each predicted instance into one square box:

* center: the first located node of ``center`` (names matched exactly, or by the
  part after the last ``/``, so ``"TTI"`` matches ``"skel-60143b/TTI"``);
* half side: ``scale`` x the distance from the center to the mean of the
  ``reference`` nodes (all must be located), else of ``reference_fallback``,
  floored at ``min_half_side`` pixels;
* otherwise (no center, or no reference): ``fallback="extent"`` centers the box on
  the extent of the instance's located points with half side
  ``fallback_scale`` x half the extent's longer side; ``"skip"`` makes no box.

The part model then runs on those boxes through the box-seeded top-down route
(``boxes="given"``), and :func:`merge_part_predictions` appends its points to the
instance each box came from. Nothing here runs unless a part model is given
(:func:`sleap_nn.inference.run.predict` ``part_model_paths``), so the default
inference path is unchanged.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import sleap_io as sio
from loguru import logger

FALLBACKS = ("extent", "skip")


def _as_list(v: Union[None, str, Sequence[str]]) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    return [str(x) for x in v]


def _match(names: Sequence[str], wanted: str) -> Optional[int]:
    """Index of ``wanted`` in ``names``: exact, else by the part after the last '/'."""
    if wanted in names:
        return list(names).index(wanted)
    short = [n.rsplit("/", 1)[-1] for n in names]
    hits = [i for i, n in enumerate(short) if n == wanted.rsplit("/", 1)[-1]]
    return hits[0] if len(hits) == 1 else None


@dataclass
class KeypointCropRule:
    """How to build a part crop box from one predicted instance's keypoints.

    Attributes:
        center: Candidate names of the node the box is centered on; the first one
            that is in the skeleton and located is used.
        reference: Nodes whose mean point sets the box size (all must be located).
        reference_fallback: Used when a ``reference`` node is missing.
        scale: Half side of the box = ``scale`` x |center - reference point|.
        fallback: ``"extent"`` (box on the located points' extent) or ``"skip"``
            (no box) when the center or both references are missing.
        fallback_scale: Half side of an extent box = ``fallback_scale`` x half the
            extent's longer side.
        min_half_side: Floor of the half side in pixels.
    """

    center: List[str]
    reference: List[str] = field(default_factory=list)
    reference_fallback: List[str] = field(default_factory=list)
    scale: float = 2.0
    fallback: str = "extent"
    fallback_scale: float = 1.0
    min_half_side: float = 1.0

    def __post_init__(self):
        """Normalize node lists and check the fallback mode."""
        self.center = _as_list(self.center)
        self.reference = _as_list(self.reference)
        self.reference_fallback = _as_list(self.reference_fallback)
        if not self.center:
            raise ValueError("KeypointCropRule needs at least one center node.")
        if self.fallback not in FALLBACKS:
            raise ValueError(
                f"fallback must be one of {FALLBACKS}; got {self.fallback!r}."
            )
        if self.scale <= 0 or self.fallback_scale <= 0:
            raise ValueError("scale and fallback_scale must be positive.")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KeypointCropRule":
        """Build a rule from a dict with the attribute names as keys."""
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown KeypointCropRule keys: {sorted(unknown)}.")
        return cls(**d)

    @classmethod
    def load(cls, rule: Union[str, Path, Dict[str, Any], "KeypointCropRule"]):
        """A rule from itself, a dict, or a ``.json`` / ``.yaml`` file."""
        if isinstance(rule, KeypointCropRule):
            return rule
        if isinstance(rule, dict):
            return cls.from_dict(rule)
        path = Path(str(rule))
        text = path.read_text()
        if path.suffix.lower() == ".json":
            return cls.from_dict(json.loads(text))
        from omegaconf import OmegaConf

        return cls.from_dict(OmegaConf.to_container(OmegaConf.create(text)))

    def to_dict(self) -> Dict[str, Any]:
        """The rule as a plain dict (provenance, files)."""
        return asdict(self)

    def resolve(self, node_names: Sequence[str]) -> Dict[str, List[int]]:
        """Node indices of the rule's names in ``node_names`` (missing ones dropped)."""
        idx = lambda names: [  # noqa: E731
            i for i in (_match(node_names, n) for n in names) if i is not None
        ]
        ref, ref_fb = idx(self.reference), idx(self.reference_fallback)
        return dict(
            center=idx(self.center),
            reference=ref if len(ref) == len(self.reference) else [],
            reference_fallback=(
                ref_fb if len(ref_fb) == len(self.reference_fallback) else []
            ),
        )

    def box(
        self, points: np.ndarray, resolved: Dict[str, List[int]]
    ) -> Tuple[Optional[np.ndarray], bool]:
        """``((4,) xyxy or None, used_fallback)`` for one instance's ``(N, 2)`` points."""
        pts = np.asarray(points, dtype=np.float64)
        ok = np.isfinite(pts).all(-1)
        center = next((pts[i] for i in resolved["center"] if ok[i]), None)
        half = None
        if center is not None:
            for ref in (resolved["reference"], resolved["reference_fallback"]):
                if ref and ok[ref].all():
                    d = float(np.linalg.norm(center - pts[ref].mean(0)))
                    if d > 0:
                        half = self.scale * d
                        break
        used_fallback = False
        if center is None or half is None:
            if self.fallback == "skip" or not ok.any():
                return None, False
            lo, hi = pts[ok].min(0), pts[ok].max(0)
            center = (lo + hi) / 2
            half = self.fallback_scale * float((hi - lo).max()) / 2
            used_fallback = True
        half = max(half, self.min_half_side)
        return (
            np.array(
                [center[0] - half, center[1] - half, center[0] + half, center[1] + half]
            ),
            used_fallback,
        )


def instance_boxes(
    lf: sio.LabeledFrame, rule: KeypointCropRule, cache: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Part boxes of a frame's predicted instances: ``(K, 4)``, ``(K,)``, instances.

    Box ``k`` belongs to ``instances[k]`` and is scored by its instance score.
    """
    cache = {} if cache is None else cache
    boxes, scores, insts = [], [], []
    for inst in lf.predicted_instances:
        key = id(inst.skeleton)
        if key not in cache:
            cache[key] = rule.resolve(inst.skeleton.node_names)
        box, _ = rule.box(inst.numpy(), cache[key])
        if box is None:
            continue
        boxes.append(box)
        score = getattr(inst, "score", None)
        scores.append(1.0 if score is None else float(score))
        insts.append(inst)
    return (
        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        np.asarray(scores, dtype=np.float32),
        insts,
    )


def keypoint_frame_box_fn(rule: KeypointCropRule):
    """A ``LabelsProvider.frame_box_fn`` giving each frame's part boxes."""
    cache: dict = {}

    def frame_box_fn(lf):
        boxes, scores, _ = instance_boxes(lf, rule, cache)
        return boxes, scores

    return frame_box_fn


def _merged_skeleton(
    base: sio.Skeleton, part: sio.Skeleton, prefix: Optional[str]
) -> Tuple[sio.Skeleton, List[str]]:
    part_names = [f"{prefix or ''}{n}" for n in part.node_names]
    clash = set(part_names) & set(base.node_names)
    if clash:
        raise ValueError(
            f"Part nodes {sorted(clash)[:5]} are also body nodes; pass part_prefix."
        )
    names = list(base.node_names) + part_names
    sk = sio.Skeleton(list(names), name=f"{base.name or 'skeleton'}+part")
    for e in base.edges:
        sk.add_edge(e.source.name, e.destination.name)
    rename = dict(zip(part.node_names, part_names))
    for e in part.edges:
        sk.add_edge(rename[e.source.name], rename[e.destination.name])
    for s in list(base.symmetries) + list(part.symmetries):
        a, b = [n.name for n in s.nodes]
        a, b = rename.get(a, a), rename.get(b, b)
        if a in names and b in names:
            sk.add_symmetry(a, b)
    return sk, part_names


def merge_part_predictions(
    base: sio.Labels,
    part: sio.Labels,
    rule: KeypointCropRule,
    part_prefix: Optional[str] = None,
    tol: float = 0.5,
) -> Tuple[sio.Labels, Dict[str, int]]:
    """Append each part instance's points to the body instance its box came from.

    A part instance is matched to a body instance through its linked crop box:
    the box the rule builds from that body instance (within ``tol`` pixels per
    coordinate). Body instances with no part prediction get NaN part points. The
    merged instances keep the body instance's score, track and boxes.

    Returns:
        ``(labels, counts)`` with ``n_instances``, ``n_merged`` and ``n_unmatched``
        (part instances whose box matches no body instance).
    """
    video_map = dict(zip(part.videos, base.videos))
    part_frames = {}
    for lf in part.labeled_frames:
        part_frames[(video_map.get(lf.video, lf.video), lf.frame_idx)] = lf
    skeletons: Dict[Tuple[int, int], Tuple[sio.Skeleton, List[str]]] = {}
    cache: dict = {}
    frames, n_inst, n_merged, n_unmatched = [], 0, 0, 0
    for lf in base.labeled_frames:
        boxes, _, insts = instance_boxes(lf, rule, cache)
        plf = part_frames.get((lf.video, lf.frame_idx))
        found: Dict[int, Any] = {}
        if plf is not None:
            linked = {id(b.instance): b for b in plf.bboxes if b.instance is not None}
            for pinst in plf.predicted_instances:
                b = linked.get(id(pinst))
                if b is None or not len(boxes):
                    n_unmatched += 1
                    continue
                d = np.abs(boxes - np.asarray(b.xyxy, np.float32)[None]).max(1)
                k = int(np.argmin(d))
                if d[k] > tol or k in found:
                    n_unmatched += 1
                    continue
                found[k] = pinst
        by_inst = {id(inst): found.get(k) for k, inst in enumerate(insts)}
        new_insts, relink = [], {}
        part_sk = part.skeletons[0] if part.skeletons else None
        for inst in lf.instances:
            if not isinstance(inst, sio.PredictedInstance) or part_sk is None:
                new_insts.append(inst)
                continue
            key = (id(inst.skeleton), id(part_sk))
            if key not in skeletons:
                skeletons[key] = _merged_skeleton(inst.skeleton, part_sk, part_prefix)
            sk, _ = skeletons[key]
            xys = inst.numpy(scores=True)
            pinst = by_inst.get(id(inst))
            if pinst is not None:
                pxys = pinst.numpy(scores=True)
                n_merged += 1
            else:
                pxys = np.full((len(part_sk.node_names), 3), np.nan)
            allx = np.concatenate([xys, pxys], 0)
            new = sio.PredictedInstance.from_numpy(
                allx[:, :2],
                skeleton=sk,
                point_scores=allx[:, 2],
                score=inst.score,
                track=inst.track,
                tracking_score=getattr(inst, "tracking_score", None),
            )
            relink[id(inst)] = new
            new_insts.append(new)
            n_inst += 1
        bboxes = []
        for b in lf.bboxes:
            nb = copy.copy(b)
            if getattr(b, "instance", None) is not None and id(b.instance) in relink:
                nb.instance = relink[id(b.instance)]
            bboxes.append(nb)
        frames.append(
            sio.LabeledFrame(
                video=lf.video,
                frame_idx=lf.frame_idx,
                instances=new_insts,
                bboxes=bboxes,
            )
        )
    sks = [sk for sk, _ in skeletons.values()] or list(base.skeletons)
    out = sio.Labels(
        frames,
        videos=base.videos,
        skeletons=sks,
        tracks=base.tracks,
        provenance=dict(base.provenance),
    )
    out.provenance["part_stage"] = dict(
        rule=rule.to_dict(),
        part_prefix=part_prefix,
        part_provenance=dict(part.provenance),
    )
    return out, dict(n_instances=n_inst, n_merged=n_merged, n_unmatched=n_unmatched)


def run_part_stage(
    base: sio.Labels,
    part_model_paths: Sequence[str],
    rule: Union[str, Path, Dict[str, Any], KeypointCropRule],
    *,
    device: str = "auto",
    batch_size: int = 4,
    precision: str = "32",
    node_filter: Optional[Union[str, List[str]]] = None,
    strip_prefix: Optional[Union[bool, str]] = None,
    peak_threshold: Optional[float] = None,
    part_prefix: Optional[str] = None,
) -> sio.Labels:
    """Run a part model on crops built from ``base``'s predicted keypoints.

    Args:
        base: Body predictions (e.g. ``predict(video, model_paths=[detector, body])``).
        part_model_paths: The part model (a centered-instance crop model).
        rule: A :class:`KeypointCropRule`, a dict, or a ``.json`` / ``.yaml`` file.
        device: Device of the part model (as for the body run).
        batch_size: Frames per batch.
        precision: ``"32"``, ``"bf16"`` or ``"fp16"`` (as for the body run).
        node_filter: Node filter of the part model's head (names or a prefix).
        strip_prefix: ``strip_prefix`` of that node filter.
        peak_threshold: Part model peak threshold (default: the model's).
        part_prefix: Prefix for the part nodes in the merged skeleton.

    Returns:
        Merged labels: one instance per body instance, body nodes then part nodes.
    """
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.providers import LabelsProvider

    rule = KeypointCropRule.load(rule)
    kwargs: Dict[str, Any] = dict(
        device=device, batch_size=batch_size, boxes="given", box_score_threshold=0.0
    )
    if precision != "32":
        kwargs["precision"] = precision
    if node_filter is not None:
        kwargs["node_filter"] = node_filter
        kwargs["strip_prefix"] = strip_prefix
    if peak_threshold is not None:
        kwargs["peak_threshold"] = peak_threshold
    predictor = Predictor.from_model_paths(list(part_model_paths), **kwargs)
    provider = LabelsProvider(
        labels=base,
        batch_size=batch_size,
        only_predicted_frames=True,
        frame_box_fn=keypoint_frame_box_fn(rule),
    )
    predict_kwargs = {"make_labels": True}
    if peak_threshold is not None:
        predict_kwargs["peak_threshold"] = peak_threshold
    part = predictor.predict(provider, **predict_kwargs)
    merged, counts = merge_part_predictions(base, part, rule, part_prefix=part_prefix)
    logger.info(
        f"Part stage: {counts['n_merged']} of {counts['n_instances']} instances got "
        f"part points ({counts['n_unmatched']} part predictions unmatched)."
    )
    return merged
