"""Crops built from an earlier stage's keypoints: a part stage after top-down pose.

Some body parts leave the crop a top-down pose model sees -- a long tail, say,
lies outside the body box the detector and the body model use. A *part model* is a
second centered-instance model trained on crops around such a part (e.g. a square
centered on the tail base). At inference its crops have to come from the body
model's predictions, so detector, body and part run as one pipeline::

    detector -> body boxes -> body model -> keypoints
             -> KeypointCropRule -> part boxes -> part model -> part keypoints
             -> merged instances (body nodes + part nodes)

A :class:`KeypointCropRule` turns each predicted instance into one square box:

* center: the first located node of ``center`` (names matched exactly, or by the
  part after the last ``/``, so ``"tail_base"`` matches ``"lab_a/tail_base"``);
* half side: ``scale`` x the distance from the center to the mean of the
  ``reference`` nodes (all must be located), else of ``reference_fallback``,
  floored at ``min_half_side`` pixels;
* otherwise (no center, or no reference): ``fallback="extent"`` centers the box on
  the extent of the instance's located points with half side
  ``fallback_scale`` x half the extent's longer side; ``"skip"`` makes no box;
* optionally (``session_floor``, off by default), a floor on the half side of
  ``session_floor`` x the median half side of the rule's boxes in the same video
  (see :func:`session_floors`), about the same center. A short reference (an
  animal rearing or seen end-on, so that its reference points nearly overlap)
  otherwise makes a crop too small to hold the part.

The part model then runs on those boxes through the box-seeded top-down route
(``boxes="given"``), and :func:`merge_part_predictions` appends its points to the
instance each box came from. Nothing here runs unless a part model is given
(:func:`sleap_nn.inference.run.predict` ``part_model_paths``), so the default
inference path is unchanged.
"""

from __future__ import annotations

import copy
import json
from array import array
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
        session_floor: If set, the half side is also floored at ``session_floor``
            x the median half side of the rule's boxes in the same video (a
            session). ``None`` (default): no such floor.
        session_min_boxes: A video with fewer boxes than this uses the median over
            all videos of the labels instead of its own.
    """

    center: List[str]
    reference: List[str] = field(default_factory=list)
    reference_fallback: List[str] = field(default_factory=list)
    scale: float = 2.0
    fallback: str = "extent"
    fallback_scale: float = 1.0
    min_half_side: float = 1.0
    session_floor: Optional[float] = None
    session_min_boxes: int = 10

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
        if self.session_floor is not None and not self.session_floor > 0:
            raise ValueError("session_floor must be positive (or None for no floor).")
        if self.session_min_boxes < 1:
            raise ValueError("session_min_boxes must be at least 1.")

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
        self, points: np.ndarray, resolved: Dict[str, List[int]], floor: float = 0.0
    ) -> Tuple[Optional[np.ndarray], bool]:
        """``((4,) xyxy or None, used_fallback)`` for one instance's ``(N, 2)`` points.

        ``floor`` is an extra floor on the half side in pixels (the video's session
        floor from :func:`session_floors`; 0 for none).
        """
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
        half = max(half, self.min_half_side, floor)
        return (
            np.array(
                [center[0] - half, center[1] - half, center[0] + half, center[1] + half]
            ),
            used_fallback,
        )

    def boxes_tensor(
        self, points: Any, resolved: Dict[str, List[int]]
    ) -> Tuple[Any, Any, Any]:
        """Vectorized tensor equivalent of :meth:`box` (floor=0) on ``points``'s device.

        Computes square box centers, raw half-sides (with ``min_half_side`` applied,
        before any ``session_floor``), and validity masks for a whole batch of
        ``(B, I, N, 2)`` keypoints without a GPU->CPU round-trip.

        Args:
            points: ``(B, I, N, 2)`` keypoint tensor on CPU or GPU.
            resolved: Resolved node index dict from :meth:`resolve`.

        Returns:
            ``(center (B, I, 2), raw_half (B, I), valid (B, I))`` on ``points.device``.
        """
        import torch

        device = points.device
        calc_dtype = torch.float32 if device.type == "mps" else torch.float64
        pts = points.detach().to(dtype=calc_dtype)
        b, n_inst = int(pts.shape[0]), int(pts.shape[1])
        ok = torch.isfinite(pts).all(dim=-1)  # (B, I, N)
        any_ok = ok.any(dim=-1)  # (B, I)

        has_center = torch.zeros((b, n_inst), dtype=torch.bool, device=device)
        center = torch.zeros((b, n_inst, 2), dtype=calc_dtype, device=device)
        for ci in resolved["center"]:
            use = (~has_center) & ok[..., ci]
            center = torch.where(use[..., None], pts[..., ci, :], center)
            has_center = has_center | use

        has_half = torch.zeros((b, n_inst), dtype=torch.bool, device=device)
        half = torch.zeros((b, n_inst), dtype=calc_dtype, device=device)
        for ref in (resolved["reference"], resolved["reference_fallback"]):
            if not ref:
                continue
            if len(ref) == 1:
                ri = ref[0]
                ref_ok = ok[..., ri]
                ref_pt = pts[..., ri, :]
            else:
                ref_idx = torch.tensor(ref, dtype=torch.long, device=device)
                ref_ok = ok.index_select(-1, ref_idx).all(dim=-1)
                ref_pt = pts.index_select(-2, ref_idx).mean(dim=-2)
            diff = torch.where(
                ref_ok[..., None], center - ref_pt, torch.zeros_like(center)
            )
            d = torch.linalg.norm(diff, dim=-1)
            use = has_center & (~has_half) & ref_ok & (d > 0.0)
            half = torch.where(use, float(self.scale) * d, half)
            has_half = has_half | use

        need_fb = (~has_center) | (~has_half)
        if self.fallback == "skip":
            valid = has_center & has_half
        else:
            valid = (~need_fb) | any_ok
            use_fb = need_fb & any_ok
            big = torch.finfo(calc_dtype).max
            lo = torch.where(ok[..., None], pts, pts.new_full((), big)).amin(dim=-2)
            hi = torch.where(ok[..., None], pts, pts.new_full((), -big)).amax(dim=-2)
            fb_center = (lo + hi) * 0.5
            fb_half = float(self.fallback_scale) * (hi - lo).amax(dim=-1) * 0.5
            center = torch.where(use_fb[..., None], fb_center, center)
            half = torch.where(use_fb, fb_half, half)

        raw_half = half.clamp(min=float(self.min_half_side))
        return center, raw_half, valid


def instance_boxes(
    lf: sio.LabeledFrame,
    rule: KeypointCropRule,
    cache: Optional[dict] = None,
    floor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    """Part boxes of a frame's predicted instances: ``(K, 4)``, ``(K,)``, instances.

    Box ``k`` belongs to ``instances[k]`` and is scored by its instance score.
    ``floor`` is passed to :meth:`KeypointCropRule.box`.
    """
    cache = {} if cache is None else cache
    boxes, scores, insts = [], [], []
    for inst in lf.predicted_instances:
        key = id(inst.skeleton)
        if key not in cache:
            cache[key] = rule.resolve(inst.skeleton.node_names)
        box, _ = rule.box(inst.numpy(), cache[key], floor)
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


def session_floors(
    labels: sio.Labels, rule: KeypointCropRule
) -> Dict[sio.Video, float]:
    """Half-side floor per video: ``rule.session_floor`` x the video's median half side.

    The median is over the boxes the rule builds (without this floor) from the
    predicted instances of each video in ``labels``; a video with fewer than
    ``rule.session_min_boxes`` boxes uses the median over all videos. Empty when
    ``rule.session_floor`` is ``None`` or there are no boxes.

    The floor needs the first-stage predictions of the whole video before any
    crop is made; streaming or real-time use would need a running median instead.
    """
    if rule.session_floor is None:
        return {}
    cache: dict = {}
    halves: Dict[sio.Video, List[float]] = {}
    for lf in labels.labeled_frames:
        boxes, _, _ = instance_boxes(lf, rule, cache)
        if len(boxes):
            halves.setdefault(lf.video, []).extend(
                ((boxes[:, 2] - boxes[:, 0]) / 2).tolist()
            )
    every = [h for v in halves.values() for h in v]
    if not every:
        return {}
    pooled = float(np.median(every))
    return {
        video: rule.session_floor
        * (float(np.median(v)) if len(v) >= rule.session_min_boxes else pooled)
        for video, v in halves.items()
    }


def keypoint_frame_box_fn(
    rule: KeypointCropRule, floors: Optional[Dict[sio.Video, float]] = None
):
    """A ``LabelsProvider.frame_box_fn`` giving each frame's part boxes.

    ``floors``: per-video half-side floors from :func:`session_floors` (none if
    ``None`` or empty).
    """
    cache: dict = {}
    floors = floors or {}

    def frame_box_fn(lf):
        boxes, scores, _ = instance_boxes(lf, rule, cache, floors.get(lf.video, 0.0))
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
    floors: Optional[Dict[sio.Video, float]] = None,
) -> Tuple[sio.Labels, Dict[str, int]]:
    """Append each part instance's points to the body instance its box came from.

    A part instance is matched to a body instance through its linked crop box:
    the box the rule builds from that body instance (within ``tol`` pixels per
    coordinate), with the per-video ``floors`` (default: :func:`session_floors`
    of ``base``, empty unless the rule sets ``session_floor``). Body instances with
    no part prediction get NaN part points. The merged instances keep the body
    instance's score, track and boxes.

    Returns:
        ``(labels, counts)`` with ``n_instances``, ``n_merged`` and ``n_unmatched``
        (part instances whose box matches no body instance).
    """
    if floors is None:
        floors = session_floors(base, rule)
    video_map = dict(zip(part.videos, base.videos))
    part_frames = {}
    for lf in part.labeled_frames:
        part_frames[(video_map.get(lf.video, lf.video), lf.frame_idx)] = lf
    skeletons: Dict[Tuple[int, int], Tuple[sio.Skeleton, List[str]]] = {}
    cache: dict = {}
    frames, n_inst, n_merged, n_unmatched = [], 0, 0, 0
    for lf in base.labeled_frames:
        boxes, _, insts = instance_boxes(lf, rule, cache, floors.get(lf.video, 0.0))
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
    from sleap_nn.inference.provenance import _add_part_exports

    _add_part_exports(out.provenance, part.provenance.get("exported_models") or [])
    if floors:
        out.provenance["part_stage"]["session_floors"] = {
            base.videos.index(v): round(float(f), 3)
            for v, f in floors.items()
            if v in base.videos
        }
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
    export_runtime: str = "auto",
    memory_format: str = "contiguous",
    upload_batches: bool = False,
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
        export_runtime: Runtime when the part model is a network-only export
            directory (see ``Predictor.from_model_paths``).
        memory_format: Weight layout of the part model (as for the body run).
        upload_batches: Copy each batch of frames to the part model's GPU before
            cropping (as for the body run).

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
    if export_runtime != "auto":
        kwargs["export_runtime"] = export_runtime
    if memory_format != "contiguous":
        kwargs["memory_format"] = memory_format
    if upload_batches:
        kwargs["upload_batches"] = True
    predictor = Predictor.from_model_paths(list(part_model_paths), **kwargs)
    floors = session_floors(base, rule)
    if floors:
        logger.info(
            f"Part stage: half sides floored at {rule.session_floor} x the median "
            f"of each of {len(floors)} videos."
        )
    provider = LabelsProvider(
        labels=base,
        batch_size=batch_size,
        only_predicted_frames=True,
        frame_box_fn=keypoint_frame_box_fn(rule, floors),
    )
    predict_kwargs = {"make_labels": True}
    if peak_threshold is not None:
        predict_kwargs["peak_threshold"] = peak_threshold
    part = predictor.predict(provider, **predict_kwargs)
    merged, counts = merge_part_predictions(
        base, part, rule, part_prefix=part_prefix, floors=floors
    )
    logger.info(
        f"Part stage: {counts['n_merged']} of {counts['n_instances']} instances got "
        f"part points ({counts['n_unmatched']} part predictions unmatched)."
    )
    return merged


def _heap_push(h: array, x: float) -> None:
    """Push ``x`` onto the min-heap ``h`` (a typed float64 array)."""
    h.append(x)
    i = len(h) - 1
    while i:
        p = (i - 1) >> 1
        if h[p] <= x:
            break
        h[i] = h[p]
        i = p
    h[i] = x


def _heap_sift(h: array, i: int) -> None:
    """Restore the min-heap ``h`` after the value at ``i`` was replaced."""
    x = h[i]
    n = len(h)
    while i:  # up
        p = (i - 1) >> 1
        if h[p] <= x:
            break
        h[i] = h[p]
        i = p
    while True:  # down
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n and h[c + 1] < h[c]:
            c += 1
        if h[c] >= x:
            break
        h[i] = h[c]
        i = c
    h[i] = x


def _heap_pop(h: array) -> float:
    """Remove and return the smallest value of the min-heap ``h``."""
    last = h.pop()
    if not h:
        return last
    top = h[0]
    n = len(h)
    i = 0
    while True:
        c = 2 * i + 1
        if c >= n:
            break
        if c + 1 < n and h[c + 1] < h[c]:
            c += 1
        if h[c] >= last:
            break
        h[i] = h[c]
        i = c
    h[i] = last
    return top


class RunningMedian:
    """Exact median of a growing set of floats, equal to ``np.median`` of the same values.

    Two binary heaps in typed float64 arrays: the lower half as a max-heap (stored
    negated) and the upper half as a min-heap. ``add`` costs O(log n) and ``median``
    O(1), so the cost per value stays flat however long the history gets. Memory: 8
    bytes per value (the arrays over-allocate by about an eighth as they grow), e.g.
    about 60 MB for a day of 3 animals at 30 frames/s. NaN values are counted, not
    stored: with any NaN the median is NaN, as ``np.median`` gives.
    """

    __slots__ = ("_lo", "_hi", "_nan")

    def __init__(self) -> None:
        """Start empty."""
        self._lo = array("d")  # lower half, negated (max-heap)
        self._hi = array("d")  # upper half (min-heap)
        self._nan = 0

    def __len__(self) -> int:
        """Number of values added."""
        return len(self._lo) + len(self._hi) + self._nan

    def add(self, x: float) -> None:
        """Add one value."""
        x = float(x)
        if x != x:
            self._nan += 1
            return
        lo, hi = self._lo, self._hi
        if lo and x > -lo[0]:
            _heap_push(hi, x)
        else:
            _heap_push(lo, -x)
        if len(lo) > len(hi) + 1:
            _heap_push(hi, -_heap_pop(lo))
        elif len(hi) > len(lo):
            _heap_push(lo, -_heap_pop(hi))

    def remove(self, x: float) -> None:
        """Remove one value added before (O(n) search; for undoing a failed step).

        Raises:
            ValueError: ``x`` is not in the set.
        """
        x = float(x)
        if x != x:
            if not self._nan:
                raise ValueError("No NaN value to remove.")
            self._nan -= 1
            return
        lo, hi = self._lo, self._hi
        for h, key in ((lo, -x), (hi, x)):
            try:
                i = h.index(key)
            except ValueError:
                continue
            last = h.pop()
            if i < len(h):
                h[i] = last
                _heap_sift(h, i)
            if len(lo) > len(hi) + 1:
                _heap_push(hi, -_heap_pop(lo))
            elif len(hi) > len(lo):
                _heap_push(lo, -_heap_pop(hi))
            return
        raise ValueError(f"{x} is not in the set.")

    def median(self) -> float:
        """The median of every value added (NaN when empty or when any value is NaN)."""
        lo, hi = self._lo, self._hi
        if self._nan or not lo:
            return float("nan")
        if len(lo) > len(hi):
            return -lo[0]
        return (-lo[0] + hi[0]) / 2

    def values(self) -> np.ndarray:
        """The values added (not in order; NaN values left out), as float64."""
        return np.concatenate(
            [
                -np.frombuffer(self._lo, dtype=np.float64),
                np.frombuffer(self._hi, dtype=np.float64),
            ]
        )


class StreamingPartStage:
    """The part stage run on each batch right after the pose stage (``part_streaming``).

    Experimental: it may change or be removed without a deprecation period.

    :func:`run_part_stage` runs after the whole video: it reads every frame a
    second time and floors each crop at the median crop of the whole video
    (``session_floor``). This runs the part model on the frames already in memory,
    batch by batch, with a running floor: ``session_floor`` x the median half side
    of the rule's boxes seen so far in that video (including the current batch),
    once at least ``session_min_boxes`` boxes were seen; before that, the median
    over every video so far, or no floor. Nothing waits for the end of the video.

    How the running floor differs from the two-pass one: each frame's floor includes
    the later frames of its own batch (so results depend on the batch size); the
    first ``session_min_boxes`` boxes of the first video get no floor at all (a clip
    that starts with short reference distances, e.g. rearing, gets small crops
    there); and every run (each ``--frames`` chunk) starts a new floor.

    The medians are exact running medians (:class:`RunningMedian`): the cost per
    frame does not grow with the history. With a session floor the history takes 8
    bytes per box for its video plus 8 for the all-video median (about 0.12 GB per
    day for 3 animals at 30 frames/s); without one nothing is stored.

    The floor state belongs to the stage, so two prediction runs interleaved on one
    predictor (e.g. two ``predict_streaming`` generators) share it; run them one
    after the other, or on separate predictors.

    Each pose instance gets the part points of the crop built from its own
    keypoints (part slot ``k`` is pose slot ``k``); points below the part model's
    threshold, and instances without a crop, get NaN.

    Args:
        layer: The part predictor's layer (a top-down layer seeded by given boxes).
        rule: The crop rule.
        body_node_names: Node names of the pose model's output.
        n_part_nodes: Number of part-model nodes (its skeleton's). Batches where
            no instance gets a crop are padded to it, whatever node count the part
            layer's empty path reports. ``None``: the part layer's count.
    """

    def __init__(
        self,
        layer: Any,
        rule: KeypointCropRule,
        body_node_names: Sequence[str],
        n_part_nodes: Optional[int] = None,
    ):
        """Resolve the rule on the pose nodes."""
        self.layer = layer
        self.rule = rule
        self.resolved = rule.resolve(list(body_node_names))
        self.n_body_nodes = len(body_node_names)
        self.n_part_nodes = n_part_nodes
        self.halves: Dict[int, RunningMedian] = {}
        self.every = RunningMedian()
        self.last_floors: Dict[int, float] = {}

    def reset(self) -> None:
        """Forget the running floors (call at the start of each prediction run)."""
        self.halves = {}
        self.every = RunningMedian()
        self.last_floors = {}
        self._last = None

    def _floor(self, video: int) -> float:
        rule = self.rule
        if rule.session_floor is None:
            return 0.0
        own = self.halves.get(video)
        if own is not None and len(own) >= rule.session_min_boxes:
            return rule.session_floor * float(own.median())
        if len(self.every) >= rule.session_min_boxes:
            return rule.session_floor * float(self.every.median())
        return 0.0

    def __call__(self, images: Any, outputs: Any, video_indices: Any = None) -> Any:
        """``outputs`` with the part points appended to every instance's nodes.

        A call that raises leaves the running floors as they were.
        """
        import torch

        if outputs.pred_keypoints is None:
            return outputs
        kpts_t = outputs.pred_keypoints
        B = int(kpts_t.shape[0])
        vids = (
            np.zeros(B, dtype=np.int64)
            if video_indices is None
            else np.asarray(
                video_indices.cpu() if torch.is_tensor(video_indices) else video_indices
            ).astype(np.int64)
        )
        added: List[Tuple[int, float]] = []
        floors_before = dict(self.last_floors)
        self._last = None
        try:
            out = self._call(images, outputs, kpts_t, vids, added)
        except BaseException:
            # Leave the floors as they were: a failed call adds nothing to the history.
            self._undo(added, floors_before)
            raise
        self._last = (added, floors_before)
        return out

    def _undo(
        self, added: List[Tuple[int, float]], floors_before: Dict[int, float]
    ) -> None:
        for video, half in reversed(added):
            med = self.halves[video]
            med.remove(half)
            if not len(med):
                del self.halves[video]
            self.every.remove(half)
        self.last_floors = floors_before

    def undo_last(self) -> None:
        """Take back the last call's floor additions (a caller's later step failed).

        Does nothing if :meth:`forget_last` ran since, or the call itself failed.
        """
        if getattr(self, "_last", None) is not None:
            self._undo(*self._last)
            self._last = None

    def forget_last(self) -> None:
        """Make the last call final (:meth:`undo_last` will not take it back)."""
        self._last = None

    def _call(
        self,
        images: Any,
        outputs: Any,
        kpts_t: Any,
        vids: np.ndarray,
        added: List[Tuple[int, float]],
    ) -> Any:
        import attrs
        import torch

        B, n_inst = int(kpts_t.shape[0]), int(kpts_t.shape[1])
        device = kpts_t.device
        if n_inst == 0:
            boxes_t = torch.full(
                (B, 1, 4), float("nan"), dtype=torch.float32, device=device
            )
            scores_t = torch.full(
                (B, 1), float("nan"), dtype=torch.float32, device=device
            )
        else:
            center, raw_half, valid = self.rule.boxes_tensor(kpts_t, self.resolved)
            if self.rule.session_floor is None:
                final_half = raw_half
            else:
                # Match ``(box[2] - box[0]) / 2`` from KeypointCropRule.box
                # bit-for-bit in float64 (avoids 1-ULP rounding drift).
                cx_raw = center[..., 0]
                box_half = ((cx_raw + raw_half) - (cx_raw - raw_half)) * 0.5
                raw_half_np = (
                    box_half.masked_fill(~valid, float("nan")).cpu().numpy()
                )
                floors = np.zeros(B, dtype=np.float64)
                has_any_b = np.zeros(B, dtype=bool)
                for b in range(B):
                    vid = int(vids[b])
                    row = raw_half_np[b]
                    for i in range(n_inst):
                        hval = float(row[i])
                        if hval != hval:
                            continue
                        has_any_b[b] = True
                        med = self.halves.get(vid)
                        if med is None:
                            med = self.halves[vid] = RunningMedian()
                        med.add(hval)
                        self.every.add(hval)
                        added.append((vid, hval))
                for b in range(B):
                    if has_any_b[b]:
                        vid = int(vids[b])
                        fl = self._floor(vid)
                        self.last_floors[vid] = fl
                        floors[b] = fl
                floors_t = torch.from_numpy(floors).to(
                    device=device, dtype=raw_half.dtype, non_blocking=True
                )[:, None]
                final_half = torch.maximum(raw_half, floors_t)

            cx, cy = center[..., 0], center[..., 1]
            boxes_t = (
                torch.stack(
                    [cx - final_half, cy - final_half, cx + final_half, cy + final_half],
                    dim=-1,
                )
                .to(torch.float32)
                .masked_fill(~valid[..., None], float("nan"))
            )
            if outputs.instance_scores is not None:
                inst_s = outputs.instance_scores.detach().to(
                    device=device, dtype=torch.float32
                )
                scores_t = torch.where(
                    torch.isfinite(inst_s), inst_s, torch.ones_like(inst_s)
                )
            else:
                scores_t = torch.ones((B, n_inst), dtype=torch.float32, device=device)
            scores_t = scores_t.masked_fill(~valid, float("nan"))

        part = self.layer.predict(
            images,
            frame_boxes=boxes_t,
            frame_box_scores=scores_t,
        )
        device = outputs.pred_keypoints.device
        pk = part.pred_keypoints
        pv = part.pred_peak_values
        n_part = self.n_part_nodes if self.n_part_nodes is not None else pk.shape[-2]
        if pk.shape[-2] != n_part:
            if pk.shape[1] and not torch.isnan(pk).all():
                raise ValueError(
                    f"The part model returned {pk.shape[-2]} nodes; its skeleton has "
                    f"{n_part}."
                )
            # No crop in this batch: the part layer's empty path may report another
            # node count (1 for an exported network); every part point is NaN.
            pk = torch.full((B, 0, n_part, 2), float("nan"))
            pv = torch.full((B, 0, n_part), float("nan"))
        # Part slot k comes from box k = pose slot k; pad or cut to the pose slots.
        full_k = torch.full((B, n_inst, n_part, 2), float("nan"), device=device)
        full_v = torch.full((B, n_inst, n_part), float("nan"), device=device)
        m = min(n_inst, pk.shape[1])
        full_k[:, :m] = pk[:, :m].to(device)
        full_v[:, :m] = pv[:, :m].to(device)
        vals = outputs.pred_peak_values
        if vals is None:
            vals = torch.full(
                outputs.pred_keypoints.shape[:3], float("nan"), device=device
            )
        return attrs.evolve(
            outputs,
            pred_keypoints=torch.cat([outputs.pred_keypoints, full_k], dim=2),
            pred_peak_values=torch.cat([vals.to(device), full_v], dim=2),
            pred_crop_keypoints=None,
        )


def track_on_body_nodes(
    labels: sio.Labels,
    body_skeleton: sio.Skeleton,
    tracker_config: Any,
    progress_callback: Optional[Any] = None,
) -> sio.Labels:
    """Track a streamed part stage's labels on the pose nodes only, as two passes do.

    ``labels`` hold the pose nodes first and the part nodes after them. Tracking
    runs on copies of the instances cut to ``body_skeleton``'s nodes; the tracks,
    the tracking scores and the instances the tracker keeps per frame are then
    carried back to the full instances (in place, as
    :func:`~sleap_nn.inference.tracking.apply_tracking` works).
    """
    from sleap_nn.inference.tracking import apply_tracking

    n_body = len(body_skeleton.nodes)
    back: Dict[int, Any] = {}
    full_of: Dict[int, Any] = {}  # body frame -> full frame (a frame may repeat)
    frames = []
    for lf in labels.labeled_frames:
        insts = []
        for inst in lf.instances:
            arr = inst.numpy(scores=True)
            b = sio.PredictedInstance.from_numpy(
                arr[:n_body, :2],
                skeleton=body_skeleton,
                point_scores=arr[:n_body, 2],
                score=inst.score,
                track=inst.track,
                tracking_score=inst.tracking_score,
            )
            back[id(b)] = inst
            insts.append(b)
        blf = sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, instances=insts)
        full_of[id(blf)] = lf
        frames.append(blf)
    body = sio.Labels(
        labeled_frames=frames,
        videos=labels.videos,
        skeletons=[body_skeleton],
        provenance=dict(labels.provenance),
    )
    body = apply_tracking(body, tracker_config, progress_callback)
    ordered = []
    for blf in body.labeled_frames:  # the tracker keeps the frame objects
        lf = full_of.pop(id(blf))
        kept = []
        for b in blf.instances:
            inst = back[id(b)]
            inst.track = b.track
            inst.tracking_score = b.tracking_score
            kept.append(inst)
        lf.instances = kept
        ordered.append(lf)
    labels.labeled_frames = ordered + list(full_of.values())
    labels.tracks = list(body.tracks)
    if "tracking" in body.provenance:
        labels.provenance["tracking"] = body.provenance["tracking"]
    else:
        labels.provenance.pop("tracking", None)
    return labels


def build_streaming_part_stage(
    base_skeleton: sio.Skeleton,
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
    export_runtime: str = "auto",
    memory_format: str = "contiguous",
) -> Tuple[StreamingPartStage, sio.Skeleton]:
    """``(StreamingPartStage, merged skeleton)`` for a pose predictor's skeleton.

    The part model is loaded as :func:`run_part_stage` loads it (``export_runtime``:
    the runtime when the part model is a network-only export directory;
    ``memory_format``: the part model's weight layout, as for the body run). The
    part model cuts its crops from the frames the body predictor hands it, so
    ``upload_batches`` reaches it through the body run. The merged
    skeleton (pose nodes, then the part nodes with ``part_prefix``) is the one
    :func:`merge_part_predictions` would build.
    """
    from sleap_nn.inference.predictor import Predictor

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
    if export_runtime != "auto":
        kwargs["export_runtime"] = export_runtime
    if memory_format != "contiguous":
        kwargs["memory_format"] = memory_format
    part = Predictor.from_model_paths(list(part_model_paths), **kwargs)
    merged, _ = _merged_skeleton(base_skeleton, part.skeleton, part_prefix)
    layer = part._scoped_postprocess_layer(peak_threshold=peak_threshold)
    stage = StreamingPartStage(
        layer, rule, base_skeleton.node_names, n_part_nodes=len(part.skeleton.nodes)
    )
    stage.part_predictor = part  # for the run's provenance (as run_part_stage's)
    stage.body_skeleton = base_skeleton  # tracking runs on these nodes only
    return stage, merged
