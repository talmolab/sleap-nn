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
  ``fallback_scale`` x half the extent's longer side; ``"skip"`` makes no box;
* optionally (``session_floor``, off by default), a floor on the half side of
  ``session_floor`` x the median half side of the rule's boxes in the same video
  (see :func:`session_floors`), about the same center. A short reference (a mouse
  rearing or seen nose-on, whose ears nearly cover the tail base) otherwise makes
  a crop too small to hold the part.

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

    :func:`run_part_stage` runs after the whole video: it reads every frame a
    second time and floors each crop at the median crop of the whole video
    (``session_floor``). This runs the part model on the frames already in memory,
    batch by batch, with a running floor: ``session_floor`` x the median half side
    of the rule's boxes seen so far in that video (including the current batch),
    once at least ``session_min_boxes`` boxes were seen; before that, the median
    over every video so far, or no floor. Nothing waits for the end of the video.
    The medians are exact running medians (:class:`RunningMedian`): the cost per
    frame does not grow with the history, and the history takes 8 bytes per box
    for its video plus 8 for the all-video median.

    Each pose instance gets the part points of the crop built from its own
    keypoints (part slot ``k`` is pose slot ``k``); points below the part model's
    threshold, and instances without a crop, get NaN.

    Args:
        layer: The part predictor's layer (a top-down layer seeded by given boxes).
        rule: The crop rule.
        body_node_names: Node names of the pose model's output.
    """

    def __init__(
        self, layer: Any, rule: KeypointCropRule, body_node_names: Sequence[str]
    ):
        """Resolve the rule on the pose nodes."""
        self.layer = layer
        self.rule = rule
        self.resolved = rule.resolve(list(body_node_names))
        self.halves: Dict[int, RunningMedian] = {}
        self.every = RunningMedian()
        self.last_floors: Dict[int, float] = {}

    def reset(self) -> None:
        """Forget the running floors (call at the start of each prediction run)."""
        self.halves = {}
        self.every = RunningMedian()
        self.last_floors = {}

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
        """``outputs`` with the part points appended to every instance's nodes."""
        import attrs
        import torch

        if outputs.pred_keypoints is None:
            return outputs
        kpts = outputs.pred_keypoints.detach().float().cpu().numpy()
        B, n_inst = kpts.shape[:2]
        vids = (
            np.zeros(B, dtype=np.int64)
            if video_indices is None
            else np.asarray(
                video_indices.cpu() if torch.is_tensor(video_indices) else video_indices
            ).astype(np.int64)
        )
        raw: Dict[Tuple[int, int], np.ndarray] = {}
        for b in range(B):
            for i in range(n_inst):
                if np.isnan(kpts[b, i]).all():
                    continue
                box, _ = self.rule.box(kpts[b, i], self.resolved, 0.0)
                if box is not None:
                    raw[(b, i)] = box
                    half = float(box[2] - box[0]) / 2
                    self.halves.setdefault(int(vids[b]), RunningMedian()).add(half)
                    self.every.add(half)
        boxes = np.full((B, max(n_inst, 1), 4), np.nan, dtype=np.float32)
        scores = np.full((B, max(n_inst, 1)), np.nan, dtype=np.float32)
        inst_scores = (
            outputs.instance_scores.detach().float().cpu().numpy()
            if outputs.instance_scores is not None
            else np.ones((B, n_inst), dtype=np.float32)
        )
        for (b, i), _ in raw.items():
            floor = self._floor(int(vids[b]))
            self.last_floors[int(vids[b])] = floor
            box, _ = self.rule.box(kpts[b, i], self.resolved, floor)
            boxes[b, i] = box
            s = inst_scores[b, i]
            scores[b, i] = s if np.isfinite(s) else 1.0
        part = self.layer.predict(
            images,
            frame_boxes=torch.from_numpy(boxes),
            frame_box_scores=torch.from_numpy(scores),
        )
        device = outputs.pred_keypoints.device
        pk = part.pred_keypoints
        pv = part.pred_peak_values
        n_part = pk.shape[-2]
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
) -> Tuple[StreamingPartStage, sio.Skeleton]:
    """``(StreamingPartStage, merged skeleton)`` for a pose predictor's skeleton.

    The part model is loaded as :func:`run_part_stage` loads it (``export_runtime``:
    the runtime when the part model is a network-only export directory); the merged
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
    part = Predictor.from_model_paths(list(part_model_paths), **kwargs)
    merged, _ = _merged_skeleton(base_skeleton, part.skeleton, part_prefix)
    layer = part._scoped_postprocess_layer(peak_threshold=peak_threshold)
    stage = StreamingPartStage(layer, rule, base_skeleton.node_names)
    stage.part_provenance = dict(
        model_paths=[str(p) for p in part_model_paths],
        node_filter=node_filter,
        peak_threshold=peak_threshold,
    )
    return stage, merged
