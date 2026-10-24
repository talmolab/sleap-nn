"""COCO-faithful keypoint evaluation (OKS AP/AR) for SLEAP labels.

This module reproduces ``pycocotools.cocoeval.COCOeval(iouType="keypoints")`` on
``sio.Labels`` without requiring pycocotools, and can optionally hand the same data
to pycocotools for cross-checking (``backend="pycocotools"``).

What it does differently from :class:`sleap_nn.evaluation.Evaluator`'s ``oks_voc``
metrics (which are kept for back-compat):

* **Per-threshold matching.** Detections are greedily matched to GT separately at
  each OKS threshold (0.50:0.05:0.95), as COCOeval does, instead of once at
  threshold 0 and then re-thresholded.
* **Every prediction is scored.** Unmatched predictions (extra detections after the
  GT runs out, detections with OKS 0 to every GT, and detections on frames with no
  GT instances) are false positives. ``Evaluator.voc_metrics`` silently drops them.
* **Ignore regions.** Crowd boxes and persons without keypoints (``UserBoundingBox``
  with no linked instance) are "ignore" GT: a detection matched to one is neither a
  TP nor an FP (COCO's ``num_keypoints == 0 or iscrowd`` rule).
* **Per-node sigmas** (``COCO_KEYPOINT_SIGMAS``, provenance ``node_sigmas``, or a
  user list/dict) instead of one scalar ``oks_stddev``.
* **COCO object area.** GT area comes from the source annotation's segmentation area
  when available (``/human_meta`` in ``.slp`` files written by the human-pose
  converter), else the GT box area, else the visible-keypoint box (the native
  evaluator's only option, which under-estimates the area and so deflates OKS).
* **maxDets=20 per image and the medium/large area ranges.**
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import attrs
import numpy as np
import sleap_io as sio
from loguru import logger

# Official COCO-17 keypoint sigmas (pycocotools ``Params.setKpParams``), by node name.
COCO_KEYPOINT_SIGMAS: Dict[str, float] = {
    "nose": 0.026,
    "left_eye": 0.025,
    "right_eye": 0.025,
    "left_ear": 0.035,
    "right_ear": 0.035,
    "left_shoulder": 0.079,
    "right_shoulder": 0.079,
    "left_elbow": 0.072,
    "right_elbow": 0.072,
    "left_wrist": 0.062,
    "right_wrist": 0.062,
    "left_hip": 0.107,
    "right_hip": 0.107,
    "left_knee": 0.087,
    "right_knee": 0.087,
    "left_ankle": 0.089,
    "right_ankle": 0.089,
}

# Official CrowdPose-14 sigmas (crowdposetools): COCO's body sigmas + head/neck.
CROWDPOSE_KEYPOINT_SIGMAS: Dict[str, float] = {
    **{k: v for k, v in COCO_KEYPOINT_SIGMAS.items() if not k.endswith(("eye", "ear"))},
    "head": 0.079,
    "neck": 0.079,
}
del CROWDPOSE_KEYPOINT_SIGMAS["nose"]

SIGMA_PRESETS: Dict[str, Dict[str, float]] = {
    "coco": COCO_KEYPOINT_SIGMAS,
    "ochuman": COCO_KEYPOINT_SIGMAS,
    "crowdpose": CROWDPOSE_KEYPOINT_SIGMAS,
}

# COCOeval keypoint area ranges (px^2): all / medium / large.
COCO_KPT_AREA_RANGES: Dict[str, Tuple[float, float]] = {
    "all": (0.0, 1e10),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, 1e10),
}

# Dataset evaluation protocols, keyed by ``provenance["dataset"]``. "labeled" ignore
# = xtcocotools ``keypoints`` (mmpose COCO/OCHuman); CrowdPose follows xtcocotools
# ``keypoints_crowd``: ignore GT whose source ``num_keypoints`` (v=2 count) is 0, and
# area = 0.53 * box area (``use_area=False``; the JSON has no area).
EVAL_PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "coco": dict(sigmas="coco", ignore_rule="labeled", area_source="auto"),
    "ochuman": dict(sigmas="coco", ignore_rule="labeled", area_source="auto"),
    "crowdpose": dict(
        sigmas="crowdpose", ignore_rule="num_keypoints", area_source="bbox"
    ),
}

SigmaSpec = Union[None, float, str, Sequence[float], Mapping[str, float]]


def _as_dict(value: Any) -> Optional[dict]:
    """Coerce a provenance value that may have been stringified back to a dict."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None
    return None


def resolve_node_sigmas(
    skeleton: sio.Skeleton,
    sigmas: SigmaSpec = None,
    provenance: Optional[Mapping[str, Any]] = None,
    default: float = 0.025,
) -> np.ndarray:
    """Resolve one OKS sigma per skeleton node.

    Resolution order (first hit wins per node):

    1. ``sigmas``: a float (all nodes), a list (one per node), a ``{node: sigma}``
       dict, a preset name (``"coco"``, ``"crowdpose"``, ``"ochuman"``), or
       ``"provenance"`` (only step 2).
    2. ``provenance["node_sigmas"]`` of the GT labels (``{native_name: sigma}``),
       remapped through ``provenance["union_map"]`` (``{native: union}``) when the
       labels use a union skeleton.
    3. ``COCO_KEYPOINT_SIGMAS`` by node name.
    4. ``default`` (logged).

    Args:
        skeleton: Skeleton whose node order the result follows.
        sigmas: Explicit sigma specification (see above).
        provenance: GT ``Labels.provenance``.
        default: Fallback sigma for nodes with no other source.

    Returns:
        Float array of shape ``(n_nodes,)``.
    """
    names = list(skeleton.node_names)
    n = len(names)
    if sigmas is not None and not isinstance(sigmas, (str, Mapping)):
        if np.isscalar(sigmas):
            return np.full(n, float(sigmas))
        arr = np.asarray(sigmas, dtype="float64")
        if arr.shape != (n,):
            raise ValueError(f"Expected {n} sigmas (one per node), got {arr.shape}.")
        return arr

    table: Dict[str, float] = {}
    explicit = None
    if isinstance(sigmas, str) and sigmas != "provenance":
        if sigmas not in SIGMA_PRESETS:
            raise ValueError(
                f"Unknown sigma preset {sigmas!r}; choose from "
                f"{sorted(SIGMA_PRESETS)} or 'provenance'."
            )
        explicit = SIGMA_PRESETS[sigmas]
    elif isinstance(sigmas, Mapping):
        explicit = dict(sigmas)
    if explicit is not None:
        table.update(explicit)

    if provenance is not None and explicit is None:
        prov = _as_dict(provenance.get("node_sigmas"))
        if prov:
            union_map = _as_dict(provenance.get("union_map")) or {}
            for native, s in prov.items():
                table.setdefault(union_map.get(native, native), float(s))
                table.setdefault(native, float(s))

    out = np.full(n, np.nan)
    missing = []
    for i, name in enumerate(names):
        if name in table:
            out[i] = table[name]
        elif sigmas != "provenance" and name in COCO_KEYPOINT_SIGMAS:
            out[i] = COCO_KEYPOINT_SIGMAS[name]
        else:
            out[i] = default
            missing.append(name)
    if missing and len(missing) < n:
        logger.info(
            f"OKS sigmas: {len(missing)}/{n} nodes have no sigma source; using "
            f"{default} for {missing[:6]}{'...' if len(missing) > 6 else ''}."
        )
    return out


@attrs.define
class CocoImage:
    """Ground truth and detections for one image, in COCO terms.

    Attributes:
        key: ``(video_key, frame_idx)`` identifying the frame.
        image_id: Source COCO ``image_id`` when known, else a running index.
        gt_points: ``(G, K, 2)`` GT keypoints; NaN = unlabeled (COCO ``v=0``).
        gt_area: ``(G,)`` GT object area (px^2).
        gt_bbox: ``(G, 4)`` GT box ``xywh`` (NaN when unknown; only needed for
            ignore GT without keypoints).
        gt_iscrowd: ``(G,)`` crowd flags.
        gt_ignore: ``(G,)`` COCO ignore flags (crowd or no keypoints).
        dt_points: ``(D, K, 2)`` predicted keypoints (NaN = missing).
        dt_scores: ``(D,)`` instance scores.
    """

    key: Tuple[str, int]
    image_id: int
    gt_points: np.ndarray
    gt_area: np.ndarray
    gt_bbox: np.ndarray
    gt_iscrowd: np.ndarray
    gt_ignore: np.ndarray
    dt_points: np.ndarray
    dt_scores: np.ndarray

    @property
    def dt_area(self) -> np.ndarray:
        """Detection area as ``COCO.loadRes`` defines it: keypoint-extent box."""
        if not len(self.dt_points):
            return np.zeros(0)
        with warnings.catch_warnings():
            # All-NaN detections: their area is 0, as COCO's would be.
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanmin(self.dt_points, axis=1)
            hi = np.nanmax(self.dt_points, axis=1)
        return np.nan_to_num(np.prod(hi - lo, axis=-1))


def _read_human_meta(labels: sio.Labels) -> Optional[dict]:
    """Read ``/human_meta`` (per-annotation area/iscrowd/ids) if the file has it."""
    path = (labels.provenance or {}).get("filename")
    if not path or not Path(path).exists():
        return None
    try:
        import h5py

        with h5py.File(path, "r") as f:
            if "human_meta" not in f:
                return None
            g = f["human_meta"]
            ann = g["annotations"][:]
            frame_image_id = g["frame_image_id"][:]
    except (OSError, KeyError):
        return None
    if len(frame_image_id) != len(labels.labeled_frames):
        logger.warning(
            "/human_meta frame count does not match the labels; ignoring it."
        )
        return None
    inst_area, bbox_area, inst_nkpt = {}, {}, {}
    for row in ann:
        lf, a = int(row["lf_idx"]), float(row["area"])
        if row["inst_idx"] >= 0:
            inst_area[lf, int(row["inst_idx"])] = a
            inst_nkpt[lf, int(row["inst_idx"])] = int(row["num_keypoints"])
        if row["bbox_idx"] >= 0:
            bbox_area[lf, int(row["bbox_idx"])] = a
    return dict(
        frame_image_id=frame_image_id,
        inst_area=inst_area,
        bbox_area=bbox_area,
        inst_nkpt=inst_nkpt,
    )


def _category(box: Any) -> str:
    cat = getattr(box, "category", None)
    return getattr(cat, "name", "") or ""


def _box_xywh(box: Any) -> np.ndarray:
    return np.array([box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1], dtype="float64")


def _kpt_box_area(points: np.ndarray) -> float:
    vis = ~np.isnan(points).any(axis=-1)
    if not vis.any():
        return 0.0
    p = points[vis]
    return float(np.prod(p.max(0) - p.min(0)))


def _video_key(video: Optional[sio.Video]) -> str:
    from sleap_nn.evaluation import _video_key as vk

    return vk(video)


def labels_to_coco_images(
    labels_gt: sio.Labels,
    labels_pr: sio.Labels,
    area_source: str = "auto",
    bbox_area_scale: float = 0.53,
    use_ignore_regions: bool = True,
    user_labels_only: bool = True,
    ignore_rule: str = "labeled",
) -> List[CocoImage]:
    """Pair GT and predicted frames into COCO-style per-image records.

    Every GT labeled frame is an image, including frames with no instances
    (``is_negative`` or box-only frames): detections there are false positives, as
    in COCOeval. Predicted frames with no GT frame are not scored (COCOeval only
    scores ``imgIds`` from the GT).

    Args:
        labels_gt: Ground truth labels.
        labels_pr: Predicted labels (``PredictedInstance.score`` ranks detections).
        area_source: GT area source: ``"meta"`` (``/human_meta`` segmentation area),
            ``"bbox"`` (linked GT ``UserBoundingBox`` w*h * ``bbox_area_scale``),
            ``"keypoints"`` (visible-keypoint box, the native evaluator's choice) or
            ``"auto"`` (meta -> bbox -> keypoints, per instance).
        bbox_area_scale: Multiplier on box areas used as GT area. ``0.53`` is
            xtcocotools' ``use_area=False`` rule (mmpose's CrowdPose / AIC eval);
            ``1.0`` over-states the area (on COCO val, AP is inflated by ~0.1).
        use_ignore_regions: Add crowd boxes and keypoint-less person boxes as
            COCO "ignore" GT.
        user_labels_only: Only user ``Instance``s are GT.
        ignore_rule: Which posed GT are "ignore": ``"labeled"`` -- none (a GT with
            any labeled keypoint counts; xtcocotools ``keypoints``, i.e. mmpose's
            COCO / OCHuman eval) -- or ``"num_keypoints"`` -- those whose *source*
            ``num_keypoints`` field is 0 (pycocotools on the raw JSON;
            xtcocotools ``keypoints_crowd``, i.e. CrowdPose, where the field counts
            only ``v=2``). Needs ``/human_meta``. The two agree on COCO.

    Returns:
        A list of :class:`CocoImage`, one per GT labeled frame.
    """
    if area_source not in ("auto", "meta", "bbox", "keypoints"):
        raise ValueError(f"Unknown area_source {area_source!r}.")
    if ignore_rule not in ("labeled", "num_keypoints"):
        raise ValueError(f"Unknown ignore_rule {ignore_rule!r}.")
    meta = _read_human_meta(labels_gt)
    if meta is None and (area_source == "meta" or ignore_rule == "num_keypoints"):
        raise ValueError(
            f"area_source={area_source!r} / ignore_rule={ignore_rule!r} need the "
            "GT file's /human_meta group."
        )
    use_meta_area = meta is not None and area_source in ("auto", "meta")

    n_nodes = len(labels_gt.skeletons[0].nodes)
    match = labels_gt.match(labels_pr)
    gt_to_pr = {id(g): p for p, g in match.video_map.items() if g is not None}
    pr_index: Dict[Tuple[int, int], sio.LabeledFrame] = {}
    for lf in labels_pr.labeled_frames:
        pr_index[id(lf.video), lf.frame_idx] = lf

    images = []
    for lf_idx, lf in enumerate(labels_gt.labeled_frames):
        image_id = int(meta["frame_image_id"][lf_idx]) if meta is not None else lf_idx
        insts = lf.user_instances if user_labels_only else lf.instances
        inst_pos = {id(inst): i for i, inst in enumerate(lf.instances)}
        box_of_inst = {}
        ignore_boxes = []
        for b_idx, box in enumerate(getattr(lf, "bboxes", []) or []):
            if getattr(box, "is_predicted", False):
                continue
            linked = getattr(box, "instance", None)
            cat = _category(box)
            if linked is not None:
                box_of_inst[id(linked)] = (b_idx, box)
            elif use_ignore_regions and cat in ("person", "crowd", ""):
                ignore_boxes.append((b_idx, box, cat == "crowd"))

        pts, area, bbox, crowd, ignore = [], [], [], [], []
        for inst in insts:
            p = inst.numpy()[:, :2].astype("float64")
            linked = box_of_inst.get(id(inst))
            if np.isnan(p).any(axis=-1).all():
                # A person with no labeled keypoint: COCO's num_keypoints == 0,
                # an ignore region when its box is known.
                if linked is not None and use_ignore_regions:
                    ignore_boxes.append((linked[0], linked[1], False))
                continue
            a = np.nan
            ig = False
            if ignore_rule == "num_keypoints":
                ig = meta["inst_nkpt"].get((lf_idx, inst_pos[id(inst)]), 1) == 0
            if use_meta_area:
                a = meta["inst_area"].get((lf_idx, inst_pos[id(inst)]), np.nan)
            if np.isnan(a) and area_source in ("auto", "bbox") and linked:
                xywh = _box_xywh(linked[1])
                a = xywh[2] * xywh[3] * bbox_area_scale
            if np.isnan(a):
                a = _kpt_box_area(p)
            pts.append(p)
            area.append(a)
            bbox.append(_box_xywh(linked[1]) if linked else np.full(4, np.nan))
            crowd.append(False)
            ignore.append(ig)
        for b_idx, box, is_crowd in ignore_boxes:
            xywh = _box_xywh(box)
            a = np.nan
            if use_meta_area:
                a = meta["bbox_area"].get((lf_idx, b_idx), np.nan)
            if np.isnan(a):
                a = xywh[2] * xywh[3] * bbox_area_scale
            pts.append(np.full((n_nodes, 2), np.nan))
            area.append(a)
            bbox.append(xywh)
            crowd.append(is_crowd)
            ignore.append(True)

        lf_pr = None
        video_pr = gt_to_pr.get(id(lf.video))
        if video_pr is not None:
            lf_pr = pr_index.get((id(video_pr), lf.frame_idx))
        dpts, dsc = [], []
        if lf_pr is not None:
            for inst in lf_pr.instances:
                if not isinstance(inst, sio.PredictedInstance):
                    continue
                dpts.append(inst.numpy()[:, :2].astype("float64"))
                dsc.append(float(inst.score))

        images.append(
            CocoImage(
                key=(_video_key(lf.video), int(lf.frame_idx)),
                image_id=image_id,
                gt_points=np.asarray(pts).reshape(-1, n_nodes, 2),
                gt_area=np.asarray(area, dtype="float64"),
                gt_bbox=np.asarray(bbox, dtype="float64").reshape(-1, 4),
                gt_iscrowd=np.asarray(crowd, dtype=bool),
                gt_ignore=np.asarray(ignore, dtype=bool),
                dt_points=np.asarray(dpts).reshape(-1, n_nodes, 2),
                dt_scores=np.asarray(dsc, dtype="float64"),
            )
        )
    if meta is not None:
        # COCOeval visits images in image_id order; this fixes score-tie order.
        images.sort(key=lambda im: im.image_id)
    return images


def coco_oks_matrix(img: CocoImage, sigmas: np.ndarray) -> np.ndarray:
    """OKS between every detection and GT of one image, exactly as COCOeval.

    For GT with labeled keypoints: mean over labeled GT nodes of
    ``exp(-d^2 / (2 * area * (2 sigma)^2))``. For GT without keypoints (ignore
    regions): distance from each predicted keypoint to the GT box enlarged to
    3x its size, averaged over all nodes. Missing (NaN) predicted keypoints get
    KS = 0.

    Returns:
        ``(D, G)`` array.
    """
    D, G = len(img.dt_points), len(img.gt_points)
    if D == 0 or G == 0:
        return np.zeros((D, G))
    var = (2 * sigmas) ** 2  # (K,)
    dt = img.dt_points[:, None]  # (D, 1, K, 2)
    gt = img.gt_points[None]  # (1, G, K, 2)
    vis = ~np.isnan(img.gt_points).any(-1)  # (G, K)
    k1 = vis.sum(-1)  # (G,)

    d = dt - gt
    # Ignore GT (no keypoints): distance to the box doubled on every side.
    bb = img.gt_bbox
    x0 = bb[:, 0] - bb[:, 2]
    x1 = bb[:, 0] + 2 * bb[:, 2]
    y0 = bb[:, 1] - bb[:, 3]
    y1 = bb[:, 1] + 2 * bb[:, 3]
    xd, yd = dt[..., 0], dt[..., 1]  # (D, 1, K)
    bx = np.maximum(0, x0[None, :, None] - xd) + np.maximum(0, xd - x1[None, :, None])
    by = np.maximum(0, y0[None, :, None] - yd) + np.maximum(0, yd - y1[None, :, None])
    no_kpt = (k1 == 0)[None, :, None]
    dx = np.where(no_kpt, bx, d[..., 0])
    dy = np.where(no_kpt, by, d[..., 1])

    e = (dx**2 + dy**2) / var / (img.gt_area[None, :, None] + np.spacing(1)) / 2
    ks = np.exp(-e)
    ks = np.where(np.isnan(ks), 0.0, ks)  # missing predicted keypoints
    weight = np.where(no_kpt, True, vis[None])  # (1|D, G, K)
    n = np.where(k1 == 0, ks.shape[-1], k1)  # (G,)
    return (ks * weight).sum(-1) / n[None]


def _evaluate_image(
    oks: np.ndarray,
    img: CocoImage,
    area_rng: Tuple[float, float],
    max_dets: int,
    thresholds: np.ndarray,
):
    """COCOeval.evaluateImg for one image/area range. Returns None if empty."""
    G, D = len(img.gt_points), len(img.dt_scores)
    if G == 0 and D == 0:
        return None
    g_ig = img.gt_ignore | (img.gt_area < area_rng[0]) | (img.gt_area > area_rng[1])
    gtind = np.argsort(g_ig, kind="mergesort")
    dtind = np.argsort(-img.dt_scores, kind="mergesort")[:max_dets]
    g_ig = g_ig[gtind]
    crowd = img.gt_iscrowd[gtind]
    ious = oks[dtind][:, gtind] if oks.size else oks
    T, Dk = len(thresholds), len(dtind)
    gtm = np.zeros((T, G), dtype=bool)
    dtm = np.zeros((T, Dk), dtype=bool)
    dt_ig = np.zeros((T, Dk), dtype=bool)
    if ious.size:
        for t_i, t in enumerate(thresholds):
            for d_i in range(Dk):
                best = min(t, 1 - 1e-10)
                m = -1
                for g_i in range(G):
                    if gtm[t_i, g_i] and not crowd[g_i]:
                        continue
                    if m > -1 and not g_ig[m] and g_ig[g_i]:
                        break
                    if ious[d_i, g_i] < best:
                        continue
                    best = ious[d_i, g_i]
                    m = g_i
                if m == -1:
                    continue
                dt_ig[t_i, d_i] = g_ig[m]
                dtm[t_i, d_i] = True
                gtm[t_i, m] = True
    dt_area = img.dt_area[dtind]
    out_rng = (dt_area < area_rng[0]) | (dt_area > area_rng[1])
    dt_ig = dt_ig | (~dtm & out_rng[None])
    return dict(
        scores=img.dt_scores[dtind],
        dtm=dtm,
        dt_ig=dt_ig,
        n_gt=int((~g_ig).sum()),
    )


def _accumulate(results: list, thresholds: np.ndarray, rec_thrs: np.ndarray):
    """COCOeval.accumulate for one area range / maxDets setting."""
    T, R = len(thresholds), len(rec_thrs)
    results = [r for r in results if r is not None]
    precision = -np.ones((T, R))
    recall = -np.ones(T)
    if not results:
        return precision, recall
    scores = np.concatenate([r["scores"] for r in results])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([r["dtm"] for r in results], axis=1)[:, order]
    dt_ig = np.concatenate([r["dt_ig"] for r in results], axis=1)[:, order]
    npig = sum(r["n_gt"] for r in results)
    if npig == 0:
        return precision, recall
    tps = np.logical_and(dtm, ~dt_ig)
    fps = np.logical_and(~dtm, ~dt_ig)
    tp_sum = np.cumsum(tps, axis=1).astype(float)
    fp_sum = np.cumsum(fps, axis=1).astype(float)
    for t in range(T):
        tp, fp = tp_sum[t], fp_sum[t]
        nd = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        recall[t] = rc[-1] if nd else 0
        pr = np.maximum.accumulate(pr[::-1])[::-1]
        inds = np.searchsorted(rc, rec_thrs, side="left")
        q = np.zeros(R)
        valid = inds < nd
        q[valid] = pr[inds[valid]]
        precision[t] = q
    return precision, recall


def _mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(x.mean()) if x.size else -1.0


def resolve_protocol(
    labels_gt: sio.Labels, protocol: Optional[str]
) -> Optional[Dict[str, Any]]:
    """The :data:`EVAL_PROTOCOLS` preset for ``protocol`` (``None`` for none).

    ``"auto"`` picks the preset of the GT ``provenance["dataset"]`` and returns
    ``None`` when there is no such preset; an explicit unknown name raises.
    """
    if protocol is None:
        return None
    if protocol == "auto":
        name = (labels_gt.provenance or {}).get("dataset")
        return EVAL_PROTOCOLS.get(name) if isinstance(name, str) else None
    if protocol not in EVAL_PROTOCOLS:
        raise ValueError(
            f"Unknown COCO eval protocol {protocol!r}; choose 'auto' or one of "
            f"{sorted(EVAL_PROTOCOLS)}."
        )
    return EVAL_PROTOCOLS[protocol]


def coco_keypoint_metrics(
    labels_gt: sio.Labels,
    labels_pr: sio.Labels,
    sigmas: SigmaSpec = None,
    area_source: Optional[str] = None,
    bbox_area_scale: float = 0.53,
    max_dets: int = 20,
    use_ignore_regions: bool = True,
    ignore_rule: Optional[str] = None,
    backend: str = "native",
    images: Optional[List[CocoImage]] = None,
    protocol: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute COCO keypoint AP/AR (COCOeval ``keypoints`` summary) on Labels.

    Args:
        labels_gt: Ground truth labels.
        labels_pr: Predicted labels.
        sigmas: Per-node OKS sigmas spec; see :func:`resolve_node_sigmas`.
        area_source: GT area source; see :func:`labels_to_coco_images`. ``None``
            takes the protocol's, else ``"auto"``.
        bbox_area_scale: Multiplier on box-derived GT areas.
        max_dets: Detections kept per image (COCO: 20).
        use_ignore_regions: Treat crowd / keypoint-less boxes as ignore GT.
        ignore_rule: ``"labeled"`` or ``"num_keypoints"``; see
            :func:`labels_to_coco_images`. ``None`` takes the protocol's, else
            ``"labeled"``.
        backend: ``"native"`` (numpy reimplementation) or ``"pycocotools"``.
        images: Precomputed :func:`labels_to_coco_images` output (reuse across
            calls).
        protocol: ``"auto"`` (from the GT ``provenance["dataset"]``, no preset
            when the dataset has none) or a key of ``EVAL_PROTOCOLS``. The preset
            fills ``ignore_rule`` / ``area_source`` when they are ``None``, and
            ``sigmas`` when neither they nor the GT provenance give any.

    Returns:
        Dict with ``coco.AP``, ``coco.AP50``, ``coco.AP75``, ``coco.AP_M``,
        ``coco.AP_L``, ``coco.AR``, ``coco.AR50``, ``coco.AR75``, ``coco.AR_M``,
        ``coco.AR_L``, plus ``coco.sigmas``, counts and the ``(T, R)`` precision
        array for the ``all`` range.
    """
    proto = resolve_protocol(labels_gt, protocol)
    if proto is not None:
        ignore_rule = ignore_rule or proto["ignore_rule"]
        area_source = area_source or proto["area_source"]
        if sigmas is None and not (labels_gt.provenance or {}).get("node_sigmas"):
            sigmas = proto["sigmas"]
    area_source = area_source or "auto"
    ignore_rule = ignore_rule or "labeled"
    skeleton = labels_gt.skeletons[0]
    sig = resolve_node_sigmas(skeleton, sigmas, provenance=labels_gt.provenance)
    if images is None:
        images = labels_to_coco_images(
            labels_gt,
            labels_pr,
            area_source=area_source,
            bbox_area_scale=bbox_area_scale,
            use_ignore_regions=use_ignore_regions,
            ignore_rule=ignore_rule,
        )
    if backend == "pycocotools":
        return _pycocotools_metrics(images, sig, max_dets=max_dets)
    if backend != "native":
        raise ValueError(f"Unknown backend {backend!r}.")

    thresholds = np.linspace(0.5, 0.95, 10)
    rec_thrs = np.linspace(0.0, 1.00, 101)
    ious = [coco_oks_matrix(img, sig) for img in images]
    out: Dict[str, Any] = {}
    for rng_name, rng in COCO_KPT_AREA_RANGES.items():
        res = [
            _evaluate_image(o, img, rng, max_dets, thresholds)
            for o, img in zip(ious, images)
        ]
        precision, recall = _accumulate(res, thresholds, rec_thrs)
        suffix = {"all": "", "medium": "_M", "large": "_L"}[rng_name]
        out[f"coco.AP{suffix}"] = _mean_valid(precision)
        out[f"coco.AR{suffix}"] = _mean_valid(recall)
        if rng_name == "all":
            out["coco.AP50"] = _mean_valid(precision[0])
            out["coco.AP75"] = _mean_valid(precision[5])
            out["coco.AR50"] = float(recall[0])
            out["coco.AR75"] = float(recall[5])
            out["coco.precision"] = precision
            out["coco.recall"] = recall
    out["coco.sigmas"] = sig
    out.update(_counts(images))
    return out


def coco_images_to_dicts(
    images: List[CocoImage], n_nodes: int
) -> Tuple[dict, List[dict]]:
    """Convert :class:`CocoImage` records to a COCO GT dict and a results list.

    NaN predicted keypoints are written far off-image (so their KS is 0, as in the
    native path); NaN GT keypoints become ``v=0``.
    """
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person", "keypoints": [""] * n_nodes}],
    }
    results = []
    ann_id = 1
    for img_i, img in enumerate(images):
        iid = img_i + 1
        gt["images"].append({"id": iid, "file_name": f"{img.key}"})
        for g in range(len(img.gt_points)):
            p = img.gt_points[g]
            vis = ~np.isnan(p).any(-1)
            kp = np.zeros((n_nodes, 3))
            kp[vis, :2] = p[vis]
            kp[vis, 2] = 2
            bb = img.gt_bbox[g]
            if np.isnan(bb).any():
                if vis.any():
                    lo, hi = p[vis].min(0), p[vis].max(0)
                    bb = np.r_[lo, hi - lo]
                else:
                    bb = np.zeros(4)
            gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": iid,
                    "category_id": 1,
                    "keypoints": kp.ravel().tolist(),
                    "num_keypoints": 0 if img.gt_ignore[g] else int(vis.sum()),
                    "area": float(img.gt_area[g]),
                    "bbox": [float(v) for v in bb],
                    "iscrowd": int(img.gt_iscrowd[g]),
                }
            )
            ann_id += 1
        for d in range(len(img.dt_scores)):
            p = np.nan_to_num(img.dt_points[d], nan=-1e6)
            kp = np.concatenate([p, np.ones((n_nodes, 1))], 1)
            results.append(
                {
                    "image_id": iid,
                    "category_id": 1,
                    "keypoints": kp.ravel().tolist(),
                    "score": float(img.dt_scores[d]),
                }
            )
    return gt, results


def _pycocotools_metrics(
    images: List[CocoImage], sigmas: np.ndarray, max_dets: int = 20
) -> Dict[str, Any]:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt, results = coco_images_to_dicts(images, len(sigmas))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results) if results else COCO()
        ev = COCOeval(coco_gt, coco_dt, "keypoints")
        ev.params.kpt_oks_sigmas = np.asarray(sigmas, dtype="float64")
        ev.params.maxDets = [max_dets]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    keys = ["AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_M", "AR_L"]
    out = {f"coco.{k}": float(v) for k, v in zip(keys, s)}
    out["coco.sigmas"] = np.asarray(sigmas)
    out.update(_counts(images))
    return out


def _counts(images: List[CocoImage]) -> Dict[str, int]:
    return {
        "coco.n_images": len(images),
        "coco.n_gt": int(sum((~im.gt_ignore).sum() for im in images)),
        "coco.n_gt_ignore": int(sum(im.gt_ignore.sum() for im in images)),
        "coco.n_dt": int(sum(len(im.dt_scores) for im in images)),
    }


def frame_image_ids(labels: sio.Labels) -> List[Optional[int]]:
    """COCO ``image_id`` of every labeled frame.

    Read from ``/human_meta`` when the file has it, else from the trailing digits
    of the frame's image file name (``000000000139.jpg`` -> ``139``) for image-list
    videos. Frames of single-file videos have no image id (``None``).
    """
    meta = _read_human_meta(labels)
    if meta is not None:
        return [int(i) for i in meta["frame_image_id"]]
    ids: List[Optional[int]] = []
    for lf in labels.labeled_frames:
        fn = lf.video.filename
        if not isinstance(fn, list) or not 0 <= lf.frame_idx < len(fn):
            ids.append(None)
            continue
        m = re.search(r"(\d+)\.\w+$", str(fn[lf.frame_idx]))
        ids.append(int(m.group(1)) if m else None)
    return ids


def attach_coco_detections(
    labels: sio.Labels,
    detections: Union[str, Path, List[dict]],
    score_threshold: float = 0.0,
    category_id: Optional[int] = 1,
    category_name: str = "person",
    replace: bool = True,
) -> Dict[str, int]:
    """Attach COCO-format detections as ``PredictedBoundingBox``es, in place.

    Loads a COCO results file (``[{"image_id", "bbox": xywh, "score",
    "category_id"}]``, e.g. the SimpleBaseline ``COCO_val2017_detections_AP_H_56_
    person.json`` every top-down paper uses) onto the matching labeled frames, so a
    box-seeded top-down predictor can read them as its stage 1.

    Args:
        labels: Labels whose frames correspond to the detection ``image_id``s
            (via ``/human_meta`` or the trailing digits of each image file name).
        detections: Path to the JSON results file, or the parsed list.
        score_threshold: Drop detections scoring below this.
        category_id: Keep only this category (``None`` keeps all).
        category_name: Category name for the attached boxes.
        replace: Remove existing predicted boxes on those frames first.

    Returns:
        Counts: ``n_boxes`` attached, ``n_frames`` touched, ``n_unmatched`` boxes
        whose image is not in ``labels``.
    """
    if not isinstance(detections, list):
        with open(detections) as f:
            detections = json.load(f)
    lf_of_image = {}
    for iid, lf in zip(frame_image_ids(labels), labels.labeled_frames):
        if iid is not None:
            lf_of_image[iid] = lf
    if not lf_of_image:
        raise ValueError(
            "No labeled frame has a COCO image id: the labels need /human_meta or "
            "image-list videos whose file names end in the image id."
        )
    category = sio.Category(category_name)
    touched, n, unmatched = set(), 0, 0
    for det in detections:
        if category_id is not None and det.get("category_id", 1) != category_id:
            continue
        if det["score"] < score_threshold:
            continue
        lf = lf_of_image.get(int(det["image_id"]))
        if lf is None:
            unmatched += 1
            continue
        if replace and id(lf) not in touched:
            lf.bboxes = [b for b in lf.bboxes if not b.is_predicted]
        touched.add(id(lf))
        lf.bboxes.append(
            sio.PredictedBoundingBox.from_xywh(
                *det["bbox"], score=float(det["score"]), category=category
            )
        )
        n += 1
    return dict(n_boxes=n, n_frames=len(touched), n_unmatched=unmatched)
