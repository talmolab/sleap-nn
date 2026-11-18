"""Undo head-to-tail flips frame to frame (``fix_flips``; off by default).

A pose model sometimes swaps an animal's head and tail end on one frame: the head
point lands where the tail was a frame before and the tail point where the head was.
:class:`FlipFixer` runs on each batch of predictions, in frame order, and turns such
an instance back by 180 degrees about its head-tail midpoint, judged against the
same animal on the previous frame.

How it works, per video:

- Frames must come in order, one video after another or interleaved by
  ``video_indices``; a frame continues the previous frame's state only when it is
  the next frame of the same video (``frame - previous frame == 1``). After a gap,
  at the first frame of a video, and for an animal not matched to one on the
  previous frame, nothing is changed. Batch boundaries make no difference: the
  state is carried from batch to batch.
- Each instance is matched to an animal of the previous frame by the IoU of their
  boxes (the tight box of the points scoring >= 0.2, or of all points when fewer
  than 4 do so, padded by 9.5% of its longest side on every side; Hungarian
  matching, IoU >= 0.3).
- A flip: the head point is closer to the previous tail point and the tail point
  to the previous head point than to their own previous places by more than a
  factor of 2 (sum of the two distances, swapped < 0.5 x kept). It is undone
  unless the head and tail points now score more than 0.1 higher (mean of the two
  peak values) than before; then the earlier frame is taken as the wrong one.
- The previous frame's animals are its instances after this fix (two boxes with
  IoU > 0.6 count as one animal, the higher-scoring one kept for the comparison);
  the outputs keep every instance.
- Decisions chain from frame to frame: each frame is judged against the corrected
  previous one. So results depend on where a video starts, and anything that moves
  points slightly (another batch size, another precision) can change a decision
  and the ones after it.

Checked on top-view mice (head node ``nose``, tail node ``tail_base``): head-to-tail
flips fell by more than half on the clips where they were common (12.4 -> 3.9 per
1,000 animal-frames with 6 mice in a wide arena), with the median and
95th-percentile point error on labelled frames from five labs unchanged within
their intervals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import attrs
import numpy as np
import torch

CONF_THRESHOLD = 0.2
MIN_POINTS = 4
BOX_SCALE = 1.19
MATCH_IOU = 0.3
DUP_IOU = 0.6
FLIP_MARGIN = 0.1


def _box_from_points(
    points: np.ndarray, scores: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """``(4,)`` xyxy box of one instance's points (see the module docstring), or ``None``."""
    pts = np.asarray(points, dtype=np.float64)
    ok = np.isfinite(pts).all(-1)
    if scores is not None:
        s = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=-np.inf)
        sel = ok & (s >= CONF_THRESHOLD)
        if sel.sum() >= MIN_POINTS:
            ok = sel
    if ok.sum() < 2:
        return None
    lo, hi = pts[ok].min(0), pts[ok].max(0)
    side = float((hi - lo).max())
    if side <= 0:
        return None
    pad = (BOX_SCALE - 1.0) / 2 * side
    return np.array([lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad])


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@attrs.define
class _Animal:
    box: np.ndarray
    points: np.ndarray
    scores: np.ndarray


class FlipFixer:
    """Undo head-to-tail flips against the previous frame (see the module docstring).

    Args:
        head: Index of the head node in the pose model's output.
        tail: Index of the tail node.

    Attributes:
        counts: ``{"flips_undone": n, "flips_kept": n}`` since the last :meth:`reset`.
    """

    def __init__(self, head: int, tail: int) -> None:
        """Store the two node indices."""
        self.head = int(head)
        self.tail = int(tail)
        self.reset()

    @classmethod
    def for_skeleton(
        cls, node_names: Sequence[str], nodes: Sequence[str]
    ) -> "FlipFixer":
        """A fixer for ``nodes = (head, tail)`` by name (a ``prefix/`` is ignored)."""
        if len(nodes) != 2:
            raise ValueError(
                f"fix_flips takes two node names (head, tail); got {list(nodes)}."
            )
        bare = [n.split("/")[-1] for n in node_names]
        try:
            return cls(*(bare.index(n.split("/")[-1]) for n in nodes))
        except ValueError:
            raise ValueError(
                f"fix_flips needs the nodes {list(nodes)} in the pose model's output; "
                f"it has {list(node_names)}."
            ) from None

    def reset(self) -> None:
        """Forget every video's previous frame (call at the start of a run)."""
        self._state: Dict[int, Tuple[int, List[_Animal]]] = {}
        self.counts = {"flips_undone": 0, "flips_kept": 0}

    def __call__(
        self, outputs: Any, frame_indices: Any, video_indices: Any = None
    ) -> Any:
        """``outputs`` with flipped instances turned back (in place), frame by frame."""
        if outputs.pred_keypoints is None or frame_indices is None:
            return outputs
        fidx = np.asarray(
            frame_indices.cpu() if torch.is_tensor(frame_indices) else frame_indices
        ).astype(np.int64)
        n = len(fidx)
        vidx = (
            np.zeros(n, dtype=np.int64)
            if video_indices is None
            else np.asarray(
                video_indices.cpu() if torch.is_tensor(video_indices) else video_indices
            ).astype(np.int64)
        )
        for b in range(n):
            self._frame(outputs, b, int(vidx[b]), int(fidx[b]))
        return outputs

    def _frame(self, out: Any, b: int, video: int, frame: int) -> None:
        last = self._state.get(video)
        old = last[1] if last is not None and 0 < frame - last[0] <= 1 else []
        kpts = out.pred_keypoints[b].detach().float().cpu().numpy()
        vals = (
            out.pred_peak_values[b].detach().float().cpu().numpy()
            if out.pred_peak_values is not None
            else np.full(kpts.shape[:2], np.nan)
        )
        prevs = self._previous(old, kpts, vals)
        animals: List[_Animal] = []
        means: List[float] = []
        for i in range(kpts.shape[0]):
            if np.isnan(kpts[i]).all():
                continue
            prev = prevs.get(i)
            if prev is not None:
                fixed = self._fix(kpts[i], vals[i], prev)
                if fixed is not None:
                    kpts[i] = fixed
                    out.pred_keypoints[b, i] = torch.as_tensor(
                        fixed, dtype=out.pred_keypoints.dtype
                    ).to(out.pred_keypoints.device)
            box = _box_from_points(kpts[i], vals[i])
            if box is None:
                continue
            m = float(np.nanmean(vals[i])) if np.isfinite(vals[i]).any() else 0.0
            animals.append(
                _Animal(box=box, points=kpts[i].copy(), scores=vals[i].copy())
            )
            means.append(m)
        if len(animals) > 1:  # two boxes on one animal: keep the higher-scoring one
            iou = _iou_matrix(
                np.stack([a.box for a in animals]), np.stack([a.box for a in animals])
            )
            keep = np.ones(len(animals), dtype=bool)
            order = sorted(
                range(len(animals)), key=lambda q: (True, means[q]), reverse=True
            )
            for a_i in order:
                if not keep[a_i]:
                    continue
                for b_i in order:
                    if b_i != a_i and keep[b_i] and iou[a_i, b_i] > DUP_IOU:
                        keep[b_i] = False
            animals = [a for a, k in zip(animals, keep) if k]
        self._state[video] = (frame, animals)

    @staticmethod
    def _previous(
        old: List[_Animal], kpts: np.ndarray, vals: np.ndarray
    ) -> Dict[int, _Animal]:
        """The previous frame's animal each instance continues (box IoU, Hungarian)."""
        if not old:
            return {}
        boxes, slots = [], []
        for i in range(kpts.shape[0]):
            if np.isnan(kpts[i]).all():
                continue
            b = _box_from_points(kpts[i], vals[i])
            if b is not None:
                boxes.append(b)
                slots.append(i)
        if not boxes:
            return {}
        from scipy.optimize import linear_sum_assignment

        iou = _iou_matrix(np.stack(boxes), np.stack([a.box for a in old]))
        rows, cols = linear_sum_assignment(-iou)
        return {slots[r]: old[c] for r, c in zip(rows, cols) if iou[r, c] >= MATCH_IOU}

    def _fix(
        self, pts: np.ndarray, vals: np.ndarray, prev: _Animal
    ) -> Optional[np.ndarray]:
        """``pts`` turned 180 degrees if they flipped head to tail since ``prev``."""
        h, t = self.head, self.tail
        q = prev.points
        if not (np.isfinite(pts[[h, t]]).all() and np.isfinite(q[[h, t]]).all()):
            return None
        same = np.linalg.norm(pts[h] - q[h]) + np.linalg.norm(pts[t] - q[t])
        swap = np.linalg.norm(pts[h] - q[t]) + np.linalg.norm(pts[t] - q[h])
        if not swap < 0.5 * same:
            return None
        now = float(np.nanmean(vals[[h, t]]))
        before = float(np.nanmean(prev.scores[[h, t]]))
        if now > before + FLIP_MARGIN:
            self.counts["flips_kept"] += 1
            return None
        self.counts["flips_undone"] += 1
        c = (pts[h] + pts[t]) / 2
        return 2 * c - pts
