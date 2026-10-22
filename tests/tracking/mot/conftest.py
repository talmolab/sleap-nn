"""Shared synthetic scenes and helpers for the MOT tracker tests.

Everything here is a plain function (import with ``from conftest import ...``;
``tests/tracking/mot`` has no ``__init__.py``, so pytest puts this directory on
``sys.path``) plus a few fixtures wrapping them.

A *scene* is a list of ``(frame_idx, objects)`` frames -- exactly what a MOT
tracker's ``prepare()`` / ``track()`` consume, no ``sio.Labels`` involved -- plus the
ground-truth identity of every object (``-1`` for duplicates and false positives).

Typical use::

    traj = crossing_trajectories(n_frames=30)
    scene = make_scene(traj, embeddings=True)
    names = run_tracker(build_tracker("sort"), scene.frames)
    assert count_id_switches(names, scene.gt) == 0
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import attrs
import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.tracking.mot import MOT_TRACKERS, MOTParams

# Five-node animal: nose/tail along the heading, two lateral nodes so the box is
# never degenerate (h > 0) when the animal is axis-aligned.
SKELETON = sio.Skeleton(
    nodes=["head", "thorax", "tail", "left", "right"], name="mot_animal"
)
SKELETON_1 = sio.Skeleton(nodes=["centroid"], name="mot_point")

BODY_LENGTH = 40.0  # px, head -> tail
BODY_WIDTH = 16.0  # px, left -> right
IMAGE_SHAPE = (240, 320)  # (height, width) for masks
MASK_HALF_SIZE = (14, 9)  # (half width, half height) of a mask rectangle

Number = Union[int, float]


# --------------------------------------------------------------------- scenes
@attrs.define
class Scene:
    """A synthetic multi-animal video.

    Attributes:
        frames: ``(frame_idx, objects)`` per frame, in temporal order.
        gt: Ground-truth identity per object, aligned with ``frames`` (``-1``
            for duplicates / false positives).
        carrier: ``"pose"`` or ``"mask"``.
        skeleton: Skeleton of the pose objects (``None`` for masks).
        id_vectors: ``(n_ids, D)`` per-identity unit vectors, or ``None``.
    """

    frames: List[Tuple[int, list]]
    gt: List[List[int]]
    carrier: str = "pose"
    skeleton: Optional[sio.Skeleton] = None
    id_vectors: Optional[np.ndarray] = None

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return len(self.frames)

    @property
    def frame_indices(self) -> List[int]:
        """Frame indices, in order."""
        return [fi for fi, _ in self.frames]

    def objects(self) -> list:
        """Every object of every frame, flattened in order."""
        return [obj for _, objs in self.frames for obj in objs]


def linear_trajectories(
    starts: Sequence[Sequence[Number]],
    velocities: Sequence[Sequence[Number]],
    n_frames: int,
) -> np.ndarray:
    """Constant-velocity trajectories.

    Args:
        starts: ``(N, 2)`` start positions ``(x, y)``.
        velocities: ``(N, 2)`` per-frame displacements.
        n_frames: Number of frames ``T``.

    Returns:
        ``(T, N, 2)`` positions.
    """
    starts = np.asarray(starts, dtype=np.float64).reshape(-1, 2)
    velocities = np.asarray(velocities, dtype=np.float64).reshape(-1, 2)
    t = np.arange(n_frames, dtype=np.float64)[:, None, None]
    return starts[None] + t * velocities[None]


def static_trajectories(
    positions: Sequence[Sequence[Number]], n_frames: int
) -> np.ndarray:
    """``(T, N, 2)`` animals that do not move."""
    return linear_trajectories(positions, np.zeros((len(positions), 2)), n_frames)


def crossing_trajectories(
    n_frames: int = 30,
    speed: float = 6.0,
    offset: float = 12.0,
    center: Tuple[float, float] = (160.0, 120.0),
) -> np.ndarray:
    """Two animals crossing head-on along ``x``, meeting at the middle frame.

    Args:
        n_frames: Number of frames.
        speed: Per-frame speed of each animal (px).
        offset: Vertical separation of the two lanes (px); 0 = exact crossing.
        center: Meeting point.

    Returns:
        ``(T, 2, 2)`` positions: animal 0 moves right, animal 1 moves left.
    """
    half = 0.5 * (n_frames - 1) * speed
    cx, cy = center
    starts = [(cx - half, cy - 0.5 * offset), (cx + half, cy + 0.5 * offset)]
    return linear_trajectories(starts, [(speed, 0.0), (-speed, 0.0)], n_frames)


def with_gaps(trajectories: np.ndarray, gaps: Dict[int, Sequence[int]]) -> np.ndarray:
    """Copy of ``trajectories`` with the animal absent (NaN) on some frames.

    Args:
        trajectories: ``(T, N, 2)`` positions.
        gaps: ``{animal: frame positions}`` (positions into the ``T`` axis).

    Returns:
        The new ``(T, N, 2)`` array; NaN rows produce no detection.
    """
    out = np.array(trajectories, dtype=np.float64, copy=True)
    for animal, frames in gaps.items():
        out[list(frames), animal] = np.nan
    return out


def identity_vectors(n_ids: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    """``(n_ids, dim)`` well-separated unit vectors (orthonormal if ``n_ids <= dim``)."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((max(n_ids, dim), dim))
    if n_ids <= dim:
        q, _ = np.linalg.qr(g.T)
        return np.ascontiguousarray(q.T[:n_ids])
    return g[:n_ids] / np.linalg.norm(g[:n_ids], axis=1, keepdims=True)


def pose_points(
    center: Sequence[Number],
    heading: float = 0.0,
    length: float = BODY_LENGTH,
    width: float = BODY_WIDTH,
    n_nodes: int = 5,
) -> np.ndarray:
    """``(K, 2)`` keypoints of an animal centered on ``center``.

    Five nodes: head, thorax (= center), tail along the heading (radians), and two
    lateral nodes. ``n_nodes=1`` gives the single center point.
    """
    c = np.asarray(center, dtype=np.float64).reshape(2)
    if n_nodes == 1:
        return c[None].copy()
    d = np.array([np.cos(heading), np.sin(heading)])
    perp = np.array([-d[1], d[0]])
    pts = np.stack(
        [
            c + 0.5 * length * d,
            c,
            c - 0.5 * length * d,
            c + 0.5 * width * perp,
            c - 0.5 * width * perp,
        ]
    )
    return pts[:n_nodes]


def make_pose(
    center: Sequence[Number],
    *,
    heading: float = 0.0,
    skeleton: sio.Skeleton = SKELETON,
    score: float = 0.9,
    point_scores: Optional[np.ndarray] = None,
    nan_nodes: Sequence[int] = (),
    all_nan: bool = False,
    embedding: Optional[np.ndarray] = None,
    user: bool = False,
    track: Optional[sio.Track] = None,
    tracking_score: Optional[float] = None,
):
    """One pose detection.

    Args:
        center: Body center ``(x, y)``.
        heading: Body direction (radians).
        skeleton: :data:`SKELETON` (5 nodes) or :data:`SKELETON_1` (1 node).
        score: Instance score; also every point score unless ``point_scores``.
        point_scores: ``(K,)`` point scores.
        nan_nodes: Node indices to make invisible.
        all_nan: Make every node invisible (an invalid detection).
        embedding: Appearance vector attached as ``identity_embedding``.
        user: Build a user ``sio.Instance`` (no scores) instead.
        track: Pre-existing (stale) track to attach.
        tracking_score: Pre-existing tracking score to attach.

    Returns:
        A ``sio.PredictedInstance`` (or ``sio.Instance`` if ``user``).
    """
    pts = pose_points(center, heading=heading, n_nodes=len(skeleton.nodes))
    pts[list(nan_nodes)] = np.nan
    if all_nan:
        pts[:] = np.nan
    if user:
        obj = sio.Instance.from_numpy(pts, skeleton=skeleton)
    else:
        if point_scores is None:
            point_scores = np.full(len(pts), score, dtype=np.float64)
        obj = sio.PredictedInstance.from_numpy(
            pts,
            skeleton=skeleton,
            score=float(score),
            point_scores=np.asarray(point_scores, dtype=np.float64),
        )
    if embedding is not None:
        obj.identity_embedding = sio.Embedding(np.asarray(embedding, dtype=np.float32))
    obj.track = track
    if tracking_score is not None:
        obj.tracking_score = tracking_score
    return obj


def make_mask(
    center: Sequence[Number],
    *,
    half_size: Tuple[int, int] = MASK_HALF_SIZE,
    shape: Tuple[int, int] = IMAGE_SHAPE,
    score: float = 0.9,
    embedding: Optional[np.ndarray] = None,
    empty: bool = False,
    track: Optional[sio.Track] = None,
) -> sio.PredictedSegmentationMask:
    """One rectangular ``sio.PredictedSegmentationMask`` centered on ``center``.

    Args:
        center: ``(x, y)`` (clipped to the image).
        half_size: ``(half width, half height)`` in px.
        shape: Image ``(height, width)``.
        score: Mask score.
        embedding: Appearance vector attached as ``identity_embedding``.
        empty: Return an empty (all-False) mask (an invalid detection).
        track: Pre-existing (stale) track to attach.

    Returns:
        The mask.
    """
    data = np.zeros(shape, dtype=bool)
    if not empty:
        x, y = (int(round(v)) for v in center)
        hw, hh = half_size
        y0, y1 = max(0, y - hh), min(shape[0], y + hh)
        x0, x1 = max(0, x - hw), min(shape[1], x + hw)
        data[y0:y1, x0:x1] = True
    mask = sio.PredictedSegmentationMask.from_numpy(data, score=float(score))
    if embedding is not None:
        mask.identity_embedding = sio.Embedding(np.asarray(embedding, dtype=np.float32))
    mask.track = track
    return mask


def _per_det(value, t: int, k: int):
    """``value`` if scalar, else ``value[t, k]``."""
    if np.ndim(value) == 0:
        return value
    return np.asarray(value)[t, k]


def make_scene(
    trajectories: np.ndarray,
    *,
    carrier: str = "pose",
    frame_indices: Optional[Sequence[int]] = None,
    scores: Union[float, np.ndarray] = 0.9,
    heading: Union[float, np.ndarray] = 0.0,
    n_nodes: int = 5,
    nan_node_prob: float = 0.0,
    min_visible: int = 2,
    position_noise: float = 0.0,
    embeddings: bool = False,
    embedding_dim: int = 16,
    embedding_noise: float = 0.05,
    duplicates: Sequence[Tuple[int, int]] = (),
    duplicate_offset: Tuple[float, float] = (3.0, 2.0),
    duplicate_score: float = 0.5,
    false_positives: Sequence[Tuple[int, Tuple[float, float], float]] = (),
    shuffle: bool = False,
    user: bool = False,
    stale_track: Optional[sio.Track] = None,
    seed: int = 0,
) -> Scene:
    """Build a scene from ``(T, N, 2)`` trajectories (NaN = animal absent).

    Deterministic for a given ``seed``: building twice gives equal (but distinct)
    objects, e.g. for determinism tests.

    Args:
        trajectories: ``(T, N, 2)`` centers; a NaN row means no detection.
        carrier: ``"pose"`` or ``"mask"``.
        frame_indices: Frame index of each of the ``T`` frames (default
            ``range(T)``; use it for strides and gaps).
        scores: Detection score, scalar or ``(T, N)``.
        heading: Body heading (radians), scalar or ``(T, N)``.
        n_nodes: 5 (:data:`SKELETON`) or 1 (:data:`SKELETON_1`).
        nan_node_prob: Probability of each node being invisible (pose).
        min_visible: Nodes kept visible regardless of ``nan_node_prob``.
        position_noise: Std (px) of Gaussian noise added to every keypoint /
            mask center.
        embeddings: Attach per-identity vectors (with noise) to every detection.
        embedding_dim: Vector dimension.
        embedding_noise: Std of the Gaussian noise added before normalizing.
        duplicates: ``(frame position, animal)`` pairs that get a second,
            slightly offset detection (gt ``-1``).
        duplicate_offset: Offset ``(dx, dy)`` of duplicates.
        duplicate_score: Score of duplicates.
        false_positives: ``(frame position, (x, y), score)`` extra detections
            (gt ``-1``).
        shuffle: Shuffle detection order within each frame.
        user: Build user ``sio.Instance`` poses (no scores).
        stale_track: Attach this track to every object (stale-label tests).
        seed: Random seed.

    Returns:
        The :class:`Scene`.
    """
    traj = np.asarray(trajectories, dtype=np.float64)
    n_frames, n_ids = traj.shape[:2]
    rng = np.random.default_rng(seed)
    if frame_indices is None:
        frame_indices = list(range(n_frames))
    frame_indices = [int(f) for f in frame_indices]
    assert len(frame_indices) == n_frames
    skeleton = SKELETON if n_nodes == 5 else SKELETON_1
    id_vecs = identity_vectors(n_ids, embedding_dim, seed=seed) if embeddings else None

    def vector(animal: int) -> Optional[np.ndarray]:
        if id_vecs is None:
            return None
        v = id_vecs[animal] + embedding_noise * rng.standard_normal(embedding_dim)
        return v / np.linalg.norm(v)

    def build(center, animal, t, score):
        center = np.asarray(center, dtype=np.float64)
        if position_noise > 0:
            center = center + position_noise * rng.standard_normal(2)
        emb = vector(animal) if animal >= 0 else None
        if carrier == "mask":
            return make_mask(center, score=score, embedding=emb, track=stale_track)
        nan_nodes: List[int] = []
        if nan_node_prob > 0 and n_nodes > 1:
            drop = rng.random(n_nodes) < nan_node_prob
            keep = rng.permutation(n_nodes)[:min_visible]
            drop[keep] = False
            nan_nodes = list(np.flatnonzero(drop))
        k = max(animal, 0)
        return make_pose(
            center,
            heading=float(_per_det(heading, t, k)),
            skeleton=skeleton,
            score=score,
            nan_nodes=nan_nodes,
            embedding=emb,
            user=user,
            track=stale_track,
            tracking_score=None if stale_track is None else 0.7,
        )

    dup_set = {(int(t), int(k)) for t, k in duplicates}
    frames: List[Tuple[int, list]] = []
    gt: List[List[int]] = []
    for t in range(n_frames):
        objs, ids = [], []
        for k in range(n_ids):
            if not np.isfinite(traj[t, k]).all():
                continue
            score = float(_per_det(scores, t, k))
            objs.append(build(traj[t, k], k, t, score))
            ids.append(k)
            if (t, k) in dup_set:
                dup_center = traj[t, k] + np.asarray(duplicate_offset)
                dup = build(dup_center, -1, t, duplicate_score)
                if embeddings and id_vecs is not None:
                    dup.identity_embedding = sio.Embedding(vector(k).astype(np.float32))
                objs.append(dup)
                ids.append(-1)
        for ft, center, score in false_positives:
            if int(ft) == t:
                objs.append(build(center, -1, t, float(score)))
                ids.append(-1)
        if shuffle and objs:
            order = rng.permutation(len(objs))
            objs = [objs[i] for i in order]
            ids = [ids[i] for i in order]
        frames.append((frame_indices[t], objs))
        gt.append(ids)
    return Scene(
        frames=frames,
        gt=gt,
        carrier=carrier,
        skeleton=None if carrier == "mask" else skeleton,
        id_vectors=id_vecs,
    )


# ------------------------------------------------------------------- running
def build_tracker(
    name: str,
    *,
    carrier: str = "pose",
    n_nodes: Optional[int] = 5,
    preset: str = "animal",
    track_name_offset: int = 0,
    **overrides: Any,
):
    """Instantiate a MOT tracker by name with preset + overrides.

    Args:
        name: One of ``MOT_TRACKER_NAMES``.
        carrier: ``"pose"`` or ``"mask"``.
        n_nodes: Skeleton size passed to the tracker (``None``: unknown).
        preset: ``"animal"`` or ``"paper"``.
        track_name_offset: Value for ``tracker.track_name_offset``.
        **overrides: :class:`MOTParams` overrides.

    Returns:
        The tracker.
    """
    params = MOTParams.for_tracker(name, preset=preset, **overrides)
    tracker = MOT_TRACKERS[name](params, carrier=carrier, n_nodes=n_nodes)
    tracker.track_name_offset = track_name_offset
    return tracker


def track_frames(
    tracker,
    frames: Sequence[Tuple[int, Sequence]],
    *,
    prepare: bool = True,
    finalize: bool = True,
) -> List[list]:
    """Run ``prepare -> track per frame -> finalize``; return each ``track()`` output.

    Args:
        tracker: A MOT tracker.
        frames: ``(frame_idx, objects)`` per frame.
        prepare: Call ``tracker.prepare(frames)`` first.
        finalize: Call ``tracker.finalize()`` at the end.

    Returns:
        The list returned by ``track()`` for every frame.
    """
    frames = [(fi, list(objs)) for fi, objs in frames]
    if prepare:
        tracker.prepare(frames)
    outputs = [tracker.track(objs, fi) for fi, objs in frames]
    if finalize:
        tracker.finalize()
    return outputs


def track_names(frames: Sequence[Tuple[int, Sequence]]) -> List[List[Optional[str]]]:
    """Current track name of every object (``None`` if untracked), per frame."""
    return [
        [None if getattr(o, "track", None) is None else o.track.name for o in objs]
        for _, objs in frames
    ]


def run_tracker(
    tracker,
    frames: Sequence[Tuple[int, Sequence]],
    *,
    prepare: bool = True,
    finalize: bool = True,
) -> List[List[Optional[str]]]:
    """Track ``frames`` and return per-frame track names read AFTER the run.

    Names are read at the end because tracks are labeled retroactively at
    confirmation (and may be relabeled by linking in ``finalize``).

    Args:
        tracker: A MOT tracker.
        frames: ``(frame_idx, objects)`` per frame.
        prepare: Call ``tracker.prepare(frames)`` first.
        finalize: Call ``tracker.finalize()`` at the end.

    Returns:
        ``names[t][i]``: the track name of object ``i`` of frame ``t`` or ``None``.
    """
    frames = [(fi, list(objs)) for fi, objs in frames]
    track_frames(tracker, frames, prepare=prepare, finalize=finalize)
    return track_names(frames)


# ------------------------------------------------------------------- scoring
def names_by_identity(
    names: Sequence[Sequence[Optional[str]]], gt: Sequence[Sequence[int]]
) -> Dict[int, List[Optional[str]]]:
    """Per ground-truth identity, its predicted name on each frame it is present.

    Args:
        names: Output of :func:`run_tracker`.
        gt: ``Scene.gt``.

    Returns:
        ``{gt_id: [name or None per frame where gt_id is detected]}`` (gt ``-1``
        excluded).
    """
    out: Dict[int, List[Optional[str]]] = {}
    for frame_names, frame_gt in zip(names, gt):
        for name, gid in zip(frame_names, frame_gt):
            if gid >= 0:
                out.setdefault(int(gid), []).append(name)
    return out


def count_id_switches(
    names: Sequence[Sequence[Optional[str]]], gt: Sequence[Sequence[int]]
) -> int:
    """Number of identity switches against ground truth.

    For each ground-truth identity, walk its detections in time and count every
    change of its (non-``None``) predicted name; untracked detections neither
    count nor break the walk. Also counts a name shared by two identities in the
    same frame (two objects with one track) as one switch per extra identity.

    Args:
        names: Output of :func:`run_tracker`.
        gt: ``Scene.gt``.

    Returns:
        The number of switches (0 = perfect identity preservation).
    """
    switches = 0
    for seq in names_by_identity(names, gt).values():
        previous = None
        for name in seq:
            if name is None:
                continue
            if previous is not None and name != previous:
                switches += 1
            previous = name
    for frame_names, frame_gt in zip(names, gt):
        tracked = [n for n, g in zip(frame_names, frame_gt) if n is not None and g >= 0]
        switches += len(tracked) - len(set(tracked))
    return switches


def distinct_names(names: Sequence[Sequence[Optional[str]]]) -> List[str]:
    """Sorted distinct non-``None`` names."""
    return sorted({n for frame in names for n in frame if n is not None})


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def two_animal_scene() -> Scene:
    """Two well-separated animals moving in parallel for 20 frames, with vectors."""
    traj = linear_trajectories([(60, 60), (60, 180)], [(4, 0), (4, 0)], 20)
    return make_scene(traj, embeddings=True)


@pytest.fixture
def crossing_scene() -> Scene:
    """Two animals crossing head-on (lanes 12 px apart), with vectors."""
    return make_scene(crossing_trajectories(n_frames=30), embeddings=True)
