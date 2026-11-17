"""``PoseSeededTopDownLayer``: run the box detector only when needed on video.

The scheduling logic is tested with a scripted stand-in for the two-stage
forward (``TopDownLayer.predict``), so it needs no trained model: a "world" of
animals per frame answers a detector call with every visible animal, and a
seeded call with the animal under each given box. Trained-model tests live in
``tests/test_bbox_topdown.py``.
"""

import numpy as np
import pytest
import torch

from sleap_nn.inference.layers import seeded as S
from sleap_nn.inference.layers.seeded import (
    PoseSeededTopDownLayer,
    SeedConfig,
    _stack,
    box_from_points,
)
from sleap_nn.inference.layers.topdown import TopDownLayer
from sleap_nn.inference.outputs import Outputs

N_NODES = 5


def test_seed_config():
    assert SeedConfig.from_any(7).detect_every == 7
    assert SeedConfig.from_any({"detect_every": 3, "motion": True}).motion
    cfg = SeedConfig()
    assert SeedConfig.from_any(cfg) is cfg
    with pytest.raises(ValueError):
        SeedConfig(detect_every=0)
    with pytest.raises(ValueError):
        SeedConfig(box_points="some")
    with pytest.raises(TypeError):
        SeedConfig.from_any("30")
    assert SeedConfig().to_dict()["box_scale"] == pytest.approx(1.19)


def test_box_from_points():
    pts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0], [100.0, 100.0]])
    scores = np.array([0.9, 0.9, 0.9, 0.9, 0.05])
    cfg = SeedConfig(box_scale=1.0, box_points="all")
    np.testing.assert_allclose(box_from_points(pts, scores, cfg), [0, 0, 100, 100])
    # "conf" leaves out the unsure point; padding is 10% of the longest side.
    cfg = SeedConfig(box_scale=1.2, box_points="conf", conf_threshold=0.2, min_points=4)
    np.testing.assert_allclose(box_from_points(pts, scores, cfg), [-1, -1, 11, 5])
    # Too few confident points: all points are used.
    cfg = SeedConfig(box_scale=1.0, box_points="conf", min_points=5)
    np.testing.assert_allclose(box_from_points(pts, scores, cfg), [0, 0, 100, 100])
    assert box_from_points(np.full((3, 2), np.nan), None, cfg) is None
    assert box_from_points(np.array([[1.0, 1.0], [np.nan, np.nan]]), None, cfg) is None


def test_stack_pads_instance_axis():
    a = Outputs(
        pred_keypoints=torch.zeros(1, 1, N_NODES, 2),
        pred_bbox_classes=torch.zeros(1, 1, dtype=torch.long),
        frame_indices=None,
    )
    b = Outputs(
        pred_keypoints=torch.ones(1, 3, N_NODES, 2),
        pred_bbox_classes=torch.ones(1, 3, dtype=torch.long),
    )
    out = _stack([a, b])
    assert out.pred_keypoints.shape == (2, 3, N_NODES, 2)
    assert torch.isnan(out.pred_keypoints[0, 1:]).all()
    assert out.pred_bbox_classes[0].tolist() == [0, -1, -1]
    assert out.pred_bbox_classes[1].tolist() == [1, 1, 1]


# --------------------------------------------------------------------------- #
# Scheduling, with a scripted two-stage forward
# --------------------------------------------------------------------------- #
class _Stage:
    """Stage-1 stand-in: only ``_to_4d_tensor`` is used by the seeded layer."""

    def __init__(self, name):
        self.name = name

    @staticmethod
    def _to_4d_tensor(image):
        return torch.as_tensor(image).permute(0, 3, 1, 2)


class _World:
    """Animals per frame: ``{frame: [(x, y) or None (hidden), ...]}``; size 200."""

    def __init__(self, positions, half=10.0, flipped=(), det_miss=()):
        self.positions = positions
        self.half = half
        self.flipped = set(flipped)  # frames whose poses come out head to tail
        self.det_miss = set(det_miss)  # (frame, animal) pairs the detector misses
        self.calls = []  # (stage name, frames)

    def animal(self, x, y, score=0.9):
        offs = np.linspace(-self.half, self.half, N_NODES)
        pts = np.stack([x + offs, y + offs / 2], -1)
        return pts, np.full(N_NODES, score)

    def predict(self, layer, image, frame_boxes=None, frame_box_scores=None, **_):
        frames = [int(f) for f in np.asarray(image)[:, 0, 0, 0]]
        name = layer.centroid_layer.name
        self.calls.append((name, frames))
        kp, pv, bx, bs = [], [], [], []
        for k, f in enumerate(frames):
            world = self.positions[f]
            rows = []
            if name == "detector":
                for a_i, p in enumerate(world):
                    if p is not None and (f, a_i) not in self.det_miss:
                        pts, sc = self.animal(*p)
                        h = 1.19 * 2 * self.half / 2
                        rows.append(
                            (pts, sc, [p[0] - h, p[1] - h, p[0] + h, p[1] + h], 0.95)
                        )
            else:
                for j in range(frame_boxes.shape[1]):
                    box = frame_boxes[k, j].numpy()
                    if not np.isfinite(box).all():
                        continue
                    c = (box[:2] + box[2:]) / 2
                    seen = [p for p in world if p is not None]
                    if seen:
                        p = min(seen, key=lambda q: np.hypot(q[0] - c[0], q[1] - c[1]))
                        near = np.hypot(p[0] - c[0], p[1] - c[1]) < 2 * self.half
                    else:
                        near = False
                    pts, sc = self.animal(
                        *(p if near else c), score=0.9 if near else 0.05
                    )
                    rows.append((pts, sc, box, float(frame_box_scores[k, j])))
            if f in self.flipped:
                rows = [(pts[::-1].copy(), sc, box, s) for pts, sc, box, s in rows]
            kp.append(rows)
        n = max(1, max(len(r) for r in kp))
        K = torch.full((len(frames), n, N_NODES, 2), float("nan"))
        V = torch.full((len(frames), n, N_NODES), float("nan"))
        B = torch.full((len(frames), n, 4), float("nan"))
        BS = torch.full((len(frames), n), float("nan"))
        for k, rows in enumerate(kp):
            for i, (pts, sc, box, s) in enumerate(rows):
                K[k, i] = torch.as_tensor(pts)
                V[k, i] = torch.as_tensor(sc)
                B[k, i] = torch.as_tensor(box, dtype=torch.float32)
                BS[k, i] = s
        return Outputs(
            pred_keypoints=K, pred_peak_values=V, pred_bboxes=B, pred_bbox_scores=BS
        )


def _layer(monkeypatch, world, **cfg):
    base = TopDownLayer.__new__(TopDownLayer)
    TopDownLayer.__init__(
        base,
        centroid_layer=_Stage("detector"),
        centered_instance_layer=None,
        crop_size=(32, 32),
    )
    monkeypatch.setattr(TopDownLayer, "predict", world.predict)
    return PoseSeededTopDownLayer(base, _Stage("seed"), SeedConfig(**cfg))


def _frames(idx):
    """Frames whose first pixel carries the frame number (read by the stand-in)."""
    img = np.zeros((len(idx), 200, 200, 1), dtype=np.float32)
    img[:, 0, 0, 0] = idx
    return img


def _run(layer, frames, batch, video=0):
    outs = []
    for s in range(0, len(frames), batch):
        f = np.asarray(frames[s : s + batch])
        outs.append(
            layer.predict(
                _frames(f), frame_indices=f, video_indices=np.full(len(f), video)
            )
        )
    return outs


def _detector_frames(world):
    return [f for name, fs in world.calls if name == "detector" for f in fs]


@pytest.mark.parametrize("batch", [1, 4])
def test_schedule(monkeypatch, batch):
    world = _World({f: [(50.0 + f, 60.0), (140.0, 120.0 - f)] for f in range(10)})
    layer = _layer(monkeypatch, world, detect_every=4)
    outs = _run(layer, list(range(10)), batch)
    assert _detector_frames(world) == [0, 4, 8]
    assert layer.seed_stats["frames"] == 10
    assert layer.seed_stats["detector_frames"] == 3
    kp = torch.cat([o.pred_keypoints for o in outs])
    assert kp.shape[0] == 10 and not torch.isnan(kp[:, :2]).any()
    # Seeded frames follow the moving animals.
    assert kp[9, :, N_NODES // 2, 0].sort().values[0] == pytest.approx(59.0)


def test_detect_every_one_is_detector_only(monkeypatch):
    world = _World({f: [(50.0, 60.0)] for f in range(5)})
    layer = _layer(monkeypatch, world, detect_every=1)
    _run(layer, list(range(5)), 2)
    assert _detector_frames(world) == [0, 1, 2, 3, 4]
    assert all(name == "detector" for name, _ in world.calls)


def test_lost_animal_triggers_detector(monkeypatch):
    # The animal vanishes at frame 3 (its seeded crop scores low) and a new one
    # appears at frame 3 elsewhere.
    pos = {f: [(50.0, 60.0)] if f < 3 else [(150.0, 150.0)] for f in range(8)}
    world = _World(pos)
    layer = _layer(monkeypatch, world, detect_every=100)
    outs = _run(layer, list(range(8)), 1)
    assert _detector_frames(world) == [0, 4]
    assert layer.seed_stats["detector_lost"] == 1
    kp = torch.cat([o.pred_keypoints for o in outs])
    assert kp[7, 0, N_NODES // 2, 0] == pytest.approx(150.0)


def test_converging_boxes_drop_one(monkeypatch):
    # Two animals; from frame 2 the second is hidden, so both seeded crops land
    # on the first: one is dropped and the detector runs on the next frame.
    pos = {f: [(50.0, 60.0), (58.0, 60.0) if f < 2 else None] for f in range(6)}
    world = _World(pos)
    layer = _layer(monkeypatch, world, detect_every=100, lost_score=0.0)
    _run(layer, list(range(6)), 1)
    assert layer.seed_stats["duplicates_dropped"] >= 1
    assert 3 in _detector_frames(world)


def test_streams_restart_on_jumps_and_videos(monkeypatch):
    world = _World({f: [(50.0, 60.0)] for f in range(40)})
    layer = _layer(monkeypatch, world, detect_every=100)
    _run(layer, [0, 1, 2, 20, 21, 22], 3)
    assert _detector_frames(world) == [0, 20]
    _run(layer, [23, 24], 2, video=1)  # another video: a new stream
    assert _detector_frames(world) == [0, 20, 23]
    layer.reset_streams()
    _run(layer, [24], 1, video=1)
    assert _detector_frames(world) == [0, 20, 23, 24]


def test_streams_in_one_batch_run_together(monkeypatch):
    # Two chunks of one video interleaved in each batch: one round per frame pair.
    world = _World({f: [(50.0, 60.0)] for f in range(110)})
    layer = _layer(monkeypatch, world, detect_every=100)
    for t in range(3):
        f = np.array([t, 100 + t])
        layer.predict(_frames(f), frame_indices=f, video_indices=np.zeros(2))
    assert world.calls[0] == ("detector", [0, 100])
    assert world.calls[1] == ("seed", [1, 101])
    assert world.calls[2] == ("seed", [2, 102])


def test_without_frame_indices_runs_plain(monkeypatch):
    world = _World({0: [(50.0, 60.0)]})
    layer = _layer(monkeypatch, world)
    seen = []
    monkeypatch.setattr(
        TopDownLayer,
        "predict",
        lambda self, image, **kw: seen.append(self.centroid_layer.name) or Outputs(),
    )
    layer.predict(_frames([0]))
    assert seen == ["detector"]


def test_outside_frame_is_lost():
    assert S._outside_fraction(np.array([-10.0, 0, 10, 10]), 100, 100) == pytest.approx(
        0.5
    )
    assert S._outside_fraction(np.array([0.0, 0, 10, 10]), 100, 100) == pytest.approx(
        0.0
    )


@pytest.mark.parametrize("every", [100, 2])
def test_flip_fix(monkeypatch, every):
    """A one-frame head-to-tail flip is turned back; without the fix it stays."""
    pos = {f: [(50.0 + 2 * f, 60.0)] for f in range(6)}
    world = _World(pos, flipped={3})
    layer = _layer(monkeypatch, world, detect_every=every, fix_flips=True)
    layer.flip_index = (0, N_NODES - 1)
    outs = _run(layer, list(range(6)), 1)
    kp = torch.cat([o.pred_keypoints for o in outs])
    head = kp[:, 0, 0, 0]
    assert (head[1:] > head[:-1]).all()  # the head stays on the same end
    assert layer.seed_stats["flips_undone"] == 1
    world = _World(pos, flipped={3})
    layer = _layer(monkeypatch, world, detect_every=every)
    kp = torch.cat([o.pred_keypoints for o in _run(layer, list(range(6)), 1)])
    assert kp[3, 0, 0, 0] > kp[3, 0, -1, 0]


@pytest.mark.parametrize("keep", [False, True])
def test_keep_unmatched_at_detector_frames(monkeypatch, keep):
    """A carried animal the detector misses on its frame stays with keep_unmatched."""
    pos = {f: [(50.0, 60.0), (140.0, 120.0)] for f in range(8)}
    world = _World(pos, det_miss={(4, 1)})
    layer = _layer(monkeypatch, world, detect_every=4, keep_unmatched=keep)
    outs = _run(layer, list(range(8)), 1)
    n = [int((~torch.isnan(o.pred_keypoints[0, :, 0, 0])).sum()) for o in outs]
    assert n[4] == (2 if keep else 1)
    # Without it the missed animal stays missing until the next scheduled detector run.
    assert n[5:] == ([2, 2, 2] if keep else [1, 1, 1])


def test_track_ids_follow_animals(monkeypatch):
    """emit_tracks: each animal keeps one id through seeded and detector frames."""
    pos = {f: [(50.0 + f, 60.0), (140.0, 120.0 - f)] for f in range(9)}
    world = _World(pos)
    layer = _layer(monkeypatch, world, detect_every=4, emit_tracks=True)
    _run(layer, list(range(9)), 2)
    log = layer.track_log
    assert sorted(log) == [(0, f) for f in range(9)]
    first = log[(0, 0)]
    assert sorted(first) == [0, 1]
    for f in range(9):
        # the animal on the left keeps its id; slot order may change at detector frames
        assert sorted(log[(0, f)]) == [0, 1]


class _DetStage(_Stage):
    """Stage 1 with a ``predict`` (the background detector calls it directly)."""

    def __init__(self, world):
        super().__init__("detector")
        self.world = world

    def predict(self, image):
        f = int(np.asarray(image)[0, 0, 0, 0])
        self.world.calls.append(("background", [f]))
        rows = []
        for a_i, p in enumerate(self.world.positions[f]):
            if p is not None and (f, a_i) not in self.world.det_miss:
                h = 1.19 * self.world.half
                rows.append(([p[0] - h, p[1] - h, p[0] + h, p[1] + h], 0.95))
        n = max(1, len(rows))
        B = torch.full((1, n, 4), float("nan"))
        BS = torch.full((1, n), float("nan"))
        C = torch.full((1, n, 2), float("nan"))
        for i, (box, s) in enumerate(rows):
            B[0, i] = torch.as_tensor(box)
            BS[0, i] = s
            C[0, i] = torch.as_tensor([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        return Outputs(
            pred_centroids=C,
            pred_centroid_values=BS.clone(),
            pred_bboxes=B,
            pred_bbox_scores=BS,
        )


class _InlineExecutor:
    """Runs a submitted call at once: its result is merged on the next frame."""

    def submit(self, fn, *a):
        from concurrent.futures import Future

        fut = Future()
        fut.set_result(fn(*a))
        return fut

    def shutdown(self, wait=True):
        pass


def _async_layer(monkeypatch, world, inline=True, **cfg):
    layer = _layer(monkeypatch, world, async_detect=True, **cfg)
    layer.centroid_layer = _DetStage(world)
    if inline:
        layer._executor = _InlineExecutor()
    return layer


def _background_frames(world):
    return [f for name, fs in world.calls if name == "background" for f in fs]


def test_async_detect_off_the_critical_path(monkeypatch):
    """Due detector runs go to the background; every frame after the first is seeded."""
    world = _World({f: [(50.0 + f, 60.0), (140.0, 120.0 - f)] for f in range(10)})
    layer = _async_layer(monkeypatch, world, detect_every=4)
    outs = _run(layer, list(range(10)), 1)
    assert _detector_frames(world) == [0]  # only the first frame waits for it
    assert _background_frames(world) == [4, 8]
    assert layer.seed_stats["detector_async"] == 2
    assert layer.seed_stats["async_reanchored"] == 4  # both animals, twice
    assert layer.seed_stats["async_added"] == 0
    kp = torch.cat([o.pred_keypoints for o in outs])
    assert not torch.isnan(kp[:, :2]).any()
    assert kp[9, :, N_NODES // 2, 0].sort().values[0] == pytest.approx(59.0)


def test_async_detect_adds_a_new_animal(monkeypatch):
    """An animal that appears is picked up from the background run one frame later."""
    pos = {f: [(50.0, 60.0)] + ([(140.0, 120.0)] if f >= 3 else []) for f in range(9)}
    world = _World(pos)
    layer = _async_layer(monkeypatch, world, detect_every=4)
    outs = _run(layer, list(range(9)), 1)
    n = [int((~torch.isnan(o.pred_keypoints[0, :, 0, 0])).sum()) for o in outs]
    # asked on frame 4 (scheduled), merged before frame 5
    assert _background_frames(world) == [4, 8]
    assert n == [1, 1, 1, 1, 1, 2, 2, 2, 2]
    assert layer.seed_stats["async_added"] == 1


def test_async_detect_keeps_missed_animals(monkeypatch):
    """A carried animal the background detector misses is kept."""
    pos = {f: [(50.0, 60.0), (140.0, 120.0)] for f in range(8)}
    world = _World(pos, det_miss={(4, 1)})
    layer = _async_layer(monkeypatch, world, detect_every=4)
    outs = _run(layer, list(range(8)), 1)
    n = [int((~torch.isnan(o.pred_keypoints[0, :, 0, 0])).sum()) for o in outs]
    assert n == [2] * 8


def test_async_detect_all_lost_runs_in_line(monkeypatch):
    """With nothing left to carry the stream waits for the detector."""
    pos = {f: [(50.0, 60.0)] if f < 3 or f >= 5 else [None] for f in range(7)}
    world = _World(pos)
    layer = _async_layer(monkeypatch, world, detect_every=30)
    _run(layer, list(range(7)), 1)
    # frame 3: the animal is gone, so it is lost and frame 4 has nothing to carry
    assert _detector_frames(world)[:2] == [0, 4]
    assert _background_frames(world) == []


def test_async_detect_with_a_thread(monkeypatch):
    """The real background thread: same animals found, whatever the timing."""
    world = _World({f: [(50.0 + f, 60.0), (140.0, 120.0 - f)] for f in range(12)})
    layer = _async_layer(monkeypatch, world, inline=False, detect_every=3)
    outs = _run(layer, list(range(12)), 2)
    kp = torch.cat([o.pred_keypoints for o in outs])
    assert kp.shape[0] == 12 and not torch.isnan(kp[:, :2]).any()
    assert layer.seed_stats["detector_async"] >= 1
    layer.reset_streams()  # shuts the thread down
    assert layer._executor is None
