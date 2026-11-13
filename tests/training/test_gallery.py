"""W&B prediction gallery: panel selection, drawing and callback safety."""

import os
import random

import numpy as np
import pytest
import torch
import wandb

from sleap_nn.training import gallery
from sleap_nn.training.callbacks import WandBGalleryCallback


@pytest.fixture(autouse=True)
def wandb_offline():
    """W&B offline (no network) and no run left open."""
    mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"
    yield
    if wandb.run is not None:
        wandb.finish()
    if mode is None:
        os.environ.pop("WANDB_MODE", None)
    else:
        os.environ["WANDB_MODE"] = mode


# --------------------------------------------------------------------------- #
# Panel selection
# --------------------------------------------------------------------------- #
def _groups(sizes):
    return np.concatenate([np.full(n, g) for g, n in enumerate(sizes)])


def test_select_panel_is_deterministic_and_stratified():
    sizes = [50, 7, 300, 1, 20]
    groups = _groups(sizes)
    names = [f"set{i}" for i in range(len(sizes))]
    hard = np.random.default_rng(0).random(len(groups))
    a = gallery.select_panel(groups, names, 16, hard)
    b = gallery.select_panel(groups, names, 16, hard)
    assert a == b and len(a) == 16
    per = np.bincount(groups[[i for i, _ in a]], minlength=len(sizes))
    # Equal shares (3 or 4 each); the 1-sample set gives its spare slots away.
    assert per[3] == 1 and all(p >= 3 for p in per[[0, 1, 2, 4]])
    # Every set with two or more slots shows its hardest sample, after the typical.
    for g in (0, 1, 2, 4):
        mine = [(i, r) for i, r in a if groups[i] == g]
        roles = [r for _, r in mine]
        assert roles[-1] == "hard" and roles[0] == "typical"
        members = np.flatnonzero(groups == g)
        assert mine[-1][0] == members[np.argmax(hard[members])]
    # Display order: by set.
    assert list(groups[[i for i, _ in a]]) == sorted(groups[[i for i, _ in a]])


def test_select_panel_picks_do_not_depend_on_other_sets():
    groups = _groups([40, 40, 40])
    names = ["a", "b", "c"]
    full = gallery.select_panel(groups, names, 9)
    alone = gallery.select_panel(groups[groups == 1] * 0, ["b"], 3)
    offset = np.flatnonzero(groups == 1)[0]
    assert [i for i, _ in full if groups[i] == 1] == [i + offset for i, _ in alone]


def test_select_panel_small_and_single_sets():
    # Fewer samples than the panel: all of them, in order.
    small = gallery.select_panel(_groups([4, 6]), ["a", "b"], 64)
    assert sorted(i for i, _ in small) == list(range(10))
    # One set: the whole panel comes from it, a quarter of it the hardest.
    groups = np.zeros(500, dtype=int)
    hard = np.arange(500, dtype=float)
    one = gallery.select_panel(groups, ["only"], 64, hard)
    assert len(one) == 64 and len({i for i, _ in one}) == 64
    assert [i for i, r in one if r == "hard"] == list(range(499, 483, -1))
    # More sets than slots: one sample each from sets spread over the list.
    many = gallery.select_panel(_groups([3] * 100), [str(i) for i in range(100)], 10)
    assert len(many) == 10 and len({i // 3 for i, _ in many}) == 10
    assert gallery.select_panel(np.zeros(0, dtype=int), ["a"], 8) == []


def test_hardness_scores():
    boxes = [
        np.array([[0, 0, 10, 10]]),
        np.array([[0, 0, 50, 50], [0, 0, 4, 4]]),
        np.array([[0, 0, 50, 50], [0, 0, 40, 40]]),
        np.zeros((0, 4)),
    ]
    h = gallery.box_hardness(boxes)
    # Most boxes first, then the smallest box; empty frames last.
    assert list(np.argsort(-h)) == [1, 2, 0, 3]
    inst = gallery.instance_hardness(np.array([[0, 0, 10, 10], [0, 0, 5, 5]]))
    assert inst[1] > inst[0]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def test_draw_keypoints_draws_only_the_given_nodes():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    gt = np.array([[10, 10], [50, 50], [np.nan, np.nan], [30, 5]], dtype=float)
    pred = gt + 2.0
    out = gallery.draw_keypoints(img, [0, 1], [(0, 1), (1, 3)], gt=gt, pred=pred)
    assert out is not img and img.sum() == 0 and out.shape == img.shape
    assert out[10, 10:15].any() and out[50, 50:55].any()
    assert not out[0:3, 27:34].any()  # node 3 is not one of the drawn nodes
    # Training targets (no predictions): ground truth and its edges.
    tgt = gallery.draw_keypoints(img, [0, 1], [(0, 1)], gt=gt)
    assert tgt[30, 30].any()  # on the 0-1 edge
    assert gallery.keypoint_error(gt, pred, [0, 1, 2]) == pytest.approx(2 * 2**0.5)


def test_bbox_wandb_image_boxes_and_scores():
    img = np.zeros((480, 1280, 3), dtype=np.uint8)
    gt = np.array([[100, 100, 300, 200], [np.nan] * 4])
    pred = np.array([[110, 100, 300, 210]])
    im = gallery.bbox_wandb_image(
        img, gt, "cap", ["body"], pred_boxes=pred, pred_scores=np.array([0.7])
    )
    assert (im._height, im._width) == (240, 640)  # long side 640
    boxes = im._boxes
    gt_box = boxes["ground_truth"]._val
    pr_box = boxes["predictions"]._val
    assert len(gt_box) == 1 and len(pr_box) == 1
    assert gt_box[0]["position"] == {"minX": 50, "maxX": 150, "minY": 50, "maxY": 100}
    assert pr_box[0]["scores"] == {"score": pytest.approx(0.7)}
    assert im._caption == "cap" and str(im._path).endswith(".jpg")


def test_bbox_wandb_image_cached_jpeg_and_top_predictions(tmp_path):
    img = np.full((480, 1280, 3), 128, dtype=np.uint8)
    path = str(tmp_path / "f.jpg")
    scale, hw = gallery.encode_display_jpeg(img, path, 640)
    assert scale == 0.5 and hw == (240, 640)
    pred = np.tile(np.array([[0, 0, 10, 10]], dtype=float), (80, 1))
    scores = np.linspace(0.05, 0.9, 80)
    im = gallery.bbox_wandb_image(
        None,
        np.zeros((0, 4)),
        "cap",
        ["body"],
        pred_boxes=pred,
        pred_scores=scores,
        cached=(path, scale),
        max_preds=50,
    )
    kept = [b["scores"]["score"] for b in im._boxes["predictions"]._val]
    assert len(kept) == 50 and min(kept) == pytest.approx(scores[-50])
    assert (im._height, im._width) == (240, 640)


# --------------------------------------------------------------------------- #
# Callback safety
# --------------------------------------------------------------------------- #
class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Dropout()
        self.b.eval()  # a submodule deliberately in eval mode

    @property
    def device(self):
        return torch.device("cpu")


def test_guarded_restores_rng_and_modes_and_swallows_errors():
    cb = WandBGalleryCallback("keypoints", ["v"], ["t"])
    module = _Module()
    module.train()
    module.b.eval()
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    expect = (random.random(), np.random.rand(), torch.rand(1).item())
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    def noisy(trainer, pl_module):
        random.random(), np.random.rand(), torch.rand(5)
        pl_module.eval()

    cb._guarded(noisy, None, module)
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expect
    assert module.training and module.a.training and not module.b.training
    assert cb.enabled

    def boom(trainer, pl_module):
        raise RuntimeError("drawing failed")

    cb._guarded(boom, None, module)  # does not raise
    assert not cb.enabled


def test_log_frequency_cap():
    class T:
        max_epochs = 200
        check_val_every_n_epoch = 1
        val_check_interval = 1.0

    cb = WandBGalleryCallback("bbox", ["v"], ["t"], max_logs=50)
    logged = [c for c in range(200) if cb._should_log(T, c)]
    assert len(logged) <= 51 and logged[0] == 0 and logged[-1] == 199
    cb = WandBGalleryCallback("bbox", ["v"], ["t"], max_logs=50)
    T.max_epochs = 10
    assert [c for c in range(10) if cb._should_log(T, c)] == list(range(10))


def test_time_budget(monkeypatch):
    import sleap_nn.training.callbacks as cbmod

    now = [0.0]
    monkeypatch.setattr(cbmod.time, "perf_counter", lambda: now[0])
    cb = WandBGalleryCallback("bbox", ["v"], ["t"], max_fraction=0.01)
    cb._planned, cb._stride, cb._t0 = 10, 1, 0.0
    cb.event_seconds, cb.panel_seconds = [3.0], 2.0  # first event: 1 s + 2 s panel
    now[0] = 50.0  # 1% of 50 s = 0.5 s < 1 s already spent + 1 s next
    assert cb._over_budget(1)
    now[0] = 250.0  # 2.5 s budget >= 1 + 1
    assert not cb._over_budget(1)
    assert not cb._over_budget(9)  # the last planned validation always logs
