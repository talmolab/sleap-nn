"""Multi-file validation: per-file val loss and per-file epoch-end evaluation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.training.callbacks import (
    EpochEndEvaluationCallback,
    ValFile,
    select_nodes,
)
from sleap_nn.training.lightning_modules import LightningModel
from sleap_nn.training.losses import compute_weighted_mse_loss
from sleap_nn.training.model_trainer import ModelTrainer

NODES = ["a/x", "a/y", "b/x", "b/y", "b/z"]


def test_select_nodes():
    assert select_nodes(NODES, None) == NODES
    assert select_nodes(NODES, "b/") == ["b/x", "b/y", "b/z"]
    # Lists come back in model order.
    assert select_nodes(NODES, ["b/z", "a/x"]) == ["a/x", "b/z"]
    with pytest.raises(ValueError, match="prefix"):
        select_nodes(NODES, "c/")
    with pytest.raises(ValueError, match="not model nodes"):
        select_nodes(NODES, ["a/x", "nope"])


class _LossStub:
    """The per-file val-loss methods of `LightningModel` on a bare object."""

    _get_node_weights = LightningModel._get_node_weights
    _accumulate_per_file_val_loss = LightningModel._accumulate_per_file_val_loss
    _log_per_file_val_loss = LightningModel._log_per_file_val_loss

    def __init__(self, names):
        self.val_file_names = names
        self._val_file_num = self._val_file_den = None
        self._trainer = None
        self.device = torch.device("cpu")
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


def test_per_file_val_loss_is_each_files_weighted_mse():
    g = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(3):
        b = 4
        batches.append(
            {
                "pred": torch.rand(b, 5, 6, 6, generator=g),
                "gt": torch.rand(b, 5, 6, 6, generator=g),
                "node_weights": (torch.rand(b, 1, 5, generator=g) > 0.3).float(),
                "labels_idx": torch.randint(0, 2, (b,), generator=g),
            }
        )
    stub = _LossStub(["a", "b"])
    for batch in batches:
        stub._accumulate_per_file_val_loss(batch["pred"], batch["gt"], batch)
    stub._log_per_file_val_loss()

    cat = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    expected = []
    for f, name in enumerate(["a", "b"]):
        sel = cat["labels_idx"] == f
        want = compute_weighted_mse_loss(
            cat["pred"][sel], cat["gt"][sel], cat["node_weights"][sel].squeeze(1)
        )
        assert stub.logged[f"val/{name}/loss"] == pytest.approx(float(want), rel=1e-6)
        expected.append(float(want))
    assert stub.logged["val/macro/loss"] == pytest.approx(np.mean(expected))
    # The sums reset for the next epoch.
    assert stub._val_file_num is None


def test_per_file_val_loss_noop_without_labels_idx_or_names():
    pred = torch.rand(2, 5, 4, 4)
    stub = _LossStub(["a", "b"])
    stub._accumulate_per_file_val_loss(pred, pred, {})
    stub._log_per_file_val_loss()
    assert stub.logged == {}
    single = _LossStub(None)
    single._accumulate_per_file_val_loss(pred, pred, {"labels_idx": torch.zeros(2)})
    single._log_per_file_val_loss()
    assert single.logged == {}


def _entry(f, video_idx, crop_id, gt, pred):
    common = {"labels_idx": f, "video_idx": video_idx, "frame_idx": 0}
    common["crop_id"] = crop_id
    return (
        {**common, "gt_instances": gt[None], "num_instances": 1},
        {
            **common,
            "pred_peaks": pred[None],
            "pred_scores": np.ones((1, len(pred))),
        },
    )


@pytest.fixture
def two_files():
    """File `a` labels a/* (1 video), file `b` labels b/* (2 videos)."""
    skeleton = sio.Skeleton(nodes=list(NODES))
    videos_a = [sio.Video(filename="a.mp4")]
    videos_b = [sio.Video(filename="b0.mp4"), sio.Video(filename="b1.mp4")]
    files = [
        ValFile("a", videos_a, ["a/x", "a/y"]),
        ValFile("b", videos_b, ["b/x", "b/y", "b/z"]),
    ]
    rng = np.random.default_rng(0)
    gts, preds = [], []
    for crop in range(6):
        f = crop % 2
        pts = rng.uniform(20, 80, (5, 2))
        gt = pts.copy()
        gt[[2, 3, 4] if f == 0 else [0, 1]] = np.nan  # Other file's nodes.
        # File `a`: exact; file `b`: every point 3 px off. Every node is predicted.
        pred = pts + (0.0 if f == 0 else np.array([3.0, 0.0]))
        g, p = _entry(
            f, video_idx=(crop // 2) % 2 if f else 0, crop_id=crop, gt=gt, pred=pred
        )
        gts.append(g)
        preds.append(p)
    cb = EpochEndEvaluationCallback(
        skeleton=skeleton, videos=videos_a, val_files=files, oks_scale=100.0
    )
    return cb, gts, preds


def test_callback_scores_each_file_on_its_own_nodes_and_videos(two_files):
    cb, gts, preds = two_files
    # Pooled labels hold every file's videos; entries map to their own file's.
    assert len(cb.videos) == 3
    labels = cb._build_gt_labels(gts, sio, np)
    assert [lf.video.filename for lf in labels.labeled_frames] == [
        "a.mp4",
        "b0.mp4",
        "a.mp4",
        "b1.mp4",
        "a.mp4",
        "b0.mp4",
    ]

    per_file = cb._evaluate_files(gts, preds, epoch=0)
    a, b = per_file["a"], per_file["b"]
    assert a["mOKS"]["mOKS"] == pytest.approx(1.0)
    assert a["distance_metrics"]["avg"] == pytest.approx(0.0, abs=1e-6)
    assert b["distance_metrics"]["avg"] == pytest.approx(3.0, abs=1e-6)
    # Each file is scored on its own nodes only: the other file's (unlabeled)
    # nodes are not counted as false-positive visible predictions.
    assert a["visibility_metrics"]["precision"] == pytest.approx(1.0)
    assert b["visibility_metrics"]["precision"] == pytest.approx(1.0)
    pooled = cb._evaluate(gts, preds, epoch=0)
    assert pooled["distance_metrics"]["avg"] == pytest.approx(1.8, abs=1e-6)


def test_callback_logs_per_file_and_macro_metrics(two_files):
    cb, gts, preds = two_files
    per_file = cb._evaluate_files(gts, preds, epoch=0)
    trainer = MagicMock()
    trainer.callback_metrics = {}
    trainer.loggers = []
    cb._log_file_metrics(trainer, per_file, epoch=0)
    m = {k: float(v) for k, v in trainer.callback_metrics.items()}
    assert m["eval/b/distance/avg"] == pytest.approx(3.0, abs=1e-6)
    assert m["eval/macro/distance/avg"] == pytest.approx(1.5, abs=1e-6)
    assert m["eval/macro/mOKS"] == pytest.approx(
        (m["eval/a/mOKS"] + m["eval/b/mOKS"]) / 2
    )
    assert "eval/val/mOKS" not in m  # The pooled keys are logged separately.

    # A file with no entries gets the worst values and is left out of the mean.
    per_file["b"] = None
    trainer.callback_metrics = {}
    cb._log_file_metrics(trainer, per_file, epoch=0)
    m = {k: float(v) for k, v in trainer.callback_metrics.items()}
    assert m["eval/b/mOKS"] == 0.0
    assert m["eval/macro/mOKS"] == pytest.approx(m["eval/a/mOKS"])


def test_callback_single_file_unchanged():
    skeleton = sio.Skeleton(nodes=list(NODES))
    videos = [sio.Video(filename="a.mp4")]
    cb = EpochEndEvaluationCallback(
        skeleton=skeleton, videos=videos, val_files=[ValFile("a", videos, NODES)]
    )
    assert cb.val_files is None and cb.videos is videos


def _trainer_stub(config, val_labels, warp=True):
    t = SimpleNamespace(
        config=OmegaConf.create(config),
        val_labels=val_labels,
        model_type="centered_instance",
        skeletons=[sio.Skeleton(nodes=list(NODES))],
    )
    for name in ("_val_file_names", "_val_files", "_check_per_file_monitor"):
        setattr(t, name, getattr(ModelTrainer, name).__get__(t))
    t._RESERVED_VAL_NAMES = ModelTrainer._RESERVED_VAL_NAMES
    t._uses_warp_pipeline = lambda: warp
    return t


def _file(nodes):
    sk = sio.Skeleton(nodes=nodes)
    return sio.Labels(skeletons=[sk], videos=[sio.Video(filename="v.mp4")])


def test_val_file_names_and_nodes():
    a, b = _file(["a/x", "a/y"]), _file(["b/x", "b/y", "b/z"])
    cfg = {
        "data_config": {
            "val_labels_path": ["/d/hcm-2025.val.pkg.slp", "/d/other.slp"],
            "val_names": None,
        },
        "trainer_config": {"eval": {"enabled": True, "file_nodes": None}},
    }
    t = _trainer_stub(cfg, [a, b])
    assert t._val_file_names() == ["hcm-2025.val", "other"]
    files = t._val_files()
    assert [f.node_names for f in files] == [["a/x", "a/y"], ["b/x", "b/y", "b/z"]]
    assert files[1].videos == list(b.videos)

    cfg["trainer_config"]["eval"]["file_nodes"] = {"other": ["b/z", "b/x"]}
    t = _trainer_stub(cfg, [a, b])
    assert t._val_files()[1].node_names == ["b/x", "b/z"]

    cfg["trainer_config"]["eval"]["file_nodes"] = {"other": "a/"}
    with pytest.raises(ValueError, match="other"):
        _trainer_stub(cfg, [a, b])._val_files()
    cfg["trainer_config"]["eval"]["file_nodes"] = {"nope": "b/"}
    with pytest.raises(ValueError, match="unknown validation files"):
        _trainer_stub(cfg, [a, b])._val_files()

    # Repeated or reserved stems are suffixed; one file has no names.
    cfg = {"data_config": {"val_labels_path": ["/x/val.slp", "/y/val.slp"]}}
    assert _trainer_stub(cfg, [a, b])._val_file_names() == ["val-0", "val-1"]
    cfg = {"data_config": {"val_labels_path": ["/x/m.slp", "/y/m.slp"]}}
    assert _trainer_stub(cfg, [a, b])._val_file_names() == ["m-0", "m-1"]
    assert _trainer_stub(cfg, [a])._val_file_names() is None
    # Split from the training files: names come from train_labels_path.
    cfg = {"data_config": {"train_labels_path": ["/t/p.slp", "/t/q.pkg.slp"]}}
    assert _trainer_stub(cfg, [a, b])._val_file_names() == ["p", "q"]

    cfg = {"data_config": {"val_labels_path": ["/x/a.slp", "/y/b.slp"]}}
    cfg["data_config"]["val_names"] = ["one", "two"]
    assert _trainer_stub(cfg, [a, b])._val_file_names() == ["one", "two"]
    for bad in (["one"], ["x", "x"], ["macro", "two"], ["a/b", "c"]):
        cfg["data_config"]["val_names"] = bad
        with pytest.raises(ValueError, match="val_names"):
            _trainer_stub(cfg, [a, b])._val_file_names()


def test_per_file_monitor_check():
    a, b = _file(["a/x"]), _file(["b/x"])
    cfg = {
        "data_config": {"val_names": ["a", "b"]},
        "trainer_config": {"eval": {"enabled": False}},
    }
    t = _trainer_stub(cfg, [a, b])
    t._check_per_file_monitor("val/macro/loss")
    t._check_per_file_monitor("val/a/loss")
    t._check_per_file_monitor("val/loss")
    t._check_per_file_monitor("eval/val/mOKS")
    with pytest.raises(ValueError, match="eval.enabled"):
        t._check_per_file_monitor("eval/macro/oks_voc_mAP")
    with pytest.raises(ValueError, match="warp"):
        _trainer_stub(cfg, [a, b], warp=False)._check_per_file_monitor("val/b/loss")
    with pytest.raises(ValueError, match="several validation files"):
        _trainer_stub(cfg, [a])._check_per_file_monitor("val/macro/loss")


# --- Multi-skeleton heads: eval without the file system, per-node losses, viz,
# --- post-training eval.


def _unreadable_match(self, other, *args, **kwargs):
    """`Labels.match` on embedded videos whose originals sit on an unreadable mount."""
    raise PermissionError(13, "Permission denied", "/root/vast/x/0021.jpg")


def test_match_labels_videos_pairs_shared_videos_without_the_file_system(
    monkeypatch,
):
    from sleap_nn.evaluation import match_labels_videos

    monkeypatch.setattr(sio.Labels, "match", _unreadable_match)
    videos = [sio.Video(filename="a.mp4"), sio.Video(filename="b.mp4")]
    gt = sio.Labels(videos=list(videos))
    pr = sio.Labels(videos=[videos[1]])
    assert match_labels_videos(gt, pr).video_map == {videos[1]: videos[1]}


def test_match_labels_videos_falls_back_to_index_pairing(monkeypatch):
    from sleap_nn.evaluation import match_labels_videos

    monkeypatch.setattr(sio.Labels, "match", _unreadable_match)
    gt = sio.Labels(videos=[sio.Video("/d/x.pkg.slp"), sio.Video("/d/y.pkg.slp")])
    # `sleap-nn predict` keeps the input's video order (other directory here).
    pr = sio.Labels(videos=[sio.Video("/e/x.pkg.slp"), sio.Video("/e/y.pkg.slp")])
    vmap = match_labels_videos(gt, pr).video_map
    assert [(p.filename, g.filename) for p, g in vmap.items()] == [
        ("/e/x.pkg.slp", "/d/x.pkg.slp"),
        ("/e/y.pkg.slp", "/d/y.pkg.slp"),
    ]
    # Different files: index pairing would be a guess, so the error stands.
    pr = sio.Labels(videos=[sio.Video("/e/y.pkg.slp"), sio.Video("/e/x.pkg.slp")])
    with pytest.raises(PermissionError):
        match_labels_videos(gt, pr)


def test_callback_scores_files_whose_originals_are_unreadable(two_files, monkeypatch):
    """The epoch-end eval scored 0 for every file once `Labels.match` raised."""
    monkeypatch.setattr(sio.Labels, "match", _unreadable_match)
    cb, gts, preds = two_files
    per_file = cb._evaluate_files(gts, preds, epoch=0)
    assert per_file["a"]["mOKS"]["mOKS"] == pytest.approx(1.0)
    assert per_file["b"]["distance_metrics"]["avg"] == pytest.approx(3.0, abs=1e-6)
    assert cb._evaluate(gts, preds, epoch=0) is not None


class _NodeLossStub:
    """The per-node train-loss methods of `LightningModel` on a bare object."""

    _get_node_weights = LightningModel._get_node_weights
    _accumulate_node_loss = LightningModel._accumulate_node_loss

    def __init__(self):
        self._epoch_node_loss_sum = None
        self._epoch_node_loss_count = 0


def test_per_node_loss_averages_only_samples_that_label_the_node():
    g = torch.Generator().manual_seed(0)
    stub = _NodeLossStub()
    batches = []
    for _ in range(3):
        mse = torch.rand(4, 5, 3, 3, generator=g)
        # Samples of file `a` label a/* only, file `b` labels b/* only.
        f = torch.randint(0, 2, (4,), generator=g)
        w = torch.stack(
            [torch.tensor([1.0, 1, 0, 0, 0] if x == 0 else [0, 0, 1, 1, 1]) for x in f]
        )
        batch = {"node_weights": w[:, None]}
        stub._accumulate_node_loss(mse, batch)
        batches.append((mse.mean(dim=(2, 3)), w))
    per_sample = torch.cat([b[0] for b in batches])
    w = torch.cat([b[1] for b in batches])
    got = stub._epoch_node_loss_sum / stub._epoch_node_loss_count
    for k in range(5):
        want = per_sample[w[:, k] > 0, k].mean()
        assert float(got[k]) == pytest.approx(float(want), rel=1e-6)
    # Without node weights every sample counts (the previous behavior).
    plain = _NodeLossStub()
    mse = torch.rand(4, 5, 3, 3, generator=g)
    plain._accumulate_node_loss(mse, {})
    want = torch.sum(mse, dim=(0, 2, 3)) / (4 * 3 * 3)
    got = plain._epoch_node_loss_sum / plain._epoch_node_loss_count
    assert torch.allclose(got, want)


def _viz_data(k):
    from sleap_nn.training.utils import VisualizationData

    rng = np.random.default_rng(k)
    return VisualizationData(
        image=rng.random((16, 16, 1)),
        pred_confmaps=rng.random((8, 8, k)),
        pred_peaks=rng.random((1, k, 2)),
        pred_peak_values=rng.random((1, k)),
        gt_instances=rng.random((1, k, 2)),
        node_names=[f"n{i}" for i in range(k)],
        output_scale=0.5,
    )


def test_restrict_viz_nodes():
    from sleap_nn.training.callbacks import _restrict_viz_nodes

    data = _viz_data(5)
    sub = _restrict_viz_nodes(data, [2, 4])
    assert sub.node_names == ["n2", "n4"]
    assert sub.pred_confmaps.shape == (8, 8, 2)
    np.testing.assert_array_equal(sub.pred_peaks, data.pred_peaks[:, [2, 4]])
    np.testing.assert_array_equal(sub.gt_instances, data.gt_instances[:, [2, 4]])
    np.testing.assert_array_equal(
        sub.pred_peak_values, data.pred_peak_values[:, [2, 4]]
    )


class _VizDataset(list):
    """Samples carrying `labels_idx`, with the warp dataset's `table.labels_idx`."""

    def __init__(self, labels_idx):
        super().__init__(
            {"labels_idx": int(f), "i": i} for i, f in enumerate(labels_idx)
        )
        self.table = SimpleNamespace(labels_idx=np.asarray(labels_idx))


def test_viz_draws_one_sample_per_val_file_on_its_own_nodes(tmp_path, monkeypatch):
    from sleap_nn.training.callbacks import UnifiedVizCallback

    lightning_model = SimpleNamespace(get_visualization_data=lambda s: _viz_data(5))
    cb = UnifiedVizCallback(
        model_trainer=SimpleNamespace(lightning_model=lightning_model),
        train_dataset=_VizDataset([1, 1]),
        val_dataset=_VizDataset([0, 1, 1, 0, 1]),
        model_type="centered_instance",
        save_local=True,
        local_save_dir=tmp_path,
        train_file_nodes=[[0, 1], [2, 3, 4]],
        val_file_nodes=[[0, 1], [2, 3, 4]],
        val_file_names=["a", "b"],
    )
    drawn = []
    monkeypatch.setattr(
        cb, "_save_local_viz", lambda data, prefix, epoch: drawn.append((prefix, data))
    )
    trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=3,
        strategy=SimpleNamespace(barrier=lambda: None),
    )
    cb.on_train_epoch_end(trainer, None)
    assert [p for p, _ in drawn] == ["train", "validation.a", "validation.b"]
    assert drawn[0][1].node_names == ["n2", "n3", "n4"]  # Train sample of file 1.
    assert drawn[1][1].node_names == ["n0", "n1"]
    assert drawn[2][1].node_names == ["n2", "n3", "n4"]


def test_viz_file_nodes():
    a, b = _file(["a/x", "a/y"]), _file(["b/x", "b/y", "b/z"])
    cfg = {
        "data_config": {"val_names": ["a", "b"]},
        "trainer_config": {"eval": {"file_nodes": {"b": ["b/z"]}}},
    }
    t = _trainer_stub(cfg, [a, b])
    t.train_labels = [b, a]
    t._viz_file_nodes = ModelTrainer._viz_file_nodes.__get__(t)
    assert t._viz_file_nodes() == ([[2, 3, 4], [0, 1]], [[0, 1], [4]], ["a", "b"])
    # One skeleton with every model node: nothing to restrict.
    full = _file(list(NODES))
    t = _trainer_stub({"data_config": {}}, [full])
    t.train_labels = [full]
    t._viz_file_nodes = ModelTrainer._viz_file_nodes.__get__(t)
    assert t._viz_file_nodes() == (None, None, None)


def test_post_train_file_nodes_and_gt_on_those_nodes():
    from sleap_nn.train import _labels_on_nodes, _post_train_file_nodes

    a, b = _file(["a/x", "a/y"]), _file(["b/z", "b/x", "b/y", "b/tail"])
    cfg = {
        "data_config": {"val_names": ["a", "b"]},
        "trainer_config": {"eval": {"file_nodes": {"b": ["b/x", "b/z"]}}},
    }
    t = _trainer_stub(cfg, [a, b])
    # Val files: their ValFile nodes (file_nodes applied), in model order.
    assert _post_train_file_nodes(t, "val.1", b) == ["b/x", "b/z"]
    # Other splits: the model nodes the file's skeleton holds.
    assert _post_train_file_nodes(t, "test.0", b) == ["b/x", "b/y", "b/z"]
    full = _file(list(NODES))
    assert _post_train_file_nodes(t, "test.0", full) is None
    with pytest.raises(ValueError, match="no model node"):
        _post_train_file_nodes(t, "test.0", _file(["c/x"]))

    sk = b.skeletons[0]
    inst = sio.Instance.from_numpy(
        np.array([[1.0, 1], [2, 2], [3, 3], [4, 4]]), skeleton=sk
    )
    b.append(sio.LabeledFrame(b.videos[0], 0, instances=[inst]))
    out = _labels_on_nodes(b, ["b/x", "b/y", "b/z"])
    assert out.skeletons[0].node_names == ["b/x", "b/y", "b/z"]
    np.testing.assert_array_equal(
        out[0].instances[0].numpy(), [[2.0, 2], [3, 3], [1, 1]]
    )
