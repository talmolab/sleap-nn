"""Frame in, pose out: ``open_stream`` / ``PoseStream`` against ``predict`` at batch 1."""

import numpy as np
import pytest
import sleap_io as sio
import torch

# Part stage on the same 2-node model, with a running session floor.
RULE = {
    "center": ["A"],
    "reference": [["A"], ["B"]],
    "scale": 1.0,
    "min_half_side": 4.0,
    "session_floor": 1.0,
    "session_min_boxes": 2,
}


def _ckpts(centroid, centered):
    return [str(centroid), str(centered)]


def _frames(labels_path):
    labels = sio.load_slp(str(labels_path))
    return [np.asarray(lf.image) for lf in labels.labeled_frames]


@pytest.fixture
def shifted_video(minimal_instance, tmp_path):
    """12 lossless frames (PNG) made from the fixture frame, shifted a little each time."""
    from PIL import Image

    base = _frames(minimal_instance)[0]
    base = base[..., 0] if base.ndim == 3 else base
    paths = []
    for i in range(12):
        img = np.roll(base, shift=(3 * i, -2 * i), axis=(0, 1))
        if i % 4 == 3:
            img = img[:, ::-1]  # mirrored now and then
        path = tmp_path / f"f{i:02d}.png"
        Image.fromarray(np.ascontiguousarray(img)).save(path)
        paths.append(path.as_posix())
    return sio.Video.from_filename(paths)


def _video_frames(video):
    return [np.asarray(video[i]) for i in range(len(video))]


def _assert_same(ref, got, n_nodes):
    frames = sorted(ref.labeled_frames, key=lambda f: f.frame_idx)
    assert len(frames) == len(got)
    seen = 0
    for lf, poses in zip(frames, got):
        assert lf.frame_idx == poses.frame_idx
        insts = lf.predicted_instances
        assert len(poses) == len(insts)
        if insts:
            a = np.stack([i.numpy() for i in insts]).astype(np.float32)
            np.testing.assert_array_equal(a, poses.points)  # bit-identical
            s = np.array([i.score for i in insts], dtype=np.float32)
            np.testing.assert_array_equal(s, poses.instance_scores)
            seen += len(insts)
        assert poses.points.shape[1:] == (n_nodes, 2)
    return seen


def test_stream_matches_predict_over_frames_with_floor_and_options(
    shifted_video,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """12 frames pushed one by one equal predict at batch 1 with part streaming.

    Session floor running, non-zero thresholds, max_instances and filter options
    passed through.
    """
    from sleap_nn.inference.filters import FilterConfig
    from sleap_nn.inference.run import open_stream, predict

    ckpts = _ckpts(
        minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt
    )
    kw = dict(
        model_paths=ckpts,
        device="cpu",
        peak_threshold=0.05,
        centroid_threshold=0.05,
        max_instances=2,
        filter_config=FilterConfig(min_peak_value=0.05, min_visible_nodes=1),
        part_model_paths=[ckpts[1]],
        part_crop_rule=RULE,
        part_peak_threshold=0.05,
        part_prefix="part/",
    )
    ref = predict(shifted_video, batch_size=1, part_streaming=True, **kw)
    with open_stream(**kw) as stream:
        got = [stream.push(f) for f in _video_frames(shifted_video)]
        floors = stream._predictor.part_stage.last_floors
        assert max(floors.values()) > RULE["min_half_side"]  # the floor enlarged crops
        assert stream.node_names == ["A", "B", "part/A", "part/B"]
    assert _assert_same(ref, got, 4) >= 12


@pytest.fixture(scope="module")
def detector_pair(tmp_path_factory, pytestconfig):
    """A tiny box detector + box-scaled centered-instance model (as test_bbox_topdown)."""
    from pathlib import Path

    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.fixtures.datasets import config as config_fixture
    from tests.test_bbox_topdown import _train_config

    tmp_path = tmp_path_factory.mktemp("stream_det")
    base = config_fixture.__wrapped__(
        Path(pytestconfig.rootdir) / "tests/assets/datasets"
    )
    dirs = {}
    for name, model_type, pre in (
        ("det", "bbox_centernet", {}),
        (
            "ci_bbox",
            "centered_instance",
            {"crop_mode": "bbox", "crop_size": None, "crop_bbox_aspect": "square"},
        ),
    ):
        cfg = _train_config(base, tmp_path, model_type, name, **pre)
        ModelTrainer.get_model_trainer_from_config(cfg).train()
        dirs[name] = (tmp_path / name).as_posix()
    return dirs


def test_stream_matches_predict_with_a_box_detector(
    shifted_video, detector_pair, minimal_instance_centered_instance_ckpt
):
    """Box-detector first stage (the shipped pipeline's shape), with a part stage."""
    from sleap_nn.inference.run import open_stream, predict

    kw = dict(
        model_paths=[detector_pair["det"], detector_pair["ci_bbox"]],
        device="cpu",
        peak_threshold=0.0,
        bbox_nms_iou=0.7,
        part_model_paths=[str(minimal_instance_centered_instance_ckpt)],
        part_crop_rule=RULE,
        part_peak_threshold=0.0,
        part_prefix="part/",
    )
    ref = predict(shifted_video, batch_size=1, part_streaming=True, **kw)
    with open_stream(**kw) as stream:
        got = [stream.push(f) for f in _video_frames(shifted_video)]
    _assert_same(ref, got, 4)
    assert any(len(p) and np.isfinite(p.boxes).all() for p in got)  # boxes reported


def _kw(centroid, centered, part=False):
    ckpts = _ckpts(centroid, centered)
    kw = dict(model_paths=ckpts, device="cpu", peak_threshold=0.0)
    if part:
        kw.update(
            part_model_paths=[ckpts[1]],
            part_crop_rule=RULE,
            part_peak_threshold=0.0,
            part_prefix="part/",
        )
    return kw


def test_stream_channels_layouts_and_bad_input(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Grey and colour frames, BGR (numpy and tensor), views of any layout, loud errors."""
    from sleap_nn.inference.run import open_stream

    kw = _kw(minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt)
    frame = _frames(minimal_instance)[0]
    grey = frame[..., 0] if frame.ndim == 3 else frame
    rgb = np.stack([grey, grey // 2, grey // 3], axis=-1).astype(np.uint8)
    with open_stream(**kw) as stream:
        assert stream.push(grey).points.shape[1:] == (2, 2)
    with open_stream(**kw) as rgb_s, open_stream(**kw, channel_order="bgr") as bgr_s:
        a = rgb_s.push(rgb)
        bgr = np.ascontiguousarray(rgb[..., ::-1])  # the same frame, BGR
        np.testing.assert_array_equal(a.points, bgr_s.push(bgr).points)
        tensor_bgr = torch.from_numpy(bgr)
        np.testing.assert_array_equal(a.points, bgr_s.push(tensor_bgr).points)
        flipped_twice = rgb[..., ::-1][..., ::-1]  # negative-stride view of RGB
        np.testing.assert_array_equal(a.points, rgb_s.push(flipped_twice).points)
        fortran = np.asfortranarray(rgb)  # not C-contiguous
        np.testing.assert_array_equal(a.points, rgb_s.push(fortran).points)
        strided = torch.from_numpy(rgb.copy()).permute(1, 0, 2).contiguous()
        np.testing.assert_array_equal(
            a.points, rgb_s.push(strided.permute(1, 0, 2)).points
        )
    with open_stream(**kw) as stream:
        with pytest.raises(TypeError, match="uint8"):
            stream.push(grey.astype(np.float32))
        with pytest.raises(TypeError):
            stream.push(grey.tolist())
        with pytest.raises(ValueError, match="C = 1 or 3"):
            stream.push(np.zeros((8, 8, 2), np.uint8))
        with pytest.raises(ValueError, match="C = 1 or 3"):
            stream.push(np.zeros((1, 8, 8, 1), np.uint8))
        stream.push(grey)
        with pytest.raises(ValueError, match="reset"):
            stream.push(grey[:-4])
        stream.reset()
        assert stream.push(grey[:-4]).frame_idx == 0
    with pytest.raises(ValueError, match="channel_order"):
        open_stream(**kw, channel_order="xyz")


def test_stream_refuses_models_without_keypoints(
    minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt
):
    """A lone centroid model or a lone centered-instance model raise at open."""
    from sleap_nn.inference.run import open_stream

    with pytest.raises(ValueError, match="end in keypoints"):
        open_stream([str(minimal_instance_centroid_ckpt)], device="cpu")
    with pytest.raises(ValueError, match="first stage"):
        open_stream([str(minimal_instance_centered_instance_ckpt)], device="cpu")


def test_stream_keep_reset_frame_idx_and_close(
    tmp_path,
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Zero animals; explicit frame_idx; keep + reset gives two videos; close twice."""
    from sleap_nn.inference.run import open_stream

    kw = _kw(
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
        part=True,
    )
    frame = _frames(minimal_instance)[0]
    with open_stream(**{**kw, "peak_threshold": 2.0}) as stream:  # no peak passes
        empty = stream.push(frame)
        assert len(empty) == 0
        assert empty.points.shape == (0, 4, 2) and empty.boxes.shape == (0, 4)
        assert empty.instance_scores.shape == (0,)
    stream = open_stream(**kw, keep=True)
    assert stream.push(frame, frame_idx=100).frame_idx == 100
    assert stream.push(frame).frame_idx == 101
    stream.reset()
    stream.reset()  # nothing pushed since: no empty video
    assert stream.push(frame).frame_idx == 0
    for o in stream._outputs:  # kept on the CPU, without images
        assert o.pred_keypoints.device.type == "cpu"
        assert o.processed_image is None
    stream.close()
    stream.close()  # twice: nothing happens
    with pytest.raises(RuntimeError, match="closed"):
        stream.push(frame)
    labels = stream.labels()
    assert len(labels.videos) == 2
    by_video = [
        [lf.frame_idx for lf in labels.labeled_frames if lf.video is v]
        for v in labels.videos
    ]
    assert by_video == [[100, 101], [0]]
    with pytest.raises(ValueError, match="one sio.Video per video"):
        stream.labels(sio.Video(filename="x", open_backend=False))
    stream.save(tmp_path / "s.slp")
    back = sio.load_slp(str(tmp_path / "s.slp"))
    assert len(back.labeled_frames) == 3 and len(back.videos) == 2
    with open_stream(**kw) as plain:
        with pytest.raises(ValueError, match="keep=True"):
            plain.labels()
    assert plain._predictor is None  # closed without keep: the models are let go


def _state(stream):
    part = stream._predictor.part_stage
    return (
        stream._next_idx,
        stream._shape,
        {v: (len(m), m.median()) for v, m in part.halves.items()},
        len(part.every),
        dict(part.last_floors),
        len(stream._outputs),
    )


def test_failed_push_leaves_the_stream_as_it_was(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
):
    """An exception inside push (here in the part model) changes no state."""
    from sleap_nn.inference.run import open_stream

    kw = _kw(
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
        part=True,
    )
    frame = _frames(minimal_instance)[0]
    other = np.roll(frame, 5, axis=0)
    ref = open_stream(**kw)
    want = [ref.push(f) for f in (frame, other, frame)]

    stream = open_stream(**kw, keep=True)
    stream.push(frame)
    part = stream._predictor.part_stage
    before = _state(stream)
    real = part.layer.predict

    def boom(*a, **k):
        raise RuntimeError("part model failed")

    monkeypatch.setattr(part.layer, "predict", boom)
    with pytest.raises(RuntimeError, match="part model failed"):
        stream.push(other)
    assert _state(stream) == before
    monkeypatch.setattr(part.layer, "predict", real)
    for w, f in zip(want[1:], (other, frame)):
        g = stream.push(f)
        assert w.frame_idx == g.frame_idx
        np.testing.assert_array_equal(w.points, g.points)


def test_warmup_reaches_the_pose_network(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
):
    """With frame_shape the pose network runs on 1-4 crops at open, part model or not.

    Without frame_shape nothing runs at open.
    """
    from sleap_nn.inference.layers.centered_instance import CenteredInstanceLayer
    from sleap_nn.inference.run import open_stream

    sizes = []
    real = CenteredInstanceLayer.predict

    def counted(self, crops, *a, **k):
        sizes.append(int(crops.shape[0]))
        return real(self, crops, *a, **k)

    monkeypatch.setattr(CenteredInstanceLayer, "predict", counted)
    kw = _kw(minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt)
    frame = _frames(minimal_instance)[0]
    open_stream(**kw)
    assert sizes == []
    # stage 1 finds nothing on empty frames here, so only the warm-up crops reach it
    stream = open_stream(**{**kw, "peak_threshold": 2.0}, frame_shape=frame.shape)
    assert {1, 2, 3, 4} <= set(sizes)
    assert stream._next_idx == 0 and stream._shape is None


def test_failure_after_the_part_stage_leaves_the_stream_as_it_was(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
):
    """A failure after the part stage returned (here in reading the outputs back,
    where a late CUDA error or a Ctrl-C lands) also leaves the floor as it was."""
    import sleap_nn.inference.stream as S
    from sleap_nn.inference.run import open_stream

    kw = _kw(
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
        part=True,
    )
    frame = _frames(minimal_instance)[0]
    other = np.roll(frame, 5, axis=0)
    ref = open_stream(**kw)
    want = [ref.push(f) for f in (frame, other, frame)]
    stream = open_stream(**kw, keep=True)
    stream.push(frame)
    before = _state(stream)
    real = S._frame_poses_from_outputs

    def boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(S, "_frame_poses_from_outputs", boom)
    with pytest.raises(KeyboardInterrupt):
        stream.push(other)
    assert _state(stream) == before
    monkeypatch.setattr(S, "_frame_poses_from_outputs", real)
    for w, f in zip(want[1:], (other, frame)):
        g = stream.push(f)
        assert w.frame_idx == g.frame_idx
        np.testing.assert_array_equal(w.points, g.points)


def test_tiled_single_instance_model_accepted(minimal_instance_single_instance_ckpt):
    """A tiled single-instance model gives keypoints: not refused at open."""
    from sleap_nn.inference.layers.tiled import TiledLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.stream import _check_keypoint_layer

    p = Predictor.from_model_paths(
        [str(minimal_instance_single_instance_ckpt)], device="cpu"
    )
    _check_keypoint_layer(TiledLayer(p.layer, tile_size=64, overlap=16))


def test_public_surface(
    minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt
):
    """Warm-up runs at open only; channel_order and keep are fixed; open_stream sits
    beside predict."""
    from sleap_nn.inference import open_stream, predict  # noqa: F401
    from sleap_nn.inference.stream import PoseStream

    assert not hasattr(PoseStream, "warmup")
    stream = open_stream(
        **_kw(minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt)
    )
    with pytest.raises(AttributeError):
        stream.keep = True
    with pytest.raises(AttributeError):
        stream.channel_order = "bgr"
    assert stream.keep is False and stream.channel_order == "rgb"


_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=_CUDA)])
def test_warmup_boxes_on_the_device_real_boxes_reach(
    shifted_video,
    detector_pair,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
    device,
):
    """The warm-up feeds the pose and part models as a real frame does.

    A real frame's boxes come from the first stage on its device (the card, for a
    CUDA detector) and the crop windows are computed there; with a part stage and
    ``frames_on_gpu`` the frame is uploaded once and the part model cuts its crops
    from that card copy. If the warm-up handed its dummy boxes or frame over on
    the host, the first real frame would load those GPU operations' code (CUDA
    lazy module loading) and be the slow one (seen with TensorRT engines and
    frames on the GPU: 28 ms instead of 11). On the CPU everything is on the
    host; the CUDA case fails when the warm-up's dummy boxes or the part model's
    dummy frame stay on the host.
    """
    from sleap_nn.inference.layers.bbox import BBoxStageLayer
    from sleap_nn.inference.layers.topdown import TopDownLayer
    from sleap_nn.inference.run import open_stream

    given, real, frames_in = [], [], []
    orig = BBoxStageLayer.predict
    orig_td = TopDownLayer.predict

    def spy(self, image, *a, boxes=None, **k):
        out = orig(self, image, *a, boxes=boxes, **k)
        if getattr(self, "use_given_boxes", False):
            given.append(torch.as_tensor(boxes).device.type)
        elif isinstance(out.pred_centroids, torch.Tensor):
            if float(torch.as_tensor(image).abs().max()) == 0:
                # an empty frame: a trained detector finds nothing there (this
                # tiny one does), so only the given-box warm-up reaches stage 2
                out.pred_centroids = torch.full_like(out.pred_centroids, float("nan"))
            else:
                real.append(out.pred_centroids.device.type)
        return out

    def spy_td(self, image, *a, frame_boxes=None, **k):
        # (layer, device of the frame) for calls that cut at least one crop
        if frame_boxes is not None and bool(torch.isfinite(frame_boxes).all(-1).any()):
            dev = image.device.type if isinstance(image, torch.Tensor) else "cpu"
            frames_in.append((id(self), dev))
        return orig_td(self, image, *a, frame_boxes=frame_boxes, **k)

    monkeypatch.setattr(BBoxStageLayer, "predict", spy)
    monkeypatch.setattr(TopDownLayer, "predict", spy_td)
    frames = _video_frames(shifted_video)
    kw = dict(
        model_paths=[detector_pair["det"], detector_pair["ci_bbox"]],
        device=device,
        peak_threshold=0.0,
        bbox_nms_iou=0.7,
        part_model_paths=[str(minimal_instance_centered_instance_ckpt)],
        part_crop_rule=RULE,
        part_peak_threshold=0.0,
        part_prefix="part/",
        frames_on_gpu=device == "cuda",  # the real-time settings
    )
    with open_stream(**kw, frame_shape=frames[0].shape) as stream:
        part_id = id(stream._predictor.part_stage.layer)
        warm, warm_part = set(given), {d for i, d in frames_in if i == part_id}
        real.clear()
        frames_in.clear()
        for f in frames:
            stream.push(f)
        real_part = {d for i, d in frames_in if i == part_id}
    assert warm and real and warm_part and real_part
    assert set(real) <= warm, (set(real), warm)  # pose boxes
    assert real_part <= warm_part, (real_part, warm_part)  # the part model's frame
