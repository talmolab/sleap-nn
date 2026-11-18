"""Streaming part stage and frames-on-GPU fixes from the review of #859 / #862."""

import numpy as np
import pytest
import sleap_io as sio
import torch
from _pytest.logging import LogCaptureFixture
from loguru import logger

from sleap_nn.inference.keypoint_crops import KeypointCropRule, StreamingPartStage
from sleap_nn.inference.outputs import Outputs

RULE = dict(center=["A"], reference=["B"], scale=2.0, min_half_side=8.0)
RULE_FLOOR = dict(RULE, session_floor=1.0, session_min_boxes=3)


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(caplog.handler, format="{message}", level=0)
    yield caplog
    logger.remove(handler_id)


def _frame(minimal_instance):
    img = np.asarray(sio.load_slp(str(minimal_instance)).labeled_frames[0].image)
    return img[..., 0] if img.ndim == 3 else img


@pytest.fixture
def two_videos(minimal_instance, tmp_path):
    """Two lossless (PNG) videos of shifted fixture frames; the second has an empty frame."""
    from PIL import Image

    base = _frame(minimal_instance)
    videos = []
    for v in range(2):
        paths = []
        for i in range(6):
            img = np.roll(base, shift=(2 * i + v, -3 * i), axis=(0, 1))
            if v == 1 and i == 2:
                img = np.zeros_like(base)  # nothing to find
            path = tmp_path / f"v{v}_{i}.png"
            Image.fromarray(np.ascontiguousarray(img)).save(path)
            paths.append(path.as_posix())
        videos.append(sio.Video.from_filename(paths))
    frames = [
        sio.LabeledFrame(video=video, frame_idx=i)
        for video in videos
        for i in range(len(video))
    ]
    return sio.Labels(labeled_frames=frames, videos=videos)


class _FakePart:
    """A part layer whose empty path reports one node, as an exported network does."""

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

    def predict(self, images, frame_boxes, frame_box_scores):
        B = frame_boxes.shape[0]
        valid = ~torch.isnan(frame_boxes).any(dim=-1)
        if not valid.any():  # as an exported network's empty path: K slots, 1 node, NaN
            K = frame_boxes.shape[1]
            return Outputs(
                pred_keypoints=torch.full((B, K, 1, 2), float("nan")),
                pred_peak_values=torch.full((B, K, 1), float("nan")),
            )
        k = frame_boxes.shape[1]
        return Outputs(
            pred_keypoints=torch.ones(B, k, self.n_nodes, 2),
            pred_peak_values=torch.ones(B, k, self.n_nodes),
        )


def _outputs(kpts):
    return Outputs(pred_keypoints=torch.from_numpy(np.asarray(kpts, np.float32)))


def test_no_floor_stores_nothing():
    """Without a session floor the stage keeps no history and reports no floor."""
    stage = StreamingPartStage(_FakePart(2), KeypointCropRule(**RULE), ["A", "B"])
    rng = np.random.default_rng(0)
    for _ in range(50):
        stage(torch.zeros(1, 1, 8, 8), _outputs(rng.normal(50, 10, (1, 4, 2, 2))))
    assert stage.halves == {} and len(stage.every) == 0 and stage.last_floors == {}


def test_batch_without_crops_keeps_the_part_node_count():
    """fallback="skip" and no crop in a batch: the instances still get the part
    model's node count (the empty path of an exported network reports 1)."""
    rule = KeypointCropRule(**dict(RULE, fallback="skip"))
    stage = StreamingPartStage(_FakePart(2), rule, ["A", "B"], n_part_nodes=2)
    k = np.full((1, 2, 2, 2), np.nan, np.float32)
    k[0, :, 1] = [[10, 10], [40, 40]]  # B found, center A missing: no crop
    out = stage(torch.zeros(1, 1, 8, 8), _outputs(k))
    assert out.pred_keypoints.shape == (1, 2, 4, 2)
    assert torch.isnan(out.pred_keypoints[0, :, 2:]).all()
    k[0, 0, 0] = [12, 12]  # one crop now: the part layer's own count (2) is used
    out = stage(torch.zeros(1, 1, 8, 8), _outputs(k))
    assert out.pred_keypoints.shape == (1, 2, 4, 2)
    with pytest.raises(ValueError, match="skeleton has 3"):
        StreamingPartStage(_FakePart(2), rule, ["A", "B"], n_part_nodes=3)(
            torch.zeros(1, 1, 8, 8), _outputs(k)
        )


def test_build_takes_the_node_count_from_the_part_skeleton(
    minimal_instance_centered_instance_ckpt,
):
    from sleap_nn.inference.keypoint_crops import build_streaming_part_stage

    stage, merged = build_streaming_part_stage(
        sio.Skeleton(["A", "B"]),
        [str(minimal_instance_centered_instance_ckpt)],
        RULE,
        device="cpu",
        part_prefix="part/",
    )
    assert stage.n_part_nodes == 2 and len(merged.nodes) == 4
    assert stage.body_skeleton.node_names == ["A", "B"]


def _export_part(ckpt, tmp_path):
    from click.testing import CliRunner

    from sleap_nn.export.cli import export

    part_dir = tmp_path / "part"
    result = CliRunner().invoke(
        export,
        [str(ckpt), "-o", str(part_dir), "--network-only", "--input-dtype", "float32"],
    )
    assert result.exit_code == 0, result.output
    return str(part_dir)


@pytest.mark.parametrize("part_kind", ["checkpoint", "onnx_export"])
def test_streaming_over_batches_and_videos(
    part_kind,
    two_videos,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """Batches of 4 over two videos (12 frames, one empty), with tracking.

    Against two passes: the same frames, instances, tracks and pose points; part
    points too when no floor applies. With a floor: the floor builds up per video
    and is recorded; the provenance has the two-pass keys. The checkpoint variant
    runs in every CI job; the export variant (runtime="onnx") needs onnxruntime.
    """
    from sleap_nn.inference.run import predict
    from sleap_nn.inference.tracking import TrackerConfig

    if part_kind == "onnx_export":
        pytest.importorskip("onnxruntime")
        part = _export_part(minimal_instance_centered_instance_ckpt, tmp_path)
        extra = dict(runtime="onnx")
    else:
        part = str(minimal_instance_centered_instance_ckpt)
        extra = {}
    kw = dict(
        model_paths=[
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
        ],
        device="cpu",
        batch_size=4,
        peak_threshold=0.05,
        part_model_paths=[part],
        part_crop_rule=RULE,
        part_prefix="part/",
        tracker_config=TrackerConfig(),
        **extra,
    )
    two = predict(two_videos, **kw)
    one = predict(two_videos, part_streaming=True, **kw)
    assert [n.name for n in one.skeletons[0].nodes] == ["A", "B", "part/A", "part/B"]

    def rows(labels):
        out = {}
        for lf in labels.labeled_frames:
            key = (labels.videos.index(lf.video), lf.frame_idx)
            out[key] = [
                (i.track.name if i.track else None, i.numpy()) for i in lf.instances
            ]
        return out

    a, b = rows(two), rows(one)
    assert a.keys() == b.keys() and len(a) == 12
    assert not a[(1, 2)]  # the empty frame
    assert sum(len(v) for v in b.values()) >= 11
    for k in a:
        assert [t for t, _ in a[k]] == [t for t, _ in b[k]]  # same tracks, same order
        for (_, pa), (_, pb) in zip(a[k], b[k]):
            np.testing.assert_array_equal(pa[:2], pb[:2])  # pose points: identical
            np.testing.assert_allclose(pa[2:], pb[2:], atol=1e-3)  # part points
    prov = one.provenance["part_stage"]
    assert set(two.provenance["part_stage"]) <= set(prov) and prov["streaming"] is True
    assert "session_floors" not in prov  # no floor: no key, as two passes
    pp = prov["part_provenance"]
    from pathlib import Path

    # provenance stores resolved POSIX paths (the same on every system)
    assert pp["model_paths"] == [Path(part).resolve().as_posix()]
    assert pp["model_type"] and pp["inference_config"]
    floored = predict(
        two_videos, part_streaming=True, **dict(kw, part_crop_rule=RULE_FLOOR)
    )
    floors = floored.provenance["part_stage"]["session_floors"]
    assert set(floors) == {0, 1} and all(f > 0 for f in floors.values())


def test_streaming_tracks_on_pose_nodes_when_part_points_mislead(
    tmp_path,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
):
    """Part points that would pull the tracker to the other animal do not: tracks equal
    those of the pose points alone, and each instance keeps its own part points.

    The stage-1/2 layer and the part stage are stand-ins with designed points; the
    tracking runs through Predictor.predict as with --part_streaming.
    """
    import attrs
    from PIL import Image

    from sleap_nn.inference.layers.topdown import TopDownLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

    paths = []
    for f in range(4):
        img = np.zeros((96, 96), np.uint8)
        img[0, 0] = f  # the frame index, read back by the stand-in layer
        path = tmp_path / f"f{f}.png"
        Image.fromarray(img).save(path)
        paths.append(path.as_posix())
    video = sio.Video.from_filename(paths)

    def body(f):  # two animals 50 px apart; listed in reverse order on odd frames
        a = np.array([[10 + f, 20], [10 + f, 30]], np.float32)
        b = np.array([[60 + f, 20], [60 + f, 30]], np.float32)
        return [a, b] if f % 2 == 0 else [b, a]

    def part(f):  # 3 part points per instance; from frame 1 on, at the OTHER animal
        out = []
        for i, pts in enumerate(body(f)):
            x = pts[0, 0]
            if f > 0:
                x = 60 + f if x < 40 else 10 + f
            out.append(np.array([[x, 40], [x, 45], [x, 50]], np.float32))
        return out

    def fake_predict(self, image, **kw):
        frames = [int(np.asarray(image)[b].reshape(-1)[0]) for b in range(len(image))]
        k = torch.from_numpy(np.stack([np.stack(body(f)) for f in frames]))
        return Outputs(
            pred_keypoints=k,
            pred_peak_values=torch.ones(k.shape[:3]),
            instance_scores=torch.ones(k.shape[:2]),
        )

    class _Tails:
        body_skeleton = sio.Skeleton(["A", "B"])

        def reset(self):
            pass

        def __call__(self, images, outputs, video_indices=None):
            frames = [
                int(np.asarray(images)[b].reshape(-1)[0]) for b in range(len(images))
            ]
            t = torch.from_numpy(np.stack([np.stack(part(f)) for f in frames]))
            return attrs.evolve(
                outputs,
                pred_keypoints=torch.cat([outputs.pred_keypoints, t], dim=2),
                pred_peak_values=torch.cat(
                    [outputs.pred_peak_values, torch.ones(t.shape[:3])], dim=2
                ),
            )

    monkeypatch.setattr(TopDownLayer, "predict", fake_predict)
    p = Predictor.from_model_paths(
        [
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
        ],
        device="cpu",
        batch_size=2,
    )
    p.part_stage, p.skeleton = _Tails(), sio.Skeleton(["A", "B", "t1", "t2", "t3"])
    p.tracker_config = TrackerConfig()
    got = p.predict(video, make_labels=True)

    def designed(with_part):
        sk = p.skeleton if with_part else _Tails.body_skeleton
        frames = []
        for f in range(4):
            insts = [
                sio.PredictedInstance.from_numpy(
                    np.concatenate([bp, pp]) if with_part else bp,
                    skeleton=sk,
                    point_scores=np.ones(5 if with_part else 2),
                    score=1.0,
                )
                for bp, pp in zip(body(f), part(f))
            ]
            frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=insts))
        lab = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[sk])
        return apply_tracking(lab, TrackerConfig())

    def tracks(
        labels,
    ):  # per frame: the track of the animal at x < 40, then of the other
        out = []
        for lf in sorted(labels.labeled_frames, key=lambda q: q.frame_idx):
            by_x = sorted(lf.instances, key=lambda i: i.numpy()[0, 0])
            out.append(tuple(i.track.name for i in by_x))
        return out

    pose_only, with_part = tracks(designed(False)), tracks(designed(True))
    assert pose_only != with_part  # the case discriminates: part points mislead
    assert tracks(got) == pose_only
    for lf in got.labeled_frames:  # each instance keeps its own (designed) part points
        for inst in lf.instances:
            pts = inst.numpy()
            mine = [
                pp
                for bp, pp in zip(body(lf.frame_idx), part(lf.frame_idx))
                if np.array_equal(bp, pts[:2])
            ]
            assert len(mine) == 1 and np.array_equal(pts[2:], mine[0])


def test_body_tracking_with_a_repeated_frame():
    """A frame requested twice (overlapping --frames ranges) is kept twice and tracked
    as tracking the pose points alone does."""
    from sleap_nn.inference.keypoint_crops import track_on_body_nodes
    from sleap_nn.inference.tracking import TrackerConfig, apply_tracking

    body_sk, full_sk = sio.Skeleton(["A", "B"]), sio.Skeleton(["A", "B", "t"])
    video = sio.Video(filename="v", open_backend=False)

    def labels(full):
        frames = []
        for f in (0, 1, 1, 2):
            pts = np.array([[10 + f, 10], [20 + f, 10], [30 + f, 10]], np.float64)
            sk = full_sk if full else body_sk
            n = 3 if full else 2
            inst = sio.PredictedInstance.from_numpy(
                pts[:n], skeleton=sk, point_scores=np.ones(n), score=1.0
            )
            frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=[inst]))
        return sio.Labels(labeled_frames=frames, videos=[video], skeletons=[sk])

    got = track_on_body_nodes(labels(True), body_sk, TrackerConfig())
    want = apply_tracking(labels(False), TrackerConfig())
    assert [lf.frame_idx for lf in got.labeled_frames] == [0, 1, 1, 2]
    assert [lf.instances[0].track.name for lf in got.labeled_frames] == [
        lf.instances[0].track.name for lf in want.labeled_frames
    ]
    assert all(lf.instances[0].numpy().shape == (3, 2) for lf in got.labeled_frames)


def test_predict_part_streaming_needs_a_part_model(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    from sleap_nn.inference.run import predict

    with pytest.raises(ValueError, match="part_streaming needs part_model_paths"):
        predict(
            str(minimal_instance),
            model_paths=[
                str(minimal_instance_centroid_ckpt),
                str(minimal_instance_centered_instance_ckpt),
            ],
            device="cpu",
            part_streaming=True,
        )


def test_bottomup_pipelined_path_keeps_the_part_stage(
    minimal_instance,
    minimal_instance_bottomup_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """With a part stage, predict_streaming does not take the paf_workers pipeline
    (which has no part stage) and the merged skeleton's part nodes are filled."""
    from sleap_nn.inference.keypoint_crops import build_streaming_part_stage
    from sleap_nn.inference.predictor import Predictor

    p = Predictor.from_model_paths(
        [str(minimal_instance_bottomup_ckpt)], device="cpu", paf_workers=2
    )
    stage, merged = build_streaming_part_stage(
        p.skeleton,
        [str(minimal_instance_centered_instance_ckpt)],
        RULE,
        device="cpu",
        peak_threshold=0.0,
        part_prefix="part/",
    )
    p.part_stage, p.skeleton = stage, merged
    outs = list(p.predict_streaming(str(minimal_instance)))
    assert all(o.pred_keypoints.shape[2] == 4 for o in outs)


def test_frames_on_gpu_without_cuda_warns(
    caplog, minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt
):
    from sleap_nn.inference.predictor import Predictor

    Predictor.from_model_paths(
        [
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
        ],
        device="cpu",
        frames_on_gpu=True,
    )
    assert "no stage of this model runs on CUDA" in caplog.text
    assert "frames stay on CPU" in caplog.text


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_part_stage_gets_the_gpu_copy_of_the_frames(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    monkeypatch,
):
    """Both options on: the part stage receives the frames already on the card."""
    from sleap_nn.inference.run import predict

    seen = []
    real = StreamingPartStage.__call__

    def spy(self, images, outputs, video_indices=None):
        seen.append(images.device.type if torch.is_tensor(images) else "numpy")
        return real(self, images, outputs, video_indices)

    monkeypatch.setattr(StreamingPartStage, "__call__", spy)
    ckpts = [
        str(minimal_instance_centroid_ckpt),
        str(minimal_instance_centered_instance_ckpt),
    ]
    labels = predict(
        str(minimal_instance),
        model_paths=ckpts,
        device="cuda",
        peak_threshold=0.0,
        frames_on_gpu=True,
        part_streaming=True,
        part_model_paths=[ckpts[1]],
        part_crop_rule=RULE,
        part_prefix="part/",
    )
    assert seen and set(seen) == {"cuda"}
    assert any(np.isfinite(i.numpy()[2:]).any() for lf in labels for i in lf.instances)


def test_real_part_layer_on_a_batch_without_crops(
    minimal_instance_centered_instance_ckpt,
):
    """The real part layer, fallback "skip", no crop in the batch: pose + part nodes."""
    from sleap_nn.inference.keypoint_crops import build_streaming_part_stage

    stage, merged = build_streaming_part_stage(
        sio.Skeleton(["A", "B"]),
        [str(minimal_instance_centered_instance_ckpt)],
        dict(RULE, fallback="skip"),
        device="cpu",
        part_prefix="part/",
    )
    k = np.full((2, 3, 2, 2), np.nan, np.float32)
    k[:, :2, 1] = [[20, 20], [60, 60]]  # B found, center A missing: no crop anywhere
    out = stage(torch.zeros(2, 1, 96, 96, dtype=torch.uint8), _outputs(k))
    assert out.pred_keypoints.shape == (2, 3, 4, 2) and len(merged.nodes) == 4
    assert torch.isnan(out.pred_keypoints[:, :, 2:]).all()
