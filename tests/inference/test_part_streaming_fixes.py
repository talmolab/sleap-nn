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
        if not valid.any():
            return Outputs(
                pred_keypoints=torch.zeros(B, 0, 1, 2),
                pred_peak_values=torch.zeros(B, 0, 1),
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


def test_streaming_over_batches_videos_with_tracking_and_export_part(
    two_videos,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """Batches of 4 over two videos (12 frames, one empty), floor building up, tracking,
    a network-only export as the part model with runtime set: the pose points and
    the tracks equal the two-pass run's; provenance has the two-pass keys."""
    pytest.importorskip("onnxruntime")
    from click.testing import CliRunner

    from sleap_nn.export.cli import export
    from sleap_nn.inference.run import predict
    from sleap_nn.inference.tracking import TrackerConfig

    part_dir = tmp_path / "part"
    result = CliRunner().invoke(
        export,
        [
            str(minimal_instance_centered_instance_ckpt),
            "-o",
            str(part_dir),
            "--network-only",
            "--input-dtype",
            "float32",
        ],
    )
    assert result.exit_code == 0, result.output
    kw = dict(
        model_paths=[
            str(minimal_instance_centroid_ckpt),
            str(minimal_instance_centered_instance_ckpt),
        ],
        device="cpu",
        batch_size=4,
        peak_threshold=0.05,
        part_model_paths=[str(part_dir)],
        part_crop_rule=RULE_FLOOR,
        part_prefix="part/",
        runtime="onnx",
        tracker_config=TrackerConfig(),
    )
    two = predict(two_videos, **kw)
    one = predict(two_videos, part_streaming=True, **kw)
    assert [n.name for n in one.skeletons[0].nodes] == ["A", "B", "part/A", "part/B"]

    def rows(labels):
        out = {}
        for lf in labels.labeled_frames:
            key = (labels.videos.index(lf.video), lf.frame_idx)
            out[key] = [
                (i.track.name if i.track else None, i.numpy()[:2].tolist())
                for i in lf.instances
            ]
        return out

    a, b = rows(two), rows(one)
    assert a.keys() == b.keys() and len(a) == 12
    assert a == b  # same tracks, same pose points, in the same order
    assert not a[(1, 2)]  # the empty frame
    assert sum(len(v) for v in b.values()) >= 11
    prov = one.provenance["part_stage"]
    assert set(two.provenance["part_stage"]) <= set(prov)
    assert prov["streaming"] is True
    assert set(prov["session_floors"]) == {0, 1}
    assert all(f > 0 for f in prov["session_floors"].values())
    assert "model_paths" in prov["part_provenance"] or prov["part_provenance"]
    no_floor = predict(two_videos, part_streaming=True, **dict(kw, part_crop_rule=RULE))
    assert no_floor.provenance["part_stage"]["session_floors"] is None


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
