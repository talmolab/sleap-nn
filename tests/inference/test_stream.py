"""Frame in, pose out: ``open_stream`` / ``PoseStream`` against ``predict`` at batch 1."""

import numpy as np
import pytest
import sleap_io as sio
import torch

RULE = {
    "center": ["A"],
    "reference": [["A"], ["B"]],
    "scale": 1.0,
    "min_half_side": 4.0,
}


def _kw(centroid_ckpt, centered_ckpt, part=True):
    ckpts = [str(centroid_ckpt), str(centered_ckpt)]
    kw = dict(model_paths=ckpts, device="cpu", peak_threshold=0.0)
    if part:
        kw.update(
            part_model_paths=[ckpts[1]],
            part_crop_rule=RULE,
            part_peak_threshold=0.0,
            part_prefix="part/",
        )
    return kw


def _frames(labels_path):
    labels = sio.load_slp(str(labels_path))
    return [np.asarray(lf.image) for lf in labels.labeled_frames]


def test_stream_matches_predict_at_batch_1(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Frames pushed one by one give predict's points at batch 1 (several animals, part stage)."""
    from sleap_nn.inference.run import open_stream, predict

    kw = _kw(minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt)
    ref = predict(str(minimal_instance), batch_size=1, part_streaming=True, **kw)
    with open_stream(**kw) as stream:
        got = [stream.push(f) for f in _frames(minimal_instance)]
    assert stream.node_names == ["A", "B", "part/A", "part/B"]
    for lf, poses in zip(ref.labeled_frames, got):
        insts = lf.predicted_instances
        assert len(poses) == len(insts) and len(insts) >= 2
        a = np.stack([i.numpy() for i in insts]).astype(np.float32)
        np.testing.assert_array_equal(a, poses.points)  # bit-identical
        np.testing.assert_array_equal(
            np.array([i.score for i in insts], dtype=np.float32), poses.instance_scores
        )


def test_stream_channels_and_bad_input(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Grey and colour frames, BGR order, and loud errors on wrong input."""
    from sleap_nn.inference.run import open_stream

    kw = _kw(
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
        part=False,
    )
    frame = _frames(minimal_instance)[0]
    grey = frame[..., 0] if frame.ndim == 3 else frame
    rgb = np.stack([grey, grey // 2, grey // 3], axis=-1).astype(np.uint8)
    with open_stream(**kw) as stream:
        p_grey = stream.push(grey)
        assert p_grey.points.shape[1:] == (2, 2)
    with (
        open_stream(**kw) as rgb_stream,
        open_stream(**kw, channel_order="bgr") as bgr_stream,
    ):
        a = rgb_stream.push(rgb)
        b = bgr_stream.push(np.ascontiguousarray(rgb[..., ::-1]))  # the same frame, BGR
        np.testing.assert_array_equal(a.points, b.points)
        c = bgr_stream.push(np.ascontiguousarray(rgb[..., ::-1]).copy())
        assert c.frame_idx == 1
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
        t = stream.push(torch.from_numpy(np.ascontiguousarray(grey[:-4])))
        assert t.frame_idx == 1
    with pytest.raises(ValueError, match="channel_order"):
        open_stream(**kw, channel_order="xyz")
    with pytest.raises(NotImplementedError):
        open_stream(**kw, tracker=object())


def test_stream_empty_frame_keep_save_and_close(
    tmp_path,
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Zero animals, keep / save, warm-up at open, context manager."""
    from sleap_nn.inference.run import open_stream

    kw = _kw(minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt)
    frame = _frames(minimal_instance)[0]
    with open_stream(**{**kw, "peak_threshold": 2.0}) as stream:  # no peak can pass
        empty = stream.push(frame)
        assert len(empty) == 0
        assert empty.points.shape == (0, 4, 2) and empty.boxes.shape == (0, 4)
        assert empty.instance_scores.shape == (0,)
    with open_stream(**kw, keep=True, frame_shape=frame.shape) as stream:
        stream.push(frame)
        full = stream.push(frame)
        assert len(full) >= 2 and stream.last_latency_ms > 0
        stream.save(tmp_path / "s.slp")
    back = sio.load_slp(str(tmp_path / "s.slp"))
    assert len(back.labeled_frames) == 2
    assert len(back.labeled_frames[1].predicted_instances) == len(full)
    with pytest.raises(RuntimeError, match="closed"):
        stream.push(frame)
    with open_stream(**kw) as stream:
        with pytest.raises(ValueError, match="keep=True"):
            stream.labels()
