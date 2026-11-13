"""Part stage: crops built from an earlier stage's keypoints (keypoint_crops.py)."""

import json

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.inference.keypoint_crops import (
    KeypointCropRule,
    instance_boxes,
    keypoint_frame_box_fn,
    merge_part_predictions,
    run_part_stage,
)


def _resolved(rule, names):
    return rule.resolve(names)


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #
def test_rule_box_center_and_reference():
    """Square box on the center node, half side = scale x |center - ref mean|."""
    rule = KeypointCropRule(center=["base"], reference=["l", "r"], scale=2.0)
    names = ["nose", "l", "r", "base"]
    pts = np.array([[0, 0], [8, -3], [8, 3], [10, 0]], float)
    box, fb = rule.box(pts, _resolved(rule, names))
    # Reference mean (8, 0) is 2 px from the center: half side 4.
    np.testing.assert_allclose(box, [6, -4, 14, 4])
    assert not fb


def test_rule_reference_fallback_and_suffix_names():
    """A missing reference node switches to reference_fallback; names match by suffix."""
    rule = KeypointCropRule(
        center=["TTI", "tail_base"],
        reference=["Ear_L", "Ear_R"],
        reference_fallback=["Nose"],
        scale=1.0,
    )
    names = ["skel-1/Nose", "skel-1/Ear_L", "skel-1/Ear_R", "skel-1/TTI"]
    pts = np.array([[0, 0], [np.nan, np.nan], [5, 5], [10, 0]], float)
    box, fb = rule.box(pts, _resolved(rule, names))
    np.testing.assert_allclose(box, [0, -10, 20, 10])
    assert not fb
    # The second center candidate is used when the first is not in the skeleton.
    names2 = ["Nose", "Ear_L", "Ear_R", "tail_base"]
    box2, _ = rule.box(pts, _resolved(rule, names2))
    np.testing.assert_allclose(box2, box)


def test_rule_fallbacks():
    """No center: an extent box ('extent') or no box ('skip'); min_half_side floor."""
    names = ["a", "b", "c"]
    pts = np.array([[0, 0], [4, 2], [np.nan, np.nan]], float)
    rule = KeypointCropRule(center=["c"], reference=["a"], fallback_scale=1.5)
    box, fb = rule.box(pts, _resolved(rule, names))
    # Extent 4 x 2 around (2, 1): half side 1.5 x 4 / 2 = 3.
    np.testing.assert_allclose(box, [-1, -2, 5, 4])
    assert fb
    skip = KeypointCropRule(center=["c"], reference=["a"], fallback="skip")
    assert skip.box(pts, _resolved(skip, names)) == (None, False)
    tiny = KeypointCropRule(center=["a"], reference=["b"], scale=0.01, min_half_side=5)
    box, _ = tiny.box(pts, _resolved(tiny, names))
    np.testing.assert_allclose(box, [-5, -5, 5, 5])
    nothing = np.full((3, 2), np.nan)
    assert rule.box(nothing, _resolved(rule, names)) == (None, False)


def test_rule_load_and_validation(tmp_path):
    """Rules load from dicts, .json and .yaml files and reject bad fields."""
    d = dict(center=["base"], reference=["l", "r"], scale=2.0)
    assert KeypointCropRule.load(d) == KeypointCropRule(**d)
    (tmp_path / "r.json").write_text(json.dumps(d))
    (tmp_path / "r.yaml").write_text("center: [base]\nreference: [l, r]\nscale: 2.0\n")
    assert KeypointCropRule.load(tmp_path / "r.json") == KeypointCropRule(**d)
    assert KeypointCropRule.load(str(tmp_path / "r.yaml")) == KeypointCropRule(**d)
    assert KeypointCropRule.load(KeypointCropRule(**d)).to_dict()["scale"] == 2.0
    with pytest.raises(ValueError, match="Unknown"):
        KeypointCropRule.from_dict(dict(center=["a"], size=3))
    with pytest.raises(ValueError, match="fallback"):
        KeypointCropRule(center=["a"], fallback="grow")
    with pytest.raises(ValueError, match="center"):
        KeypointCropRule(center=[])


def _frame(skeleton, pts_list, scores=(0.9, 0.8)):
    video = sio.Video(filename="fake.mp4")
    insts = [
        sio.PredictedInstance.from_numpy(
            np.asarray(p, float),
            skeleton=skeleton,
            point_scores=np.ones(len(p)),
            score=s,
        )
        for p, s in zip(pts_list, scores)
    ]
    return video, sio.LabeledFrame(video=video, frame_idx=3, instances=insts)


def test_instance_boxes_and_frame_box_fn():
    """One box per predicted instance, scored by it; user instances are skipped."""
    sk = sio.Skeleton(["a", "b"])
    _, lf = _frame(sk, [[[0, 0], [1, 0]], [[10, 10], [10, 12]]])
    lf.instances.append(
        sio.Instance.from_numpy(np.array([[5.0, 5], [6, 5]]), skeleton=sk)
    )
    rule = KeypointCropRule(center=["a"], reference=["b"], scale=1.0)
    boxes, scores, insts = instance_boxes(lf, rule)
    np.testing.assert_allclose(boxes, [[-1, -1, 1, 1], [8, 8, 12, 12]])
    np.testing.assert_allclose(scores, [0.9, 0.8])
    assert insts == lf.predicted_instances
    b2, s2 = keypoint_frame_box_fn(rule)(lf)
    np.testing.assert_allclose(b2, boxes)
    np.testing.assert_allclose(s2, scores)


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #
def _part_labels(video, boxes, part_sk, which=(0, 1)):
    frames_insts, bxs = [], []
    for k in which:
        p = sio.PredictedInstance.from_numpy(
            np.array([[100.0 + k, 200.0]]),
            skeleton=part_sk,
            point_scores=np.array([0.5]),
            score=0.7,
        )
        frames_insts.append(p)
        bxs.append(sio.PredictedBoundingBox.from_xyxy(*boxes[k], score=0.7, instance=p))
    lf = sio.LabeledFrame(video=video, frame_idx=3, instances=frames_insts, bboxes=bxs)
    return sio.Labels([lf], videos=[video], skeletons=[part_sk])


def test_merge_part_predictions():
    """Part points are appended to the instance whose rule box made their crop."""
    sk = sio.Skeleton(["a", "b"])
    sk.add_edge("a", "b")
    video, lf = _frame(sk, [[[0, 0], [1, 0]], [[10, 10], [10, 12]]])
    base = sio.Labels([lf], videos=[video], skeletons=[sk])
    rule = KeypointCropRule(center=["a"], reference=["b"], scale=1.0)
    boxes, _, _ = instance_boxes(lf, rule)
    part_sk = sio.Skeleton(["tip"])
    # Part predictions in reverse box order: matching is by box, not by order.
    part = _part_labels(video, boxes, part_sk, which=(1, 0))
    merged, counts = merge_part_predictions(base, part, rule, part_prefix="tail/")
    assert counts == dict(n_instances=2, n_merged=2, n_unmatched=0)
    m = merged[0].instances
    assert m[0].skeleton.node_names == ["a", "b", "tail/tip"]
    assert len(m[0].skeleton.edges) == 1
    np.testing.assert_allclose(m[0].numpy()[2], [100, 200])
    np.testing.assert_allclose(m[1].numpy()[2], [101, 200])
    np.testing.assert_allclose(m[1].numpy()[:2], [[10, 10], [10, 12]])
    assert m[0].score == pytest.approx(0.9)
    assert merged.provenance["part_stage"]["rule"]["center"] == ["a"]
    # One part prediction missing: that instance gets NaN part points.
    merged, counts = merge_part_predictions(
        base, _part_labels(video, boxes, part_sk, which=(0,)), rule, part_prefix="t/"
    )
    assert counts["n_merged"] == 1
    assert np.isnan(merged[0].instances[1].numpy()[2]).all()


def test_merge_refuses_clashing_names():
    """Part nodes named like body nodes need part_prefix."""
    sk = sio.Skeleton(["a", "b"])
    video, lf = _frame(sk, [[[0, 0], [1, 0]], [[10, 10], [10, 12]]])
    base = sio.Labels([lf], videos=[video], skeletons=[sk])
    rule = KeypointCropRule(center=["a"], reference=["b"], scale=1.0)
    boxes, _, _ = instance_boxes(lf, rule)
    part = _part_labels(video, boxes, sio.Skeleton(["a"]))
    with pytest.raises(ValueError, match="part_prefix"):
        merge_part_predictions(base, part, rule)


# --------------------------------------------------------------------------- #
# End to end on the fixture models
# --------------------------------------------------------------------------- #
RULE = dict(center=["A"], reference=["B"], scale=2.0, min_half_side=8.0)


def _base(minimal_instance, centroid, centered):
    from sleap_nn.inference.run import predict

    return predict(
        str(minimal_instance),
        model_paths=[str(centroid), str(centered)],
        device="cpu",
        peak_threshold=0.0,
    )


def test_run_part_stage_end_to_end(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Body -> keypoint crops -> part model -> merged instances, every instance."""
    base = _base(
        minimal_instance,
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
    )
    n = sum(len(lf.predicted_instances) for lf in base)
    assert n > 0
    merged = run_part_stage(
        base,
        [str(minimal_instance_centered_instance_ckpt)],
        RULE,
        device="cpu",
        peak_threshold=0.0,
        part_prefix="part/",
    )
    insts = [i for lf in merged for i in lf.predicted_instances]
    assert len(insts) == n
    assert insts[0].skeleton.node_names == ["A", "B", "part/A", "part/B"]
    assert all(np.isfinite(i.numpy()[2:]).all() for i in insts)


def test_predict_with_part_model_matches_stagewise(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """predict(part_model_paths=...) = the pose run, then run_part_stage."""
    from sleap_nn.inference.run import predict

    base = _base(
        minimal_instance,
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
    )
    stagewise = run_part_stage(
        base,
        [str(minimal_instance_centered_instance_ckpt)],
        RULE,
        device="cpu",
        peak_threshold=0.0,
        part_prefix="part/",
    )
    out = tmp_path / "pred.slp"
    ckpts = [
        str(minimal_instance_centroid_ckpt),
        str(minimal_instance_centered_instance_ckpt),
    ]
    labels = predict(
        str(minimal_instance),
        model_paths=ckpts,
        device="cpu",
        peak_threshold=0.0,
        part_model_paths=[ckpts[1]],
        part_crop_rule=RULE,
        part_peak_threshold=0.0,
        part_prefix="part/",
        output_path=str(out),
    )
    a = np.stack([i.numpy() for lf in labels for i in lf.predicted_instances])
    b = np.stack([i.numpy() for lf in stagewise for i in lf.predicted_instances])
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)
    saved = sio.load_slp(str(out))
    assert saved.skeletons[0].node_names == ["A", "B", "part/A", "part/B"]
    with pytest.raises(ValueError, match="part_crop_rule"):
        predict(
            str(minimal_instance),
            model_paths=ckpts,
            device="cpu",
            part_model_paths=[ckpts[1]],
        )


def test_predict_without_part_model_is_unchanged(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """No part model: the same instances and skeleton as before the part stage existed."""
    base = _base(
        minimal_instance,
        minimal_instance_centroid_ckpt,
        minimal_instance_centered_instance_ckpt,
    )
    assert base.skeletons[0].node_names == ["A", "B"]
    assert "part_stage" not in base.provenance


def test_cli_part_model(
    minimal_instance,
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """sleap-nn predict --part_model/--part_crop_rule writes merged instances."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli

    rule = tmp_path / "rule.json"
    rule.write_text(json.dumps(RULE))
    out = tmp_path / "pred.slp"
    common = [
        "predict",
        "-i",
        str(minimal_instance),
        "-m",
        str(minimal_instance_centroid_ckpt),
        "-m",
        str(minimal_instance_centered_instance_ckpt),
        "--device",
        "cpu",
        "--peak_threshold",
        "0.0",
        "-o",
        str(out),
    ]
    res = CliRunner().invoke(
        cli,
        common
        + [
            "--part_model",
            str(minimal_instance_centered_instance_ckpt),
            "--part_crop_rule",
            str(rule),
            "--part_peak_threshold",
            "0.0",
            "--part_prefix",
            "part/",
        ],
    )
    assert res.exit_code == 0, res.output
    assert sio.load_slp(str(out)).skeletons[0].node_names == [
        "A",
        "B",
        "part/A",
        "part/B",
    ]
    res = CliRunner().invoke(
        cli, common + ["--part_model", str(minimal_instance_centered_instance_ckpt)]
    )
    assert res.exit_code != 0 and "--part_crop_rule" in res.output
