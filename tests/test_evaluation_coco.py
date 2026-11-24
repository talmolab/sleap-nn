"""Tests for the COCO-faithful keypoint evaluator (sleap_nn.evaluation_coco)."""

import numpy as np
import pytest
import sleap_io as sio

from sleap_nn.evaluation import Evaluator
from sleap_nn.evaluation_coco import (
    COCO_KEYPOINT_SIGMAS,
    attach_coco_detections,
    coco_keypoint_metrics,
    labels_to_coco_images,
    resolve_node_sigmas,
)

NODES = list(COCO_KEYPOINT_SIGMAS)


def _skeleton():
    return sio.Skeleton(nodes=NODES, name="coco")


def _make_labels(rng, n_frames=40, noise=0.08, n_fp=1, dup=0.3):
    """GT with posed people, 0-keypoint boxes and crowd boxes, plus noisy preds."""
    skel = _skeleton()
    video = sio.Video(filename="fake.mp4", open_backend=False)
    gt_lfs, pr_lfs = [], []
    for f in range(n_frames):
        gt_lf = sio.LabeledFrame(video=video, frame_idx=f)
        pr_lf = sio.LabeledFrame(video=video, frame_idx=f)
        for _ in range(rng.integers(0, 4)):
            c = rng.uniform(100, 500, size=2)
            s = rng.uniform(20, 120)
            pts = c + rng.normal(0, s / 3, size=(17, 2))
            pts[rng.random(17) < 0.3] = np.nan
            if np.isnan(pts).all():
                pts[0] = c
            inst = sio.Instance.from_numpy(pts, skeleton=skel)
            gt_lf.instances.append(inst)
            lo, hi = np.nanmin(pts, 0), np.nanmax(pts, 0)
            gt_lf.bboxes.append(
                sio.UserBoundingBox.from_xywh(*(lo - 5), *(hi - lo + 10), instance=inst)
            )
            if rng.random() < 0.85:
                filled = np.where(np.isnan(pts), np.nanmean(pts, 0), pts)
                p = filled + rng.normal(0, noise * s, size=filled.shape)
                score = float(rng.uniform(0.3, 1.0))
                pr_lf.instances.append(
                    sio.PredictedInstance.from_numpy(
                        p, skeleton=skel, point_scores=np.ones(17), score=score
                    )
                )
                if rng.random() < dup:
                    pr_lf.instances.append(
                        sio.PredictedInstance.from_numpy(
                            p + rng.normal(0, 2 * noise * s, size=p.shape),
                            skeleton=skel,
                            point_scores=np.ones(17),
                            score=score * 0.9,
                        )
                    )
        if rng.random() < 0.3:  # unlabeled person -> ignore region
            gt_lf.bboxes.append(sio.UserBoundingBox.from_xywh(20, 20, 60, 120))
        if rng.random() < 0.2:  # crowd region
            gt_lf.bboxes.append(
                sio.UserBoundingBox.from_xywh(
                    300, 300, 150, 100, category=sio.Category("crowd")
                )
            )
        for _ in range(rng.poisson(n_fp)):
            p = rng.uniform(0, 600, size=2) + rng.normal(0, 30, size=(17, 2))
            pr_lf.instances.append(
                sio.PredictedInstance.from_numpy(
                    p,
                    skeleton=skel,
                    point_scores=np.ones(17),
                    score=float(rng.uniform(0.2, 1.0)),
                )
            )
        gt_lfs.append(gt_lf)
        pr_lfs.append(pr_lf)
    gt = sio.Labels(labeled_frames=gt_lfs, videos=[video], skeletons=[skel])
    pr = sio.Labels(labeled_frames=pr_lfs, videos=[video], skeletons=[skel])
    return gt, pr


def test_resolve_node_sigmas():
    """Sigmas resolve from presets, scalars, node names and union provenance."""
    skel = _skeleton()
    coco = np.array(list(COCO_KEYPOINT_SIGMAS.values()))
    np.testing.assert_allclose(resolve_node_sigmas(skel), coco)  # by node name
    np.testing.assert_allclose(resolve_node_sigmas(skel, "coco"), coco)
    np.testing.assert_allclose(resolve_node_sigmas(skel, 0.05), np.full(17, 0.05))
    with pytest.raises(ValueError):
        resolve_node_sigmas(skel, [0.1, 0.2])

    # Union skeleton: provenance sigmas are keyed by native names + union_map.
    union = sio.Skeleton(nodes=["crowdpose_head", "neck_union", "left_hip"])
    prov = {
        "node_sigmas": {"head": 0.079, "neck": 0.08},
        "union_map": "{'head': 'crowdpose_head', 'neck': 'neck_union'}",
    }
    np.testing.assert_allclose(
        resolve_node_sigmas(union, None, prov), [0.079, 0.08, 0.107]
    )


def test_exact_predictions_score_one():
    """Predictions equal to the GT score AP = AR = 1."""
    rng = np.random.default_rng(0)
    gt, _ = _make_labels(rng)
    pr_lfs = []
    for lf in gt.labeled_frames:
        pr_lfs.append(
            sio.LabeledFrame(
                video=lf.video,
                frame_idx=lf.frame_idx,
                instances=[
                    sio.PredictedInstance.from_numpy(
                        np.nan_to_num(i.numpy(), nan=0.0),
                        skeleton=gt.skeletons[0],
                        point_scores=np.ones(17),
                        score=1.0,
                    )
                    for i in lf.user_instances
                ],
            )
        )
    pr = sio.Labels(labeled_frames=pr_lfs, videos=gt.videos, skeletons=gt.skeletons)
    m = coco_keypoint_metrics(gt, pr)
    assert m["coco.AP"] == pytest.approx(1.0)
    assert m["coco.AR"] == pytest.approx(1.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools(seed):
    """The numpy evaluator reproduces pycocotools on the same records."""
    pytest.importorskip("pycocotools")
    gt, pr = _make_labels(np.random.default_rng(seed))
    images = labels_to_coco_images(gt, pr)
    assert sum(im.gt_ignore.sum() for im in images) > 0
    native = coco_keypoint_metrics(gt, pr, images=images)
    ref = coco_keypoint_metrics(gt, pr, images=images, backend="pycocotools")
    for k in ["AP", "AP50", "AP75", "AP_M", "AP_L", "AR", "AR50", "AR75"]:
        assert native[f"coco.{k}"] == pytest.approx(ref[f"coco.{k}"], abs=1e-9), k


def test_unmatched_predictions_count_as_false_positives():
    """count_unmatched_as_fp adds dropped predictions as FPs."""
    gt, pr = _make_labels(np.random.default_rng(3), n_fp=2)
    sig = resolve_node_sigmas(gt.skeletons[0])
    base = Evaluator(gt, pr, oks_stddev=sig).voc_metrics()
    fixed = Evaluator(gt, pr, oks_stddev=sig, count_unmatched_as_fp=True)
    assert fixed.unmatched_pred_scores.size > 0
    assert fixed.voc_metrics()["oks_voc.mAP"] <= base["oks_voc.mAP"]


def test_attach_coco_detections(tmp_path):
    """COCO detection JSON rows attach as PredictedBoundingBoxes by image_id."""
    skel = _skeleton()
    video = sio.Video(
        filename=["/x/000000000042.jpg", "/x/000000000007.jpg"], open_backend=False
    )
    labels = sio.Labels(
        labeled_frames=[
            sio.LabeledFrame(video=video, frame_idx=0),
            sio.LabeledFrame(video=video, frame_idx=1),
        ],
        videos=[video],
        skeletons=[skel],
    )
    dets = [
        {"image_id": 42, "category_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9},
        {"image_id": 7, "category_id": 1, "bbox": [5, 6, 7, 8], "score": 0.1},
        {"image_id": 99, "category_id": 1, "bbox": [5, 6, 7, 8], "score": 0.9},
    ]
    counts = attach_coco_detections(labels, dets, score_threshold=0.2)
    assert counts == {"n_boxes": 1, "n_frames": 1, "n_unmatched": 1}
    box = labels.labeled_frames[0].bboxes[0]
    assert box.is_predicted and box.score == pytest.approx(0.9)
    assert (box.x1, box.y1, box.x2, box.y2) == (1, 2, 4, 6)


def test_protocol_resolution():
    """Explicit settings beat the protocol preset; 'auto' reads the provenance."""
    from sleap_nn.evaluation_coco import EVAL_PROTOCOLS, resolve_protocol

    gt, _ = _make_labels(np.random.default_rng(0), n_frames=2)
    assert resolve_protocol(gt, None) is None
    assert resolve_protocol(gt, "auto") is None  # no provenance dataset
    gt.provenance["dataset"] = "crowdpose"
    assert resolve_protocol(gt, "auto") == EVAL_PROTOCOLS["crowdpose"]
    assert resolve_protocol(gt, "coco") == EVAL_PROTOCOLS["coco"]
    with pytest.raises(ValueError, match="Unknown COCO eval protocol"):
        resolve_protocol(gt, "mpii")
    # CrowdPose's num_keypoints rule needs /human_meta, which this GT lacks...
    with pytest.raises(ValueError, match="human_meta"):
        coco_keypoint_metrics(gt, gt, protocol="auto")
    # ... unless the rule is overridden.
    m = coco_keypoint_metrics(gt, gt, protocol="auto", ignore_rule="labeled")
    assert m["coco.n_images"] == 2


def test_keypointless_instance_is_an_ignore_region():
    """A posed GT with no labeled keypoint and a box is a COCO ignore region."""
    skel = _skeleton()
    video = sio.Video(filename="fake.mp4", open_backend=False)
    empty = sio.Instance.from_numpy(np.full((17, 2), np.nan), skeleton=skel)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[empty])
    lf.bboxes.append(sio.UserBoundingBox.from_xywh(10, 10, 50, 100, instance=empty))
    gt = sio.Labels(labeled_frames=[lf], videos=[video], skeletons=[skel])
    (img,) = labels_to_coco_images(gt, gt)
    assert img.gt_ignore.tolist() == [True]
    np.testing.assert_allclose(img.gt_bbox[0], [10, 10, 50, 100])
    np.testing.assert_allclose(img.gt_area, [50 * 100 * 0.53])


def test_frame_image_ids_need_image_lists():
    """Image ids come from image-list file names; single-file videos have none."""
    from sleap_nn.evaluation_coco import frame_image_ids

    skel = _skeleton()
    video = sio.Video(filename="movie_000123.mp4", open_backend=False)
    labels = sio.Labels(
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=0)],
        videos=[video],
        skeletons=[skel],
    )
    assert frame_image_ids(labels) == [None]
    with pytest.raises(ValueError, match="No labeled frame has a COCO image id"):
        attach_coco_detections(labels, [])


def test_transfer_predicted_boxes():
    """Predicted boxes copy across files by video and frame, adding frames."""
    from sleap_nn.evaluation_coco import transfer_predicted_boxes

    skel = _skeleton()
    video = sio.Video(filename="fake.mp4", open_backend=False)
    labels = sio.Labels(
        labeled_frames=[sio.LabeledFrame(video=video, frame_idx=0)],
        videos=[video],
        skeletons=[skel],
    )
    video_b = sio.Video(filename="fake.mp4", open_backend=False)
    inst = sio.PredictedInstance.from_numpy(
        np.zeros((17, 2)), skeleton=skel, point_scores=np.ones(17), score=1.0
    )
    box_lfs = [
        sio.LabeledFrame(
            video=video_b,
            frame_idx=0,
            instances=[inst],
            bboxes=[
                sio.PredictedBoundingBox(
                    x1=0, y1=0, x2=1, y2=1, score=0.9, instance=inst
                ),
                sio.PredictedBoundingBox(x1=0, y1=0, x2=2, y2=2, score=0.1),
                sio.UserBoundingBox(x1=0, y1=0, x2=3, y2=3),
            ],
        ),
        sio.LabeledFrame(
            video=video_b,
            frame_idx=5,
            bboxes=[sio.PredictedBoundingBox(x1=0, y1=0, x2=4, y2=4, score=0.5)],
        ),
    ]
    box_labels = sio.Labels(labeled_frames=box_lfs, videos=[video_b], skeletons=[skel])
    counts = transfer_predicted_boxes(labels, box_labels, score_threshold=0.2)
    assert counts == {"n_boxes": 2, "n_frames": 2, "n_new_frames": 1, "n_unmatched": 0}
    first = labels.find(video, 0)[0].bboxes
    assert len(first) == 1 and first[0].instance is None and first[0].score == 0.9
    assert labels.find(video, 5)[0].bboxes[0].x2 == 4


def test_eval_cli_coco(tmp_path):
    """sleap-nn eval --coco reports coco.* next to the legacy oks_voc metrics."""
    from click.testing import CliRunner

    from sleap_nn.cli import cli, parse_oks_sigmas

    gt, pr = _make_labels(np.random.default_rng(4), n_frames=10)
    gt.save(tmp_path / "gt.slp")
    pr.save(tmp_path / "pr.slp")
    metrics_path = tmp_path / "metrics.npz"
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "-g",
            str(tmp_path / "gt.slp"),
            "-p",
            str(tmp_path / "pr.slp"),
            "--coco",
            "--oks_sigmas",
            "coco",
            "--coco_protocol",
            "none",
            "--save_metrics",
            str(metrics_path),
        ],
    )
    assert result.exit_code == 0, result.output
    metrics = np.load(metrics_path, allow_pickle=True)["metrics"].item()
    ref = coco_keypoint_metrics(
        sio.load_slp(tmp_path / "gt.slp"), sio.load_slp(tmp_path / "pr.slp")
    )
    assert metrics["coco_metrics"]["coco.AP"] == pytest.approx(ref["coco.AP"])
    assert "oks_voc.mAP" in metrics["voc_metrics"]

    assert parse_oks_sigmas(None) is None
    assert parse_oks_sigmas("crowdpose") == "crowdpose"
    assert parse_oks_sigmas("0.05") == 0.05
    assert parse_oks_sigmas("0.1,0.2") == [0.1, 0.2]
    import click

    with pytest.raises(click.BadParameter):
        parse_oks_sigmas("cocoo")


def test_coco_images_score_occluded_estimates():
    """GT occluded estimates (COCO v=1) are scored; unlabeled nodes are not."""
    from sleap_nn.evaluation_coco import labels_to_coco_images

    skel = sio.Skeleton(["a", "b", "c"])
    video = sio.Video(filename="v.mp4")
    xy = np.array([[10.0, 10.0], [50.0, 10.0], [50.0, 50.0]])
    gt = sio.Instance.from_numpy(xy, skeleton=skel)
    gt.set_hidden(["b"], keep_location=True)
    gt.set_unlabeled(["c"])
    pr = sio.PredictedInstance.from_numpy(xy, skeleton=skel, score=1.0)
    labels_gt = sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=0, instances=[gt])],
        videos=[video],
        skeletons=[skel],
    )
    labels_pr = sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=0, instances=[pr])],
        videos=[video],
        skeletons=[skel],
    )
    (img,) = labels_to_coco_images(labels_gt, labels_pr)
    np.testing.assert_allclose(img.gt_points[0, :2], xy[:2])
    assert np.isnan(img.gt_points[0, 2]).all()
