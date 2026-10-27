"""Tests for `sleap-nn detect-boxes` (detector boxes matched into training labels)."""

import json

import cv2
import h5py
import numpy as np
import pytest
import sleap_io as sio
import torch
from click.testing import CliRunner

from sleap_nn.data.topdown_warp import build_instance_table
from sleap_nn.inference.detect_boxes import (
    FrameDetections,
    TileSpec,
    default_output_path,
    detect_boxes,
    detect_with_layer,
    gt_people,
    iou_matrix,
    load_coco_detections,
    load_detection_chunks,
    match_detections,
    merge_tile_detections,
    save_coco_detections,
    write_detector_labels,
)

NODES = ["head", "hip", "foot"]
PERSON = sio.Category(name="person")


def _inst(sk, x0, y0, w=20.0, h=40.0):
    pts = np.array([[x0, y0], [x0 + w / 2, y0 + h / 2], [x0 + w, y0 + h]])
    return sio.Instance.from_numpy(pts, skeleton=sk)


def _write_images(tmp_path, n, hw=(120, 160)):
    paths = []
    for i in range(n):
        p = tmp_path / f"{i + 1:012d}.png"
        cv2.imwrite(str(p), np.full((*hw, 3), 40 * i, np.uint8))
        paths.append(str(p))
    return paths


def _labels(tmp_path):
    """Two frames of labels with every kind of ground-truth person.

    Frame 0: a boxed person, a keypoint-less person, a crowd box and an unboxed
    pose instance (keypoint-extent GT). Frame 1: nobody.
    """
    sk = sio.Skeleton(nodes=list(NODES))
    video = sio.Video.from_filename(_write_images(tmp_path, 2))
    a = _inst(sk, 10, 10)  # box [8, 8, 32, 52]
    c = _inst(sk, 110, 20)  # no linked box: extent [110, 20, 130, 60]
    lf0 = sio.LabeledFrame(video=video, frame_idx=0, instances=[a, c])
    lf0.bboxes.append(sio.UserBoundingBox(8, 8, 32, 52, instance=a, category=PERSON))
    lf0.bboxes.append(sio.UserBoundingBox(50, 10, 80, 70, category=PERSON))  # no kpts
    crowd = sio.UserBoundingBox(85, 60, 150, 110, category=PERSON)
    crowd.metadata["iscrowd"] = 1
    lf0.bboxes.append(crowd)
    lf1 = sio.LabeledFrame(video=video, frame_idx=1)
    labels = sio.Labels([lf0, lf1], skeletons=[sk], videos=[video])
    return labels, a, c


def _dets(per_frame, detector="det"):
    boxes = [
        np.array([b for b, _ in fr], np.float64).reshape(-1, 4) for fr in per_frame
    ]
    scores = [np.array([s for _, s in fr], np.float64) for fr in per_frame]
    return FrameDetections(boxes=boxes, scores=scores, detector=detector)


def _pred(lf):
    return [b for b in lf.bboxes if b.is_predicted]


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10], [0, 0, 5, 10]], np.float64)
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float64)
    np.testing.assert_allclose(iou_matrix(a, b), [[1, 0], [0.5, 0]])
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)


def test_gt_people_crowd_keypointless_and_extent(tmp_path):
    labels, a, c = _labels(tmp_path)
    boxes, insts, kinds = gt_people(labels[0])
    # The crowd box is ignored; the keypoint-less box is a target; the unboxed
    # instance is its keypoint extent; the boxed one is not added twice.
    assert kinds == ["box", "nokpt", "kpt_extent"]
    assert insts[0] is a and insts[1] is None and insts[2] is c
    np.testing.assert_array_equal(boxes[2], [110, 20, 130, 60])
    # A crowd region of category "crowd" (COCO SLP 3.0 files) is not a person box.
    labels[0].bboxes.append(sio.UserBoundingBox(0, 0, 9, 9, category="crowd"))
    assert len(gt_people(labels[0])[0]) == 3
    # Other categories are not people unless asked for (then the person boxes
    # are not GT, so both pose instances are keypoint-extent people).
    assert gt_people(labels[0], category="crowd")[2] == [
        "nokpt",
        "kpt_extent",
        "kpt_extent",
    ]


def test_gt_people_extent_counts_occluded_points_and_skips_empty(tmp_path):
    labels, _, c = _labels(tmp_path)
    c.points["visible"][2] = False  # occluded estimate keeps its location
    sk = labels.skeletons[0]
    empty = sio.Instance.from_numpy(np.full((3, 2), np.nan), skeleton=sk)
    labels[0].instances.append(empty)
    boxes, insts, kinds = gt_people(labels[0])
    np.testing.assert_array_equal(boxes[2], [110, 20, 130, 60])
    assert all(i is not empty for i in insts)


def test_match_links_people_and_leaves_others_unlinked(tmp_path):
    labels, a, c = _labels(tmp_path)
    existing = list(labels[0].bboxes)
    dets = _dets(
        [
            [
                ((9, 9, 33, 50), 0.9),  # person a
                ((52, 12, 79, 68), 0.8),  # the keypoint-less person
                ((111, 21, 131, 58), 0.7),  # keypoint extent of c
                ((90, 70, 140, 100), 0.6),  # in the crowd region: unmatched
                ((10, 10, 30, 50), 0.29),  # below the score threshold
            ],
            [((0, 0, 50, 50), 0.95)],  # a frame with nobody
        ]
    )
    stats = match_detections(labels, dets)
    pred0 = _pred(labels[0])
    assert labels[0].bboxes[: len(existing)] == existing  # same indices
    assert [b.instance for b in pred0] == [a, None, c, None]
    assert [b.score for b in pred0] == [0.9, 0.8, 0.7, 0.6]
    assert pred0[3].metadata["match_iou"] is None
    assert pred0[0].metadata["match_iou"] == pytest.approx(
        iou_matrix(np.array([[8, 8, 32, 52.0]]), np.array([[9, 9, 33, 50.0]]))[0, 0]
    )
    assert [b.instance for b in _pred(labels[1])] == [None]
    assert {b.source for lf in labels for b in _pred(lf)} == {"det"}
    assert stats["people"] == 2 and stats["matched"] == 2
    assert stats["boxes_added"] == 5 and stats["boxes_linked"] == 2
    assert stats["dets_matched_to_kptless_people"] == 1
    assert set(stats["by_gt_box_kind"]) == {"box", "kpt_extent"}


def test_keypointless_person_keeps_its_detection_from_a_neighbour(tmp_path):
    """A detection on a keypoint-less person is not handed to the labeled one."""
    labels, a, _ = _labels(tmp_path)
    # One detection on the keypoint-less person [50, 10, 80, 70] that also
    # overlaps a's box a little: matched to the keypoint-less person.
    dets = _dets([[((30, 10, 80, 70), 0.9)], []])
    stats = match_detections(labels, dets)
    assert _pred(labels[0])[0].instance is None
    assert stats["matched"] == 0 and stats["dets_matched_to_kptless_people"] == 1


def test_min_iou_and_score_threshold_are_configurable(tmp_path):
    labels, a, _ = _labels(tmp_path)
    det = ((8, 8, 32, 14), 0.2)  # IoU 6/44 ~= 0.136 with a's box
    stats = match_detections(labels, _dets([[det], []]), score_threshold=0.1)
    assert stats["matched"] == 1 and _pred(labels[0])[0].instance is a
    labels, a, _ = _labels(tmp_path)
    stats = match_detections(
        labels, _dets([[det], []]), score_threshold=0.1, min_iou=0.2
    )
    assert stats["matched"] == 0 and _pred(labels[0])[0].instance is None
    labels, _, _ = _labels(tmp_path)
    stats = match_detections(labels, _dets([[det], []]))  # 0.2 < 0.3: not stored
    assert stats["boxes_added"] == 0 and not _pred(labels[0])


def test_ties_link_one_detection(tmp_path):
    """Two identical detections of one person: exactly one is linked (the first)."""
    labels, a, _ = _labels(tmp_path)
    stats = match_detections(labels, _dets([[((8, 8, 32, 52), 0.9)] * 2, []]))
    assert [b.instance for b in _pred(labels[0])] == [a, None]
    assert stats["mean_iou_matched"] == 1.0


def test_no_detections(tmp_path):
    labels, _, _ = _labels(tmp_path)
    stats = match_detections(labels, _dets([[], []]))
    assert stats["matched"] == 0 and stats["unmatched"] == 2
    assert stats["recall_iou50"] == 0.0 and stats["boxes_added"] == 0
    assert not any(_pred(lf) for lf in labels)
    with pytest.raises(ValueError, match="detection frames"):
        match_detections(labels, _dets([[]]))


def test_dry_run_counts_without_adding(tmp_path):
    labels, _, _ = _labels(tmp_path)
    stats = match_detections(
        labels, _dets([[((9, 9, 33, 50), 0.9)], []]), add_boxes=False
    )
    assert stats["boxes_added"] == 1 and stats["boxes_linked"] == 1
    assert not _pred(labels[0])


def test_replace_predicted(tmp_path):
    labels, a, _ = _labels(tmp_path)
    labels[0].bboxes.append(
        sio.PredictedBoundingBox(0, 0, 5, 5, category=PERSON, score=0.9, source="old")
    )
    stats = match_detections(
        labels, _dets([[((9, 9, 33, 50), 0.9)], []]), replace_predicted=True
    )
    assert [b.source for b in _pred(labels[0])] == ["det"]
    assert stats["predicted_boxes_removed"] == 1


def test_boxes_share_one_category_and_round_trip(tmp_path):
    labels, a, c = _labels(tmp_path)
    src = tmp_path / "src.slp"
    labels.save(str(src))
    with h5py.File(src, "a") as f:
        f.create_group("human_meta").create_dataset("frame_image_id", data=[1, 2])
    labels = sio.load_slp(str(src), open_videos=False)
    dets = _dets(
        [[((9, 9, 33, 50), 0.9), ((111, 21, 131, 58), 0.7), ((90, 70, 140, 100), 0.6)]]
        + [[((0, 0, 50, 50), 0.95)]]
    )
    stats = match_detections(labels, dets)
    person = next(cat for cat in labels.categories if cat.name == "person")
    assert all(b.category is person for lf in labels for b in _pred(lf))
    out = tmp_path / "src.det.slp"
    stats_path = write_detector_labels(labels, stats, src, out)
    assert json.loads(stats_path.read_text())["matched"] == 2

    back = sio.load_slp(str(out), open_videos=False)
    assert [cat.name for cat in back.categories] == ["person"]
    assert back.provenance["detector_boxes"]["boxes_added"] == 4
    assert back.provenance["detector_boxes_source"] == str(src)
    lf = back[0]
    insts = lf.user_instances
    assert [b.instance for b in _pred(lf)] == [insts[0], insts[1], None]
    assert _pred(back[1])[0].instance is None
    with h5py.File(out) as f:
        np.testing.assert_array_equal(f["human_meta/frame_image_id"][:], [1, 2])
    # What pose training reads: the linked detector boxes, GT for the rest.
    table = build_instance_table(
        [back],
        NODES,
        bbox_categories=["person"],
        crop_bbox_source="detector",
        crop_bbox_detector="det",
        crop_bbox_min_score=0.3,
    )
    assert table.box_sources.get("detector") == 2 and not table.box_sources.get("gt")
    np.testing.assert_allclose(table.box[0], [9, 9, 33, 50])

    with pytest.raises(FileExistsError):
        write_detector_labels(back, stats, src, out)
    with pytest.raises(ValueError, match="overwrite the input"):
        write_detector_labels(back, stats, src, src, overwrite=True)


def test_per_box_categories_are_refused(tmp_path):
    labels, _, _ = _labels(tmp_path)
    src = tmp_path / "src.slp"
    labels.save(str(src))
    for i in range(3):
        labels[1].bboxes.append(
            sio.PredictedBoundingBox(0, 0, 5, 5, category=sio.Category("person"))
        )
    with pytest.raises(ValueError, match="share one"):
        write_detector_labels(labels, {}, src, tmp_path / "out.slp")


def test_coco_detections_by_image_id_and_file_name(tmp_path):
    labels, a, _ = _labels(tmp_path)
    path0 = labels.videos[0].filename[0]
    dets = [
        {"image_id": 1, "category_id": 1, "bbox": [9, 9, 24, 41], "score": 0.9},
        {"image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5], "score": 0.9},
        {"image_id": 7, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.9},
        {"file_name": path0, "bbox": [52, 12, 27, 56], "score": 0.8},
        {"file_name": "elsewhere/000000000002.png", "bbox": [1, 1, 2, 2], "score": 0.5},
    ]
    fd = load_coco_detections(labels, dets, detector="dfine")
    np.testing.assert_array_equal(fd.boxes[0], [[9, 9, 33, 50], [52, 12, 79, 68]])
    np.testing.assert_array_equal(fd.boxes[1], [[1, 1, 3, 3]])  # by base name
    assert fd.info["n_unmatched_detections"] == 1 and fd.detector == "dfine"
    assert len(load_coco_detections(labels, dets, category_id=None).boxes[0]) == 3

    path = tmp_path / "dets.json"
    exact = FrameDetections(
        boxes=[np.array([[0.1, 0.2, 10.3, 20.7]]), np.zeros((0, 4))],
        scores=[np.array([0.123456789]), np.zeros(0)],
        detector="x",
    )
    assert save_coco_detections(labels, exact, path) == 1
    back = load_coco_detections(labels, path)
    np.testing.assert_array_equal(back.boxes[0], exact.boxes[0])  # bit-exact
    np.testing.assert_array_equal(back.scores[0], exact.scores[0])
    assert back.detector == "dets"


def test_detection_chunks(tmp_path):
    labels, a, _ = _labels(tmp_path)
    paths = list(labels.videos[0].filename)
    listing = tmp_path / "list.txt"
    listing.write_text("\n".join(paths[::-1]) + "\n")  # list order != frame order
    d = tmp_path / "chunks"
    d.mkdir()
    common = dict(model=np.array("org/dfine"))
    np.savez(
        d / "chunk_00000.npz",
        img_idx=np.array([1, 0, 1], np.int32),
        boxes=np.array([[9, 9, 33, 50], [0, 0, 5, 5], [52, 12, 79, 68]], np.float32),
        scores=np.array([0.9, 0.5, 0.8], np.float32),
        image_idx=np.array([0], np.int32),
        image_hw=np.array([[120, 160]], np.int32),
        **common,
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_detection_chunks(labels, d, listing)
    np.savez(
        d / "chunk_00001.npz",
        img_idx=np.zeros(0, np.int32),
        boxes=np.zeros((0, 4), np.float32),
        scores=np.zeros(0, np.float32),
        image_idx=np.array([1], np.int32),
        image_hw=np.array([[120, 160]], np.int32),
        **common,
    )
    fd = load_detection_chunks(labels, d, listing)
    assert fd.detector == "org/dfine" and fd.image_hw == [(120, 160)] * 2
    np.testing.assert_array_equal(fd.boxes[0], [[9, 9, 33, 50], [52, 12, 79, 68]])
    np.testing.assert_array_equal(fd.boxes[1], [[0, 0, 5, 5]])
    stats = match_detections(labels, fd)
    assert stats["matched"] == 1 and stats["people_kpts_outside_image"] == 0
    listing.write_text(paths[0] + "\n")
    with pytest.raises(ValueError):
        load_detection_chunks(labels, d, listing)


def test_tile_starts_and_merge():
    spec = TileSpec()
    xs = spec.starts(3760)
    assert len(xs) == 9 and xs[0] == 0 and xs[-1] == 3120 and xs[1] == 390
    assert spec.starts(500) == [0]
    boxes = np.array(
        [
            [100, 0, 200, 300],  # a person, wholly inside a tile
            [102, 0, 201, 300],  # duplicate from the neighbouring tile
            [150, 0, 199, 300],  # truncated at an inner tile edge, inside the first
            [400, 0, 500, 300],  # someone else
        ],
        np.float64,
    )
    scores = np.array([0.8, 0.9, 0.95, 0.5])
    edge = np.array([False, False, True, False])
    keep = merge_tile_detections(boxes, scores, edge)
    np.testing.assert_array_equal(keep, [1, 3])  # by score; edge box suppressed
    keep = merge_tile_detections(boxes, scores, np.zeros(4, bool), contain=1.0)
    np.testing.assert_array_equal(keep, [2, 1, 3])
    assert len(merge_tile_detections(boxes, scores, edge, max_detections=1)) == 1


class _CenterBox(torch.nn.Module):
    """A bbox_detr stand-in: one person box over the middle half of each input."""

    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, x):
        self.shapes.append(tuple(x.shape))
        b = x.shape[0]
        logits = torch.full((b, 2, 3), -10.0)
        logits[:, 0, 1] = 10.0  # class 1 = "person"
        logits[:, 1, 0] = 5.0  # class 0 = "dog"
        boxes = torch.tensor([0.5, 0.5, 0.5, 0.5]).expand(b, 2, 4)
        return {"pred_logits": logits, "pred_boxes": boxes}


def _layer(model):
    from sleap_nn.inference.bbox import BBoxDecoder
    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.bbox import BBoxLayer
    from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig

    return BBoxLayer(
        backend=TorchBackend(model=model, device="cpu"),
        decoder=BBoxDecoder("bbox_detr", ["dog", "person"], score_threshold=0.01),
        output_stride=16,
        max_stride=16,
        preprocess_config=PreprocessConfig(
            max_height=64, max_width=64, resize_mode="squash"
        ),
        postprocess_config=PostprocessConfig(peak_threshold=0.01, max_instances=10),
    )


def test_detect_with_layer(tmp_path):
    labels, _, _ = _labels(tmp_path)
    model = _CenterBox()
    fd = detect_with_layer(labels, _layer(model), detector="m", batch_size=2)
    assert model.shapes[0][-2:] == (64, 64)
    assert fd.image_hw == [(120, 160)] * 2 and fd.detector == "m"
    for boxes, scores in zip(fd.boxes, fd.scores):
        np.testing.assert_allclose(boxes, [[40, 30, 120, 90]], atol=1e-4)
        assert scores[0] > 0.99  # only the person class is kept
    with pytest.raises(ValueError, match="no 'cat' class"):
        detect_with_layer(labels, _layer(model), detector="m", category="cat")


def test_detect_with_layer_groups_mixed_sizes(tmp_path):
    """Mixed-size frames are batched by size; results stay on their own frames."""
    sizes = [(120, 160), (60, 80), (120, 160), (60, 80)]
    paths = []
    for i, hw in enumerate(sizes):
        p = tmp_path / f"{i:03d}.png"
        cv2.imwrite(str(p), np.zeros((*hw, 3), np.uint8))
        paths.append(str(p))
    video = sio.Video.from_filename(paths)
    frames = [sio.LabeledFrame(video=video, frame_idx=i) for i in range(4)]
    labels = sio.Labels(frames, skeletons=[sio.Skeleton(NODES)], videos=[video])
    model = _CenterBox()
    fd = detect_with_layer(labels, _layer(model), detector="m", batch_size=2)
    assert [s[0] for s in model.shapes] == [2, 2]  # two full batches, not four
    assert labels.labeled_frames == frames
    for (h, w), boxes in zip(sizes, fd.boxes):
        np.testing.assert_allclose(
            boxes, [[w / 4, h / 4, 3 * w / 4, 3 * h / 4]], atol=1e-4
        )
    assert fd.image_hw == sizes
    model = _CenterBox()
    detect_with_layer(
        labels, _layer(model), detector="m", batch_size=2, order_by_size=False
    )
    assert [s[0] for s in model.shapes] == [1, 1, 1, 1]


def test_detect_with_layer_tiled(tmp_path):
    sk = sio.Skeleton(nodes=list(NODES))
    video = sio.Video.from_filename(_write_images(tmp_path, 1, hw=(48, 400)))
    labels = sio.Labels(
        [sio.LabeledFrame(video=video, frame_idx=0)], skeletons=[sk], videos=[video]
    )
    model = _CenterBox()
    spec = TileSpec(width=160, stride=120)
    fd = detect_with_layer(labels, _layer(model), detector="m", tiling=spec)
    assert spec.starts(400) == [0, 120, 240]
    assert sum(s[0] for s in model.shapes) == 3  # one forward row per tile
    # Each tile's middle half, shifted to frame coordinates; none overlap.
    np.testing.assert_allclose(
        fd.boxes[0][np.argsort(fd.boxes[0][:, 0])],
        [[40, 12, 120, 36], [160, 12, 240, 36], [280, 12, 360, 36]],
        atol=1e-4,
    )


def test_detect_boxes_model_path(tmp_path):
    labels, a, _ = _labels(tmp_path)
    src = tmp_path / "train.slp"
    labels.save(str(src))
    stats, out = detect_boxes(
        src,
        model=tmp_path / "my-detector",
        layer_factory=lambda *args: _layer(_CenterBox()),
        save_detections=tmp_path / "raw.json",
    )
    assert out == tmp_path / "train.bboxdetr.slp" and out.exists()
    assert stats["boxes_added"] == 2 and stats["dataset"] == "train"
    back = sio.load_slp(str(out), open_videos=False)
    assert {b.source for lf in back for b in _pred(lf)} == {"my-detector"}
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert len(raw) == 2 and raw[0]["image_id"] == 1


def test_detect_boxes_argument_checks(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        detect_boxes(tmp_path / "a.slp")
    with pytest.raises(ValueError, match="image_list"):
        detect_boxes(tmp_path / "a.slp", detection_chunks=tmp_path)
    with pytest.raises(ValueError, match="model run only"):
        detect_boxes(tmp_path / "a.slp", detections="d.json", tiling=TileSpec())
    assert default_output_path("x/coco.slp", "dfinex").as_posix() == (
        "x/coco.dfinex.slp"
    )


def test_cli_detect_boxes(tmp_path):
    from sleap_nn.cli import cli

    labels, _, _ = _labels(tmp_path)
    src = tmp_path / "train.slp"
    labels.save(str(src))
    dets = tmp_path / "dfine.json"
    dets.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 1, "bbox": [9, 9, 24, 41], "score": 0.9},
                {"image_id": 2, "category_id": 1, "bbox": [0, 0, 9, 9], "score": 0.4},
            ]
        )
    )
    runner = CliRunner()
    args = ["detect-boxes", "-i", str(src), "--detections", str(dets)]
    res = runner.invoke(cli, args + ["--dry_run"])
    assert res.exit_code == 0, res.output
    assert "matched 1" in res.output and not (tmp_path / "train.dets.slp").exists()

    res = runner.invoke(cli, args + ["--tag", "dfinex", "--score_threshold", "0.5"])
    assert res.exit_code == 0, res.output
    out = tmp_path / "train.dfinex.slp"
    stats = json.loads((tmp_path / "train.dfinex.json").read_text())
    assert stats["boxes_added"] == 1 and stats["det_thr"] == 0.5
    assert stats["detector"] == "dfine"
    assert sio.load_slp(str(out), open_videos=False).provenance["detector_boxes"]

    res = runner.invoke(cli, args + ["--tag", "dfinex"])
    assert res.exit_code != 0 and "exists" in res.output
    res = runner.invoke(cli, ["detect-boxes", "-i", str(src)])
    assert res.exit_code != 0 and "exactly one" in res.output
