"""``bbox_detr`` with both experimental options: keypoints and ignore boxes.

Ignore boxes (``data_config.ignore_box_category``) ride the DETR data path with the
animal boxes and are split off before the set targets; the keypoint branch carries
one row of points per box. These tests check that the two stay aligned: the
keypoint rows of the set targets are exactly the animal boxes' rows, through the
training augmentations, and that a model trained with both predicts instances
with points linked to their boxes and never an ignore box.
"""

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf

from sleap_nn.data.bboxes import SetTargetEncoder, collate_set_targets
from tests.test_bbox_detr_keypoints import (
    NAMES,
    _blob_labels,
    _check_sample,
    _kpt_config,
)

#: An ignore region in an empty corner of the blob frames (128 x 96).
IGNORE_BOX = (110.0, 2.0, 126.0, 14.0)


def _blob_labels_with_ignore(tmp_path):
    labels = _blob_labels(tmp_path)
    for lf in labels:
        lf.bboxes.append(sio.UserBoundingBox.from_xyxy(*IGNORE_BOX, category="ignore"))
    return labels


AUG_CASES = {
    "plain": dict(),
    "flips_rot90": dict(flip_p=0.5, vflip_p=0.5, rot90_p=0.5),
    "crop": dict(crop_p=1.0, flip_p=0.5),
    "zoom_out_frame": dict(
        zoom_out_p=1.0, zoom_out_min=0.5, zoom_out_max=0.8, zoom_out_canvas="frame"
    ),
    # Pasted animals no smaller than 0.6x and not overlapping others, so each blob
    # stays apart from the other animals' blobs of its channel.
    "copy_paste": dict(
        paste_p=1.0,
        paste_max=3,
        paste_dilate=0.6,
        paste_feather=0.0,
        paste_scale_min=0.6,
        paste_overlap=0.0,
        flip_p=0.5,
    ),
}
#: Blob-centroid tolerance (px): a crop can cut a blob at the frame edge.
TOLERANCE = {"crop": 2.5}


@pytest.mark.parametrize("case", sorted(AUG_CASES))
def test_keypoint_rows_follow_animal_boxes_with_ignore_boxes(tmp_path, case):
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    labels = _blob_labels_with_ignore(tmp_path)
    ds = DetrBBoxDataset(
        labels=[labels],
        target_encoder=SetTargetEncoder(),
        categories=["object"],
        max_stride=32,
        ensure_rgb=True,
        apply_aug=True,
        max_hw=(96, 128),
        resize_mode="letterbox",
        detr_aug={
            "crop_p": 0.0,
            "shortest_sides": [96],
            "crop_range": [64, 96],
            "flip_p": 0.0,
            **AUG_CASES[case],
        },
        ignore_box_category="ignore",
        keypoints={"part_names": NAMES, "supervise_occluded": False},
    )
    # The ignore box is a box of every frame, with no points and no weight.
    for i in range(len(ds)):
        entry = ds.lf_idx_list[i]
        assert entry["bbox_classes"].tolist()[-1] == -2
        assert entry["kpts"].shape[0] == entry["bboxes"].shape[0] == 3
        assert np.isnan(entry["kpts"][-1, :, :2]).all()
        assert (entry["kpts"][-1, :, 2] == 0).all()
    assert ds._kpt_box_counts == [[6, 6, 6]]  # ignore boxes are not counted

    ignores = targets = 0
    for seed in range(8):
        torch.manual_seed(seed)
        sample = ds[seed % len(ds)]
        n = int(sample["set_count"])
        ignores += int(sample["num_ignore"])
        # Every animal target has its own pose (all animals are posed, no ignore
        # box became a keypoint target) and its points sit on its own blobs.
        w = sample["set_keypoint_weights"]
        assert (w[:n].sum(-1) > 0).all() and (w[n:] == 0).all()
        t, _ = _check_sample(sample, tol=TOLERANCE.get(case, 1.6))
        targets += t
        # The points of a target are never those of the ignore box's row.
        batch = torch.utils.data.default_collate([sample])
        (target,) = collate_set_targets(batch)
        assert len(target["keypoints"]) == len(target["boxes"]) == n
        assert len(target["ignore_boxes"]) == int(sample["num_ignore"])
    assert targets >= 8 and ignores >= 1


def _ignore_file(minimal_instance, tmp_path):
    """``minimal_instance`` with an ignore box in an empty corner of its frame."""
    labels = sio.load_slp(minimal_instance.as_posix())
    labels[0].bboxes.append(
        sio.UserBoundingBox.from_xyxy(10, 10, 60, 60, category="ignore")
    )
    path = tmp_path / "with_ignore.pkg.slp"
    labels.save(path.as_posix(), embed=True)
    return path.as_posix()


def test_train_and_predict_with_keypoints_and_ignore_boxes(
    config, minimal_instance, tmp_path, monkeypatch
):
    import sleap_nn.training.lightning_modules as lightning_modules
    from sleap_nn.inference.layers.bbox import BBoxLayer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.training.model_trainer import ModelTrainer

    # Spy on the loss targets: (boxes, ignore boxes, keypoint rows) per image.
    seen = []
    collate = lightning_modules.collate_set_targets

    def spy(batch):
        targets = collate(batch)
        seen.extend(
            (len(t["boxes"]), len(t.get("ignore_boxes", ())), len(t["keypoints"]))
            for t in targets
        )
        return targets

    monkeypatch.setattr(lightning_modules, "collate_set_targets", spy)
    path = _ignore_file(minimal_instance, tmp_path)
    cfg = _kpt_config(config, tmp_path, run_name="kpt_ignore", flip_p=0.5, rot90_p=0.5)
    cfg.data_config.train_labels_path = [path]
    cfg.data_config.val_labels_path = [path]
    cfg.data_config.ignore_box_category = "ignore"
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.visualize_preds_during_training = False
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()
    logged = trainer.trainer.callback_metrics
    for term in ("loss_keypoints", "loss_keypoint_vis", "loss_ce"):
        assert torch.isfinite(logged[f"train/{term}"])
    # The ignore box reached the criterion as an ignore box, and the keypoint rows
    # are the animal boxes' rows.
    assert seen and all(n == k for n, _, k in seen)
    assert any(m > 0 for _, m, _ in seen), "no ignore box reached the loss"

    run_dir = tmp_path / "kpt_ignore"
    saved = OmegaConf.load(run_dir / "training_config.yaml")
    detr = saved.model_config.head_configs.bbox_detr.detr
    assert list(detr.categories) == ["object"]  # the ignore category is no class
    assert list(detr.keypoints.part_names) == ["A", "B"]
    assert saved.data_config.ignore_box_category == "ignore"

    pred = Predictor.from_model_paths(
        [run_dir.as_posix()], device="cpu", peak_threshold=0.0
    )
    assert isinstance(pred.layer, BBoxLayer)
    assert pred.layer.decoder.categories == ["object"]
    pred.layer.decoder.visibility_threshold = 0.0  # untrained: keep every point
    labels = pred.predict(minimal_instance.as_posix())
    lf = labels[0]
    assert lf.instances and len(lf.instances) == len(lf.bboxes)
    for inst, box in zip(lf.instances, lf.bboxes):
        assert box.instance is inst and inst.numpy().shape == (2, 2)
        assert box.category is None  # never the ignore category
