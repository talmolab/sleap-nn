"""Unit tests for bbox_detr dataset, export, rescoring, and distillation fixes."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Negative/empty primary frames under zoom_out_canvas="frame"
# ─────────────────────────────────────────────────────────────────────────────


def test_detr_bbox_dataset_negative_frame_stays_empty_under_frame_canvas(monkeypatch):
    """Empty primary frames zoom onto a black canvas instead of pasting onto positive frames."""
    from sleap_nn.data import custom_datasets
    from sleap_nn.data.custom_datasets import DetrBBoxDataset

    ds = DetrBBoxDataset.__new__(DetrBBoxDataset)
    ds.ensure_rgb = False
    ds.ensure_grayscale = False
    ds.intensity_aug = None
    ds.geometric_aug = None
    ds.max_stride = 1
    ds.apply_aug = True
    ds.detr_aug = {
        "crop_p": 0.0,
        "shortest_sides": [64],
        "crop_range": [32, 64],
        "flip_p": 0.0,
        "vflip_p": 0.0,
        "rot90_p": 0.0,
        "zoom_out_p": 1.0,
        "zoom_out_min": 0.5,
        "zoom_out_max": 0.8,
        "zoom_out_canvas": "frame",
        "paste_p": 0.0,
    }
    ds.train_hw = (64, 64)
    ds.resize_mode = "letterbox"
    ds._kpt_names = None
    ds._kpt_flip_order = ()
    ds._capture_hulls = False

    seen_zoom_out_p = []
    real_detr_train_transform = custom_datasets.detr_train_transform

    def _spy_detr_train_transform(*args, **kwargs):
        seen_zoom_out_p.append(kwargs.get("zoom_out_p"))
        return real_detr_train_transform(*args, **kwargs)

    monkeypatch.setattr(
        custom_datasets, "detr_train_transform", _spy_detr_train_transform
    )

    zoom_on_frame_mock = MagicMock(
        side_effect=AssertionError("_zoom_out_on_frame must not run for empty frames")
    )
    ds._zoom_out_on_frame = zoom_on_frame_mock

    empty_sample = {
        "image": torch.full((1, 3, 64, 64), 128, dtype=torch.uint8),
        "instances": torch.empty((1, 0, 4, 2), dtype=torch.float32),
        "_classes": torch.empty((0,), dtype=torch.long),
    }
    out = ds._apply_common_preprocessing(empty_sample)
    zoom_on_frame_mock.assert_not_called()
    assert seen_zoom_out_p == [1.0]
    assert out["instances"].shape == (1, 0, 4, 2)
    assert out["_classes"].shape == (0,)
    # Black canvas padding appeared from zero-canvas zoom-out on the empty frame.
    assert (out["image"] == 0).any()

    # Positive frame with boxes DOES use _zoom_out_on_frame when zoom_out_canvas="frame".
    called = []

    def _fake_zoom_out_on_frame(img, boxes, classes, zo_range, *pts):
        called.append((boxes.shape[0], zo_range))
        return (img, boxes, classes, *pts)

    ds._zoom_out_on_frame = _fake_zoom_out_on_frame
    pos_corners = torch.tensor(
        [[[[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]]],
        dtype=torch.float32,
    )
    pos_sample = {
        "image": torch.full((1, 3, 64, 64), 128, dtype=torch.uint8),
        "instances": pos_corners,
        "_classes": torch.tensor([0], dtype=torch.long),
    }
    out_pos = ds._apply_common_preprocessing(pos_sample)
    assert seen_zoom_out_p == [1.0, 0.0]
    assert len(called) == 1
    assert called[0][0] == 1
    assert out_pos["instances"].shape[1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 4-Keypoint Export + Native vis_mult + Canonical Keypoint Propagation
# ─────────────────────────────────────────────────────────────────────────────


class _DummyDetrModel(nn.Module):
    """Minimal bbox_detr stand-in with optional 4-keypoint branch."""

    def __init__(self, num_keypoints: int = 4) -> None:
        super().__init__()
        self.detr = nn.Module()
        self.detr.num_keypoints = num_keypoints

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        out = {
            "pred_logits": torch.zeros(b, 5, 2, device=x.device),
            "pred_boxes": torch.full((b, 5, 4), 0.5, device=x.device),
        }
        if self.detr.num_keypoints:
            out["pred_keypoints"] = torch.full(
                (b, 5, self.detr.num_keypoints, 2), 0.5, device=x.device
            )
            out["pred_keypoint_logits"] = torch.zeros(
                b, 5, self.detr.num_keypoints, device=x.device
            )
        return out


def test_bbox_onnx_wrapper_4_tensor_keypoint_outputs_and_cli_cleanup():
    """BBoxONNXWrapper exports 4 tensors when bbox_detr has keypoints and 2 otherwise."""
    import sleap_nn.export.cli as export_cli
    from sleap_nn.export.wrappers.bbox import BBoxONNXWrapper

    assert not hasattr(export_cli, "_has_detr_keypoints")

    w_kpt = BBoxONNXWrapper(_DummyDetrModel(num_keypoints=4), model_type="bbox_detr")
    assert w_kpt.has_keypoints is True
    assert w_kpt.output_names == [
        "pred_logits",
        "pred_boxes",
        "pred_keypoints",
        "pred_keypoint_logits",
    ]
    out_kpt = w_kpt(torch.zeros(2, 3, 32, 32, dtype=torch.float32))
    assert list(out_kpt.keys()) == w_kpt.output_names
    assert out_kpt["pred_keypoints"].shape == (2, 5, 4, 2)
    assert out_kpt["pred_keypoint_logits"].shape == (2, 5, 4)

    w_box = BBoxONNXWrapper(_DummyDetrModel(num_keypoints=0), model_type="bbox_detr")
    assert w_box.has_keypoints is False
    assert w_box.output_names == ["pred_logits", "pred_boxes"]
    out_box = w_box(torch.zeros(2, 3, 32, 32, dtype=torch.float32))
    assert list(out_box.keys()) == ["pred_logits", "pred_boxes"]


def test_decode_set_and_decode_set_keypoints_vis_mult():
    """Optional bbox_vis_beta rescores DETR boxes while preserving raw keypoint vis."""
    from sleap_nn.inference.bbox import (
        BBoxDecoder,
        decode_set,
        decode_set_keypoints,
    )

    # Query 0: high p_det (sigmoid(2.0)~0.8808), low vis (sigmoid(-3.0)~0.0474)
    # Query 1: moderate p_det (sigmoid(1.0)~0.7311), high vis (sigmoid(3.0)~0.9526)
    pred_logits = torch.tensor([[[2.0, -5.0], [1.0, -5.0]]])
    pred_boxes = torch.tensor([[[0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]]])
    pred_kpts = torch.tensor(
        [
            [
                [[0.2, 0.2], [0.22, 0.22], [0.25, 0.25], [0.28, 0.28]],
                [[0.7, 0.7], [0.72, 0.72], [0.75, 0.75], [0.78, 0.78]],
            ]
        ]
    )
    pred_kpt_logits = torch.tensor([[[-3.0, -3.0, -3.0, -3.0], [3.0, 3.0, 3.0, 3.0]]])

    p0 = torch.sigmoid(torch.tensor(2.0)).item()
    p1 = torch.sigmoid(torch.tensor(1.0)).item()
    v0 = torch.sigmoid(torch.tensor(-3.0)).item()
    v1 = torch.sigmoid(torch.tensor(3.0)).item()

    # Default bbox_vis_beta=0.0: Query 0 ranks first (100% backward compatible).
    b_raw, s_raw, c_raw = decode_set(
        pred_logits,
        pred_boxes,
        (100, 100),
        score_threshold=0.0,
        max_detections=5,
        pred_keypoint_logits=pred_kpt_logits,
        bbox_vis_beta=0.0,
    )
    torch.testing.assert_close(s_raw[0, :2], torch.tensor([p0, p1]))

    # With bbox_vis_beta=1.0: Query 1 overtakes Query 0, and Query 0 falls below 0.1.
    for beta in (0.5, 1.0):
        expected_q0 = p0 * (v0**beta)
        expected_q1 = p1 * (v1**beta)
        b_vm, s_vm, _ = decode_set(
            pred_logits,
            pred_boxes,
            (100, 100),
            score_threshold=0.0,
            max_detections=5,
            pred_keypoint_logits=pred_kpt_logits,
            bbox_vis_beta=beta,
        )
        torch.testing.assert_close(
            s_vm[0, :2], torch.tensor([expected_q1, expected_q0]), atol=1e-6, rtol=1e-6
        )
        # decode_set_keypoints matches decode_set scores & preserves raw vis in point_scores
        kb, ks, kc, kp, kv = decode_set_keypoints(
            pred_logits,
            pred_boxes,
            pred_kpts,
            pred_kpt_logits,
            (100, 100),
            score_threshold=0.0,
            max_detections=5,
            visibility_threshold=0.0,
            bbox_vis_beta=beta,
        )
        torch.testing.assert_close(ks, s_vm, equal_nan=True)
        torch.testing.assert_close(kb, b_vm, equal_nan=True)
        torch.testing.assert_close(kv[0, 0], torch.full((4,), v1), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(kv[0, 1], torch.full((4,), v0), atol=1e-6, rtol=1e-6)

    # Thresholding operates on rescored score: Query 0 (~0.042 at beta=1) is dropped at thr=0.1.
    dec = BBoxDecoder(
        "bbox_detr",
        categories=["mouse"],
        score_threshold=0.1,
        max_detections=5,
        num_keypoints=4,
        bbox_vis_beta=1.0,
    )
    raw = {
        "pred_logits": pred_logits,
        "pred_boxes": pred_boxes,
        "pred_keypoints": pred_kpts,
        "pred_keypoint_logits": pred_kpt_logits,
    }
    _, s_thr, _ = dec(raw, (100, 100))
    assert torch.isfinite(s_thr[0]).sum().item() == 1
    torch.testing.assert_close(s_thr[0, 0], torch.tensor(p1 * v1), atol=1e-6, rtol=1e-6)

    # Negative beta rejected cleanly
    with pytest.raises(ValueError, match="bbox_vis_beta"):
        decode_set(pred_logits, pred_boxes, (100, 100), bbox_vis_beta=-0.5)
    with pytest.raises(ValueError, match="bbox_vis_beta"):
        BBoxDecoder("bbox_detr", categories=["mouse"], bbox_vis_beta=-1.0)


def test_canonical_bbox_keypoints_propagate_through_stage1_and_topdown():
    """Stage-1 canonical keypoints propagate alongside Stage-2 pose in TopDownLayer."""
    from sleap_nn.inference.bbox import BBoxDecoder
    from sleap_nn.inference.layers.backends import TorchBackend
    from sleap_nn.inference.layers.bbox import BBoxLayer, BBoxStageLayer
    from sleap_nn.inference.layers.configs import PostprocessConfig, PreprocessConfig
    from sleap_nn.inference.layers.topdown import TopDownLayer
    from sleap_nn.inference.outputs import Outputs

    class _DetNet(nn.Module):
        def forward(self, x):
            b = x.shape[0]
            return {
                "pred_logits": torch.tensor([[[3.0, -5.0]]]).expand(b, 1, 2).clone(),
                "pred_boxes": torch.tensor([[[0.5, 0.5, 0.4, 0.4]]])
                .expand(b, 1, 4)
                .clone(),
                "pred_keypoints": torch.tensor(
                    [[[[0.4, 0.4], [0.45, 0.45], [0.55, 0.45], [0.6, 0.6]]]]
                )
                .expand(b, 1, 4, 2)
                .clone(),
                "pred_keypoint_logits": torch.full((b, 1, 4), 2.0),
            }

    class _Stage2Stub:
        use_gt_peaks = False

        def predict(self, crops):
            n = crops.shape[0]
            return Outputs(
                pred_keypoints=torch.full((n, 1, 6, 2), 16.0),
                pred_peak_values=torch.full((n, 1, 6), 0.9),
            )

    decoder = BBoxDecoder(
        "bbox_detr",
        categories=["mouse"],
        score_threshold=0.1,
        max_detections=4,
        num_keypoints=4,
        visibility_threshold=0.5,
        bbox_vis_beta=1.0,
    )
    img = torch.zeros(2, 3, 64, 64, dtype=torch.float32)

    # Standalone BBoxLayer preserves both pred_keypoints and pred_bbox_keypoints.
    standalone = BBoxLayer(
        backend=TorchBackend(_DetNet(), device="cpu"),
        decoder=decoder,
        output_stride=16,
        preprocess_config=PreprocessConfig(),
        postprocess_config=PostprocessConfig(peak_threshold=0.1, max_instances=4),
    )
    out1 = standalone.predict(img)
    assert out1.pred_bbox_keypoints is not None
    assert out1.pred_bbox_keypoints.shape == (2, 4, 4, 2)
    assert out1.pred_bbox_keypoint_vis.shape == (2, 4, 4)
    torch.testing.assert_close(
        out1.pred_bbox_keypoints, out1.pred_keypoints, equal_nan=True
    )

    # Two-stage TopDownLayer keeps Stage-2 pose (6 nodes) in pred_keypoints AND
    # preserves Stage-1 canonical keypoints (4 nodes) in pred_bbox_keypoints!
    stage1 = BBoxStageLayer(
        backend=TorchBackend(_DetNet(), device="cpu"),
        decoder=decoder,
        output_stride=16,
        preprocess_config=PreprocessConfig(),
        postprocess_config=PostprocessConfig(peak_threshold=0.1, max_instances=4),
    )
    td = TopDownLayer(
        centroid_layer=stage1,
        centered_instance_layer=_Stage2Stub(),
        crop_size=(32, 32),
    )
    out2 = td.predict(img)
    assert out2.pred_keypoints.shape == (2, 4, 6, 2)  # Stage-2 body pose
    assert out2.pred_bbox_keypoints is not None
    assert out2.pred_bbox_keypoints.shape == (2, 4, 4, 2)  # Stage-1 canonical kpts
    assert out2.pred_bbox_keypoint_vis.shape == (2, 4, 4)
    # Valid slot 0 has finite canonical keypoints; padded slots 1..3 are NaN.
    assert torch.isfinite(out2.pred_bbox_keypoints[:, 0]).all()
    assert torch.isnan(out2.pred_bbox_keypoints[:, 1:]).all()
    # Outputs.slim() keeps canonical keypoints pickle-safe on CPU
    slim = out2.slim()
    assert slim.pred_bbox_keypoints is not None
    assert slim.pred_bbox_keypoints.device.type == "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Distillation Telemetry (training_log.csv) & canonical_node_indices
# ─────────────────────────────────────────────────────────────────────────────


def test_distillation_mask_unlabeled_nodes_and_canonical_node_indices(
    minimal_instance_centered_instance_ckpt,
):
    """canonical_node_indices keeps specified shared nodes unmasked during distillation."""
    from omegaconf import OmegaConf

    from sleap_nn.config.trainer_config import DistillationConfig
    from sleap_nn.training.distillation import attach_distillation
    from sleap_nn.training.lightning_modules import LightningModel

    # Config defaults & validation
    assert DistillationConfig(teacher="t").mask_unlabeled_nodes is True
    assert DistillationConfig(teacher="t").canonical_node_indices is None
    assert DistillationConfig(
        teacher="t", canonical_node_indices=[0, 2]
    ).canonical_node_indices == [0, 2]
    for bad in ([-1], [True], "0"):
        with pytest.raises(ValueError, match="canonical_node_indices"):
            DistillationConfig(teacher="t", canonical_node_indices=bad)

    ckpt = minimal_instance_centered_instance_ckpt
    cfg = OmegaConf.load(ckpt / "training_config.yaml")
    # Teacher has ["A", "B"]; student has ["B", "C", "A"] -> shared student_idx=[0, 2], teacher_idx=[1, 0]
    cfg.model_config.head_configs.centered_instance.confmaps.part_names = [
        "B",
        "C",
        "A",
    ]
    cfg.model_config.backbone_config.unet.filters = 8
    torch.manual_seed(0)
    student = LightningModel.get_lightning_model_from_config(cfg)

    x = torch.rand(2, 1, 96, 96)
    out, bb = student.model(x, return_backbone_outputs=True)

    # Sample 0 has only node 0 ("B") labeled; node 2 ("A") unlabeled.
    # Sample 1 has only node 2 ("A") labeled; node 0 ("B") unlabeled.
    w = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    d_masked = attach_distillation(
        student, {"teacher": str(ckpt), "alpha": 0.0, "mask_unlabeled_nodes": True}
    )
    t = d_masked.teacher(x)["CenteredInstanceConfmapsHead"][:, [1, 0]]
    s = out["CenteredInstanceConfmapsHead"][:, [0, 2]]
    per = ((s - t) ** 2).mean(
        dim=(2, 3)
    )  # (2, 2): [:, 0]=node "B"(idx 0), [:, 1]=node "A"(idx 2)

    # Default mask_unlabeled_nodes=True, canonical_node_indices=None: only labeled (0,0) and (1,1) count
    loss_masked = d_masked.loss_terms(x, out, bb, {}, w)["heatmap"]
    torch.testing.assert_close(loss_masked, (per[0, 0] + per[1, 1]) / 2.0)

    # With canonical_node_indices=[2]: node "A" (student idx 2, shared pos 1) is ALWAYS distilled
    # (weight=1 on both samples), while non-canonical node "B" (student idx 0) stays masked on sample 1!
    d_canon = attach_distillation(
        student,
        {
            "teacher": str(ckpt),
            "alpha": 0.0,
            "mask_unlabeled_nodes": True,
            "canonical_node_indices": [2],
        },
    )
    loss_canon = d_canon.loss_terms(x, out, bb, {}, w)["heatmap"]
    torch.testing.assert_close(loss_canon, (per[0, 0] + per[0, 1] + per[1, 1]) / 3.0)


def test_distillation_logs_telemetry_columns_to_training_log_csv(
    config, tmp_path, minimal_instance_centered_instance_ckpt
):
    """Distillation writes train/distill_*_loss telemetry columns into training_log.csv."""
    import csv

    from omegaconf import OmegaConf

    from sleap_nn.training.model_trainer import ModelTrainer

    ckpt = minimal_instance_centered_instance_ckpt
    t_cfg = OmegaConf.load(ckpt / "training_config.yaml")
    cfg = config.copy()
    cfg.data_config.preprocessing.crop_size = 96
    cfg.model_config.head_configs.centered_instance = OmegaConf.create(
        OmegaConf.to_container(t_cfg.model_config.head_configs.centered_instance)
    )
    cfg.model_config.head_configs.single_instance = None
    cfg.model_config.backbone_config.unet.filters = 8
    cfg.model_config.backbone_config.unet.max_stride = 8
    cfg.model_config.backbone_config.unet.output_stride = 2
    cfg.trainer_config.max_epochs = 1
    cfg.trainer_config.min_train_steps_per_epoch = 2
    cfg.trainer_config.save_ckpt = True
    cfg.trainer_config.ckpt_dir = str(tmp_path)
    cfg.trainer_config.run_name = "kd_csv"
    cfg.trainer_config.trainer_accelerator = "cpu"
    cfg.trainer_config.distillation = {
        "teacher": str(ckpt),
        "alpha": 0.5,
        "mask_unlabeled_nodes": True,
    }

    trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    trainer.train()

    csv_path = tmp_path / "kd_csv" / "training_log.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 1
    row = rows[0]
    for col in (
        "train/distill_sup_loss",
        "train/distill_heatmap_loss",
        "train/distill_feat_loss",
        "train/distill_weight",
    ):
        assert col in row, f"Missing {col} in training_log.csv headers {list(row)}"
        val = float(row[col])
        assert val == val and val >= 0.0, f"Invalid value for {col}: {row[col]}"
