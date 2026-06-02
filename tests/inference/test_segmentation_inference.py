"""Inference-pipeline tests for bottom-up instance segmentation (PR 3).

The headline is a *deterministic GT roundtrip*: feed ground-truth maps produced
by the GT generators through the inference grouping / ``SegmentationLayer`` and
confirm the recovered masks match the input masks (validates the coordinate
conventions + the GT-generation <-> inference contract without needing a trained
model). A lighter train -> predict test exercises the full
``loaders -> SegmentationLayer -> Predictor -> sio.Labels`` plumbing.
"""

import numpy as np
import torch

import sleap_io as sio

from sleap_nn.data.segmentation_maps import (
    generate_center_heatmap,
    generate_center_offsets,
    generate_foreground_mask,
)
from sleap_nn.inference.layers.configs import PostprocessConfig
from sleap_nn.inference.layers.segmentation import SegmentationLayer
from sleap_nn.inference.preprocess_info import PreprocInfo
from sleap_nn.inference.segmentation import (
    find_center_peaks,
    group_instances_from_offsets,
)


def _disk(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2


# ---------------------------------------------------------------------------
# find_center_peaks — plateau robustness
# ---------------------------------------------------------------------------


def test_find_center_peaks_handles_plateau():
    """A 2-pixel max plateau still yields exactly one peak (strict NMS drops it)."""
    hm = torch.zeros(1, 1, 16, 16)
    hm[0, 0, 8, 8] = 1.0
    hm[0, 0, 9, 8] = 1.0  # tied plateau
    peaks, vals = find_center_peaks(hm, threshold=0.2)
    assert peaks.shape[0] == 1
    assert abs(float(vals[0]) - 1.0) < 1e-6


def test_find_center_peaks_two_separate():
    """Two separated peaks yield two centers."""
    hm = torch.zeros(1, 1, 32, 32)
    hm[0, 0, 5, 5] = 1.0
    hm[0, 0, 25, 25] = 0.8
    peaks, vals = find_center_peaks(hm, threshold=0.2)
    assert peaks.shape[0] == 2


def test_find_center_peaks_below_threshold():
    """Nothing above threshold -> no peaks."""
    hm = torch.full((1, 1, 16, 16), 0.05)
    peaks, _ = find_center_peaks(hm, threshold=0.2)
    assert peaks.shape[0] == 0


# ---------------------------------------------------------------------------
# Deterministic GT roundtrip: GT maps -> grouping -> recovered masks
# ---------------------------------------------------------------------------


def test_gt_roundtrip_grouping_two_instances():
    """GT maps for two disks roundtrip to two masks with near-perfect IoU.

    Uses an *even* centroid (cy=90) that lands between grid rows — the case the
    plateau-robust peak finder must handle.
    """
    H, W, s = 128, 128, 2
    m1 = _disk(H, W, 30, 40, 18)
    m2 = _disk(H, W, 90, 95, 18)  # cy=90 even -> plateau on the +stride/2 grid
    masks = [m1, m2]

    fg = generate_foreground_mask(masks, (H, W), output_stride=s)
    center = generate_center_heatmap(masks, (H, W), output_stride=s, sigma=6.0)
    offsets, _ = generate_center_offsets(masks, (H, W), output_stride=s)

    instances = group_instances_from_offsets(
        fg, center, offsets, fg_threshold=0.5, peak_threshold=0.2, output_stride=s
    )
    assert len(instances) == 2

    # Downsample GT to output stride for an exact per-instance comparison.
    import torch.nn.functional as F

    def ds(m):
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        return (
            F.interpolate(t, size=(H // s, W // s), mode="area")[0, 0] > 0.5
        ).numpy()

    gt_ds = [ds(m) for m in masks]
    for inst in instances:
        pm = inst["mask"]
        best = max((pm & g).sum() / ((pm | g).sum() or 1) for g in gt_ds)
        assert best > 0.9


def test_segmentation_layer_postprocess_gt_roundtrip():
    """SegmentationLayer.postprocess maps GT maps to full-res masks -> sio.Labels."""
    H, W, s = 128, 128, 2
    m1 = _disk(H, W, 30, 40, 18)
    m2 = _disk(H, W, 90, 95, 18)
    masks = [m1, m2]
    fg = generate_foreground_mask(masks, (H, W), output_stride=s)
    center = generate_center_heatmap(masks, (H, W), output_stride=s, sigma=6.0)
    offsets, _ = generate_center_offsets(masks, (H, W), output_stride=s)

    # Build a layer without constructing a backend (postprocess is backend-free).
    layer = SegmentationLayer.__new__(SegmentationLayer)
    layer.output_stride = s
    layer.fg_threshold = 0.5
    layer.min_mask_area = 0
    layer.postprocess_config = PostprocessConfig(peak_threshold=0.2)

    info = PreprocInfo(
        original_size=(H, W),
        processed_size=(H, W),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=s,
    )
    raw = {
        "SegmentationHead": fg,
        "InstanceCenterHead": center,
        "CenterOffsetHead": offsets,
    }
    out = layer.postprocess(raw, info)
    assert len(out.pred_masks) == 1
    assert len(out.pred_masks[0]) == 2
    for inst in out.pred_masks[0]:
        fm = inst["mask"]
        assert fm.shape == (H, W)
        best = max((fm & g).sum() / ((fm | g).sum() or 1) for g in masks)
        assert best > 0.85  # full-res recovery (downsample/upsample boundary loss)

    # Package + .slp roundtrip.
    video = sio.Video.from_filename("dummy.mp4")
    out = type(out)(
        pred_masks=out.pred_masks,
        frame_indices=torch.tensor([0]),
        video_indices=torch.tensor([0]),
    )
    labels = out.to_labels(skeleton=None, videos=[video])
    assert len(labels.masks) == 2


def test_segmentation_layer_min_mask_area_drops_tiny_masks():
    """``min_mask_area`` drops tiny spurious masks while keeping large ones.

    Suppresses over-segmentation (the inference layer otherwise emits small
    noise blobs alongside real instances). Area is measured in original-image
    pixels after mapping back from the output-stride grid.
    """
    H, W, s = 128, 128, 2
    big = _disk(H, W, 40, 40, 22)  # ~1520 px
    tiny = _disk(H, W, 95, 95, 4)  # ~50 px
    masks = [big, tiny]
    fg = generate_foreground_mask(masks, (H, W), output_stride=s)
    center = generate_center_heatmap(masks, (H, W), output_stride=s, sigma=6.0)
    offsets, _ = generate_center_offsets(masks, (H, W), output_stride=s)
    raw = {
        "SegmentationHead": fg,
        "InstanceCenterHead": center,
        "CenterOffsetHead": offsets,
    }
    info = PreprocInfo(
        original_size=(H, W),
        processed_size=(H, W),
        eff_scale=torch.tensor([1.0]),
        input_scale=1.0,
        output_stride=s,
    )

    def _layer(min_area):
        layer = SegmentationLayer.__new__(SegmentationLayer)
        layer.output_stride = s
        layer.fg_threshold = 0.5
        layer.min_mask_area = min_area
        layer.postprocess_config = PostprocessConfig(peak_threshold=0.2)
        return layer

    # No filter: both instances (incl. the tiny one) are recovered.
    out0 = _layer(0).postprocess(raw, info)
    assert len(out0.pred_masks[0]) == 2

    # With a floor between the two areas, only the large mask survives.
    out1 = _layer(400).postprocess(raw, info)
    assert len(out1.pred_masks[0]) == 1
    assert int(out1.pred_masks[0][0]["mask"].sum()) >= 400


# ---------------------------------------------------------------------------
# Full train -> predict plumbing (loaders -> layer -> Predictor -> Labels)
# ---------------------------------------------------------------------------


def _seg_train_config(seg_path, tmp_path, max_epochs=1):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train_labels_path": [seg_path],
                "val_labels_path": [seg_path],
                "validation_fraction": 0.1,
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "cache_img_path": None,
                "use_existing_imgs": False,
                "delete_cache_imgs_after_training": True,
                "preprocessing": {
                    "ensure_rgb": False,
                    "ensure_grayscale": True,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "crop_size": None,
                    "min_crop_size": None,
                },
                "use_augmentations_train": False,
                "augmentation_config": None,
            },
            "model_config": {
                "init_weights": "default",
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 8,
                        "filters_rate": 1.5,
                        "max_stride": 16,
                        "convs_per_block": 2,
                        "stacks": 1,
                        "stem_stride": None,
                        "middle_block": True,
                        "up_interpolate": True,
                        "output_stride": 2,
                    }
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "bottomup": None,
                    "centered_instance": None,
                    "multi_class_bottomup": None,
                    "multi_class_topdown": None,
                    "bottomup_segmentation": {
                        "segmentation": {"output_stride": 2, "loss_weight": 1.0},
                        "center": {
                            "sigma": 5.0,
                            "output_stride": 2,
                            "loss_weight": 1.0,
                        },
                        "offsets": {"output_stride": 2, "loss_weight": 0.1},
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "val_data_loader": {"batch_size": 1, "num_workers": 0},
                "model_ckpt": {"save_top_k": 1, "save_last": True},
                "early_stopping": {
                    "stop_training_on_plateau": False,
                    "min_delta": 1e-08,
                    "patience": 20,
                },
                "trainer_devices": 1,
                "trainer_device_indices": None,
                "trainer_accelerator": "cpu",
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 1,
                "train_steps_per_epoch": None,
                "max_epochs": max_epochs,
                "seed": 1000,
                "keep_viz": False,
                "use_wandb": False,
                "save_ckpt": True,
                "ckpt_dir": str(tmp_path),
                "run_name": "seg_infer",
                "resume_ckpt_path": None,
                "optimizer_name": "Adam",
                "optimizer": {"lr": 1e-3, "amsgrad": False},
                "lr_scheduler": {
                    "reduce_lr_on_plateau": {
                        "threshold": 1e-07,
                        "threshold_mode": "rel",
                        "cooldown": 3,
                        "patience": 5,
                        "factor": 0.5,
                        "min_lr": 1e-08,
                    }
                },
                "online_hard_keypoint_mining": {
                    "online_mining": False,
                    "hard_to_easy_ratio": 2.0,
                    "min_hard_keypoints": 2,
                    "max_hard_keypoints": None,
                    "loss_scale": 5.0,
                },
                "eval": {"enabled": False},
            },
        }
    )


def test_segmentation_train_predict_wiring(minimal_instance_seg, tmp_path):
    """Exercise the full train -> load -> predict segmentation plumbing.

    Trains a segmentation model, then loads + predicts via the new Predictor
    pipeline. Asserts the plumbing (loaders -> SegmentationLayer -> Predictor ->
    sio.Labels) works and the result saves to .slp. The tiny 1-epoch model is
    not expected to produce good masks; mask *quality* is covered
    deterministically by the GT-roundtrip tests above.
    """
    from sleap_nn.training.model_trainer import ModelTrainer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.segmentation import SegmentationLayer
    from sleap_nn.inference.loaders import load_model_assets

    cfg = _seg_train_config(minimal_instance_seg.as_posix(), tmp_path, max_epochs=1)
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    run_dir = (tmp_path / "seg_infer").as_posix()

    # loaders detect the model type and build a SegmentationLayer.
    assets, model_types = load_model_assets([run_dir], device="cpu")
    assert model_types == ["bottomup_segmentation"]

    pred = Predictor.from_model_paths([run_dir], peak_threshold=0.05, device="cpu")
    assert isinstance(pred.layer, SegmentationLayer)

    out = pred.predict(minimal_instance_seg.as_posix(), make_labels=True)
    assert isinstance(out, sio.Labels)
    # Structure: masks is a flat list (possibly empty for an undertrained model);
    # any emitted mask is a PredictedSegmentationMask with a finite score.
    for m in out.masks:
        assert isinstance(m, sio.PredictedSegmentationMask)
        assert np.isfinite(m.score)

    # The output saves + reloads as a valid .slp.
    out_path = tmp_path / "preds.slp"
    out.save(out_path.as_posix())
    sio.load_slp(out_path.as_posix())
