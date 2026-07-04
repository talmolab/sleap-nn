"""Tests for the ``semantic_segmentation`` model type (issue #686).

Whole-frame binary foreground/background segmentation: a lone ``SegmentationHead``
on the full frame with NO instance grouping, tiling-compatible, decoded by
thresholding the foreground probability into ONE instance-less mask per frame.
"""

import numpy as np
import pytest
import sleap_io as sio
import torch
from omegaconf import OmegaConf


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_semantic_config_string_path():
    """``get_head_configs('semantic_segmentation')`` builds a lone fg head."""
    from sleap_nn.config.get_config import get_head_configs

    hc = get_head_configs("semantic_segmentation")
    assert hc.semantic_segmentation is not None
    assert hc.semantic_segmentation.segmentation.output_stride == 2
    assert hc.semantic_segmentation.segmentation.loss_weight == 1.0
    # Fg-only: no anchor_part (whole-frame, no crop) and no sibling seg types.
    assert not hasattr(hc.semantic_segmentation.segmentation, "anchor_part")
    assert hc.bottomup_segmentation is None
    assert hc.centered_instance_segmentation is None


def test_semantic_config_dict_path():
    """The dict path round-trips custom ``output_stride`` / ``loss_weight``."""
    from sleap_nn.config.get_config import get_head_configs

    d = {
        "semantic_segmentation": {
            "segmentation": {"output_stride": 4, "loss_weight": 2.0}
        }
    }
    hc = get_head_configs(d)
    assert hc.semantic_segmentation.segmentation.output_stride == 4
    assert hc.semantic_segmentation.segmentation.loss_weight == 2.0


def test_semantic_get_model_type_from_cfg():
    """A semantic head config is detected as ``semantic_segmentation``."""
    from sleap_nn.config.get_config import get_head_configs
    from sleap_nn.config.utils import get_model_type_from_cfg

    hc = get_head_configs("semantic_segmentation")
    cfg = OmegaConf.create({"model_config": {"head_configs": OmegaConf.structured(hc)}})
    assert get_model_type_from_cfg(cfg) == "semantic_segmentation"


def test_semantic_check_tiling_accepts():
    """``check_tiling`` admits semantic_segmentation (tiling-compatible)."""
    from sleap_nn.config.get_config import get_head_configs
    from sleap_nn.config.utils import check_tiling

    hc = get_head_configs("semantic_segmentation")
    cfg = OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "tiling": {
                        "enabled": True,
                        "tile_size": 256,
                        "overlap": 64,
                        "min_overlap_fraction": 0.25,
                    }
                }
            },
            "model_config": {
                "backbone_config": {"unet": {"max_stride": 16, "output_stride": 2}},
                "head_configs": OmegaConf.structured(hc),
            },
        }
    )
    # Must not raise (the allowlist accepts semantic_segmentation).
    out = check_tiling(cfg)
    assert out.data_config.preprocessing.tiling.tile_size == 256


# --------------------------------------------------------------------------- #
# Architecture head
# --------------------------------------------------------------------------- #
def test_semantic_get_head_builds_one_seg_head():
    """``get_head`` builds exactly one ``SegmentationHead`` (no center/offset)."""
    from sleap_nn.architectures.heads import SegmentationHead
    from sleap_nn.architectures.model import get_head

    head_cfg = OmegaConf.create(
        {"segmentation": {"output_stride": 2, "loss_weight": 1.0}}
    )
    heads = get_head("semantic_segmentation", head_cfg)
    assert len(heads) == 1
    assert isinstance(heads[0], SegmentationHead)


# --------------------------------------------------------------------------- #
# Datasets — fg-only target, no center/offset/weight keys
# --------------------------------------------------------------------------- #
def _seg_labels(minimal_instance, radius=30):
    labels = sio.load_slp(str(minimal_instance))
    video = labels.videos[0]
    lf0 = labels[0]
    h, w = video.shape[1], video.shape[2]
    yy, xx = np.ogrid[:h, :w]
    masks = []
    for inst in lf0.instances:
        pts = inst.numpy()
        cx, cy = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        masks.append(sio.UserSegmentationMask.from_numpy(blob))
    seg_lf = sio.LabeledFrame(
        video=video, frame_idx=lf0.frame_idx, instances=lf0.instances, masks=masks
    )
    return sio.Labels(videos=[video], labeled_frames=[seg_lf])


_FORBIDDEN_KEYS = ("center_heatmap", "center_offsets", "foreground_weight")


def test_semantic_dataset_fg_only(minimal_instance):
    """The plain dataset yields ONLY ``foreground_mask`` (no grouping targets)."""
    from sleap_nn.data.custom_datasets import SemanticSegmentationDataset

    labels = _seg_labels(minimal_instance)
    ds = SemanticSegmentationDataset(
        labels=[labels],
        seg_head_config=OmegaConf.create({"output_stride": 2}),
        max_stride=16,
        ensure_grayscale=True,
    )
    assert len(ds) == 1
    sample = ds[0]
    assert "foreground_mask" in sample
    assert "image" in sample
    for k in _FORBIDDEN_KEYS:
        assert k not in sample, f"semantic dataset must not emit {k}"
    # foreground_mask is a single-channel binary target at output stride.
    fg = sample["foreground_mask"]
    assert fg.shape[-3] == 1  # (1, 1, H/s, W/s)
    assert fg.max() <= 1.0 and fg.min() >= 0.0
    assert fg.sum() > 0  # circular blobs -> nonempty foreground


def test_semantic_tiled_dataset_fg_only(minimal_instance):
    """The tiled dataset yields ONLY ``foreground_mask`` (no ownership filter)."""
    from sleap_nn.data.custom_datasets import SemanticSegmentationTiledDataset

    labels = _seg_labels(minimal_instance)
    tiling = OmegaConf.create(
        {
            "enabled": True,
            "tile_size": 160,
            "overlap": 48,
            "sampling": "grid",
            "min_overlap_fraction": 0.25,
            "tile_fg_fraction": 0.5,
            "center_jitter": 0.0,
            "min_visible_keypoints": 0,
            "samples_per_frame": 2,
        }
    )
    ds = SemanticSegmentationTiledDataset(
        labels=[labels],
        seg_head_config=OmegaConf.create({"output_stride": 2}),
        max_stride=16,
        ensure_grayscale=True,
        tiling=tiling,
    )
    assert len(ds) > 0
    saw_fg = False
    for i in range(len(ds)):
        sample = ds[i]
        assert "foreground_mask" in sample
        assert "tile_origin" in sample
        for k in _FORBIDDEN_KEYS:
            assert k not in sample
        if float(sample["foreground_mask"].sum()) > 0:
            saw_fg = True
    # At least one grid tile covers foreground (union of all masks touching it).
    assert saw_fg


# --------------------------------------------------------------------------- #
# LightningModule — sigmoid-dict forward + val/fg_iou
# --------------------------------------------------------------------------- #
def _backbone_head_configs():
    backbone_config = OmegaConf.create(
        {
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
        }
    )
    head_configs = OmegaConf.create(
        {
            "single_instance": None,
            "centroid": None,
            "bottomup": None,
            "centered_instance": None,
            "multi_class_bottomup": None,
            "multi_class_topdown": None,
            "bottomup_segmentation": None,
            "centered_instance_segmentation": None,
            "semantic_segmentation": {
                "segmentation": {"output_stride": 2, "loss_weight": 1.0},
            },
        }
    )
    return backbone_config, head_configs


def test_semantic_lightning_forward_returns_sigmoid_dict():
    """``forward`` returns ``{"SegmentationHead": sigmoid(logits)}`` (probabilities).

    This is load-bearing: the whole-frame ``SemanticSegmentationLayer`` and the
    tiled wrapper read ``SegmentationHead`` as probabilities (thresholded at
    ``fg_threshold``; Gaussian-averaged across tile overlaps).
    """
    from sleap_nn.training.lightning_modules import SemanticSegmentationLightningModule

    backbone_config, head_configs = _backbone_head_configs()
    module = SemanticSegmentationLightningModule(
        model_type="semantic_segmentation",
        backbone_type="unet",
        backbone_config=backbone_config,
        head_configs=head_configs,
        init_weights="default",
    )
    module.eval()
    x = torch.rand(1, 1, 1, 64, 64)
    with torch.no_grad():
        out = module.forward(x)
    assert isinstance(out, dict) and set(out.keys()) == {"SegmentationHead"}
    fg = out["SegmentationHead"]
    # Sigmoid probabilities: strictly within [0, 1].
    assert float(fg.min()) >= 0.0 and float(fg.max()) <= 1.0


# --------------------------------------------------------------------------- #
# End-to-end train -> autodetect -> predict (plain + tiled)
# --------------------------------------------------------------------------- #
def _semantic_train_config(seg_path, tmp_path, tiling=False, max_epochs=1):
    preprocessing = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "max_width": None,
        "max_height": None,
        "scale": 1.0,
        "crop_size": None,
        "min_crop_size": None,
    }
    if tiling:
        preprocessing["tiling"] = {
            "enabled": True,
            "tile_size": 160,
            "overlap": 48,
            "sampling": "foreground",
            "min_overlap_fraction": 0.25,
            "tile_fg_fraction": 0.5,
            "center_jitter": 0.0,
            "min_visible_keypoints": 0,
            "samples_per_frame": 2,
            "blend": "gaussian",
            "sigma_scale": 0.125,
            "tile_batch_size": 8,
            "accumulator_device": "cpu",
            "cpu_thresh": 0.40,
        }
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
                "preprocessing": preprocessing,
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
                    "bottomup_segmentation": None,
                    "centered_instance_segmentation": None,
                    "semantic_segmentation": {
                        "segmentation": {"output_stride": 2, "loss_weight": 1.0},
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
                "run_name": "sem_infer" + ("_tiled" if tiling else ""),
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


@pytest.mark.parametrize("tiling", [False, True])
def test_semantic_train_predict_wiring(minimal_instance_seg, tmp_path, tiling):
    """Full train -> load (autodetect) -> predict, plain and tiled.

    Asserts the plumbing (loaders -> Semantic[Tiled]SegmentationLayer -> Predictor
    -> sio.Labels) works and emits at most ONE instance-less
    ``PredictedSegmentationMask`` per frame.
    """
    from sleap_nn.training.model_trainer import ModelTrainer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.segmentation import SemanticSegmentationLayer
    from sleap_nn.inference.layers.tiled import TiledSemanticSegmentationLayer
    from sleap_nn.inference.loaders import load_model_assets

    cfg = _semantic_train_config(
        minimal_instance_seg.as_posix(), tmp_path, tiling=tiling, max_epochs=1
    )
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    run_dir = (tmp_path / cfg.trainer_config.run_name).as_posix()

    # loaders autodetect the model type.
    assets, model_types = load_model_assets([run_dir], device="cpu")
    assert model_types == ["semantic_segmentation"]

    # fg_threshold / min_mask_area thread from_model_paths -> load_model_assets
    # -> _build_semantic_segmentation -> SemanticSegmentationLayer.
    pred = Predictor.from_model_paths(
        [run_dir], fg_threshold=0.1, min_mask_area=0, device="cpu"
    )
    expected = TiledSemanticSegmentationLayer if tiling else SemanticSegmentationLayer
    assert isinstance(pred.layer, expected)

    out = pred.predict(minimal_instance_seg.as_posix(), make_labels=True)
    assert isinstance(out, sio.Labels)
    for lf in out:
        # Semantic => at most ONE mask per frame (no grouping).
        assert len(lf.masks) <= 1
        for m in lf.masks:
            assert isinstance(m, sio.PredictedSegmentationMask)
            # Instance-less: a whole-frame foreground has no instance.
            assert m.instance is None
            assert np.isfinite(m.score)

    # The output saves + reloads as a valid .slp.
    out_path = tmp_path / f"preds_{'tiled' if tiling else 'plain'}.slp"
    out.save(out_path.as_posix())
    assert out_path.exists()


def test_semantic_predict_fg_threshold_min_area_thread(minimal_instance_seg, tmp_path):
    """``fg_threshold`` / ``min_mask_area`` reach the built ``SemanticSegmentationLayer``."""
    from sleap_nn.training.model_trainer import ModelTrainer
    from sleap_nn.inference.predictor import Predictor

    cfg = _semantic_train_config(
        minimal_instance_seg.as_posix(), tmp_path, max_epochs=1
    )
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    run_dir = (tmp_path / "sem_infer").as_posix()

    pred = Predictor.from_model_paths(
        [run_dir], min_mask_area=500, fg_threshold=0.7, device="cpu"
    )
    assert pred.layer.min_mask_area == 500
    assert abs(pred.layer.fg_threshold - 0.7) < 1e-9


# --------------------------------------------------------------------------- #
# Eval — matching-free whole-frame foreground
# --------------------------------------------------------------------------- #
def _fg_slp(video, cx_cy_r_list, tmp, name, predicted, score=0.9):
    h, w = video.shape[1], video.shape[2]
    yy, xx = np.ogrid[:h, :w]
    union = np.zeros((h, w), dtype=bool)
    for cx, cy, r in cx_cy_r_list:
        union |= ((yy - cy) ** 2 + (xx - cx) ** 2) <= r**2
    if predicted:
        mask = sio.PredictedSegmentationMask.from_numpy(union, score=score)
    else:
        mask = sio.UserSegmentationMask.from_numpy(union)
    lf = sio.LabeledFrame(video=video, frame_idx=0, masks=[mask])
    labels = sio.Labels(videos=[video], labeled_frames=[lf])
    path = tmp / name
    labels.save(path.as_posix())
    return path.as_posix()


def test_semantic_eval_matching_free(minimal_instance, tmp_path):
    """``run_evaluation(match_method='semantic')`` scores whole-frame fg IoU/clDice.

    No Hungarian matching, no IoU threshold: a single union foreground per frame.
    """
    from sleap_nn.evaluation import run_evaluation

    video = sio.load_slp(str(minimal_instance)).videos[0]
    blobs = [(100, 100, 30), (150, 120, 30)]
    gt = _fg_slp(video, blobs, tmp_path, "gt.slp", predicted=False)
    pr = _fg_slp(
        video,
        [(101, 101, 28), (151, 121, 28)],
        tmp_path,
        "pr.slp",
        predicted=True,
    )

    metrics = run_evaluation(
        ground_truth_path=gt, predicted_path=pr, match_method="semantic"
    )
    assert set(metrics.keys()) == {"semantic_metrics"}
    sm = metrics["semantic_metrics"]
    assert sm["n_frames"] == 1
    assert 0.5 < sm["mean_iou"] < 1.0
    assert not np.isnan(sm["mean_boundary_iou"])


def test_semantic_eval_callback_foreground_mode():
    """The foreground-mode callback unions masks and scores IoU without matching."""
    from sleap_nn.training.callbacks import SegmentationEvaluationCallback

    cb = SegmentationEvaluationCallback(foreground=True)
    assert cb.foreground is True

    gt = np.zeros((32, 32), bool)
    gt[5:20, 5:20] = True
    pr = np.zeros((32, 32), bool)
    pr[6:19, 6:19] = True

    m = cb._compute_metrics_foreground([{"masks": [pr]}], [{"masks": [gt]}])
    assert m["n_frames"] == 1
    assert 0.5 < m["fg_mean_iou"] < 1.0
    assert m["fg_frame_recall"] == 1.0

    # An empty prediction against a non-empty GT is scored (IoU 0), not skipped.
    m2 = cb._compute_metrics_foreground([{"masks": []}], [{"masks": [gt]}])
    assert m2["fg_mean_iou"] == 0.0
    assert m2["fg_frame_recall"] == 0.0

    # An empty-GT frame is skipped (no foreground to score).
    m3 = cb._compute_metrics_foreground([{"masks": [pr]}], [{"masks": []}])
    assert m3["n_frames"] == 0
