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


@pytest.mark.parametrize("dataset_cls", ["semantic", "bottomup"])
def test_seg_target_registers_under_padding(minimal_instance, dataset_cls):
    """Whole-frame fg target must register to the image grid when the frame is padded.

    Regression for the ~half-pad offset: when the input H/W are not multiples of
    ``max_stride`` the image is padded bottom-right, but the pre-fix code resized the
    GT masks with a single ``F.interpolate`` straight to the PADDED size — stretching
    them across the pad region and displacing the foreground target from the image by
    ~pad/2 (worse toward the bottom-right). The masks must instead be carried through
    the SAME bottom-right pad as the image. ``max_stride=40`` is not a divisor of the
    384x384 frame, so it pads to 400x400 (pad 16), exactly the arabidopsis/soy
    ``pad_to_stride`` scenario.
    """
    import torch.nn.functional as F
    from sleap_nn.data.custom_datasets import (
        BottomUpSegmentationDataset,
        SemanticSegmentationDataset,
    )
    from sleap_nn.data.resizing import apply_pad_to_stride
    from sleap_nn.data.segmentation_maps import generate_foreground_mask

    labels = _seg_labels(minimal_instance)
    max_stride = 40  # 384 % 40 == 24 -> bottom-right pad of 16 px (400x400)
    seg_head = OmegaConf.create({"output_stride": 1})  # no downsample: crispest check
    if dataset_cls == "semantic":
        ds = SemanticSegmentationDataset(
            labels=[labels],
            seg_head_config=seg_head,
            max_stride=max_stride,
            ensure_grayscale=True,
        )
    else:
        ds = BottomUpSegmentationDataset(
            labels=[labels],
            seg_head_config=seg_head,
            center_head_config=OmegaConf.create({"output_stride": 1, "sigma": 4.0}),
            offset_head_config=OmegaConf.create({"output_stride": 1}),
            max_stride=max_stride,
            ensure_grayscale=True,
        )

    sample = ds[0]
    got = sample["foreground_mask"][0, 0].numpy() > 0.5
    assert got.shape == (400, 400)  # padded, not the raw 384x384

    # Masks captured at index-build (original resolution, pre-preprocessing).
    mask_arrays = [np.asarray(m, dtype=bool) for m in ds.lf_idx_list[0]["masks"]]

    # Correct reference: pad each mask bottom-right the SAME way as the image.
    correct = (
        generate_foreground_mask(
            [
                apply_pad_to_stride(
                    torch.from_numpy(m.astype(np.float32))[None, None],
                    max_stride=max_stride,
                )
                .squeeze()
                .numpy()
                > 0.5
                for m in mask_arrays
            ],
            img_hw=got.shape,
            output_stride=1,
        )[0, 0].numpy()
        > 0.5
    )
    # Buggy reference: STRETCH each mask straight to the padded size (pre-fix path).
    stretched = (
        generate_foreground_mask(
            [
                F.interpolate(
                    torch.from_numpy(m.astype(np.float32))[None, None],
                    size=got.shape,
                    mode="area",
                )
                .squeeze()
                .numpy()
                > 0.5
                for m in mask_arrays
            ],
            img_hw=got.shape,
            output_stride=1,
        )[0, 0].numpy()
        > 0.5
    )

    # The two references genuinely differ (the pad shift is real) -> test is
    # non-trivial. Use an exact set difference rather than an IoU threshold so the
    # assertion does not depend on fixture geometry / radius / max_stride.
    assert not np.array_equal(correct, stretched)
    # After the fix the target equals the pad-correct reference exactly...
    assert np.array_equal(got, correct)
    # ...and is clearly NOT the stretched (pre-fix) target.
    assert not np.array_equal(got, stretched)


def _chain_reference(mask_arrays, max_hw, scale, max_stride, output_stride):
    """Foreground target from masks carried through the PUBLIC image chain.

    Mirrors :meth:`BaseDataset._apply_common_preprocessing` for the whole-frame
    masks: size-match -> scale -> stride-pad (the exact helpers the image rides),
    binarized once, then reduced to the union foreground. Any dataset that skips,
    reorders, or re-implements a leg (e.g. the pre-fix single stretch-to-padded)
    diverges from this reference.
    """
    from sleap_nn.data.resizing import (
        apply_pad_to_stride,
        apply_resizer,
        apply_sizematcher,
    )
    from sleap_nn.data.segmentation_maps import generate_foreground_mask

    mt = torch.from_numpy(
        np.stack([m.astype(np.float32) for m in mask_arrays])
    ).unsqueeze(0)
    mt, _ = apply_sizematcher(mt, max_height=max_hw[0], max_width=max_hw[1])
    mt, _ = apply_resizer(mt, torch.zeros(1), scale=scale)
    mt = apply_pad_to_stride(mt, max_stride=max_stride)
    masks = [mt[0, k].numpy() > 0.5 for k in range(mt.shape[1])]
    img_hw = (int(mt.shape[-2]), int(mt.shape[-1]))
    return (
        generate_foreground_mask(masks, img_hw=img_hw, output_stride=output_stride)[
            0, 0
        ].numpy()
        > 0.5
    )


@pytest.mark.parametrize("dataset_cls", ["semantic", "bottomup"])
@pytest.mark.parametrize(
    "leg, max_stride, scale, max_hw",
    [
        ("pad", 40, 1.0, (None, None)),  # bottom-right stride pad only (384->400)
        ("scale", 1, 0.5, (None, None)),  # scale/resizer only (384->192)
        ("sizematch", 1, 1.0, (256, 256)),  # size-matcher only (384->256)
        ("sizematch_nonsquare", 1, 1.0, (300, 500)),  # aspect-preserving + asym. pad
        ("combo", 32, 0.5, (320, 320)),  # all three legs at once
    ],
)
def test_seg_target_matches_full_preprocessing_chain(
    minimal_instance, dataset_cls, leg, max_stride, scale, max_hw
):
    """The fg target must ride the FULL size-match / scale / stride-pad chain.

    The prior regression test only padded (max_stride, scale=1, max_hw=None), so the
    size-matcher and scale legs were never exercised — a dataset that dropped or
    re-implemented either leg would still pass. Each parametrization here makes a
    DIFFERENT leg the one that changes the resolution, and the target is checked
    against a reference carried through the identical public helpers.
    """
    from sleap_nn.data.custom_datasets import (
        BottomUpSegmentationDataset,
        SemanticSegmentationDataset,
    )

    labels = _seg_labels(minimal_instance)
    seg_head = OmegaConf.create({"output_stride": 1})  # crispest check
    common = dict(
        labels=[labels],
        seg_head_config=seg_head,
        max_stride=max_stride,
        scale=scale,
        max_hw=max_hw,
        ensure_grayscale=True,
    )
    if dataset_cls == "semantic":
        ds = SemanticSegmentationDataset(**common)
    else:
        ds = BottomUpSegmentationDataset(
            center_head_config=OmegaConf.create({"output_stride": 1, "sigma": 4.0}),
            offset_head_config=OmegaConf.create({"output_stride": 1}),
            **common,
        )

    got = ds[0]["foreground_mask"][0, 0].numpy() > 0.5

    # Masks captured at index-build (original resolution, full-frame).
    mask_arrays = [np.asarray(m, dtype=bool) for m in ds.lf_idx_list[0]["masks"]]
    correct = _chain_reference(mask_arrays, max_hw, scale, max_stride, output_stride=1)

    assert got.shape == correct.shape
    assert np.array_equal(got, correct), f"target diverges from full chain on leg={leg}"


@pytest.mark.parametrize("dataset_cls", ["semantic", "bottomup"])
def test_seg_heterogeneous_mask_shapes_register_without_crash(
    minimal_instance, dataset_cls
):
    """Ragged (offset-carrying) decoded masks must place correctly, not crash.

    ``decode_mask_to_image_res`` top-left-pads an offset mask to a per-instance
    shape, so a frame's decoded masks can have DIFFERENT (H, W) and none may match
    the frame. The pre-fix code, whenever preprocessing changed the size, ran the
    masks through ``_resize_masks_like_image`` -> ``np.stack`` and raised
    ``ValueError: all input arrays must have the same shape``; the fix canvases each
    mask onto the full-frame grid at its absolute (top-left-anchored) origin first.

    ``max_stride=40`` forces a bottom-right pad (384 -> 400), which is exactly the
    condition that made the pre-fix ``np.stack`` fire — so this test discriminates
    the fix from the old code (a no-size-change config would NOT, since
    ``generate_foreground_mask`` clips ragged masks itself and never stacks). The
    placed masks sit well inside 384, so the bottom-right pad only extends the
    canvas; the fg target must equal the exact union at 400x400.
    """
    from sleap_nn.data.custom_datasets import (
        BottomUpSegmentationDataset,
        SemanticSegmentationDataset,
    )

    labels = _seg_labels(minimal_instance)
    common = dict(
        labels=[labels],
        seg_head_config=OmegaConf.create({"output_stride": 1}),
        max_stride=40,  # 384 -> 400 pad: the size change that fires the pre-fix stack
        ensure_grayscale=True,
    )
    if dataset_cls == "semantic":
        ds = SemanticSegmentationDataset(**common)
    else:
        ds = BottomUpSegmentationDataset(
            center_head_config=OmegaConf.create({"output_stride": 1, "sigma": 4.0}),
            offset_head_config=OmegaConf.create({"output_stride": 1}),
            **common,
        )

    # Simulate offset-decoded masks: differently-shaped, top-left-anchored arrays.
    m1 = np.zeros((40, 50), dtype=bool)
    m1[10:20, 12:22] = True
    m2 = np.zeros((150, 300), dtype=bool)
    m2[120:140, 250:280] = True
    ds.lf_idx_list[0]["masks"] = [m1, m2]

    sample = ds[0]  # pre-fix: raised ValueError in _resize_masks_like_image's np.stack
    fg = sample["foreground_mask"][0, 0].numpy() > 0.5
    assert fg.shape == (
        400,
        400,
    )  # padded (proves the size change that fired the stack)

    expected = np.zeros((400, 400), dtype=bool)
    expected[10:20, 12:22] = True
    expected[120:140, 250:280] = True
    assert np.array_equal(fg, expected)


def test_masks_to_frame_canvas_handles_ragged_shapes():
    """``_masks_to_frame_canvas`` unifies ragged masks onto the frame grid."""
    from sleap_nn.data.custom_datasets import _masks_to_frame_canvas

    m1 = np.zeros((40, 50), dtype=bool)
    m1[5:8, 5:8] = True
    m2 = np.zeros((120, 200), dtype=bool)
    m2[100:110, 150:160] = True

    t = _masks_to_frame_canvas([m1, m2], (384, 384))
    assert t.shape == (1, 2, 384, 384)
    assert t.dtype == torch.float32
    b = t[0].numpy() > 0.5
    assert b[0, 5:8, 5:8].all() and b[1, 100:110, 150:160].all()
    assert int(b[0].sum()) == 9 and int(b[1].sum()) == 100

    # Oversized mask (decoded extent can exceed the frame) is clamped, not an error.
    big = np.ones((500, 500), dtype=bool)
    t2 = _masks_to_frame_canvas([big], (384, 384))
    assert t2.shape == (1, 1, 384, 384) and (t2[0, 0].numpy() > 0.5).all()

    # Empty list -> a well-formed (1, 0, H, W) tensor (no instances).
    assert _masks_to_frame_canvas([], (10, 12)).shape == (1, 0, 10, 12)


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


# --------------------------------------------------------------------------- #
# Thin-structure knobs: loss weighting / pos_weight + max-pool target + OS=1
# --------------------------------------------------------------------------- #
def test_seg_head_config_loss_target_knobs():
    """The shared SegmentationHeadConfig exposes loss + target knobs (default-inert)."""
    from sleap_nn.config.get_config import get_head_configs

    s = get_head_configs("semantic_segmentation").semantic_segmentation.segmentation
    # Defaults preserve the historical symmetric-unweighted / area-0.5 behavior.
    assert (s.bce_weight, s.dice_weight) == (0.5, 0.5)
    assert s.bce_pos_weight is None
    assert s.target_maxpool is False

    d = {
        "semantic_segmentation": {
            "segmentation": {
                "output_stride": 1,
                "loss_weight": 1.0,
                "bce_weight": 0.3,
                "dice_weight": 0.7,
                "bce_pos_weight": 10.0,
                "target_maxpool": True,
            }
        }
    }
    s2 = get_head_configs(d).semantic_segmentation.segmentation
    assert (s2.output_stride, s2.bce_weight, s2.dice_weight) == (1, 0.3, 0.7)
    assert s2.bce_pos_weight == 10.0 and s2.target_maxpool is True

    # Same leaf is shared by bottomup_segmentation -> knobs available there too.
    sb = get_head_configs("bottomup_segmentation").bottomup_segmentation.segmentation
    assert hasattr(sb, "bce_pos_weight") and hasattr(sb, "target_maxpool")


def test_bce_dice_loss_pos_weight():
    """pos_weight up-weights the foreground BCE term; None == prior behavior."""
    from sleap_nn.training.losses import compute_bce_dice_loss

    y_pred = torch.zeros(1, 1, 8, 8)
    y_gt = torch.zeros(1, 1, 8, 8)
    y_gt[0, 0, 3, 3] = 1.0
    base = compute_bce_dice_loss(y_pred, y_gt)
    # None pos_weight is byte-for-byte the default-weighted loss.
    assert torch.allclose(compute_bce_dice_loss(y_pred, y_gt, pos_weight=None), base)
    weighted = compute_bce_dice_loss(
        y_pred, y_gt, bce_weight=0.3, dice_weight=0.7, pos_weight=10.0
    )
    assert float(weighted) != float(base)


def test_generate_foreground_mask_maxpool_preserves_thin():
    """Max-pool downsample keeps a thin structure that area-0.5 erodes at OS>1."""
    from sleap_nn.data.segmentation_maps import generate_foreground_mask

    m = np.zeros((64, 64), dtype=bool)
    for i in range(62):  # a ~2px-wide diagonal "lateral root"
        m[i, i] = True
        m[i, i + 1] = True
    area = generate_foreground_mask([m], (64, 64), output_stride=2, maxpool=False)
    mp = generate_foreground_mask([m], (64, 64), output_stride=2, maxpool=True)
    assert mp.sum() > area.sum()  # max-pool preserves more of the thin diagonal
    # At output_stride=1 there is no downsample, so maxpool is inert (identical).
    os1_a = generate_foreground_mask([m], (64, 64), output_stride=1, maxpool=False)
    os1_b = generate_foreground_mask([m], (64, 64), output_stride=1, maxpool=True)
    assert torch.equal(os1_a, os1_b)
    assert int(os1_a.sum()) == int(m.sum())  # full-res target == the union exactly


def test_semantic_module_reads_loss_knobs():
    """The LightningModule reads bce/dice/pos_weight from the head config."""
    from sleap_nn.training.lightning_modules import SemanticSegmentationLightningModule

    backbone_config, head_configs = _backbone_head_configs()
    head_configs.semantic_segmentation.segmentation.bce_weight = 0.3
    head_configs.semantic_segmentation.segmentation.dice_weight = 0.7
    head_configs.semantic_segmentation.segmentation.bce_pos_weight = 10.0
    module = SemanticSegmentationLightningModule(
        model_type="semantic_segmentation",
        backbone_type="unet",
        backbone_config=backbone_config,
        head_configs=head_configs,
        init_weights="default",
    )
    assert module.fg_bce_weight == 0.3
    assert module.fg_dice_weight == 0.7
    assert module.fg_bce_pos_weight == 10.0


def test_semantic_dataset_target_maxpool(minimal_instance):
    """target_maxpool on the dataset yields >= foreground vs the default target."""
    from sleap_nn.data.custom_datasets import SemanticSegmentationDataset

    labels = _seg_labels(minimal_instance)
    common = dict(labels=[labels], max_stride=16, ensure_grayscale=True)
    ds_area = SemanticSegmentationDataset(
        seg_head_config=OmegaConf.create({"output_stride": 4, "target_maxpool": False}),
        **common,
    )
    ds_mp = SemanticSegmentationDataset(
        seg_head_config=OmegaConf.create({"output_stride": 4, "target_maxpool": True}),
        **common,
    )
    fg_area = float(ds_area[0]["foreground_mask"].sum())
    fg_mp = float(ds_mp[0]["foreground_mask"].sum())
    assert fg_mp >= fg_area  # max-pool never drops foreground, and grows thin edges


def test_semantic_output_stride_1_end_to_end(minimal_instance_seg, tmp_path):
    """OS=1 + pos_weight + Dice-tilt trains end-to-end and predicts one mask/frame."""
    from sleap_nn.training.model_trainer import ModelTrainer
    from sleap_nn.inference.predictor import Predictor
    from sleap_nn.inference.layers.segmentation import SemanticSegmentationLayer

    cfg = _semantic_train_config(
        minimal_instance_seg.as_posix(), tmp_path, max_epochs=1
    )
    seg = cfg.model_config.head_configs.semantic_segmentation.segmentation
    seg.output_stride = 1
    seg.bce_weight = 0.3
    seg.dice_weight = 0.7
    seg.bce_pos_weight = 10.0
    cfg.trainer_config.run_name = "sem_os1"
    ModelTrainer.get_model_trainer_from_config(cfg).train()
    run_dir = (tmp_path / "sem_os1").as_posix()

    pred = Predictor.from_model_paths([run_dir], fg_threshold=0.5, device="cpu")
    assert isinstance(pred.layer, SemanticSegmentationLayer)
    out = pred.predict(minimal_instance_seg.as_posix(), make_labels=True)
    assert isinstance(out, sio.Labels)
    for lf in out:
        assert len(lf.masks) <= 1
        for m in lf.masks:
            assert isinstance(m, sio.PredictedSegmentationMask)
            assert m.instance is None
