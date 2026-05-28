"""Tests for :mod:`sleap_nn.inference.loaders`.

Validates that ``load_model_assets`` can independently load every
supported model type and return correct ``LoadedAssets``.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from sleap_nn.inference.loaders import (
    LoadedAssets,
    _common_lightning_kwargs,
    _detect_backbone_type,
    _load_training_config,
    load_model_assets,
)

CKPT_ROOT = Path(__file__).resolve().parents[1] / "assets" / "model_ckpts"
SINGLE_CKPT = CKPT_ROOT / "minimal_instance_single_instance"
BOTTOMUP_CKPT = CKPT_ROOT / "minimal_instance_bottomup"
MULTICLASS_BU_CKPT = CKPT_ROOT / "minimal_instance_multiclass_bottomup"
CENTROID_CKPT = CKPT_ROOT / "minimal_instance_centroid"
CENTERED_CKPT = CKPT_ROOT / "minimal_instance_centered_instance"
MULTICLASS_TD_CKPT = CKPT_ROOT / "minimal_instance_multiclass_centered_instance"


# ─────────────────────────────────────────────────────────────────────────
# Module-scoped fixtures — load each combo ONCE
# ─────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def single_assets():
    if not SINGLE_CKPT.exists():
        pytest.skip("single-instance ckpt absent")
    assets, types = load_model_assets([str(SINGLE_CKPT)], device="cpu")
    yield assets, types
    del assets
    gc.collect()


@pytest.fixture(scope="module")
def bottomup_assets():
    if not BOTTOMUP_CKPT.exists():
        pytest.skip("bottomup ckpt absent")
    assets, types = load_model_assets([str(BOTTOMUP_CKPT)], device="cpu")
    yield assets, types
    del assets
    gc.collect()


@pytest.fixture(scope="module")
def multiclass_bu_assets():
    if not MULTICLASS_BU_CKPT.exists():
        pytest.skip("multiclass-bottomup ckpt absent")
    assets, types = load_model_assets([str(MULTICLASS_BU_CKPT)], device="cpu")
    yield assets, types
    del assets
    gc.collect()


@pytest.fixture(scope="module")
def topdown_assets():
    if not (CENTROID_CKPT.exists() and CENTERED_CKPT.exists()):
        pytest.skip("topdown ckpts absent")
    assets, types = load_model_assets(
        [str(CENTROID_CKPT), str(CENTERED_CKPT)], device="cpu"
    )
    yield assets, types
    del assets
    gc.collect()


@pytest.fixture(scope="module")
def topdown_multiclass_assets():
    if not (CENTROID_CKPT.exists() and MULTICLASS_TD_CKPT.exists()):
        pytest.skip("topdown-multiclass ckpts absent")
    assets, types = load_model_assets(
        [str(CENTROID_CKPT), str(MULTICLASS_TD_CKPT)], device="cpu"
    )
    yield assets, types
    del assets
    gc.collect()


@pytest.fixture(scope="module")
def centered_only_assets():
    if not CENTERED_CKPT.exists():
        pytest.skip("centered-instance ckpt absent")
    assets, types = load_model_assets([str(CENTERED_CKPT)], device="cpu")
    yield assets, types
    del assets
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────
# Tests — LoadedAssets structure
# ─────────────────────────────────────────────────────────────────────────


def test_load_single_instance(single_assets):
    assets, types = single_assets
    assert isinstance(assets, LoadedAssets)
    assert types == ["single_instance"]
    assert assets.inference_model is not None
    assert assets.skeletons is not None and len(assets.skeletons) > 0
    assert assets.max_stride is not None
    assert assets.backbone_type is not None


def test_load_bottomup(bottomup_assets):
    assets, types = bottomup_assets
    assert types == ["bottomup"]
    assert assets.inference_model is not None
    assert assets.bottomup_config is not None
    assert assets.backbone_type is not None
    assert hasattr(assets.inference_model, "paf_scorer")


def test_load_bottomup_multiclass(multiclass_bu_assets):
    assets, types = multiclass_bu_assets
    assert types == ["multi_class_bottomup"]
    assert assets.inference_model is not None
    assert assets.bottomup_config is not None


def test_load_topdown(topdown_assets):
    assets, types = topdown_assets
    assert "centroid" in types
    assert "centered_instance" in types
    assert assets.inference_model is not None
    assert hasattr(assets.inference_model, "centroid_crop")
    assert hasattr(assets.inference_model, "instance_peaks")
    assert assets.centroid_config is not None
    assert assets.confmap_config is not None


def test_load_topdown_crop_size_resolved(topdown_assets):
    """crop_size must come from the confmap config, not be left as None."""
    assets, _ = topdown_assets
    assert assets.preprocess_config.crop_size is not None
    assert assets.preprocess_config.crop_size > 0


def test_load_topdown_multiclass(topdown_multiclass_assets):
    assets, types = topdown_multiclass_assets
    assert "centroid" in types
    assert "multi_class_topdown" in types
    assert assets.inference_model is not None


def test_load_centered_instance_only(centered_only_assets):
    """Standalone centered-instance (no centroid model)."""
    assets, types = centered_only_assets
    assert types == ["centered_instance"]
    assert assets.inference_model is not None
    assert hasattr(assets.inference_model, "centroid_crop")


# ─────────────────────────────────────────────────────────────────────────
# Tests — L1 helpers
# ─────────────────────────────────────────────────────────────────────────


def test_load_training_config_yaml():
    if not SINGLE_CKPT.exists():
        pytest.skip("single-instance ckpt absent")
    config, is_legacy = _load_training_config(str(SINGLE_CKPT))
    assert not is_legacy
    assert hasattr(config, "model_config")
    assert hasattr(config, "data_config")


def test_load_training_config_missing():
    with pytest.raises(FileNotFoundError, match="No training_config"):
        _load_training_config("/nonexistent/path")


def test_detect_backbone_type():
    if not SINGLE_CKPT.exists():
        pytest.skip("single-instance ckpt absent")
    config, _ = _load_training_config(str(SINGLE_CKPT))
    backbone = _detect_backbone_type(config)
    assert isinstance(backbone, str)
    assert len(backbone) > 0


def test_common_lightning_kwargs_keys():
    if not SINGLE_CKPT.exists():
        pytest.skip("single-instance ckpt absent")
    config, _ = _load_training_config(str(SINGLE_CKPT))
    backbone = _detect_backbone_type(config)
    kwargs = _common_lightning_kwargs(config, backbone, "single_instance")
    expected_keys = {
        "model_type",
        "backbone_type",
        "backbone_config",
        "head_configs",
        "pretrained_backbone_weights",
        "pretrained_head_weights",
        "init_weights",
        "lr_scheduler",
        "online_mining",
        "hard_to_easy_ratio",
        "min_hard_keypoints",
        "max_hard_keypoints",
        "loss_scale",
        "optimizer",
        "learning_rate",
        "amsgrad",
    }
    assert set(kwargs.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────────
# Tests — error paths
# ─────────────────────────────────────────────────────────────────────────


def test_load_model_assets_bad_path():
    with pytest.raises(FileNotFoundError):
        load_model_assets(["/nonexistent/model/dir"], device="cpu")


def test_load_model_assets_unsupported_type(tmp_path):
    """A path with a training config but an unrecognized model type."""
    from omegaconf import OmegaConf

    fake_config = OmegaConf.create(
        {
            "model_config": {
                "head_configs": {"unknown_type": {"confmaps": {}}},
                "backbone_config": {"unet": {"max_stride": 16}},
                "init_weights": "default",
            },
            "data_config": {
                "skeletons": {},
                "preprocessing": {},
            },
            "trainer_config": {
                "lr_scheduler": {},
                "optimizer_name": "adam",
                "optimizer": {"lr": 1e-3, "amsgrad": False},
                "online_hard_keypoint_mining": {
                    "online_mining": False,
                    "hard_to_easy_ratio": 2.0,
                    "min_hard_keypoints": 0,
                    "max_hard_keypoints": 0,
                    "loss_scale": 1.0,
                },
            },
        }
    )
    OmegaConf.save(fake_config, str(tmp_path / "training_config.yaml"))
    with pytest.raises((ValueError, KeyError)):
        load_model_assets([str(tmp_path)], device="cpu")
