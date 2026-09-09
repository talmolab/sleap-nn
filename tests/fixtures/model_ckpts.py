"""Dataset fixtures for unit testing."""

from pathlib import Path
from omegaconf import OmegaConf

import pytest


@pytest.fixture
def sleap_nn_model_ckpts_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/assets/model_ckpts"


@pytest.fixture
def minimal_instance_centered_instance_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_centered_instance"


@pytest.fixture
def minimal_instance_single_instance_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_single_instance"


@pytest.fixture
def minimal_instance_centroid_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for trained model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_centroid"


@pytest.fixture
def minimal_instance_bottomup_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for BottomUP model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_bottomup"


@pytest.fixture
def minimal_instance_multi_class_bottomup_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for BottomUp ID model."""
    return Path(sleap_nn_model_ckpts_dir) / "minimal_instance_multiclass_bottomup"


@pytest.fixture
def minimal_instance_multi_class_topdown_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for topdown ID model."""
    return (
        Path(sleap_nn_model_ckpts_dir) / "minimal_instance_multiclass_centered_instance"
    )


@pytest.fixture
def single_instance_with_metrics_ckpt(sleap_nn_model_ckpts_dir):
    """Checkpoint file for topdown ID model."""
    return Path(sleap_nn_model_ckpts_dir) / "single_instance_with_metrics"


# Embedding (re-ID) model fixtures. Kept here rather than in
# `tests/export/conftest.py` so the inference and CLI suites can build the same
# loadable model directory instead of each growing its own copy of this config.
def _build_embedding_training_config(backbone_type="unet", in_channels=1, crop_size=32):
    """Build a minimal, loadable training config for an embedding model."""
    from omegaconf import OmegaConf

    embedding_dim = 16
    max_stride = 16
    backbone_leaf = {
        "in_channels": in_channels,
        "kernel_size": 3,
        "filters": 8,
        "filters_rate": 1.5,
        "max_stride": max_stride,
        "stem_stride": None,
        "middle_block": True,
        "up_interpolate": True,
        "stacks": 1,
        "convs_per_block": 2,
        "output_stride": 2,
    }
    objective = {
        "positives": {"scope": "global_id"},
        "negatives": {
            "sources": ["in_batch"],
            "exclude_same_track": True,
            "restrict_same_video": False,
        },
        "loss": {"name": "supcon", "temperature": 0.1, "margin": 0.2},
        "sampler": {"kind": "pk", "groups_per_batch": 2, "samples_per_group": 4},
        "use_projection": True,
        "projection_dim": embedding_dim,
    }
    config = OmegaConf.create(
        {
            "data_config": {
                "preprocessing": {
                    "scale": 1.0,
                    "crop_size": crop_size,
                    "max_height": 64,
                    "max_width": 64,
                    "ensure_rgb": in_channels == 3,
                    "ensure_grayscale": in_channels == 1,
                },
                "skeletons": None,
            },
            "model_config": {
                "backbone_type": backbone_type,
                "backbone_config": {backbone_type: backbone_leaf},
                "head_configs": {
                    "embedding": {
                        "embedding": {
                            "embedding_dim": embedding_dim,
                            "num_fc_layers": 1,
                            "num_fc_units": 32,
                            "pool": "gem",
                            "normalize": True,
                            "output_stride": max_stride,
                            "loss_weight": 1.0,
                            "freeze_backbone": False,
                            "objective": objective,
                        }
                    },
                },
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "init_weights": "xavier",
            },
            "trainer_config": {
                "lr_scheduler": None,
                "optimizer_name": "Adam",
                "optimizer": {"lr": 1e-3, "amsgrad": False},
                "online_hard_keypoint_mining": {
                    "online_mining": False,
                    "hard_to_easy_ratio": 2.0,
                    "min_hard_keypoints": 2,
                    "max_hard_keypoints": None,
                    "loss_scale": 5.0,
                },
            },
        }
    )
    return config


@pytest.fixture
def minimal_embedding_model_dir(tmp_path):
    """A tiny (random-weights) embedding model directory: best.ckpt + config.

    Mirrors what a real training run leaves on disk so the export CLI can load and
    export it. Built in-process (no training) for speed and portability.
    """
    import torch
    from omegaconf import OmegaConf

    from sleap_nn.training.lightning_modules import EmbeddingLightningModule

    config = _build_embedding_training_config()
    module = EmbeddingLightningModule(
        model_type="embedding",
        backbone_type="unet",
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        init_weights="xavier",
    ).eval()

    model_dir = tmp_path / "embedding_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": {},
            "pytorch-lightning_version": "2.0.0",
            "epoch": 0,
            "global_step": 0,
        },
        model_dir / "best.ckpt",
    )
    OmegaConf.save(config, model_dir / "training_config.yaml")
    return model_dir
