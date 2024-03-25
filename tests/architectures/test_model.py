import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.architectures.model import Model, get_backbone, get_head
from sleap_nn.architectures.heads import Head


def test_get_backbone():
    # unet
    base_unet_model_config = OmegaConf.create(
        {
            "backbone_type": "unet",
            "backbone_config": {
                "init_weights": "default",
                "in_channels": 1,
                "kernel_size": 3,
                "filters": 16,
                "filters_rate": 2,
                "down_blocks": 4,
                "up_blocks": 3,
                "convs_per_block": 2,
            },
        }
    )

    backbone = get_backbone(
        base_unet_model_config.backbone_type, base_unet_model_config.backbone_config
    )
    assert isinstance(backbone, torch.nn.Module)

    # convnext
    base_convnext_model_config = OmegaConf.create(
        {
            "backbone_type": "convnext",
            "backbone_config": {
                "init_weights": "default",
                "pretrained_weights": "",
                "in_channels": 1,
                "kernel_size": 3,
                "filters_rate": 2,
                "up_blocks": 2,
                "convs_per_block": 2,
                "arch": {
                    "depths": [3, 3, 9, 3],
                    "channels": [96, 192, 384, 768],
                },
                "stem_patch_kernel": 4,
                "stem_patch_stride": 2,
            },
        }
    )

    backbone = get_backbone(
        base_convnext_model_config.backbone_type,
        base_convnext_model_config.backbone_config,
    )
    assert isinstance(backbone, torch.nn.Module)

    with pytest.raises(KeyError):
        _ = get_backbone("invalid_input", base_unet_model_config.backbone_config)


def test_get_head():
    base_unet_head_config = OmegaConf.create(
        {
            "head_type": "SingleInstanceConfmapsHead",
            "head_config": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            },
        }
    )

    head = get_head(base_unet_head_config.head_type, base_unet_head_config.head_config)
    assert isinstance(head, Head)

    with pytest.raises(KeyError):
        _ = get_head("invalid_input", base_unet_head_config.head_config)


def test_unet_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_unet_model_config = OmegaConf.create(
        {
            "backbone_type": "unet",
            "backbone_config": {
                "in_channels": 1,
                "kernel_size": 3,
                "filters": 64,
                "filters_rate": 2,
                "down_blocks": 4,
                "up_blocks": 4,
                "convs_per_block": 2,
            },
        }
    )

    base_unet_head_config = OmegaConf.create(
        {
            "head_type": "SingleInstanceConfmapsHead",
            "head_config": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            },
        }
    )

    model = Model(
        backbone_config=base_unet_model_config, head_configs=[base_unet_head_config]
    ).to(device)

    assert model.backbone_config == base_unet_model_config
    assert model.head_configs == [base_unet_head_config]

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z[base_unet_head_config.head_type].shape == (1, 13, 192, 192)
    assert z[base_unet_head_config.head_type].dtype == torch.float32

    model = Model.from_config(
        backbone_config=base_unet_model_config, head_configs=[base_unet_head_config]
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z[base_unet_head_config.head_type].shape == (1, 13, 192, 192)
    assert z[base_unet_head_config.head_type].dtype == torch.float32


def test_convnext_model():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    base_convnext_model_config = OmegaConf.create(
        {
            "backbone_type": "convnext",
            "backbone_config": {
                "init_weights": "default",
                "pretrained_weights": "",
                "in_channels": 1,
                "kernel_size": 3,
                "filters_rate": 2,
                "up_blocks": 4,
                "convs_per_block": 2,
                "arch": {
                    "depths": [3, 3, 9, 3],
                    "channels": [16, 32, 64, 128],
                },
                "stem_patch_kernel": 4,
                "stem_patch_stride": 2,
            },
        }
    )

    base_convnext_head_config = OmegaConf.create(
        {
            "head_type": "SingleInstanceConfmapsHead",
            "head_config": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            },
        }
    )

    model = Model(
        backbone_config=base_convnext_model_config,
        head_configs=[base_convnext_head_config],
    ).to(device)

    assert model.backbone_config == base_convnext_model_config
    assert model.head_configs == [base_convnext_head_config]

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z[base_convnext_head_config.head_type].shape == (1, 13, 192, 192)
    assert z[base_convnext_head_config.head_type].dtype == torch.float32

    model = Model.from_config(
        backbone_config=base_convnext_model_config,
        head_configs=[base_convnext_head_config],
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z[base_convnext_head_config.head_type].shape == (1, 13, 192, 192)
    assert z[base_convnext_head_config.head_type].dtype == torch.float32
