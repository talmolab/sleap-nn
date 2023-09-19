import torch
from omegaconf import OmegaConf

from sleap_nn.architectures.model import Model, get_backbone, get_head
from sleap_nn.architectures.heads import Head


def test_get_backbone():
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

    backbone = get_backbone(
        base_unet_model_config.backbone_type, base_unet_model_config.backbone_config
    )
    assert isinstance(backbone, torch.nn.Module)


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

    assert type(z) is list
    assert len(z) == 1
    assert z[0].shape == (1, 13, 192, 192)
    assert z[0].dtype == torch.float32

    model = Model.from_config(
        backbone_config=base_unet_model_config, head_configs=[base_unet_head_config]
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is list
    assert len(z) == 1
    assert z[0].shape == (1, 13, 192, 192)
    assert z[0].dtype == torch.float32
