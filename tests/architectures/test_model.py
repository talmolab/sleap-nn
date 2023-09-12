import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.architectures.model import Model


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
            "part_names": [f"{i}" for i in range(13)],
            "sigma": 5.0,
            "output_stride": 1,
            "loss_weight": 1.0,
        }
    )

    model = Model(
        model_config=base_unet_model_config, head_config=base_unet_head_config
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert z.shape == (1, 13, 192, 192)
    assert z.dtype == torch.float32

    model = Model.from_config(
        model_config=base_unet_model_config, head_config=base_unet_head_config
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert z.shape == (1, 13, 192, 192)
    assert z.dtype == torch.float32
