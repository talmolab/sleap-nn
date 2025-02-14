import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig

from sleap_nn.architectures.model import Model, get_backbone, get_head
from sleap_nn.architectures.heads import Head


def test_get_backbone():
    """Test `get_backbone` function."""
    # unet
    base_unet_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 16,
            "filters_rate": 2,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "output_stride": 1,
        }
    )

    backbone = get_backbone(
        "unet",
        base_unet_model_config,
    )
    assert isinstance(backbone, torch.nn.Module)

    # convnext
    base_convnext_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 1,
        }
    )

    backbone = get_backbone(
        "convnext",
        base_convnext_model_config,
    )
    assert isinstance(backbone, torch.nn.Module)

    with pytest.raises(KeyError):
        _ = get_backbone("invalid_input", base_unet_model_config)

    # swint
    base_convnext_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "patch_size": [4, 4],
            "window_size": [7, 7],
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_stride": 4,
            "output_stride": 1,
        }
    )

    backbone = get_backbone(
        "swint",
        base_convnext_model_config,
    )
    assert isinstance(backbone, torch.nn.Module)

    with pytest.raises(KeyError):
        _ = get_backbone("invalid_input", base_unet_model_config)


def test_get_head():
    base_unet_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    head = get_head("single_instance", base_unet_head_config)
    assert isinstance(head[0], Head)

    with pytest.raises(Exception):
        _ = get_head("invalid_input", base_unet_head_config)


def test_unet_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_unet_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 16,
            "filters_rate": 2,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "output_stride": 1,
        }
    )

    base_unet_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="unet",
        backbone_config=base_unet_model_config,
        head_configs=base_unet_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_unet_model_config
    assert model.head_configs == base_unet_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    # filter rate = 1.5
    base_unet_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 16,
            "filters_rate": 1.5,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": True,
            "output_stride": 1,
        }
    )

    base_unet_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="unet",
        backbone_config=base_unet_model_config,
        head_configs=base_unet_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_unet_model_config
    assert model.head_configs == base_unet_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    # upsampling stack with TransposeConv layers
    base_unet_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "kernel_size": 3,
            "filters": 16,
            "filters_rate": 1.5,
            "max_stride": 16,
            "convs_per_block": 2,
            "stacks": 1,
            "stem_stride": None,
            "middle_block": True,
            "up_interpolate": False,
            "output_stride": 1,
        }
    )

    base_unet_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="unet",
        backbone_config=base_unet_model_config,
        head_configs=base_unet_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_unet_model_config
    assert model.head_configs == base_unet_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32


def test_convnext_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_convnext_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 1,
        }
    )

    base_convnext_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="convnext",
        backbone_config=base_convnext_model_config,
        head_configs=base_convnext_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_convnext_model_config
    assert model.head_configs == base_convnext_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    model = Model.from_config(
        backbone_type="convnext",
        backbone_config=base_convnext_model_config,
        head_configs=base_convnext_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    # stride = 4
    base_convnext_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 4,
            "output_stride": 1,
        }
    )

    base_convnext_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="convnext",
        backbone_config=base_convnext_model_config,
        head_configs=base_convnext_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_convnext_model_config
    assert model.head_configs == base_convnext_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    model = Model.from_config(
        backbone_type="convnext",
        backbone_config=base_convnext_model_config,
        head_configs=base_convnext_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    # transposeconv as upsampling stack
    base_convnext_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": False,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 4,
            "output_stride": 1,
        }
    )

    base_convnext_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="convnext",
        backbone_config=base_convnext_model_config,
        head_configs=base_convnext_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_convnext_model_config
    assert model.head_configs == base_convnext_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    model = Model.from_config(
        backbone_type="convnext",
        backbone_config=base_convnext_model_config,
        head_configs=base_convnext_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32


def test_swint_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # stride = 4
    base_swint_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "patch_size": [4, 4],
            "window_size": [7, 7],
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_stride": 4,
            "output_stride": 1,
        }
    )

    base_swint_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="swint",
        backbone_config=base_swint_model_config,
        head_configs=base_swint_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_swint_model_config
    assert model.head_configs == base_swint_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    model = Model.from_config(
        backbone_type="swint",
        backbone_config=base_swint_model_config,
        head_configs=base_swint_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    # transposeConv for upsampling stack
    base_swint_model_config = OmegaConf.create(
        {
            "in_channels": 1,
            "model_type": "tiny",
            "arch": None,
            "patch_size": [4, 4],
            "window_size": [7, 7],
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": False,
            "stem_patch_stride": 4,
            "stem_stride": None,
            "output_stride": 1,
        }
    )

    base_swint_head_config = OmegaConf.create(
        {
            "confmaps": {
                "part_names": [f"{i}" for i in range(13)],
                "sigma": 5.0,
                "output_stride": 1,
                "loss_weight": 1.0,
            }
        }
    )

    model = Model(
        backbone_type="swint",
        backbone_config=base_swint_model_config,
        head_configs=base_swint_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    assert model.backbone_config == base_swint_model_config
    assert model.head_configs == base_swint_head_config

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32

    model = Model.from_config(
        backbone_type="swint",
        backbone_config=base_swint_model_config,
        head_configs=base_swint_head_config,
        input_expand_channels=1,
        model_type="single_instance",
    ).to(device)

    x = torch.rand(1, 1, 192, 192).to(device)
    model.eval()

    with torch.no_grad():
        z = model(x)

    assert type(z) is dict
    assert len(z.keys()) == 1
    assert z["SingleInstanceConfmapsHead"].shape == (1, 13, 192, 192)
    assert z["SingleInstanceConfmapsHead"].dtype == torch.float32
