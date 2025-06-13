"""Test TrainingModule classes."""

import numpy as np
from omegaconf import OmegaConf
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.training.lightning_modules import (
    TopDownCenteredInstanceLightningModule,
    SingleInstanceLightningModule,
    CentroidLightningModule,
    BottomUpLightningModule,
)
from torch.nn.functional import mse_loss


def test_topdown_centered_instance_model(config, tmp_path: str):

    # unet
    model = TopDownCenteredInstanceLightningModule(
        config=config,
        model_type="centered_instance",
        backbone_type="unet",
    )
    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_topdown_centered_instance_model_1/",
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["confidence_maps"]
    preds = model(input_["instance_image"])

    # check the output shape
    assert preds.shape == (1, 2, 80, 80)

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm)) < 1e-3

    # convnext with pretrained weights
    OmegaConf.update(
        config, "model_config.pre_trained_weights", "ConvNeXt_Tiny_Weights"
    )
    OmegaConf.update(config, "data_config.preprocessing.is_rgb", True)
    OmegaConf.update(config, "model_config.backbone_config.unet", None)
    OmegaConf.update(
        config,
        "model_config.backbone_config.convnext",
        {
            "in_channels": 3,
            "model_type": "tiny",
            "arch": None,
            "kernel_size": 3,
            "filters_rate": 2,
            "convs_per_block": 2,
            "up_interpolate": True,
            "stem_patch_kernel": 4,
            "stem_patch_stride": 2,
            "output_stride": 2,
            "max_stride": 16,
        },
    )
    model = TopDownCenteredInstanceLightningModule(
        config=config,
        model_type="centered_instance",
        backbone_type="convnext",
    )
    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_topdown_centered_instance_model_2/",
    )
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["confidence_maps"]
    preds = model(input_["instance_image"])

    # check the output shape
    assert preds.shape == (1, 2, 80, 80)
    assert all(
        np.abs(
            next(model.parameters())[0, 0, 0, :].detach().numpy()
            - np.array([-0.1019, -0.1258, -0.0777, -0.0484])
        )
        < 1e-4
    )


def test_centroid_model(config, tmp_path: str):
    """Test CentroidLightningModule training."""
    OmegaConf.update(
        config,
        "model_config.head_configs.centroid",
        config.model_config.head_configs.centered_instance,
    )
    del config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centroid["confmaps"].part_names

    model = CentroidLightningModule(
        config=config, model_type="centroid", backbone_type="unet"
    )

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_centroid_model_1/"
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["centroids_confidence_maps"]
    preds = model(input_["image"])

    # check the output shape
    assert preds.shape == (1, 1, 192, 192)

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm.squeeze(dim=1))) < 1e-3

    # torch dataset
    model = CentroidLightningModule(
        config=config, backbone_type="unet", model_type="centroid"
    )

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_centroid_model_2/"
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))
    input_cm = input_["centroids_confidence_maps"]
    preds = model(input_["image"])

    # check the output shape
    assert preds.shape == (1, 1, 192, 192)

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm.squeeze(dim=1))) < 1e-3


def test_single_instance_model(config, tmp_path: str):
    """Test the SingleInstanceLightningModule training."""
    head_config = config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.single_instance", head_config)
    del config.model_config.head_configs.single_instance.confmaps.anchor_part

    OmegaConf.update(config, "model_config.init_weights", "xavier")

    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_single_instance_model_1/",
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))
    model = SingleInstanceLightningModule(
        config=config,
        backbone_type="unet",
        model_type="single_instance",
    )

    img = input_["image"]
    img_shape = img.shape[-2:]
    preds = model(img)

    # check the output shape
    assert preds.shape == (
        1,
        2,
        int(
            img_shape[0]
            / config.model_config.head_configs.single_instance.confmaps.output_stride
        ),
        int(
            img_shape[1]
            / config.model_config.head_configs.single_instance.confmaps.output_stride
        ),
    )

    # check the loss value
    input_["confidence_maps"] = input_["confidence_maps"][:, :, :2, :, :]
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_["confidence_maps"].squeeze(dim=1))) < 1e-3

    # torch dataset
    OmegaConf.update(
        config,
        "trainer_config.save_ckpt_path",
        f"{tmp_path}/test_single_instance_model_2/",
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))
    model = SingleInstanceLightningModule(
        config=config,
        backbone_type="unet",
        model_type="single_instance",
    )

    img = input_["image"]
    img_shape = img.shape[-2:]
    preds = model(img)

    # check the output shape
    assert preds.shape == (
        1,
        2,
        int(
            img_shape[0]
            / config.model_config.head_configs.single_instance.confmaps.output_stride
        ),
        int(
            img_shape[1]
            / config.model_config.head_configs.single_instance.confmaps.output_stride
        ),
    )

    # check the loss value
    input_["confidence_maps"] = input_["confidence_maps"][:, :, :2, :, :]
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_["confidence_maps"].squeeze(dim=1))) < 1e-3


def test_bottomup_model(config, tmp_path: str):
    """Test BottomUp model training."""
    config_copy = config.copy()

    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": [("part1", "part2")],
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del config.model_config.head_configs.centered_instance
    config.model_config.head_configs.bottomup["pafs"] = paf
    config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_bottomup_model_1/"
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    input_ = next(iter(model_trainer.train_data_loader))

    model = BottomUpLightningModule(
        config=config, backbone_type="unet", model_type="bottomup"
    )

    preds = model(input_["image"])

    # check the output shape
    loss = model.training_step(input_, 0)
    assert preds["MultiInstanceConfmapsHead"].shape == (1, 2, 192, 192)
    assert preds["PartAffinityFieldsHead"].shape == (1, 2, 96, 96)

    # with edges as None
    config = config_copy
    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.bottomup", head_config)
    paf = {
        "edges": None,
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del config.model_config.head_configs.centered_instance
    config.model_config.head_configs.bottomup["pafs"] = paf
    config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    OmegaConf.update(
        config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_bottomup_model_2/"
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer(config)
    model_trainer._create_data_loaders_torch_dataset()
    skeletons = model_trainer.skeletons
    input_ = next(iter(model_trainer.train_data_loader))

    model = BottomUpLightningModule(
        config=model_trainer.config,
        backbone_type="unet",
        model_type="bottomup",
    )

    preds = model(input_["image"])

    # check the output shape
    loss = model.training_step(input_, 0)
    assert preds["MultiInstanceConfmapsHead"].shape == (1, 2, 192, 192)
    assert preds["PartAffinityFieldsHead"].shape == (1, 2, 96, 96)
