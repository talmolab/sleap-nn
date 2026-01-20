"""Test TrainingModule classes."""

import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf
import wandb
from sleap_nn.data.custom_datasets import (
    get_train_val_dataloaders,
    get_train_val_datasets,
)
import sleap_io as sio
from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.training.lightning_modules import (
    TopDownCenteredInstanceLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
    SingleInstanceLightningModule,
    CentroidLightningModule,
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
    LightningModel,
)
from torch.nn.functional import mse_loss
import torch
import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger
from sleap_nn.config.training_job_config import TrainingJobConfig


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True)
def cleanup_wandb():
    """Ensure wandb runs in offline mode and is cleaned up after each test.

    This fixture:
    1. Sets WANDB_MODE=offline to prevent network hangs on CI
    2. Cleans up any active wandb run after the test to prevent state leakage
    """
    # Save original mode and force offline to prevent network hangs on CI
    original_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"

    yield

    # Finish any active wandb run to prevent contamination between tests
    if wandb.run is not None:
        wandb.finish()

    # Restore original WANDB_MODE
    if original_mode is not None:
        os.environ["WANDB_MODE"] = original_mode
    else:
        os.environ.pop("WANDB_MODE", None)


def test_topdown_centered_instance_model(
    config, tmp_path: str, minimal_instance_centered_instance_ckpt
):

    # unet
    model = TopDownCenteredInstanceLightningModule(
        model_type="centered_instance",
        backbone_config=config.model_config.backbone_config,
        backbone_type="unet",
        head_configs=config.model_config.head_configs,
        pretrained_backbone_weights=(
            Path(minimal_instance_centered_instance_ckpt) / "best.ckpt"
        ).as_posix(),
        pretrained_head_weights=config.model_config.pretrained_head_weights,
        init_weights=config.model_config.init_weights,
        lr_scheduler=config.trainer_config.lr_scheduler,
        optimizer="AdamW",
    )
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(
        config, "trainer_config.run_name", "test_topdown_centered_instance_model_1"
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
    input_cm = input_["confidence_maps"]
    preds = model(input_["instance_image"])

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm[0])) < 1e-3

    # check the output shape
    assert preds.shape == (1, 2, 80, 80)

    # convnext with pretrained weights
    OmegaConf.update(config, "model_config.backbone_config.unet", None)
    OmegaConf.update(
        config,
        "model_config.backbone_config.convnext",
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
            "output_stride": 2,
            "max_stride": 32,
        },
    )
    OmegaConf.update(
        config,
        "model_config.backbone_config.convnext.pre_trained_weights",
        "ConvNeXt_Tiny_Weights",
    )
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    model = LightningModel.get_lightning_model_from_config(
        config=model_trainer.config
    )  # passing model trainer config modifies the in_channels according to the pretrained model weights
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(
        config, "trainer_config.run_name", "test_topdown_centered_instance_model_2"
    )
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
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
        model_type="centroid",
        backbone_config="unet_medium_rf",
        backbone_type="unet",
        head_configs=config.model_config.head_configs,
    )

    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_centroid_model_1")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
    input_cm = input_["centroids_confidence_maps"]
    preds = model(input_["image"])

    # check the output shape
    assert preds.shape == (1, 1, 192, 192)

    # check the loss value
    loss = model.training_step(input_, 0)
    assert abs(loss - mse_loss(preds, input_cm.squeeze(dim=1))) < 1e-3

    # torch dataset
    model = LightningModel.get_lightning_model_from_config(config=config)

    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_centroid_model_2")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
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
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_single_instance_model_1")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
    model = SingleInstanceLightningModule(
        model_type="single_instance",
        backbone_config="unet_medium_rf",
        backbone_type="unet",
        head_configs=config.model_config.head_configs,
        lr_scheduler=None,
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
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_single_instance_model_2")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
    model = LightningModel.get_lightning_model_from_config(config=config)

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
        "edges": [("A", "B")],
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.bottomup["confmaps"].anchor_part
    del config.model_config.head_configs.centered_instance
    config.model_config.head_configs.bottomup["pafs"] = paf
    config.model_config.head_configs.bottomup.confmaps.loss_weight = 1.0

    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_bottomup_model_1")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))

    model = LightningModel.get_lightning_model_from_config(config=config)

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

    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_bottomup_model_2")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(config)
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    skeletons = model_trainer.skeletons
    input_ = next(iter(train_data_loader))

    model = LightningModel.get_lightning_model_from_config(config=model_trainer.config)

    preds = model(input_["image"])

    # check the output shape
    loss = model.training_step(input_, 0)
    assert preds["MultiInstanceConfmapsHead"].shape == (1, 2, 192, 192)
    assert preds["PartAffinityFieldsHead"].shape == (1, 2, 96, 96)


def test_multi_class_bottomup_model(config, tmp_path: str, minimal_instance):
    """Test BottomUp model training."""
    config_copy = config.copy()
    tracked_labels = sio.load_slp(minimal_instance)
    tracks = 0
    for lf in tracked_labels:
        for instance in lf.instances:
            instance.track = sio.Track(f"{tracks}")
            tracks += 1
    tracked_labels.update()

    head_config = config.model_config.head_configs.centered_instance
    OmegaConf.update(
        config, "model_config.head_configs.multi_class_bottomup", head_config
    )
    class_maps = {
        "classes": None,
        "sigma": 4,
        "output_stride": 4,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.multi_class_bottomup["confmaps"].anchor_part
    del config.model_config.head_configs.centered_instance
    config.model_config.head_configs.multi_class_bottomup["class_maps"] = class_maps
    config.model_config.head_configs.multi_class_bottomup.confmaps.loss_weight = 1.0

    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(
        config, "trainer_config.run_name", "test_multi_class_bottomup_model_1"
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    model_trainer = ModelTrainer.get_model_trainer_from_config(
        config, train_labels=[tracked_labels], val_labels=[tracked_labels]
    )
    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))

    model = LightningModel.get_lightning_model_from_config(config=model_trainer.config)

    preds = model(input_["image"])

    # check the output shape
    loss = model.training_step(input_, 0)
    assert preds["MultiInstanceConfmapsHead"].shape == (1, 2, 192, 192)
    assert preds["ClassMapsHead"].shape == (1, 2, 96, 96)


def test_mutli_class_topdown_centered(config, tmp_path: str, minimal_instance):
    # unet
    tracked_labels = sio.load_slp(minimal_instance)
    tracks = 0
    for lf in tracked_labels:
        for instance in lf.instances:
            instance.track = sio.Track(f"{tracks}")
            tracks += 1
    tracked_labels.update()

    confmaps = config.model_config.head_configs.centered_instance
    class_vectors = {
        "classes": None,
        "num_fc_layers": 1,
        "num_fc_units": 64,
        "output_stride": 16,
        "loss_weight": 1.0,
    }
    del config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.multi_class_topdown", confmaps)
    config.model_config.head_configs["multi_class_topdown"][
        "class_vectors"
    ] = class_vectors
    config.model_config.head_configs.multi_class_topdown.confmaps.loss_weight = 1.0

    model_trainer = ModelTrainer.get_model_trainer_from_config(
        config, train_labels=[tracked_labels], val_labels=[tracked_labels]
    )

    model = LightningModel.get_lightning_model_from_config(config=model_trainer.config)
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(
        config,
        "trainer_config.run_name",
        "test_topdown_centered_instance_multiclass_model_1",
    )
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")

    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_data_loader, val_data_loader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
    )
    input_ = next(iter(train_data_loader))
    input_cm = input_["confidence_maps"]
    preds = model(input_["instance_image"])

    # check the output shape
    assert preds["CenteredInstanceConfmapsHead"].shape == (1, 2, 80, 80)
    assert preds["ClassVectorsHead"].shape == (1, 2)


def test_incorrect_model_type(config, caplog, tmp_path: str):
    """Test the SingleInstanceLightningModule training."""
    head_config = config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centered_instance
    config["model_config"]["head_configs"]["topdown"] = head_config
    OmegaConf.update(config, "model_config.head_configs.topdown", head_config)
    del config.model_config.head_configs.topdown.confmaps.anchor_part

    OmegaConf.update(config, "model_config.init_weights", "xavier")

    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(config, "trainer_config.run_name", "test_single_instance_model_1")
    OmegaConf.update(config, "data_config.data_pipeline_fw", "torch_dataset")
    with pytest.raises(Exception):
        _ = LightningModel.get_lightning_model_from_config(config)
    assert "Incorrect model type." in caplog.text


def test_load_trained_ckpts(config, tmp_path, minimal_instance_centered_instance_ckpt):
    """Test loading trained weights for backbone and head layers."""
    if torch.cuda.is_available():
        OmegaConf.update(config, "trainer_config.trainer_accelerator", "cuda")
    else:
        OmegaConf.update(config, "trainer_config.trainer_accelerator", "cpu")
    OmegaConf.update(config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(
        config, "trainer_config.run_name", "test_model_trainer_load_trained_ckpts"
    )
    OmegaConf.update(config, "trainer_config.save_ckpt", True)
    OmegaConf.update(config, "trainer_config.use_wandb", True)
    OmegaConf.update(config, "data_config.preprocessing.crop_size", None)
    OmegaConf.update(config, "data_config.preprocessing.min_crop_size", 100)
    OmegaConf.update(
        config,
        "model_config.pretrained_backbone_weights",
        (Path(minimal_instance_centered_instance_ckpt) / "best.ckpt").as_posix(),
    )
    OmegaConf.update(
        config,
        "model_config.pretrained_head_weights",
        (Path(minimal_instance_centered_instance_ckpt) / "best.ckpt").as_posix(),
    )

    # check loading trained weights for backbone
    ckpt = torch.load(
        (Path(minimal_instance_centered_instance_ckpt) / "best.ckpt").as_posix(),
        map_location="cuda" if torch.cuda.is_available() else "cpu",
        weights_only=False,
    )
    first_layer_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    # load head ckpts
    head_layer_ckpt = (
        ckpt["state_dict"]["model.head_layers.0.CenteredInstanceConfmapsHead.0.weight"][
            0, 0, :
        ]
        .cpu()
        .numpy()
    )

    lightning_module = LightningModel.get_lightning_model_from_config(config=config)
    model_ckpt = next(lightning_module.parameters())[0, 0, :].detach().cpu().numpy()

    assert np.all(np.abs(first_layer_ckpt - model_ckpt) < 1e-6)

    model_ckpt = (
        next(lightning_module.model.head_layers.parameters())[0, 0, :]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(head_layer_ckpt - model_ckpt) < 1e-6)


def test_load_trained_keras_weights(
    sleap_centered_instance_model_path, minimal_instance, caplog
):
    """Test loading trained keras (.h5) weights for model initialization."""
    sleap_nn_config = TrainingJobConfig.load_sleap_config(
        Path(sleap_centered_instance_model_path) / "training_config.json"
    )
    sleap_nn_config.model_config.pretrained_backbone_weights = (
        Path(sleap_centered_instance_model_path) / "best_model.h5"
    )
    sleap_nn_config.model_config.pretrained_head_weights = (
        Path(sleap_centered_instance_model_path) / "best_model.h5"
    )

    trainer = ModelTrainer.get_model_trainer_from_config(
        config=sleap_nn_config,
        train_labels=[sio.load_slp(minimal_instance)],
        val_labels=[sio.load_slp(minimal_instance)],
    )

    lightning_module = LightningModel.get_lightning_model_from_config(
        config=trainer.config
    )
    assert "Loading backbone weights from" in caplog.text
    assert "Successfully loaded 28/28 weights from legacy model" in caplog.text
    assert "Loading head weights from" in caplog.text
    assert "Successfully loaded 2/2 weights from legacy model" in caplog.text

    with pytest.raises(Exception):
        sleap_nn_config.model_config.pretrained_backbone_weights = (
            Path(sleap_centered_instance_model_path) / "best_model.pth"
        )
        trainer = ModelTrainer.get_model_trainer_from_config(
            config=sleap_nn_config,
            train_labels=[sio.load_slp(minimal_instance)],
            val_labels=[sio.load_slp(minimal_instance)],
        )
        _ = LightningModel.get_lightning_model_from_config(config=trainer.config)
    assert "Unsupported file extension for pretrained backbone weights." in caplog.text

    with pytest.raises(Exception):
        sleap_nn_config.model_config.pretrained_backbone_weights = None
        sleap_nn_config.model_config.pretrained_head_weights = (
            Path(sleap_centered_instance_model_path) / "best_model.pth"
        )
        trainer = ModelTrainer.get_model_trainer_from_config(
            config=sleap_nn_config,
            train_labels=[sio.load_slp(minimal_instance)],
            val_labels=[sio.load_slp(minimal_instance)],
        )
        _ = LightningModel.get_lightning_model_from_config(config=trainer.config)
    assert "Unsupported file extension for pretrained head weights." in caplog.text
