import sleap_io as sio
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import Predictor
from loguru import logger
from _pytest.logging import LogCaptureFixture


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


def test_topdown_predictor(
    caplog,
    centered_instance_video,
    minimal_instance,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
    minimal_instance_bottomup_ckpt,
    minimal_instance_single_instance_ckpt,
    tmp_path,
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    # check loading diff head ckpt
    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt],
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    predictor.make_pipeline(
        centered_instance_video.as_posix(),
        frames=[x for x in range(100)],
    )

    output = predictor.predict(
        make_labels=True,
    )
    assert isinstance(output, sio.Labels)
    assert len(output) == 100

    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": False,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_centered_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # check loading diff head ckpt for centroid
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centroid
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_centered_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # check loading diff head ckpt for centroid
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": False,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centroid
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_centered_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # check loading diff head ckpt for centroid
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centroid
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_centered_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # check loading diff head ckpt for centroid
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": False,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centroid
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_centered_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # check loading diff head ckpt for centroid
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)


def test_multiclass_topdown_predictor(
    caplog,
    minimal_instance,
    minimal_instance_multi_class_topdown_ckpt,
    minimal_instance_centroid_ckpt,
    minimal_instance_bottomup_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_topdown_ckpt],
        peak_threshold=0.03,
        preprocess_config=OmegaConf.create(preprocess_config),
    )
    predictor.make_pipeline(
        minimal_instance.as_posix(),
    )
    output = predictor.predict(
        make_labels=True,
    )
    assert isinstance(output, sio.Labels)
    assert len(output) == 1

    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": False,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_topdown_ckpt],
        backbone_ckpt_path=Path(minimal_instance_multi_class_topdown_ckpt)
        / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_multi_class_topdown_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_topdown_ckpt],
        backbone_ckpt_path=Path(minimal_instance_multi_class_topdown_ckpt)
        / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_multi_class_topdown_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_topdown_ckpt],
        backbone_ckpt_path=Path(minimal_instance_multi_class_topdown_ckpt)
        / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_multi_class_topdown_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_topdown_ckpt],
        backbone_ckpt_path=Path(minimal_instance_multi_class_topdown_ckpt)
        / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_multi_class_topdown_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_topdown_ckpt],
        backbone_ckpt_path=Path(minimal_instance_multi_class_topdown_ckpt)
        / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_multi_class_topdown_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.instance_peaks.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)


def test_single_instance_predictor(
    small_robot_minimal_video,
    small_robot_minimal,
    minimal_instance_single_instance_ckpt,
):
    """Test SingleInstancePredictor module."""

    # check loading diff head ckpt
    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
    predictor = Predictor.from_model_paths(
        [minimal_instance_single_instance_ckpt],
        peak_threshold=0.3,
        preprocess_config=OmegaConf.create(preprocess_config),
    )
    predictor.make_pipeline(
        small_robot_minimal_video.as_posix(),
        frames=[x for x in range(100)],
    )

    # run predict
    output = predictor.predict(
        make_labels=False,
    )
    assert isinstance(output, list)
    assert len(output) == 25
    assert output[0]["pred_instance_peaks"].shape[0] == 4
    assert isinstance(output[0], dict)
    assert "pred_confmaps" not in output[0].keys()


def test_bottomup_predictor(
    caplog,
    minimal_instance,
    minimal_instance_bottomup_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
):
    """Test BottomUpPredictor module."""
    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
    predictor = Predictor.from_model_paths(
        [minimal_instance_bottomup_ckpt],
        peak_threshold=0.05,
        preprocess_config=OmegaConf.create(preprocess_config),
    )
    predictor.make_pipeline(
        centered_instance_video.as_posix(),
        frames=[x for x in range(100)],
    )
    output = predictor.predict(
        make_labels=True,
    )
    assert isinstance(output, sio.Labels)
    assert len(output) == 100

    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
    }

    # load only backbone and head ckpt as None
    predictor = Predictor.from_model_paths(
        [minimal_instance_bottomup_ckpt],
        backbone_ckpt_path=Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.05,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_centered_instance_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.torch_model.model.parameters())[0, 0, :]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)


def test_multi_class_bottomup_predictor(
    caplog,
    centered_instance_video,
    minimal_instance,
    minimal_instance_multi_class_bottomup_ckpt,
    minimal_instance_multi_class_topdown_ckpt,
):
    """Test BottomUpPredictor module."""
    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_bottomup_ckpt],
        peak_threshold=0.03,
        preprocess_config=OmegaConf.create(preprocess_config),
    )
    predictor.make_pipeline(
        centered_instance_video.as_posix(),
        frames=[x for x in range(100)],
    )
    output = predictor.predict(
        make_labels=True,
    )
    assert isinstance(output, sio.Labels)
    assert len(output) == 100

    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
    }

    # load only backbone and head ckpt as None
    predictor = Predictor.from_model_paths(
        [minimal_instance_multi_class_bottomup_ckpt],
        backbone_ckpt_path=Path(minimal_instance_multi_class_topdown_ckpt)
        / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_multi_class_topdown_ckpt) / "best.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    backbone_ckpt = (
        ckpt["state_dict"][
            "model.backbone.encoders.0.encoder_stack.0.blocks.stack0_enc0_conv0.weight"
        ][0, 0, :]
        .cpu()
        .numpy()
    )

    model_weights = (
        next(predictor.inference_model.torch_model.model.parameters())[0, 0, :]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)
