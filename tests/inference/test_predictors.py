import shutil
import sleap_io as sio
from pathlib import Path
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import Predictor, _filter_user_labeled_frames
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
        "scale": None,
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
        "scale": None,
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


def _scale_only_preprocess_config(scale):
    """A ``preprocess_config`` overriding only ``scale`` (rest ``None``)."""
    return OmegaConf.create(
        {
            "ensure_rgb": None,
            "ensure_grayscale": None,
            "crop_size": None,
            "max_width": None,
            "max_height": None,
            "scale": scale,
        }
    )


def test_single_instance_predictor_input_scale_override_reaches_inference_model(
    minimal_instance_single_instance_ckpt,
):
    """A ``preprocess_config.scale`` override must reach the inference model.

    Regression test: every legacy ``*Predictor._initialize_inference_model``
    used to build its ``input_scale=`` kwarg from the *training config's*
    baked-in scale (e.g. ``self.confmap_config.data_config.preprocessing.scale``)
    instead of the already-resolved ``self.preprocess_config.scale`` -- so an
    explicit override was applied to the forward image resize (via
    ``self.preprocess_config["scale"]`` elsewhere) but NOT to the coordinate
    rescale-back step, corrupting output coordinates. Fixed to read
    ``self.preprocess_config.scale`` uniformly.
    """
    override = 0.37
    predictor = Predictor.from_model_paths(
        [minimal_instance_single_instance_ckpt],
        preprocess_config=_scale_only_preprocess_config(override),
    )
    assert predictor.inference_model.input_scale == override


def test_bottomup_predictor_input_scale_override_reaches_inference_model(
    minimal_instance_bottomup_ckpt,
):
    override = 0.37
    predictor = Predictor.from_model_paths(
        [minimal_instance_bottomup_ckpt],
        preprocess_config=_scale_only_preprocess_config(override),
    )
    assert predictor.inference_model.input_scale == override


def test_topdown_predictor_input_scale_override_reaches_both_stages(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
):
    """Both the centroid-crop and centered-instance stages must see the override."""
    override = 0.37
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt, minimal_instance_centered_instance_ckpt],
        preprocess_config=_scale_only_preprocess_config(override),
    )
    assert predictor.inference_model.centroid_crop.input_scale == override
    assert predictor.inference_model.instance_peaks.input_scale == override


def _copy_ckpt_with_scale(src: Path, dst: Path, scale: float) -> Path:
    """Copy a checkpoint dir to *dst*, overriding ``data_config.preprocessing.scale``."""
    shutil.copytree(src, dst)
    cfg = OmegaConf.load(str(dst / "training_config.yaml"))
    cfg.data_config.preprocessing.scale = scale
    OmegaConf.save(cfg, str(dst / "training_config.yaml"))
    return dst


def test_topdown_predictor_default_scale_is_per_stage_not_shared(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    tmp_path,
):
    """Each stage must default to ITS OWN trained scale, not a shared value.

    Regression test for the `track`-side half of #725: `TopDownPredictor`
    used to build both `centroid_crop_layer.input_scale` and
    `instance_peaks_layer.input_scale` from the same shared
    `self.preprocess_config.scale`, resolved from whichever config
    (`centroid_config` or `confmap_config`) happened to fill it first. Both
    fixture checkpoints train at scale=1.0 by default, which never exercises
    this path, so the scales are patched apart here. `Predictor.from_model_paths`
    also used to call `TopDownPredictor.from_trained_models` twice (a discarded
    centroid-only call, then the real call), mutating the shared
    `preprocess_config` object in place on the first call and corrupting the
    second -- covered here too since this test goes through `from_model_paths`,
    not `TopDownPredictor.from_trained_models` directly.
    """
    centroid_dir = _copy_ckpt_with_scale(
        minimal_instance_centroid_ckpt, tmp_path / "centroid", scale=0.5
    )
    centered_dir = _copy_ckpt_with_scale(
        minimal_instance_centered_instance_ckpt,
        tmp_path / "centered_instance",
        scale=1.0,
    )
    predictor = Predictor.from_model_paths(
        [str(centroid_dir), str(centered_dir)],
        preprocess_config=_scale_only_preprocess_config(None),
    )
    assert predictor.inference_model.centroid_crop.input_scale == 0.5
    assert predictor.inference_model.instance_peaks.input_scale == 1.0


def test_multiclass_topdown_predictor_default_scale_is_per_stage_not_shared(
    minimal_instance_centroid_ckpt,
    minimal_instance_multi_class_topdown_ckpt,
    tmp_path,
):
    """Same regression as above, for `TopDownMultiClassPredictor`."""
    centroid_dir = _copy_ckpt_with_scale(
        minimal_instance_centroid_ckpt, tmp_path / "centroid", scale=0.5
    )
    confmap_dir = _copy_ckpt_with_scale(
        minimal_instance_multi_class_topdown_ckpt,
        tmp_path / "multiclass_centered_instance",
        scale=1.0,
    )
    predictor = Predictor.from_model_paths(
        [str(centroid_dir), str(confmap_dir)],
        preprocess_config=_scale_only_preprocess_config(None),
    )
    assert predictor.inference_model.centroid_crop.input_scale == 0.5
    assert predictor.inference_model.instance_peaks.input_scale == 1.0


def test_topdown_predictor_mismatched_scale_coords_differ_from_shared_scale_bug(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    tmp_path,
):
    """End-to-end guard: pins that fixed per-stage scale changes predicted coords.

    The attribute-level tests above check that each stage keeps its own
    `input_scale`, but don't exercise the actual forward pass. This one does,
    on a real video with a real (deliberately mismatched-scale) checkpoint
    pair. Instance *count* alone is not a reliable target here: forcing the
    pre-fix bug (both stages sharing one scale) on these tiny fixture models
    produces the SAME instance count across a 0.15x-3x scale range -- the
    models are too small/robust for lost detections to show up in a count.
    Predicted *coordinates* do shift measurably (confirmed up to ~67px on a
    96x96 crop), so this test compares those instead.

    Deliberately NOT a golden/snapshot test pinning a literal recorded array:
    an earlier version of this test did that and was flaky across CI
    platforms (mac/ubuntu/windows CPU backends produce measurably different
    conv/pooling reduction order, enough to shift which pixel wins an
    argmax-based peak -- tens of px apart on some elements, not just
    sub-pixel float noise). Instead, this reconstructs the pre-fix bug
    in-process (forcing the confmap stage to share the centroid stage's
    scale, exactly what the bug did) and asserts the two coordinate sets
    differ substantially -- a comparison that's robust to whatever a given
    CI platform's own numerics look like, since both sides run on the same
    platform in the same process.
    """
    centroid_dir = _copy_ckpt_with_scale(
        minimal_instance_centroid_ckpt, tmp_path / "centroid", scale=1.0
    )
    centered_dir = _copy_ckpt_with_scale(
        minimal_instance_centered_instance_ckpt,
        tmp_path / "centered_instance",
        scale=0.5,
    )

    def _predict_coords(force_shared_scale: bool) -> np.ndarray:
        predictor = Predictor.from_model_paths(
            [str(centroid_dir), str(centered_dir)],
            peak_threshold=0.03,
            max_instances=6,
            preprocess_config=_scale_only_preprocess_config(None),
        )
        if force_shared_scale:
            # Reconstruct the pre-fix bug: both stages share one scale.
            predictor.inference_model.instance_peaks.input_scale = (
                predictor.inference_model.centroid_crop.input_scale
            )
        predictor.make_pipeline(
            centered_instance_video.as_posix(), frames=list(range(5))
        )
        output = predictor.predict(make_labels=True)
        return np.concatenate([inst.numpy() for lf in output for inst in lf.instances])

    fixed_coords = _predict_coords(force_shared_scale=False)
    buggy_coords = _predict_coords(force_shared_scale=True)

    assert fixed_coords.shape == buggy_coords.shape
    assert np.nanmax(np.abs(fixed_coords - buggy_coords)) > 5.0


def test_single_instance_predictor_input_scale_override_end_to_end_coords_in_bounds(
    small_robot_minimal_video,
    minimal_instance_single_instance_ckpt,
):
    """End-to-end: an ``--input_scale`` override must not corrupt output coordinates.

    Before the fix, overriding ``scale`` away from the training config's value
    applied the override to the forward image resize but NOT to the
    coordinate rescale-back step (still using the stale training-config
    scale), producing physically-impossible out-of-frame coordinates (e.g.
    x > frame_width). Predicted keypoints must always land within the source
    frame's bounds, regardless of ``--input_scale``.
    """
    import sleap_io as sio

    video = sio.load_video(small_robot_minimal_video.as_posix())
    height, width = video.shape[1], video.shape[2]

    predictor = Predictor.from_model_paths(
        [minimal_instance_single_instance_ckpt],
        peak_threshold=0.3,
        preprocess_config=_scale_only_preprocess_config(1.5),
    )
    predictor.make_pipeline(
        small_robot_minimal_video.as_posix(),
        frames=[x for x in range(20)],
    )
    output = predictor.predict(make_labels=True)
    assert isinstance(output, sio.Labels)

    saw_instance = False
    for lf in output:
        for inst in lf.instances:
            pts = inst.numpy()
            valid = ~np.isnan(pts).any(axis=1)
            if not valid.any():
                continue
            saw_instance = True
            xs, ys = pts[valid, 0], pts[valid, 1]
            assert np.all(
                (xs >= 0) & (xs <= width)
            ), f"x coords out of [0, {width}] bounds: {xs[(xs < 0) | (xs > width)]}"
            assert np.all(
                (ys >= 0) & (ys <= height)
            ), f"y coords out of [0, {height}] bounds: {ys[(ys < 0) | (ys > height)]}"
    assert (
        saw_instance
    ), "test produced no instances to check -- fixture/threshold issue"


def test_filter_user_labeled_frames(tmp_path):
    """Test _filter_user_labeled_frames helper function."""
    import imageio.v3 as iio
    import os

    # Create a minimal video from temporary image files
    for i in range(10):
        img = np.zeros((100, 100, 1), dtype=np.uint8)
        iio.imwrite(os.path.join(tmp_path, f"frame_{i:03d}.png"), img[:, :, 0])

    video = sio.Video.from_filename(str(tmp_path))
    skeleton = sio.Skeleton(nodes=["A", "B"])

    # Create labeled frames: frames 0, 2, 5 have user instances
    lf0 = sio.LabeledFrame(
        video=video,
        frame_idx=0,
        instances=[
            sio.Instance(skeleton=skeleton, points={"A": [10, 10], "B": [20, 20]})
        ],
    )
    lf2 = sio.LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[
            sio.Instance(skeleton=skeleton, points={"A": [15, 15], "B": [25, 25]})
        ],
    )
    lf5 = sio.LabeledFrame(
        video=video,
        frame_idx=5,
        instances=[
            sio.Instance(skeleton=skeleton, points={"A": [30, 30], "B": [40, 40]})
        ],
    )

    labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[lf0, lf2, lf5]
    )

    # Test 1: exclude_user_labeled=False should return original frames
    frames = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    result = _filter_user_labeled_frames(
        labels, video, frames, exclude_user_labeled=False
    )
    assert result == frames

    # Test 2: exclude_user_labeled=True should filter out user-labeled frames
    result = _filter_user_labeled_frames(
        labels, video, frames, exclude_user_labeled=True
    )
    assert result == [1, 3, 4, 6, 7, 8, 9]

    # Test 3: frames=None should build full list and filter
    result = _filter_user_labeled_frames(labels, video, None, exclude_user_labeled=True)
    assert result == [1, 3, 4, 6, 7, 8, 9]

    # Test 4: frames=None with exclude_user_labeled=False should return None
    result = _filter_user_labeled_frames(
        labels, video, None, exclude_user_labeled=False
    )
    assert result is None

    # Test 5: Empty labels (no user-labeled frames) should return original frames
    empty_labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=[])
    result = _filter_user_labeled_frames(
        empty_labels, video, frames, exclude_user_labeled=True
    )
    assert result == frames

    # Test 6: All frames are user-labeled should return empty list
    all_user_labeled = [0, 2, 5]
    result = _filter_user_labeled_frames(
        labels, video, all_user_labeled, exclude_user_labeled=True
    )
    assert result == []


def test_filter_user_labeled_frames_with_predicted_instances(tmp_path):
    """Test that _filter_user_labeled_frames only filters frames with user instances, not predicted."""
    import imageio.v3 as iio
    import os

    # Create a minimal video from temporary image files
    for i in range(5):
        img = np.zeros((100, 100, 1), dtype=np.uint8)
        iio.imwrite(os.path.join(tmp_path, f"frame_{i:03d}.png"), img[:, :, 0])

    video = sio.Video.from_filename(str(tmp_path))
    skeleton = sio.Skeleton(nodes=["A", "B"])

    # Create a frame with only predicted instances (should NOT be filtered)
    lf_pred = sio.LabeledFrame(
        video=video,
        frame_idx=1,
        instances=[
            sio.PredictedInstance(
                skeleton=skeleton,
                points={"A": [10, 10], "B": [20, 20]},
                score=0.9,
            )
        ],
    )

    # Create a frame with user instances (should be filtered)
    lf_user = sio.LabeledFrame(
        video=video,
        frame_idx=2,
        instances=[
            sio.Instance(skeleton=skeleton, points={"A": [15, 15], "B": [25, 25]})
        ],
    )

    labels = sio.Labels(
        videos=[video], skeletons=[skeleton], labeled_frames=[lf_pred, lf_user]
    )

    frames = [0, 1, 2, 3, 4]
    result = _filter_user_labeled_frames(
        labels, video, frames, exclude_user_labeled=True
    )

    # Frame 1 has only predicted instances so should NOT be filtered
    # Frame 2 has user instances so should be filtered
    assert result == [0, 1, 3, 4]


def test_predictor_make_pipeline_with_labels_and_video_index(
    small_robot_minimal,
    minimal_instance_single_instance_ckpt,
):
    """Test make_pipeline with sio.Labels object and video_index parameter.

    This tests the code path where make_pipeline receives a Labels object
    with video_index specified, which triggers the filtering logic.
    """
    # Load the SLP file as a Labels object
    labels = sio.load_slp(small_robot_minimal.as_posix())

    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    # Create a SingleInstancePredictor
    predictor = Predictor.from_model_paths(
        [minimal_instance_single_instance_ckpt],
        peak_threshold=0.3,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    # Call make_pipeline with Labels object and video_index
    predictor.make_pipeline(
        labels,
        video_index=0,
        frames=[0, 1, 2],
    )

    # Verify the pipeline was created correctly
    assert predictor.pipeline is not None
    assert len(predictor.videos) == 1
    assert predictor.videos[0] == labels.videos[0]


def test_topdown_predictor_with_labels_and_video_index(
    minimal_instance,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
):
    """Test TopDownPredictor make_pipeline with sio.Labels and video_index."""
    # Load SLP file as Labels object
    labels = sio.load_slp(minimal_instance.as_posix())

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

    # Call make_pipeline with Labels object and video_index
    predictor.make_pipeline(
        labels,
        video_index=0,
        frames=[0],
    )

    assert predictor.pipeline is not None
    assert len(predictor.videos) == 1


def test_bottomup_predictor_with_labels_and_video_index(
    minimal_instance,
    minimal_instance_bottomup_ckpt,
):
    """Test BottomUpPredictor make_pipeline with sio.Labels and video_index."""
    labels = sio.load_slp(minimal_instance.as_posix())

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
        labels,
        video_index=0,
        frames=[0],
    )

    assert predictor.pipeline is not None
    assert len(predictor.videos) == 1


def test_centroid_predictor_with_labels_and_video_index(
    minimal_instance,
    minimal_instance_centroid_ckpt,
):
    """Test CentroidPredictor make_pipeline with sio.Labels and video_index."""
    labels = sio.load_slp(minimal_instance.as_posix())

    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        peak_threshold=0.03,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    predictor.make_pipeline(
        labels,
        video_index=0,
        frames=[0],
    )

    assert predictor.pipeline is not None
    assert len(predictor.videos) == 1


def test_multiclass_topdown_predictor_with_labels_and_video_index(
    minimal_instance,
    minimal_instance_multi_class_topdown_ckpt,
    minimal_instance_centroid_ckpt,
):
    """Test TopDownMultiClassPredictor make_pipeline with sio.Labels and video_index."""
    labels = sio.load_slp(minimal_instance.as_posix())

    preprocess_config = {
        "ensure_rgb": None,
        "ensure_grayscale": None,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
        "scale": None,
    }

    # TopDownMultiClassPredictor requires both centroid and instance models
    # when using video_index (no ground truth available)
    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt, minimal_instance_multi_class_topdown_ckpt],
        peak_threshold=0.03,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    predictor.make_pipeline(
        labels,
        video_index=0,
        frames=[0],
    )

    assert predictor.pipeline is not None
    assert len(predictor.videos) == 1
