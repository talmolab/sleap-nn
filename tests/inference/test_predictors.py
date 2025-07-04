import sleap_io as sio
from pathlib import Path
from typing import Text
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import Predictor, run_inference
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
    minimal_instance,
    minimal_instance_ckpt,
    minimal_instance_centroid_ckpt,
    minimal_instance_bottomup_ckpt,
    tmp_path,
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    # for centered instance model
    # check if labels are created from ckpt

    pred_labels = run_inference(
        model_paths=[minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        return_confmaps=False,
        make_labels=True,
        peak_threshold=0.0,
        device="cpu",
        output_path=f"{tmp_path}/test.pkg.slp",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 2
    lf = pred_labels[0]

    assert Path(f"{tmp_path}/test.pkg.slp").exists

    # check if the predicted labels have same video and skeleton as the ground truth labels
    gt_labels = sio.load_slp(minimal_instance)
    gt_lf = gt_labels[0]

    skl = pred_labels.skeletons[0]
    gt_skl = gt_labels.skeletons[0]
    assert [a.name for a in skl.nodes] == [a.name for a in gt_skl.nodes]
    assert len(skl.edges) == len(gt_skl.edges)
    for a, b in zip(skl.edges, gt_skl.edges):
        assert a[0].name == b[0].name and a[1].name == b[1].name
    assert skl.symmetries == gt_skl.symmetries

    assert lf.frame_idx == gt_lf.frame_idx
    assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape
    assert lf.instances[1].numpy().shape == gt_lf.instances[1].numpy().shape
    assert lf.image.shape == gt_lf.image.shape

    # with video_index
    preds = run_inference(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        video_index=0,
        frames=[0],
        make_labels=True,
        device="cpu",
        peak_threshold=0.0,
    )
    assert isinstance(preds, sio.Labels)
    assert len(preds) == 1

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        device="cpu",
        make_labels=False,
        peak_threshold=0.0,
        integral_refinement="integral",
    )
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert len(preds[0]["instance_image"]) == 2
    assert len(preds[0]["centroid"]) == 2
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()

    # if model parameter is not set right
    with pytest.raises(ValueError):
        config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
        config_copy = config.copy()
        head_config = config_copy.model_config.head_configs.centered_instance
        del config_copy.model_config.head_configs.centered_instance
        OmegaConf.update(config_copy, "model_config.head_configs.topdown", head_config)
        OmegaConf.save(config_copy, f"{minimal_instance_ckpt}/training_config.yaml")
        preds = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            make_labels=False,
        )
    assert "Could not create predictor" in caplog.text

    OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

    # centroid + centroid instance model
    pred_labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        make_labels=True,
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        device="cpu",
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6

    # centroid model
    max_instances = 6
    pred_labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        make_labels=False,
        max_instances=max_instances,
        device="cpu",
        peak_threshold=0.1,
    )
    assert len(pred_labels) == 1
    assert (
        pred_labels[0]["centroid"].shape[-2] <= max_instances
    )  # centroids (1,max_instances,2)

    # Provider = VideoReader
    # centroid + centered-instance model inference

    pred_labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        make_labels=True,
        max_instances=6,
        device="cpu",
        peak_threshold=[0.0, 0.0],
        integral_refinement="integral",
        frames=[x for x in range(100)],
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 100

    # Provider = VideoReader
    # error in Videoreader but graceful execution

    pred_labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        make_labels=True,
        max_instances=6,
        device="cpu",
        frames=[1100, 1101, 1102, 1103],
        peak_threshold=0.1,
    )

    # Provider = VideoReader
    # centroid model not provided

    with pytest.raises(
        ValueError,
    ):
        pred_labels = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            make_labels=True,
            max_instances=6,
            device="cpu",
            frames=[x for x in range(100)],
            peak_threshold=0.1,
        )
    assert "Error when reading video frame." in caplog.text

    # test with tracking
    pred_labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        make_labels=True,
        max_instances=2,
        post_connect_single_breaks=True,
        max_tracks=None,
        device="cpu",
        peak_threshold=0.1,
        frames=[x for x in range(20)],
        tracking=True,
    )

    assert len(pred_labels.tracks) <= 2  # should be less than max tracks

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None

    # test with tracking (no max inst provided and post connect breaks is true)
    with pytest.raises(ValueError):
        pred_labels = run_inference(
            model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            make_labels=True,
            max_instances=None,
            post_connect_single_breaks=True,
            max_tracks=None,
            device="cpu",
            peak_threshold=0.1,
            frames=[x for x in range(20)],
            tracking=True,
        )
        assert "Max_tracks (and max instances) is None" in caplog.text

    # check loading diff head ckpt for centered instance
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": False,
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
        "anchor_part": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(
        Path(minimal_instance_bottomup_ckpt) / "best.ckpt", map_location="cpu"
    )
    head_layer_ckpt = ckpt["state_dict"]["model.head_layers.0.0.weight"][
        0, 0, :
    ].numpy()

    model_weights = (
        next(
            predictor.inference_model.instance_peaks.torch_model.model.head_layers.parameters()
        )[0, 0, :]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(head_layer_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None - centered instance
    predictor = Predictor.from_model_paths(
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
        "anchor_part": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        Path(minimal_instance_centroid_ckpt) / "best.ckpt", map_location="cpu"
    )
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
        "anchor_part": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        Path(minimal_instance_centroid_ckpt) / "best.ckpt", map_location="cpu"
    )
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
        "anchor_part": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        Path(minimal_instance_centroid_ckpt) / "best.ckpt", map_location="cpu"
    )
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
        "anchor_part": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        Path(minimal_instance_centroid_ckpt) / "best.ckpt", map_location="cpu"
    )
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
        "anchor_part": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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
        Path(minimal_instance_centroid_ckpt) / "best.ckpt", map_location="cpu"
    )
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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


def test_single_instance_predictor(
    minimal_instance,
    minimal_instance_ckpt,
    minimal_instance_bottomup_ckpt,
):
    """Test SingleInstancePredictor module."""
    # provider as LabelsReader
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        head_config = config.model_config.head_configs.centered_instance
        del config.model_config.head_configs.centered_instance
        OmegaConf.update(
            config, "model_config.head_configs.single_instance", head_config
        )
        del config.model_config.head_configs.single_instance.confmaps.anchor_part
        OmegaConf.update(config, "data_config.preprocessing.scale", 0.9)

        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        pred_labels = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            make_labels=True,
            max_instances=6,
            device="cpu",
            peak_threshold=0.1,
        )
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 1
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same video and skeleton as the ground truth labels
        gt_labels = sio.load_slp(minimal_instance)
        gt_lf = gt_labels[0]
        skl = pred_labels.skeletons[0]
        gt_skl = gt_labels.skeletons[0]
        assert [a.name for a in skl.nodes] == [a.name for a in gt_skl.nodes]
        assert len(skl.edges) == len(gt_skl.edges)
        for a, b in zip(skl.edges, gt_skl.edges):
            assert a[0].name == b[0].name and a[1].name == b[1].name
        assert skl.symmetries == gt_skl.symmetries
        assert lf.frame_idx == gt_lf.frame_idx
        assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape

        # check if dictionaries are created when make labels is set to False
        preds = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            make_labels=False,
            peak_threshold=0.3,
            device="cpu",
        )
        assert isinstance(preds, list)
        assert len(preds) == 1
        assert isinstance(preds[0], dict)
        assert "pred_confmaps" not in preds[0].keys()

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    # slp file with video index
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        head_config = config.model_config.head_configs.centered_instance
        del config.model_config.head_configs.centered_instance
        OmegaConf.update(
            config, "model_config.head_configs.single_instance", head_config
        )
        del config.model_config.head_configs.single_instance.confmaps.anchor_part
        OmegaConf.update(config, "data_config.preprocessing.scale", 0.9)

        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        pred_labels = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            video_index=0,
            frames=[0],
            device="cpu",
            make_labels=True,
            peak_threshold=0.1,
        )
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 1
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same video and skeleton as the ground truth labels
        gt_labels = sio.load_slp(minimal_instance)
        gt_lf = gt_labels[0]
        skl = pred_labels.skeletons[0]
        gt_skl = gt_labels.skeletons[0]
        assert [a.name for a in skl.nodes] == [a.name for a in gt_skl.nodes]
        assert len(skl.edges) == len(gt_skl.edges)
        for a, b in zip(skl.edges, gt_skl.edges):
            assert a[0].name == b[0].name and a[1].name == b[1].name
        assert skl.symmetries == gt_skl.symmetries
        assert lf.frame_idx == gt_lf.frame_idx
        assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    # provider as VideoReader
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        head_config = config.model_config.head_configs.centered_instance
        del config.model_config.head_configs.centered_instance
        OmegaConf.update(
            config, "model_config.head_configs.single_instance", head_config
        )
        del config.model_config.head_configs.single_instance.confmaps.anchor_part
        OmegaConf.update(config, "data_config.preprocessing.scale", 0.9)

        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        pred_labels = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            make_labels=True,
            device="cpu",
            peak_threshold=0.0,
        )
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 1100
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same skeleton as the GT labels
        gt_labels = sio.load_slp(minimal_instance)
        skl = pred_labels.skeletons[0]
        gt_skl = gt_labels.skeletons[0]
        assert [a.name for a in skl.nodes] == [a.name for a in gt_skl.nodes]
        assert len(skl.edges) == len(gt_skl.edges)
        for a, b in zip(skl.edges, gt_skl.edges):
            assert a[0].name == b[0].name and a[1].name == b[1].name
        assert skl.symmetries == gt_skl.symmetries
        assert lf.frame_idx == 0

        # check if dictionaries are created when make labels is set to False
        preds = run_inference(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            make_labels=False,
            device="cpu",
            peak_threshold=0.3,
            frames=[x for x in range(100)],
        )
        assert isinstance(preds, list)
        assert len(preds) == 25
        assert preds[0]["pred_instance_peaks"].shape[0] == 4
        assert isinstance(preds[0], dict)
        assert "pred_confmaps" not in preds[0].keys()

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        head_config = config.model_config.head_configs.centered_instance
        del config.model_config.head_configs.centered_instance
        OmegaConf.update(
            config, "model_config.head_configs.single_instance", head_config
        )
        del config.model_config.head_configs.single_instance.confmaps.anchor_part
        OmegaConf.update(config, "data_config.preprocessing.scale", 0.9)

        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check loading diff head ckpt
        preprocess_config = {
            "ensure_rgb": False,
            "ensure_grayscale": True,
            "crop_hw": None,
            "max_width": None,
            "max_height": None,
        }

        predictor = Predictor.from_model_paths(
            [minimal_instance_ckpt],
            backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
            head_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
            peak_threshold=0.1,
            preprocess_config=OmegaConf.create(preprocess_config),
        )

        ckpt = torch.load(
            Path(minimal_instance_bottomup_ckpt) / "best.ckpt", map_location="cpu"
        )
        head_layer_ckpt = (
            ckpt["state_dict"]["model.head_layers.0.0.weight"][0, 0, :].cpu().numpy()
        )

        model_weights = (
            next(predictor.inference_model.torch_model.model.head_layers.parameters())[
                0, 0, :
            ]
            .detach()
            .cpu()
            .numpy()
        )

        assert np.all(np.abs(head_layer_ckpt - model_weights) < 1e-6)

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        head_config = config.model_config.head_configs.centered_instance
        del config.model_config.head_configs.centered_instance
        OmegaConf.update(
            config, "model_config.head_configs.single_instance", head_config
        )
        del config.model_config.head_configs.single_instance.confmaps.anchor_part
        OmegaConf.update(config, "data_config.preprocessing.scale", 0.9)

        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check loading diff head ckpt
        preprocess_config = {
            "ensure_rgb": False,
            "ensure_grayscale": False,
            "crop_hw": None,
            "max_width": None,
            "max_height": None,
        }

        predictor = Predictor.from_model_paths(
            [minimal_instance_bottomup_ckpt],
            backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
            head_ckpt_path=None,
            peak_threshold=0.03,
            max_instances=6,
            preprocess_config=OmegaConf.create(preprocess_config),
        )

        ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
        backbone_ckpt = (
            ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
                0, 0, :
            ]
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

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")


def test_bottomup_predictor(
    caplog, minimal_instance, minimal_instance_bottomup_ckpt, minimal_instance_ckpt
):
    """Test BottomUpPredictor module."""
    # provider as LabelsReader

    # check if labels are created from ckpt
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.03,
        device="cpu",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6
    lf = pred_labels[0]

    # check if the predicted labels have same video and skeleton as the ground truth labels
    gt_labels = sio.load_slp(minimal_instance)
    gt_lf = gt_labels[0]
    skl = pred_labels.skeletons[0]
    gt_skl = gt_labels.skeletons[0]
    assert [a.name for a in skl.nodes] == [a.name for a in gt_skl.nodes]
    assert len(skl.edges) == len(gt_skl.edges)
    for a, b in zip(skl.edges, gt_skl.edges):
        assert a[0].name == b[0].name and a[1].name == b[1].name
    assert skl.symmetries == gt_skl.symmetries
    assert lf.frame_idx == gt_lf.frame_idx
    assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
        device="cpu",
    )
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # with video_index
    preds = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        video_index=0,
        frames=[0],
        make_labels=True,
        device="cpu",
    )
    assert isinstance(preds, sio.Labels)
    assert len(preds) == 1

    # with higher threshold
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        make_labels=True,
        max_instances=6,
        peak_threshold=1.0,
        device="cpu",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 0

    # change to video reader
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.03,
        frames=[x for x in range(100)],
        device="cpu",
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 100
    assert len(pred_labels[0].instances) <= 6

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
        frames=[x for x in range(100)],
        device="cpu",
    )
    assert isinstance(preds, list)
    assert len(preds) == 25
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # test with tracking
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.03,
        tracking=True,
        candidates_method="local_queues",
        max_tracks=6,
        post_connect_single_breaks=True,
        device="cpu",
    )

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None
            assert instance.tracking_score == 1

    assert len(pred_labels.tracks) <= 6  # should be less than max tracks

    # check loading diff head ckpt
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_bottomup_ckpt],
        backbone_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    head_layer_ckpt = (
        ckpt["state_dict"]["model.head_layers.0.0.weight"][0, 0, :].cpu().numpy()
    )

    model_weights = (
        next(predictor.inference_model.torch_model.model.head_layers.parameters())[
            0, 0, :
        ]
        .detach()
        .cpu()
        .numpy()
    )

    assert np.all(np.abs(head_layer_ckpt - model_weights) < 1e-6)

    # load only backbone and head ckpt as None
    predictor = Predictor.from_model_paths(
        [minimal_instance_bottomup_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=None,
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt", map_location="cpu")
    backbone_ckpt = (
        ckpt["state_dict"]["model.backbone.enc.encoder_stack.0.blocks.0.weight"][
            0, 0, :
        ]
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


def test_tracking_only_pipeline(
    minimal_instance_centroid_ckpt, minimal_instance_ckpt, centered_instance_video
):
    """Test tracking-only pipeline."""
    labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        post_connect_single_breaks=True,
    )
    labels.save("preds.slp")

    tracked_labels = run_inference(
        data_path="preds.slp",
        tracking=True,
        post_connect_single_breaks=True,
        max_instances=2,
    )

    assert len(tracked_labels.tracks) == 2

    # neither model nor tracking is provided
    with pytest.raises(Exception):
        labels = run_inference(
            data_path=centered_instance_video.as_posix(), tracking=False
        )
