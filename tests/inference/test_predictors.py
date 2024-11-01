import sleap_io as sio
from pathlib import Path
from typing import Text
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import Predictor, main


def test_topdown_predictor(
    minimal_instance,
    minimal_instance_ckpt,
    minimal_instance_centroid_ckpt,
    minimal_instance_bottomup_ckpt,
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    # for centered instance model
    # check if labels are created from ckpt

    pred_labels = main(
        model_paths=[minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        return_confmaps=False,
        make_labels=True,
        peak_threshold=0.0,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 2
    lf = pred_labels[0]

    # check if the predicted labels have same video and skeleton as the ground truth labels
    gt_labels = sio.load_slp(minimal_instance)
    gt_lf = gt_labels[0]
    assert pred_labels.skeletons == gt_labels.skeletons
    assert lf.frame_idx == gt_lf.frame_idx
    assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape
    assert lf.instances[1].numpy().shape == gt_lf.instances[1].numpy().shape
    assert lf.image.shape == gt_lf.image.shape

    # check if dictionaries are created when make labels is set to False
    preds = main(
        model_paths=[minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
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
        preds = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            provider="LabelsReader",
            make_labels=False,
        )

    OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

    # centroid + centroid instance model
    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6

    # centroid model
    max_instances = 6
    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=False,
        max_instances=max_instances,
        peak_threshold=0.1,
    )
    assert len(pred_labels) == 1
    assert (
        pred_labels[0]["centroids"].shape[-2] <= max_instances
    )  # centroids (1,1,max_instances,2)

    # Provider = VideoReader
    # centroid + centered-instance model inference

    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        provider="VideoReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        integral_refinement="integral",
        videoreader_start_idx=0,
        videoreader_end_idx=100,
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 100

    # Unrecognized provider
    with pytest.raises(
        Exception,
        match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
    ):
        pred_labels = main(
            model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            provider="Reader",
            make_labels=True,
            max_instances=6,
        )

    # Provider = VideoReader
    # error in Videoreader but graceful execution

    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        provider="VideoReader",
        make_labels=True,
        max_instances=6,
        videoreader_start_idx=1100,
        videoreader_end_idx=1103,
        peak_threshold=0.1,
    )

    # Provider = VideoReader
    # centroid model not provided

    with pytest.raises(
        ValueError,
        match="Ground truth data was not detected... Please load both models when predicting on non-ground-truth data.",
    ):
        pred_labels = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            provider="VideoReader",
            make_labels=True,
            max_instances=6,
            videoreader_start_idx=0,
            videoreader_end_idx=100,
            peak_threshold=0.1,
        )

    # test with tracking
    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        provider="VideoReader",
        make_labels=True,
        max_instances=2,
        peak_threshold=0.1,
        videoreader_start_idx=0,
        videoreader_end_idx=20,
        tracking=True,
    )

    assert len(pred_labels.tracks) <= 2  # should be less than max tracks

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None
            assert instance.tracking_score == 1

    # check loading diff head ckpt for centered instance
    preprocess_config = {
        "is_rgb": False,
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_bottomup_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_bottomup_ckpt) / "best.ckpt")
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

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

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
        "is_rgb": False,
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

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

    ckpt = torch.load(Path(minimal_instance_centroid_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

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

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

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
        "is_rgb": False,
        "crop_hw": None,
        "max_width": None,
        "max_height": None,
    }

    predictor = Predictor.from_model_paths(
        [minimal_instance_centroid_ckpt],
        backbone_ckpt_path=Path(minimal_instance_ckpt) / "best.ckpt",
        head_ckpt_path=Path(minimal_instance_centroid_ckpt) / "best.ckpt",
        peak_threshold=0.03,
        max_instances=6,
        preprocess_config=OmegaConf.create(preprocess_config),
    )

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

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

    ckpt = torch.load(Path(minimal_instance_centroid_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

    model_weights = (
        next(predictor.inference_model.centroid_crop.torch_model.model.parameters())[
            0, 0, :
        ]
        .detach()
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
        pred_labels = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            provider="LabelsReader",
            make_labels=True,
            max_instances=6,
            peak_threshold=0.1,
        )
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 1
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same video and skeleton as the ground truth labels
        gt_labels = sio.load_slp(minimal_instance)
        gt_lf = gt_labels[0]
        assert pred_labels.skeletons == gt_labels.skeletons
        assert lf.frame_idx == gt_lf.frame_idx
        assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape

        # check if dictionaries are created when make labels is set to False
        preds = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            provider="LabelsReader",
            make_labels=False,
            peak_threshold=0.3,
        )
        assert isinstance(preds, list)
        assert len(preds) == 1
        assert isinstance(preds[0], dict)
        assert "pred_confmaps" not in preds[0].keys()

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
        pred_labels = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            provider="VideoReader",
            make_labels=True,
            peak_threshold=0.3,
        )
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 1100
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same skeleton as the GT labels
        gt_labels = sio.load_slp(minimal_instance)
        assert pred_labels.skeletons == gt_labels.skeletons
        assert lf.frame_idx == 0

        # check if dictionaries are created when make labels is set to False
        preds = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            provider="VideoReader",
            make_labels=False,
            peak_threshold=0.3,
            videoreader_end_idx=100,
        )
        assert isinstance(preds, list)
        assert len(preds) == 25
        assert preds[0]["pred_instance_peaks"].shape[0] == 4
        assert isinstance(preds[0], dict)
        assert "pred_confmaps" not in preds[0].keys()

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    # unrecognized provider
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        head_config = config.model_config.head_configs.centered_instance
        del config.model_config.head_configs.centered_instance
        OmegaConf.update(
            config, "model_config.head_configs.single_instance", head_config
        )
        del config.model_config.head_configs.single_instance.confmaps.anchor_part
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        with pytest.raises(
            Exception,
            match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
        ):
            preds = main(
                model_paths=[minimal_instance_ckpt],
                data_path="./tests/assets/centered_pair_small.mp4",
                provider="Reader",
                make_labels=False,
                peak_threshold=0.3,
            )

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
            "is_rgb": False,
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

        ckpt = torch.load(Path(minimal_instance_bottomup_ckpt) / "best.ckpt")
        head_layer_ckpt = ckpt["state_dict"]["model.head_layers.0.0.weight"][
            0, 0, :
        ].numpy()

        model_weights = (
            next(predictor.inference_model.torch_model.model.head_layers.parameters())[
                0, 0, :
            ]
            .detach()
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
            "is_rgb": False,
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

        ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
        backbone_ckpt = ckpt["state_dict"][
            "model.backbone.enc.encoder_stack.0.blocks.0.weight"
        ][0, 0, :].numpy()

        model_weights = (
            next(predictor.inference_model.torch_model.model.parameters())[0, 0, :]
            .detach()
            .numpy()
        )

        assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")


def test_bottomup_predictor(
    minimal_instance, minimal_instance_bottomup_ckpt, minimal_instance_ckpt
):
    """Test BottomUpPredictor module."""
    # provider as LabelsReader

    # check if labels are created from ckpt
    pred_labels = main(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.03,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6
    lf = pred_labels[0]

    # check if the predicted labels have same video and skeleton as the ground truth labels
    gt_labels = sio.load_slp(minimal_instance)
    gt_lf = gt_labels[0]
    assert pred_labels.skeletons == gt_labels.skeletons
    assert lf.frame_idx == gt_lf.frame_idx
    assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape

    # check if dictionaries are created when make labels is set to False
    preds = main(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
    )
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # with higher threshold
    pred_labels = main(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=1.0,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 0

    # change to video reader
    pred_labels = main(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        provider="VideoReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.03,
        videoreader_start_idx=0,
        videoreader_end_idx=100,
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 100
    assert len(pred_labels[0].instances) <= 6

    # check if dictionaries are created when make labels is set to False
    preds = main(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        provider="VideoReader",
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
        videoreader_start_idx=0,
        videoreader_end_idx=100,
    )
    assert isinstance(preds, list)
    assert len(preds) == 25
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # unrecognized provider
    with pytest.raises(
        Exception,
        match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
    ):
        preds = main(
            model_paths=[minimal_instance_bottomup_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            provider="Reader",
            make_labels=True,
            max_instances=6,
            peak_threshold=0.03,
        )

    # test with tracking
    pred_labels = main(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.03,
        tracking=True,
    )

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None
            assert instance.tracking_score == 1

    assert len(pred_labels.tracks) <= 6  # should be less than max tracks

    # check loading diff head ckpt
    preprocess_config = {
        "is_rgb": False,
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

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
    head_layer_ckpt = ckpt["state_dict"]["model.head_layers.0.0.weight"][
        0, 0, :
    ].numpy()
    print(f"head_layer_ckpt: {head_layer_ckpt}")

    model_weights = (
        next(predictor.inference_model.torch_model.model.head_layers.parameters())[
            0, 0, :
        ]
        .detach()
        .numpy()
    )
    print(model_weights)

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

    ckpt = torch.load(Path(minimal_instance_ckpt) / "best.ckpt")
    backbone_ckpt = ckpt["state_dict"][
        "model.backbone.enc.encoder_stack.0.blocks.0.weight"
    ][0, 0, :].numpy()

    model_weights = (
        next(predictor.inference_model.torch_model.model.parameters())[0, 0, :]
        .detach()
        .numpy()
    )

    assert np.all(np.abs(backbone_ckpt - model_weights) < 1e-6)
