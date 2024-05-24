import sleap_io as sio
from typing import Text
import pytest
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import Predictor


def test_topdown_predictor(
    minimal_instance, minimal_instance_ckpt, minimal_instance_centroid_ckpt
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    # for centered instance model
    # check if labels are created from ckpt

    predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])
    pred_labels = predictor.predict(make_labels=True)
    assert predictor.centroid_config is None
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
    preds = predictor.predict(make_labels=False)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()

    # if model parameter is not set right
    with pytest.raises(ValueError):
        config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
        model_name = config.model_config.head_configs[0].head_type
        config.model_config.head_configs[0].head_type = "instance"
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")
        predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])

    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    config.model_config.head_configs[0].head_type = model_name
    OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

    # centroid + centroid instance model
    predictor = Predictor.from_model_paths(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt]
    )
    pred_labels = predictor.predict(make_labels=True)
    assert predictor.centroid_config is not None
    assert predictor.confmap_config is not None
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= config.inference_config.data.max_instances

    # centroid model
    predictor = Predictor.from_model_paths(model_paths=[minimal_instance_centroid_ckpt])
    pred_labels = predictor.predict(make_labels=False)
    assert predictor.confmap_config is None
    assert len(pred_labels) == 1
    assert pred_labels[0]["centroids"].shape == (1, 1, 2, 2)

    # Provider = VideoReader
    # centroid + centered-instance model inference
    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    centroid_config = OmegaConf.load(
        f"{minimal_instance_centroid_ckpt}/training_config.yaml"
    )
    _centroid_config = centroid_config.copy()
    _config = config.copy()
    try:
        OmegaConf.update(config, "inference_config.data.provider", "VideoReader")
        OmegaConf.update(
            centroid_config, "inference_config.data.provider", "VideoReader"
        )
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.update(
            centroid_config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")
        OmegaConf.save(
            centroid_config, f"{minimal_instance_centroid_ckpt}/training_config.yaml"
        )
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt]
        )
        print("Predictor created!!")
        pred_labels = predictor.predict(make_labels=True)
        assert predictor.centroid_config is not None
        assert predictor.confmap_config is not None
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 100
    finally:
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")
        OmegaConf.save(
            _centroid_config, f"{minimal_instance_centroid_ckpt}/training_config.yaml"
        )

    # Unrecognized provider
    config = OmegaConf.load(f"{minimal_instance_centroid_ckpt}/training_config.yaml")
    _config = config.copy()
    try:
        OmegaConf.update(config, "inference_config.data.provider", "Reader")
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.save(config, f"{minimal_instance_centroid_ckpt}/training_config.yaml")
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt]
        )
        with pytest.raises(
            Exception,
            match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
        ):
            pred_labels = predictor.predict(make_labels=True)

    finally:
        OmegaConf.save(
            _config, f"{minimal_instance_centroid_ckpt}/training_config.yaml"
        )


def test_single_instance_predictor(minimal_instance, minimal_instance_ckpt):
    """Test SingleInstancePredictor."""
    # provider as LabelsReader
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
        OmegaConf.update(config, "inference_config.data.max_height", 500)
        OmegaConf.update(config, "inference_config.data.max_width", 500)
        config.model_config.head_configs[0].head_type = "SingleInstanceConfmapsHead"
        del config.model_config.head_configs[0].head_config.anchor_part
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])
        pred_labels = predictor.predict(make_labels=True)
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
        preds = predictor.predict(make_labels=False)
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
        OmegaConf.update(config, "inference_config.data.provider", "VideoReader")
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
        config.model_config.head_configs[0].head_type = "SingleInstanceConfmapsHead"
        del config.model_config.head_configs[0].head_config.anchor_part
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])
        pred_labels = predictor.predict(make_labels=True)
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 100
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same skeleton as the GT labels
        gt_labels = sio.load_slp(minimal_instance)
        assert pred_labels.skeletons == gt_labels.skeletons
        assert lf.frame_idx == 0

        # check if dictionaries are created when make labels is set to False
        preds = predictor.predict(make_labels=False)
        assert isinstance(preds, list)
        assert len(preds) == 25
        assert preds[0]["pred_instance_peaks"].shape[0] == 4
        assert isinstance(preds[0], dict)
        assert "pred_confmaps" not in preds[0].keys()
        print("end of video reader")

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    # unrecognized provider
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        OmegaConf.update(config, "inference_config.data.provider", "Reader")
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
        config.model_config.head_configs[0].head_type = "SingleInstanceConfmapsHead"
        del config.model_config.head_configs[0].head_config.anchor_part
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])
        with pytest.raises(
            Exception,
            match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
        ):
            pred_labels = predictor.predict(make_labels=True)

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")


def test_bottomup_predictor(minimal_instance, minimal_instance_bottomup_ckpt):
    """Test BottomUpPredictor."""
    # provider as LabelsReader

    # check if labels are created from ckpt
    predictor = Predictor.from_model_paths(model_paths=[minimal_instance_bottomup_ckpt])
    pred_labels = predictor.predict(make_labels=True)
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

    # check if dictionaries are created when make labels is set to False
    preds = predictor.predict(make_labels=False)
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert preds[0]["pred_instance_peaks"].is_nested
    assert tuple(preds[0]["pred_instance_peaks"][0].shape) == (2, 2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape) == (2, 2)
    assert tuple(preds[0]["instance_scores"][0].shape) == (2,)

    # with higher threshold
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
    )
    orig_config = train_config.copy()
    try:
        OmegaConf.update(train_config, "inference_config.peak_threshold", 1.0)
        OmegaConf.save(
            train_config, f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
        )

        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_bottomup_ckpt]
        )
        pred_labels = predictor.predict(make_labels=True)
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 1
        assert len(pred_labels[0].instances) == 0

    finally:
        OmegaConf.save(
            orig_config, f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
        )

    # change to video reader
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
    )
    orig_config = train_config.copy()
    try:
        OmegaConf.update(train_config, "inference_config.data.provider", "VideoReader")
        OmegaConf.update(
            train_config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.save(
            train_config, f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
        )

        # check if labels are created from ckpt
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_bottomup_ckpt]
        )
        pred_labels = predictor.predict(make_labels=True)
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_bottomup_ckpt]
        )
        pred_labels = predictor.predict(make_labels=True)
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 100
        assert len(pred_labels[0].instances) == 2

        # check if dictionaries are created when make labels is set to False
        preds = predictor.predict(make_labels=False)
        assert isinstance(preds, list)
        assert len(preds) == 25
        assert isinstance(preds[0], dict)
        assert "pred_confmaps" not in preds[0].keys()
        assert preds[0]["pred_instance_peaks"].is_nested
        assert tuple(preds[0]["pred_instance_peaks"][0].shape) == (2, 2, 2)
        assert tuple(preds[0]["pred_peak_values"][0].shape) == (2, 2)
        assert tuple(preds[0]["instance_scores"][0].shape) == (2,)

    finally:
        OmegaConf.save(
            orig_config, f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
        )

    # unrecognized provider
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
    )
    orig_config = train_config.copy()
    try:
        OmegaConf.update(train_config, "inference_config.data.provider", "Reader")
        OmegaConf.save(
            train_config, f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
        )

        # check if labels are created from ckpt
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_bottomup_ckpt]
        )
        with pytest.raises(
            Exception,
            match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
        ):
            pred_labels = predictor.predict(make_labels=True)

    finally:
        # save the original config back
        OmegaConf.save(
            orig_config, f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
        )
