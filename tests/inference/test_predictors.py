import sleap_io as sio
from typing import Text
import pytest
from omegaconf import OmegaConf
from sleap_nn.inference.predictors import main


def test_topdown_predictor(
    minimal_instance,
    minimal_instance_ckpt,
    minimal_instance_centroid_ckpt,
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
        peak_threshold=0.1,
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
        batch_size=1,
    )
    assert isinstance(preds, list)
    assert len(preds) == 2
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
        peak_threshold=0.0,
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6

    # centroid model
    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt],
        data_path="./tests/assets/minimal_instance.pkg.slp",
        provider="LabelsReader",
        make_labels=False,
        max_instances=6,
        peak_threshold=0.1,
    )
    assert len(pred_labels) == 1
    assert pred_labels[0]["centroids"].shape == (1, 1, 2, 2)

    # Provider = VideoReader
    # centroid + centered-instance model inference

    pred_labels = main(
        model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt],
        data_path="./tests/assets/centered_pair_small.mp4",
        provider="VideoReader",
        make_labels=True,
        max_instances=6,
        peak_threshold=0.0,
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


def test_single_instance_predictor(minimal_instance, minimal_instance_ckpt):
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
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        pred_labels = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/minimal_instance.pkg.slp",
            provider="LabelsReader",
            make_labels=True,
            max_instances=6,
            peak_threshold=0.3,
            max_height=500,
            max_width=500,
            scale=0.9,
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
            max_height=500,
            max_width=500,
            scale=0.9,
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
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        pred_labels = main(
            model_paths=[minimal_instance_ckpt],
            data_path="./tests/assets/centered_pair_small.mp4",
            provider="VideoReader",
            make_labels=True,
            peak_threshold=0.3,
            scale=0.9,
        )
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 100
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
            scale=0.9,
        )
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
                scale=0.9,
            )

    finally:
        # save the original config back
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")


def test_bottomup_predictor(minimal_instance, minimal_instance_bottomup_ckpt):
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
    assert len(pred_labels[0].instances) == 6
    print(pred_labels[0].instances)
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
    assert len(pred_labels[0].instances) == 6

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
