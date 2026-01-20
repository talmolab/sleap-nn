import sleap_io as sio
from pathlib import Path
from typing import Text
import numpy as np
import pytest
from omegaconf import OmegaConf
from sleap_nn.predict import run_inference
from loguru import logger
from _pytest.logging import LogCaptureFixture
import torch


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
    tmp_path,
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    # for centered instance model
    # check if labels are created from ckpt

    pred_labels = run_inference(
        model_paths=[minimal_instance_centered_instance_ckpt],
        data_path=minimal_instance.as_posix(),
        return_confmaps=False,
        make_labels=True,
        peak_threshold=0.0,
        device="cpu",
        output_path=f"{tmp_path}/test.pkg.slp",
        integral_refinement=None,
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
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=minimal_instance.as_posix(),
        video_index=0,
        frames=[0],
        make_labels=True,
        output_path=tmp_path / "test.slp",
        device="cpu",
        max_instances=6,
        peak_threshold=0.0,
        integral_refinement=None,
    )
    assert isinstance(preds, sio.Labels)
    assert len(preds) == 1

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_centered_instance_ckpt],
        data_path=minimal_instance.as_posix(),
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
        config = OmegaConf.load(
            f"{minimal_instance_centered_instance_ckpt}/training_config.yaml"
        )
        config_copy = config.copy()
        head_config = config_copy.model_config.head_configs.centered_instance
        del config_copy.model_config.head_configs.centered_instance
        OmegaConf.update(config_copy, "model_config.head_configs.topdown", head_config)
        OmegaConf.save(
            config_copy,
            f"{minimal_instance_centered_instance_ckpt}/training_config.yaml",
        )
        preds = run_inference(
            model_paths=[minimal_instance_centered_instance_ckpt],
            data_path=minimal_instance.as_posix(),
            make_labels=False,
            integral_refinement=None,
        )
    assert "Could not create predictor" in caplog.text

    OmegaConf.save(
        config, f"{minimal_instance_centered_instance_ckpt}/training_config.yaml"
    )

    # centroid + centroid instance model
    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        device="cpu",
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6

    # centroid + centroid instance model with labels
    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        input_labels=sio.load_slp(minimal_instance.as_posix()),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        device="cpu",
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6

    # centroid + centroid instance model with labels (return confmaps)
    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        input_labels=sio.load_slp(minimal_instance.as_posix()),
        return_confmaps=True,
        make_labels=False,
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        device="cpu",
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, list)
    assert len(pred_labels) == 1
    assert "pred_centroid_confmaps" in pred_labels[0].keys()
    assert "pred_instance_peaks" in pred_labels[0].keys()

    # centroid model
    max_instances = 6
    pred_labels = run_inference(
        model_paths=[minimal_instance_centroid_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=False,
        max_instances=max_instances,
        device="cpu",
        peak_threshold=0.1,
        integral_refinement=None,
    )
    assert len(pred_labels) == 1
    assert (
        pred_labels[0]["centroid"].shape[-2] <= max_instances
    )  # centroids (1,max_instances,2)

    # Provider = VideoReader
    # centroid + centered-instance model inference

    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        device="cpu",
        peak_threshold=[0.0, 0.0],
        integral_refinement="integral",
        frames=[x for x in range(10)],
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 10

    # Provider = VideoReader
    # error in Videoreader but graceful execution

    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        device="cpu",
        frames=[1100, 1101, 1102, 1103],
        peak_threshold=0.1,
        integral_refinement=None,
    )

    # Provider = VideoReader
    # centroid model not provided

    with pytest.raises(
        ValueError,
    ):
        pred_labels = run_inference(
            model_paths=[minimal_instance_centered_instance_ckpt],
            data_path=centered_instance_video.as_posix(),
            make_labels=True,
            output_path=tmp_path / "test.slp",
            max_instances=6,
            device="cpu",
            frames=[x for x in range(10)],
            peak_threshold=0.1,
            integral_refinement=None,
        )
    assert "Error when reading video frame." in caplog.text

    # test with tracking
    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
        max_tracks=None,
        device="cpu",
        peak_threshold=0.1,
        frames=[x for x in range(20)],
        tracking=True,
        integral_refinement=None,
    )

    assert len(pred_labels.tracks) <= 2  # should be less than max tracks

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None

    # test with tracking and tracking_target_instance_count is not provided
    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        post_connect_single_breaks=True,
        max_tracks=None,
        device="cpu",
        peak_threshold=0.1,
        frames=[x for x in range(5)],
        tracking=True,
        integral_refinement=None,
    )

    assert len(pred_labels.tracks) <= 2  # should be less than max tracks

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None

    # test with tracking (no max inst provided and post connect breaks is true)
    with pytest.raises(ValueError):
        pred_labels = run_inference(
            model_paths=[
                minimal_instance_centroid_ckpt,
                minimal_instance_centered_instance_ckpt,
            ],
            data_path=centered_instance_video.as_posix(),
            make_labels=True,
            output_path=tmp_path / "test.slp",
            max_instances=None,
            post_connect_single_breaks=True,
            max_tracks=None,
            device="cpu",
            peak_threshold=0.1,
            frames=[x for x in range(20)],
            tracking=True,
            integral_refinement=None,
        )
        assert "tracking_target_instance_count and max_instances" in caplog.text


def test_multiclass_topdown_predictor(
    caplog,
    minimal_instance,
    minimal_instance_multi_class_topdown_ckpt,
    minimal_instance_centroid_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Test TopDownPredictor class for running inference on centroid and centered instance models."""
    # for centered instance model
    # check if labels are created from ckpt

    pred_labels = run_inference(
        model_paths=[minimal_instance_multi_class_topdown_ckpt],
        data_path=minimal_instance.as_posix(),
        return_confmaps=False,
        make_labels=True,
        peak_threshold=0.0,
        device="cpu",
        output_path=f"{tmp_path}/test.pkg.slp",
        integral_refinement=None,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 2
    assert len(pred_labels.tracks) == 2
    lf = pred_labels[0]
    assert lf.instances[0].track is not None

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
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_multi_class_topdown_ckpt,
        ],
        data_path=minimal_instance.as_posix(),
        video_index=0,
        frames=[0],
        make_labels=True,
        output_path=tmp_path / "test.slp",
        device="cpu",
        peak_threshold=0.0,
        integral_refinement=None,
    )
    assert isinstance(preds, sio.Labels)
    assert len(preds) == 1
    assert len(preds.tracks) == 2

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_multi_class_topdown_ckpt],
        data_path=minimal_instance.as_posix(),
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
    assert "pred_class_vectors" not in preds[0].keys()

    # if model parameter is not set right
    with pytest.raises(ValueError):
        config = OmegaConf.load(
            f"{minimal_instance_multi_class_topdown_ckpt}/training_config.yaml"
        )
        config_copy = config.copy()
        head_config = config_copy.model_config.head_configs.multi_class_topdown
        del config_copy.model_config.head_configs.multi_class_topdown
        OmegaConf.update(config_copy, "model_config.head_configs.topdown", head_config)
        OmegaConf.save(
            config_copy,
            f"{minimal_instance_multi_class_topdown_ckpt}/training_config.yaml",
        )
        preds = run_inference(
            model_paths=[minimal_instance_multi_class_topdown_ckpt],
            data_path=minimal_instance.as_posix(),
            make_labels=False,
            integral_refinement=None,
        )
    assert "Could not create predictor" in caplog.text

    OmegaConf.save(
        config, f"{minimal_instance_multi_class_topdown_ckpt}/training_config.yaml"
    )

    # centroid + centroid instance model
    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_multi_class_topdown_ckpt,
        ],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=[0.0, 0.0],
        device="cpu",
        integral_refinement="integral",
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) <= 6
    assert len(pred_labels.tracks) <= 6
    assert pred_labels[0].instances[0].track is not None

    # Provider = VideoReader
    # centroid + centered-instance model inference

    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_multi_class_topdown_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        device="cpu",
        peak_threshold=[0.0, 0.0],
        integral_refinement="integral",
        frames=[x for x in range(10)],
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 10
    assert len(pred_labels.tracks) <= 6
    assert pred_labels[0].instances[0].track is not None

    # Provider = VideoReader
    # error in Videoreader but graceful execution

    pred_labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_multi_class_topdown_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        integral_refinement=None,
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
            model_paths=[minimal_instance_multi_class_topdown_ckpt],
            data_path=centered_instance_video.as_posix(),
            make_labels=True,
            output_path=tmp_path / "test.slp",
            max_instances=6,
            device="cpu",
            frames=[x for x in range(10)],
            peak_threshold=0.1,
            integral_refinement=None,
        )
    assert "Error when reading video frame." in caplog.text


def test_single_instance_predictor(
    small_robot_minimal_video,
    small_robot_minimal,
    minimal_instance_single_instance_ckpt,
    tmp_path,
):
    """Test SingleInstancePredictor module."""
    # provider as LabelsReader
    gt_labels = sio.load_slp(small_robot_minimal)

    # check if labels are created from ckpt
    pred_labels = run_inference(
        model_paths=[minimal_instance_single_instance_ckpt],
        data_path=small_robot_minimal.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        device="cpu",
        peak_threshold=0.1,
        integral_refinement=None,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == len(gt_labels)
    assert len(pred_labels[0].instances) == 1
    lf = pred_labels[0]

    # check if the predicted labels have same video and skeleton as the ground truth labels

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
        model_paths=[minimal_instance_single_instance_ckpt],
        data_path=small_robot_minimal.as_posix(),
        make_labels=False,
        peak_threshold=0.3,
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()

    # slp file with video index

    # check if labels are created from ckpt
    pred_labels = run_inference(
        model_paths=[minimal_instance_single_instance_ckpt],
        data_path=small_robot_minimal.as_posix(),
        video_index=0,
        frames=[0],
        device="cpu",
        make_labels=True,
        output_path=tmp_path / "test.slp",
        peak_threshold=0.1,
        integral_refinement=None,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 1
    lf = pred_labels[0]

    # check if the predicted labels have same video and skeleton as the ground truth labels
    gt_labels = sio.load_slp(small_robot_minimal)
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

    # provider as VideoReader

    # check if labels are created from ckpt
    pred_labels = run_inference(
        model_paths=[minimal_instance_single_instance_ckpt],
        data_path=small_robot_minimal_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        device="cpu",
        peak_threshold=0.1,
        integral_refinement=None,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == len(sio.Video(small_robot_minimal_video))
    assert len(pred_labels[0].instances) == 1
    lf = pred_labels[0]

    # check if the predicted labels have same skeleton as the GT labels
    gt_labels = sio.load_slp(small_robot_minimal)
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
        model_paths=[minimal_instance_single_instance_ckpt],
        data_path=small_robot_minimal_video.as_posix(),
        make_labels=False,
        device="cpu",
        peak_threshold=0.3,
        frames=[x for x in range(10)],
        integral_refinement=None,
    )
    assert isinstance(preds, list)
    assert len(preds) == 3  # 10 frames / batch_size=4 = 3 batches
    assert preds[0]["pred_instance_peaks"].shape[0] == 4
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()

    # check loading diff head ckpt
    preprocess_config = {
        "ensure_rgb": False,
        "ensure_grayscale": True,
        "crop_size": None,
        "max_width": None,
        "max_height": None,
    }


def test_bottomup_predictor(
    caplog,
    minimal_instance,
    minimal_instance_bottomup_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Test BottomUpPredictor module."""
    # provider as LabelsReader

    # check if labels are created from ckpt
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.05,
        device="cpu",
        integral_refinement=None,
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
        data_path=minimal_instance.as_posix(),
        make_labels=False,
        max_instances=6,
        peak_threshold=0.05,
        device="cpu",
        integral_refinement=None,
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
        data_path=minimal_instance.as_posix(),
        video_index=0,
        frames=[0],
        make_labels=True,
        output_path=tmp_path / "test.slp",
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds, sio.Labels)
    assert len(preds) == 1

    # with higher threshold
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=1.0,
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 0

    # change to video reader
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.05,
        frames=[x for x in range(10)],
        device="cpu",
        integral_refinement=None,
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 10
    assert len(pred_labels[0].instances) <= 6

    # change to video object
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        input_video=sio.load_video(centered_instance_video.as_posix()),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.05,
        frames=[x for x in range(10)],
        device="cpu",
        integral_refinement=None,
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 10
    assert len(pred_labels[0].instances) <= 6

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path=centered_instance_video.as_posix(),
        make_labels=False,
        max_instances=6,
        peak_threshold=0.05,
        frames=[x for x in range(10)],
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds, list)
    assert len(preds) == 3  # 10 frames / batch_size=4 = 3 batches
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # test with tracking
    pred_labels = run_inference(
        model_paths=[minimal_instance_bottomup_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.05,
        tracking=True,
        candidates_method="local_queues",
        max_tracks=6,
        device="cpu",
        integral_refinement=None,
    )

    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None
            assert instance.tracking_score == 1

    assert len(pred_labels.tracks) <= 6  # should be less than max tracks


def test_multi_class_bottomup_predictor(
    caplog,
    centered_instance_video,
    minimal_instance,
    minimal_instance_multi_class_bottomup_ckpt,
    minimal_instance_multi_class_topdown_ckpt,
    tmp_path,
):
    """Test BottomUpPredictor module."""
    # provider as LabelsReader

    # check if labels are created from ckpt
    pred_labels = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.03,
        device="cpu",
        integral_refinement=None,
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
    assert len(pred_labels.tracks) == 2
    for lf in pred_labels:
        for instance in lf.instances:
            assert instance.track is not None

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        data_path=minimal_instance.as_posix(),
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds, list)
    assert len(preds) == 1
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert "pred_class_maps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # check with labels object
    preds_with_labels = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        input_labels=sio.load_slp(minimal_instance.as_posix()),
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds_with_labels, list)
    assert len(preds_with_labels) == 1
    assert isinstance(preds_with_labels[0], dict)
    assert "pred_confmaps" not in preds_with_labels[0].keys()
    assert "pred_class_maps" not in preds_with_labels[0].keys()
    assert isinstance(preds_with_labels[0]["pred_instance_peaks"], list)
    assert tuple(preds_with_labels[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds_with_labels[0]["pred_peak_values"][0].shape)[1:] == (2,)

    # with video_index
    preds = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        data_path=minimal_instance.as_posix(),
        video_index=0,
        frames=[0],
        make_labels=True,
        output_path=tmp_path / "test.slp",
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds, sio.Labels)
    assert len(preds) == 1
    assert len(preds.tracks) == 2

    # change to video reader
    pred_labels = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.03,
        frames=[x for x in range(10)],
        device="cpu",
        integral_refinement=None,
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 10
    assert len(pred_labels[0].instances) <= 6
    assert len(pred_labels.tracks) <= 6

    # change to video object
    pred_labels = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        input_video=sio.load_video(centered_instance_video.as_posix()),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=6,
        peak_threshold=0.03,
        frames=[x for x in range(10)],
        device="cpu",
        integral_refinement=None,
    )

    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 10
    assert len(pred_labels[0].instances) <= 6
    assert len(pred_labels.tracks) <= 6

    # check if dictionaries are created when make labels is set to False
    preds = run_inference(
        model_paths=[minimal_instance_multi_class_bottomup_ckpt],
        data_path=centered_instance_video.as_posix(),
        make_labels=False,
        max_instances=6,
        peak_threshold=0.03,
        frames=[x for x in range(10)],
        device="cpu",
        integral_refinement=None,
    )
    assert isinstance(preds, list)
    assert len(preds) == 3  # 10 frames / batch_size=4 = 3 batches
    assert isinstance(preds[0], dict)
    assert "pred_confmaps" not in preds[0].keys()
    assert isinstance(preds[0]["pred_instance_peaks"], list)
    assert tuple(preds[0]["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(preds[0]["pred_peak_values"][0].shape)[1:] == (2,)


def test_tracking_only_pipeline(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    minimal_instance,
    small_robot_minimal_video,
    tmp_path,
):
    """Test tracking-only pipeline."""
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        device="cpu" if torch.backends.mps.is_available() else "auto",
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
    )
    labels.save(f"{tmp_path}/preds.slp")

    tracked_labels = run_inference(
        data_path=f"{tmp_path}/preds.slp",
        tracking=True,
        post_connect_single_breaks=True,
        max_instances=2,
        integral_refinement=None,
        tracking_target_instance_count=2,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )

    assert len(tracked_labels.tracks) == 2
    assert len(tracked_labels) == 10

    # track-only pipeline with video file
    with pytest.raises(ValueError):
        labels = run_inference(
            data_path=centered_instance_video.as_posix(),
            tracking=True,
            integral_refinement=None,
            device="cpu" if torch.backends.mps.is_available() else "auto",
        )

    # track-only pipeline with select frames
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    labels.save(f"{tmp_path}/preds.slp")

    tracked_labels = run_inference(
        data_path=f"{tmp_path}/preds.slp",
        tracking=True,
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
        max_instances=2,
        integral_refinement=None,
        frames=[x for x in range(0, 5)],
    )

    assert len(tracked_labels.tracks) == 2
    assert len(tracked_labels) == 5

    # track-only pipeline with video index
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        post_connect_single_breaks=True,
        tracking_target_instance_count=2,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    # Use a regular video file (not .pkg.slp) to avoid sleap-io video matching issues
    labels.videos.append(sio.load_video(small_robot_minimal_video))
    assert len(labels.videos) == 2
    labels.save(f"{tmp_path}/preds.slp")

    tracked_labels = run_inference(
        data_path=f"{tmp_path}/preds.slp",
        tracking=True,
        post_connect_single_breaks=True,
        max_instances=2,
        tracking_target_instance_count=2,
        integral_refinement=None,
        frames=[x for x in range(0, 5)],
        video_index=0,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )

    assert len(tracked_labels.tracks) == 2
    assert len(tracked_labels) == 5

    # neither model nor tracking is provided
    with pytest.raises(Exception):
        labels = run_inference(
            data_path=centered_instance_video.as_posix(),
            tracking=False,
            integral_refinement=None,
            device="cpu" if torch.backends.mps.is_available() else "auto",
        )

    # neither max_instances nor tracking_target_instance_count is provided
    with pytest.raises(ValueError):
        labels = run_inference(
            data_path=centered_instance_video.as_posix(),
            tracking=True,
            integral_refinement=None,
            post_connect_single_breaks=True,
            device="cpu" if torch.backends.mps.is_available() else "auto",
        )

    # racking_target_instance_count is provided
    labels = run_inference(
        data_path=minimal_instance.as_posix(),
        tracking=True,
        integral_refinement=None,
        post_connect_single_breaks=True,
        max_instances=2,
        output_path=tmp_path / "test.slp",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    for lf in labels:
        assert len(lf.instances) <= 2


def test_legacy_topdown_predictor(
    minimal_instance,
    sleap_centroid_model_path,
    sleap_centered_instance_model_path,
    tmp_path,
):
    """Test legacy topdown predictor."""
    pred_labels = run_inference(
        model_paths=[sleap_centroid_model_path, sleap_centered_instance_model_path],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        integral_refinement="integral",
        max_instances=2,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    gt_labels = sio.load_slp(minimal_instance)

    assert np.all(
        np.isclose(
            pred_labels[0].instances[0].numpy(),
            gt_labels[0].instances[1].numpy(),
            atol=6,
        )
    ) or np.all(
        np.isclose(
            pred_labels[0].instances[0].numpy(),
            gt_labels[0].instances[0].numpy(),
            atol=6,
        )
    )
    assert np.all(
        np.isclose(
            pred_labels[0].instances[1].numpy(),
            gt_labels[0].instances[0].numpy(),
            atol=6,
        )
    ) or np.all(
        np.isclose(
            pred_labels[0].instances[1].numpy(),
            gt_labels[0].instances[1].numpy(),
            atol=6,
        )
    )


def test_legacy_bottomup_predictor(
    minimal_instance, sleap_bottomup_model_path, tmp_path
):
    """Test legacy bottomup predictor."""
    pred_labels = run_inference(
        model_paths=[sleap_bottomup_model_path],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        integral_refinement="integral",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    gt_labels = sio.load_slp(minimal_instance)

    assert np.all(
        np.isclose(
            pred_labels[0].instances[0].numpy(),
            gt_labels[0].instances[1].numpy(),
            atol=6,
        )
    ) or np.all(
        np.isclose(
            pred_labels[0].instances[0].numpy(),
            gt_labels[0].instances[0].numpy(),
            atol=6,
        )
    )
    assert np.all(
        np.isclose(
            pred_labels[0].instances[1].numpy(),
            gt_labels[0].instances[0].numpy(),
            atol=6,
        )
    ) or np.all(
        np.isclose(
            pred_labels[0].instances[1].numpy(),
            gt_labels[0].instances[1].numpy(),
            atol=6,
        )
    )


def test_legacy_single_instance_predictor(
    small_robot_minimal, sleap_single_instance_model_path, tmp_path
):
    """Test legacy single instance predictor."""
    pred_labels = run_inference(
        model_paths=[sleap_single_instance_model_path],
        data_path=small_robot_minimal.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        integral_refinement="integral",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    gt_labels = sio.load_slp(small_robot_minimal)


def test_legacy_multiclass_bottomup_predictor(
    minimal_instance,
    sleap_bottomup_multiclass_model_path,
    tmp_path,
):
    """Test legacy multiclass bottomup predictor."""
    pred_labels = run_inference(
        model_paths=[sleap_bottomup_multiclass_model_path],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        integral_refinement="integral",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    gt_labels = sio.load_slp(minimal_instance)


def test_legacy_multiclass_topdown_predictor(
    minimal_instance,
    sleap_centroid_model_path,
    sleap_topdown_multiclass_model_path,
    tmp_path,
):
    """Test legacy multiclass topdown predictor."""
    pred_labels = run_inference(
        model_paths=[sleap_centroid_model_path, sleap_topdown_multiclass_model_path],
        data_path=minimal_instance.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        integral_refinement="integral",
        max_instances=2,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    gt_labels = sio.load_slp(minimal_instance)


def test_predict_main(
    centered_instance_video,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
    tmp_path,
):
    """Test CLI prediction interface works end-to-end.

    This is a single subprocess call that validates:
    - CLI argument parsing works
    - Multiple model paths work
    - Frame range parsing works
    - Tracking options work via CLI
    - Output file is created

    Note: Tracking cleaning logic is tested more thoroughly in
    test_topdown_predictor_with_tracking_cleaning using direct API calls.
    """
    import subprocess

    # Single comprehensive test with tracking and cleaning options
    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "torch-cpu",
        "sleap-nn",
        "track",
        "--model_paths",
        minimal_instance_centroid_ckpt,
        "--model_paths",
        minimal_instance_centered_instance_ckpt,
        "--data_path",
        centered_instance_video.as_posix(),
        "--max_instances",
        "6",
        "--output_path",
        f"{tmp_path}/test.slp",
        "--frames",
        "0-4",  # Reduced from 0-9 for faster test
        "--tracking",
        "--no_empty_frames",
        "--tracking_target_instance_count",
        "1",
        "--tracking_pre_cull_to_target",
        "1",
        "--device",
        "cpu" if torch.backends.mps.is_available() else "auto",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert Path(f"{tmp_path}/test.slp").exists()

    labels = sio.load_slp(f"{tmp_path}/test.slp")
    assert len(labels) == 5
    assert len(labels[0].instances) == 1  # Pre-cull worked


def test_topdown_predictor_with_tracking_cleaning(
    minimal_instance_centroid_ckpt,
    minimal_instance_centered_instance_ckpt,
    centered_instance_video,
    tmp_path,
):
    """Test topdown predictor with tracking cleaning."""
    # pre tracking cleaning
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        tracking=True,
        tracking_target_instance_count=1,
        tracking_pre_cull_to_target=1,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    assert len(labels[0].instances) == 1

    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        tracking=True,
        tracking_clean_instance_count=1,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    assert len(labels[0].instances) == 1

    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        make_labels=True,
        output_path=tmp_path / "test.slp",
        max_instances=2,
        peak_threshold=0.1,
        frames=[x for x in range(0, 10)],
        integral_refinement="integral",
        tracking=True,
        tracking_clean_instance_count=2,
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    assert len(labels[0].instances) == 2

    # no empty frames
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=centered_instance_video.as_posix(),
        no_empty_frames=True,
        frames=[x for x in range(0, 10)],
        peak_threshold=1.0,
        output_path=tmp_path / "test.slp",
        device="cpu" if torch.backends.mps.is_available() else "auto",
    )
    assert len(labels) < 10


def test_video_index_output_path(
    minimal_instance,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
    tmp_path,
):
    """Test that video_index appends video name to output path."""
    # Run inference with video_index and no explicit output_path
    labels = run_inference(
        model_paths=[
            minimal_instance_centroid_ckpt,
            minimal_instance_centered_instance_ckpt,
        ],
        data_path=minimal_instance.as_posix(),
        video_index=0,
        make_labels=True,
        device="cpu",
        peak_threshold=0.0,
        integral_refinement=None,
    )

    # Check that output file was created with video name in path
    expected_pattern = f"*minimal_instance*.predictions.slp"
    output_files = list(Path(minimal_instance).parent.glob(expected_pattern))
    assert len(output_files) > 0, "No output file found with video name in path"

    # The output file should contain the video name
    output_file = output_files[0]
    video_name = Path(labels.videos[0].filename).stem
    assert (
        video_name in output_file.stem
    ), f"Video name '{video_name}' not found in output path '{output_file}'"

    # Clean up
    for f in output_files:
        f.unlink()
