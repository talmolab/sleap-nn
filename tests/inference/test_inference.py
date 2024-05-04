import sleap_io as sio
from typing import Text
import pytest
from omegaconf import OmegaConf
import lightning as L
import numpy as np
import torch
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import SizeMatcher
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import SizeMatcher
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
)
from sleap_nn.data.confidence_maps import make_grid_vectors, make_multi_confmaps
from sleap_nn.model_trainer import (
    TopDownCenteredInstanceModel,
    SingleInstanceModel,
    CentroidModel,
    ModelTrainer,
)
from sleap_nn.inference.inference import (
    Predictor,
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownInferenceModel,
    SingleInstanceInferenceModel,
    CentroidCrop,
)


def initialize_model(config, minimal_instance, minimal_instance_ckpt):
    """Returns data loader, trained torch model and FindInstancePeaks layer to test InferenceModels."""
    # for centered instance model
    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    torch_model = TopDownCenteredInstanceModel.load_from_checkpoint(
        f"{minimal_instance_ckpt}/best.ckpt", config=config
    )

    data_provider = LabelsReader.from_filename(minimal_instance)
    pipeline = SizeMatcher(
        data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)
    pipeline = InstanceCentroidFinder(
        pipeline,
        anchor_ind=config.inference_config.data.preprocessing.anchor_ind,
    )
    pipeline = InstanceCropper(
        pipeline, crop_hw=config.inference_config.data.preprocessing.crop_hw
    )

    data_provider = LabelsReader.from_filename(minimal_instance)
    pipeline = SizeMatcher(
        data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)
    pipeline = InstanceCentroidFinder(
        pipeline,
        anchor_ind=config.inference_config.data.preprocessing.anchor_ind,
    )
    pipeline = InstanceCropper(
        pipeline, crop_hw=config.inference_config.data.preprocessing.crop_hw
    )

    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        **dict(config.inference_config.data.data_loader),
    )
    find_peaks_layer = FindInstancePeaks(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=0.0,
        return_confmaps=False,
    )
    return data_pipeline, torch_model, find_peaks_layer


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
        model_name = config.model_config.head_configs.head_type
        OmegaConf.update(config, "model_config.head_configs.head_type", "instance")
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")
        predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])

    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    OmegaConf.update(config, "model_config.head_configs.head_type", model_name)
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
    _config = config.copy()
    try:
        OmegaConf.update(config, "inference_config.data.provider", "VideoReader")
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt]
        )
        pred_labels = predictor.predict(make_labels=True)
        assert predictor.centroid_config is not None
        assert predictor.confmap_config is not None
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 40
        assert len(pred_labels[0].instances) == 2
    finally:
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")

    # Unrecognized provider
    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    _config = config.copy()
    try:
        OmegaConf.update(config, "inference_config.data.provider", "Reader")
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")
        predictor = Predictor.from_model_paths(
            model_paths=[minimal_instance_centroid_ckpt, minimal_instance_ckpt]
        )
        with pytest.raises(
            Exception,
            match="Provider not recognised. Please use either `LabelsReader` or `VideoReader` as provider",
        ):
            pred_labels = predictor.predict(make_labels=True)

    finally:
        OmegaConf.save(_config, f"{minimal_instance_ckpt}/training_config.yaml")


def test_topdown_inference_model(
    config, minimal_instance, minimal_instance_ckpt, minimal_instance_centroid_ckpt
):
    """Test TopDownInferenceModel class for centroid and cenetered model inferences."""
    # for centered instance model
    loader, _, find_peaks_layer = initialize_model(
        config, minimal_instance, minimal_instance_ckpt
    )

    data_provider = LabelsReader.from_filename(minimal_instance, instances_key=True)
    pipeline = SizeMatcher(
        data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)
    pipeline = InstanceCentroidFinder(
        pipeline,
        anchor_ind=config.inference_config.data.preprocessing.anchor_ind,
    )

    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        **dict(config.inference_config.data.data_loader),
    )

    # if centroid layer is none and centered-instance model.
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=find_peaks_layer
    )
    outputs = []
    for x in loader:
        outputs.append(topdown_inf_layer(x))
    for i in outputs[0]:
        assert i["centroid_val"] == 1
        assert "pred_instance_peaks" in i and "pred_peak_values" in i

    # if centroid layer is none and "instances" not in data
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=FindInstancePeaksGroundTruth()
    )
    example = next(iter(data_pipeline))
    del example["instances"]

    with pytest.raises(
        Exception,
        match="Ground truth data was not detected... Please load both models when predicting on non-ground-truth data.",
    ):
        topdown_inf_layer(example)

    # centroid layer and find peaks
    config = OmegaConf.load(f"{minimal_instance_centroid_ckpt}/training_config.yaml")
    torch_model = CentroidModel.load_from_checkpoint(
        f"{minimal_instance_centroid_ckpt}/best.ckpt", config=config
    )

    data_provider = LabelsReader.from_filename(minimal_instance, instances_key=True)
    pipeline = SizeMatcher(
        data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)
    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        **dict(config.inference_config.data.data_loader),
    )

    centroid_layer = CentroidCrop(
        torch_model=torch_model,
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=False,
        max_instances=2,
        return_crops=True,
        crop_hw=(160, 160),
    )

    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=centroid_layer, instance_peaks=find_peaks_layer
    )
    outputs = topdown_inf_layer(next(iter(data_pipeline)))
    for i in outputs:
        assert i["instance_image"].shape[1:] == (1, 1, 160, 160)
        assert i["pred_instance_peaks"].shape[1:] == (2, 2)


def test_find_instance_peaks_groundtruth(
    config, minimal_instance, minimal_instance_ckpt, minimal_instance_centroid_ckpt
):
    """Test FindInstancePeaksGroundTruth class for running inference on centroid model without cenetered instance model."""
    data_provider = LabelsReader.from_filename(minimal_instance, instances_key=True)
    pipeline = SizeMatcher(
        data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)
    pipeline = InstanceCentroidFinder(
        pipeline,
        anchor_ind=config.inference_config.data.preprocessing.anchor_ind,
    )

    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        **dict(config.inference_config.data.data_loader),
    )

    p = iter(data_pipeline)
    example = next(p)
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=FindInstancePeaksGroundTruth()
    )
    output = topdown_inf_layer(example)[0]
    assert torch.isclose(
        output["pred_instance_peaks"],
        example["instances"].squeeze(),
        atol=1e-6,
        equal_nan=True,
    ).all()
    assert output["pred_peak_values"].shape == (2, 2)

    # with centroid crop class
    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    data_provider = LabelsReader.from_filename(minimal_instance, instances_key=True)
    pipeline = SizeMatcher(
        data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)
    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        **dict(config.inference_config.data.data_loader),
    )

    OmegaConf.update(config, "data_config.pipeline", "CentroidConfmaps")
    OmegaConf.update(
        config, "model_config.head_configs.head_type", "CentroidConfmapsHead"
    )
    del config.model_config.head_configs.head_config.part_names
    config = OmegaConf.load(f"{minimal_instance_centroid_ckpt}/training_config.yaml")
    model = CentroidModel.load_from_checkpoint(
        f"{minimal_instance_centroid_ckpt}/best.ckpt", config=config
    )

    layer = CentroidCrop(
        torch_model=model,
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=False,
        max_instances=2,
        return_crops=False,
        crop_hw=(160, 160),
    )

    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=layer, instance_peaks=FindInstancePeaksGroundTruth()
    )
    output = topdown_inf_layer(next(iter(data_pipeline)))[0]

    assert "pred_instance_peaks" in output.keys()
    assert output["pred_instance_peaks"].shape == (2, 2, 2)
    assert output["pred_peak_values"].shape == (2, 2)


def test_find_instance_peaks(config, minimal_instance, minimal_instance_ckpt):
    """Test FindInstancePeaks class to run inference on the Centered instance model."""
    data_pipeline, torch_model, find_peaks_layer = initialize_model(
        config, minimal_instance, minimal_instance_ckpt
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))
    keys = outputs[0].keys()
    assert "pred_instance_peaks" in keys and "pred_peak_values" in keys
    assert "pred_confmaps" not in keys
    for i in outputs:
        instance = i["pred_instance_peaks"].numpy()
        assert not np.all(np.isnan(instance))

    # high peak threshold
    find_peaks_layer = FindInstancePeaks(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=1.0,
        return_confmaps=False,
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))

    for i in outputs:
        instance = i["pred_instance_peaks"].numpy()
        assert np.all(np.isnan(instance))

    # check return confmaps
    find_peaks_layer = FindInstancePeaks(
        torch_model=torch_model, output_stride=2, peak_threshold=0, return_confmaps=True
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))
    assert "pred_confmaps" in outputs[0].keys()
    assert outputs[0]["pred_confmaps"].shape[-2:] == (80, 80)


def test_single_instance_inference_model(
    config, minimal_instance, minimal_instance_ckpt
):
    """Test SingleInstanceInferenceModel."""
    OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
    OmegaConf.update(
        config, "model_config.head_configs.head_type", "SingleInstanceConfmapsHead"
    )
    del config.model_config.head_configs.head_config.anchor_part

    torch_model = SingleInstanceModel.load_from_checkpoint(
        f"{minimal_instance_ckpt}/best.ckpt", config=config
    )

    labels = sio.load_slp(minimal_instance)
    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    provider_pipeline = LabelsReader(labels)
    pipeline = SizeMatcher(
        provider_pipeline,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = Normalizer(pipeline, is_rgb=config.inference_config.data.is_rgb)

    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        **dict(config.inference_config.data.data_loader),
    )
    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=0.0,
        return_confmaps=False,
    )

    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))
    keys = outputs[0][0].keys()
    assert "pred_instance_peaks" in keys and "pred_peak_values" in keys
    assert "pred_confmaps" not in keys
    for i in outputs:
        instance = i[0]["pred_instance_peaks"].numpy()
        assert not np.all(np.isnan(instance))

    # high peak threshold
    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=1.0,
        return_confmaps=False,
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))

    for i in outputs:
        instance = i[0]["pred_instance_peaks"].numpy()
        assert np.all(np.isnan(instance))

    # check return confmaps
    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model, output_stride=2, peak_threshold=0, return_confmaps=True
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))
    assert "pred_confmaps" in outputs[0][0].keys()
    assert outputs[0][0]["pred_confmaps"].shape[-2:] == (192, 192)


def test_single_instance_predictor(minimal_instance, minimal_instance_ckpt):
    """Test SingleInstancePredictor."""
    # provider as LabelsReader
    _config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")

    config = _config.copy()

    try:
        OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
        OmegaConf.update(
            config, "model_config.head_configs.head_type", "SingleInstanceConfmapsHead"
        )
        del config.model_config.head_configs.head_config.anchor_part
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
        OmegaConf.update(
            config, "model_config.head_configs.head_type", "SingleInstanceConfmapsHead"
        )
        del config.model_config.head_configs.head_config.anchor_part
        OmegaConf.save(config, f"{minimal_instance_ckpt}/training_config.yaml")

        # check if labels are created from ckpt
        predictor = Predictor.from_model_paths(model_paths=[minimal_instance_ckpt])
        pred_labels = predictor.predict(make_labels=True)
        assert isinstance(pred_labels, sio.Labels)
        assert len(pred_labels) == 40
        assert len(pred_labels[0].instances) == 1
        lf = pred_labels[0]

        # check if the predicted labels have same skeleton as the GT labels
        gt_labels = sio.load_slp(minimal_instance)
        assert pred_labels.skeletons == gt_labels.skeletons
        assert lf.frame_idx == 0

        # check if dictionaries are created when make labels is set to False
        preds = predictor.predict(make_labels=False)
        assert isinstance(preds, list)
        assert len(preds) == 10
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
        OmegaConf.update(config, "inference_config.data.provider", "Reader")
        OmegaConf.update(
            config,
            "inference_config.data.path",
            f"./tests/assets/centered_pair_small.mp4",
        )
        OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
        OmegaConf.update(
            config, "model_config.head_configs.head_type", "SingleInstanceConfmapsHead"
        )
        del config.model_config.head_configs.head_config.anchor_part
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


def test_centroid_inference_model(config):
    """Test CentroidCrop class to run inference on centroid models."""
    OmegaConf.update(config, "data_config.pipeline", "CentroidConfmaps")
    OmegaConf.update(
        config, "model_config.head_configs.head_type", "CentroidConfmapsHead"
    )
    del config.model_config.head_configs.head_config.part_names

    trainer = ModelTrainer(config)
    trainer._create_data_loaders()
    loader = next(iter(trainer.val_data_loader))
    trainer._initialize_model()
    model = trainer.model

    # return crops = False
    layer = CentroidCrop(
        torch_model=model,
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=False,
        max_instances=2,
        return_crops=False,
        crop_hw=(160, 160),
    )

    out = layer(loader)
    assert tuple(out["centroids"].shape) == (1, 1, 2, 2)
    assert tuple(out["centroid_vals"].shape) == (1, 2)
    assert "instance_image" not in out.keys()

    # return crops = False
    layer = CentroidCrop(
        torch_model=model,
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=False,
        max_instances=2,
        return_crops=True,
        crop_hw=(160, 160),
    )
    out = layer(loader)
    assert len(out) == 1
    out = out[0]
    assert tuple(out["centroid"].shape) == (2, 2)
    assert tuple(out["centroid_val"].shape) == (2,)
    assert tuple(out["instance_image"].shape) == (2, 1, 1, 160, 160)
    assert tuple(out["instance_bbox"].shape) == (2, 1, 4, 2)
