import pytest
from omegaconf import OmegaConf
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.resizing import resize_image
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import SizeMatcher, Resizer, PadToStride
from sleap_nn.data.instance_centroids import InstanceCentroidFinder
from sleap_nn.data.instance_cropping import InstanceCropper
from sleap_nn.training.model_trainer import (
    CentroidModel,
    ModelTrainer,
    TopDownCenteredInstanceModel,
)
from sleap_nn.inference.topdown import (
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownInferenceModel,
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
    pipeline = Normalizer(data_provider, is_rgb=config.inference_config.data.is_rgb)
    pipeline = SizeMatcher(
        pipeline,
        provider=data_provider,
        max_height=config.inference_config.data.max_height,
        max_width=config.inference_config.data.max_width,
    )
    pipeline = InstanceCentroidFinder(
        pipeline,
        anchor_ind=config.inference_config.data.preprocessing.anchor_ind,
    )
    pipeline = InstanceCropper(
        pipeline,
        crop_hw=config.inference_config.data.preprocessing.crop_hw,
    )
    pipeline = Resizer(
        pipeline, scale=1.0, image_key="instance_image", instances_key="instance"
    )
    pipeline = PadToStride(pipeline, max_stride=16, image_key="instance_image")

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


def test_centroid_inference_model(config):
    """Test CentroidCrop class to run inference on centroid models."""
    OmegaConf.update(config, "data_config.pipeline", "CentroidConfmaps")
    config.model_config.head_configs[0].head_type = "CentroidConfmapsHead"
    del config.model_config.head_configs[0].head_config.part_names

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


def test_find_instance_peaks_groundtruth(
    config, minimal_instance, minimal_instance_ckpt, minimal_instance_centroid_ckpt
):
    """Test FindInstancePeaksGroundTruth class for running inference on centroid model without centered instance model."""
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
    config.model_config.head_configs[0].head_type = "CentroidConfmapsHead"
    del config.model_config.head_configs[0].head_config.part_names
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
        peak_threshold=2.0,
        return_confmaps=False,
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))
    print(f"outputs len: {len(outputs)}")
    for i in outputs:
        instance = i["pred_instance_peaks"].numpy()
        print(f"imgs: {i['instance_image'].shape}")
        print(f"pred vals: {i['pred_peak_values']}")
        assert np.all(np.isnan(instance))

    # check return confmaps
    find_peaks_layer = FindInstancePeaks(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=0,
        return_confmaps=True,
        input_scale=0.5,
    )
    outputs = []
    for x in data_pipeline:
        x["image"] = resize_image(x["image"], 0.5)
        outputs.append(find_peaks_layer(x))
    assert "pred_confmaps" in outputs[0].keys()
    assert outputs[0]["pred_confmaps"].shape[-2:] == (40, 40)


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

    # centroid layer and "instances" not in example
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=centroid_layer, instance_peaks=FindInstancePeaksGroundTruth()
    )
    with pytest.raises(
        Exception,
        match="Ground truth data was not detected... Please load both models when predicting on non-ground-truth data.",
    ):
        outputs = topdown_inf_layer(example)
