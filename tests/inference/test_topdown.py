import pytest
from omegaconf import OmegaConf
import numpy as np
import torch
from pathlib import Path
from loguru import logger
from _pytest.logging import LogCaptureFixture

import sleap_io as sio
from sleap_nn.data.providers import process_lf
from sleap_nn.data.instance_centroids import generate_centroids
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.data.instance_cropping import generate_crops
from sleap_nn.training.lightning_modules import (
    CentroidLightningModule,
    TopDownCenteredInstanceLightningModule,
    TopDownCenteredInstanceMultiClassLightningModule,
)
from sleap_nn.inference.topdown import (
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownMultiClassFindInstancePeaks,
    TopDownInferenceModel,
    CentroidCrop,
)


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


def initialize_model(config, minimal_instance, minimal_instance_centered_instance_ckpt):
    """Returns trained torch model and FindInstancePeaks layer to test InferenceModels."""
    # for centered instance model
    config = OmegaConf.load(
        f"{minimal_instance_centered_instance_ckpt}/training_config.yaml"
    )
    torch_model = TopDownCenteredInstanceLightningModule.load_from_checkpoint(
        f"{minimal_instance_centered_instance_ckpt}/best.ckpt",
        model_type="centered_instance",
        backbone_type="unet",
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        map_location="cpu",
        weights_only=False,
    )

    find_peaks_layer = FindInstancePeaks(
        torch_model=torch_model.to("cpu"),
        output_stride=2,
        peak_threshold=0.0,
        return_confmaps=False,
    )
    return torch_model, find_peaks_layer


def test_centroid_inference_model(
    config, minimal_instance, tmp_path, minimal_instance_centroid_ckpt
):
    """Test CentroidCrop class to run inference on centroid models."""
    config = OmegaConf.load(
        (Path(minimal_instance_centroid_ckpt) / "training_config.yaml").as_posix()
    )

    labels = sio.load_slp(minimal_instance)
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["instances"] = ex["instances"].unsqueeze(dim=0)
    ex["frame_idx"] = ex["frame_idx"].unsqueeze(dim=0)
    ex["video_idx"] = ex["video_idx"].unsqueeze(dim=0)
    ex["orig_size"] = ex["orig_size"].unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    model = CentroidLightningModule.load_from_checkpoint(
        f"{minimal_instance_centroid_ckpt}/best.ckpt",
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        skeletons=None,
        model_type="centroid",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )

    # return crops = False
    layer = CentroidCrop(
        torch_model=model.to("cpu"),
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=True,
        max_instances=6,
        return_crops=False,
        crop_hw=(160, 160),
    )

    out = layer(ex)
    assert tuple(out["centroids"].shape) == (1, 1, 6, 2)
    assert tuple(out["centroid_vals"].shape) == (1, 6)
    assert "instance_image" not in out.keys()
    assert "pred_centroid_confmaps" in out.keys()

    # return crops = True
    layer = CentroidCrop(
        torch_model=model.to("cpu"),
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=True,
        max_instances=2,
        return_crops=True,
        crop_hw=(160, 160),
    )
    out = layer(ex)
    assert len(out) == 1
    out = out[0]
    assert tuple(out["centroid"].shape) == (2, 2)
    assert tuple(out["centroid_val"].shape) == (2,)
    assert tuple(out["instance_image"].shape) == (2, 1, 1, 160, 160)
    assert tuple(out["instance_bbox"].shape) == (2, 1, 4, 2)


def test_find_instance_peaks_groundtruth(
    config,
    minimal_instance,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
):
    """Test FindInstancePeaksGroundTruth class for running inference on centroid model without centered instance model."""
    labels = sio.load_slp(minimal_instance)
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["instances"] = ex["instances"].unsqueeze(dim=0)
    ex["frame_idx"] = ex["frame_idx"].unsqueeze(dim=0)
    ex["video_idx"] = ex["video_idx"].unsqueeze(dim=0)
    ex["orig_size"] = ex["orig_size"].unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    example = ex
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=CentroidCrop(
            use_gt_centroids=True, anchor_ind=0, crop_hw=(160, 160)
        ),
        instance_peaks=FindInstancePeaksGroundTruth(),
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
    labels = sio.load_slp(minimal_instance)
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["instances"] = ex["instances"].unsqueeze(dim=0)
    ex["frame_idx"] = ex["frame_idx"].unsqueeze(dim=0)
    ex["video_idx"] = ex["video_idx"].unsqueeze(dim=0)
    ex["orig_size"] = ex["orig_size"].unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])
    config = OmegaConf.load(
        f"{minimal_instance_centered_instance_ckpt}/training_config.yaml"
    )
    OmegaConf.update(
        config,
        "model_config.head_configs.centroid",
        config.model_config.head_configs.centered_instance,
    )
    del config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centroid["confmaps"].part_names
    config = OmegaConf.load(f"{minimal_instance_centroid_ckpt}/training_config.yaml")
    torch_model = CentroidLightningModule.load_from_checkpoint(
        f"{minimal_instance_centroid_ckpt}/best.ckpt",
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        model_type="centroid",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )

    centroid_layer = CentroidCrop(
        torch_model=torch_model.to("cpu"),
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
        centroid_crop=centroid_layer, instance_peaks=FindInstancePeaksGroundTruth()
    )

    output = topdown_inf_layer(ex)[0]

    assert "pred_instance_peaks" in output.keys()
    assert output["pred_instance_peaks"].shape == (2, 2, 2)
    assert output["pred_peak_values"].shape == (2, 2)
    assert output["image"].shape == (2, 1, 1, 384, 384)
    assert "pred_centroid_confmaps" not in output.keys()

    centroid_layer = CentroidCrop(
        torch_model=torch_model.to("cpu"),
        peak_threshold=0.0,
        refinement="integral",
        integral_patch_size=5,
        output_stride=2,
        return_confmaps=True,
        max_instances=2,
        return_crops=False,
        crop_hw=(160, 160),
    )

    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=centroid_layer, instance_peaks=FindInstancePeaksGroundTruth()
    )

    output = topdown_inf_layer(ex)[0]

    assert "pred_instance_peaks" in output.keys()
    assert output["pred_instance_peaks"].shape == (2, 2, 2)
    assert output["pred_peak_values"].shape == (2, 2)
    assert output["image"].shape == (2, 1, 1, 384, 384)
    assert "pred_centroid_confmaps" in output.keys()
    assert output["pred_centroid_confmaps"].shape == (2, 1, 96, 96)


def test_find_instance_peaks(
    config, minimal_instance, minimal_instance_centered_instance_ckpt
):
    """Test FindInstancePeaks class to run inference on the Centered instance model."""
    torch_model, find_peaks_layer = initialize_model(
        config, minimal_instance, minimal_instance_centered_instance_ckpt
    )
    labels = sio.load_slp(minimal_instance)
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["instances"] = ex["instances"].unsqueeze(dim=0)
    ex["frame_idx"] = ex["frame_idx"].unsqueeze(dim=0)
    ex["video_idx"] = ex["video_idx"].unsqueeze(dim=0)
    ex["orig_size"] = ex["orig_size"].unsqueeze(dim=0)
    ex["centroids"] = generate_centroids(ex["instances"], 0)
    ex["instances"], centroids = (
        ex["instances"][0, 0],
        ex["centroids"][0, 0],
    )  # n_samples=1
    ex["eff_scale"] = torch.Tensor([1.0])

    for cnt, (instance, centroid) in enumerate(zip(ex["instances"], centroids)):
        if cnt == ex["num_instances"]:
            break

        res = generate_crops(ex["image"][0], instance, centroid, (160, 160))

        res["frame_idx"] = ex["frame_idx"]
        res["video_idx"] = ex["video_idx"]
        res["num_instances"] = ex["num_instances"]
        res["orig_size"] = ex["orig_size"]
        res["instance_image"] = res["instance_image"].unsqueeze(dim=0)
        res["eff_scale"] = torch.Tensor([1.0])

        break

    outputs = []
    outputs.append(find_peaks_layer(res))
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
    outputs.append(find_peaks_layer(res))
    for i in outputs:
        instance = i["pred_instance_peaks"].numpy()
        assert np.all(np.isnan(instance))

    # check return confmaps
    find_peaks_layer = FindInstancePeaks(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=0,
        return_confmaps=True,
    )
    outputs = []
    outputs.append(find_peaks_layer(res))
    assert "pred_confmaps" in outputs[0].keys()
    assert outputs[0]["pred_confmaps"].shape[-2:] == (80, 80)


def test_find_instance_peaks_multiclass(
    config, minimal_instance, minimal_instance_multi_class_topdown_ckpt
):
    """Test FindInstancePeaks class to run inference on the Centered instance model."""
    config = OmegaConf.load(
        f"{minimal_instance_multi_class_topdown_ckpt}/training_config.yaml"
    )
    torch_model = TopDownCenteredInstanceMultiClassLightningModule.load_from_checkpoint(
        f"{minimal_instance_multi_class_topdown_ckpt}/best.ckpt",
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        model_type="multi_class_topdown",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )

    find_peaks_layer = TopDownMultiClassFindInstancePeaks(
        torch_model=torch_model.to("cpu"),
        output_stride=2,
        peak_threshold=0.0,
        return_confmaps=False,
    )

    labels = sio.load_slp(minimal_instance)
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["instances"] = ex["instances"].unsqueeze(dim=0)
    ex["frame_idx"] = ex["frame_idx"].unsqueeze(dim=0)
    ex["video_idx"] = ex["video_idx"].unsqueeze(dim=0)
    ex["orig_size"] = ex["orig_size"].unsqueeze(dim=0)
    ex["centroids"] = generate_centroids(ex["instances"], 0)
    ex["instances"], centroids = (
        ex["instances"][0, 0],
        ex["centroids"][0, 0],
    )  # n_samples=1
    ex["eff_scale"] = torch.Tensor([1.0])

    for cnt, (instance, centroid) in enumerate(zip(ex["instances"], centroids)):
        if cnt == ex["num_instances"]:
            break

        res = generate_crops(ex["image"][0], instance, centroid, (160, 160))

        res["frame_idx"] = ex["frame_idx"]
        res["video_idx"] = ex["video_idx"]
        res["num_instances"] = ex["num_instances"]
        res["orig_size"] = ex["orig_size"]
        res["instance_image"] = res["instance_image"].unsqueeze(dim=0)
        res["eff_scale"] = torch.Tensor([1.0])

        break

    outputs = []
    outputs.append(find_peaks_layer(res))
    keys = outputs[0].keys()
    assert (
        "pred_instance_peaks" in keys
        and "pred_peak_values" in keys
        and "instance_scores" in keys
    )
    assert "pred_confmaps" not in keys and "pred_class_vectors" not in keys
    for i in outputs:
        instance = i["pred_instance_peaks"].numpy()
        assert not np.all(np.isnan(instance))

        assert i["instance_scores"].shape == (1,)

    # check return confmaps
    find_peaks_layer = TopDownMultiClassFindInstancePeaks(
        torch_model=torch_model,
        output_stride=2,
        peak_threshold=0,
        return_confmaps=True,
        return_class_vectors=True,
    )
    outputs = []
    outputs.append(find_peaks_layer(res))
    assert "pred_confmaps" in outputs[0].keys()
    assert outputs[0]["pred_confmaps"].shape[-2:] == (80, 80)
    assert outputs[0]["pred_class_vectors"].shape == (1, 2)


def test_topdown_inference_model(
    caplog,
    config,
    minimal_instance,
    minimal_instance_centered_instance_ckpt,
    minimal_instance_centroid_ckpt,
):
    """Test TopDownInferenceModel class for centroid and cenetered model inferences."""
    # for centered instance model
    _, find_peaks_layer = initialize_model(
        config, minimal_instance, minimal_instance_centered_instance_ckpt
    )

    labels = sio.load_slp(minimal_instance)
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["instances"] = ex["instances"].unsqueeze(dim=0)
    ex["frame_idx"] = ex["frame_idx"].unsqueeze(dim=0)
    ex["video_idx"] = ex["video_idx"].unsqueeze(dim=0)
    ex["orig_size"] = ex["orig_size"].unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    # if gt centroids and centered-instance model.
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=CentroidCrop(
            use_gt_centroids=True, anchor_ind=0, crop_hw=(160, 160), return_crops=True
        ),
        instance_peaks=find_peaks_layer,
    )
    outputs = []
    outputs.append(topdown_inf_layer(ex))
    for i in outputs[0]:
        assert i["centroid_val"][0] == 1
        assert "pred_instance_peaks" in i and "pred_peak_values" in i

    # if gt centroids and "instances" not in data
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=CentroidCrop(
            use_gt_centroids=True, anchor_ind=0, crop_hw=(160, 160), return_crops=True
        ),
        instance_peaks=FindInstancePeaksGroundTruth(),
    )
    example = ex
    del example["instances"]

    with pytest.raises(
        Exception,
        match="Ground truth data was not detected... Please load both models when predicting on non-ground-truth data.",
    ):
        topdown_inf_layer(example)
    assert "Ground truth data was not detected." in caplog.text
    # centroid layer and find peaks
    config = OmegaConf.load(f"{minimal_instance_centroid_ckpt}/training_config.yaml")
    torch_model = CentroidLightningModule.load_from_checkpoint(
        f"{minimal_instance_centroid_ckpt}/best.ckpt",
        backbone_config=config.model_config.backbone_config,
        head_configs=config.model_config.head_configs,
        model_type="centroid",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )

    centroid_layer = CentroidCrop(
        torch_model=torch_model.to("cpu"),
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
    outputs = topdown_inf_layer(ex)
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
    assert "Ground truth data was not detected." in caplog.text
