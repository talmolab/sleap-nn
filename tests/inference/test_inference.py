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
from sleap_nn.data.pipelines import TopdownConfmapsPipeline
from sleap_nn.model_trainer import TopDownCenteredInstanceModel
from sleap_nn.inference.inference import (
    Predictor,
    FindInstancePeaks,
    FindInstancePeaksGroundTruth,
    TopDownInferenceModel,
)


def initialize_model(minimal_instance, minimal_instance_ckpt):
    # for centered instance model
    config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    torch_model = TopDownCenteredInstanceModel.load_from_checkpoint(
        f"{minimal_instance_ckpt}/best.ckpt", config=config
    )
    data_pipeline = TopdownConfmapsPipeline(config.inference_config.data)

    labels = sio.load_slp(minimal_instance)
    provider_pipeline = LabelsReader(labels)
    pipeline = data_pipeline.make_training_pipeline(data_provider=provider_pipeline)

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


def test_topdown_centered_predictor(minimal_instance, minimal_instance_ckpt):
    # for centered instance model
    # check if labels are created from ckpt
    data_pipeline, _, find_peaks_layer = initialize_model(
        minimal_instance, minimal_instance_ckpt
    )

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


def test_topdown_inference_model(minimal_instance, minimal_instance_ckpt):
    # for centered instance model
    data_pipeline, _, find_peaks_layer = initialize_model(
        minimal_instance, minimal_instance_ckpt
    )

    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=find_peaks_layer
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(topdown_inf_layer(x))
    for i in outputs:
        assert i["centroid_val"] == 1
        assert "pred_instance_peaks" in i and "pred_peak_values" in i

    # if centroid layer is none and "instances" not in data
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=FindInstancePeaksGroundTruth()
    )
    example = next(iter(data_pipeline))

    with pytest.raises(
        Exception,
        match="Ground truth data was not detected... Please load both models when predicting on non-ground-truth data.",
    ):
        topdown_inf_layer(example)


def test_find_instance_peaks_groundtruth(minimal_instance, minimal_instance_ckpt):
    data_pipeline, _, _ = initialize_model(minimal_instance, minimal_instance_ckpt)
    p = iter(data_pipeline)
    e1 = next(p)
    e2 = next(p)
    example = e1.copy()
    example["instance"] = e1["instance"].unsqueeze(dim=1)
    example["centroid"] = e1["centroid"].unsqueeze(dim=1)
    inst = e2["instance"].unsqueeze(dim=1) + 300
    cent = e2["centroid"].unsqueeze(dim=1) + 300
    inst[0, 0, 0, 0, 0] = torch.nan
    example["instances"] = torch.cat(
        [example["instance"], inst, torch.full(example["instance"].shape, torch.nan)],
        dim=2,
    )
    example["centroids"] = torch.cat(
        [example["centroid"], cent, torch.full(example["centroid"].shape, torch.nan)],
        dim=2,
    )
    example["num_instances"] = [2, 2, 2, 2]
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=FindInstancePeaksGroundTruth()
    )
    output = topdown_inf_layer(example)
    assert torch.isclose(
        output["pred_instance_peaks"], example["instances"], atol=1e-6, equal_nan=True
    ).all()
    assert output["pred_peak_values"].shape == (1, 1, 3, 2)


def test_find_instance_peaks(minimal_instance, minimal_instance_ckpt):
    data_pipeline, torch_model, find_peaks_layer = initialize_model(
        minimal_instance, minimal_instance_ckpt
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
