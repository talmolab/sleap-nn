import torch
import sleap_io as sio
from typing import Text
import pytest
from omegaconf import OmegaConf
import lightning as L
import numpy as np
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import TopdownConfmapsPipeline
from sleap_nn.model_trainer import TopDownCenteredInstanceModel
from sleap_nn.inference.inference import (
    Predictor,
    FindInstancePeaks,
    TopDownInferenceModel,
)


def test_topdown_centered_predictor(minimal_instance_ckpt, minimal_instance):
    # for centered instance model
    # check if labels are created from ckpt
    predictor = Predictor.from_model_paths(
        ckpt_paths={"centered": minimal_instance_ckpt}, model="topdown"
    )
    pred_labels = predictor.predict(make_labels=True)
    assert predictor.centroid_config is None
    assert isinstance(pred_labels, sio.Labels)
    assert len(pred_labels) == 1
    assert len(pred_labels[0].instances) == 2
    lf = pred_labels[0]

    # check if the predicted labels have same video and skeleton as the ground truth labels
    gt_labels = sio.load_slp(minimal_instance)
    gt_lf = gt_labels[0]
    assert pred_labels.videos == gt_labels.videos
    assert pred_labels.skeleton == gt_labels.skeleton
    assert lf.frame_idx == gt_lf.frame_idx
    assert lf.instances[0].numpy().shape == gt_lf.instances[0].numpy().shape
    assert lf.instances[1].numpy().shape == gt_lf.instances[1].numpy().shape

    # TODO: check if dictionaries are created when make labels is set to False
    preds = predictor.predict(make_labels=False)

    # if model parameter is not set right
    with pytest.raises(
        ValueError, match=f"Could not create predictor from model name:\nTop"
    ):
        predictor = Predictor.from_model_paths(
            ckpt_paths={"centered": minimal_instance_ckpt}, model="Top"
        )

    # if neither of centered nor centroid ckpts are given.
    with pytest.raises(
        ValueError,
        match="Either the centroid or topdown confidence map model must be provided.",
    ):
        predictor = Predictor.from_model_paths(
            ckpt_paths={"single_instance": minimal_instance_ckpt}, model="topdown"
        )


def initialize_model(minimal_instance, minimal_instance_ckpt, config):
    # for centered instance model
    torch_model = TopDownCenteredInstanceModel.load_from_checkpoint(
        minimal_instance_ckpt, config=config
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


def test_topdown_inference_model(minimal_instance, minimal_instance_ckpt, config):
    # for centered instance model
    data_pipeline, _, find_peaks_layer = initialize_model(
        minimal_instance, minimal_instance_ckpt, config
    )
    topdown_inf_layer = TopDownInferenceModel(
        centroid_crop=None, instance_peaks=find_peaks_layer
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(topdown_inf_layer(x))
    for i in outputs:
        assert i["centroid_val"] == 1
        assert "pred_instance_peaks" in i.keys() and "pred_peak_values" in i.keys()


def test_find_instance_peaks(minimal_instance, minimal_instance_ckpt, config):
    data_pipeline, torch_model, find_peaks_layer = initialize_model(
        minimal_instance, minimal_instance_ckpt, config
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
    print("-------------------------confmaps: ", outputs[0]["pred_confmaps"].shape)
    assert outputs[0]["pred_confmaps"].shape[-2:] == (80, 80)
