import sleap_io as sio
from typing import Text
import pytest
from omegaconf import OmegaConf
import lightning as L
import numpy as np
from pathlib import Path
from torch.utils.data.dataloader import DataLoader
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.pipelines import (
    TopdownConfmapsPipeline,
    SingleInstanceConfmapsPipeline,
)
from sleap_nn.model_trainer import TopDownCenteredInstanceModel, SingleInstanceModel
from sleap_nn.inference.inference import (
    Predictor,
    FindInstancePeaks,
    TopDownInferenceModel,
    SingleInstanceInferenceModel,
)


def test_topdown_centered_predictor(minimal_instance_ckpt, minimal_instance):
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
        assert "pred_instance_peaks" in i.keys() and "pred_peak_values" in i.keys()


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


def test_single_instance_inference_model(
    config, minimal_instance, minimal_instance_ckpt
):
    OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
    OmegaConf.update(
        config, "model_config.head_configs.head_type", "SingleInstanceConfmapsHead"
    )
    del config.model_config.head_configs.head_config.anchor_part

    torch_model = SingleInstanceModel.load_from_checkpoint(
        f"{minimal_instance_ckpt}/best.ckpt", config=config
    )
    data_pipeline = SingleInstanceConfmapsPipeline(config.inference_config.data)

    labels = sio.load_slp(minimal_instance)
    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    provider_pipeline = LabelsReader(labels)
    pipeline = data_pipeline.make_training_pipeline(data_provider=provider_pipeline)

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
    keys = outputs[0].keys()
    assert "pred_instance_peaks" in keys and "pred_peak_values" in keys
    assert "pred_confmaps" not in keys
    for i in outputs:
        instance = i["pred_instance_peaks"].numpy()
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
        instance = i["pred_instance_peaks"].numpy()
        assert np.all(np.isnan(instance))

    # check return confmaps
    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model, output_stride=2, peak_threshold=0, return_confmaps=True
    )
    outputs = []
    for x in data_pipeline:
        outputs.append(find_peaks_layer(x))
    assert "pred_confmaps" in outputs[0].keys()
    assert outputs[0]["pred_confmaps"].shape[-2:] == (192, 192)


def test_single_instance_predictor(minimal_instance, minimal_instance_ckpt):
    # store the original config
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
