import sleap_io as sio
from omegaconf import OmegaConf
import numpy as np
import torch
from sleap_nn.data.resizing import resize_image
from sleap_nn.training.lightning_modules import (
    SingleInstanceLightningModule,
)
from sleap_nn.inference.single_instance import (
    SingleInstanceInferenceModel,
)
from sleap_nn.data.providers import process_lf
from sleap_nn.data.normalization import apply_normalization


def test_single_instance_inference_model(minimal_instance, minimal_instance_ckpt):
    """Test SingleInstanceInferenceModel."""
    config = OmegaConf.load(f"{minimal_instance_ckpt}/initial_config.yaml")
    head_config = config.model_config.head_configs.centered_instance
    del config.model_config.head_configs.centered_instance
    OmegaConf.update(config, "model_config.head_configs.single_instance", head_config)
    del config.model_config.head_configs.single_instance.confmaps.anchor_part

    training_config = OmegaConf.load(f"{minimal_instance_ckpt}/training_config.yaml")
    head_config = training_config.model_config.head_configs.centered_instance
    del training_config.model_config.head_configs.centered_instance
    OmegaConf.update(
        training_config, "model_config.head_configs.single_instance", head_config
    )
    del training_config.model_config.head_configs.single_instance.confmaps.anchor_part

    torch_model = SingleInstanceLightningModule.load_from_checkpoint(
        f"{minimal_instance_ckpt}/best.ckpt",
        config=training_config,
        model_type="single_instance",
        backbone_type="unet",
    )

    labels = sio.load_slp(minimal_instance)
    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    ex = process_lf(labels[0], 0, 2)
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model.to("cpu"),
        output_stride=2,
        peak_threshold=0.0,
        return_confmaps=False,
    )

    outputs = []
    outputs.append(find_peaks_layer(ex))
    keys = outputs[0][0].keys()
    assert "pred_instance_peaks" in keys and "pred_peak_values" in keys
    assert "pred_confmaps" not in keys
    for i in outputs:
        instance = i[0]["pred_instance_peaks"].numpy()
        assert not np.all(np.isnan(instance))

    # high peak threshold
    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model.to("cpu"),
        output_stride=2,
        peak_threshold=1.0,
        return_confmaps=False,
        input_scale=0.5,
    )
    outputs = []
    ex["image"] = resize_image(ex["image"], 0.5)
    outputs.append(find_peaks_layer(ex))

    for i in outputs:
        instance = i[0]["pred_instance_peaks"].numpy()
        assert np.all(np.isnan(instance))

    # check return confmaps
    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model.to("cpu"),
        output_stride=2,
        peak_threshold=0,
        return_confmaps=True,
    )
    outputs = []
    outputs.append(find_peaks_layer(ex))
    assert "pred_confmaps" in outputs[0][0].keys()
    assert outputs[0][0]["pred_confmaps"].shape[-2:] == (96, 96)
