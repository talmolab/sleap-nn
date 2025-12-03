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


def test_single_instance_inference_model(
    small_robot_minimal, minimal_instance_single_instance_ckpt
):
    """Test SingleInstanceInferenceModel."""
    training_config = OmegaConf.load(
        f"{minimal_instance_single_instance_ckpt}/training_config.yaml"
    )

    torch_model = SingleInstanceLightningModule.load_from_checkpoint(
        f"{minimal_instance_single_instance_ckpt}/best.ckpt",
        backbone_config=training_config.model_config.backbone_config,
        head_configs=training_config.model_config.head_configs,
        model_type="single_instance",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )

    labels = sio.load_slp(small_robot_minimal)
    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    find_peaks_layer = SingleInstanceInferenceModel(
        torch_model=torch_model.to("cpu"),
        output_stride=4,
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
        output_stride=4,
        peak_threshold=3.0,
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
        output_stride=4,
        peak_threshold=0,
        return_confmaps=True,
    )
    outputs = []
    outputs.append(find_peaks_layer(ex))
    assert "pred_confmaps" in outputs[0][0].keys()
    assert outputs[0][0]["pred_confmaps"].shape[-2:] == (40, 70)
