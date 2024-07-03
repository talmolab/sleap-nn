import sleap_io as sio
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data.dataloader import DataLoader
from sleap_nn.data.resizing import resize_image
from sleap_nn.data.providers import LabelsReader
from sleap_nn.data.normalization import Normalizer
from sleap_nn.data.resizing import SizeMatcher
from sleap_nn.training.model_trainer import (
    SingleInstanceModel,
)
from sleap_nn.inference.single_instance import (
    SingleInstanceInferenceModel,
)


def test_single_instance_inference_model(
    config, minimal_instance, minimal_instance_ckpt
):
    """Test SingleInstanceInferenceModel."""
    OmegaConf.update(config, "data_config.pipeline", "SingleInstanceConfmaps")
    config.model_config.head_configs["confmaps"].head_type = (
        "SingleInstanceConfmapsHead"
    )
    del config.model_config.head_configs["confmaps"].head_config.anchor_part

    torch_model = SingleInstanceModel.load_from_checkpoint(
        f"{minimal_instance_ckpt}/best.ckpt", config=config
    )

    labels = sio.load_slp(minimal_instance)
    # Making our minimal 2-instance example into a single instance example.
    for lf in labels:
        lf.instances = lf.instances[:1]

    provider_pipeline = LabelsReader(labels)
    pipeline = Normalizer(provider_pipeline, is_rgb=False)
    pipeline = SizeMatcher(
        pipeline,
        max_height=None,
        max_width=None,
        provider=provider_pipeline,
    )

    pipeline = pipeline.sharding_filter()
    data_pipeline = DataLoader(
        pipeline,
        batch_size=4,
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
        input_scale=0.5,
    )
    outputs = []
    for x in data_pipeline:
        x["image"] = resize_image(x["image"], 0.5)
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
