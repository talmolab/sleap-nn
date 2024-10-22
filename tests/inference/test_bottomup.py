from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
import torch
import shutil
import sleap_io as sio
from sleap_nn.data.providers import process_lf
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.training.model_trainer import BottomUpModel
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.inference.bottomup import (
    BottomUpInferenceModel,
)


def test_bottomup_inference_model(
    minimal_instance, minimal_instance_bottomup_ckpt, tmp_path: str
):
    """Test BottomUpInferenceModel."""
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
    )
    OmegaConf.update(
        train_config, "trainer_config.save_ckpt_path", f"{tmp_path}/test_model_trainer/"
    )
    OmegaConf.update(
        train_config,
        "data_config.train_labels_path",
        "./tests/assets/minimal_instance.pkg.slp",
    )
    OmegaConf.update(
        train_config,
        "data_config.val_labels_path",
        "./tests/assets/minimal_instance.pkg.slp",
    )

    labels = sio.load_slp(minimal_instance)
    ex = process_lf(labels[0], 0, 2)
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    torch_model = BottomUpModel.load_from_checkpoint(
        f"{minimal_instance_bottomup_ckpt}/best.ckpt",
        config=train_config,
        skeletons=None,
        model_type="bottomup",
    )

    inference_layer = BottomUpInferenceModel(
        torch_model=torch_model,
        paf_scorer=PAFScorer.from_config(
            config=OmegaConf.create(
                {
                    "confmaps": train_config.model_config.head_configs.bottomup[
                        "confmaps"
                    ],
                    "pafs": train_config.model_config.head_configs.bottomup["pafs"],
                }
            )
        ),
        peak_threshold=0.03,
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        pafs_output_stride=4,
        return_confmaps=False,
    )

    output = inference_layer(ex)[0]
    assert "confmaps" not in output.keys()
    assert output["pred_instance_peaks"].is_nested
    assert tuple(output["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(output["pred_peak_values"][0].shape)[1:] == (2,)

    # check return confmaps and pafs
    inference_layer = BottomUpInferenceModel(
        torch_model=torch_model,
        paf_scorer=PAFScorer.from_config(
            config=OmegaConf.create(
                {
                    "confmaps": train_config.model_config.head_configs.bottomup[
                        "confmaps"
                    ],
                    "pafs": train_config.model_config.head_configs.bottomup["pafs"],
                }
            )
        ),
        peak_threshold=0.03,
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        pafs_output_stride=4,
        return_confmaps=True,
        return_pafs=True,
        return_paf_graph=True,
    )

    output = inference_layer(ex)[0]
    assert tuple(output["confmaps"].shape) == (1, 2, 192, 192)
    assert tuple(output["part_affinity_fields"].shape) == (1, 96, 96, 2)
    assert output["pred_instance_peaks"].is_nested
    assert output["peaks"][0].shape[-1] == 2
    assert tuple(output["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(output["pred_peak_values"][0].shape)[1:] == (2,)
