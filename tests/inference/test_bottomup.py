from omegaconf import OmegaConf
import numpy as np
from pathlib import Path
import torch
import shutil
import sleap_io as sio
from sleap_nn.data.providers import process_lf
from sleap_nn.data.normalization import apply_normalization
from sleap_nn.training.lightning_modules import (
    BottomUpLightningModule,
    BottomUpMultiClassLightningModule,
)
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.inference.bottomup import (
    BottomUpInferenceModel,
    BottomUpMultiClassInferenceModel,
)


def test_bottomup_inference_model(
    minimal_instance, minimal_instance_bottomup_ckpt, tmp_path: str
):
    """Test BottomUpInferenceModel."""
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
    )
    OmegaConf.update(train_config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(train_config, "trainer_config.run_name", "test_model_trainer")
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
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    torch_model = BottomUpLightningModule.load_from_checkpoint(
        f"{minimal_instance_bottomup_ckpt}/best.ckpt",
        backbone_config=train_config.model_config.backbone_config,
        head_configs=train_config.model_config.head_configs,
        model_type="bottomup",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
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
        peak_threshold=0.2,
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        pafs_output_stride=4,
        return_confmaps=False,
    )

    output = inference_layer(ex)[0]
    assert "pred_confmaps" not in output.keys()
    assert isinstance(output["pred_instance_peaks"], list)
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
        peak_threshold=0.2,
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        pafs_output_stride=4,
        return_confmaps=True,
        return_pafs=True,
        return_paf_graph=True,
    )

    output = inference_layer(ex)[0]
    assert tuple(output["pred_confmaps"].shape) == (1, 2, 192, 192)
    assert tuple(output["pred_part_affinity_fields"].shape) == (1, 96, 96, 2)
    assert isinstance(output["pred_instance_peaks"], list)
    assert output["peaks"][0].shape[-1] == 2
    assert tuple(output["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(output["pred_peak_values"][0].shape)[1:] == (2,)


def test_multiclass_bottomup_inference_model(
    minimal_instance, minimal_instance_multi_class_bottomup_ckpt, tmp_path: str
):
    """Test BottomUpInferenceModel."""
    train_config = OmegaConf.load(
        f"{minimal_instance_multi_class_bottomup_ckpt}/training_config.yaml"
    )
    OmegaConf.update(train_config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(train_config, "trainer_config.run_name", "test_model_trainer")
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
    ex = process_lf(
        instances_list=labels[0].instances,
        img=labels[0].image,
        frame_idx=labels[0].frame_idx,
        video_idx=0,
        max_instances=2,
    )
    ex["image"] = apply_normalization(ex["image"]).unsqueeze(dim=0)
    ex["eff_scale"] = torch.Tensor([1.0])

    torch_model = BottomUpMultiClassLightningModule.load_from_checkpoint(
        f"{minimal_instance_multi_class_bottomup_ckpt}/best.ckpt",
        model_type="multi_class_bottomup",
        backbone_type="unet",
        backbone_config=train_config.model_config.backbone_config,
        head_configs=train_config.model_config.head_configs,
        map_location="cpu",
        weights_only=False,
    )

    inference_layer = BottomUpMultiClassInferenceModel(
        torch_model=torch_model,
        peak_threshold=0.2,
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        class_maps_output_stride=2,
        return_confmaps=False,
        return_class_maps=False,
    )

    output = inference_layer(ex)[0]
    assert "pred_confmaps" not in output.keys()
    assert "pred_class_maps" not in output.keys()
    assert isinstance(output["pred_instance_peaks"], list)
    assert tuple(output["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(output["pred_peak_values"][0].shape)[1:] == (2,)

    # check return confmaps and pafs
    inference_layer = BottomUpMultiClassInferenceModel(
        torch_model=torch_model,
        peak_threshold=0.2,
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        class_maps_output_stride=2,
        return_confmaps=True,
        return_class_maps=True,
    )

    output = inference_layer(ex)[0]
    assert tuple(output["pred_confmaps"].shape) == (1, 2, 192, 192)
    assert tuple(output["pred_class_maps"].shape) == (1, 2, 192, 192)
    assert isinstance(output["pred_instance_peaks"], list)
    assert tuple(output["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(output["pred_peak_values"][0].shape)[1:] == (2,)


def test_bottomup_max_peaks_per_node_guardrail(
    minimal_instance, minimal_instance_bottomup_ckpt, tmp_path: str
):
    """Test BottomUpInferenceModel max_peaks_per_node guardrail.

    This test verifies that when max_peaks_per_node is set to a very low value,
    the model returns empty predictions instead of attempting PAF scoring which
    would cause a combinatorial explosion.
    """
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
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
    ex["eff_scale"] = torch.Tensor([1.0])

    torch_model = BottomUpLightningModule.load_from_checkpoint(
        f"{minimal_instance_bottomup_ckpt}/best.ckpt",
        backbone_config=train_config.model_config.backbone_config,
        head_configs=train_config.model_config.head_configs,
        model_type="bottomup",
        backbone_type="unet",
        map_location="cpu",
        weights_only=False,
    )

    # Test with a very low max_peaks_per_node that will trigger the guardrail
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
        peak_threshold=0.0,  # Low threshold = many peaks
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        pafs_output_stride=4,
        max_peaks_per_node=1,  # Very low limit - will trigger guardrail
    )

    output = inference_layer(ex)[0]

    # Should return empty predictions (shape (0, n_nodes, 2))
    assert isinstance(output["pred_instance_peaks"], list)
    assert output["pred_instance_peaks"][0].shape[0] == 0  # No instances
    assert output["pred_peak_values"][0].shape[0] == 0
    assert output["instance_scores"][0].shape[0] == 0

    # Verify that without the guardrail (None), predictions are returned
    inference_layer_no_guard = BottomUpInferenceModel(
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
        peak_threshold=0.2,  # Normal threshold
        refinement="integral",
        integral_patch_size=5,
        cms_output_stride=2,
        pafs_output_stride=4,
        max_peaks_per_node=None,  # No guardrail
    )

    output_no_guard = inference_layer_no_guard(ex)[0]
    # With normal threshold and no guardrail, should get predictions
    assert isinstance(output_no_guard["pred_instance_peaks"], list)
    # The minimal instance has 2 nodes, so predictions should have shape (n_inst, 2, 2)
    assert output_no_guard["pred_instance_peaks"][0].shape[-1] == 2
