from omegaconf import OmegaConf
import numpy as np
from sleap_nn.training.model_trainer import BottomUpModel, ModelTrainer
from sleap_nn.inference.paf_grouping import PAFScorer
from sleap_nn.inference.bottomup import (
    BottomUpInferenceModel,
)


def test_bottomup_inference_model(minimal_instance_bottomup_ckpt):
    """Test BottomUpInferenceModel."""
    train_config = OmegaConf.load(
        f"{minimal_instance_bottomup_ckpt}/training_config.yaml"
    )
    OmegaConf.update(
        train_config,
        "data_config.train.labels_path",
        "./tests/assets/minimal_instance.pkg.slp",
    )
    OmegaConf.update(
        train_config,
        "data_config.val.labels_path",
        "./tests/assets/minimal_instance.pkg.slp",
    )
    # get dataloader
    trainer = ModelTrainer(train_config)
    trainer._create_data_loaders()
    loader = trainer.val_data_loader

    torch_model = BottomUpModel.load_from_checkpoint(
        f"{minimal_instance_bottomup_ckpt}/best.ckpt", config=train_config
    )

    inference_layer = BottomUpInferenceModel(
        torch_model=torch_model,
        paf_scorer=PAFScorer.from_config(
            config=OmegaConf.create(
                {
                    "confmaps": train_config.model_config.head_configs[
                        "confmaps"
                    ].head_config,
                    "pafs": train_config.model_config.head_configs["pafs"].head_config,
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

    output = inference_layer(next(iter(loader)))[0]
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
                    "confmaps": train_config.model_config.head_configs[
                        "confmaps"
                    ].head_config,
                    "pafs": train_config.model_config.head_configs["pafs"].head_config,
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

    output = inference_layer(next(iter(loader)))[0]
    assert tuple(output["confmaps"].shape) == (1, 2, 192, 192)
    assert tuple(output["part_affinity_fields"].shape) == (1, 96, 96, 2)
    assert output["pred_instance_peaks"].is_nested
    assert output["peaks"][0].shape[-1] == 2
    assert tuple(output["pred_instance_peaks"][0].shape)[1:] == (2, 2)
    assert tuple(output["pred_peak_values"][0].shape)[1:] == (2,)
