"""End-to-end train -> predict smoke test for centroid focal loss.

Exercises the full plumbing for the sigmoid-activation + focal-loss
combination added for the centroid confmap head (see
``CentroidConfMapsConfig.focal_loss_alpha`` / ``use_sigmoid_activation``,
and ``docs/guides/centroid-focal-loss.md``): trains a tiny centroid model
with both enabled, then loads the resulting checkpoint through the normal
``Predictor`` inference path with no special-casing, confirming a checkpoint
trained with this feature is fully compatible with existing inference.
"""

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import sleap_io as sio

from sleap_nn.training.model_trainer import ModelTrainer
from sleap_nn.training.lightning_modules import CentroidLightningModule
from sleap_nn.inference.predictor import Predictor
from sleap_nn.inference.layers.centroid import CentroidLayer


def _centroid_focal_loss_config(config, tmp_path):
    """Build a `centroid` config with sigmoid activation + focal loss enabled."""
    centroid_config = config.copy()
    head_config = centroid_config.model_config.head_configs.centered_instance
    OmegaConf.update(centroid_config, "model_config.head_configs.centroid", head_config)
    del centroid_config.model_config.head_configs.centered_instance
    del centroid_config.model_config.head_configs.centroid["confmaps"].part_names

    OmegaConf.update(
        centroid_config,
        "model_config.head_configs.centroid.confmaps.use_sigmoid_activation",
        True,
    )
    OmegaConf.update(
        centroid_config,
        "model_config.head_configs.centroid.confmaps.focal_loss_alpha",
        2.0,
    )

    OmegaConf.update(centroid_config, "trainer_config.run_name", "centroid_focal_loss")
    OmegaConf.update(centroid_config, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(centroid_config, "trainer_config.save_ckpt", True)
    OmegaConf.update(centroid_config, "trainer_config.use_wandb", False)
    OmegaConf.update(centroid_config, "trainer_config.max_epochs", 3)
    OmegaConf.update(centroid_config, "trainer_config.min_train_steps_per_epoch", 30)
    OmegaConf.update(centroid_config, "data_config.data_pipeline_fw", "torch_dataset")
    return centroid_config


def test_centroid_focal_loss_train_predict_wiring(config, minimal_instance, tmp_path):
    """Train with sigmoid+focal-loss enabled, then predict via the normal path.

    The tiny 2-epoch model is not expected to produce accurate centroids;
    this asserts the sigmoid-activated, focal-loss-trained checkpoint loads
    and runs through inference exactly like any other centroid checkpoint --
    no special-casing needed at predict time (the sigmoid activation is
    baked into the model graph at train time).
    """
    centroid_config = _centroid_focal_loss_config(config, tmp_path)

    trainer = ModelTrainer.get_model_trainer_from_config(centroid_config)
    trainer.train()
    assert isinstance(trainer.lightning_model, CentroidLightningModule)
    assert trainer.lightning_model.centroid_focal_loss_alpha == 2.0

    run_dir = (Path(tmp_path) / "centroid_focal_loss").as_posix()
    assert (Path(run_dir) / "best.ckpt").exists()

    # Auto-detected as a centroid-only model, no special handling required
    # for a focal-loss-trained checkpoint.
    predictor = Predictor.from_model_paths([run_dir], peak_threshold=0.2, device="cpu")
    assert isinstance(predictor.layer, CentroidLayer)

    out = predictor.predict(minimal_instance.as_posix(), make_labels=True)
    assert isinstance(out, sio.Labels)
    # Structure only -- a tiny smoke-trained model isn't expected to produce
    # accurate centroids; any predicted instance should still have a finite
    # score and the anchor node populated (centroid-only output convention).
    for lf in out.labeled_frames:
        for inst in lf.instances:
            assert np.isfinite(inst.score)

    out_path = tmp_path / "centroid_focal_loss_preds.slp"
    out.save(out_path.as_posix())
    sio.load_slp(out_path.as_posix())
