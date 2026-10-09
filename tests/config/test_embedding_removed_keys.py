"""Dead embedding config keys are gone, and old configs still load (emb-review F9).

`head_configs.embedding.embedding.loss_weight` (a single-head model has no other loss
to weigh it against) and `objective.negatives.proximity_filter_px` ("Reserved (P2);
unused in P1") were never read. Training wrote every field into
`training_config.yaml`, so every embedding config saved before their removal carries
them: loading such a config drops them with a warning instead of failing on an
unknown key.
"""

from pathlib import Path

import pytest
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.config.get_config import get_head_configs
from sleap_nn.config.model_config import (
    EmbeddingHeadConfig,
    NegativesConfig,
    resolve_embedding_objective,
)
from sleap_nn.config.training_job_config import verify_training_cfg
from sleap_nn.training.model_trainer import ModelTrainer
from tests.training.test_embedding_train_e2e import (  # noqa: F401
    _raw_config,
    tracked_slp,
)

LEAF = "model_config.head_configs.embedding.embedding"
SAMPLE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "sample_configs"
    / "config_embedding_convnext.yaml"
)


@pytest.fixture
def warnings_log():
    """Collect loguru WARNING+ messages (whitespace-collapsed)."""
    messages = []
    sink_id = logger.add(
        lambda m: messages.append(" ".join(str(m).split())), level="WARNING"
    )
    yield messages
    logger.remove(sink_id)


def _old_config(tracked_slp, tmp_path, **updates):
    """A config as an older sleap-nn saved it: with both removed keys set."""
    cfg = _raw_config(tracked_slp, tmp_path, "old_config", **updates)
    OmegaConf.update(cfg, f"{LEAF}.loss_weight", 1.0, force_add=True)
    OmegaConf.update(
        cfg, f"{LEAF}.objective.negatives.proximity_filter_px", 12.0, force_add=True
    )
    return cfg


def _removed_key_warnings(messages):
    return [m for m in messages if "never used and has been removed" in m]


def test_the_fields_are_gone():
    assert not hasattr(EmbeddingHeadConfig(), "loss_weight")
    assert not hasattr(NegativesConfig(), "proximity_filter_px")


def test_old_training_config_loads_through_the_trainer(
    tracked_slp, tmp_path, warnings_log
):
    """`ModelTrainer` (training, fine-tuning, `sleap-nn train --config`) accepts a
    config that still carries the removed keys and drops them, with a warning."""
    cfg = _old_config(tracked_slp, tmp_path)
    trainer = ModelTrainer.get_model_trainer_from_config(cfg)

    leaf = OmegaConf.select(trainer.config, LEAF)
    assert "loss_weight" not in leaf
    assert "proximity_filter_px" not in leaf.objective.negatives
    warned = _removed_key_warnings(warnings_log)
    assert len(warned) == 2
    assert any(f"{LEAF}.loss_weight" in m for m in warned)
    assert any(f"{LEAF}.objective.negatives.proximity_filter_px" in m for m in warned)
    # The caller's config is not modified.
    assert OmegaConf.select(cfg, f"{LEAF}.loss_weight") == 1.0


def test_verified_config_does_not_write_the_keys(tracked_slp, tmp_path, warnings_log):
    """A config saved from now on carries neither key, and loading it says nothing."""
    cfg = verify_training_cfg(_raw_config(tracked_slp, tmp_path, "new_config"))
    saved = tmp_path / "training_config.yaml"
    OmegaConf.save(cfg, saved.as_posix())
    text = saved.read_text()
    assert "loss_weight" not in OmegaConf.to_yaml(OmegaConf.select(cfg, LEAF))
    assert "proximity_filter_px" not in text

    verify_training_cfg(OmegaConf.load(saved.as_posix()))
    assert _removed_key_warnings(warnings_log) == []


def test_old_objective_resolves():
    """The one reader of the objective tolerates the removed negatives key."""
    objective = resolve_embedding_objective(
        {"negatives": {"proximity_filter_px": 12.0, "restrict_same_video": True}}
    )
    assert objective.negatives.restrict_same_video is True
    assert not hasattr(objective.negatives, "proximity_filter_px")


def test_old_head_dict_builds(warnings_log):
    """The Python config API (`get_head_configs` / `sleap_nn.train.train`) too."""
    old = {
        "embedding": {
            "embedding": {
                "embedding_dim": 32,
                "loss_weight": 1.0,
                "objective": {"negatives": {"proximity_filter_px": None}},
            }
        }
    }
    heads = get_head_configs(old)
    assert heads.embedding.embedding.embedding_dim == 32
    assert not hasattr(heads.embedding.embedding, "loss_weight")
    assert len(_removed_key_warnings(warnings_log)) == 2
    # The caller's dict is not modified.
    assert old["embedding"]["embedding"]["loss_weight"] == 1.0


def test_sample_config_verifies():
    """The shipped sample config no longer sets the removed key."""
    cfg = OmegaConf.load(SAMPLE_CONFIG.as_posix())
    assert "loss_weight" not in OmegaConf.select(cfg, LEAF)
    verify_training_cfg(cfg)


def test_old_model_dir_still_embeds(tmp_path):
    """`sleap-nn predict` on a model whose saved config still carries the removed
    keys: the inference path reads the raw `training_config.yaml` (the objective
    too, when it builds the LightningModule)."""
    from tests.cli.test_embedding_carrier import (
        N_FRAMES,
        _both_carriers_slp,
        _embedded_carriers,
        _model_dir,
    )

    model_dir = _model_dir(tmp_path, "old_model", detection_mode="pose")
    cfg = OmegaConf.load(model_dir / "training_config.yaml")
    OmegaConf.update(cfg, f"{LEAF}.loss_weight", 1.0, force_add=True)
    OmegaConf.update(
        cfg, f"{LEAF}.objective.negatives.proximity_filter_px", 12.0, force_add=True
    )
    OmegaConf.save(cfg, model_dir / "training_config.yaml")

    embedded = _embedded_carriers(tmp_path, model_dir, _both_carriers_slp(tmp_path))
    assert embedded == {"pose": 2 * N_FRAMES, "mask": 0}
