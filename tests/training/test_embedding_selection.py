"""Checkpoint selection for the ``embedding`` model type.

The selection metric is a retrieval metric (`eval/val/rank1` and friends), logged
by `EmbeddingEvaluationCallback` rather than by the training step — which makes
its availability, its name, and the cadence it is logged at part of the
contract:

- `ModelCheckpoint` raises, not warns, when its monitored key was never logged.
- `trainer_config.eval.select_metric` has to exist on `EvalConfig` to be
  settable from YAML at all.
"""

import math

import pytest
import sleap_io as sio
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from sleap_nn.config.trainer_config import EvalConfig
from sleap_nn.training.callbacks import EmbeddingEvaluationCallback
from sleap_nn.training.model_trainer import ModelTrainer


def _trainer(config, tmp_path, minimal_instance, model_type=None, **updates):
    cfg = config.copy()
    OmegaConf.update(cfg, "trainer_config.save_ckpt", True)
    OmegaConf.update(cfg, "trainer_config.ckpt_dir", f"{tmp_path}")
    OmegaConf.update(cfg, "trainer_config.run_name", "emb_selection_test")
    for key, value in updates.items():
        OmegaConf.update(cfg, key, value)

    labels = sio.load_slp(minimal_instance)
    trainer = ModelTrainer.get_model_trainer_from_config(
        cfg, train_labels=[labels], val_labels=[labels]
    )
    if model_type is not None:
        trainer.model_type = model_type
    return trainer


def _checkpoint_callback(trainer):
    _, callbacks = trainer._setup_loggers_callbacks(
        viz_train_dataset=None, viz_val_dataset=None
    )
    return next(c for c in callbacks if isinstance(c, ModelCheckpoint))


# ─────────────────────────────────────────────────────────────────────────
# eval.select_metric must be reachable from a config file
# ─────────────────────────────────────────────────────────────────────────
def test_select_metric_is_a_declared_eval_config_field():
    """A structured merge raises `ConfigKeyError` for undeclared keys.

    Without the field, `trainer_config.eval.select_metric` in a YAML config could
    only ever fail — so the selection-mode table in `model_trainer` was dead for
    config users and selection was always rank-1.
    """
    schema = OmegaConf.structured(EvalConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create({"select_metric": "mAP"}))
    assert merged.select_metric == "mAP"
    # The default is unchanged.
    assert OmegaConf.structured(EvalConfig).select_metric == "rank1"


@pytest.mark.parametrize(
    "metric,mode",
    [
        ("rank1", "max"),
        ("mAP", "max"),
        ("auc", "max"),
        ("knn_acc", "max"),
        ("eer", "min"),
    ],
)
def test_select_metric_drives_monitor_and_mode(
    config, tmp_path, minimal_instance, metric, mode
):
    """Each valid metric selects on its own key, with the right direction."""
    trainer = _trainer(
        config,
        tmp_path,
        minimal_instance,
        model_type="embedding",
        **{"trainer_config.eval.select_metric": metric},
    )
    ckpt = _checkpoint_callback(trainer)

    assert ckpt.monitor == f"eval/val/{metric}"
    assert ckpt.mode == mode


def test_unknown_select_metric_raises(config, tmp_path, minimal_instance):
    """A typo must not silently monitor a key that is never logged."""
    trainer = _trainer(
        config,
        tmp_path,
        minimal_instance,
        model_type="embedding",
        **{"trainer_config.eval.select_metric": "recall_at_5"},
    )
    with pytest.raises(ValueError, match="is not a valid embedding selection metric"):
        _checkpoint_callback(trainer)


# ─────────────────────────────────────────────────────────────────────────
# The checkpointer's cadence must match the metric's cadence
# ─────────────────────────────────────────────────────────────────────────
def test_checkpoint_cadence_follows_eval_frequency(config, tmp_path, minimal_instance):
    """`every_n_epochs` matches `eval.frequency` for an eval-gated monitor.

    The metric only exists on eval epochs, so "best" may only be considered
    there — otherwise the checkpointer either raises on a missing key or picks a
    checkpoint using a value measured at a different epoch.
    """
    trainer = _trainer(
        config,
        tmp_path,
        minimal_instance,
        model_type="embedding",
        **{"trainer_config.eval.frequency": 3},
    )
    assert _checkpoint_callback(trainer).every_n_epochs == 3


def test_pose_model_checkpoints_every_epoch(config, tmp_path, minimal_instance):
    """`val/loss` is logged every epoch, so the cadence stays 1."""
    trainer = _trainer(
        config, tmp_path, minimal_instance, **{"trainer_config.eval.frequency": 3}
    )
    ckpt = _checkpoint_callback(trainer)

    assert ckpt.monitor == "val/loss"
    assert ckpt.every_n_epochs == 1


def test_explicit_monitor_keeps_every_epoch(config, tmp_path, minimal_instance):
    """A user-configured monitor is theirs to schedule; we don't re-gate it."""
    trainer = _trainer(
        config,
        tmp_path,
        minimal_instance,
        model_type="embedding",
        **{
            "trainer_config.model_ckpt.monitor": "val/fg_iou",
            "trainer_config.model_ckpt.mode": "max",
            "trainer_config.eval.frequency": 3,
        },
    )
    ckpt = _checkpoint_callback(trainer)

    assert ckpt.monitor == "val/fg_iou"
    assert ckpt.every_n_epochs == 1


# ─────────────────────────────────────────────────────────────────────────
# The callback must log the monitored keys on EVERY validation epoch
# ─────────────────────────────────────────────────────────────────────────
class _FakeStrategy:
    def broadcast(self, obj, src=0):
        return obj

    def barrier(self):
        pass


class _FakeTrainer:
    def __init__(self, current_epoch):
        self.current_epoch = current_epoch
        self.is_global_zero = True
        self.sanity_checking = False
        self.loggers = []
        self.strategy = _FakeStrategy()


class _FakeModule:
    def __init__(self, predictions=None, ground_truth=None):
        self.val_predictions = predictions or []
        self.val_ground_truth = ground_truth or []
        self._collect_val_predictions = False
        self.logged = {}

    def log(self, key, value, **kwargs):
        self.logged[key] = value


def _embeddings(n_per_class=3, n_classes=2, dim=4):
    """Well-separated embeddings, so the metrics are non-degenerate."""
    import numpy as np
    import torch

    predictions, ground_truth = [], []
    for cls in range(n_classes):
        base = np.zeros(dim, dtype=np.float64)
        base[cls] = 1.0
        for i in range(n_per_class):
            vec = base + 0.01 * (i + 1)
            vec = vec / np.linalg.norm(vec)
            predictions.append({"embedding": torch.tensor(vec)})
            ground_truth.append({"label": cls})
    return predictions, ground_truth


SELECTION_KEYS = [
    "eval/val/rank1",
    "eval/val/mAP",
    "eval/val/auc",
    "eval/val/eer",
    "eval/val/knn_acc",
]


def test_callback_logs_the_keys_before_the_first_eval_epoch():
    """With frequency > 1, epoch 0 does not evaluate — but must still log.

    Logging nothing left the monitored key absent from `callback_metrics`, and
    `ModelCheckpoint` raises `MisconfigurationException: could not find the
    monitored key` the first time it looks. That killed every
    `eval.frequency > 1` run at the end of epoch 0.
    """
    callback = EmbeddingEvaluationCallback(eval_frequency=2)
    predictions, ground_truth = _embeddings()
    module = _FakeModule(predictions, ground_truth)

    callback.on_validation_epoch_end(_FakeTrainer(current_epoch=0), module)

    assert set(SELECTION_KEYS) <= set(module.logged)
    # Nothing has been measured yet, so nothing may look like a best score.
    assert all(math.isnan(module.logged[k]) for k in SELECTION_KEYS)


def test_callback_carries_the_last_values_through_non_eval_epochs():
    """An eval epoch's values persist over the epochs that skip evaluation."""
    callback = EmbeddingEvaluationCallback(eval_frequency=2)
    predictions, ground_truth = _embeddings()

    # Epoch 1: (1 + 1) % 2 == 0 -> evaluates.
    eval_module = _FakeModule(predictions, ground_truth)
    callback.on_validation_epoch_end(_FakeTrainer(current_epoch=1), eval_module)
    assert not math.isnan(eval_module.logged["eval/val/rank1"])
    assert eval_module.logged["eval/val/rank1"] == pytest.approx(1.0)

    # Epoch 2: does not evaluate, and the module collected nothing.
    skipped_module = _FakeModule()
    callback.on_validation_epoch_end(_FakeTrainer(current_epoch=2), skipped_module)

    assert set(SELECTION_KEYS) <= set(skipped_module.logged)
    assert skipped_module.logged["eval/val/rank1"] == pytest.approx(1.0)


def test_callback_survives_an_empty_val_set():
    """No collected embeddings on an eval epoch still logs, with NaN."""
    callback = EmbeddingEvaluationCallback(eval_frequency=1)
    module = _FakeModule()

    callback.on_validation_epoch_end(_FakeTrainer(current_epoch=0), module)

    assert set(SELECTION_KEYS) <= set(module.logged)
    assert all(math.isnan(module.logged[k]) for k in SELECTION_KEYS)
