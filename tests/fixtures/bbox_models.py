"""A tiny box detector + box-scaled centered-instance model, trained once per test session.

``tests/test_bbox_topdown.py`` and ``tests/inference/test_stream.py`` both need this pair
with the same training settings; training it once (instead of once per module) keeps the
slowest CI job (mac) well under its time limit.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def bbox_det_pair(tmp_path_factory, pytestconfig):
    """Directories of a CenterNet detector ("det") and a box-scaled centered-instance
    model ("ci_bbox"), trained on the one-frame fixture with test_bbox_topdown's tiny CPU
    settings. Read-only for the tests that use them."""
    from sleap_nn.training.model_trainer import ModelTrainer
    from tests.fixtures.datasets import config as config_fixture
    from tests.test_bbox_topdown import _train_config

    tmp_path = tmp_path_factory.mktemp("bbox_det_pair")
    base = config_fixture.__wrapped__(
        Path(pytestconfig.rootdir) / "tests/assets/datasets"
    )
    dirs = {}
    for name, model_type, pre in (
        ("det", "bbox_centernet", {}),
        (
            "ci_bbox",
            "centered_instance",
            {"crop_mode": "bbox", "crop_size": None, "crop_bbox_aspect": "square"},
        ),
    ):
        cfg = _train_config(base, tmp_path, model_type, name, **pre)
        ModelTrainer.get_model_trainer_from_config(cfg).train()
        dirs[name] = tmp_path / name
    return dirs
