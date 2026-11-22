"""Warnings when a box detector feeds a box-scaled crop model."""

from types import SimpleNamespace

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger
from omegaconf import OmegaConf

from sleap_nn.inference.loaders import _check_crop_stage


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


def _assets(det_padding, crop_padding, det_source="instances", crop_source="instances"):
    crop_cfg = OmegaConf.create(
        {
            "data_config": {
                "bbox_source": crop_source,
                "bbox_padding": crop_padding,
                "preprocessing": {"crop_mode": "bbox"},
            }
        }
    )
    det_cfg = OmegaConf.create(
        {"data_config": {"bbox_source": det_source, "bbox_padding": det_padding}}
    )
    return SimpleNamespace(
        confmap_config=crop_cfg,
        detector_assets=SimpleNamespace(bottomup_config=det_cfg),
    )


def _warnings(caplog):
    return [r.message for r in caplog.records if r.levelname == "WARNING"]


def test_padding_difference_is_quantified(caplog):
    caplog.clear()
    _check_crop_stage(_assets(9.0, 0.0), ["bbox_detr", "centered_instance"])
    (msg,) = _warnings(caplog)
    assert "each side of the detector's boxes is about 18 px longer" in msg
    assert "on purpose" in msg
    caplog.clear()
    _check_crop_stage(_assets(0.0, 4.0), ["bbox_detr", "centered_instance"])
    (msg,) = _warnings(caplog)
    assert "about 8 px shorter" in msg
    # User-drawn boxes are never padded: no size is claimed then.
    for source in ("bboxes", "auto"):
        caplog.clear()
        _check_crop_stage(
            _assets(9.0, 0.0, det_source=source, crop_source=source),
            ["bbox_detr", "centered_instance"],
        )
        (msg,) = _warnings(caplog)
        assert "px" not in msg and "user-drawn" in msg


@pytest.mark.parametrize("padding", [0.0, 5.0])
def test_matching_settings_do_not_warn(caplog, padding):
    caplog.clear()
    _check_crop_stage(_assets(padding, padding), ["bbox_detr", "centered_instance"])
    assert not _warnings(caplog)


def test_box_source_difference_warns(caplog):
    caplog.clear()
    _check_crop_stage(
        _assets(0.0, 0.0, det_source="bboxes", crop_source="instances"),
        ["bbox_detr", "centered_instance"],
    )
    (msg,) = _warnings(caplog)
    assert "define boxes differently" in msg
