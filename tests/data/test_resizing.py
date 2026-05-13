import torch

from sleap_nn.data.providers import process_lf
from sleap_nn.data import resizing
from sleap_nn.data.resizing import (
    apply_resizer,
    apply_pad_to_stride,
    apply_sizematcher,
)
import numpy as np
import sleap_io as sio
import pytest
from loguru import logger
from _pytest.logging import LogCaptureFixture


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


def test_apply_resizer(minimal_instance):
    """Test `apply_resizer` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    image, instances = apply_resizer(ex["image"], ex["instances"], scale=2.0)
    assert image.shape == torch.Size([1, 1, 768, 768])
    assert torch.all(instances == ex["instances"] * 2.0)


def test_apply_pad_to_stride(minimal_instance):
    """Test `apply_pad_to_stride` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    image = apply_pad_to_stride(ex["image"], max_stride=2)
    assert image.shape == torch.Size([1, 1, 384, 384])

    image = apply_pad_to_stride(ex["image"], max_stride=200)
    assert image.shape == torch.Size([1, 1, 400, 400])


def test_apply_sizematcher(caplog, minimal_instance):
    """Test `apply_sizematcher` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    resizing._SIZEMATCHER_WARNED_KEYS.clear()

    image, _ = apply_sizematcher(ex["image"], 500, 500)
    assert image.shape == torch.Size([1, 1, 500, 500])

    image, _ = apply_sizematcher(ex["image"], 700, 600)
    assert image.shape == torch.Size([1, 1, 700, 600])

    image, _ = apply_sizematcher(ex["image"])
    assert image.shape == torch.Size([1, 1, 384, 384])

    image, eff = apply_sizematcher(ex["image"], 100, 480)
    assert image.shape == torch.Size([1, 1, 100, 480])


def test_apply_sizematcher_warns_on_size_mismatch(caplog, minimal_instance):
    """Size-mismatched frames (either direction) must warn about the slow path."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    resizing._SIZEMATCHER_WARNED_KEYS.clear()
    caplog.clear()
    apply_sizematcher(ex["image"], 100, 480)  # 384x384 -> downscale
    assert any(
        "downscaled" in r.message and "slower" in r.message for r in caplog.records
    )

    resizing._SIZEMATCHER_WARNED_KEYS.clear()
    caplog.clear()
    apply_sizematcher(ex["image"], 500, 500)  # 384x384 -> upscale
    assert any(
        "upscaled" in r.message and "slower" in r.message for r in caplog.records
    )


def test_apply_sizematcher_warns_once_per_size(caplog, minimal_instance):
    """Repeated calls with the same sizes should not flood the log."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(
        instances_list=lf.instances,
        img=lf.image,
        frame_idx=lf.frame_idx,
        video_idx=0,
        max_instances=2,
    )

    resizing._SIZEMATCHER_WARNED_KEYS.clear()
    caplog.clear()

    for _ in range(5):
        apply_sizematcher(ex["image"], 100, 480)
    matches = [r for r in caplog.records if "100x480" in r.message]
    assert len(matches) == 1
