import torch

from sleap_nn.data.providers import LabelsReaderDP, process_lf
from sleap_nn.data.resizing import (
    SizeMatcher,
    Resizer,
    PadToStride,
    apply_resizer,
    apply_pad_to_stride,
    apply_sizematcher,
)
import numpy as np
import sleap_io as sio
import pytest


def test_sizematcher(minimal_instance):
    """Test SizeMatcher module for pad images to specified dimensions."""
    l = LabelsReaderDP.from_filename(minimal_instance)
    pipe = SizeMatcher(l, provider=l)
    sample = next(iter(pipe))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    # custom max height and width
    pipe = SizeMatcher(l, provider=l, max_height=500, max_width=400)
    sample = next(iter(pipe))
    instances, image = sample["instances"], sample["image"]
    assert image.shape == torch.Size([1, 1, 500, 400])
    assert instances.shape == torch.Size([1, 2, 2, 2])
    assert torch.isnan(instances[:, 2:, :, :]).all()

    pipe = SizeMatcher(l, max_height=100, max_width=500)
    with pytest.raises(
        Exception,
        match=f"Max height {100} should be greater than the current image height: {384}",
    ):
        sample = next(iter(pipe))

    pipe = SizeMatcher(l, max_height=500, max_width=100)
    with pytest.raises(
        Exception,
        match=f"Max width {100} should be greater than the current image width: {384}",
    ):
        sample = next(iter(pipe))


def test_resizer(minimal_instance):
    """Test Resizer module for resizing images based on given scale."""
    l = LabelsReaderDP.from_filename(minimal_instance)
    pipe = Resizer(l, scale=2, keep_original=False)
    sample = next(iter(pipe))
    image = sample["image"]
    assert image.shape == torch.Size([1, 1, 768, 768])
    assert "original_image" not in sample.keys()

    pipe = Resizer(l, scale=2, keep_original=True)
    sample = next(iter(pipe))
    image = sample["image"]
    assert image.shape == torch.Size([1, 1, 768, 768])
    assert "original_image" in sample.keys()


def test_padtostride(minimal_instance):
    """Test PadToStride module to pad images based on max stride."""
    l = LabelsReaderDP.from_filename(minimal_instance)
    pipe = PadToStride(l, max_stride=200)
    sample = next(iter(pipe))
    image = sample["image"]
    assert image.shape == torch.Size([1, 1, 400, 400])

    pipe = PadToStride(l, max_stride=2)
    sample = next(iter(pipe))
    image = sample["image"]
    assert image.shape == torch.Size([1, 1, 384, 384])

    pipe = PadToStride(l, max_stride=500)
    sample = next(iter(pipe))
    image = sample["image"]
    assert image.shape == torch.Size([1, 1, 500, 500])


def test_apply_resizer(minimal_instance):
    """Test `apply_resizer` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)

    image, instances = apply_resizer(ex["image"], ex["instances"], scale=2.0)
    assert image.shape == torch.Size([1, 1, 768, 768])
    assert torch.all(instances == ex["instances"] * 2.0)


def test_apply_pad_to_stride(minimal_instance):
    """Test `apply_pad_to_stride` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)

    image = apply_pad_to_stride(ex["image"], max_stride=2)
    assert image.shape == torch.Size([1, 1, 384, 384])

    image = apply_pad_to_stride(ex["image"], max_stride=200)
    assert image.shape == torch.Size([1, 1, 400, 400])


def test_apply_sizematcher(minimal_instance):
    """Test `apply_sizematcher` function."""
    labels = sio.load_slp(minimal_instance)
    lf = labels[0]
    ex = process_lf(lf, 0, 2)

    image, _ = apply_sizematcher(ex["image"], 500, 500)
    assert image.shape == torch.Size([1, 1, 500, 500])

    image, _ = apply_sizematcher(ex["image"], 700, 600)
    assert image.shape == torch.Size([1, 1, 700, 600])

    image, _ = apply_sizematcher(ex["image"])
    assert image.shape == torch.Size([1, 1, 384, 384])

    with pytest.raises(
        Exception,
        match=f"Max height {100} should be greater than the current image height: {384}",
    ):
        image = apply_sizematcher(ex["image"], max_height=100, max_width=500)

    with pytest.raises(
        Exception,
        match=f"Max width {100} should be greater than the current image width: {384}",
    ):
        image = apply_sizematcher(ex["image"], max_height=500, max_width=100)
