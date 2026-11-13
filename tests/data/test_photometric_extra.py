"""E19 appearance ops (`sleap_nn.data.photometric_extra`) and the DETR zoom-out."""

import numpy as np
import pytest
import torch

from sleap_nn.config.data_config import DetrAugmentationConfig, IntensityConfig
from sleap_nn.data.detr_augmentation import detr_train_transform, zoom_out
from sleap_nn.data.photometric_extra import ExtraPhotometric
from sleap_nn.data.topdown_warp import WarpAug


def _img(c=3, h=64, w=48, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, c), dtype=np.uint8)


def test_defaults_inactive_and_no_draws():
    e = ExtraPhotometric.from_intensity({})
    assert not e.active
    a = WarpAug.from_configs(None, {"contrast_p": 0.0})
    assert not a.extra.active
    # Same stream with and without the (inactive) extra ops.
    im = _img()
    r1, r2 = np.random.default_rng(3), np.random.default_rng(3)
    out = a.intensity(im.copy(), r1)
    assert np.array_equal(out, im)
    assert r1.random() == r2.random()


def test_config_fields_flow_into_warpaug():
    cfg = dict(
        gray_p=1.0,
        invert_p=1.0,
        gamma_min=0.5,
        gamma_max=2.0,
        gamma_p=0.0,
        downup_min=2.0,
        downup_max=3.0,
        downup_p=0.0,
        jpeg_quality_min=20,
        jpeg_quality_max=40,
        jpeg_p=0.0,
    )
    IntensityConfig(**cfg)  # validators accept them
    a = WarpAug.from_configs(None, cfg)
    assert a.extra.active and a.extra.gray_p == 1.0 and a.extra.downup == (2.0, 3.0)
    assert a.extra.jpeg_quality == (20, 40)


def test_invert_and_gray():
    im = _img()
    a = WarpAug.from_configs(None, {"invert_p": 1.0})
    out = a.intensity(im.copy(), np.random.default_rng(0))
    assert np.array_equal(out, 255 - im)
    g = WarpAug.from_configs(None, {"gray_p": 1.0}).intensity(
        im.copy(), np.random.default_rng(0)
    )
    assert np.array_equal(g[..., 0], g[..., 1]) and np.array_equal(g[..., 1], g[..., 2])
    # Gray (1-channel) crops: colour ops skip, tone ops apply.
    im1 = _img(c=1)
    out1 = WarpAug.from_configs(None, {"gray_p": 1.0, "invert_p": 1.0}).intensity(
        im1.copy(), np.random.default_rng(0)
    )
    assert out1.shape == im1.shape and np.array_equal(out1, 255 - im1)


def test_gamma_monotone_and_range():
    x = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (4, 1, 3))
    e = ExtraPhotometric(gamma=(2.0, 2.0), gamma_p=1.0)
    lut = e.tone_lut(np.random.default_rng(0))
    assert lut[0] == 0 and lut[255] == 255 and lut[128] < 128
    assert np.all(np.diff(lut.astype(int)) >= 0)
    del x


def test_channel_shuffle_and_gain():
    im = np.zeros((8, 8, 3), np.uint8)
    im[..., 0], im[..., 1], im[..., 2] = 10, 100, 200
    e = ExtraPhotometric(channel_shuffle_p=1.0)
    out = e.color([im], np.random.default_rng(1))[0]
    assert sorted(out[0, 0].tolist()) == [10, 100, 200]
    e = ExtraPhotometric(channel_gain_max=0.5, channel_gain_p=1.0)
    out = e.color([im], np.random.default_rng(1))[0]
    assert not np.array_equal(out, im)
    assert (out.astype(int) - im.astype(int)).std() > 0


def test_downup_blurs_and_keeps_shape():
    im = _img(h=128, w=96)
    e = ExtraPhotometric(downup=(4.0, 4.0), downup_p=1.0)
    out = e.degrade([im], np.random.default_rng(0))[0]
    assert out.shape == im.shape
    hf = lambda z: np.abs(np.diff(z.astype(float), axis=0)).mean()  # noqa: E731
    assert hf(out) < 0.5 * hf(im)


def test_jpeg_keeps_shape_and_channels():
    for c in (1, 3):
        im = _img(c=c)
        e = ExtraPhotometric(jpeg_quality=(10, 10), jpeg_p=1.0)
        out = e.degrade([im], np.random.default_rng(0))[0]
        assert out.shape == im.shape and out.dtype == np.uint8
        assert not np.array_equal(out, im)
    # Colour order survives (RGB in, RGB out): a red patch stays red.
    red = np.zeros((32, 32, 3), np.uint8)
    red[..., 0] = 220
    out = ExtraPhotometric(jpeg_quality=(90, 90), jpeg_p=1.0).degrade(
        [red], np.random.default_rng(0)
    )[0]
    assert out[..., 0].mean() > 200 and out[..., 2].mean() < 20


def test_zoom_out_moves_boxes_with_pixels():
    img = torch.zeros(1, 3, 100, 200, dtype=torch.uint8)
    img[..., 20:40, 50:90] = 255
    boxes = torch.tensor([[50.0, 20.0, 90.0, 40.0]])
    out, b = zoom_out(img, boxes, 0.5, (1.0, 0.0))
    assert out.shape == img.shape
    x1, y1, x2, y2 = b[0].round().int().tolist()
    assert (x1, y1, x2, y2) == (25, 60, 45, 70)
    inside = out[0, 0, y1 + 1 : y2 - 1, x1 + 1 : x2 - 1].float().mean()
    assert inside > 200
    assert out[0, 0, :50].sum() == 0  # top half is canvas


def test_detr_transform_zoom_out_p():
    DetrAugmentationConfig(zoom_out_p=0.5, zoom_out_min=0.3, zoom_out_max=1.0)
    torch.manual_seed(0)
    img = torch.full((1, 3, 120, 160), 255, dtype=torch.uint8)
    boxes = torch.tensor([[10.0, 10.0, 150.0, 110.0]])
    sizes = []
    for _ in range(20):
        out, b = detr_train_transform(
            img,
            boxes,
            (96, 128),
            "letterbox",
            crop_p=0.0,
            flip_p=0.0,
            zoom_out_p=1.0,
            zoom_out_range=(0.3, 1.0),
        )
        assert out.shape[-2:] == (96, 128)
        sizes.append(float((b[0, 2] - b[0, 0]) / (140 * 128 / 160)))
    assert min(sizes) >= 0.29 and max(sizes) <= 1.01 and np.std(sizes) > 0.05


@pytest.mark.parametrize("bad", [{"gamma_min": 0.0}, {"downup_min": 0.5}])
def test_validators(bad):
    with pytest.raises(Exception):
        IntensityConfig(**bad)
