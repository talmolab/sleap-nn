"""Appearance augmentations for unseen recording setups (E19, mouse generalist).

Off by default. Configured through new keys of ``augmentation_config.intensity``
(:class:`~sleap_nn.config.data_config.IntensityConfig`) and applied by
:class:`~sleap_nn.data.topdown_warp.WarpAug` (pose crops) and, with
``intensity.frame_photometric: true``, by
:class:`~sleap_nn.data.custom_datasets.DetrBBoxDataset` (whole detector frames).

Each op acts on a uint8 ``(H, W, C)`` image (``C`` = 1 or 3) and draws only from
the generator it is given:

* ``gray_p``: replace the colour image by its luminance, replicated to 3 channels
  (colour cameras vs IR/gray ones).
* ``channel_shuffle_p``: random permutation of the 3 channels.
* ``channel_gain_p``: multiply each channel by ``U(1 - g, 1 + g)``,
  ``g = channel_gain_max`` (colour cast: red light, tinted IR, white balance).
* ``gamma_p``: ``255 * (x / 255) ** gamma``, ``log(gamma) ~ U(log min, log max)``
  (dim or overexposed cages).
* ``invert_p``: ``255 - x`` (a light mouse on a dark floor looks like a dark
  mouse on a light floor).
* ``downup_p``: downsample by ``f ~ U(downup_min, downup_max)`` (area) and resize
  back (bilinear): a mouse filmed with few pixels, upsampled into the crop.
* ``jpeg_p``: JPEG re-encode at quality ``U[jpeg_quality_min, jpeg_quality_max]``.

Order: colour (gray, shuffle, gain), tone (gamma, invert; one lookup table),
then resolution (down-up), then compression (JPEG).
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ExtraPhotometric:
    """The E19 appearance ops (see the module docstring)."""

    gray_p: float = 0.0
    channel_shuffle_p: float = 0.0
    channel_gain_max: float = 0.0
    channel_gain_p: float = 0.0
    gamma: Tuple[float, float] = (1.0, 1.0)
    gamma_p: float = 0.0
    invert_p: float = 0.0
    downup: Tuple[float, float] = (1.0, 1.0)
    downup_p: float = 0.0
    jpeg_quality: Tuple[int, int] = (95, 95)
    jpeg_p: float = 0.0

    @classmethod
    def from_intensity(cls, intensity) -> "ExtraPhotometric":
        """From an ``augmentation_config.intensity`` section (dict-like) or None."""
        e = cls()
        if intensity is None:
            return e
        i = dict(intensity)

        def f(key, default):
            v = i.get(key)
            return default if v is None else v

        e.gray_p = float(f("gray_p", 0.0))
        e.channel_shuffle_p = float(f("channel_shuffle_p", 0.0))
        e.channel_gain_max = float(f("channel_gain_max", 0.0))
        e.channel_gain_p = float(f("channel_gain_p", 0.0))
        e.gamma = (float(f("gamma_min", 1.0)), float(f("gamma_max", 1.0)))
        e.gamma_p = float(f("gamma_p", 0.0))
        e.invert_p = float(f("invert_p", 0.0))
        e.downup = (float(f("downup_min", 1.0)), float(f("downup_max", 1.0)))
        e.downup_p = float(f("downup_p", 0.0))
        e.jpeg_quality = (
            int(f("jpeg_quality_min", 95)),
            int(f("jpeg_quality_max", 95)),
        )
        e.jpeg_p = float(f("jpeg_p", 0.0))
        if min(e.gamma) <= 0:
            raise ValueError(f"gamma_min/gamma_max must be > 0, got {e.gamma}.")
        if min(e.downup) < 1.0:
            raise ValueError(f"downup_min/downup_max must be >= 1, got {e.downup}.")
        return e

    @property
    def active(self) -> bool:
        """Whether any op can fire."""
        return any(
            p > 0
            for p in (
                self.gray_p,
                self.channel_shuffle_p,
                self.channel_gain_p,
                self.gamma_p,
                self.invert_p,
                self.downup_p,
                self.jpeg_p,
            )
        )

    def color(self, imgs, rng: np.random.Generator):
        """Gray / channel shuffle / channel gain on 3-channel images (same draw)."""
        if imgs[0].shape[-1] != 3:
            return imgs
        if self.gray_p > 0 and rng.random() < self.gray_p:
            imgs = [
                np.repeat(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)[:, :, None], 3, axis=-1)
                for im in imgs
            ]
        if self.channel_shuffle_p > 0 and rng.random() < self.channel_shuffle_p:
            perm = rng.permutation(3)
            imgs = [np.ascontiguousarray(im[:, :, perm]) for im in imgs]
        if self.channel_gain_p > 0 and rng.random() < self.channel_gain_p:
            g = rng.uniform(1 - self.channel_gain_max, 1 + self.channel_gain_max, 3)
            x = np.arange(256, dtype=np.float32)
            luts = [np.clip(np.rint(x * gi), 0, 255).astype(np.uint8) for gi in g]
            imgs = [
                np.stack([cv2.LUT(im[:, :, c], luts[c]) for c in range(3)], axis=-1)
                for im in imgs
            ]
        return imgs

    def tone_lut(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        """Gamma then inversion as one uint8 lookup table (None: identity)."""
        lut = None
        x = np.arange(256, dtype=np.float32) / 255.0
        if self.gamma_p > 0 and rng.random() < self.gamma_p:
            lo, hi = np.log(self.gamma[0]), np.log(self.gamma[1])
            g = float(np.exp(rng.uniform(lo, hi)))
            x = x**g
            lut = True
        if self.invert_p > 0 and rng.random() < self.invert_p:
            x = 1.0 - x
            lut = True
        if lut is None:
            return None
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)

    def degrade(self, imgs, rng: np.random.Generator):
        """Down-up sampling and JPEG (same draws for every image)."""
        if self.downup_p > 0 and rng.random() < self.downup_p:
            fct = float(rng.uniform(*self.downup))
            out = []
            for im in imgs:
                h, w = im.shape[:2]
                small = cv2.resize(
                    im,
                    (max(1, int(round(w / fct))), max(1, int(round(h / fct)))),
                    interpolation=cv2.INTER_AREA,
                )
                back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                out.append(back[:, :, None] if back.ndim == 2 else back)
            imgs = out
        if self.jpeg_p > 0 and rng.random() < self.jpeg_p:
            q = int(rng.integers(self.jpeg_quality[0], self.jpeg_quality[1] + 1))
            out = []
            for im in imgs:
                rgb = im.shape[-1] == 3
                enc = cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if rgb else im
                ok, buf = cv2.imencode(".jpg", enc, [cv2.IMWRITE_JPEG_QUALITY, q])
                dec = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if ok else enc
                if rgb:
                    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                out.append(dec[:, :, None] if dec.ndim == 2 else dec)
            imgs = out
        return imgs
