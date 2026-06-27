"""Tests for the embedding crop-pipeline knobs (P2 #7).

Covers ``background_fill`` (LightningModule burn-in fill), ``crop_centering`` (mask
center-of-mass vs bounding-box midpoint), and the RGB/grayscale channel option
(config resolution + inference-layer coercion + auto-sync decoupling).
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sleap_nn.training.lightning_modules import (
    EmbeddingLightningModule,
    set_embedding_burn_in_from_config,
)

_DIM = 16
_MAX_STRIDE = 16


def _make_embedding_module(in_channels=1, objective_overrides=None):
    backbone = OmegaConf.create(
        {
            "unet": {
                "in_channels": in_channels,
                "kernel_size": 3,
                "filters": 8,
                "filters_rate": 1.5,
                "max_stride": _MAX_STRIDE,
                "stem_stride": None,
                "middle_block": True,
                "up_interpolate": True,
                "stacks": 1,
                "convs_per_block": 2,
                "output_stride": 2,
            }
        }
    )
    objective = {
        "positives": {"scope": "global_id"},
        "negatives": {
            "sources": ["in_batch"],
            "exclude_same_track": True,
            "restrict_same_video": False,
        },
        "loss": {"name": "supcon", "temperature": 0.1},
        "sampler": {"kind": "pk", "groups_per_batch": 2, "samples_per_group": 4},
        "use_projection": True,
        "projection_dim": _DIM,
    }
    if objective_overrides:
        objective.update(objective_overrides)
    heads = OmegaConf.create(
        {
            "embedding": {
                "embedding": {
                    "embedding_dim": _DIM,
                    "num_fc_layers": 1,
                    "num_fc_units": 32,
                    "pool": "gem",
                    "normalize": True,
                    "output_stride": _MAX_STRIDE,
                    "loss_weight": 1.0,
                    "freeze_backbone": False,
                    "objective": objective,
                }
            }
        }
    )
    return EmbeddingLightningModule(
        model_type="embedding",
        backbone_type="unet",
        backbone_config=backbone,
        head_configs=heads,
        init_weights="xavier",
    ).eval()


# ----------------------------------------------------------------------------
# background_fill
# ----------------------------------------------------------------------------


class TestBackgroundFill:
    """The masked-out background fill in EmbeddingLightningModule._standardize."""

    def _half_mask_inputs(self):
        gray = torch.rand(2, 1, 8, 8) * 255.0
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, :, :4] = 1.0  # left half is foreground
        return gray, mask

    def test_black_is_zeroed_background(self):
        """black == the original mask-multiply (background exactly 0)."""
        m = _make_embedding_module()
        m.burn_in = True
        m.standardize = True
        m.background_fill = "black"
        gray, mask = self._half_mask_inputs()
        out = m._standardize(gray, mask)
        bg = out[mask < 0.5]
        assert torch.allclose(bg, torch.zeros_like(bg), atol=1e-6)

    def test_mean_matches_black_for_standardized(self):
        """mean == black in standardized space (foreground mean is 0)."""
        m = _make_embedding_module()
        m.burn_in = True
        m.standardize = True
        gray, mask = self._half_mask_inputs()
        m.background_fill = "black"
        black = m._standardize(gray, mask)
        m.background_fill = "mean"
        mean = m._standardize(gray, mask)
        assert torch.allclose(black, mean, atol=1e-6)

    def test_grey_is_nonzero_constant_background(self):
        """grey fills the background with a per-crop constant != 0."""
        m = _make_embedding_module()
        m.burn_in = True
        m.standardize = True
        m.background_fill = "grey"
        gray, mask = self._half_mask_inputs()
        out = m._standardize(gray, mask)
        # Foreground unchanged vs black.
        m.background_fill = "black"
        black = m._standardize(gray, mask)
        fg = mask > 0.5
        assert torch.allclose(out[fg], black[fg], atol=1e-6)
        # Background is a single (per-crop) constant, not zero.
        for i in range(out.shape[0]):
            bg_i = out[i][mask[i] < 0.5]
            assert not torch.allclose(bg_i, torch.zeros_like(bg_i))
            assert torch.allclose(bg_i, bg_i[0].expand_as(bg_i), atol=1e-5)

    def test_noise_is_random_background(self):
        """noise fills the background with per-pixel (non-constant) noise (train mode)."""
        torch.manual_seed(0)
        m = _make_embedding_module()
        m.train()  # noise is a train-only augmentation (neutral at eval)
        m.burn_in = True
        m.standardize = True
        m.background_fill = "noise"
        gray, mask = self._half_mask_inputs()
        out = m._standardize(gray, mask)
        bg = out[0][mask[0] < 0.5]
        assert bg.std() > 1e-3  # genuinely varying, not a constant

    def test_build_input_nonstandardized_fills(self):
        """The non-standardized (/255) path honors the fill too."""
        m = _make_embedding_module()
        m.burn_in = True
        m.standardize = False
        gray, mask = self._half_mask_inputs()
        m.background_fill = "grey"
        out = m._build_input(gray, mask)
        bg = out[mask < 0.5]
        assert torch.allclose(bg, torch.full_like(bg, 0.5), atol=1e-6)

    def test_noise_is_deterministic_at_eval(self):
        """noise is a train-only augmentation: at eval it falls back to neutral 0.

        Keeps the validation retrieval metric (and thus checkpoint selection)
        deterministic.
        """
        m = _make_embedding_module()
        m.burn_in = True
        m.standardize = True
        m.background_fill = "noise"
        gray, mask = self._half_mask_inputs()

        m.eval()
        out_a = m._standardize(gray, mask)
        out_b = m._standardize(gray, mask)
        # Deterministic at eval, and the background is the neutral 0 fill (== black).
        assert torch.allclose(out_a, out_b, atol=1e-6)
        bg = out_a[mask < 0.5]
        assert torch.allclose(bg, torch.zeros_like(bg), atol=1e-6)

        # During training the same call injects (varying) noise into the background.
        m.train()
        bg_train = m._standardize(gray, mask)[mask < 0.5]
        assert bg_train.std() > 1e-3


class TestSetBurnInFromConfig:
    """set_embedding_burn_in_from_config reads + validates background_fill."""

    def test_reads_background_fill(self):
        m = _make_embedding_module()
        cfg = OmegaConf.create(
            {
                "data_config": {
                    "preprocessing": {"burn_in": True, "background_fill": "noise"}
                }
            }
        )
        set_embedding_burn_in_from_config(m, cfg)
        assert m.burn_in is True
        assert m.background_fill == "noise"

    def test_defaults_to_black(self):
        m = _make_embedding_module()
        cfg = OmegaConf.create({"data_config": {"preprocessing": {"burn_in": True}}})
        set_embedding_burn_in_from_config(m, cfg)
        assert m.background_fill == "black"

    def test_rejects_unknown_fill(self):
        m = _make_embedding_module()
        cfg = OmegaConf.create(
            {"data_config": {"preprocessing": {"background_fill": "rainbow"}}}
        )
        with pytest.raises(ValueError, match="background_fill"):
            set_embedding_burn_in_from_config(m, cfg)


# ----------------------------------------------------------------------------
# crop_centering
# ----------------------------------------------------------------------------


class TestCropCentering:
    """Mask-mode crop centering: mask-COM vs bbox-midpoint."""

    def test_bbox_midpoint(self):
        from sleap_nn.data.custom_datasets import _mask_bbox_midpoint

        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 4:8] = True  # rows 2..5, cols 4..7
        cx, cy = _mask_bbox_midpoint(mask)
        assert cx == (4 + 7) / 2.0
        assert cy == (2 + 5) / 2.0

    def test_bbox_midpoint_differs_from_com_for_concave(self):
        """For an L-shaped mask the bbox midpoint != center-of-mass."""
        from sleap_nn.data.custom_datasets import _mask_bbox_midpoint
        from sleap_nn.data.segmentation_maps import _compute_mask_centroids

        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:6] = True  # tall left bar
        mask[14:18, 2:18] = True  # bottom bar -> L shape
        bx, by = _mask_bbox_midpoint(mask)
        cx, cy = _compute_mask_centroids([mask])[0]
        assert abs(bx - cx) + abs(by - cy) > 1.0

    def test_empty_mask_falls_back_to_center(self):
        from sleap_nn.data.custom_datasets import _mask_bbox_midpoint

        mask = np.zeros((8, 12), dtype=bool)
        cx, cy = _mask_bbox_midpoint(mask)
        assert (cx, cy) == (6.0, 4.0)

    def test_dataset_rejects_unknown_centering(self):
        from sleap_nn.data.custom_datasets import EmbeddingDataset

        # The centering value is validated early in __init__ (before any label work),
        # so empty labels are fine — we only need to reach the validation.
        with pytest.raises(ValueError, match="crop_centering"):
            EmbeddingDataset(
                labels=[],
                crop_size=32,
                class_names=[],
                embedding_head_config=OmegaConf.create({}),
                max_stride=16,
                crop_centering="diagonal",
            )


# ----------------------------------------------------------------------------
# RGB / grayscale channel option
# ----------------------------------------------------------------------------


class TestEmbeddingChannels:
    """The config-driven RGB/grayscale option and its inference parity."""

    def test_resolve_channels_default_grayscale(self):
        from sleap_nn.inference.loaders import _resolve_embedding_channels

        cfg = OmegaConf.create({"data_config": {"preprocessing": {}}})
        ensure_rgb, ensure_grayscale = _resolve_embedding_channels(cfg)
        assert ensure_rgb is False
        assert ensure_grayscale is True

    def test_resolve_channels_rgb_optin(self):
        from sleap_nn.inference.loaders import _resolve_embedding_channels

        cfg = OmegaConf.create({"data_config": {"preprocessing": {"ensure_rgb": True}}})
        ensure_rgb, ensure_grayscale = _resolve_embedding_channels(cfg)
        assert ensure_rgb is True
        assert ensure_grayscale is False

    def test_layer_coerces_to_grayscale(self):
        from sleap_nn.inference.layers.backends.torch_backend import TorchBackend
        from sleap_nn.inference.layers.configs import (
            PostprocessConfig,
            PreprocessConfig,
        )
        from sleap_nn.inference.layers.embedding import EmbeddingLayer

        module = _make_embedding_module(in_channels=1)
        layer = EmbeddingLayer(
            backend=TorchBackend(model=module.model, device="cpu"),
            embedding_module=module,
            embedding_dim=_DIM,
            output_stride=_MAX_STRIDE,
            max_stride=_MAX_STRIDE,
            input_channels=1,
            preprocess_config=PreprocessConfig(scale=1.0),
            postprocess_config=PostprocessConfig(),
        )
        # A 3-channel crop is coerced to grayscale and embeds without error.
        crops = torch.rand(2, 3, 32, 32) * 255.0
        out = layer.predict(crops)
        assert out.pred_embeddings.shape == (2, 1, _DIM)


# ----------------------------------------------------------------------------
# Loss-parameter validation
# ----------------------------------------------------------------------------


class TestLossParamValidation:
    """temperature / margin are validated at construction."""

    def test_nonpositive_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            _make_embedding_module(
                objective_overrides={"loss": {"name": "supcon", "temperature": 0.0}}
            )

    def test_negative_margin_raises(self):
        with pytest.raises(ValueError, match="margin"):
            _make_embedding_module(
                objective_overrides={"loss": {"name": "triplet", "margin": -0.1}}
            )

    def test_valid_params_ok(self):
        _make_embedding_module(
            objective_overrides={"loss": {"name": "supcon", "temperature": 0.05}}
        )  # no raise


# ----------------------------------------------------------------------------
# GeM exponent clamp (no NaN from a degenerate learnable p)
# ----------------------------------------------------------------------------


class TestGeMExponentClamp:
    """The GeM exponent is floored so a degenerate p cannot yield NaN/inf."""

    def test_negative_p_does_not_nan(self):
        from sleap_nn.architectures.heads import GeM

        gem = GeM(learnable=True)
        with torch.no_grad():
            gem.p.fill_(-2.0)  # degenerate: would invert/explode without the clamp
        out = gem(torch.rand(2, 4, 8, 8) * 5.0)
        assert torch.isfinite(out).all()
        assert out.shape == (2, 4)
