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


# ----------------------------------------------------------------------------
# aug_views validation (no longer a silent no-op)
# ----------------------------------------------------------------------------


class TestAugViewsValidation:
    """positives.aug_views must be 2 (the two-view contrastive setup) or raise."""

    def test_aug_views_not_two_raises(self):
        with pytest.raises(ValueError, match="aug_views"):
            _make_embedding_module(
                objective_overrides={
                    "positives": {"scope": "global_id", "aug_views": 3}
                }
            )

    def test_aug_views_two_ok(self):
        _make_embedding_module(
            objective_overrides={"positives": {"scope": "global_id", "aug_views": 2}}
        )  # no raise


# ----------------------------------------------------------------------------
# Per-channel standardize (RGB) + mask-aware forward
# ----------------------------------------------------------------------------


class TestStandardizePerChannel:
    """`_standardize` reduces per-channel: RGB channels are independently normalized."""

    def test_rgb_each_channel_zero_mean_unit_std(self):
        module = _make_embedding_module(in_channels=3)
        # Channels at very different intensities: a cross-channel ("whole-tensor") mean
        # would leave large per-channel offsets; a true per-channel standardize zeroes
        # each channel's mean and unit-scales its std.
        gray = torch.empty(2, 3, 8, 8)
        gray[:, 0] = torch.rand(2, 8, 8) * 5 + 10
        gray[:, 1] = torch.rand(2, 8, 8) * 5 + 100
        gray[:, 2] = torch.rand(2, 8, 8) * 5 + 200
        g = module._standardize(gray, torch.ones_like(gray[:, :1]))
        per_ch_mean = g.mean(dim=(2, 3))  # (B, 3)
        per_ch_std = g.std(dim=(2, 3), unbiased=False)  # (B, 3)
        assert torch.allclose(per_ch_mean, torch.zeros_like(per_ch_mean), atol=1e-4)
        assert torch.allclose(per_ch_std, torch.ones_like(per_ch_std), atol=1e-2)

    def test_grayscale_zero_mean(self):
        # For C=1 the per-channel reduction equals the old whole-tensor reduction.
        module = _make_embedding_module(in_channels=1)
        gray = torch.rand(2, 1, 8, 8) * 255.0
        g = module._standardize(gray, torch.ones_like(gray[:, :1]))
        assert torch.allclose(g.mean(dim=(2, 3)), torch.zeros(2, 1), atol=1e-4)


class TestForwardMaskAware:
    """`forward` uses the instance mask when provided (burn-in parity with training)."""

    def test_forward_uses_mask_when_provided(self):
        module = _make_embedding_module(in_channels=1)
        module.burn_in = True
        module.background_fill = "black"
        # Crops as the dataset stores them: (B, 1, C, H, W).
        img = torch.rand(2, 1, 1, 16, 16) * 255.0
        mask = torch.zeros(2, 1, 1, 16, 16)
        mask[..., :8, :] = 1.0  # half foreground
        with torch.no_grad():
            e_nomask = module(img)  # mask=None -> whole-crop standardize
            e_mask = module(img, mask)  # masked, foreground-only standardize + fill
        # The mask changes the standardization statistics, so the embeddings differ.
        assert not torch.allclose(e_nomask, e_mask, atol=1e-4)


# ----------------------------------------------------------------------------
# GroupAwareBatchSampler DDP sharding
# ----------------------------------------------------------------------------


class TestSamplerDDPSharding:
    """Under DDP each rank must draw a distinct batch stream (no wasted recompute)."""

    def _make(self, rank=0, world_size=1, seed=0):
        from sleap_nn.data.custom_datasets import GroupAwareBatchSampler

        g = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        v = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        f = np.arange(12)
        return GroupAwareBatchSampler(
            g,
            v,
            f,
            kind="pk",
            P=2,
            K=2,
            batches_per_epoch=6,
            seed=seed,
            rank=rank,
            world_size=world_size,
        )

    def test_different_ranks_draw_different_batches(self):
        b0 = list(self._make(rank=0, world_size=2))
        b1 = list(self._make(rank=1, world_size=2))
        assert b0 != b1  # disjoint streams -> the all-reduced gradient covers both

    def test_rank0_is_independent_of_world_size(self):
        # rank 0 reproduces the single-GPU stream regardless of world_size (a no-op
        # for single-GPU training, where rank=0/world_size=1).
        assert list(self._make(rank=0, world_size=1)) == list(
            self._make(rank=0, world_size=4)
        )

    def test_same_rank_and_seed_is_reproducible(self):
        assert list(self._make(rank=1, world_size=2, seed=7)) == list(
            self._make(rank=1, world_size=2, seed=7)
        )


class TestSamplerComposition:
    """The sampler's core invariant: PK batches = P distinct groups x K samples each."""

    _G = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    _V = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    _F = np.arange(12)

    def _make(self, kind="pk", P=2, K=2):
        from sleap_nn.data.custom_datasets import GroupAwareBatchSampler

        return GroupAwareBatchSampler(
            self._G,
            self._V,
            self._F,
            kind=kind,
            P=P,
            K=K,
            batches_per_epoch=8,
            seed=0,
        )

    def test_pk_batch_has_P_groups_K_each(self):
        sampler = self._make(kind="pk", P=2, K=2)
        for batch in sampler:
            assert len(batch) == 2 * 2
            groups = self._G[batch]
            uniq, counts = np.unique(groups, return_counts=True)
            assert len(uniq) == 2  # P distinct groups
            assert (counts == 2).all()  # K samples per group

    def test_within_video_batch_is_single_video(self):
        sampler = self._make(kind="within_video", P=2, K=2)
        for batch in sampler:
            assert len(np.unique(self._V[batch])) == 1  # all from one video

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            list(self._make(kind="diagonal"))


# ----------------------------------------------------------------------------
# Track/Identity-based positive/negative grouping (sleap-io #535)
# ----------------------------------------------------------------------------


class TestIdentitySampling:
    """Positives group on the real `sio.Identity` (global animal), else `sio.Track`."""

    @staticmethod
    def _skel():
        import sleap_io as sio

        return sio.Skeleton(["a", "b"])

    def _inst(self, x, *, identity=None, track=None):
        import sleap_io as sio

        return sio.Instance.from_numpy(
            np.array([[x, x], [x + 1, x + 1]], dtype=float),
            skeleton=self._skel(),
            track=track,
            identity=identity,
        )

    def _labels_two_videos_same_animals(self):
        """Same two animals in two videos, with DIFFERENT per-video track names."""
        import sleap_io as sio

        idA, idB = sio.Identity(name="mouseA"), sio.Identity(name="mouseB")
        v1, v2 = sio.Video("v1.mp4"), sio.Video("v2.mp4")
        lf1 = sio.LabeledFrame(
            video=v1,
            frame_idx=0,
            instances=[
                self._inst(0, identity=idA, track=sio.Track("v1_t0")),
                self._inst(5, identity=idB, track=sio.Track("v1_t1")),
            ],
        )
        lf2 = sio.LabeledFrame(
            video=v2,
            frame_idx=0,
            instances=[
                self._inst(0, identity=idA, track=sio.Track("v2_t0")),
                self._inst(5, identity=idB, track=sio.Track("v2_t1")),
            ],
        )
        return sio.Labels([lf1, lf2])

    def test_global_label_prefers_identity_over_track(self):
        from sleap_nn.data.custom_datasets import _global_identity_label

        inst = self._inst(0, identity=None, track=None)
        import sleap_io as sio

        inst.identity = sio.Identity(name="mouseA")
        inst.track = sio.Track("v1_t0")
        # Identity wins even when track_names_are_global is False.
        assert _global_identity_label(inst, track_names_are_global=False) == "mouseA"

    def test_vocabulary_collapses_same_animal_across_videos(self):
        """4 per-video tracks but only 2 global identities -> 2-class vocabulary."""
        from sleap_nn.data.custom_datasets import resolve_embedding_class_names

        labels = self._labels_two_videos_same_animals()
        vocab = resolve_embedding_class_names(labels_iter := [labels])
        assert vocab == ["mouseA", "mouseB"]

    def test_global_id_group_is_identity_across_videos(self):
        """Same animal in two videos gets the SAME group_id under global_id scope."""
        from sleap_nn.data.custom_datasets import (
            EmbeddingDataset,
            resolve_embedding_class_names,
        )

        labels = self._labels_two_videos_same_animals()
        class_names = resolve_embedding_class_names([labels])
        ds = EmbeddingDataset.__new__(EmbeddingDataset)
        ds.class_names = list(class_names)
        ds.id_scope = "global_id"
        ds.track_names_are_global = False
        ds._tracklet_vocab = {}
        # mouseA in video 0 and video 1 -> same group_id (the identity index).
        g_v0 = ds._resolve_group(labels[0].instances[0], 0, 0)
        g_v1 = ds._resolve_group(labels[1].instances[0], 0, 1)
        assert g_v0 is not None and g_v1 is not None
        assert g_v0[0] == g_v1[0] == class_names.index("mouseA")

    def test_tracklet_group_is_per_video_track(self):
        """Under tracklet scope the same animal in two videos is DIFFERENT groups."""
        from sleap_nn.data.custom_datasets import (
            EmbeddingDataset,
            resolve_embedding_class_names,
        )

        labels = self._labels_two_videos_same_animals()
        class_names = resolve_embedding_class_names([labels])
        ds = EmbeddingDataset.__new__(EmbeddingDataset)
        ds.class_names = list(class_names)
        ds.id_scope = "tracklet"
        ds.track_names_are_global = False
        ds._tracklet_vocab = {}
        g_v0 = ds._resolve_group(labels[0].instances[0], 0, 0)
        g_v1 = ds._resolve_group(labels[1].instances[0], 0, 1)
        # Distinct per-video tracklet group_ids ...
        assert g_v0[0] != g_v1[0]
        # ... but the eval (global) grouping still ties them to the same identity.
        assert g_v0[1] == g_v1[1] == class_names.index("mouseA")

    def test_track_fallback_requires_promise(self):
        """A track-only detection is grouped only under track_names_are_global."""
        import sleap_io as sio
        from sleap_nn.data.custom_datasets import resolve_embedding_class_names

        v = sio.Video("v.mp4")
        lf = sio.LabeledFrame(
            video=v, frame_idx=0, instances=[self._inst(0, track=sio.Track("t0"))]
        )
        labels = sio.Labels([lf])
        assert resolve_embedding_class_names([labels], track_names_are_global=True) == [
            "t0"
        ]
        assert (
            resolve_embedding_class_names([labels], track_names_are_global=False) == []
        )
