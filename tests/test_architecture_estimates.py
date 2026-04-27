"""Tests for sleap_nn.config_generator.architecture_estimates.

Golden values match the web app config picker
(``docs/configuration/config-picker/app.html``) and the canonical receptive
field formula at ``example_notebooks/receptive_field_guide.py``.
"""

import pytest

from sleap_nn.config_generator.architecture_estimates import (
    compute_augmentation_padding,
    compute_max_stride_for_animal_size,
    compute_pad_to_stride,
    compute_receptive_field,
    compute_suggested_crop_size,
    decoder_blocks,
    encoder_blocks,
    estimate_unet_params,
    recommend_default_max_stride,
)


class TestReceptiveField:
    """Receptive-field formula matches the canonical notebook + web app."""

    @pytest.mark.parametrize(
        "max_stride,expected_rf",
        [(8, 36), (16, 76), (32, 156), (64, 316), (128, 636)],
    )
    def test_canonical_values(self, max_stride, expected_rf):
        """Match the values pre-computed in the web app's RF_TABLE."""
        assert compute_receptive_field(max_stride) == expected_rf

    def test_minimum_stride(self):
        """A single down block (max_stride=2) gives RF=6."""
        # convs(1,1) + pool(2): kernels [3,3,2], strides [1,1,2]
        # rf = 1 + (3-1)*1 + (3-1)*1 + (2-1)*1 = 6
        assert compute_receptive_field(2) == 6

    def test_invalid_max_stride(self):
        """Non-power-of-2 and zero raise ValueError."""
        with pytest.raises(ValueError):
            compute_receptive_field(15)
        with pytest.raises(ValueError):
            compute_receptive_field(0)


class TestEncoderDecoderBlocks:
    """Encoder/decoder block counts match log2 expectations."""

    @pytest.mark.parametrize(
        "max_stride,expected", [(8, 3), (16, 4), (32, 5), (64, 6), (128, 7)]
    )
    def test_encoder(self, max_stride, expected):
        """log2(max_stride) gives the encoder depth."""
        assert encoder_blocks(max_stride) == expected

    @pytest.mark.parametrize(
        "max_stride,output_stride,expected",
        [(16, 1, 4), (16, 2, 3), (32, 4, 3), (64, 1, 6), (32, 1, 5)],
    )
    def test_decoder(self, max_stride, output_stride, expected):
        """Decoder depth scales with max_stride/output_stride."""
        assert decoder_blocks(max_stride, output_stride) == expected

    def test_decoder_zero_output_stride(self):
        """output_stride=0 falls back to encoder_blocks."""
        assert decoder_blocks(16, 0) == encoder_blocks(16)


class TestMaxStrideForAnimalSize:
    """Selecting the smallest max_stride whose RF covers an animal."""

    @pytest.mark.parametrize(
        "animal_size,expected",
        [
            (30, 8),  # RF 36 covers 30
            (50, 16),  # RF 76 covers 50, 36 doesn't
            (100, 32),  # RF 156 covers 100, 76 doesn't
            (200, 64),  # RF 316 covers 200, 156 doesn't
            (400, 128),  # RF 636 covers 400, 316 doesn't
            (700, 128),  # RF 636 doesn't cover; fallback to largest = 128
        ],
    )
    def test_picks_smallest_covering_stride(self, animal_size, expected):
        """For each size, picks the smallest stride whose RF covers it."""
        assert compute_max_stride_for_animal_size(animal_size) == expected

    def test_huge_animal_falls_back_to_largest(self):
        """If no stride's RF covers, return the largest candidate."""
        assert compute_max_stride_for_animal_size(10_000) == 128

    def test_custom_candidates(self):
        """Caller can constrain the candidate set."""
        assert compute_max_stride_for_animal_size(50, candidates=(16, 32)) == 16
        assert compute_max_stride_for_animal_size(500, candidates=(16, 32)) == 32


class TestEstimateUnetParams:
    """Parameter-count estimator includes middle block + uses output_stride."""

    def test_includes_middle_block(self):
        """Bottleneck block puts params in the right ballpark.

        Typical config (filters=32, max_stride=16) lands at 1–2M params.
        """
        params = estimate_unet_params(
            filters=32,
            max_stride=16,
            output_stride=1,
            in_channels=1,
            num_keypoints=24,
            filters_rate=1.5,
        )
        assert 1_000_000 < params < 2_000_000

    def test_grows_with_max_stride(self):
        """Deeper encoder = more params."""
        small = estimate_unet_params(32, 8, 1, 1, 24, 1.5)
        big = estimate_unet_params(32, 32, 1, 1, 24, 1.5)
        assert big > small * 2

    def test_output_stride_affects_decoder_depth(self):
        """Lower output_stride means more decoder blocks => more params."""
        os1 = estimate_unet_params(32, 16, 1, 1, 24, 1.5)
        os4 = estimate_unet_params(32, 16, 4, 1, 24, 1.5)
        assert os1 > os4

    def test_in_channels_affects_first_layer(self):
        """RGB input adds (3-1)*32*9 = 576 params in the first conv."""
        gray = estimate_unet_params(32, 16, 1, 1, 24, 1.5)
        rgb = estimate_unet_params(32, 16, 1, 3, 24, 1.5)
        assert rgb > gray
        assert rgb - gray == 576


class TestAugmentationPadding:
    """Padding formula for rotation/scale augmentations."""

    def test_no_aug(self):
        """No rotation, no scale -> zero padding."""
        assert compute_augmentation_padding(100, 0, 1.0) == 0

    def test_45_degree_is_sqrt2(self):
        """At 45° rotation, expansion factor is sqrt(2)."""
        # 100*sqrt(2) - 100 ~= 41.42 -> ceil 42
        assert compute_augmentation_padding(100, 45, 1.0) == 42

    def test_90_degree_is_sqrt2_too(self):
        """Web app caps rotation factor at sqrt(2) for >=45°."""
        assert compute_augmentation_padding(100, 90, 1.0) == 42

    def test_scale_only(self):
        """scale=1.2 expands bbox by 20% (rotation_factor stays at 1.0)."""
        assert compute_augmentation_padding(100, 0, 1.2) == 20

    def test_combined(self):
        """Combined rotation + scale multiplies the expansions."""
        # rot 45° (sqrt(2)) * scale 1.2 = ~1.697; pad = ceil(69.7) = 70
        assert compute_augmentation_padding(100, 45, 1.2) == 70

    def test_small_rotation(self):
        """At 30°, factor = cos+sin ~= 1.366."""
        assert compute_augmentation_padding(100, 30, 1.0) == 37


class TestSuggestedCropSize:
    """Crop-size rounding and augmentation-aware padding behavior."""

    def test_basic_round_to_stride(self):
        """No aug, no padding: round bbox up to multiple of max_stride."""
        assert compute_suggested_crop_size(90, max_stride=16) == 96
        assert compute_suggested_crop_size(96, max_stride=16) == 96
        assert compute_suggested_crop_size(100, max_stride=32) == 128

    def test_user_padding_overrides(self):
        """Explicit user_padding wins over augmentation-derived padding."""
        # bbox 100 + padding 10 = 110, round to 16 = 112
        assert (
            compute_suggested_crop_size(
                100, max_stride=16, use_augmentation=True, user_padding=10
            )
            == 112
        )

    def test_user_padding_zero_means_zero(self):
        """user_padding=0 is honored, not treated as 'falsy/use auto'."""
        assert (
            compute_suggested_crop_size(
                100,
                max_stride=16,
                use_augmentation=True,
                user_padding=0,
                rotation_max=45,
            )
            == 112
        )

    def test_aug_padding_when_enabled(self):
        """With use_augmentation and rotation, padding is added."""
        # bbox 100, rot 45° -> pad 42; total 142 -> round to 144 (16*9)
        assert (
            compute_suggested_crop_size(
                100, max_stride=16, use_augmentation=True, rotation_max=45
            )
            == 144
        )


class TestRecommendDefaultMaxStride:
    """Web-app bucket recommendation for default max_stride."""

    @pytest.mark.parametrize(
        "avg_size,scale,expected",
        [
            (30, 1.0, 8),  # < 40 → 8
            (39.9, 1.0, 8),  # boundary
            (40, 1.0, 16),  # = 40 (not < 40) → 16
            (60, 1.0, 16),  # 40 ≤ 60 ≤ 100 → 16
            (100, 1.0, 16),  # = 100 (not > 100) → 16
            (100.1, 1.0, 32),  # > 100 → 32
            (200, 1.0, 32),  # > 100 → 32
            (200, 0.5, 16),  # 200*0.5=100 → 16
            (50, 0.5, 8),  # 50*0.5=25 → 8
            (300, 0.5, 32),  # 300*0.5=150 → 32
        ],
    )
    def test_buckets_match_web_app(self, avg_size, scale, expected):
        """Bucket boundaries are <40, 40–100, >100 (web-app parity)."""
        assert recommend_default_max_stride(avg_size, scale) == expected


class TestPadToStride:
    """Padding image dimensions up to the next multiple of max_stride."""

    def test_multiples_unchanged(self):
        """Already-divisible dims pass through unchanged."""
        assert compute_pad_to_stride(64, 128, 16) == (64, 128)

    def test_rounds_up(self):
        """Non-multiples round up to the next multiple."""
        assert compute_pad_to_stride(50, 100, 16) == (64, 112)

    def test_stride_32(self):
        """Stride 32 rounds to 32-multiples."""
        assert compute_pad_to_stride(50, 100, 32) == (64, 128)
