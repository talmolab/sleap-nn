"""Tests for the Phase-0 tiling config cluster.

Covers:
  - ``sleap_nn.config.data_config.TilingConfig`` (attrs validators + nesting).
  - ``sleap_nn.config_generator.architecture_estimates`` tile-geometry helpers.
  - ``sleap_nn.config.utils.check_tiling`` / ``check_tiling_parity``.
  - ``sleap_nn.export.utils.warn_on_tiled_export``.
"""

import math

import pytest
from loguru import logger
from omegaconf import OmegaConf
from _pytest.logging import LogCaptureFixture

from sleap_nn.config.data_config import (
    DataConfig,
    PreprocessingConfig,
    TilingConfig,
)
from sleap_nn.config.utils import check_tiling, check_tiling_parity
from sleap_nn.config_generator.architecture_estimates import (
    _BACKBONE_CONTEXT_MARGIN_PX,
    compute_backbone_context_margin,
    compute_receptive_field,
    compute_suggested_tile_overlap,
    compute_suggested_tile_size,
)
from sleap_nn.export.utils import warn_on_tiled_export


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru logs into the pytest ``caplog`` fixture (repo shim)."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


# ---------------------------------------------------------------------------
# 1. TilingConfig
# ---------------------------------------------------------------------------
class TestTilingConfig:
    """Attrs-level validation, defaults and nesting of ``TilingConfig``."""

    def test_defaults(self):
        """Default construction matches the documented Phase-0 defaults."""
        t = TilingConfig()
        assert t.enabled is False
        assert t.tile_size is None
        assert t.overlap is None
        assert t.min_overlap_fraction == 0.25
        assert t.blend == "gaussian"
        assert t.sigma_scale == 0.125
        assert t.tile_batch_size is None
        assert t.accumulator_device == "auto"
        assert t.cpu_thresh == 0.40
        assert t.sampling == "foreground"
        assert t.tile_fg_fraction == 0.5
        assert t.samples_per_frame is None
        assert t.center_jitter == 0.5
        assert t.min_visible_keypoints == 1
        assert t.steps_per_epoch is None
        assert t.full_frame_pass is False

    def test_all_valid_values(self):
        """A fully-specified valid config constructs and round-trips its values."""
        t = TilingConfig(
            enabled=True,
            tile_size=64,
            overlap=0,
            min_overlap_fraction=1.0,
            blend="pyramid",
            sigma_scale=1.0,
            tile_batch_size=4,
            accumulator_device="cpu",
            cpu_thresh=0.0,
            sampling="grid",
            tile_fg_fraction=0.0,
            samples_per_frame=2,
            center_jitter=1.0,
            min_visible_keypoints=0,
            steps_per_epoch=5,
            full_frame_pass=True,
        )
        assert t.enabled is True
        assert t.tile_size == 64
        assert t.overlap == 0
        assert t.blend == "pyramid"
        assert t.sampling == "grid"
        assert t.min_visible_keypoints == 0

    @pytest.mark.parametrize("value", [1, 4096, None])
    def test_tile_size_valid(self, value):
        """tile_size accepts None or a positive int."""
        assert TilingConfig(tile_size=value).tile_size == value

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_tile_size_invalid(self, value, caplog):
        """tile_size rejects zero / negatives."""
        with pytest.raises(ValueError):
            TilingConfig(tile_size=value)
        assert "tile_size" in caplog.text

    @pytest.mark.parametrize("value", [0, 4, None])
    def test_overlap_valid(self, value):
        """Overlap accepts None or a non-negative int."""
        assert TilingConfig(overlap=value).overlap == value

    def test_overlap_invalid(self, caplog):
        """Overlap rejects negatives."""
        with pytest.raises(ValueError):
            TilingConfig(overlap=-1)
        assert "overlap" in caplog.text

    @pytest.mark.parametrize("field_name", ["tile_batch_size", "samples_per_frame"])
    @pytest.mark.parametrize("value", [0, -3])
    def test_optional_positive_int_invalid(self, field_name, value, caplog):
        """Other optional-positive-int fields reject non-positive values."""
        with pytest.raises(ValueError):
            TilingConfig(**{field_name: value})
        assert field_name in caplog.text

    def test_steps_per_epoch_valid(self):
        """steps_per_epoch accepts None or positive int."""
        assert TilingConfig(steps_per_epoch=None).steps_per_epoch is None
        assert TilingConfig(steps_per_epoch=10).steps_per_epoch == 10

    @pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
    def test_min_overlap_fraction_invalid(self, value, caplog):
        """min_overlap_fraction must be a proportion in [0, 1]."""
        with pytest.raises(ValueError):
            TilingConfig(min_overlap_fraction=value)
        assert "min_overlap_fraction" in caplog.text

    @pytest.mark.parametrize("value", [0.0, 0.25, 1.0])
    def test_min_overlap_fraction_valid(self, value):
        """min_overlap_fraction endpoints 0 and 1 are allowed."""
        assert TilingConfig(min_overlap_fraction=value).min_overlap_fraction == value

    @pytest.mark.parametrize("value", ["gaussian", "pyramid", "constant"])
    def test_blend_valid(self, value):
        """Blend accepts each supported window mode."""
        assert TilingConfig(blend=value).blend == value

    @pytest.mark.parametrize("value", ["bad", "linear", "", "Gaussian"])
    def test_blend_invalid(self, value, caplog):
        """Blend rejects unsupported window modes."""
        with pytest.raises(ValueError):
            TilingConfig(blend=value)
        assert "blend" in caplog.text

    @pytest.mark.parametrize("value", [0.001, 0.125, 1.0])
    def test_sigma_scale_valid(self, value):
        """sigma_scale must be in (0, 1]."""
        assert TilingConfig(sigma_scale=value).sigma_scale == value

    @pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
    def test_sigma_scale_invalid(self, value):
        """sigma_scale rejects 0 and values above 1."""
        with pytest.raises(ValueError):
            TilingConfig(sigma_scale=value)

    @pytest.mark.parametrize("value", ["auto", "cpu", "cuda"])
    def test_accumulator_device_valid(self, value):
        """accumulator_device accepts each supported placement."""
        assert TilingConfig(accumulator_device=value).accumulator_device == value

    @pytest.mark.parametrize("value", ["gpu", "tpu", "mps", ""])
    def test_accumulator_device_invalid(self, value, caplog):
        """accumulator_device rejects unsupported placements."""
        with pytest.raises(ValueError):
            TilingConfig(accumulator_device=value)
        assert "accumulator_device" in caplog.text

    @pytest.mark.parametrize("value", [-0.01, 1.01])
    def test_cpu_thresh_invalid(self, value, caplog):
        """cpu_thresh must be a proportion in [0, 1]."""
        with pytest.raises(ValueError):
            TilingConfig(cpu_thresh=value)
        assert "cpu_thresh" in caplog.text

    @pytest.mark.parametrize("value", ["foreground", "grid"])
    def test_sampling_valid(self, value):
        """Sampling accepts each supported strategy."""
        assert TilingConfig(sampling=value).sampling == value

    @pytest.mark.parametrize("value", ["fg", "random", ""])
    def test_sampling_invalid(self, value, caplog):
        """Sampling rejects unsupported strategies."""
        with pytest.raises(ValueError):
            TilingConfig(sampling=value)
        assert "sampling" in caplog.text

    @pytest.mark.parametrize("value", [0.0, 0.5, 0.999])
    def test_tile_fg_fraction_valid(self, value):
        """tile_fg_fraction must be in [0, 1) (never 1.0)."""
        assert TilingConfig(tile_fg_fraction=value).tile_fg_fraction == value

    @pytest.mark.parametrize("value", [1.0, 1.5, -0.1])
    def test_tile_fg_fraction_invalid(self, value, caplog):
        """tile_fg_fraction rejects 1.0 (and out-of-range values)."""
        with pytest.raises(ValueError):
            TilingConfig(tile_fg_fraction=value)
        assert "tile_fg_fraction" in caplog.text

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_center_jitter_invalid(self, value, caplog):
        """center_jitter must be a proportion in [0, 1]."""
        with pytest.raises(ValueError):
            TilingConfig(center_jitter=value)
        assert "center_jitter" in caplog.text

    def test_min_visible_keypoints_valid(self):
        """min_visible_keypoints accepts 0 and positive ints."""
        assert TilingConfig(min_visible_keypoints=0).min_visible_keypoints == 0
        assert TilingConfig(min_visible_keypoints=3).min_visible_keypoints == 3

    def test_min_visible_keypoints_invalid(self):
        """min_visible_keypoints rejects negatives."""
        with pytest.raises(ValueError):
            TilingConfig(min_visible_keypoints=-1)

    def test_nested_on_preprocessing_default(self):
        """PreprocessingConfig carries a factory-default (disabled) TilingConfig."""
        pc = PreprocessingConfig()
        assert isinstance(pc.tiling, TilingConfig)
        assert pc.tiling.enabled is False

    def test_nested_factory_is_unique_per_instance(self):
        """Each PreprocessingConfig gets its own TilingConfig (factory default)."""
        a = PreprocessingConfig()
        b = PreprocessingConfig()
        assert a.tiling is not b.tiling

    def test_structured_dataconfig_exposes_tiling(self):
        """OmegaConf.structured(DataConfig()) exposes preprocessing.tiling.enabled."""
        cfg = OmegaConf.structured(DataConfig())
        assert cfg.preprocessing.tiling.enabled is False
        assert cfg.preprocessing.tiling.min_overlap_fraction == 0.25

    def test_structured_merge_rejects_unknown_key(self):
        """A structured DataConfig rejects an unknown nested tiling key."""
        cfg = OmegaConf.structured(DataConfig())
        with pytest.raises(Exception):
            OmegaConf.merge(
                cfg, OmegaConf.create({"preprocessing": {"tiling": {"nope": 1}}})
            )


# ---------------------------------------------------------------------------
# 2. architecture_estimates: backbone margin + tile sizers
# ---------------------------------------------------------------------------
class TestComputeBackboneContextMargin:
    """``compute_backbone_context_margin`` per-backbone behavior."""

    @pytest.mark.parametrize("max_stride", [8, 16, 32])
    def test_unet_is_half_receptive_field(self, max_stride):
        """UNet margin == ceil(receptive_field / 2)."""
        rf = compute_receptive_field(max_stride)
        assert compute_backbone_context_margin("unet", max_stride) == math.ceil(rf / 2)

    @pytest.mark.parametrize("backbone", ["convnext", "swint"])
    def test_fixed_family_constant(self, backbone):
        """ConvNext / SwinT return the fixed per-family constant (128)."""
        margin = compute_backbone_context_margin(backbone, 16)
        assert margin == 128
        assert margin == _BACKBONE_CONTEXT_MARGIN_PX[backbone]

    @pytest.mark.parametrize("backbone", ["pretrained", "resnet", "foo"])
    def test_unsupported_backbone_raises(self, backbone):
        """Any other backbone raises (tiling unsupported there)."""
        with pytest.raises(ValueError):
            compute_backbone_context_margin(backbone, 16)


class TestComputeSuggestedTileSize:
    """``compute_suggested_tile_size`` divisibility, floor and monotonicity."""

    def test_divisible_by_lcm(self):
        """Result is a multiple of lcm(max_stride, output_stride)."""
        max_stride, output_stride, margin = 16, 6, 40
        divisor = math.lcm(max_stride, output_stride)
        tile = compute_suggested_tile_size(50, max_stride, output_stride, margin)
        assert tile % divisor == 0

    def test_covers_object_plus_margin(self):
        """Result >= object_multiple * max_bbox_dim + 2 * margin."""
        max_bbox_dim, margin = 50, 40
        tile = compute_suggested_tile_size(max_bbox_dim, 16, 2, margin)
        assert tile >= 2.0 * max_bbox_dim + 2 * margin

    def test_min_tile_multiples_floor(self):
        """A tiny object floors at min_tile_multiples * lcm."""
        max_stride, output_stride = 16, 2
        divisor = math.lcm(max_stride, output_stride)
        tile = compute_suggested_tile_size(
            1, max_stride, output_stride, 0, min_tile_multiples=2
        )
        assert tile == 2 * divisor

    def test_monotonic_in_bbox(self):
        """Larger objects never yield a smaller tile."""
        prev = 0
        for dim in [10, 50, 100, 250, 500]:
            tile = compute_suggested_tile_size(dim, 16, 2, 40)
            assert tile >= prev
            prev = tile

    def test_custom_object_multiple(self):
        """A larger object_multiple grows the tile."""
        small = compute_suggested_tile_size(50, 16, 2, 40, object_multiple=1.0)
        large = compute_suggested_tile_size(50, 16, 2, 40, object_multiple=4.0)
        assert large >= small


class TestComputeSuggestedTileOverlap:
    """``compute_suggested_tile_overlap`` divisibility, floor and clamp."""

    def test_divisible_by_output_stride(self):
        """Overlap is a multiple of output_stride."""
        output_stride = 4
        overlap = compute_suggested_tile_overlap(256, 60, 3.0, output_stride, 40)
        assert overlap % output_stride == 0

    def test_covers_object_and_sigma_and_margin(self):
        """Overlap covers 0.5*bbox + sigma_multiple*sigma + margin (when unclamped)."""
        tile_size, bbox, sigma, output_stride, margin = 512, 60, 3.0, 2, 40
        overlap = compute_suggested_tile_overlap(
            tile_size, bbox, sigma, output_stride, margin
        )
        needed = 0.5 * bbox + 3.0 * sigma + margin
        assert overlap >= needed

    def test_min_fraction_floor_for_small_object(self):
        """When the object is tiny, the min_overlap_fraction floor drives overlap."""
        tile_size, output_stride, frac = 256, 4, 0.25
        overlap = compute_suggested_tile_overlap(
            tile_size, 1, 0.1, output_stride, 0, min_overlap_fraction=frac
        )
        expected_floor = math.ceil(frac * tile_size / output_stride) * output_stride
        assert overlap == expected_floor

    def test_clamped_below_tile_size(self):
        """A huge object clamps overlap to tile_size - output_stride (< tile_size)."""
        tile_size, output_stride = 64, 2
        overlap = compute_suggested_tile_overlap(
            tile_size, 100000, 100.0, output_stride, 500
        )
        assert overlap == tile_size - output_stride
        assert overlap < tile_size


# ---------------------------------------------------------------------------
# 3. check_tiling
# ---------------------------------------------------------------------------
def _make_tiling_cfg(
    backbone="unet",
    enabled=True,
    tile_size=64,
    overlap=16,
    min_overlap_fraction=0.25,
    max_stride=8,
    output_stride=2,
    model="single_instance",
    pretrained_backbone_weights=None,
    include_tiling=True,
):
    """Build a realistic finalized-config for ``check_tiling`` tests."""
    if backbone == "pretrained":
        # A HuggingFace pretrained encoder surfaces as backbone_type "pretrained".
        backbone_config = {
            "pretrained": {"max_stride": max_stride, "output_stride": output_stride}
        }
    else:
        backbone_config = {
            backbone: {"max_stride": max_stride, "output_stride": output_stride}
        }

    if model == "multi_class_topdown":
        head_configs = {
            "multi_class_topdown": {
                "confmaps": {"output_stride": output_stride, "sigma": 1.5},
                "class_vectors": {"output_stride": max_stride, "classes": ["a", "b"]},
            }
        }
    else:
        head_configs = {
            model: {"confmaps": {"output_stride": output_stride, "sigma": 1.5}}
        }

    preprocessing = {}
    if include_tiling:
        preprocessing["tiling"] = {
            "enabled": enabled,
            "tile_size": tile_size,
            "overlap": overlap,
            "min_overlap_fraction": min_overlap_fraction,
        }

    return OmegaConf.create(
        {
            "data_config": {"preprocessing": preprocessing},
            "model_config": {
                "pretrained_backbone_weights": pretrained_backbone_weights,
                "backbone_config": backbone_config,
                "head_configs": head_configs,
            },
        }
    )


class TestCheckTiling:
    """Behaviors of ``check_tiling`` (in-place mutation + guards)."""

    def test_disabled_is_noop(self):
        """Disabled tiling returns the same config object untouched."""
        cfg = _make_tiling_cfg(enabled=False, tile_size=None, overlap=None)
        assert check_tiling(cfg) is cfg
        assert cfg.data_config.preprocessing.tiling.tile_size is None

    def test_missing_tiling_is_noop(self):
        """A config with no tiling section is returned unchanged."""
        cfg = _make_tiling_cfg(include_tiling=True, enabled=False)
        del cfg.data_config.preprocessing.tiling
        assert check_tiling(cfg) is cfg

    @pytest.mark.parametrize("backbone", ["unet", "convnext", "swint"])
    def test_supported_backbones_allowed(self, backbone):
        """UNet / ConvNext / SwinT are all allowed when tiling is enabled."""
        cfg = _make_tiling_cfg(backbone=backbone)
        # Should not raise.
        assert check_tiling(cfg) is cfg

    def test_unet_with_pretrained_weights_allowed(self):
        """A unet that merely loaded pretrained *weights* is NOT excluded."""
        cfg = _make_tiling_cfg(backbone="unet", pretrained_backbone_weights="foo.ckpt")
        # Should not raise (this clause was intentionally dropped).
        assert check_tiling(cfg) is cfg

    def test_pretrained_backbone_raises(self, caplog):
        """A pretrained-encoder backbone with tiling enabled raises."""
        cfg = _make_tiling_cfg(backbone="pretrained")
        with pytest.raises(ValueError):
            check_tiling(cfg)
        assert "pretrained" in caplog.text

    def test_multi_class_topdown_raises(self, caplog):
        """A multi_class_topdown (ClassVectorsHead) model with tiling raises."""
        cfg = _make_tiling_cfg(model="multi_class_topdown")
        with pytest.raises(ValueError):
            check_tiling(cfg)
        assert "ClassVectorsHead" in caplog.text or "multi_class_topdown" in caplog.text

    def test_tile_size_none_raises(self, caplog):
        """tile_size=None while enabled raises (setup must run first)."""
        cfg = _make_tiling_cfg(tile_size=None)
        with pytest.raises(ValueError):
            check_tiling(cfg)
        assert "tile_size" in caplog.text

    def test_tile_size_rounds_up_and_warns(self, caplog):
        """A non-divisible tile_size is rounded UP to lcm and a warning logged."""
        # lcm(8, 2) == 8; 60 -> 64.
        cfg = _make_tiling_cfg(tile_size=60, max_stride=8, output_stride=2, overlap=16)
        check_tiling(cfg)
        assert cfg.data_config.preprocessing.tiling.tile_size == 64
        assert "rounding up to 64" in caplog.text

    def test_overlap_none_raises(self, caplog):
        """overlap=None while enabled raises."""
        cfg = _make_tiling_cfg(overlap=None)
        with pytest.raises(ValueError):
            check_tiling(cfg)
        assert "overlap" in caplog.text

    def test_overlap_rounds_up_to_output_stride(self, caplog):
        """A non-divisible overlap is rounded UP to output_stride."""
        # output_stride=2; overlap=15 -> 16; min_overlap_fraction=0 to isolate.
        cfg = _make_tiling_cfg(
            tile_size=64, overlap=15, output_stride=2, min_overlap_fraction=0.0
        )
        check_tiling(cfg)
        assert cfg.data_config.preprocessing.tiling.overlap == 16
        assert "rounding up to 16" in caplog.text

    def test_overlap_raised_to_min_fraction_floor(self, caplog):
        """An overlap below the min_overlap_fraction floor is raised to it."""
        # floor = ceil(0.25 * 64 / 2) * 2 == 16.
        cfg = _make_tiling_cfg(
            tile_size=64, overlap=4, output_stride=2, min_overlap_fraction=0.25
        )
        check_tiling(cfg)
        assert cfg.data_config.preprocessing.tiling.overlap == 16
        assert "floor" in caplog.text

    def test_overlap_ge_tile_size_raises(self, caplog):
        """An overlap >= tile_size (no positive stride) raises."""
        cfg = _make_tiling_cfg(
            tile_size=64, overlap=64, output_stride=2, min_overlap_fraction=0.0
        )
        with pytest.raises(ValueError):
            check_tiling(cfg)
        assert "positive stride" in caplog.text

    def test_valid_geometry_unchanged(self):
        """Already-valid geometry passes through unchanged."""
        cfg = _make_tiling_cfg(tile_size=64, overlap=16)
        check_tiling(cfg)
        assert cfg.data_config.preprocessing.tiling.tile_size == 64
        assert cfg.data_config.preprocessing.tiling.overlap == 16


# ---------------------------------------------------------------------------
# 4. check_tiling_parity
# ---------------------------------------------------------------------------
class TestCheckTilingParity:
    """Behaviors of ``check_tiling_parity`` (inference-time geometry check)."""

    def test_disabled_is_noop(self):
        """Disabled tiling ignores overrides entirely."""
        cfg = _make_tiling_cfg(enabled=False, tile_size=64, overlap=16)
        assert check_tiling_parity(cfg, 999, 999) is cfg

    def test_no_overrides_returns_unchanged(self):
        """No overrides is always a no-op pass-through."""
        cfg = _make_tiling_cfg(tile_size=64, overlap=16)
        assert check_tiling_parity(cfg) is cfg

    def test_matching_overrides_ok(self):
        """Overrides equal to the trained geometry pass."""
        cfg = _make_tiling_cfg(tile_size=64, overlap=16)
        assert check_tiling_parity(cfg, 64, 16) is cfg

    def test_tile_size_override_mismatch_raises(self, caplog):
        """A divergent tile_size override raises."""
        cfg = _make_tiling_cfg(tile_size=64, overlap=16)
        with pytest.raises(ValueError):
            check_tiling_parity(cfg, tile_size_override=128)
        assert "tile_size override" in caplog.text

    def test_overlap_override_mismatch_raises(self, caplog):
        """A divergent overlap override raises."""
        cfg = _make_tiling_cfg(tile_size=64, overlap=16)
        with pytest.raises(ValueError):
            check_tiling_parity(cfg, overlap_override=8)
        assert "overlap override" in caplog.text


# ---------------------------------------------------------------------------
# 5. warn_on_tiled_export
# ---------------------------------------------------------------------------
class TestWarnOnTiledExport:
    """``warn_on_tiled_export`` emits a loguru warning only when tiling is on."""

    def test_warns_when_enabled(self, caplog):
        """Enabled tiling emits a warning."""
        cfg = OmegaConf.create(
            {"data_config": {"preprocessing": {"tiling": {"enabled": True}}}}
        )
        warn_on_tiled_export(cfg)
        assert "tiling enabled" in caplog.text
        assert "export is not yet supported" in caplog.text

    def test_silent_when_disabled(self, caplog):
        """Disabled tiling emits nothing."""
        cfg = OmegaConf.create(
            {"data_config": {"preprocessing": {"tiling": {"enabled": False}}}}
        )
        warn_on_tiled_export(cfg)
        assert caplog.text == ""

    def test_silent_when_missing(self, caplog):
        """A config with no tiling section emits nothing."""
        cfg = OmegaConf.create({"data_config": {"preprocessing": {}}})
        warn_on_tiled_export(cfg)
        assert caplog.text == ""
