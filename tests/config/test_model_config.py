"""Tests for the serializable configuration classes for specifying all model config parameters.

These configuration classes are intended to specify all
the parameters required to initialize the model config.
"""

import pytest
from omegaconf import OmegaConf
from loguru import logger

from _pytest.logging import LogCaptureFixture

from sleap_nn.config.model_config import (
    ModelConfig,
    BackboneConfig,
    UNetConfig,
    ConvNextConfig,
    SwinTConfig,
    SwinTBaseConfig,
    SwinTSmallConfig,
    HeadConfig,
    SingleInstanceConfig,
    CentroidConfig,
    CenteredInstanceConfig,
    BottomUpConfig,
    SingleInstanceConfMapsConfig,
    CentroidConfMapsConfig,
    CenteredInstanceConfMapsConfig,
    BottomUpConfMapsConfig,
    PAFConfig,
    model_mapper,
)


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


@pytest.fixture
def default_config():
    """Fixture for a default ModelConfig instance."""
    return ModelConfig(
        init_weights="default",
        backbone_config=BackboneConfig(),
        head_configs=HeadConfig(),
    )


def test_default_initialization(default_config):
    """Test default initialization of ModelConfig."""
    assert default_config.init_weights == "default"


def test_invalid_pre_trained_weights(caplog):
    """Test validation failure with an invalid pre_trained_weights."""
    with pytest.raises(ValueError):
        ModelConfig(
            backbone_config=BackboneConfig(
                convnext=ConvNextConfig(
                    pre_trained_weights="here",
                )
            ),
        )
    assert "Invalid pre-trained" in caplog.text

    with pytest.raises(ValueError):
        ModelConfig(
            backbone_config=BackboneConfig(
                swint=SwinTConfig(
                    pre_trained_weights="here",
                )
            ),
        )
    assert "Invalid pre-trained" in caplog.text


def test_update_config(default_config):
    """Test updating configuration attributes."""
    config = OmegaConf.structured(
        ModelConfig(
            init_weights="default",
            backbone_config=BackboneConfig(unet=UNetConfig()),
            head_configs=HeadConfig(),
        )
    )


def test_valid_model_type():
    """Test valid model_type values."""
    valid_types = ["tiny", "small", "base"]
    for model_type in valid_types:
        config = SwinTConfig(model_type=model_type)


def test_invalid_model_type(caplog):
    """Test validation failure with an invalid model_type."""
    with pytest.raises(ValueError):
        SwinTConfig(model_type="invalid_model_type")
    assert "Invalid model_type" in caplog.text

    with pytest.raises(ValueError):
        SwinTSmallConfig(model_type="invalid_model_type")
    assert "Invalid model_type" in caplog.text

    with pytest.raises(ValueError):
        SwinTBaseConfig(model_type="invalid_model_type")
    assert "Invalid model_type" in caplog.text


def test_model_mapper():
    """Test the model_mapper function with a sample legacy configuration."""
    legacy_config = {
        "model": {
            "backbone": {
                "unet": {
                    "filters": 64,
                    "filters_rate": 2.0,
                    "max_stride": 32,
                    "stem_stride": 8,
                    "middle_block": True,
                    "up_interpolate": False,
                    "stacks": 2,
                    "output_stride": 2,
                }
            },
            "heads": {
                "single_instance": {
                    "part_names": ["head", "thorax", "abdomen"],
                    "sigma": 3.0,
                    "output_stride": 2,
                },
            },
        },
        "backbone_type": "unet",
    }

    config = model_mapper(legacy_config)
    # Test backbone config
    assert config.backbone_config.unet is not None
    assert config.backbone_config.unet.filters == 64
    assert config.backbone_config.unet.filters_rate == 2.0
    assert config.backbone_config.unet.max_stride == 32
    assert config.backbone_config.unet.stem_stride == 8
    assert config.backbone_config.unet.middle_block is True
    assert config.backbone_config.unet.up_interpolate is False
    assert config.backbone_config.unet.stacks == 2
    assert config.backbone_config.unet.output_stride == 2

    # Test head configs
    assert config.head_configs.single_instance is not None
    assert config.head_configs.single_instance.confmaps.part_names == [
        "head",
        "thorax",
        "abdomen",
    ]
    assert config.head_configs.single_instance.confmaps.sigma == 3.0
    assert config.head_configs.single_instance.confmaps.output_stride == 2


def test_model_oneof_failure_model_config(caplog):
    """Test validation failure with oneof fields."""
    with pytest.raises(ValueError):
        ModelConfig(
            backbone_config=BackboneConfig(
                unet=UNetConfig(),
                swint=SwinTConfig(),
            )
        )
    assert "Only one attribute of this class can be set (not None).\n" in caplog.text


def test_model_oneof_failure_head_config(caplog):
    """Test validation failure with oneof fields."""
    with pytest.raises(ValueError):
        HeadConfig(
            single_instance=SingleInstanceConfig(),
            centroid=CentroidConfig(),
        )
    assert "Only one attribute of this class can be set (not None).\n" in caplog.text


def test_pretrained_backbone_config():
    """The `pretrained` backbone member (HuggingFace) is a valid oneof member."""
    from sleap_nn.config.model_config import PretrainedConfig

    cfg = BackboneConfig(
        pretrained=PretrainedConfig(
            model_name="microsoft/resnet-50", output_stride=4, max_stride=32
        )
    )
    assert cfg.which_oneof_attrib_name() == "pretrained"
    assert cfg.pretrained.source == "hf"
    assert cfg.pretrained.model_name == "microsoft/resnet-50"
    # Defaults required by check_output_strides / model_trainer / export.
    assert cfg.pretrained.in_channels == 3
    assert cfg.pretrained.output_stride == 4
    assert cfg.pretrained.max_stride == 32
    assert cfg.pretrained.weights is True
    assert cfg.pretrained.freeze is False
    assert cfg.pretrained.mode == "auto"


def test_pretrained_backbone_oneof_exclusive(caplog):
    """`pretrained` cannot be set alongside another backbone."""
    from sleap_nn.config.model_config import PretrainedConfig

    with pytest.raises(ValueError):
        BackboneConfig(unet=UNetConfig(), pretrained=PretrainedConfig())
    assert "Only one attribute of this class can be set (not None).\n" in caplog.text


def test_pretrained_backbone_struct_merge():
    """A YAML-style dict merges onto the structured schema (the training gate)."""
    from omegaconf import OmegaConf
    from sleap_nn.config.model_config import ModelConfig

    struct = OmegaConf.structured(ModelConfig())
    user = OmegaConf.create(
        {
            "backbone_config": {
                "unet": None,
                "convnext": None,
                "swint": None,
                "pretrained": {
                    "source": "hf",
                    "model_name": "facebook/convnextv2-nano-22k-224",
                    "weights": False,
                    "output_stride": 2,
                    "max_stride": 32,
                },
            }
        }
    )
    merged = OmegaConf.merge(struct, user)
    assert (
        merged.backbone_config.pretrained.model_name
        == "facebook/convnextv2-nano-22k-224"
    )
    assert merged.backbone_config.pretrained.weights is False


def test_get_head_configs_embedding_missing_leaf_raises():
    """A dict embedding head with no inner `embedding` leaf raises a clear error.

    The dict factory must point the user at the expected structure instead of
    surfacing a raw KeyError/AttributeError.
    """
    from sleap_nn.config.get_config import get_head_configs

    for bad in ({"embedding": {}}, {"embedding": {"embedding": None}}):
        with pytest.raises(ValueError, match="embedding"):
            get_head_configs(bad)


class TestEmbeddingPretrainedEncoderMode:
    """`embedding` + `pretrained` must be configurable through the DEFAULT path.

    Both features have a notion of "this model has no decoder" and they spell it
    differently: the embedding model says it with strides (`check_output_strides`
    pins `backbone.output_stride = max_stride` so the native UNet/ConvNeXt/SwinT
    decoder comes out empty), while the `pretrained` wrapper says it with
    `mode="encoder"` and treats `output_stride == max_stride` as a user error
    ("nothing to decode"). Pinning the stride is therefore what made the default
    `mode: auto` refuse to build on every hierarchical pretrained backbone.
    """

    @staticmethod
    def _cfg(backbone, mode=None, head_output_stride=32, max_stride=32):
        backbone_cfg = {"max_stride": max_stride, "output_stride": 2}
        if backbone == "pretrained":
            backbone_cfg.update(
                {"model_name": "facebook/convnextv2-nano-22k-224", "mode": mode}
            )
        return OmegaConf.create(
            {
                "model_config": {
                    "backbone_config": {backbone: backbone_cfg},
                    "head_configs": {
                        "embedding": {
                            "embedding": {
                                "embedding_dim": 128,
                                "output_stride": head_output_stride,
                            }
                        }
                    },
                }
            }
        )

    def test_auto_resolves_to_encoder(self):
        """The default. Before this, it raised "nothing to decode" at build time."""
        from sleap_nn.config.utils import check_output_strides

        cfg = check_output_strides(self._cfg("pretrained", mode="auto"))
        assert cfg.model_config.backbone_config.pretrained.mode == "encoder"

    def test_explicit_encoder_is_left_alone(self):
        from sleap_nn.config.utils import check_output_strides

        cfg = check_output_strides(self._cfg("pretrained", mode="encoder"))
        assert cfg.model_config.backbone_config.pretrained.mode == "encoder"

    def test_explicit_decoder_is_rejected_for_a_pooled_head(self):
        """A decoder under a pooled head gets no gradient; say so, not "strides"."""
        from sleap_nn.config.utils import check_output_strides

        with pytest.raises(ValueError, match="no gradient"):
            check_output_strides(self._cfg("pretrained", mode="decoder"))

    def test_pretrained_stride_is_not_pinned_to_max_stride(self):
        """The pin is the collision; the head still gets max_stride."""
        from sleap_nn.config.utils import check_output_strides

        cfg = check_output_strides(
            self._cfg("pretrained", mode="auto", head_output_stride=16, max_stride=32)
        )
        head = cfg.model_config.head_configs.embedding.embedding
        assert head.output_stride == 32
        assert cfg.model_config.backbone_config.pretrained.output_stride != 32

    def test_native_backbones_still_pin_the_stride(self):
        """UNet/ConvNeXt/SwinT express "no decoder" with strides; unchanged."""
        from sleap_nn.config.utils import check_output_strides

        for backbone in ("unet", "convnext", "swint"):
            cfg = check_output_strides(self._cfg(backbone))
            assert (
                cfg.model_config.backbone_config[backbone].output_stride == 32
            ), backbone
            assert (
                cfg.model_config.head_configs.embedding.embedding.output_stride == 32
            ), backbone
