"""Utilities for config building and validation."""

from loguru import logger
from omegaconf import DictConfig, OmegaConf


def get_model_type_from_cfg(config: DictConfig):
    """Return the model type from the config. One of [single_instance, centroid, centered_instance, bottomup]."""
    model_type = None
    for k, v in config.model_config.head_configs.items():
        if v is not None:
            model_type = k
            break
    return model_type


def get_backbone_type_from_cfg(config: DictConfig):
    """Return the backbone type from the config. One of [unet, swint, convnext]."""
    backbone_type = None
    for k, v in config.model_config.backbone_config.items():
        if v is not None:
            backbone_type = k
            break
    return backbone_type


def get_output_strides_from_heads(head_configs: DictConfig):
    """Get list of output strides from head configs."""
    output_strides_from_heads = []
    for head_type in head_configs:
        if head_configs[head_type] is not None:
            for head_layer in head_configs[head_type]:
                output_strides_from_heads.append(
                    head_configs[head_type][head_layer]["output_stride"]
                )
    return output_strides_from_heads


def check_output_strides(config: OmegaConf) -> OmegaConf:
    """Check max_stride and output_stride in backbone_config with head_config."""
    output_strides = get_output_strides_from_heads(config.model_config.head_configs)
    backbone_type = get_backbone_type_from_cfg(config)
    if output_strides:
        config.model_config.backbone_config[f"{backbone_type}"]["output_stride"] = min(
            output_strides
        )
        if config.model_config.backbone_config[f"{backbone_type}"]["max_stride"] < max(
            output_strides
        ):
            config.model_config.backbone_config[f"{backbone_type}"]["max_stride"] = max(
                output_strides
            )

    model_type = get_model_type_from_cfg(config)
    if model_type == "multi_class_topdown":
        config.model_config.head_configs.multi_class_topdown.class_vectors.output_stride = config.model_config.backbone_config[
            f"{backbone_type}"
        ][
            "max_stride"
        ]
    return config


def oneof(attrs_cls, must_be_set: bool = False):
    """Ensure that the decorated attrs class only has a single attribute set.

    This decorator is inspired by the `oneof` protobuffer field behavior.

    Args:
        attrs_cls: An attrs decorated class.
        must_be_set: If True, raise an error if none of the attributes are set. If not,
            error will only be raised if more than one attribute is set.

    Returns:
        The `attrs_cls` with an `__init__` method that checks for the number of
        attributes that are set.
    """
    # Check if the class is an attrs class at all.
    if not hasattr(attrs_cls, "__attrs_attrs__"):
        message = "Classes decorated with oneof must also be attr.s decorated."
        logger.error(message)
        raise ValueError(message)

    # Pull out attrs generated class attributes.
    attribs = attrs_cls.__attrs_attrs__
    init_fn = attrs_cls.__init__

    # Define a new __init__ function that wraps the attrs generated one.
    def new_init_fn(self, *args, **kwargs):
        # Execute the standard attrs-generated __init__.
        init_fn(self, *args, **kwargs)

        # Check for attribs with set values.
        attribs_with_value = [
            attrib for attrib in attribs if getattr(self, attrib.name) is not None
        ]

        class_name = self.__class__.__name__

        if len(attribs_with_value) > 1:
            # Raise error if more than one attribute is set.
            message = (
                f"{class_name}: Only one attribute of this class can be set (not None)."
            )
            logger.error(message)
            raise ValueError(message)

        if len(attribs_with_value) == 0 and must_be_set:
            # Raise error if none are set.
            message = f"{class_name}: At least one attribute of this class must be set."
            logger.error(message)
            raise ValueError(message)

    # Replace with wrapped __init__.
    attrs_cls.__init__ = new_init_fn

    # Define convenience method for getting the set attribute.
    def which_oneof_attrib_name(self):
        attribs_with_value = [
            attrib for attrib in attribs if getattr(self, attrib.name) is not None
        ]
        class_name = self.__class__.__name__

        if len(attribs_with_value) > 1:
            # Raise error if more than one attribute is set.
            message = (
                f"{class_name}: Only one attribute of this class can be set (not None)."
            )
            logger.error(message)
            raise ValueError(message)

        if len(attribs_with_value) == 0:
            if must_be_set:
                # Raise error if none are set.
                message = (
                    f"{class_name}: At least one attribute of this class must be set."
                )
                logger.error(message)
                raise ValueError(message)
            else:
                return None

        return attribs_with_value[0].name

    def which_oneof(self):
        attrib_name = self.which_oneof_attrib_name()

        if attrib_name is None:
            return None

        return getattr(self, attrib_name)

    attrs_cls.which_oneof_attrib_name = which_oneof_attrib_name
    attrs_cls.which_oneof = which_oneof

    return attrs_cls
