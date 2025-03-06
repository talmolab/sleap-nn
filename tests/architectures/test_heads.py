import pytest
from omegaconf import OmegaConf
from sleap_nn.architectures.heads import (
    Head,
    SingleInstanceConfmapsHead,
    CentroidConfmapsHead,
    CenteredInstanceConfmapsHead,
    MultiInstanceConfmapsHead,
    PartAffinityFieldsHead,
    ClassMapsHead,
    ClassVectorsHead,
    OffsetRefinementHead,
)
import torch

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


def test_head(caplog):
    output_stride = 1
    loss_weight = 1.0

    head = Head(output_stride=output_stride, loss_weight=loss_weight)
    with pytest.raises(NotImplementedError):
        _ = head.channels
    assert head.output_stride == output_stride
    assert head.loss_function == "mse"
    assert head.activation == "identity"
    assert head.loss_weight == loss_weight
    assert "Subclasses must implement this method." in caplog.text


def test_single_instance_confmaps_head(caplog):
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    part_names = None
    sigma = 5.0
    with pytest.raises(ValueError):
        _ = SingleInstanceConfmapsHead.from_config(
            OmegaConf.create(
                {
                    "part_names": part_names,
                    "sigma": sigma,
                    "output_stride": output_stride,
                    "loss_weight": loss_weight,
                }
            )
        )

    part_names = ["part1", "part2"]
    sigma = 5.0
    head = SingleInstanceConfmapsHead(
        part_names=part_names,
        sigma=sigma,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)

    assert output.shape[1] == head.channels
    assert output.dtype == torch.float32
    assert head.part_names == part_names
    assert head.sigma == sigma
    assert head.channels == len(part_names)
    assert "Required attribute 'part_names" in caplog.text

    base_unet_head_config = OmegaConf.create(
        {
            "part_names": [f"{i}" for i in range(13)],
            "sigma": 5.0,
            "output_stride": 1,
            "loss_weight": 1.0,
        }
    )

    head = SingleInstanceConfmapsHead.from_config(base_unet_head_config)
    assert isinstance(head, Head)


def test_centroid_confmaps_head():
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    anchor_part = "anchor_part"
    sigma = 5.0
    head = CentroidConfmapsHead(
        anchor_part=anchor_part,
        sigma=sigma,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)

    assert output.shape[1] == 1
    assert output.dtype == torch.float32
    assert head.anchor_part == anchor_part
    assert head.sigma == sigma
    assert head.channels == 1

    config = OmegaConf.create(
        {
            "anchor_part": anchor_part,
            "sigma": sigma,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = CentroidConfmapsHead.from_config(config)
    assert isinstance(head, Head)


def test_centered_instance_confmaps_head(caplog):
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    part_names = ["part1", "part2"]
    anchor_part = "anchor_part"
    sigma = 5.0
    head = CenteredInstanceConfmapsHead(
        part_names=part_names,
        anchor_part=anchor_part,
        sigma=sigma,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)

    assert output.shape[1] == len(part_names)
    assert output.dtype == torch.float32
    assert head.part_names == part_names
    assert head.anchor_part == anchor_part
    assert head.sigma == sigma
    assert head.channels == len(part_names)

    config = OmegaConf.create(
        {
            "part_names": part_names,
            "anchor_part": anchor_part,
            "sigma": sigma,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = CenteredInstanceConfmapsHead.from_config(config)
    assert isinstance(head, Head)

    with pytest.raises(ValueError):
        _ = CenteredInstanceConfmapsHead.from_config(
            OmegaConf.create(
                {
                    "part_names": None,
                    "anchor_part": anchor_part,
                    "sigma": sigma,
                    "output_stride": output_stride,
                    "loss_weight": loss_weight,
                }
            )
        )

    assert "Required attribute 'part_names'" in caplog.text


def test_multi_instance_confmaps_head(caplog):
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    part_names = ["part1", "part2"]
    sigma = 5.0
    head = MultiInstanceConfmapsHead(
        part_names=part_names,
        sigma=sigma,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)
    assert output.shape[1] == len(part_names)
    assert output.dtype == torch.float32
    assert head.part_names == part_names
    assert head.sigma == sigma
    assert head.channels == len(part_names)

    config = OmegaConf.create(
        {
            "part_names": part_names,
            "sigma": sigma,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = MultiInstanceConfmapsHead.from_config(config)
    assert isinstance(head, Head)

    with pytest.raises(ValueError):
        _ = MultiInstanceConfmapsHead.from_config(
            OmegaConf.create(
                {
                    "part_names": None,
                    "sigma": sigma,
                    "output_stride": output_stride,
                    "loss_weight": loss_weight,
                }
            )
        )
    assert "Required attribute 'part_names'" in caplog.text


def test_part_affinity_fields_head():
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    edges = [("part1", "part2"), ("part2", "part3")]
    sigma = 5.0
    head = PartAffinityFieldsHead(
        edges=edges,
        sigma=sigma,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)
    assert output.shape[1] == len(edges) * 2
    assert output.dtype == torch.float32
    assert head.edges == edges
    assert head.sigma == sigma
    assert head.channels == len(edges) * 2

    config = OmegaConf.create(
        {
            "edges": edges,
            "sigma": sigma,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = PartAffinityFieldsHead.from_config(config)
    assert isinstance(head, Head)


def test_class_maps_head():
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    classes = ["class1", "class2"]
    sigma = 5.0
    head = ClassMapsHead(
        classes=classes,
        sigma=sigma,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)
    assert output.shape[1] == len(classes)
    assert output.dtype == torch.float32
    assert head.classes == classes
    assert head.sigma == sigma
    assert head.channels == len(classes)

    config = OmegaConf.create(
        {
            "classes": classes,
            "sigma": sigma,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = ClassMapsHead.from_config(config)
    assert isinstance(head, Head)


def test_class_vectors_head():
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    classes = ["class1", "class2"]
    num_fc_layers = 2
    num_fc_units = 64
    global_pool = True
    head = ClassVectorsHead(
        classes=classes,
        num_fc_layers=num_fc_layers,
        num_fc_units=num_fc_units,
        global_pool=global_pool,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )

    output = head.make_head(sample_input.size(1))(sample_input)
    assert head.activation == "softmax"
    assert head.loss_function == "categorical_crossentropy"
    assert output.shape[1] == len(classes)
    assert output.dtype == torch.float32
    assert head.classes == classes
    assert head.num_fc_layers == num_fc_layers
    assert head.num_fc_units == num_fc_units
    assert head.global_pool == global_pool
    assert head.channels == len(classes)

    config = OmegaConf.create(
        {
            "classes": classes,
            "num_fc_layers": num_fc_layers,
            "num_fc_units": num_fc_units,
            "global_pool": global_pool,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = ClassVectorsHead.from_config(config)
    assert isinstance(head, Head)


def test_offset_refinement_head(caplog):
    output_stride = 1
    loss_weight = 1.0
    sample_input = torch.randn(1, 3, 64, 64)

    part_names = ["part1", "part2"]
    sigma_threshold = 0.2
    head = OffsetRefinementHead(
        part_names=part_names,
        sigma_threshold=sigma_threshold,
        output_stride=output_stride,
        loss_weight=loss_weight,
    )
    output = head.make_head(sample_input.size(1))(sample_input)
    assert output.shape[1] == len(part_names) * 2
    assert output.dtype == torch.float32
    assert head.part_names == part_names
    assert head.sigma_threshold == sigma_threshold
    assert head.channels == len(part_names) * 2

    config = OmegaConf.create(
        {
            "part_names": part_names,
            "sigma_threshold": sigma_threshold,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = OffsetRefinementHead.from_config(config)
    assert isinstance(head, Head)

    config = OmegaConf.create(
        {
            "anchor_part": part_names[0],
            "sigma_threshold": sigma_threshold,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    head = OffsetRefinementHead.from_config(config)
    assert isinstance(head, Head)

    config = OmegaConf.create(
        {
            "sigma_threshold": sigma_threshold,
            "output_stride": output_stride,
            "loss_weight": loss_weight,
        }
    )

    with pytest.raises(ValueError):
        _ = OffsetRefinementHead.from_config(config)
    assert "Required attribute 'part_names'" in caplog.text
