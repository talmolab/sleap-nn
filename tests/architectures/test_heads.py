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

output_stride = 1
loss_weight = 1.0
sample_input = torch.randn(1, 3, 64, 64)


def test_head():
    head = Head(output_stride=output_stride, loss_weight=loss_weight)
    assert head.output_stride == output_stride
    assert head.loss_weight == loss_weight


def test_single_instance_confmaps_head():
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
    assert head.part_names == part_names
    assert head.sigma == sigma
    assert head.channels == len(part_names)


def test_centroid_confmaps_head():
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
    assert head.anchor_part == anchor_part
    assert head.sigma == sigma
    assert head.channels == 1


def test_centered_instance_confmaps_head():
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
    assert head.part_names == part_names
    assert head.anchor_part == anchor_part
    assert head.sigma == sigma
    assert head.channels == len(part_names)


def test_multi_instance_confmaps_head():
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
    assert head.part_names == part_names
    assert head.sigma == sigma
    assert head.channels == len(part_names)


def test_part_affinity_fields_head():
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
    assert head.edges == edges
    assert head.sigma == sigma
    assert head.channels == len(edges) * 2


def test_class_maps_head():
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
    assert head.classes == classes
    assert head.sigma == sigma
    assert head.channels == len(classes)


def test_class_vectors_head():
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
    assert output.shape[1] == len(classes)
    assert head.classes == classes
    assert head.num_fc_layers == num_fc_layers
    assert head.num_fc_units == num_fc_units
    assert head.global_pool == global_pool
    assert head.channels == len(classes)


def test_offset_refinement_head():
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
    assert head.part_names == part_names
    assert head.sigma_threshold == sigma_threshold
    assert head.channels == len(part_names) * 2
