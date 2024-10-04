from pathlib import Path
from omegaconf import DictConfig
import functools
import litdata as ld
import shutil
import sleap_io as sio
from sleap_nn.data.get_data_chunks import (
    bottomup_data_chunks,
    centered_instance_data_chunks,
    centroid_data_chunks,
    single_instance_data_chunks,
)
from sleap_nn.data.providers import get_max_height_width
from sleap_nn.data.streaming_datasets import (
    BottomUpStreamingDataset,
    CenteredInstanceStreamingDataset,
    CentroidStreamingDataset,
    SingleInstanceStreamingDataset,
)


def test_bottomup_streaming_dataset(minimal_instance, sleap_data_dir, config):
    """Test BottomUpStreamingDataset class."""
    labels = sio.load_slp(minimal_instance)
    max_hw = get_max_height_width(labels)
    edge_inds = labels.skeletons[0].edge_inds

    dir_path = Path(sleap_data_dir) / "data_chunks"

    partial_func = functools.partial(
        bottomup_data_chunks,
        data_config=config.data_config,
        max_instances=2,
        max_hw=max_hw,
    )
    ld.optimize(
        fn=partial_func,
        inputs=[(x, labels.videos.index(x.video)) for x in labels],
        output_dir=str(dir_path),
        chunk_size=4,
    )

    try:
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})
        pafs_head = DictConfig({"sigma": 4, "output_stride": 4})

        dataset = BottomUpStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            confmap_head=confmap_head,
            pafs_head=pafs_head,
            edge_inds=edge_inds,
            max_stride=100,
            scale=0.5,
            input_dir=str(dir_path),
        )

        samples = list(iter(dataset))
        assert len(samples) == 1

        assert samples[0]["image"].shape == (1, 1, 200, 200)
        assert samples[0]["confidence_maps"].shape == (1, 2, 100, 100)
        assert samples[0]["part_affinity_fields"].shape == (50, 50, 2)

        # test with random crop
        config.data_config.augmentation_config.geometric["random_crop_p"] = 1.0
        config.data_config.augmentation_config.geometric["random_crop_height"] = 300
        config.data_config.augmentation_config.geometric["random_crop_width"] = 300
        dataset = BottomUpStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            confmap_head=confmap_head,
            pafs_head=pafs_head,
            edge_inds=edge_inds,
            max_stride=2,
            scale=1.0,
            input_dir=str(dir_path),
            apply_aug=True,
        )

        samples = list(iter(dataset))
        assert len(samples) == 1

        assert samples[0]["image"].shape == (1, 1, 300, 300)
        assert samples[0]["confidence_maps"].shape == (1, 2, 150, 150)
        assert samples[0]["part_affinity_fields"].shape == (75, 75, 2)

    finally:
        shutil.rmtree(dir_path)


def test_centered_instance_streaming_dataset(minimal_instance, sleap_data_dir, config):
    """Test CenteredInstanceStreamingDataset class."""
    labels = sio.load_slp(minimal_instance)
    max_hw = get_max_height_width(labels)

    dir_path = Path(sleap_data_dir) / "data_chunks"

    partial_func = functools.partial(
        centered_instance_data_chunks,
        data_config=config.data_config,
        max_instances=2,
        crop_size=(160, 160),
        anchor_ind=0,
        max_hw=max_hw,
    )
    ld.optimize(
        fn=partial_func,
        inputs=[(x, labels.videos.index(x.video)) for x in labels],
        output_dir=str(dir_path),
        chunk_size=4,
    )

    try:
        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = CenteredInstanceStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            confmap_head=confmap_head,
            crop_hw=(160, 160),
            max_stride=100,
            scale=0.5,
            input_dir=str(dir_path),
        )

        samples = list(iter(dataset))
        assert len(samples) == 2

        assert samples[0]["instance_image"].shape == (1, 1, 100, 100)
        assert samples[0]["confidence_maps"].shape == (1, 2, 50, 50)

    finally:
        shutil.rmtree(dir_path)


def test_centroid_streaming_dataset(minimal_instance, sleap_data_dir, config):
    """Test CentroidStreamingDataset class."""
    labels = sio.load_slp(minimal_instance)
    max_hw = get_max_height_width(labels)

    dir_path = Path(sleap_data_dir) / "data_chunks"

    partial_func = functools.partial(
        centroid_data_chunks,
        data_config=config.data_config,
        max_instances=2,
        anchor_ind=0,
        max_hw=max_hw,
    )

    ld.optimize(
        fn=partial_func,
        inputs=[(x, labels.videos.index(x.video)) for x in labels],
        output_dir=str(dir_path),
        chunk_size=4,
    )

    try:

        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = CentroidStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            confmap_head=confmap_head,
            max_stride=100,
            scale=0.5,
            input_dir=str(dir_path),
        )

        samples = list(iter(dataset))
        assert len(samples) == 1

        assert samples[0]["image"].shape == (1, 1, 200, 200)
        assert samples[0]["centroids_confidence_maps"].shape == (1, 1, 100, 100)

        # test with random crop
        config.data_config.augmentation_config.geometric["random_crop_p"] = 1.0
        config.data_config.augmentation_config.geometric["random_crop_height"] = 300
        config.data_config.augmentation_config.geometric["random_crop_width"] = 300
        dataset = CentroidStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            confmap_head=confmap_head,
            max_stride=2,
            scale=1.0,
            input_dir=str(dir_path),
            apply_aug=True,
        )

        samples = list(iter(dataset))
        assert len(samples) == 1

        assert samples[0]["image"].shape == (1, 1, 300, 300)
        assert samples[0]["centroids_confidence_maps"].shape == (1, 1, 150, 150)

    finally:
        shutil.rmtree(dir_path)


def test_single_instance_streaming_dataset(minimal_instance, sleap_data_dir, config):
    """Test SingleInstanceStreamingDataset class."""
    labels = sio.load_slp(minimal_instance)
    max_hw = get_max_height_width(labels)

    dir_path = Path(sleap_data_dir) / "data_chunks"

    partial_func = functools.partial(
        single_instance_data_chunks, data_config=config.data_config, max_hw=max_hw
    )

    for lf in labels:
        lf.instances = lf.instances[:1]
    ld.optimize(
        fn=partial_func,
        inputs=[(x, labels.videos.index(x.video)) for x in labels],
        output_dir=str(dir_path),
        chunk_size=4,
    )

    try:

        confmap_head = DictConfig({"sigma": 1.5, "output_stride": 2})

        dataset = SingleInstanceStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            confmap_head=confmap_head,
            max_stride=100,
            scale=0.5,
            input_dir=str(dir_path),
        )

        samples = list(iter(dataset))
        assert len(samples) == 1

        assert samples[0]["image"].shape == (1, 1, 200, 200)
        assert samples[0]["confidence_maps"].shape == (1, 2, 100, 100)

        # test with random crop
        config.data_config.augmentation_config.geometric["random_crop_p"] = 1.0
        config.data_config.augmentation_config.geometric["random_crop_height"] = 300
        config.data_config.augmentation_config.geometric["random_crop_width"] = 300
        dataset = SingleInstanceStreamingDataset(
            augmentation_config=config.data_config.augmentation_config,
            apply_aug=True,
            confmap_head=confmap_head,
            max_stride=2,
            scale=1.0,
            input_dir=str(dir_path),
        )

        samples = list(iter(dataset))
        assert len(samples) == 1

        assert samples[0]["image"].shape == (1, 1, 300, 300)
        assert samples[0]["confidence_maps"].shape == (1, 2, 150, 150)

    finally:
        shutil.rmtree(dir_path)
