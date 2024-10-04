"""Function to generate `.bin` files."""

import argparse
import functools
import litdata as ld
from pathlib import Path
from omegaconf import OmegaConf
import sleap_io as sio

from sleap_nn.data.providers import get_max_instances, get_max_height_width
from sleap_nn.data.get_data_chunks import (
    bottomup_data_chunks,
    centered_instance_data_chunks,
    centroid_data_chunks,
    single_instance_data_chunks,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str)
    parser.add_argument("--user_instances_only", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--crop_hw", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(f"{args.dir_path}/initial_config.yaml")

    train_labels = sio.load_slp(config.data_config.train_labels_path)
    max_height, max_width = get_max_height_width(train_labels)
    val_labels = sio.load_slp(config.data_config.val_labels_path)
    user_instances_only = False if args.user_instances_only == 0 else True

    max_stride = config.model_config.backbone_config.max_stride
    max_instances = get_max_instances(train_labels)

    print("Starting data-chunk generation...")

    if args.model_type == "single_instance":

        factory_get_chunks = functools.partial(
            single_instance_data_chunks,
            data_config=config.data_config,
            user_instances_only=user_instances_only,
            max_hw=(max_height, max_width),
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, train_labels.videos.index(x.video)) for x in train_labels],
            output_dir=(Path(args.dir_path) / "train_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, val_labels.videos.index(x.video)) for x in val_labels],
            output_dir=(Path(args.dir_path) / "val_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

    elif args.model_type == "centered_instance":

        factory_get_chunks = functools.partial(
            centered_instance_data_chunks,
            data_config=config.data_config,
            max_instances=max_instances,
            crop_size=(args.crop_hw, args.crop_hw),
            anchor_ind=config.model_config.head_configs.centered_instance.confmaps.anchor_part,
            user_instances_only=user_instances_only,
            max_hw=(max_height, max_width),
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, train_labels.videos.index(x.video)) for x in train_labels],
            output_dir=(Path(args.dir_path) / "train_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, val_labels.videos.index(x.video)) for x in val_labels],
            output_dir=(Path(args.dir_path) / "val_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

    elif args.model_type == "centroid":
        factory_get_chunks = functools.partial(
            centroid_data_chunks,
            data_config=config.data_config,
            max_instances=max_instances,
            anchor_ind=config.model_config.head_configs.centroid.confmaps.anchor_part,
            user_instances_only=user_instances_only,
            max_hw=(max_height, max_width),
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, train_labels.videos.index(x.video)) for x in train_labels],
            output_dir=(Path(args.dir_path) / "train_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, val_labels.videos.index(x.video)) for x in val_labels],
            output_dir=(Path(args.dir_path) / "val_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

    elif args.model_type == "bottomup":
        factory_get_chunks = functools.partial(
            bottomup_data_chunks,
            data_config=config.data_config,
            max_instances=max_instances,
            user_instances_only=user_instances_only,
            max_hw=(max_height, max_width),
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, train_labels.videos.index(x.video)) for x in train_labels],
            output_dir=(Path(args.dir_path) / "train_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

        ld.optimize(
            fn=factory_get_chunks,
            inputs=[(x, val_labels.videos.index(x.video)) for x in val_labels],
            output_dir=(Path(args.dir_path) / "val_chunks").as_posix(),
            num_workers=args.num_workers,
            chunk_size=args.chunk_size,
        )

    else:
        raise ValueError(
            f"{args.model_type} is not defined. Please choose one of `single_instance`, `centered_instance`, `centroid`, `bottomup`."
        )
