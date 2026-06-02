"""Dataset fixtures for unit testing."""

from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
import pytest
import sleap_io as sio


@pytest.fixture
def sleap_nn_data_dir(pytestconfig):
    """Dir path to sleap data."""
    return Path(pytestconfig.rootdir) / "tests/assets/datasets"


def make_seg_labels_from_slp(src_slp: str, radius: int = 30) -> "sio.Labels":
    """Build synthetic instance-segmentation labels from a keypoint ``.slp``.

    For the first labeled frame, a circular foreground mask is generated around
    the mean keypoint location of each instance. The original keypoint instances
    are retained (so the skeleton is preserved) alongside the new masks.

    Args:
        src_slp: Path to a source ``.slp`` with keypoint instances + image data.
        radius: Radius (px) of the synthetic circular mask per instance.

    Returns:
        ``sio.Labels`` with one frame carrying ``UserSegmentationMask`` objects.
    """
    labels = sio.load_slp(src_slp)
    video = labels.videos[0]
    lf0 = labels[0]
    h, w = video.shape[1], video.shape[2]
    yy, xx = np.ogrid[:h, :w]
    masks = []
    for inst in lf0.instances:
        pts = inst.numpy()
        cx, cy = np.nanmean(pts[:, 0]), np.nanmean(pts[:, 1])
        blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
        masks.append(sio.UserSegmentationMask.from_numpy(blob))
    seg_lf = sio.LabeledFrame(
        video=video, frame_idx=lf0.frame_idx, instances=lf0.instances, masks=masks
    )
    return sio.Labels(videos=[video], labeled_frames=[seg_lf])


@pytest.fixture
def minimal_instance_seg(minimal_instance, tmp_path):
    """Path to a synthetic instance-segmentation ``.pkg.slp`` (masks + image)."""
    seg_labels = make_seg_labels_from_slp(str(minimal_instance))
    out = Path(tmp_path) / "minimal_instance_seg.pkg.slp"
    seg_labels.save(out.as_posix(), embed=True)
    return out


@pytest.fixture
def minimal_instance(sleap_nn_data_dir):
    """Sleap fly .slp and video file paths."""
    return Path(sleap_nn_data_dir) / "minimal_instance.pkg.slp"


@pytest.fixture
def small_robot_minimal(sleap_nn_data_dir):
    """Sleap robot .slp path."""
    return Path(sleap_nn_data_dir) / "small_robot_minimal.slp"


@pytest.fixture
def small_robot_minimal_video(sleap_nn_data_dir):
    """Sleap robot .mp4 path."""
    return Path(sleap_nn_data_dir) / "small_robot.mp4"


@pytest.fixture
def centered_instance_video(sleap_nn_data_dir):
    """Sleap-io fly video .mp4 path."""
    return Path(sleap_nn_data_dir) / "centered_pair_small.mp4"


@pytest.fixture
def config(sleap_nn_data_dir):
    """Configuration for Sleap-NN data processing and model training."""
    init_config = OmegaConf.create(
        {
            "data_config": {
                "provider": "LabelsReader",
                "train_labels_path": [f"{sleap_nn_data_dir}/minimal_instance.pkg.slp"],
                "val_labels_path": [f"{sleap_nn_data_dir}/minimal_instance.pkg.slp"],
                "validation_fraction": 0.1,
                "test_file_path": None,
                "user_instances_only": True,
                "data_pipeline_fw": "torch_dataset",
                "cache_img_path": None,
                "use_existing_imgs": False,
                "delete_cache_imgs_after_training": True,
                "preprocessing": {
                    "ensure_rgb": False,
                    "ensure_grayscale": False,
                    "max_width": None,
                    "max_height": None,
                    "scale": 1.0,
                    "crop_size": 160,
                    "min_crop_size": None,
                },
                "use_augmentations_train": True,
                "augmentation_config": {
                    "intensity": {
                        "contrast_p": 1.0,
                    },
                    "geometric": {
                        "rotation_min": -180.0,
                        "rotation_max": 180.0,
                        "scale_min": 1.0,
                        "scale_max": 1.0,
                        "translate_width": 0,
                        "translate_height": 0,
                        "affine_p": 0.5,
                    },
                },
            },
            "model_config": {
                "init_weights": "default",
                "pretrained_backbone_weights": None,
                "pretrained_head_weights": None,
                "backbone_config": {
                    "unet": {
                        "in_channels": 1,
                        "kernel_size": 3,
                        "filters": 16,
                        "filters_rate": 1.5,
                        "max_stride": 8,
                        "convs_per_block": 2,
                        "stacks": 1,
                        "stem_stride": None,
                        "middle_block": True,
                        "up_interpolate": False,
                        "output_stride": 2,
                    }
                },
                "head_configs": {
                    "single_instance": None,
                    "centroid": None,
                    "bottomup": None,
                    "centered_instance": {
                        "confmaps": {
                            "part_names": [
                                "A",
                                "B",
                            ],
                            "anchor_part": "A",
                            "sigma": 1.5,
                            "output_stride": 2,
                        }
                    },
                },
            },
            "trainer_config": {
                "train_data_loader": {
                    "batch_size": 1,
                    "shuffle": True,
                    "num_workers": 0,
                },
                "val_data_loader": {
                    "batch_size": 1,
                    "num_workers": 0,
                },
                "model_ckpt": {
                    "save_top_k": 1,
                    "save_last": True,
                },
                "early_stopping": {
                    "stop_training_on_plateau": True,
                    "min_delta": 1e-08,
                    "patience": 20,
                },
                "trainer_devices": 1,
                "trainer_device_indices": None,
                "trainer_accelerator": "auto",
                "enable_progress_bar": False,
                "min_train_steps_per_epoch": 2,
                "train_steps_per_epoch": None,
                "max_epochs": 2,
                "seed": 1000,
                "keep_viz": True,
                "use_wandb": False,
                "save_ckpt": False,
                "ckpt_dir": ".",
                "run_name": None,
                "resume_ckpt_path": None,
                "wandb": {
                    "entity": None,
                    "project": "test",
                    "name": "test_run",
                    "wandb_mode": "offline",
                    "save_viz_imgs_wandb": True,
                    "api_key": "",
                    "prv_runid": None,
                    "group": None,
                },
                "optimizer_name": "Adam",
                "optimizer": {"lr": 0.0001, "amsgrad": False},
                "lr_scheduler": {
                    "reduce_lr_on_plateau": {
                        "threshold": 1e-07,
                        "threshold_mode": "rel",
                        "cooldown": 3,
                        "patience": 5,
                        "factor": 0.5,
                        "min_lr": 1e-08,
                    },
                },
                "online_hard_keypoint_mining": {
                    "online_mining": False,
                    "hard_to_easy_ratio": 2.0,
                    "min_hard_keypoints": 2,
                    "max_hard_keypoints": None,
                    "loss_scale": 5.0,
                },
            },
        }
    )
    return init_config
