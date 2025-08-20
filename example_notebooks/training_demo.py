# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imageio==2.37.0",
#     "ipython==9.4.0",
#     "kornia==0.8.1",
#     "marimo",
#     "matplotlib==3.9.4",
#     "numpy==2.0.2",
#     "omegaconf==2.3.0",
#     "opencv-python==4.12.0.88",
#     "pillow==11.3.0",
#     "seaborn==0.13.2",
#     "sleap-io==0.4.1",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "zmq==0.0.0",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This tutorial notebook walks through creating a config file, run training, inference, and evaluation worlflows in sleap-nn using higher-level APIs. (See docs for details on how to use our CLI).

    **_Note_**: This tutorial runs on CPU by default (or MPS on macOS). CUDA libraries are intentionally not included in the notebook’s dependencies.
    """
    )
    return


@app.cell
def _():
    # until we have sleap-nn pip pkg
    # ! pip install git+https://github.com/talmolab/sleap-nn.git
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("Download sample data..."),
            mo.download(
                "https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/train.pkg.slp",
                label="Train slp file",
                filename="./train.pkg.slp",
            ),
            mo.download(
                "https://storage.googleapis.com/sleap-data/datasets/BermanFlies/random_split1/val.pkg.slp",
                label="Val slp file",
                filename="./val.pkg.slp",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If you already have `.slp` files, provide the paths below.""")
    return


@app.cell
def _():
    path_to_train_slp_file = "train.pkg.slp"
    path_to_val_slp_file = "val.pkg.slp"
    return path_to_train_slp_file, path_to_val_slp_file


@app.cell(hide_code=True)
def _(mo):
    model_type = mo.ui.radio(
        options=["single_instance", "centroid", "centered_instance", "bottomup"],
        label="Choose the model type you want to train!",
    )
    model_type
    return (model_type,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Set-up config""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The first step in training is setting up the configuration. You can either start from one of the sample YAMLs in the repo’s sample_configs and edit it, or build the config programmatically. In this tutorial, we’ll take the functional route: compose each section (`data_config`, `model_config`, `trainer_config`) using handy functions and then create an Omegaconf config."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""First, we set-up the data config using `get_data_config()` function which has a set of defaults, and could be modified if required."""
    )
    return


@app.cell
def _(get_data_config, path_to_train_slp_file, path_to_val_slp_file):
    data_config = get_data_config(
        train_labels_path=[path_to_train_slp_file],
        val_labels_path=[path_to_val_slp_file],
        data_pipeline_fw="torch_dataset_cache_img_memory",
        use_augmentations_train=True,
        intensity_aug=["brightness"],
        geometry_aug=["rotation", "scale"],
        crop_hw=[100, 100],  # patch fix
    )
    return (data_config,)


@app.cell
def _(data_config):
    # modify the defaults (if required)

    data_config.augmentation_config.intensity.brightness_min = 0.9
    data_config.augmentation_config.intensity.brightness_max = 1.1
    return


@app.cell(hide_code=True)
def _(OmegaConf, data_config):
    print("Data Config: ")
    print("===========================")
    print(OmegaConf.to_yaml(data_config, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Next, we set-up the model config using `get_model_config()` function which sets up the parameters for building the model."""
    )
    return


@app.cell
def _(get_model_config, model_type):
    model_config = get_model_config(
        init_weight="xavier", backbone_config="unet", head_configs=f"{model_type.value}"
    )

    # modify the defaults (if required)
    model_config.backbone_config.unet.filters = 16
    return (model_config,)


@app.cell(hide_code=True)
def _(OmegaConf, model_config):
    print("Model Config: ")
    print("===========================")
    print(OmegaConf.to_yaml(model_config, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Next, we set-up the trainer config using `get_trainer_config()` function which has a set of defaults for setting up the hyperparameters for training, which could be modified if needed."""
    )
    return


@app.cell
def _(get_trainer_config, model_type):
    trainer_config = get_trainer_config(
        batch_size=4,
        num_workers=2,
        trainer_num_devices=1,
        shuffle_train=True,
        learning_rate=1e-4,
        save_ckpt=True,
        save_ckpt_path=f"{model_type.value}_training",
        lr_scheduler="reduce_lr_on_plateau",
    )

    trainer_config.max_epochs = 10
    return (trainer_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""**_Note_**: If you want to visualize the model training in [WandB](https://wandb.ai), set the following parameters:"""
    )
    return


@app.cell
def _():
    # trainer_config.use_wandb = True
    # trainer_config.wandb.entity = "<wandb entity name>"
    # trainer_config.wandb.project = "<wandb project name>"
    # trainer_config.wandb.name =  "<wandb run name>"
    # trainer_config.wandb.api_key = "<wandb API key>" # this is required to login to your account
    # trainer_config.wandb.group = "<wandb run group name>"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""**_Note_**: If you’re not using caching (memory/disk; see `data_config.data_pipeline_fw`) and your dataset/transforms aren’t picklable, set num_workers=0 on Windows/macOS (they use `spawn`). On Linux (default `fork`), multiple workers are typically safe."""
    )
    return


@app.cell(hide_code=True)
def _(OmegaConf, trainer_config):
    print("Trainer Config: ")
    print("===========================")
    print(OmegaConf.to_yaml(trainer_config, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Using the above initialized config classes, create a `TrainingJobConfig` instance, which could then be converted to a `OmegaConf` object."""
    )
    return


@app.cell
def _(TrainingJobConfig, data_config, model_config, trainer_config):
    # Create TrainingJobConfig

    training_job_config = TrainingJobConfig(
        data_config=data_config,
        model_config=model_config,
        trainer_config=trainer_config,
    )

    # Convert to omegaconf objects

    sleap_nn_cfg = training_job_config.to_sleap_nn_cfg()  # validates config structure
    return (sleap_nn_cfg,)


@app.cell(hide_code=True)
def _(OmegaConf, sleap_nn_cfg):
    print("Config: ")
    print("===========================")
    print(OmegaConf.to_yaml(sleap_nn_cfg, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Run training""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Create an instance of the `ModelTrainer` class by passing the config to the `get_model_trainer_from_config` method."""
    )
    return


@app.cell
def _(ModelTrainer, sleap_nn_cfg):
    model_trainer = ModelTrainer.get_model_trainer_from_config(sleap_nn_cfg)
    return (model_trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""The `get_model_trainer_from_config` method does the training setup by calling dataset preparation methods to establish training and validation labels (automatically splitting training data for validation if needed) and then invoking `_setup_config()` to process the loaded labels and automatically populate all configuration fields that were initially `None`. This includes computing `max_height` and `max_width` from actual image dimensions in the `sio.Labels` files, extracting skeletons from the labels data structure, and calculating other derived parameters based on the actual data characteristics. The method essentially transforms a minimal configuration into a complete configuration and ensures all required fields are populated and consistent before training begins, allowing users to start with basic parameters while the system automatically handles the complex configuration details."""
    )
    return


@app.cell(hide_code=True)
def _(OmegaConf, model_trainer):
    print("Config after `_setup_config()`: ")
    print("===========================")
    print(OmegaConf.to_yaml(model_trainer.config, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Let's create the training and validation dataloaders using `model_trainer.train_labels`, `model_trainer.val_labels`, and `model_trainer.config`. These attributes are initialized when you call `get_model_trainer_from_config()`."""
    )
    return


@app.cell
def _(get_train_val_dataloaders, get_train_val_datasets, model_trainer):
    # get dataloaders

    train_dataset, val_dataset = get_train_val_datasets(
        train_labels=model_trainer.train_labels,
        val_labels=model_trainer.val_labels,
        config=model_trainer.config,
    )
    train_dataloader, val_dataloader = get_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=model_trainer.config,
        trainer_devices=model_trainer.config.trainer_config.trainer_devices,
    )
    return (train_dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's visualize some of the sample images from the training labels!""")
    return


@app.cell(hide_code=True)
def _(model_type, plt, torch, train_dataloader):
    # visualize samples in dataloader

    # Get sample data
    sample = next(iter(train_dataloader))
    img_key = "image" if model_type.value != "centered_instance" else "instance_image"
    instance_key = (
        "instances"
        if model_type.value == "single_instance" or model_type.value == "bottomup"
        else None
    )
    if instance_key is None:
        instance_key = "centroids" if model_type.value == "centroid" else "instance"

    # Print sample info
    print("Sample keys and shapes:")
    print("=" * 50)
    for key in sample:
        print(f"`{key}` shape: {sample[key].shape} dtype: {sample[key].dtype}")

    batch_size = sample[img_key].shape[0]
    print(f"\n Batch Visualization - {model_type.value} model")

    n_cols = min(4, batch_size)  # Max 4 columns for readability
    n_rows = (batch_size + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if batch_size == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()

    # Plot each image in the batch
    for batch_idx in range(batch_size):
        ax = axes[batch_idx]

        img = sample[img_key][batch_idx, 0].numpy().transpose(1, 2, 0)
        cmap = "gray"

        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Batch {batch_idx}")
        ax.axis("off")

        # Add keypoints/centroids
        if instance_key in sample:
            pts = sample[instance_key][batch_idx, 0]

            if instance_key == "instances":
                for inst_idx, pt in enumerate(pts):
                    if not torch.isnan(pt).all():  # Check if instance is valid
                        ax.plot(
                            pt[:, 0],
                            pt[:, 1],
                            "go",
                            markersize=4,
                            alpha=0.8,
                            label=f"GT Instances" if inst_idx == 0 else "",
                        )
            elif instance_key == "centroids":
                # Plot centroids
                if not torch.isnan(pts).all():
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        "go",
                        markersize=6,
                        alpha=0.8,
                        label="Centroids",
                    )
            else:
                if not torch.isnan(pts).all():
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        "go",
                        markersize=5,
                        alpha=0.8,
                        label="Instances",
                    )

        if batch_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    for _idx in range(batch_size, len(axes)):
        fig.delaxes(axes[_idx])

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""We can instantiate the `LightningModule` from the config by calling `get_lightning_model_from_config`."""
    )
    return


@app.cell
def _(LightningModel, model_trainer):
    # create lightning model from config

    lightning_model = LightningModel.get_lightning_model_from_config(
        model_trainer.config
    )
    return (lightning_model,)


@app.cell(hide_code=True)
def _(lightning_model):
    lightning_model.model
    return


@app.cell(hide_code=True)
def _(lightning_model, mo):
    mo.md(
        f"""Total number of parameters: {sum(p.numel() for p in lightning_model.parameters())}"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Next to start the training process, we call the `train()` method of the `ModelTrainer` class. The `train()` method internally handles the complete training pipeline by automatically creating and configuring all necessary components, including dataloaders and Lightning modules. After creating a ModelTrainer instance using the `get_model_trainer_from_config` function, directly call this `train` method to initiate the entire training process without needing to manually set up individual components."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    run_train = mo.ui.run_button(label="Run training!")
    run_train
    return (run_train,)


@app.cell
def _(mo, model_trainer, run_train):
    # Call the `train` method to start training the model!

    if not run_train.value:
        mo.stop("Click `Run training!` to start.")
    model_trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Run inference / Get evaluation metrics""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Once we have the checkpoints, we can run inference on either a `.slp` file or a `.mp4` with the trained model."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    run_inf = mo.ui.run_button(label="Run Inference!")
    run_inf
    return (run_inf,)


@app.cell
def _(
    mo,
    model_type,
    path_to_val_slp_file,
    run_inf,
    run_inference,
    sleap_nn_cfg,
):
    # Running inference on val dataset

    if not run_inf.value:
        mo.stop("Click `Run Inference!` to start.")

    pred_labels = run_inference(
        data_path=path_to_val_slp_file,
        model_paths=[sleap_nn_cfg.trainer_config.save_ckpt_path],
        output_path=f"predictions_{model_type.value}.slp",
    )
    return (pred_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Evaluate the model against ground truth and compute metrics. (Make sure gt_labels contains ground-truth annotations.)"""
    )
    return


@app.cell
def _(Evaluator, path_to_val_slp_file, pred_labels, sio):
    # get eval metrics
    gt_labels = sio.load_slp(path_to_val_slp_file)

    evaluator = Evaluator(
        ground_truth_instances=gt_labels,
        predicted_instances=pred_labels,
    )

    metrics = evaluator.evaluate()

    print(f"Evaluation metrics:")
    print(f"OKS mAP: {metrics['voc_metrics']['oks_voc.mAP']}")
    print(f"Dist p90: {metrics['distance_metrics']['p90']}")
    return (gt_labels,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""**_Note (for centroid-only inference)_**: The centroid model is essentially the first stage of TopDown model workflow, which only predicts centers, not keypoints. In centroid-only inference, each predicted centroid is matched (by Euclidean distance) to the nearest ground-truth instance, and the ground-truth keypoints are copied for display. Therefore, an OKS mAP of 1.0 just means all instances were detected—it does not reflect pose/keypoint accuracy. To evaluate keypoints, run the second stage (the pose model) rather than centroid-only inference."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Sample predictions:""")
    return


@app.cell(hide_code=True)
def _(cv2, mo, np, plt, random):
    import matplotlib.animation as animation
    from pathlib import Path
    import imageio

    def plot_preds_gif(
        gt_labels, pred_labels, num_frames=20, frame_duration=500, random_seed=42
    ):
        """Create a GIF animation comparing ground truth vs predictions over random frames."""

        # Set random seed for reproducible frame selection
        random.seed(random_seed)

        # Randomly sample frames
        selected_frames = random.sample(list(range(len(pred_labels))), num_frames)

        # Create frames for GIF
        frames = []

        for frame_idx in range(num_frames):
            # Create figure for this frame
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # Get the labeled frame
            lf_idx = selected_frames[frame_idx]
            gt_lf = gt_labels[lf_idx]
            pred_lf = pred_labels[lf_idx]

            # Ensure we're plotting keypoints for the same frame
            assert (
                gt_lf.frame_idx == pred_lf.frame_idx
            ), f"Frame mismatch at {lf_idx}: GT={gt_lf.frame_idx}, Pred={pred_lf.frame_idx}"

            # Plot image
            ax.imshow(gt_lf.image, cmap="gray")

            # Plot ground truth instances
            for idx, instance in enumerate(gt_lf.instances):
                if not instance.is_empty:
                    gt_pts = instance.numpy()
                    ax.plot(
                        gt_pts[:, 0],
                        gt_pts[:, 1],
                        "go",
                        markersize=8,
                        alpha=0.8,
                        label="Ground Truth" if idx == 0 else "",
                    )

            # Plot predicted instances
            for idx, instance in enumerate(pred_lf.instances):
                if not instance.is_empty:
                    pred_pts = instance.numpy()
                    ax.plot(
                        pred_pts[:, 0],
                        pred_pts[:, 1],
                        "rx",
                        markersize=8,
                        alpha=0.8,
                        label="Predictions" if idx == 0 else "",
                    )

            # Add legend
            ax.legend(loc="upper right", fontsize=10)
            ax.axis("off")

            # Add overall title
            fig.suptitle(
                f"Ground Truth vs Predictions Animation ({num_frames} frames, {frame_duration}ms per frame)",
                fontsize=10,
                fontweight="bold",
            )

            plt.tight_layout()

            # Convert plot to image array
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frames.append(img_array)
            plt.close(fig)  # Close to free memory

        # Save as MP4 video
        output_path = f"gt_vs_pred_animation_{random_seed}.mp4"

        # Get video dimensions from first frame
        height, width = frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 1000 / frame_duration  # Convert ms to fps
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames to video
        for frame in frames:
            # Convert RGB to BGR (OpenCV expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()

        return mo.md(
            f"""
        ## 🎬 Animation Created!
    
        **Frames:** {num_frames} randomly selected frames  
        **Duration:** {frame_duration}ms per frame  
        **Random Seed:** {random_seed}  
        **Total Animation Time:** {(num_frames * frame_duration) / 1000:.1f} seconds
    
        **File saved as:** `{output_path}`
    
        You can now download and view the GIF file!
        """
        )

    return (plot_preds_gif,)


@app.cell(hide_code=True)
def _(gt_labels, mo, plot_preds_gif, pred_labels):
    random_seed = 42

    # Create the animation
    plot_preds_gif(
        gt_labels,
        pred_labels,
        num_frames=20,
        frame_duration=200,
        random_seed=random_seed,
    )

    mo.video(f"gt_vs_pred_animation_{random_seed}.mp4")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To view predictions of a certain frame:""")
    return


@app.cell(hide_code=True)
def _(mo):
    lf_index = mo.ui.number(start=0, stop=100, label="LF index")
    return (lf_index,)


@app.cell
def _(lf_index, mo):
    mo.hstack([lf_index, mo.md(f"Has value: {lf_index.value}")])
    return


@app.cell(hide_code=True)
def _(gt_labels, lf_index, plt, pred_labels):
    _fig, _ax = plt.subplots(1, 1, figsize=(5 * 1, 5 * 1))

    # Plot each frame
    gt_lf = gt_labels[lf_index.value]
    pred_lf = pred_labels[lf_index.value]

    # Ensure we're plotting keypoints for the same frame
    assert (
        gt_lf.frame_idx == pred_lf.frame_idx
    ), f"Frame mismatch at {lf_index.value}: GT={gt_lf.frame_idx}, Pred={pred_lf.frame_idx}"

    _ax.imshow(gt_lf.image, cmap="gray")
    _ax.set_title(
        f"Frame {gt_lf.frame_idx} (lf idx: {lf_index.value})",
        fontsize=12,
        fontweight="bold",
    )

    # Plot ground truth instances
    for idx, instance in enumerate(gt_lf.instances):
        if not instance.is_empty:
            gt_pts = instance.numpy()
            _ax.plot(
                gt_pts[:, 0],
                gt_pts[:, 1],
                "go",
                markersize=6,
                alpha=0.8,
                label="GT" if idx == 0 else "",
            )

    # Plot predicted instances
    for idx, instance in enumerate(pred_lf.instances):
        if not instance.is_empty:
            pred_pts = instance.numpy()
            _ax.plot(
                pred_pts[:, 0],
                pred_pts[:, 1],
                "rx",
                markersize=6,
                alpha=0.8,
                label="Pred" if idx == 0 else "",
            )

    # Add legend
    _ax.legend(loc="upper right", fontsize=8)

    _ax.axis("off")

    plt.suptitle(f"Ground Truth vs Predictions", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import cv2
    import torch
    import pprint
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torchvision import transforms

    from omegaconf import OmegaConf

    import sleap_io as sio
    import random

    from sleap_nn.architectures.model import Model
    from sleap_nn.training.model_trainer import ModelTrainer

    from sleap_nn.config.get_config import (
        get_data_config,
        get_head_configs,
        get_model_config,
        get_trainer_config,
    )
    from sleap_nn.config.training_job_config import TrainingJobConfig

    from sleap_nn.data.custom_datasets import (
        get_train_val_dataloaders,
        get_train_val_datasets,
    )

    from sleap_nn.training.lightning_modules import LightningModel

    from sleap_nn.predict import run_inference
    from sleap_nn.evaluation import Evaluator

    torch.set_default_dtype(torch.float32)
    return (
        Evaluator,
        LightningModel,
        ModelTrainer,
        OmegaConf,
        TrainingJobConfig,
        cv2,
        get_data_config,
        get_model_config,
        get_train_val_dataloaders,
        get_train_val_datasets,
        get_trainer_config,
        mo,
        np,
        plt,
        random,
        run_inference,
        sio,
        torch,
    )


if __name__ == "__main__":
    app.run()
