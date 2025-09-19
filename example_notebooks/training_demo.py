# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "imageio==2.37.0",
#     "ipython==9.4.0",
#     "kornia==0.8.1",
#     "marimo",
#     "matplotlib==3.10.6",
#     "numpy==2.3.3",
#     "omegaconf==2.3.0",
#     "opencv-python==4.11.0.86",
#     "pillow==11.3.0",
#     "seaborn==0.13.2",
#     "sleap-io>=0.5.3",
#     "sleap-nn>=0.0.1",
#     "torch==2.7.1",
#     "torchvision==0.22.1",
#     "zmq==0.0.0",
# ]
# ///

import marimo

__generated_with = "0.15.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # import all necessary modules

    import marimo as mo
    import cv2
    import torch
    import pprint
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torchvision import transforms
    from pathlib import Path
    import imageio.v3 as iio

    import matplotlib.animation as animation
    import imageio

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
        Path,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    This tutorial notebook walks through creating a config file, run training, inference, and evaluation worlflows in sleap-nn using higher-level APIs. (See docs for details on how to use our CLI).

    **_Note_**: This tutorial runs on CPU by default (or MPS on macOS). CUDA libraries are intentionally not included in the notebook’s dependencies.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    **_Tips on using marimo:
    _**

    Marimo notebooks are designed for a seamless, automated workflow. After you select the model type, all cells will execute automatically—no need to run them one by one. However, training and inference will start only when you click the **Run Training**/ **Run Inference** button, giving you full control over when to begin model training or run inference.

    If you want to tweak values in a specific cell, edit the cell and click its yellow highlighted Run button on the right of the cell block; that cell will execute, and Marimo will automatically re-run only the downstream cells that depend on it, leaving unrelated cells unchanged. If you need a full refresh, use highlighted `Run all` button in the bottom right corner!
    """
    )
    return


@app.cell
def _():
    # until we have sleap-nn pip pkg:
    # In the manage packages tab to the left, add `git+https://github.com/talmolab/sleap-nn.git` dependency!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("Download sample data and move them to your current working dir..."),
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
    mo.md(
        r"""
    If you already have `.slp` files to work with, modify
    the paths below.
    """
    )
    return


@app.cell
def _():
    path_to_train_slp_file = "train.pkg.slp"
    path_to_val_slp_file = "val.pkg.slp"
    return path_to_train_slp_file, path_to_val_slp_file


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""#### Choose the model type you want to train! (To start simple, you could choose single-instance)"""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    model_type = mo.ui.radio(
        options=["single_instance", "centroid", "centered_instance", "bottomup"],
        value="single_instance",
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
        r"""The first step in training is setting up the configuration. You can either start from one of the sample YAMLs in the repo and edit it, or build the config programmatically. In this tutorial, we’ll take the functional route: compose each section (`data_config`, `model_config`, `trainer_config`) using handy functions and then create an Omegaconf config."""
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
    )
    return (data_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's see how the `data_config` section looks like:""")
    return


@app.cell(hide_code=True)
def _(OmegaConf, data_config):
    print("Data Config: ")
    print("===========================")
    print(OmegaConf.to_yaml(data_config, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If required, we could also modify the default values as given below:""")
    return


@app.cell
def _(data_config):
    data_config.augmentation_config.intensity.brightness_min = 0.9
    data_config.augmentation_config.intensity.brightness_max = 1.1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Next, we set-up the model config using `get_model_config()` function which sets up the parameters for building the model. We will be using the `unet` model as the backbone here and head config would be updated based on the model type you chose before!"""
    )
    return


@app.cell
def _(get_model_config, model_type):
    model_config = get_model_config(
        init_weight="xavier", backbone_config="unet", head_configs=f"{model_type.value}"
    )
    return (model_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's print and see how the `model_config` looks like:""")
    return


@app.cell(hide_code=True)
def _(OmegaConf, model_config):
    print("Model Config: ")
    print("===========================")
    print(OmegaConf.to_yaml(model_config, resolve=True, sort_keys=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""If required, we could modify the default values as given below:""")
    return


@app.cell
def _(model_config):
    model_config.backbone_config.unet.filters = 16
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
        max_epochs=10,
        ckpt_dir=".",
        run_name=f"{model_type.value}_training",
        lr_scheduler="reduce_lr_on_plateau",
    )
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
    # trainer_config.wandb.save_viz_imgs_wandb = False
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
def _(mo):
    mo.md(r"""Let's look into the generated `trainer_config`:""")
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
def _(mo):
    mo.md(
        r"""The entire configuration that would be given as input to the training modules:"""
    )
    return


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
        r"""The `get_model_trainer_from_config` method does the training setup by calling dataset preparation methods to establish training and validation labels (automatically splitting training data for validation if needed) and then invoking `setup_config()` to process the loaded labels and automatically populate all configuration fields that were initially `None`. This includes computing `max_height` and `max_width` from actual image dimensions in the `sio.Labels` files, extracting skeletons from the labels data structure, and calculating other derived parameters based on the actual data characteristics. The method essentially transforms a minimal configuration into a complete configuration and ensures all required fields are populated and consistent before training begins, allowing users to start with basic parameters while the system automatically handles the complex configuration details."""
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""Let's take a look at the config now, which now has all fields set (esp. `data_config.preprocessing.max_width`, `data_config.preprocessing.max_width` and `data_config.skeletons`)"""
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
def _(mo):
    mo.md(r"""Let's take a look at the model created from our config:""")
    return


@app.cell(hide_code=True)
def _(lightning_model):
    lightning_model.model
    return


@app.cell(hide_code=True)
def _(lightning_model, mo):
    mo.md(
        f"""#### Total number of parameters: {sum(p.numel() for p in lightning_model.parameters())}"""
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
    if not run_train.value:
        mo.stop("Click `Run training!` to start.")

    # Call the `train` method to start training the model!
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
    Path,
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
        model_paths=[
            (
                Path(sleap_nn_cfg.trainer_config.ckpt_dir)
                / sleap_nn_cfg.trainer_config.run_name
            ).as_posix()
        ],
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
    def plot_preds_video(
        gt_labels,
        pred_labels,
        num_frames=20,
        frame_duration=500,  # ms per frame
        random_seed=42,
        output_path=None,
    ):
        """Create an MP4 comparing ground truth vs predictions over random frames."""
        assert len(pred_labels) == len(
            gt_labels
        ), "GT and predictions must be the same length."

        # Don't sample more frames than available
        num_frames = min(num_frames, len(pred_labels))
        if num_frames == 0:
            raise ValueError("No frames available to plot.")

        random.seed(random_seed)
        selected_frames = random.sample(range(len(pred_labels)), num_frames)

        frames_bgr = []

        for i in range(num_frames):
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)  # dpi sets pixel size

            lf_idx = selected_frames[i]
            gt_lf = gt_labels[lf_idx]
            pred_lf = pred_labels[lf_idx]

            # Sanity check: same underlying frame
            assert (
                gt_lf.frame_idx == pred_lf.frame_idx
            ), f"Frame mismatch at {lf_idx}: GT={gt_lf.frame_idx}, Pred={pred_lf.frame_idx}"

            # Background image
            ax.imshow(
                getattr(gt_lf, "image", None), cmap="gray", interpolation="nearest"
            )

            # Ground-truth keypoints
            for k, inst in enumerate(getattr(gt_lf, "instances", [])):
                if not inst.is_empty:
                    pts = inst.numpy()
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        "go",
                        markersize=8,
                        alpha=0.8,
                        label="Ground Truth" if k == 0 else None,
                    )

            # Predicted keypoints
            for k, inst in enumerate(getattr(pred_lf, "instances", [])):
                if not inst.is_empty:
                    pts = inst.numpy()
                    ax.plot(
                        pts[:, 0],
                        pts[:, 1],
                        "rx",
                        markersize=8,
                        alpha=0.8,
                        label="Predictions" if k == 0 else None,
                    )

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc="upper right", fontsize=10)

            ax.axis("off")
            fig.suptitle(
                f"Ground Truth vs Predictions ({num_frames} frames, {frame_duration} ms/frame)",
                fontsize=10,
                fontweight="bold",
            )
            fig.tight_layout()

            # --- Safe pixel grab on Agg: use buffer_rgba(), then drop alpha
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # RGBA bytes
            rgba = buf.reshape(h, w, 4)
            img_rgb = rgba[..., :3]  # drop alpha

            # OpenCV expects BGR
            frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            frames_bgr.append(frame_bgr)

            plt.close(fig)

        # Output path
        if output_path is None:
            output_path = f"gt_vs_pred_animation_{random_seed}.mp4"

        # Write MP4 (OpenCV wants (width, height))
        height, width = frames_bgr[0].shape[:2]
        fps = 1000.0 / frame_duration
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for f in frames_bgr:
            if f.dtype != np.uint8:
                f = f.astype(np.uint8)
            if f.ndim != 3 or f.shape[2] != 3:
                raise ValueError(f"Unexpected frame shape {f.shape}")
            out.write(f)
        out.release()

        total_secs = (num_frames * frame_duration) / 1000.0
        return mo.md(
            f"""
    ## 🎬 Video Created!

    **Frames:** {num_frames} randomly selected  
    **Duration:** {frame_duration} ms per frame  
    **Random Seed:** {random_seed}  
    **Total Time:** {total_secs:.1f} s

    **Saved as:** `{output_path}`
    """
        )

    return (plot_preds_video,)


@app.cell(hide_code=True)
def _(gt_labels, mo, plot_preds_video, pred_labels):
    random_seed = 42

    # Create the animation
    plot_preds_video(
        gt_labels,
        pred_labels,
        num_frames=20,
        frame_duration=200,
        random_seed=random_seed,
    )

    # mo.video(f"./gt_vs_pred_animation_{random_seed}.mp4",
    #         autoplay=True,
    #     loop=True,
    #     controls=True,
    #     muted=True,
    #     width=800,
    #         height=800)

    mo.video(
        f"gt_vs_pred_animation_{random_seed}.mp4",
        autoplay=True,
        loop=True,
        controls=True,
        muted=True,
    )
    return


@app.cell(hide_code=True)
def _(mo, pred_labels):
    lf_index = mo.ui.number(start=0, stop=len(pred_labels) - 1, label="LF index")
    return (lf_index,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""To view predictions of a certain frame:""")
    return


@app.cell(hide_code=True)
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


if __name__ == "__main__":
    app.run()
