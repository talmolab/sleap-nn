# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo",
#     "sleap-nn @ git+https://github.com/talmolab/sleap-nn.git",
#     "torch",
#     "torchvision",
#     "imageio-ffmpeg",
#     "matplotlib",
# ]
#
# # GPU on molab: turn it on with the notebook-specs button in the app header, then
# # run this notebook — it picks up the GPU automatically (no code changes).
# #
# # Why pin a CUDA build of PyTorch? molab's GPU is an NVIDIA RTX Pro 6000
# # (Blackwell), and the default PyPI torch wheel ships no Blackwell kernels. We
# # therefore install torch/torchvision from PyTorch's CUDA 13.0 index on
# # Linux/Windows; macOS falls back to the default wheel (CPU/MPS). On a machine
# # with no NVIDIA GPU the cu130 wheel still runs fine on CPU. Older GPU drivers
# # that predate CUDA 13 can switch these URLs to `.../whl/cu128`.
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# torchvision = [{ index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" }]
# ///
#
# Tip: pin sleap-nn to a tag/commit for reproducibility, e.g.
#   "sleap-nn @ git+https://github.com/talmolab/sleap-nn.git@v0.2.0"

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        # 🪰 SLEAP-NN — top-down training, evaluation & tracking (end-to-end)

        Trains a **top-down** pose model on the **flies13** (`wt_gold.13pt`) dataset —
        two flies per frame, a 13-node skeleton — and runs it on a fresh video **with
        tracking**. A top-down model is two networks: a **centroid** model that locates
        each fly (anchored on the `thorax`), and a **centered-instance** model that
        predicts the 13-node skeleton inside the crop around each centroid.

        This notebook runs **straight through** — download → build configs (saved to
        YAML) → train → evaluate on the test split → tracked inference on a clip →
        render. A GPU is strongly recommended (this is full-resolution 1024×1024 data);
        on molab, enable the GPU from the notebook-specs button.
        """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import shutil
    import urllib.request
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    from omegaconf import OmegaConf

    import sleap_io as sio

    from sleap_nn.config.get_config import (
        get_data_config,
        get_model_config,
        get_trainer_config,
    )
    from sleap_nn.config.training_job_config import TrainingJobConfig
    from sleap_nn.train import run_training
    from sleap_nn.inference import predict
    from sleap_nn.inference.tracking import TrackerConfig
    from sleap_nn.evaluation import Evaluator

    return (
        Evaluator,
        OmegaConf,
        Path,
        TrackerConfig,
        TrainingJobConfig,
        get_data_config,
        get_model_config,
        get_trainer_config,
        mo,
        plt,
        predict,
        run_training,
        shutil,
        sio,
        torch,
        urllib,
    )


@app.cell(hide_code=True)
def _(mo, torch):
    if torch.cuda.is_available():
        device, device_name = "cuda", torch.cuda.get_device_name(0)
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device, device_name = "mps", "Apple MPS"
    else:
        device, device_name = "cpu", "CPU"
    mo.md(f"**Using `{device}`** — {device_name}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 1. Download the dataset

        **flies13** (`wt_gold.13pt`, `tracking_split2`) — `1024×1024` grayscale frames
        with two flies each, as `.pkg.slp` files (labels with embedded images) — plus a
        clip for inference.
        """)
    return


@app.cell
def _(Path, urllib):
    if not Path("train.pkg.slp").exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/tracking_split2/train.pkg.slp",
            "train.pkg.slp",
        )
    if not Path("val.pkg.slp").exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/tracking_split2/val.pkg.slp",
            "val.pkg.slp",
        )
    if not Path("test.pkg.slp").exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/tracking_split2/test.pkg.slp",
            "test.pkg.slp",
        )
    if not Path("fly_clip.mp4").exists():
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt/clips/talk_title_slide%4013150-14500.mp4",
            "fly_clip.mp4",
        )
    train_slp, val_slp, test_slp, clip_mp4 = (
        "train.pkg.slp",
        "val.pkg.slp",
        "test.pkg.slp",
        "fly_clip.mp4",
    )
    return clip_mp4, test_slp, train_slp, val_slp


@app.cell(hide_code=True)
def _(mo, sio, test_slp, train_slp, val_slp):
    train_labels = sio.load_slp(train_slp)
    val_labels = sio.load_slp(val_slp)
    test_labels = sio.load_slp(test_slp)
    skeleton = train_labels.skeletons[0]
    _img = train_labels[0].image

    mo.md(f"""
        | | |
        |---|---|
        | Train / val / test frames | **{len(train_labels)} / {len(val_labels)} / {len(test_labels)}** |
        | Frame size | **{_img.shape[1]}×{_img.shape[0]}**, {_img.shape[2]} channel(s) |
        | Instances / frame | **{max(len(lf.instances) for lf in train_labels)}** (multi-animal → tracking) |
        | Skeleton | **{len(skeleton.nodes)}** nodes — {", ".join(n.name for n in skeleton.nodes)} |
        """)
    return skeleton, test_labels, train_labels


@app.cell(hide_code=True)
def _(plt, skeleton, train_labels):
    _fig, _axes = plt.subplots(1, 3, figsize=(12, 4))
    for _ax, _lf in zip(_axes, train_labels[:3]):
        _ax.imshow(_lf.image, cmap="gray")
        for _k, _inst in enumerate(_lf.instances):
            _pts = _inst.numpy()
            _col = ["lime", "red"][_k % 2]
            for _s, _d in skeleton.edge_inds:
                _ax.plot(
                    [_pts[_s, 0], _pts[_d, 0]],
                    [_pts[_s, 1], _pts[_d, 1]],
                    "-",
                    color=_col,
                    linewidth=1,
                    alpha=0.8,
                )
            _ax.scatter(_pts[:, 0], _pts[:, 1], c=_col, s=8, zorder=3)
        _ax.set_title(f"frame {_lf.frame_idx}", fontsize=9)
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 2. Build the model configs (→ YAML)

        Each model's config is composed with the `sleap_nn.config` builders and saved
        to a YAML file. Training then reads those YAMLs — the same path as
        `sleap-nn train configs/centroid.yaml`. Shared settings: **±180° rotation**
        augmentation, **`thorax`** anchor, **25 epochs**, initial LR **1e-4**, eval
        metrics logged each epoch.
        """)
    return


@app.cell
def _(
    OmegaConf,
    Path,
    TrainingJobConfig,
    get_data_config,
    get_model_config,
    get_trainer_config,
):
    # ── Centroid: locate each fly (input scale 0.5, stride 16→2, 16 filters ×2, σ3.5).
    centroid_cfg = TrainingJobConfig(
        data_config=get_data_config(
            train_labels_path=["train.pkg.slp"],
            val_labels_path=["val.pkg.slp"],
            data_pipeline_fw="torch_dataset_cache_img_memory",
            scale=0.5,
            use_augmentations_train=True,
            geometry_aug="rotation",
        ),
        model_config=get_model_config(backbone_config="unet", head_configs="centroid"),
        trainer_config=get_trainer_config(
            batch_size=4,
            num_workers=0,
            max_epochs=25,
            learning_rate=1e-4,
            lr_scheduler="reduce_lr_on_plateau",
            save_ckpt=True,
            ckpt_dir="models",
            run_name="centroid",
            visualize_preds_during_training=True,
            trainer_num_devices=1,
        ),
    ).to_sleap_nn_cfg()
    centroid_cfg.model_config.backbone_config.unet.filters = 16
    centroid_cfg.model_config.backbone_config.unet.filters_rate = 2.0
    centroid_cfg.model_config.backbone_config.unet.max_stride = 16
    centroid_cfg.model_config.backbone_config.unet.output_stride = 2
    centroid_cfg.model_config.head_configs.centroid.confmaps.sigma = 3.5
    centroid_cfg.model_config.head_configs.centroid.confmaps.anchor_part = "thorax"
    centroid_cfg.model_config.head_configs.centroid.confmaps.output_stride = 2
    centroid_cfg.data_config.augmentation_config.geometric.rotation_min = -180.0
    centroid_cfg.data_config.augmentation_config.geometric.rotation_max = 180.0
    centroid_cfg.data_config.augmentation_config.geometric.affine_p = 1.0
    centroid_cfg.trainer_config.eval.enabled = True

    # ── Centered-instance: 13-node skeleton per crop (crop 160, stride 32→4, 24 filters ×1.5, σ2.5).
    centered_cfg = TrainingJobConfig(
        data_config=get_data_config(
            train_labels_path=["train.pkg.slp"],
            val_labels_path=["val.pkg.slp"],
            data_pipeline_fw="torch_dataset_cache_img_memory",
            scale=1.0,
            crop_size=160,
            use_augmentations_train=True,
            geometry_aug="rotation",
        ),
        model_config=get_model_config(
            backbone_config="unet", head_configs="centered_instance"
        ),
        trainer_config=get_trainer_config(
            batch_size=8,
            num_workers=0,
            max_epochs=25,
            learning_rate=1e-4,
            lr_scheduler="reduce_lr_on_plateau",
            save_ckpt=True,
            ckpt_dir="models",
            run_name="centered_instance",
            visualize_preds_during_training=True,
            trainer_num_devices=1,
        ),
    ).to_sleap_nn_cfg()
    centered_cfg.model_config.backbone_config.unet.filters = 24
    centered_cfg.model_config.backbone_config.unet.filters_rate = 1.5
    centered_cfg.model_config.backbone_config.unet.max_stride = 32
    centered_cfg.model_config.backbone_config.unet.output_stride = 4
    centered_cfg.model_config.head_configs.centered_instance.confmaps.sigma = 2.5
    centered_cfg.model_config.head_configs.centered_instance.confmaps.anchor_part = (
        "thorax"
    )
    centered_cfg.model_config.head_configs.centered_instance.confmaps.output_stride = 4
    centered_cfg.data_config.augmentation_config.geometric.rotation_min = -180.0
    centered_cfg.data_config.augmentation_config.geometric.rotation_max = 180.0
    centered_cfg.data_config.augmentation_config.geometric.affine_p = 1.0
    centered_cfg.trainer_config.eval.enabled = True

    # To log to Weights & Biases (needs an API key), uncomment and fill in:
    # for _cfg in (centroid_cfg, centered_cfg):
    #     _cfg.trainer_config.use_wandb = True
    #     _cfg.trainer_config.wandb.entity = "<entity>"
    #     _cfg.trainer_config.wandb.project = "<project>"
    #     _cfg.trainer_config.wandb.api_key = "<key>"

    Path("configs").mkdir(exist_ok=True)
    OmegaConf.save(centroid_cfg, "configs/centroid.yaml")
    OmegaConf.save(centered_cfg, "configs/centered_instance.yaml")
    return centered_cfg, centroid_cfg


@app.cell(hide_code=True)
def _(OmegaConf, centered_cfg, centroid_cfg, mo):
    mo.accordion(
        {
            "📄 configs/centroid.yaml": mo.md(
                f"```yaml\n{OmegaConf.to_yaml(centroid_cfg, resolve=True)}\n```"
            ),
            "📄 configs/centered_instance.yaml": mo.md(
                f"```yaml\n{OmegaConf.to_yaml(centered_cfg, resolve=True)}\n```"
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 3. Train

        Trains both stages from their YAMLs via `run_training` (the function behind
        `sleap-nn train`). This is the slow step — minutes per model on a GPU.
        """)
    return


@app.cell
def _(OmegaConf, centered_cfg, centroid_cfg, run_training, shutil):
    # Reuse the canonical model dirs on re-run (sleap-nn would otherwise append `-2`).
    shutil.rmtree("models/centroid", ignore_errors=True)
    shutil.rmtree("models/centered_instance", ignore_errors=True)

    run_training(OmegaConf.load("configs/centroid.yaml"))
    run_training(OmegaConf.load("configs/centered_instance.yaml"))

    centroid_dir, centered_instance_dir = "models/centroid", "models/centered_instance"
    _ = (centroid_cfg, centered_cfg)  # depend on the config cell
    return centered_instance_dir, centroid_dir


@app.cell(hide_code=True)
def _(Path, centered_instance_dir, centroid_dir, mo, shutil):
    _ = (centroid_dir, centered_instance_dir)  # wait for training
    _zip = shutil.make_archive("trained_models", "zip", root_dir="models")
    mo.vstack(
        [
            mo.md(
                "### Download the trained models\n"
                "Each dir has `best.ckpt` + `training_config.yaml`. Run locally with "
                "`sleap-nn predict -i video.mp4 -m models/centroid/ "
                "-m models/centered_instance/ --tracking --max_instances 2 "
                "--candidates_method local_queues -o preds.slp`."
            ),
            mo.download(
                data=Path(_zip).read_bytes(),
                filename="trained_models.zip",
                label="⬇️ Download trained models (.zip)",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 4. Evaluate on the test split

        Run the trained top-down model on the held-out **test** labels and compute pose
        metrics (OKS mAP, PCK, keypoint distance) against ground truth.
        """)
    return


@app.cell
def _(Evaluator, centered_instance_dir, centroid_dir, device, mo, predict, test_labels):
    test_preds = predict(
        test_labels, model_paths=[centroid_dir, centered_instance_dir], device=device
    )
    metrics = Evaluator(
        ground_truth_instances=test_labels, predicted_instances=test_preds
    ).evaluate()

    mo.md(f"""
        | metric | value |
        |---|---|
        | OKS mAP | **{metrics['voc_metrics']['oks_voc.mAP']:.3f}** |
        | mPCK | **{metrics['pck_metrics']['mPCK']:.3f}** |
        | distance p50 | **{metrics['distance_metrics']['p50']:.2f}** px |
        | distance p90 | **{metrics['distance_metrics']['p90']:.2f}** px |
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 5. Tracked inference on a clip

        Run the model on a fresh clip **with tracking** so each fly keeps a consistent
        identity — capped at **2 instances** using the **local-queues** candidate method
        (CLI equivalent: `--tracking --max_instances 2 --candidates_method local_queues`).
        """)
    return


@app.cell
def _(
    TrackerConfig, centered_instance_dir, centroid_dir, clip_mp4, device, mo, predict
):
    clip_preds = predict(
        clip_mp4,
        model_paths=[centroid_dir, centered_instance_dir],
        device=device,
        frames=list(range(300)),  # first 300 frames; set to None for the whole clip
        max_instances=2,
        tracker_config=TrackerConfig(
            candidates_method="local_queues", max_tracks=2, window_size=5
        ),
        output_path="fly_clip.tracked.slp",
    )
    _n_tracks = len(
        {inst.track.name for lf in clip_preds for inst in lf.instances if inst.track}
    )
    mo.md(
        f"Tracked **{len(clip_preds)}** frames · "
        f"**{sum(len(lf.instances) for lf in clip_preds)}** instances · "
        f"**{_n_tracks}** track(s) → `fly_clip.tracked.slp`"
    )
    return (clip_preds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Render the tracked predictions (`sio.render_video`)""")
    return


@app.cell
def _(clip_preds, mo, sio):
    sio.render_video(
        clip_preds,
        "fly_clip.tracked.viz.mp4",
        fps=30,
        color_by="track",
        show_progress=False,
    )
    mo.video("fly_clip.tracked.viz.mp4", controls=True, loop=True, muted=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## Next steps

        - **Run on a cluster**: the saved `configs/*.yaml` train with
          `sleap-nn train configs/centroid.yaml` (then the centered-instance one).
        - **Tracking knobs**: tune `window_size` / `scoring_method`, or add a motion
          model (`--use_flow` / `--use_kalman`) — see the
          [tracking guide](https://nn.sleap.ai/guides/tracking/).
        - **Render options**: `sio.render_video(..., show_trails=True, trail_length=10)`
          for motion trails — see [io.sleap.ai](https://io.sleap.ai).
        """)
    return


if __name__ == "__main__":
    app.run()
