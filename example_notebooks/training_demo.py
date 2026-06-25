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

        This notebook trains a **top-down** pose model on the **flies13**
        (`wt_gold.13pt`) dataset — two flies per frame, a 13-node skeleton — and
        runs it on a fresh video **with tracking**.

        A top-down model is **two networks**:

        1. a **centroid** model that locates each fly (anchored on the `thorax`), and
        2. a **centered-instance** model that predicts the 13-node skeleton inside the
           crop around each centroid.

        We build each model's config, **save it to YAML**, train from the YAML
        (exactly what the `sleap-nn train config.yaml` CLI does), **evaluate on the
        held-out test split**, then run **tracked** inference on a clip and render the
        result with `sleap-io render`.

        > **Runs on molab.** Everything downloads itself. The heavy steps (training,
        > inference) are behind **Run** buttons. Uses the GPU automatically when one is
        > available — recommended here, since this is a full-resolution (1024×1024)
        > dataset.
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ### Tips on using marimo

        marimo runs cells **automatically** based on their dependencies. The only
        manual steps are the **Run** buttons, so the expensive work happens only when
        you ask for it. Edit a value (epochs, threshold) and marimo re-runs just the
        downstream cells that depend on it.
        """)
    return


@app.cell
def _():
    # Imports.
    import marimo as mo
    import urllib.request
    from pathlib import Path

    import matplotlib.pyplot as plt
    import torch
    from omegaconf import OmegaConf

    import sleap_io as sio

    # Config builders (compose programmatically, then save to YAML).
    from sleap_nn.config.get_config import (
        get_data_config,
        get_model_config,
        get_trainer_config,
    )
    from sleap_nn.config.training_job_config import TrainingJobConfig

    # Training entry point (consumes the same config the `sleap-nn train` CLI does).
    from sleap_nn.train import run_training

    # New inference API + tracker config.
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
        sio,
        torch,
        urllib,
    )


@app.cell
def _(mo):
    # `script` mode is `uv run training_demo.py` (used for testing / CI): every step
    # runs automatically with tiny settings. Interactively, the heavy steps wait for
    # the buttons.
    is_script_mode = mo.app_meta().mode == "script"
    is_script_mode
    return (is_script_mode,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 1. Compute device""")
    return


@app.cell
def _(mo, torch):
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = "mps"
        device_name = "Apple MPS"
    else:
        device = "cpu"
        device_name = "CPU"

    mo.md(f"**Using `{device}`** — {device_name}")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 2. Download the dataset

        **flies13** (`wt_gold.13pt`, `tracking_split2`) — `1024×1024` grayscale frames
        with **two flies** each and a 13-node skeleton, as `.pkg.slp` files (labels
        with embedded images). We use the **train / val / test** splits plus a clip
        for inference.
        """)
    return


@app.cell
def _(Path, urllib):
    def fetch(url, dest):
        """Download `url` to `dest` once (skips if it already exists)."""
        dest = Path(dest)
        if not dest.exists():
            urllib.request.urlretrieve(url, dest)
        return dest

    _base = "https://storage.googleapis.com/sleap-data/datasets/wt_gold.13pt"
    train_path = fetch(f"{_base}/tracking_split2/train.pkg.slp", "train.pkg.slp")
    val_path = fetch(f"{_base}/tracking_split2/val.pkg.slp", "val.pkg.slp")
    test_path = fetch(f"{_base}/tracking_split2/test.pkg.slp", "test.pkg.slp")
    clip_path = fetch(
        f"{_base}/clips/talk_title_slide%4013150-14500.mp4", "fly_clip.mp4"
    )
    train_path, val_path, test_path, clip_path
    return clip_path, test_path, train_path, val_path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Inspect the labels""")
    return


@app.cell
def _(mo, sio, test_path, train_path, val_path):
    train_labels = sio.load_slp(train_path.as_posix())
    val_labels = sio.load_slp(val_path.as_posix())
    test_labels = sio.load_slp(test_path.as_posix())

    skeleton = train_labels.skeletons[0]
    _img = train_labels[0].image

    mo.md(f"""
        | | |
        |---|---|
        | Train / val / test frames | **{len(train_labels)} / {len(val_labels)} / {len(test_labels)}** |
        | Frame size | **{_img.shape[1]}×{_img.shape[0]}**, {_img.shape[2]} channel(s) |
        | Instances / frame | **{max(len(lf.instances) for lf in train_labels)}** (multi-animal → tracking) |
        | Skeleton | **{len(skeleton.nodes)}** nodes, **{len(skeleton.edges)}** edges |
        | Nodes | {", ".join(n.name for n in skeleton.nodes)} |
        """)
    return skeleton, test_labels, train_labels, val_labels


@app.cell(hide_code=True)
def _(plt, skeleton, train_labels):
    def plot_instances(ax, image, instances, sk):
        """Draw an image plus each instance's skeleton (edges + nodes)."""
        ax.imshow(image, cmap="gray")
        colors = ["lime", "red", "cyan", "magenta", "yellow"]
        for k, inst in enumerate(instances):
            pts = inst.numpy()
            col = colors[k % len(colors)]
            for src, dst in sk.edge_inds:
                ax.plot(
                    [pts[src, 0], pts[dst, 0]],
                    [pts[src, 1], pts[dst, 1]],
                    "-",
                    color=col,
                    linewidth=1,
                    alpha=0.8,
                )
            ax.scatter(pts[:, 0], pts[:, 1], c=col, s=8, zorder=3)
        ax.axis("off")

    _n = 3
    _fig, _axes = plt.subplots(1, _n, figsize=(4 * _n, 4))
    for _i, _ax in enumerate(_axes):
        _lf = train_labels[_i]
        plot_instances(_ax, _lf.image, _lf.instances, skeleton)
        _ax.set_title(f"frame {_lf.frame_idx}", fontsize=9)
    _fig.tight_layout()
    _fig
    return (plot_instances,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 3. Build the model configs (→ YAML)

        We compose each model's config with the `sleap_nn.config` builders, apply the
        architecture settings below, and **save each to a YAML file**. Training then
        reads those YAMLs — the exact same path as the CLI:

        ```bash
        sleap-nn train configs/centroid.yaml
        sleap-nn train configs/centered_instance.yaml
        ```

        **Shared:** ±180° rotation augmentation · `thorax` anchor · 25 epochs ·
        initial LR `1e-4` · eval metrics logged each epoch.

        | | centroid | centered-instance |
        |---|---|---|
        | input scale | 0.5 | 1.0 |
        | crop size | — | 160 |
        | max stride | 16 | 32 |
        | output stride | 2 | 4 |
        | filters / rate | 16 / 2.0 | 24 / 1.5 |
        | confmap sigma | 3.5 | 2.5 |
        """)
    return


@app.cell
def _(mo):
    epochs = mo.ui.slider(1, 100, value=25, label="Epochs (per model)")
    epochs
    return (epochs,)


@app.cell
def _(
    OmegaConf,
    Path,
    TrainingJobConfig,
    epochs,
    get_data_config,
    get_model_config,
    get_trainer_config,
    is_script_mode,
    mo,
    train_path,
    val_path,
):
    def build_cfg(
        model_type,
        *,
        scale,
        crop_size,
        filters,
        filters_rate,
        max_stride,
        output_stride,
        sigma,
        batch_size,
        run_name,
        max_epochs,
    ):
        data = get_data_config(
            train_labels_path=[train_path.as_posix()],
            val_labels_path=[val_path.as_posix()],
            data_pipeline_fw="torch_dataset_cache_img_memory",
            scale=scale,
            crop_size=crop_size,
            use_augmentations_train=True,
            geometry_aug="rotation",
        )
        model = get_model_config(backbone_config="unet", head_configs=model_type)
        trainer = get_trainer_config(
            batch_size=batch_size,
            num_workers=0,  # bump up on Linux (data is cached) for faster loading
            max_epochs=max_epochs,
            learning_rate=1e-4,
            lr_scheduler="reduce_lr_on_plateau",
            save_ckpt=True,
            ckpt_dir="models",
            run_name=run_name,
            visualize_preds_during_training=True,
            trainer_num_devices=1,
        )
        cfg = TrainingJobConfig(
            data_config=data, model_config=model, trainer_config=trainer
        ).to_sleap_nn_cfg()

        u = cfg.model_config.backbone_config.unet
        u.filters, u.filters_rate = filters, filters_rate
        u.max_stride, u.output_stride = max_stride, output_stride

        h = getattr(cfg.model_config.head_configs, model_type).confmaps
        h.sigma, h.anchor_part, h.output_stride = sigma, "thorax", output_stride

        g = cfg.data_config.augmentation_config.geometric
        g.rotation_min, g.rotation_max, g.affine_p = -180.0, 180.0, 1.0

        cfg.trainer_config.eval.enabled = True  # log eval metrics each epoch

        # To log to Weights & Biases (needs an API key), uncomment:
        # cfg.trainer_config.use_wandb = True
        # cfg.trainer_config.wandb.entity = "<entity>"
        # cfg.trainer_config.wandb.project = "<project>"
        # cfg.trainer_config.wandb.api_key = "<key>"
        return cfg

    n_epochs = 1 if is_script_mode else epochs.value

    centroid_cfg = build_cfg(
        "centroid",
        scale=0.5,
        crop_size=None,
        filters=16,
        filters_rate=2.0,
        max_stride=16,
        output_stride=2,
        sigma=3.5,
        batch_size=4,
        run_name="centroid",
        max_epochs=n_epochs,
    )
    centered_cfg = build_cfg(
        "centered_instance",
        scale=1.0,
        crop_size=160,
        filters=24,
        filters_rate=1.5,
        max_stride=32,
        output_stride=4,
        sigma=2.5,
        batch_size=8,
        run_name="centered_instance",
        max_epochs=n_epochs,
    )

    Path("configs").mkdir(exist_ok=True)
    centroid_yaml = "configs/centroid.yaml"
    centered_yaml = "configs/centered_instance.yaml"
    OmegaConf.save(centroid_cfg, centroid_yaml)
    OmegaConf.save(centered_cfg, centered_yaml)

    mo.md(f"Saved **`{centroid_yaml}`** and **`{centered_yaml}`**.")
    return centered_cfg, centered_yaml, centroid_cfg, centroid_yaml


@app.cell(hide_code=True)
def _(OmegaConf, centroid_cfg, mo):
    mo.accordion(
        {
            "📄 View `centroid.yaml`": mo.md(
                f"```yaml\n{OmegaConf.to_yaml(centroid_cfg, resolve=True)}\n```"
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(OmegaConf, centered_cfg, mo):
    mo.accordion(
        {
            "📄 View `centered_instance.yaml`": mo.md(
                f"```yaml\n{OmegaConf.to_yaml(centered_cfg, resolve=True)}\n```"
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 4. Train

        Trains both stages from their YAMLs via `run_training` (the function behind
        `sleap-nn train`). Full-resolution flies are heavier than the quick-start
        datasets — a GPU is strongly recommended. Click **Run training**.
        """)
    return


@app.cell
def _(mo):
    run_train = mo.ui.run_button(label="🚀 Run training")
    run_train
    return (run_train,)


@app.cell
def _(
    OmegaConf,
    centered_yaml,
    centroid_yaml,
    is_script_mode,
    mo,
    run_training,
    run_train,
):
    mo.stop(
        not (run_train.value or is_script_mode),
        mo.md(
            "⬆️ Click **Run training** to train the centroid + centered-instance models."
        ),
    )

    import shutil as _shutil

    # Reuse canonical model dirs on re-run (sleap-nn would otherwise append `-2`).
    for _d in ("models/centroid", "models/centered_instance"):
        _shutil.rmtree(_d, ignore_errors=True)

    _overrides = {}
    if is_script_mode:  # tiny CI run
        _overrides = {
            "trainer_config.train_steps_per_epoch": 5,
            "trainer_config.min_train_steps_per_epoch": 5,
            "trainer_config.visualize_preds_during_training": False,
        }

    for _yaml in (centroid_yaml, centered_yaml):
        _cfg = OmegaConf.load(_yaml)
        for _k, _v in _overrides.items():
            OmegaConf.update(_cfg, _k, _v)
        run_training(_cfg)

    centroid_dir = "models/centroid"
    centered_instance_dir = "models/centered_instance"
    centroid_dir, centered_instance_dir
    return centered_instance_dir, centroid_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 5. Download the trained models

        Each model directory holds `best.ckpt` (weights) + `training_config.yaml`
        (full config incl. skeleton + preprocessing). Download both as a zip to run
        inference locally:

        ```bash
        sleap-nn predict -i my_video.mp4 \
            -m models/centroid/ -m models/centered_instance/ \
            --tracking --max_instances 2 --candidates_method local_queues \
            -o predictions.slp
        ```
        """)
    return


@app.cell(hide_code=True)
def _(Path, centered_instance_dir, centroid_dir, mo):
    import shutil

    _ = (centroid_dir, centered_instance_dir)  # wait for training
    _zip = shutil.make_archive("trained_models", "zip", root_dir="models")
    mo.download(
        data=Path(_zip).read_bytes(),
        filename="trained_models.zip",
        label="⬇️ Download trained models (.zip)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## 6. Evaluate on the test split

        Run the trained top-down model on the held-out **test** labels and compute
        pose metrics (OKS mAP, PCK, keypoint distance) against ground truth.
        """)
    return


@app.cell
def _(mo):
    run_eval = mo.ui.run_button(label="📊 Run evaluation")
    run_eval
    return (run_eval,)


@app.cell(hide_code=True)
def _(
    Evaluator,
    centered_instance_dir,
    centroid_dir,
    device,
    is_script_mode,
    mo,
    predict,
    sio,
    test_labels,
    test_path,
):
    mo.stop(
        not run_eval.value,
        mo.md("⬆️ Click **Run evaluation** to score on the test split."),
    )

    _gt = test_labels
    if is_script_mode:  # only score a few frames in CI
        _gt = sio.Labels(
            videos=test_labels.videos,
            skeletons=test_labels.skeletons,
            labeled_frames=list(test_labels.labeled_frames[:10]),
        )

    eval_preds = predict(
        _gt, model_paths=[centroid_dir, centered_instance_dir], device=device
    )
    metrics = Evaluator(
        ground_truth_instances=_gt, predicted_instances=eval_preds
    ).evaluate()

    mo.md(f"""
        ### Test-set metrics
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
        ## 7. Tracked inference on a clip

        Run the model on a fresh clip **with tracking** so each fly keeps a
        consistent identity. We cap at **2 instances** and use the **local-queues**
        candidate method — both passed to the new `predict` API via a `TrackerConfig`
        (the CLI equivalent is `--tracking --max_instances 2 --candidates_method
        local_queues`).
        """)
    return


@app.cell
def _(mo):
    n_frames = mo.ui.slider(20, 1350, value=200, label="Frames to predict")
    run_infer = mo.ui.run_button(label="🔮 Run tracked inference")
    mo.vstack([n_frames, run_infer])
    return n_frames, run_infer


@app.cell
def _(
    TrackerConfig,
    centered_instance_dir,
    centroid_dir,
    clip_path,
    device,
    is_script_mode,
    mo,
    n_frames,
    predict,
    run_infer,
):
    mo.stop(
        not (run_infer.value or is_script_mode),
        mo.md("⬆️ Click **Run tracked inference** to predict + track on the clip."),
    )

    frames = list(range(5)) if is_script_mode else list(range(n_frames.value))

    clip_preds = predict(
        clip_path.as_posix(),
        model_paths=[centroid_dir, centered_instance_dir],
        device=device,
        frames=frames,
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
        f"**{_n_tracks}** track(s) · saved to `fly_clip.tracked.slp`"
    )
    return (clip_preds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Render the tracked predictions (`sio.render_video`)""")
    return


@app.cell(hide_code=True)
def _(clip_preds, mo, sio):
    # `sio.render_video` overlays the tracked skeletons (one color per track) on the
    # video — the same renderer behind the `sleap-io render` CLI.
    if len(clip_preds) == 0:
        _view = mo.md(
            "_No tracked frames to render — train longer or lower the threshold._"
        )
    else:
        _out = "fly_clip.tracked.viz.mp4"
        sio.render_video(
            clip_preds, _out, fps=30, color_by="track", show_progress=False
        )
        _view = mo.video(_out, controls=True, loop=True, muted=True)
    _view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        ## Next steps

        - **Run locally / on a cluster**: the saved `configs/*.yaml` train with
          `sleap-nn train configs/centroid.yaml` (then the centered-instance one).
        - **Tracking knobs**: tune `window_size`, `scoring_method`, or switch to a
          motion model (`--use_flow` / `--use_kalman`) — see the
          [tracking guide](https://nn.sleap.ai/guides/tracking/).
        - **Render options**: `sleap-io render … --trails --trail-length 10` for
          motion trails; see [io.sleap.ai](https://io.sleap.ai).

        Docs: [nn.sleap.ai](https://nn.sleap.ai) ·
        [Inference API](https://nn.sleap.ai/guides/inference-api/) ·
        [Training](https://nn.sleap.ai/guides/training/)
        """)
    return


if __name__ == "__main__":
    app.run()
