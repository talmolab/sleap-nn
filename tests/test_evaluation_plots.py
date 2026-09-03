"""Tests for `sleap_nn.evaluation_plots`."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from _pytest.logging import LogCaptureFixture
from loguru import logger

from sleap_nn.evaluation_plots import (
    DASHBOARD_PANELS,
    MetricsNotAvailable,
    load_node_names,
    load_training_log,
    plot_error_by_node,
    plot_error_distribution,
    plot_metrics,
    plot_pck_by_node,
    plot_pck_curve,
    plot_precision_recall,
    plot_training_curve,
    plot_visibility,
    worst_instances,
)

NODE_NAMES = ["head", "thorax", "abdomen"]


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    """Route loguru records into pytest's `caplog`."""
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,
    )
    yield caplog
    logger.remove(handler_id)


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure a test opened so the plot cache stays bounded."""
    yield
    plt.close("all")


def make_metrics(n_instances: int = 8, n_nodes: int = 3) -> dict:
    """Build a metrics dictionary shaped like `Evaluator.evaluate()` output.

    Args:
        n_instances: Number of matched instances to synthesize.
        n_nodes: Number of skeleton nodes.

    Returns:
        A metrics dictionary with the keypoint (``match_method="oks"``) groups.
    """
    rng = np.random.default_rng(0)
    dists = rng.uniform(0.5, 5.0, size=(n_instances, n_nodes))
    thresholds = np.linspace(1, 10, 10)
    pcks = dists[..., None] < thresholds.reshape(1, 1, -1)

    return {
        "voc_metrics": {
            "oks_voc.match_score_thresholds": np.linspace(0.5, 0.95, 10),
            "oks_voc.recall_thresholds": np.linspace(0, 1, 101),
            "oks_voc.precisions": rng.uniform(0.5, 1.0, size=(10, 101)),
            "oks_voc.recalls": rng.uniform(0.5, 1.0, size=10),
            "oks_voc.mAP": 0.82,
            "oks_voc.mAR": 0.79,
        },
        "mOKS": {"mOKS": 0.9},
        "distance_metrics": {
            "frame_idxs": list(range(n_instances)),
            "video_paths": ["video.mp4"] * n_instances,
            "dists": dists,
            "avg": float(dists.mean()),
            "p50": float(np.percentile(dists, 50)),
            "p90": float(np.percentile(dists, 90)),
            "p99": float(np.percentile(dists, 99)),
        },
        "pck_metrics": {
            "thresholds": thresholds,
            "pcks": pcks,
            "mPCK_parts": pcks.mean(axis=0).mean(axis=-1),
            "mPCK": float(pcks.mean()),
        },
        "visibility_metrics": {
            "tp": 100,
            "fp": 5,
            "tn": 10,
            "fn": 3,
            "precision": 100 / 105,
            "recall": 100 / 103,
        },
    }


def make_model_dir(
    tmp_path: Path,
    metrics: dict | None = None,
    with_config: bool = True,
    with_log: bool = True,
    split: str = "val",
) -> Path:
    """Write a minimal model directory to disk.

    Args:
        tmp_path: Directory to build inside.
        metrics: Metrics to save, or `make_metrics()` output when `None`.
        with_config: Whether to write a `training_config.yaml` with a skeleton.
        with_log: Whether to write a `training_log.csv`.
        split: Split name for the metrics file.

    Returns:
        The model directory path.
    """
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)

    np.savez(
        model_dir / f"metrics.{split}.0.npz",
        metrics=metrics if metrics is not None else make_metrics(),
    )

    if with_config:
        config = {
            "data_config": {"skeletons": [{"nodes": [{"name": n} for n in NODE_NAMES]}]}
        }
        with open(model_dir / "training_config.yaml", "w") as f:
            yaml.safe_dump(config, f)

    if with_log:
        (model_dir / "training_log.csv").write_text(
            "epoch,train_loss,val_loss\n0,,0.5\n0,0.6,0.5\n1,0.3,0.25\n"
        )

    return model_dir


# --- Metadata loading ---------------------------------------------------------


def test_load_node_names_from_model_dir(tmp_path):
    model_dir = make_model_dir(tmp_path)

    assert load_node_names(model_dir) == NODE_NAMES


def test_load_node_names_from_yaml_path(tmp_path):
    model_dir = make_model_dir(tmp_path)

    assert load_node_names(model_dir / "training_config.yaml") == NODE_NAMES


def test_load_node_names_missing_config(tmp_path):
    model_dir = make_model_dir(tmp_path, with_config=False)

    assert load_node_names(model_dir) is None


def test_load_node_names_without_skeletons(tmp_path):
    model_dir = make_model_dir(tmp_path, with_config=False)
    with open(model_dir / "training_config.yaml", "w") as f:
        yaml.safe_dump({"data_config": {}}, f)

    assert load_node_names(model_dir) is None


def test_load_node_names_empty_config(tmp_path):
    (tmp_path / "training_config.yaml").write_text("")

    assert load_node_names(tmp_path) is None


def test_load_training_log(tmp_path):
    model_dir = make_model_dir(tmp_path)

    log = load_training_log(model_dir)

    assert set(log) == {"epoch", "train_loss", "val_loss"}
    assert log["epoch"].tolist() == [0.0, 0.0, 1.0]


def test_load_training_log_missing(tmp_path):
    model_dir = make_model_dir(tmp_path, with_log=False)

    assert load_training_log(model_dir) is None


def test_load_training_log_header_only(tmp_path):
    model_dir = make_model_dir(tmp_path, with_log=False)
    (model_dir / "training_log.csv").write_text("epoch,train_loss\n")

    assert load_training_log(model_dir) is None


# --- Individual plots ---------------------------------------------------------


def test_plot_error_distribution():
    ax = plot_error_distribution(make_metrics())

    assert ax.get_xlabel() == "Localization error (px)"
    assert ax.patches  # histogram bars


def test_plot_error_distribution_reports_tail():
    metrics = make_metrics(n_instances=200)
    metrics["distance_metrics"]["dists"][0, 0] = 500.0

    ax = plot_error_distribution(metrics)

    assert "beyond p99" in ax.get_title()
    assert ax.get_xlim()[1] < 500.0  # a lone outlier does not stretch the axis


def test_plot_error_distribution_with_no_finite_distances():
    metrics = make_metrics()
    metrics["distance_metrics"]["dists"] = np.full((4, 3), np.nan)

    ax = plot_error_distribution(metrics)

    assert ax.get_title() == "Error distribution"


def test_plot_error_by_node_orders_worst_first():
    metrics = make_metrics()
    # Make "abdomen" clearly the worst node.
    metrics["distance_metrics"]["dists"][:, 2] += 100.0

    ax = plot_error_by_node(metrics, node_names=NODE_NAMES)

    assert [t.get_text() for t in ax.get_yticklabels()][0] == "abdomen"


def test_plot_error_by_node_falls_back_to_indices():
    ax = plot_error_by_node(make_metrics())

    assert set(t.get_text() for t in ax.get_yticklabels()) == {"0", "1", "2"}


def test_plot_error_by_node_warns_on_name_mismatch(caplog):
    plot_error_by_node(make_metrics(), node_names=["only_one"])

    assert "the metrics file and the skeleton disagree" in caplog.text


def test_plot_error_by_node_with_empty_distances():
    metrics = make_metrics()
    metrics["distance_metrics"]["dists"] = np.zeros((0, 3))

    ax = plot_error_by_node(metrics)

    assert "no data" in ax.get_title()


def test_plot_pck_curve():
    ax = plot_pck_curve(make_metrics())

    assert ax.get_ylabel() == "PCK"
    assert len(ax.lines) == 1
    assert ax.get_ylim() == (0, 1)


def test_plot_pck_by_node():
    ax = plot_pck_by_node(make_metrics(), node_names=NODE_NAMES)

    assert set(t.get_text() for t in ax.get_yticklabels()) == set(NODE_NAMES)


def test_plot_pck_by_node_with_no_parts():
    metrics = make_metrics()
    metrics["pck_metrics"]["mPCK_parts"] = np.array([])

    ax = plot_pck_by_node(metrics)

    assert "no data" in ax.get_title()


def test_plot_precision_recall_draws_one_line_per_threshold():
    ax = plot_precision_recall(make_metrics())

    assert len(ax.lines) == 10
    assert "mAP=0.820" in ax.get_title()


def test_plot_precision_recall_without_curves():
    metrics = make_metrics()
    metrics["voc_metrics"] = {}

    ax = plot_precision_recall(metrics)

    assert len(ax.lines) == 0
    assert ax.get_title() == "Precision-recall"


def test_plot_visibility_matrix_orientation():
    ax = plot_visibility(make_metrics())

    # Rows are predictions, columns ground truth: [[tp, fp], [fn, tn]].
    assert ax.images[0].get_array().tolist() == [[100.0, 5.0], [3.0, 10.0]]
    assert "P=0.952" in ax.get_title()


def test_plot_training_curve(tmp_path):
    log = load_training_log(make_model_dir(tmp_path))

    ax = plot_training_curve(log)

    assert {line.get_label() for line in ax.lines} == {"train", "val"}
    assert ax.get_yscale() == "log"


def test_plot_training_curve_without_epoch_column():
    log = {"train_loss": np.array([1.0, 0.5])}

    ax = plot_training_curve(log)

    assert len(ax.lines) == 1


def test_plot_training_curve_skips_all_nan_series():
    log = {
        "epoch": np.array([0.0, 1.0]),
        "train_loss": np.array([np.nan, np.nan]),
        "val_loss": np.array([1.0, 0.5]),
    }

    ax = plot_training_curve(log)

    assert {line.get_label() for line in ax.lines} == {"val"}


# --- worst_instances ----------------------------------------------------------


def test_worst_instances_orders_by_mean_error():
    metrics = make_metrics()
    metrics["distance_metrics"]["dists"][3] += 100.0

    rows = worst_instances(metrics, n=3)

    assert rows[0]["frame_idx"] == 3
    assert rows[0]["error"] > rows[1]["error"] > rows[2]["error"]
    assert rows[0]["video_path"] == "video.mp4"
    assert rows[0]["max_error"] >= rows[0]["error"]


def test_worst_instances_caps_at_n():
    assert len(worst_instances(make_metrics(n_instances=20), n=4)) == 4


def test_worst_instances_skips_all_nan_rows():
    metrics = make_metrics(n_instances=3)
    metrics["distance_metrics"]["dists"][1] = np.nan

    rows = worst_instances(metrics)

    assert len(rows) == 2
    assert 1 not in [r["frame_idx"] for r in rows]


def test_worst_instances_without_frame_metadata():
    metrics = make_metrics(n_instances=2)
    metrics["distance_metrics"]["frame_idxs"] = []
    metrics["distance_metrics"]["video_paths"] = []

    rows = worst_instances(metrics)

    assert rows[0]["frame_idx"] is None
    assert rows[0]["video_path"] is None


def test_worst_instances_with_empty_distances():
    metrics = make_metrics()
    metrics["distance_metrics"]["dists"] = np.zeros((0, 3))

    assert worst_instances(metrics) == []


def test_worst_instances_requires_distance_metrics():
    with pytest.raises(MetricsNotAvailable):
        worst_instances({"mOKS": {"mOKS": 0.5}})


# --- plot_metrics -------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(DASHBOARD_PANELS))
def test_plot_metrics_each_kind(tmp_path, kind):
    model_dir = make_model_dir(tmp_path)

    fig = plot_metrics(model_dir, kind=kind)

    assert len(fig.axes) == 1


def test_plot_metrics_dashboard_draws_every_panel(tmp_path):
    model_dir = make_model_dir(tmp_path)

    fig = plot_metrics(model_dir, kind="dashboard")

    # Six metric panels plus the training curve, rounded up to a 2-column grid.
    assert len(fig.axes) == 8
    titles = [ax.get_title() for ax in fig.axes]
    assert "Training curve" in titles


def test_plot_metrics_dashboard_without_training_log(tmp_path):
    model_dir = make_model_dir(tmp_path, with_log=False)

    fig = plot_metrics(model_dir, kind="dashboard")

    assert "Training curve" not in [ax.get_title() for ax in fig.axes]


def test_plot_metrics_dashboard_with_partial_metrics(tmp_path):
    metrics = {"distance_metrics": make_metrics()["distance_metrics"]}
    model_dir = make_model_dir(tmp_path, metrics=metrics, with_log=False)

    fig = plot_metrics(model_dir, kind="dashboard")

    titles = [ax.get_title() for ax in fig.axes]
    assert len(titles) == 2
    assert titles[0].startswith("Error distribution")
    assert titles[1] == "Error by node"


def test_plot_metrics_dashboard_with_nothing_plottable(tmp_path):
    model_dir = make_model_dir(
        tmp_path, metrics={"mOKS": {"mOKS": 0.5}}, with_log=False
    )

    with pytest.raises(MetricsNotAvailable, match="No plottable metrics"):
        plot_metrics(model_dir, kind="dashboard")


def test_plot_metrics_uses_node_names_from_config(tmp_path):
    model_dir = make_model_dir(tmp_path)

    fig = plot_metrics(model_dir, kind="pck_by_node")

    labels = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    assert labels == set(NODE_NAMES)


def test_plot_metrics_explicit_node_names_win(tmp_path):
    model_dir = make_model_dir(tmp_path)

    fig = plot_metrics(model_dir, kind="pck_by_node", node_names=["a", "b", "c"])

    assert {t.get_text() for t in fig.axes[0].get_yticklabels()} == {"a", "b", "c"}


def test_plot_metrics_accepts_npz_path(tmp_path):
    model_dir = make_model_dir(tmp_path)

    fig = plot_metrics(model_dir / "metrics.val.0.npz", kind="pck_by_node")

    # Node names still resolve, from the config beside the npz.
    assert {t.get_text() for t in fig.axes[0].get_yticklabels()} == set(NODE_NAMES)


def test_plot_metrics_training_curve_kind(tmp_path):
    model_dir = make_model_dir(tmp_path)

    fig = plot_metrics(model_dir, kind="training_curve")

    assert fig.axes[0].get_title() == "Training curve"


def test_plot_metrics_training_curve_without_log(tmp_path):
    model_dir = make_model_dir(tmp_path, with_log=False)

    with pytest.raises(FileNotFoundError, match="training_log.csv"):
        plot_metrics(model_dir, kind="training_curve")


def test_plot_metrics_saves_file(tmp_path):
    model_dir = make_model_dir(tmp_path)
    save_path = tmp_path / "eval.png"

    plot_metrics(model_dir, kind="pck_curve", save_path=save_path)

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_metrics_rejects_unknown_kind(tmp_path):
    model_dir = make_model_dir(tmp_path)

    with pytest.raises(ValueError, match="Unknown plot kind"):
        plot_metrics(model_dir, kind="nonsense")


def test_plot_metrics_reports_missing_group(tmp_path):
    metrics = {"distance_metrics": make_metrics()["distance_metrics"]}
    model_dir = make_model_dir(tmp_path, metrics=metrics)

    with pytest.raises(MetricsNotAvailable, match="pck_metrics"):
        plot_metrics(model_dir, kind="pck_curve")


def test_plot_metrics_respects_split(tmp_path):
    model_dir = make_model_dir(tmp_path, split="train")

    fig = plot_metrics(model_dir, kind="pck_curve", split="train")

    assert fig.axes[0].get_title() == "PCK vs threshold"
