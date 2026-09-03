"""Plots for the evaluation metrics computed by :mod:`sleap_nn.evaluation`.

A trained model directory already carries everything needed to judge the model:
ground truth, predictions, a metrics ``.npz`` per split, and the training log.
This module draws that data. Nothing here recomputes a metric -- it reads what
:func:`sleap_nn.evaluation.run_evaluation` already saved.

The entry point is :func:`plot_metrics`, which takes a model directory (or a
metrics ``.npz`` directly) and a ``kind``::

    from sleap_nn.evaluation_plots import plot_metrics

    plot_metrics("models/my_model", kind="dashboard", save_path="eval.png")

Individual plots take an existing axes, so they compose into your own figures::

    fig, ax = plt.subplots()
    plot_error_by_node(metrics, ax=ax, node_names=names)

The companion to the plots is :func:`worst_instances`, which returns the frames
with the largest localization error. Those carry a video path and frame index,
so a numeric outlier can be handed straight to ``sleap_io.render_image`` and
looked at.

Note:
    This module needs only numpy, matplotlib, and PyYAML -- no torch. It can be
    imported and used without a GPU or a training environment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib

matplotlib.use(
    "Agg"
)  # Use non-interactive backend to avoid tkinter issues on Windows CI
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import yaml
from loguru import logger

from sleap_nn.evaluation import load_metrics

#: Keyword for a horizontal box plot. matplotlib 3.10 deprecated the `vert`
#: bool in favor of `orientation`; the dependency is unpinned, so pick per
#: installed version.
_HORIZONTAL_BOXPLOT = (
    {"orientation": "horizontal"}
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) >= (3, 10)
    else {"vert": False}
)

#: Plot kinds accepted by :func:`plot_metrics`, mapped to the metrics group each
#: one needs. ``"dashboard"`` and ``"training_curve"`` are handled separately.
PLOT_KINDS = {
    "error_distribution": "distance_metrics",
    "error_by_node": "distance_metrics",
    "pck_curve": "pck_metrics",
    "pck_by_node": "pck_metrics",
    "precision_recall": "voc_metrics",
    "visibility": "visibility_metrics",
}

#: Panels drawn by the ``"dashboard"`` kind, in layout order.
DASHBOARD_PANELS = (
    "error_distribution",
    "error_by_node",
    "pck_curve",
    "pck_by_node",
    "precision_recall",
    "visibility",
)


class MetricsNotAvailable(KeyError):
    """Raised when a requested plot's metrics are absent from the file.

    A metrics file only carries the groups its match method computes -- a
    segmentation or bbox model has no ``pck_metrics``, for instance -- so asking
    for a keypoint plot on those metrics is a usage error, not a bug.
    """


def load_node_names(path: Union[str, Path]) -> Optional[List[str]]:
    """Read skeleton node names from a model directory's training config.

    Args:
        path: Path to a model directory, or directly to a ``training_config.yaml``.

    Returns:
        The node names of the first skeleton, or ``None`` if the config is
        missing or carries no skeleton. Metrics files do not store node names,
        so this is how per-node plots get their labels.
    """
    path = Path(path)
    config_path = (
        path if path.suffix in (".yaml", ".yml") else path / "training_config.yaml"
    )
    if not config_path.exists():
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    skeletons = (config or {}).get("data_config", {}).get("skeletons") or []
    if not skeletons:
        return None
    nodes = skeletons[0].get("nodes") or []
    return [n["name"] for n in nodes if "name" in n] or None


def load_training_log(path: Union[str, Path]) -> Optional[Dict[str, np.ndarray]]:
    """Read the per-epoch training log from a model directory.

    Args:
        path: Path to a model directory, or directly to a ``training_log.csv``.

    Returns:
        A dictionary of column name to values with the epoch rows collapsed (the
        logger writes several rows per epoch, one per logged metric), or ``None``
        if the log is missing or empty.
    """
    path = Path(path)
    log_path = path if path.suffix == ".csv" else path / "training_log.csv"
    if not log_path.exists():
        return None

    rows = np.genfromtxt(log_path, delimiter=",", names=True, dtype=float)
    rows = np.atleast_1d(rows)
    if rows.size == 0 or rows.dtype.names is None:
        return None
    return {name: rows[name] for name in rows.dtype.names}


def _require(metrics: Dict[str, Any], group: str, kind: str) -> Dict[str, Any]:
    """Return a metrics group, raising a clear error when it is absent.

    Args:
        metrics: The loaded metrics dictionary.
        group: The group key required, e.g. ``"pck_metrics"``.
        kind: The plot kind being drawn, used in the error message.

    Returns:
        The requested metrics group.

    Raises:
        MetricsNotAvailable: If the group is not present.
    """
    if group not in metrics:
        raise MetricsNotAvailable(
            f"Plot {kind!r} needs {group!r}, which these metrics do not contain "
            f"(available: {sorted(metrics)}). Keypoint plots are only computed "
            f'for match_method="oks"; centroid, mask, semantic and bbox '
            f"evaluations report a different set."
        )
    return metrics[group]


def _resolve_node_names(node_names: Optional[Sequence[str]], n_nodes: int) -> List[str]:
    """Return one label per node, falling back to indices.

    Args:
        node_names: Caller-supplied names, possibly ``None`` or the wrong length.
        n_nodes: Number of nodes that must be labeled.

    Returns:
        A list of exactly `n_nodes` labels.
    """
    if node_names is not None and len(node_names) == n_nodes:
        return list(node_names)
    if node_names is not None:
        logger.warning(
            f"Ignoring {len(node_names)} node names for {n_nodes} nodes "
            "(the metrics file and the skeleton disagree)."
        )
    return [str(i) for i in range(n_nodes)]


def plot_error_distribution(
    metrics: Dict[str, Any], ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot the distribution of localization error with its percentile marks.

    Args:
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into. A new figure is created when ``None``.

    Returns:
        The axes drawn into.
    """
    dm = _require(metrics, "distance_metrics", "error_distribution")
    ax = ax or plt.subplots()[1]

    dists = np.asarray(dm["dists"], dtype=float).ravel()
    finite = dists[np.isfinite(dists)]
    title = "Error distribution"
    if finite.size:
        upper = float(np.percentile(finite, 99))
        n_beyond = int((finite > upper).sum())
        ax.hist(
            finite,
            bins=50,
            range=(0.0, max(upper, np.spacing(1))),
            color="#4C72B0",
            alpha=0.85,
        )
        for key, color in (("p50", "#55A868"), ("p90", "#DD8452"), ("p99", "#C44E52")):
            value = dm.get(key)
            if value is not None and np.isfinite(value) and value <= upper:
                ax.axvline(
                    value, color=color, linestyle="--", linewidth=1.5, label=f"{key}"
                )
        ax.legend(frameon=False, fontsize=8)
        if n_beyond:
            title += f" ({n_beyond} beyond p99, max {finite.max():.1f} px)"

    ax.set_xlabel("Localization error (px)")
    ax.set_ylabel("Keypoints")
    ax.set_title(title)
    return ax


def plot_error_by_node(
    metrics: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    node_names: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """Plot per-node localization error as a box plot, worst node first.

    Args:
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into. A new figure is created when ``None``.
        node_names: Skeleton node names. Indices are used when omitted.

    Returns:
        The axes drawn into.
    """
    dm = _require(metrics, "distance_metrics", "error_by_node")
    ax = ax or plt.subplots()[1]

    dists = np.asarray(dm["dists"], dtype=float)
    if dists.ndim != 2 or dists.size == 0:
        ax.set_title("Error by node (no data)")
        return ax

    names = _resolve_node_names(node_names, dists.shape[1])
    per_node = [d[np.isfinite(d)] for d in dists.T]
    means = np.array([d.mean() if d.size else np.nan for d in per_node])
    order = np.argsort(np.nan_to_num(means, nan=-np.inf))[::-1]

    ax.boxplot(
        [per_node[i] if per_node[i].size else [np.nan] for i in order],
        showfliers=False,
        medianprops={"color": "#C44E52"},
        **_HORIZONTAL_BOXPLOT,
    )
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Localization error (px)")
    ax.set_title("Error by node")
    return ax


def plot_pck_curve(metrics: Dict[str, Any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot PCK against the distance threshold.

    Args:
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into. A new figure is created when ``None``.

    Returns:
        The axes drawn into.
    """
    pm = _require(metrics, "pck_metrics", "pck_curve")
    ax = ax or plt.subplots()[1]

    thresholds = np.asarray(pm["thresholds"], dtype=float)
    pcks = np.asarray(pm["pcks"], dtype=float)
    if pcks.size:
        # (n_instances, n_nodes, n_thresholds) -> mean over instances and nodes.
        ax.plot(thresholds, pcks.mean(axis=(0, 1)), color="#4C72B0", marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Threshold (px)")
    ax.set_ylabel("PCK")
    ax.set_title("PCK vs threshold")
    return ax


def plot_pck_by_node(
    metrics: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    node_names: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """Plot mean PCK per node as a horizontal bar chart, worst node first.

    Args:
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into. A new figure is created when ``None``.
        node_names: Skeleton node names. Indices are used when omitted.

    Returns:
        The axes drawn into.
    """
    pm = _require(metrics, "pck_metrics", "pck_by_node")
    ax = ax or plt.subplots()[1]

    parts = np.asarray(pm["mPCK_parts"], dtype=float)
    if parts.size == 0:
        ax.set_title("PCK by node (no data)")
        return ax

    names = _resolve_node_names(node_names, parts.size)
    order = np.argsort(parts)
    ax.barh([names[i] for i in order], parts[order], color="#4C72B0")
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("mPCK")
    ax.set_title("PCK by node")
    return ax


def plot_precision_recall(
    metrics: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    match_score: str = "oks",
) -> plt.Axes:
    """Plot precision-recall curves across match-score thresholds.

    Args:
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into. A new figure is created when ``None``.
        match_score: Which VOC family to draw, ``"oks"`` or ``"pck"``.

    Returns:
        The axes drawn into.
    """
    vm = _require(metrics, "voc_metrics", "precision_recall")
    ax = ax or plt.subplots()[1]

    prefix = f"{match_score}_voc."
    precisions = np.asarray(vm.get(f"{prefix}precisions", []), dtype=float)
    recalls = np.asarray(vm.get(f"{prefix}recall_thresholds", []), dtype=float)
    thresholds = np.asarray(vm.get(f"{prefix}match_score_thresholds", []), dtype=float)

    if precisions.ndim == 2 and precisions.size and recalls.size:
        colors = plt.cm.viridis(np.linspace(0, 1, precisions.shape[0]))
        for i, row in enumerate(precisions):
            label = f"{thresholds[i]:.2f}" if i < thresholds.size else None
            ax.plot(recalls, row, color=colors[i], linewidth=1.2, label=label)
        if thresholds.size:
            ax.legend(
                title=f"{match_score.upper()} thr",
                fontsize=6,
                title_fontsize=7,
                frameon=False,
                ncol=2,
            )

    mAP = vm.get(f"{prefix}mAP")
    title = "Precision-recall"
    if mAP is not None and np.isfinite(mAP):
        title += f" (mAP={mAP:.3f})"
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    return ax


def plot_visibility(metrics: Dict[str, Any], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot the node-visibility confusion matrix.

    Args:
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into. A new figure is created when ``None``.

    Returns:
        The axes drawn into.
    """
    vm = _require(metrics, "visibility_metrics", "visibility")
    ax = ax or plt.subplots()[1]

    matrix = np.array(
        [
            [float(vm.get("tp", 0)), float(vm.get("fp", 0))],
            [float(vm.get("fn", 0)), float(vm.get("tn", 0))],
        ]
    )
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            f"{int(value)}",
            ha="center",
            va="center",
            color="white" if value > matrix.max() / 2 else "black",
        )
    ax.set_xticks([0, 1], ["visible", "not visible"])
    ax.set_yticks([0, 1], ["visible", "not visible"])
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Predicted")

    precision, recall = vm.get("precision"), vm.get("recall")
    title = "Node visibility"
    if precision is not None and recall is not None:
        title += f" (P={precision:.3f}, R={recall:.3f})"
    ax.set_title(title)
    return ax


def plot_training_curve(
    log: Dict[str, np.ndarray], ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot train and validation loss against epoch.

    Args:
        log: A training log as returned by :func:`load_training_log`.
        ax: Axes to draw into. A new figure is created when ``None``.

    Returns:
        The axes drawn into.
    """
    ax = ax or plt.subplots()[1]

    epochs = log.get("epoch")
    for key, color, label in (
        ("train_loss", "#4C72B0", "train"),
        ("val_loss", "#DD8452", "val"),
    ):
        values = log.get(key)
        if values is None:
            continue
        finite = np.isfinite(values)
        if not finite.any():
            continue
        x = epochs[finite] if epochs is not None else np.arange(finite.sum())
        ax.plot(x, values[finite], color=color, label=label, marker=".")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curve")
    ax.legend(frameon=False, fontsize=8)
    return ax


def worst_instances(metrics: Dict[str, Any], n: int = 10) -> List[Dict[str, Any]]:
    """Return the instances with the largest mean localization error.

    Each entry carries the video path and frame index the metrics file recorded
    alongside the distances, so a numeric outlier can be rendered and inspected::

        for row in worst_instances(metrics, n=3):
            sio.render_image(labels, f"{row['frame_idx']}.png", ...)

    Args:
        metrics: The loaded metrics dictionary.
        n: Maximum number of instances to return.

    Returns:
        Dictionaries with keys ``"frame_idx"``, ``"video_path"``, ``"error"``
        (mean over visible nodes), and ``"max_error"``, ordered worst first.
        Instances with no finite distances are omitted.

    Raises:
        MetricsNotAvailable: If the metrics carry no ``distance_metrics``.
    """
    dm = _require(metrics, "distance_metrics", "worst_instances")
    dists = np.asarray(dm["dists"], dtype=float)
    if dists.ndim != 2 or dists.size == 0:
        return []

    frame_idxs = list(dm.get("frame_idxs", []))
    video_paths = list(dm.get("video_paths", []))

    finite = np.isfinite(dists)
    counts = finite.sum(axis=1)
    means = np.divide(
        np.where(finite, dists, 0.0).sum(axis=1),
        counts,
        out=np.full(dists.shape[0], np.nan),
        where=counts > 0,
    )
    maxes = np.where(counts > 0, np.where(finite, dists, -np.inf).max(axis=1), np.nan)

    order = np.argsort(np.nan_to_num(means, nan=-np.inf))[::-1]
    rows = []
    for i in order:
        if not np.isfinite(means[i]):
            continue
        rows.append(
            {
                "frame_idx": int(frame_idxs[i]) if i < len(frame_idxs) else None,
                "video_path": video_paths[i] if i < len(video_paths) else None,
                "error": float(means[i]),
                "max_error": float(maxes[i]),
            }
        )
        if len(rows) >= n:
            break
    return rows


def _plot_dashboard(
    metrics: Dict[str, Any],
    node_names: Optional[Sequence[str]],
    log: Optional[Dict[str, np.ndarray]],
) -> matplotlib.figure.Figure:
    """Draw every available panel into one figure.

    Args:
        metrics: The loaded metrics dictionary.
        node_names: Skeleton node names, or ``None``.
        log: A training log, or ``None`` to omit that panel.

    Returns:
        The composed figure.

    Raises:
        MetricsNotAvailable: If no panel could be drawn from these metrics.
    """
    panels = [k for k in DASHBOARD_PANELS if PLOT_KINDS[k] in metrics]
    if log is not None:
        panels.append("training_curve")
    if not panels:
        raise MetricsNotAvailable(
            f"No plottable metrics found (available: {sorted(metrics)})."
        )

    n_cols = 2 if len(panels) > 1 else 1
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 4.2 * n_rows))
    flat = np.atleast_1d(np.asarray(axes)).ravel()

    for ax, kind in zip(flat, panels):
        if kind == "training_curve":
            plot_training_curve(log, ax=ax)
        else:
            _draw(kind, metrics, ax, node_names)
    for ax in flat[len(panels) :]:
        ax.axis("off")

    fig.tight_layout()
    return fig


def _draw(
    kind: str,
    metrics: Dict[str, Any],
    ax: plt.Axes,
    node_names: Optional[Sequence[str]],
) -> None:
    """Dispatch a single metrics plot onto an axes.

    Args:
        kind: One of the keys of :data:`PLOT_KINDS`.
        metrics: The loaded metrics dictionary.
        ax: Axes to draw into.
        node_names: Skeleton node names, or ``None``.
    """
    if kind == "error_distribution":
        plot_error_distribution(metrics, ax=ax)
    elif kind == "error_by_node":
        plot_error_by_node(metrics, ax=ax, node_names=node_names)
    elif kind == "pck_curve":
        plot_pck_curve(metrics, ax=ax)
    elif kind == "pck_by_node":
        plot_pck_by_node(metrics, ax=ax, node_names=node_names)
    elif kind == "precision_recall":
        plot_precision_recall(metrics, ax=ax)
    elif kind == "visibility":
        plot_visibility(metrics, ax=ax)


def plot_metrics(
    path: Union[str, Path],
    kind: str = "dashboard",
    split: str = "val",
    dataset_idx: int = 0,
    node_names: Optional[Sequence[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> matplotlib.figure.Figure:
    """Plot the evaluation metrics of a trained model.

    Args:
        path: A model directory or a metrics ``.npz`` file. When a directory is
            given, node names are read from its ``training_config.yaml`` and the
            training curve from its ``training_log.csv``.
        kind: Which plot to draw. One of the keys of :data:`PLOT_KINDS`,
            ``"training_curve"``, or ``"dashboard"`` (the default) for every
            panel the metrics support.
        split: Split to load when ``path`` is a directory: ``"train"``,
            ``"val"``, or ``"test"``. Ignored for a direct ``.npz`` path.
        dataset_idx: Dataset index to load for multi-dataset training. Ignored
            for a direct ``.npz`` path.
        node_names: Skeleton node names for per-node plots. Read from the model
            directory when omitted; falls back to node indices.
        save_path: If given, save the figure here.

    Returns:
        The figure that was drawn.

    Raises:
        ValueError: If `kind` is not a recognized plot kind.
        FileNotFoundError: If no metrics file is found, or if
            ``kind="training_curve"`` and no training log is present.
        MetricsNotAvailable: If the metrics do not contain what `kind` needs.

    Example:
        >>> plot_metrics("models/my_model", kind="dashboard", save_path="eval.png")
    """
    if kind not in PLOT_KINDS and kind not in ("dashboard", "training_curve"):
        raise ValueError(
            f"Unknown plot kind {kind!r}. Expected one of "
            f"{sorted(set(PLOT_KINDS) | {'dashboard', 'training_curve'})}."
        )

    path = Path(path)
    model_dir = path.parent if path.suffix == ".npz" else path

    if node_names is None:
        node_names = load_node_names(model_dir)
    log = load_training_log(model_dir)

    if kind == "training_curve":
        if log is None:
            raise FileNotFoundError(f"No training_log.csv found in {model_dir}.")
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        plot_training_curve(log, ax=ax)
        fig.tight_layout()
    else:
        metrics = load_metrics(str(path), split=split, dataset_idx=dataset_idx)
        if kind == "dashboard":
            fig = _plot_dashboard(metrics, node_names, log)
        else:
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            _draw(kind, metrics, ax, node_names)
            fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {kind} plot to {save_path}")
    return fig
