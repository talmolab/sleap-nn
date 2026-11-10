"""Main module for sleap_nn package."""

import os
import sys
from loguru import logger


def _get_rank() -> int:
    """Return this process's global rank, or -1 if no launcher set one.

    Checked in the same order Lightning uses (``lightning.fabric.utilities.
    rank_zero._get_rank``), so ``RANK`` agrees with ``trainer.global_rank``:

    - ``RANK``: the global rank, set by ``torchrun`` (also on multi-node runs,
      where ``LOCAL_RANK`` is 0 once per node).
    - ``LOCAL_RANK``: set by Lightning for the processes it spawns itself
      (``sleap-nn train`` with ``trainer_devices > 1``), and by ``torchrun``.
    - ``SLURM_PROCID``: the global rank under Slurm's ``srun``, which starts one
      task per GPU and sets neither of the above. Lightning's
      ``SLURMEnvironment`` uses it, so without this check every srun task
      would see -1 and act as rank 0.
    - ``JSM_NAMESPACE_RANK``: the global rank under LSF's ``jsrun``.

    ``LOCAL_RANK`` comes before ``SLURM_PROCID`` because a Lightning-spawned
    child inside a Slurm job inherits ``SLURM_PROCID=0`` from its parent.
    """
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK"):
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return int(value)
    return -1


# Rank for distributed training, read once at import (before the Lightning
# Trainer exists). -1 means "not distributed"; 0 and -1 both mean the main
# process, which alone writes the run folder's files during training.
RANK = _get_rank()

# Rank the log filter below uses: only ranks 0 and -1 print INFO and WARNING.
# At import it is read from LOCAL_RANK alone, as before, NOT from SLURM_PROCID:
# commands other than training can be started as several independent srun tasks
# (for example ``srun -n 4`` running a wrapper that predicts a different video in
# each task), and each of those keeps its own logs. Training switches it to
# ``RANK`` when it starts (`_use_training_rank_for_logs`), so the DDP processes
# of one training run print each message once.
_LOG_RANK = int(os.environ.get("LOCAL_RANK", -1))


def _use_training_rank_for_logs() -> None:
    """Quiet INFO/WARNING logs on every training process but global rank 0.

    Called at the start of training (``sleap-nn train``, ``run_training``,
    ``ModelTrainer.train``). Re-reads the environment, so it also sees a rank set
    after import. ERROR messages still print on every rank.
    """
    global _LOG_RANK
    _LOG_RANK = _get_rank()


# Force-disable Apple Metal (MPS) acceleration when ``SLEAP_NN_DISABLE_MPS=1``.
#
# MPS on Apple Silicon has known driver bugs. On the GitHub ``macos-14`` CI
# runner in particular, real MPS compute intermittently *hangs* or raises
# ``RuntimeError: MPS backend out of memory`` even for tiny allocations.
# Setting this env var makes torch report MPS as unavailable, so torch itself,
# Lightning's ``accelerator="auto"`` selection, and every ``--device auto``
# resolution in this package all fall back to CPU.
#
# Applied here at package import (rather than in a pytest fixture) so it also
# takes effect inside ``sleap-nn`` CLI *subprocesses* spawned by the tests —
# those inherit the env var but not any in-process patching. torch is imported
# only when the flag is set, keeping normal CLI startup lean. No-op otherwise.
if os.environ.get("SLEAP_NN_DISABLE_MPS") == "1":
    import torch

    if hasattr(torch.backends, "mps"):
        torch.backends.mps.is_available = lambda: False
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
        torch.mps.is_available = lambda: False


# Configure loguru for distributed training
def _should_log(record):
    """Filter function to control logging based on rank."""
    # Always log ERROR acrossall ranks
    if record["level"].no >= logger.level("ERROR").no:
        return True

    # Log info only on rank 0
    if _LOG_RANK in [0, -1] and record["level"].no >= logger.level("INFO").no:
        return True

    return False


# Remove default handler and add custom one
logger.remove()


def _safe_print(msg, stream_name="stdout"):
    """Print with fallback for encoding errors.

    ``stream_name`` (``"stdout"`` or ``"stderr"``) is looked up on ``sys`` at
    call time rather than bound once at sink-creation time, so this respects
    later redirection of ``sys.stdout``/``sys.stderr`` (e.g.
    ``contextlib.redirect_stdout``, pytest's output capture).
    """
    stream = getattr(sys, stream_name)
    try:
        print(msg, end="", file=stream)
    except UnicodeEncodeError:
        # Fallback: replace unencodable characters with '?'
        encoding = getattr(stream, "encoding", None) or "utf-8"
        print(
            msg.encode(encoding, errors="replace").decode(encoding),
            end="",
            file=stream,
        )


def _add_default_sink(stream_name="stdout"):
    """(Re)install the standard sleap_nn log sink, targeting ``stream_name``.

    Centralizes the sink config (level/filter/format) so callers only choose
    *where* logs go, not how they're formatted -- used both for the default
    stdout setup below and for :func:`redirect_logs_to_stderr`.
    """
    logger.add(
        lambda msg: _safe_print(msg, stream_name=stream_name),
        level="DEBUG",
        filter=_should_log,
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        colorize=False,
    )


# Add logger with the custom filter
# Disable colorization to avoid ANSI codes in captured output
_add_default_sink("stdout")


def redirect_logs_to_stderr() -> None:
    """Redirect sleap_nn's log sink to stderr.

    Call this once, early, whenever stdout must be reserved for a
    machine-readable channel -- e.g. the CLI's ``--gui`` mode, which emits one
    JSON progress line per batch on stdout for a GUI subprocess reader to
    parse (see ``sleap_nn.cli._gui_progress_callback``). Without this, a
    plain-text log line (e.g. "Loaded inference model | ...") can interleave
    with the JSON lines on the same fd and break a naive per-line
    ``json.loads()`` reader. Idempotent -- safe to call more than once.
    """
    logger.remove()
    _add_default_sink("stderr")


__version__ = "0.3.5"

# Public API
from sleap_nn.evaluation import load_metrics


def load_models(model_paths, **kwargs):
    """Load trained model(s) into a ready-to-run :class:`Predictor`.

    A discoverable, top-level convenience wrapper around
    :meth:`sleap_nn.inference.Predictor.from_model_paths`. Pass one model
    directory (single-instance / bottom-up / centroid) or a centroid +
    centered-instance pair (top-down); a lone centroid directory is
    auto-detected. Accepts every keyword argument of ``from_model_paths``
    (e.g. ``device``, ``batch_size``, ``peak_threshold``, ``tracker_config``)
    and returns a reusable ``Predictor`` you can call ``.predict(...)`` on
    repeatedly.

    Example:
        >>> import sleap_nn
        >>> predictor = sleap_nn.load_models(
        ...     ["models/centroid/", "models/centered_instance/"], device="cuda"
        ... )
        >>> labels = predictor.predict("video.mp4")

    For a one-shot call, use :func:`sleap_nn.predict` instead.
    """
    from sleap_nn.inference import Predictor

    return Predictor.from_model_paths(model_paths, **kwargs)


# Lazily surface the high-level inference entry points at the top level. These
# are resolved on first access via PEP 562 so that ``import sleap_nn`` (e.g. CLI
# startup or a ``__version__`` check) does not eagerly import the heavy
# inference stack (torch, the predictor/layer modules).
_LAZY_ATTRS = {"predict": "predict", "Predictor": "Predictor"}


def __getattr__(name):
    """Resolve lazily-exposed inference entry points (``predict``, ``Predictor``)."""
    if name in _LAZY_ATTRS:
        from sleap_nn import inference

        return getattr(inference, _LAZY_ATTRS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include the lazily-exposed names in ``dir(sleap_nn)`` for discoverability."""
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = ["load_metrics", "load_models", "predict", "Predictor", "__version__"]
