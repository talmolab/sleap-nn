"""Main module for sleap_nn package."""

import os
import sys
from loguru import logger

# Get RANK for distributed training
RANK = int(os.environ.get("LOCAL_RANK", -1))


# Configure loguru for distributed training
def _should_log(record):
    """Filter function to control logging based on rank."""
    # Always log ERROR acrossall ranks
    if record["level"].no >= logger.level("ERROR").no:
        return True

    # Log info only on rank 0
    if RANK in [0, -1] and record["level"].no >= logger.level("INFO").no:
        return True

    return False


# Remove default handler and add custom one
logger.remove()


def _safe_print(msg):
    """Print with fallback for encoding errors."""
    try:
        print(msg, end="")
    except UnicodeEncodeError:
        # Fallback: replace unencodable characters with '?'
        print(
            msg.encode(sys.stdout.encoding, errors="replace").decode(
                sys.stdout.encoding
            ),
            end="",
        )


# Add logger with the custom filter
# Disable colorization to avoid ANSI codes in captured output
logger.add(
    _safe_print,
    level="DEBUG",
    filter=_should_log,
    format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
    colorize=False,
)

__version__ = "0.3.0"

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
