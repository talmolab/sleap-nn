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

__version__ = "0.1.0a4"

# Public API
from sleap_nn.evaluation import load_metrics

__all__ = ["load_metrics", "__version__"]
