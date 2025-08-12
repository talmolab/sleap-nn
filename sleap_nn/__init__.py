"""Main module for sleap_nn package."""

import os
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

# Add logger with the custom filter
logger.add(
    lambda msg: print(msg, end=""),
    level="DEBUG",
    filter=_should_log,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)

__version__ = "0.0.1"
