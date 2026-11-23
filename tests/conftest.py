"""Config for pytests."""

import gc
import os

import pytest

from tests.fixtures.datasets import *
from tests.fixtures.inference import *
from tests.fixtures.legacy_models import *
from tests.fixtures.model_ckpts import *
from tests.fixtures.bbox_models import *
from tests.fixtures.legacy_sleap_json_configs import *
from tests.fixtures.predictors import *

# Collect garbage in the test process before every fork (DataLoader workers use fork
# on Linux). A test can leave a video reader, and with it an FFmpeg decoder and its
# frame threads, in uncollected cyclic garbage: for example a video provider kept in
# a MagicMock's recorded calls. A forked worker inherits that garbage but not the
# decoder's threads; if the worker's own garbage collector frees it, FFmpeg waits for
# ever for threads that do not exist there, and the test waits for ever for the
# worker's batch. Collecting before the fork frees such objects here, where their
# threads exist. (Windows has no fork and no os.register_at_fork.)
#
# Guard: after that collection, any FFmpeg thread ("av:*") still alive belongs to a
# video reader that is open, not garbage. A forked child that reads through it or
# frees it hangs the same way, so the test during which such a fork happens fails
# and names the problem. (Linux only: it reads thread names from /proc.)
_FORKS_WITH_DECODER = []


def _ffmpeg_threads():
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        return 0
    n = 0
    for t in tasks:
        try:
            with open(f"/proc/self/task/{t}/comm") as f:
                n += f.read().startswith("av:")
        except OSError:
            pass
    return n


def _before_fork():
    gc.collect()
    n = _ffmpeg_threads()
    if n:
        _FORKS_WITH_DECODER.append(n)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_before_fork)


@pytest.fixture(autouse=True)
def _no_fork_with_an_open_video_decoder():
    _FORKS_WITH_DECODER.clear()
    yield
    if _FORKS_WITH_DECODER:
        n = max(_FORKS_WITH_DECODER)
        _FORKS_WITH_DECODER.clear()
        pytest.fail(
            f"This test forked a child process while {n} FFmpeg decoder threads were "
            "alive: a video reader is still open in the test process. Close videos "
            "before starting fork-based workers; a forked child that reads through "
            "or frees an inherited decoder waits for ever."
        )
