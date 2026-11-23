"""Config for pytests."""

from tests.fixtures.datasets import *
from tests.fixtures.inference import *
from tests.fixtures.legacy_models import *
from tests.fixtures.model_ckpts import *
from tests.fixtures.bbox_models import *
from tests.fixtures.legacy_sleap_json_configs import *
from tests.fixtures.predictors import *

# DIAGNOSTIC, NOT FOR MERGE: log when the number of live FFmpeg (PyAV) threads changes.
import os as _diag_os

_DIAG_AV = [0]


def _diag_av_threads():
    n = 0
    if _diag_os.path.isdir("/proc/self/task"):
        for t in _diag_os.listdir("/proc/self/task"):
            try:
                with open(f"/proc/self/task/{t}/comm") as f:
                    n += f.read().startswith("av:")
            except OSError:
                pass
    return n


def _diag_fd(item):
    from _pytest.faulthandler import fault_handler_stderr_fd_key

    return item.config.stash[fault_handler_stderr_fd_key]


def pytest_runtest_setup(item):
    n = _diag_av_threads()
    if n != _DIAG_AV[0]:
        _DIAG_AV[0] = n
        _diag_os.write(
            _diag_fd(item),
            f"\n[diag-av] {n} PyAV threads at setup of {item.nodeid}\n".encode(),
        )


def pytest_runtest_teardown(item, nextitem):
    n = _diag_av_threads()
    if n != _DIAG_AV[0]:
        _DIAG_AV[0] = n
        _diag_os.write(
            _diag_fd(item),
            f"\n[diag-av] {n} PyAV threads at teardown of {item.nodeid}\n".encode(),
        )
