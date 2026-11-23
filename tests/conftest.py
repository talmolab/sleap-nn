"""Config for pytests."""

import gc
import os

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
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=gc.collect)
