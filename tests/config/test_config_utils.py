"""Tests for the utilities for config building and validation."""

import os

import attr
import pytest
from omegaconf import OmegaConf
from typing import Optional, Text
from loguru import logger

from _pytest.logging import LogCaptureFixture

from sleap_nn.config import utils


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=False,  # Set to 'True' if your test is spawning child processes.
    )
    yield caplog
    logger.remove(handler_id)


def test_one_of(caplog):
    """Test of decorator."""

    @utils.oneof
    @attr.s(auto_attribs=True)
    class ExclusiveClass:
        a: Optional[Text] = None
        b: Optional[Text] = None

    c = ExclusiveClass(a="hello")

    assert c.which_oneof_attrib_name() == "a"
    assert c.which_oneof() == "hello"

    with pytest.raises(ValueError):
        c = ExclusiveClass(a="hello", b="too many values!")
    assert "Only one attribute" in caplog.text


def test_usable_cpu_count():
    """The usable CPU count is a sane positive integer."""
    count = utils.usable_cpu_count()
    assert isinstance(count, int)
    assert count >= 1
    assert count <= max(1, (os.cpu_count() or 1))


def test_usable_cpu_count_respects_affinity(monkeypatch):
    """The affinity mask wins over the machine's total core count."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    assert utils.usable_cpu_count() == 3


def test_usable_cpu_count_floor(monkeypatch):
    """An empty/unavailable count still yields at least one CPU."""
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(), raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert utils.usable_cpu_count() == 1


def test_resolve_auto_num_workers_streaming(monkeypatch):
    """Streaming can't pickle video backends, so `auto` must resolve to 0."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    assert utils.resolve_auto_num_workers("torch_dataset") == 0


@pytest.mark.parametrize(
    "data_pipeline_fw",
    ["torch_dataset_cache_img_memory", "torch_dataset_cache_img_disk"],
)
@pytest.mark.parametrize(
    "cpus,expected",
    [
        (1, 0),
        (2, 1),
        (4, 3),
        (8, utils.MAX_AUTO_NUM_WORKERS),
        (64, utils.MAX_AUTO_NUM_WORKERS),
    ],
)
def test_resolve_auto_num_workers_cached(monkeypatch, data_pipeline_fw, cpus, expected):
    """With caching, `auto` leaves one core free and is capped."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: cpus)
    assert utils.resolve_auto_num_workers(data_pipeline_fw) == expected


def _num_workers_cfg(train, val, data_pipeline_fw="torch_dataset_cache_img_memory"):
    return OmegaConf.create(
        {
            "data_config": {"data_pipeline_fw": data_pipeline_fw},
            "trainer_config": {
                "train_data_loader": {"num_workers": train},
                "val_data_loader": {"num_workers": val},
            },
        }
    )


def test_check_num_workers_resolves_auto(monkeypatch):
    """`auto` is replaced in-config by a concrete int for both loaders."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 4)
    cfg = utils.check_num_workers(_num_workers_cfg("auto", "auto"))
    assert cfg.trainer_config.train_data_loader.num_workers == 3
    assert cfg.trainer_config.val_data_loader.num_workers == 3


def test_check_num_workers_is_case_insensitive(monkeypatch):
    """`Auto`/`AUTO` are accepted the same as `auto`."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 4)
    cfg = utils.check_num_workers(_num_workers_cfg("Auto", "AUTO"))
    assert cfg.trainer_config.train_data_loader.num_workers == 3
    assert cfg.trainer_config.val_data_loader.num_workers == 3


def test_check_num_workers_leaves_explicit_ints(monkeypatch):
    """An explicit worker count is never overridden."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 64)
    cfg = utils.check_num_workers(_num_workers_cfg(0, 2))
    assert cfg.trainer_config.train_data_loader.num_workers == 0
    assert cfg.trainer_config.val_data_loader.num_workers == 2


def test_check_num_workers_per_loader(monkeypatch):
    """Each loader resolves independently."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 4)
    cfg = utils.check_num_workers(_num_workers_cfg("auto", 0))
    assert cfg.trainer_config.train_data_loader.num_workers == 3
    assert cfg.trainer_config.val_data_loader.num_workers == 0


def test_check_num_workers_is_idempotent(monkeypatch):
    """Re-running on an already-resolved config changes nothing."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 4)
    cfg = utils.check_num_workers(_num_workers_cfg("auto", "auto"))
    cfg = utils.check_num_workers(cfg)
    assert cfg.trainer_config.train_data_loader.num_workers == 3


def test_check_num_workers_streaming(monkeypatch):
    """The streaming pipeline resolves `auto` to 0 regardless of core count."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 64)
    cfg = utils.check_num_workers(
        _num_workers_cfg("auto", "auto", data_pipeline_fw="torch_dataset")
    )
    assert cfg.trainer_config.train_data_loader.num_workers == 0
    assert cfg.trainer_config.val_data_loader.num_workers == 0


GB = 1024**3


@pytest.fixture
def fake_memory(monkeypatch):
    """Drive `resolve_auto_num_workers`'s memory bound with synthetic sizes."""
    from sleap_nn.data import utils as data_utils

    def _setup(raw_cache_gb, available_gb, cpus=32):
        monkeypatch.setattr(utils, "usable_cpu_count", lambda: cpus)
        monkeypatch.setattr(
            data_utils,
            "estimate_cache_memory",
            lambda train_labels, val_labels, num_workers, memory_buffer: {
                "raw_cache_bytes": int(raw_cache_gb * GB),
                "python_overhead_bytes": 0,
                "available_bytes": int(available_gb * GB),
            },
        )

    return _setup


def test_auto_num_workers_bounded_by_memory(fake_memory):
    """Workers multiply the in-memory cache, so RAM can bind below the CPU cap."""
    from sleap_nn.data.utils import worker_memory_overhead_factor

    # Cache of 8 GB with 16 GB free: after the 20% buffer there is
    # 16/1.2 - 8 = 5.33 GB of headroom, and each worker costs 8 GB * factor.
    fake_memory(raw_cache_gb=8, available_gb=16)
    expected = int((16 / 1.2 - 8) // (8 * worker_memory_overhead_factor()))

    assert (
        utils.resolve_auto_num_workers(
            "torch_dataset_cache_img_memory", train_labels=[object()], val_labels=[]
        )
        == expected
    )
    assert expected < utils.MAX_AUTO_NUM_WORKERS


def test_auto_num_workers_memory_bound_agrees_with_estimator(monkeypatch):
    """The analytic bound matches the estimator it inverts.

    `resolve_auto_num_workers` solves `estimate_cache_memory` for the largest
    worker count rather than re-estimating per candidate, so the two must not
    drift: the chosen count must fit and one more must not.
    """
    from sleap_nn.data import utils as data_utils

    class _FakeVM:
        available = int(16 * GB)

    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    monkeypatch.setattr(data_utils, "check_memory", lambda labels: int(2 * GB))
    monkeypatch.setattr(data_utils.psutil, "virtual_memory", lambda: _FakeVM())

    labels = [[object()]]
    n = utils.resolve_auto_num_workers(
        "torch_dataset_cache_img_memory", train_labels=labels, val_labels=labels
    )

    assert data_utils.estimate_cache_memory(labels, labels, num_workers=n)["sufficient"]
    if n < utils.MAX_AUTO_NUM_WORKERS:
        assert not data_utils.estimate_cache_memory(labels, labels, num_workers=n + 1)[
            "sufficient"
        ]


def test_auto_num_workers_plenty_of_memory(fake_memory):
    """With ample RAM the CPU cap is what binds."""
    fake_memory(raw_cache_gb=0.5, available_gb=256)
    assert (
        utils.resolve_auto_num_workers(
            "torch_dataset_cache_img_memory", train_labels=[object()], val_labels=[]
        )
        == utils.MAX_AUTO_NUM_WORKERS
    )


def test_auto_num_workers_when_cache_cannot_fit_at_all(fake_memory):
    """A cache too big for zero workers means the run falls back to disk caching.

    Disk caching holds no large in-process cache, so workers are cheap again and
    returning 0 here would needlessly serialize the run.
    """
    fake_memory(raw_cache_gb=64, available_gb=16)
    assert (
        utils.resolve_auto_num_workers(
            "torch_dataset_cache_img_memory", train_labels=[object()], val_labels=[]
        )
        == utils.MAX_AUTO_NUM_WORKERS
    )


def test_auto_num_workers_disk_cache_is_cpu_bound_only(monkeypatch):
    """Disk caching keeps no big in-process cache, so memory never bounds it."""
    from sleap_nn.data import utils as data_utils

    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)

    def _boom(*args, **kwargs):
        raise AssertionError("memory should not be estimated for disk caching")

    monkeypatch.setattr(data_utils, "estimate_cache_memory", _boom)
    assert (
        utils.resolve_auto_num_workers(
            "torch_dataset_cache_img_disk", train_labels=[object()], val_labels=[]
        )
        == utils.MAX_AUTO_NUM_WORKERS
    )


def test_auto_num_workers_without_labels_skips_memory_bound(monkeypatch):
    """Omitting labels falls back to the CPU bound rather than erroring."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    assert (
        utils.resolve_auto_num_workers("torch_dataset_cache_img_memory")
        == utils.MAX_AUTO_NUM_WORKERS
    )


def test_auto_num_workers_empty_cache(fake_memory):
    """A zero-byte cache must not divide by zero."""
    fake_memory(raw_cache_gb=0, available_gb=16)
    assert (
        utils.resolve_auto_num_workers(
            "torch_dataset_cache_img_memory", train_labels=[object()], val_labels=[]
        )
        == utils.MAX_AUTO_NUM_WORKERS
    )


def test_auto_num_workers_streaming_ignores_memory(fake_memory):
    """Streaming is 0 regardless of how much memory is free."""
    fake_memory(raw_cache_gb=0.001, available_gb=1024)
    assert (
        utils.resolve_auto_num_workers(
            "torch_dataset", train_labels=[object()], val_labels=[]
        )
        == 0
    )


def test_check_num_workers_resolves_once_for_both_loaders(monkeypatch):
    """The estimate is expensive, so both loaders share a single resolution."""
    calls = []

    def _spy(data_pipeline_fw, **kwargs):
        calls.append(data_pipeline_fw)
        return 2

    monkeypatch.setattr(utils, "resolve_auto_num_workers", _spy)
    cfg = utils.check_num_workers(_num_workers_cfg("auto", "auto"))

    assert len(calls) == 1
    assert cfg.trainer_config.train_data_loader.num_workers == 2
    assert cfg.trainer_config.val_data_loader.num_workers == 2


def test_suggest_workers_when_caching_with_zero_workers(monkeypatch, caplog):
    """A cached run left at 0 workers is decoding serially for no reason."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    utils.check_num_workers(_num_workers_cfg(0, 0))

    assert "Data caching is enabled" in caplog.text
    assert "train_data_loader.num_workers` (0)" in caplog.text
    assert "val_data_loader.num_workers` (0)" in caplog.text
    assert f"to {utils.MAX_AUTO_NUM_WORKERS}" in caplog.text


def test_suggest_workers_disk_cache(monkeypatch, caplog):
    """Disk caching supports workers too, so it gets the same hint."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    utils.check_num_workers(
        _num_workers_cfg(0, 0, data_pipeline_fw="torch_dataset_cache_img_disk")
    )
    assert "Data caching is enabled" in caplog.text


def test_no_suggestion_when_streaming(monkeypatch, caplog):
    """Streaming must stay at 0 workers, so suggesting more would be wrong."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    utils.check_num_workers(_num_workers_cfg(0, 0, data_pipeline_fw="torch_dataset"))
    assert "Data caching is enabled" not in caplog.text


def test_no_suggestion_when_already_sufficient(monkeypatch, caplog):
    """No nagging a user who is already at or above the suggestion."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    utils.check_num_workers(
        _num_workers_cfg(utils.MAX_AUTO_NUM_WORKERS, utils.MAX_AUTO_NUM_WORKERS)
    )
    assert "Data caching is enabled" not in caplog.text

    utils.check_num_workers(_num_workers_cfg(16, 16))
    assert "Data caching is enabled" not in caplog.text


def test_suggestion_names_only_the_low_loader(monkeypatch, caplog):
    """Only the loaders actually below the suggestion are named."""
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)
    utils.check_num_workers(_num_workers_cfg(utils.MAX_AUTO_NUM_WORKERS, 0))

    assert "val_data_loader.num_workers` (0)" in caplog.text
    assert "train_data_loader.num_workers" not in caplog.text


def test_suggestion_respects_memory_bound(fake_memory, caplog):
    """The hint must never suggest more workers than RAM allows.

    Otherwise it would talk users into a setting that pushes the run past the
    memory check and silently downgrades it to disk caching.
    """
    fake_memory(raw_cache_gb=8, available_gb=16)
    utils.check_num_workers(
        _num_workers_cfg(0, 0), train_labels=[object()], val_labels=[]
    )

    assert "Data caching is enabled" in caplog.text
    assert f"to {utils.MAX_AUTO_NUM_WORKERS}" not in caplog.text


def test_no_suggestion_when_memory_allows_none(fake_memory, caplog):
    """Nothing to suggest when the dataset leaves no room for workers."""
    fake_memory(raw_cache_gb=11, available_gb=16)
    utils.check_num_workers(
        _num_workers_cfg(0, 0), train_labels=[object()], val_labels=[]
    )
    assert "Data caching is enabled" not in caplog.text


def test_no_spurious_auto_log_for_explicit_config(fake_memory, caplog):
    """Resolving speculatively must not claim an `auto` the user never set."""
    fake_memory(raw_cache_gb=8, available_gb=16)
    utils.check_num_workers(
        _num_workers_cfg(0, 0), train_labels=[object()], val_labels=[]
    )
    assert "`num_workers: auto` limited to" not in caplog.text


def test_auto_log_still_emitted_when_requested(fake_memory, caplog):
    """...but it is still reported when `auto` really was requested."""
    fake_memory(raw_cache_gb=8, available_gb=16)
    utils.check_num_workers(
        _num_workers_cfg("auto", "auto"), train_labels=[object()], val_labels=[]
    )
    assert "`num_workers: auto` limited to" in caplog.text


def test_suggestion_skips_estimate_when_nothing_to_suggest(monkeypatch):
    """The memory estimate is skipped when no hint is possible."""
    from sleap_nn.data import utils as data_utils

    def _boom(*args, **kwargs):
        raise AssertionError("should not estimate memory with nothing to suggest")

    monkeypatch.setattr(data_utils, "estimate_cache_memory", _boom)
    monkeypatch.setattr(utils, "usable_cpu_count", lambda: 32)

    utils.check_num_workers(
        _num_workers_cfg(utils.MAX_AUTO_NUM_WORKERS, utils.MAX_AUTO_NUM_WORKERS),
        train_labels=[object()],
        val_labels=[],
    )
