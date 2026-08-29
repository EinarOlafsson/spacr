"""The memory preferences govern the caches that actually retain data."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import matplotlib
import numpy as np
import pytest
from PySide6.QtGui import QPixmap

matplotlib.use("Agg")

pytestmark = pytest.mark.qt


def _thumbnail(cache, name: str, used: float, side: int = 1024):
    key = (name, 1, side, side)
    cache._store(key, QPixmap(side, side))
    cache._last_used[key] = float(used)
    return key


def test_the_ceiling_is_one_pool_across_real_thumbnail_caches(qapp):
    from spacr.qt import resource_cleanup as cleanup
    from spacr.qt.crop_thumbs import CropThumbnails

    now = time.time()
    older, newer = CropThumbnails(), CropThumbnails()
    old_key = _thumbnail(older, "old", now - 20)
    new_key = _thumbnail(newer, "new", now - 10)

    result = cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=600,
        ceiling_mb=5,
        headroom_short=False,
        owners=(older, newer),
    )

    assert old_key not in older._cache
    assert new_key in newer._cache
    assert result.before_mb >= 8
    assert result.after_mb < result.before_mb
    assert len(result.dropped) == 1


def test_idle_time_evicts_a_real_merged_field_with_measured_bytes(tmp_path):
    from spacr import crops
    from spacr.qt import resource_cleanup as cleanup

    path = tmp_path / "field.npy"
    np.save(path, np.zeros((64, 64, 5), dtype=np.uint16))
    field = crops.open_merged_field(str(path))
    key = next(iter(crops._FIELD_CACHE))
    now = time.time()
    crops._FIELD_CACHE_USED[key] = now - 121

    rows = crops.cache_budget_entries()
    assert rows[0][1] >= field.array.nbytes
    result = cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=2,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(crops,),
    )

    assert key not in crops._FIELD_CACHE
    assert result.dropped


def test_headroom_rechecks_after_each_real_eviction(qapp, monkeypatch):
    from spacr.qt import memory_budget, resource_cleanup
    from spacr.qt.crop_thumbs import CropThumbnails

    now = time.time()
    cache = CropThumbnails()
    oldest = _thumbnail(cache, "oldest", now - 20, side=64)
    newest = _thumbnail(cache, "newest", now - 10, side=64)
    readings = iter((True, True, False))
    monkeypatch.setattr(
        memory_budget, "headroom_is_short", lambda: next(readings))

    result = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=600,
        ceiling_mb=10_000,
        owners=(cache,),
    )

    assert result.pressure is True
    assert oldest not in cache._cache
    assert newest in cache._cache


def test_selected_figure_and_pixmap_survive_an_idle_sweep(qtbot, monkeypatch):
    import matplotlib.pyplot as plt

    from spacr.qt import preferences, resource_cleanup
    from spacr.qt.widgets.figure_queue import FigureQueue

    monkeypatch.setattr(preferences, "get_figure_format", lambda: "png")
    queue = FigureQueue(ram_cap=4)
    qtbot.addWidget(queue)
    queue.set_live_canvas_enabled(False)
    first = plt.figure()
    first.gca().plot(np.arange(16), np.arange(16))
    second = plt.figure()
    second.gca().plot(np.arange(16), np.arange(16) * 2)
    queue.add_figure(first)
    queue.add_figure(second)
    now = time.time()
    queue._figure_last_used.update({0: now - 120, 1: now - 120})
    queue._ram_last_used.update({0: now - 120, 1: now - 120})

    result = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(queue,),
    )

    assert not queue.has_live_figure(0)
    assert 0 not in queue._ram
    assert queue.has_live_figure(1)
    assert 1 in queue._ram
    assert len(result.retained_in_use) == 2
    plt.close("all")


def test_decoded_outline_arrays_are_registered_and_idle_evictable():
    from spacr.qt import annotate_engine, resource_cleanup

    annotate_engine.forget_outline_masks()
    mask = annotate_engine._foreground_mask(
        np.arange(64, dtype=np.uint8).reshape(8, 8), 0.0, 1.0)
    key = next(iter(annotate_engine._MASK_CACHE))
    now = time.time()
    annotate_engine._MASK_CACHE_USED[key] = now - 61

    rows = annotate_engine.cache_budget_entries()
    assert rows[0][1] == mask.nbytes
    resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(annotate_engine,),
    )

    assert key not in annotate_engine._MASK_CACHE
    annotate_engine.forget_outline_masks()


def test_live_owners_are_weakly_registered_and_the_sweep_is_bounded(
        qapp):
    from spacr.qt import resource_cleanup
    from spacr.qt.crop_thumbs import CropThumbnails

    now = time.time()
    cache = CropThumbnails()
    keys = [_thumbnail(cache, str(index), now - index - 1, side=32)
            for index in range(3)]
    assert cache in resource_cleanup._loaded_cache_owners()

    result = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=600,
        ceiling_mb=10_000,
        headroom_short=True,
        max_entries=1,
        owners=(cache,),
    )

    assert sum(key not in cache._cache for key in keys) == 1
    assert result.complete is False


def test_pressure_releases_registered_models_and_existing_gpu_cache(
        monkeypatch):
    from spacr.qt import resource_cleanup

    released = []
    resource_cleanup.MODEL_RELEASERS.append(
        lambda: released.append("model") or 1)
    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: False)
    monkeypatch.setattr(
        resource_cleanup,
        "clear_vram",
        lambda **_kwargs: resource_cleanup.Reclaim(
            "vram", before=9 * 1024, after=3 * 1024),
    )
    try:
        result = resource_cleanup.sweep_memory_budget(
            headroom_short=True, owners=())
    finally:
        resource_cleanup.MODEL_RELEASERS.pop()

    assert released == ["model"]
    assert result.models_released == 1
    assert result.vram_freed == 6 * 1024


def test_pressure_never_releases_a_model_or_allocator_during_a_run(
        monkeypatch):
    from spacr.qt import resource_cleanup

    resource_cleanup.MODEL_RELEASERS.append(
        lambda: pytest.fail("released an in-use model"))
    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: True)
    monkeypatch.setattr(
        resource_cleanup, "clear_vram",
        lambda **_kwargs: pytest.fail("emptied an in-use allocator"),
    )
    try:
        result = resource_cleanup.sweep_memory_budget(
            headroom_short=True, owners=())
    finally:
        resource_cleanup.MODEL_RELEASERS.pop()

    assert result.models_released == 0
    assert result.vram_freed == 0


def test_the_qapplication_owns_one_periodic_budget_timer(qapp):
    from PySide6.QtCore import QTimer

    from spacr.qt import resource_cleanup

    assert resource_cleanup.install_budget_sweep() is True
    timer = qapp.findChild(QTimer, "LiveCacheBudgetSweep")
    assert timer is not None and timer.isActive()
    assert timer.interval() == resource_cleanup.BUDGET_SWEEP_INTERVAL_MS
    assert resource_cleanup.install_budget_sweep() is True
    assert len(qapp.findChildren(QTimer, "LiveCacheBudgetSweep")) == 1


def test_a_cache_retries_registration_after_qapplication_construction():
    """The real launcher registers cleanup before QApplication exists."""
    script = textwrap.dedent(
        """
        from spacr.qt import resource_cleanup
        assert resource_cleanup.install_budget_sweep() is False
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        from spacr.qt.crop_thumbs import CropThumbnails
        cache = CropThumbnails()
        assert cache in resource_cleanup._loaded_cache_owners()
        assert resource_cleanup.install_budget_sweep() is True
        assert resource_cleanup._BUDGET_TIMER.isActive()
        """
    )
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True,
        text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
