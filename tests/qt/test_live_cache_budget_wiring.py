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


def test_decoded_timelapse_frames_are_measured_idle_and_reproducible(
        qapp, tmp_path):
    from spacr.qt import resource_cleanup
    from spacr.qt.widgets.timelapse_preview import FrameSequence

    path = tmp_path / "frame.npy"
    expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
    np.save(path, expected)
    sequence = FrameSequence("files", [path], 1, [0])
    sequence._register_cache_budget()
    assert np.array_equal(sequence.frame(0), expected)
    now = time.time()
    sequence._cache_last_used[0] = now - 61

    assert sequence in resource_cleanup._loaded_cache_owners()
    result = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(sequence,),
    )

    assert 0 not in sequence._cache
    assert result.before_mb == pytest.approx(expected.nbytes / 1024 ** 2)
    assert np.array_equal(sequence.frame(0), expected)
    assert sequence.read_count == 2


def test_derived_mask_cache_pins_the_displayed_result(qtbot):
    from spacr.qt import resource_cleanup
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel

    panel = TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    old_key, current_key = ("old",), ("current",)
    old = np.zeros((2, 8, 8), dtype=np.int32)
    current = np.ones((2, 8, 8), dtype=np.int32)
    panel._mask_cache.update({old_key: old, current_key: current})
    panel._masks = current
    now = time.time()
    panel._mask_cache_last_used.update({
        old_key: now - 61,
        current_key: now - 61,
    })

    result = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(panel,),
    )

    assert old_key not in panel._mask_cache
    assert panel._mask_cache[current_key] is current
    assert len(result.retained_in_use) == 1
    panel.shutdown()


def test_composited_movie_frames_rebuild_byte_identically(qtbot):
    from spacr.qt import resource_cleanup
    from spacr.qt.widgets.timelapse_movie import FovMovie

    movie = FovMovie()
    qtbot.addWidget(movie)
    images = np.stack((
        np.arange(64, dtype=np.uint8).reshape(8, 8),
        np.arange(64, dtype=np.uint8).reshape(8, 8)[::-1],
    ))
    movie.set_sequence(images)
    key = (0, True, True)
    expected = movie._cache[key].copy()
    now = time.time()
    for cached_key in movie._cache:
        movie._cache_last_used[cached_key] = now - 61

    assert movie in resource_cleanup._loaded_cache_owners()
    resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(movie,),
    )

    assert key not in movie._cache
    assert np.array_equal(movie._rendered(0), expected)


def test_the_real_warm_outline_model_obeys_the_budget(monkeypatch):
    from spacr.qt import annotate_engine, resource_cleanup

    class Parameter:
        def numel(self):
            return 256

        def element_size(self):
            return 4

    class Network:
        def parameters(self):
            return [Parameter(), Parameter()]

        def buffers(self):
            return [Parameter()]

    class Model:
        net = Network()

    model = Model()
    now = time.time()
    monkeypatch.setattr(annotate_engine, "_cellpose_outline_model", model)
    monkeypatch.setattr(
        annotate_engine, "_cellpose_outline_last_used", now - 61)
    monkeypatch.setattr(annotate_engine, "_cellpose_outline_in_use", 1)

    rows = annotate_engine.cache_budget_entries()
    model_row = next(row for row in rows if row[0][0] == "model")
    assert model_row[1] == 3 * 256 * 4
    pinned = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(annotate_engine,),
    )
    assert annotate_engine._cellpose_outline_model is model
    assert pinned.retained_in_use

    annotate_engine._cellpose_outline_in_use = 0
    released = resource_cleanup.sweep_memory_budget(
        now=now,
        idle_minutes=1,
        ceiling_mb=10_000,
        headroom_short=False,
        owners=(annotate_engine,),
    )
    assert annotate_engine._cellpose_outline_model is None
    assert released.dropped


def test_cuda_allocator_cache_obeys_idle_age_and_ceiling(monkeypatch):
    from spacr.qt import resource_cleanup

    mib = 1024 * 1024

    class Cuda:
        def __init__(self):
            self.reserved = [48 * mib, 48 * mib]
            self.allocated = [8 * mib, 8 * mib]
            self.clears = 0

        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def device_count(self):
            return 2

        def memory_reserved(self, device):
            return self.reserved[device]

        def memory_allocated(self, device):
            return self.allocated[device]

        def empty_cache(self):
            self.clears += 1
            self.reserved = list(self.allocated)

    cuda = Cuda()
    fake_torch = type("Torch", (), {"cuda": cuda})()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(resource_cleanup, "_a_run_is_active", lambda: False)
    monkeypatch.setattr(resource_cleanup, "_CUDA_CACHE_BYTES", None)
    monkeypatch.setattr(resource_cleanup, "_CUDA_CACHE_LAST_USED", 0.0)

    first = resource_cleanup.sweep_memory_budget(
        now=100.0,
        idle_minutes=1,
        ceiling_mb=1_000,
        headroom_short=False,
        owners=(),
    )
    assert first.vram_freed == 0
    assert cuda.clears == 0

    idle = resource_cleanup.sweep_memory_budget(
        now=161.0,
        idle_minutes=1,
        ceiling_mb=1_000,
        headroom_short=False,
        owners=(),
    )
    assert idle.vram_freed == 80 * mib
    assert cuda.clears == 1

    cuda.reserved = [48 * mib, 48 * mib]
    resource_cleanup._CUDA_CACHE_BYTES = None
    resource_cleanup._CUDA_CACHE_LAST_USED = 0.0
    capped = resource_cleanup.sweep_memory_budget(
        now=200.0,
        idle_minutes=600,
        ceiling_mb=64,
        headroom_short=False,
        owners=(),
    )
    assert capped.vram_freed == 80 * mib
    assert cuda.clears == 2
