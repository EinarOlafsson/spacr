"""The four cleanup buttons, along the paths a real machine takes.

``tests/qt/test_resource_cleanup.py`` already asserts the promises this
module makes — that nothing is killed, that a run in flight is left alone,
that every figure reported was measured. What is exercised here is the rest
of the surface: the fallbacks that only run when the usual reading is
unavailable (no ``psutil``, no readable ``/proc``, no ``QThreadPool``), the
aggressive form of ``clear_ram`` that drops the expensive caches, and the
launch/pre-run wiring that the two performance modes press.

Where a branch only runs on hardware this box must not touch — an
initialised CUDA context — the *reading* is stood in for and everything
around it is the real function. Nothing here allocates VRAM, and nothing
here starts a thread.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

from PySide6.QtGui import QPixmap, QPixmapCache
from PySide6.QtWidgets import QWidget

from spacr.qt import resource_cleanup as rc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def test_a_drive_line_names_free_total_and_the_percentage_used():
    """One line per drive has to be readable without doing the division."""
    entry = rc.DiskEntry("/data", total=1000, used=750, free=250)
    line = entry.summary()
    assert "/data" in line
    assert "250 B free" in line
    assert "1000 B" in line
    assert "75% used" in line
    assert entry.percent_used == pytest.approx(75.0)


def test_a_drive_of_unknown_size_is_not_a_division_by_zero():
    assert rc.DiskEntry("/nowhere", 0, 0, 0).percent_used == 0.0


def test_a_disk_report_prints_its_note_under_the_drives_not_instead_of_them():
    """A folder that could not be read must not hide the drives that could."""
    report = rc.DiskReport(
        (rc.DiskEntry("/a", 2048, 1024, 1024),
         rc.DiskEntry("/b", 4096, 1024, 3072)),
        note="1 folder(s) could not be read.")
    lines = report.summary().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("/a:")
    assert lines[1].startswith("/b:")
    assert lines[2] == "1 folder(s) could not be read."
    assert report.tightest.path == "/a"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def test_the_process_size_is_still_read_when_psutil_is_missing(monkeypatch):
    """``/proc/self/statm`` is the fallback, and it has to give a real figure.

    ``None`` in ``sys.modules`` is what an absent package looks like to
    ``import``: it raises ``ImportError`` rather than returning a stub, so
    the fallback runs exactly as it would on a machine without psutil.
    """
    with_psutil = rc.process_rss()
    monkeypatch.setitem(sys.modules, "psutil", None)
    without_psutil = rc.process_rss()
    assert without_psutil > 0
    # Same process, two readings of the same quantity — MB apart at worst.
    assert abs(without_psutil - with_psutil) < 256 * 1024 * 1024


def test_a_process_size_that_cannot_be_read_reports_zero_not_a_guess(
        monkeypatch):
    """Zero means "could not measure", and a cleanup then says so."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(os, "sysconf",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    assert rc.process_rss() == 0

    reclaim = rc.clear_ram()
    assert reclaim.measured is False
    assert "could not be read" in reclaim.note
    assert "nothing to measure" in reclaim.summary()


def test_torch_is_never_imported_just_to_ask_about_it(monkeypatch):
    """The VRAM button must not be the thing that allocates the memory."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert rc._torch_if_loaded() is None
    assert rc.cuda_reserved() is None

    sentinel = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", sentinel)
    assert rc._torch_if_loaded() is sentinel


def _fake_torch(reserved, *, available=True, initialised=True, raises=None):
    """A stand-in for torch that answers about a GPU this box must not touch.

    Real ``torch.cuda`` on this machine answers "not available" (the display
    GPU is deliberately hidden from the test run), so the branch that reads a
    reserved figure has no hardware path here. The reading is stood in for;
    everything around it — the releasers, ``empty_cache``, the subtraction,
    the note — is the real function.
    """
    readings = list(reserved)
    calls = {"empty_cache": 0}

    def memory_reserved():
        if raises == "reserved":
            raise RuntimeError("no context")
        return readings.pop(0) if len(readings) > 1 else readings[0]

    def empty_cache():
        calls["empty_cache"] += 1
        if raises == "empty_cache":
            raise RuntimeError("driver said no")

    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        is_initialized=lambda: initialised,
        memory_reserved=memory_reserved,
        empty_cache=empty_cache,
    )
    module._calls = calls
    return module


def test_a_torch_that_raises_when_asked_reports_nothing_to_measure(
        monkeypatch):
    """A driver that errors is "no reading", never a fabricated zero."""
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch([0], raises="reserved"))
    assert rc.cuda_reserved() is None
    reclaim = rc.clear_vram()
    assert reclaim.measured is False
    assert "CUDA context" in reclaim.note


def test_a_hidden_gpu_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch([0], available=False))
    assert rc.cuda_reserved() is None
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch([0], initialised=False))
    assert rc.cuda_reserved() is None


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def test_the_aggressive_cleanup_drops_the_icon_cache_the_mild_one_keeps_it():
    """The whole difference between the two forms of ``clear_ram``."""
    from spacr.qt import iconset

    iconset.veil_color.cache_clear()
    iconset.veil_color("dark")
    iconset.veil_color("light")
    assert iconset.veil_color.cache_info().currsize == 2

    rc.clear_ram(aggressive=False)
    assert iconset.veil_color.cache_info().currsize == 2, \
        "the mild cleanup must not cost the next screen its icons"

    dropped = rc.clear_ram(aggressive=True)
    assert iconset.veil_color.cache_info().currsize == 0
    assert any("iconset.veil_color (2 entries)" in line
               for line in dropped.details)


def test_an_empty_lru_is_not_reported_as_dropped():
    """A cache that held nothing did not free anything, and says nothing."""
    from spacr.qt import iconset

    iconset.veil_color.cache_clear()
    rc.clear_ram(aggressive=True)
    assert not any("veil_color" in line
                   for line in rc.clear_ram(aggressive=True).details)


def test_a_cache_module_nobody_imported_is_left_unimported(monkeypatch):
    """Importing a module to clear it would allocate rather than free."""
    monkeypatch.delitem(sys.modules, "spacr.qt.widgets.animation_zoom",
                        raising=False)
    rc.clear_ram(aggressive=True)
    assert "spacr.qt.widgets.animation_zoom" not in sys.modules


def test_a_dict_cache_is_emptied_and_named_with_its_size(monkeypatch):
    import spacr.crops as crops

    monkeypatch.setattr(crops, "_FORMAT_CACHE", {"a": 1, "b": 2})
    details = rc.clear_ram().details
    assert crops._FORMAT_CACHE == {}
    assert any("spacr.crops._FORMAT_CACHE (2 entries)" in line
               for line in details)


def test_a_dict_cache_whose_module_is_not_imported_is_skipped(monkeypatch):
    monkeypatch.delitem(sys.modules, "spacr.qt.widgets.data_filter_panel",
                        raising=False)
    rc.clear_ram()
    assert "spacr.qt.widgets.data_filter_panel" not in sys.modules


def test_a_cache_that_refuses_to_clear_does_not_sink_the_cleanup(monkeypatch):
    """One broken cache must not cost the user the other four."""
    import spacr.crops as crops

    class Stubborn(dict):
        def clear(self):
            raise RuntimeError("held open")

    monkeypatch.setattr(crops, "_FORMAT_CACHE", Stubborn(a=1))
    monkeypatch.setattr(crops, "_DB_FORMAT_CACHE", {"b": 2})
    details = rc.clear_ram().details
    assert crops._DB_FORMAT_CACHE == {}
    assert not any("crops._FORMAT_CACHE " in line for line in details)


def test_a_live_thumbnail_cache_is_emptied_by_the_aggressive_cleanup(qtbot):
    """The LRU is found through the widget that owns it, not the heap."""
    from spacr.qt.crop_thumbs import CropThumbnails

    holder = QWidget()
    qtbot.addWidget(holder)
    holder._thumbs = CropThumbnails()
    # A crop that cannot be read is cached as a failure, which is still an
    # entry — enough to prove the sweep empties it.
    holder._thumbs.pixmap("/definitely/not/a/crop.png")
    assert len(holder._thumbs) == 1

    details = rc.clear_ram(aggressive=True).details
    assert len(holder._thumbs) == 0
    assert any("thumbnail cache(s), 1 thumbnails" in line for line in details)

    # A second sweep finds it empty and says nothing about it.
    assert not any("thumbnail" in line
                   for line in rc.clear_ram(aggressive=True).details)


def test_a_widget_attribute_that_merely_shares_the_name_is_left_alone(qtbot):
    """``_thumbs`` on some other widget is not a cache to clear."""
    holder = QWidget()
    qtbot.addWidget(holder)
    holder._thumbs = ["not", "a", "cache"]
    rc.clear_ram(aggressive=True)
    assert holder._thumbs == ["not", "a", "cache"]


def test_the_qt_pixmap_cache_is_dropped_by_the_aggressive_cleanup(qapp):
    """The pixmaps are dropped on Qt versions without ``totalUsed`` too."""
    QPixmapCache.clear()
    pixmap = QPixmap(64, 64)
    pixmap.fill()
    assert QPixmapCache.insert("spacr-cov-w3-7", pixmap)
    assert QPixmapCache.find("spacr-cov-w3-7") is not None

    rc.clear_ram(aggressive=True)
    assert QPixmapCache.find("spacr-cov-w3-7") is None


def test_the_pixmap_cache_reading_pyside_does_not_offer_is_not_a_crash(qapp):
    """Whatever it can or cannot read, it must not throw out of Clear RAM.

    ``totalUsed`` is absent from PySide6's QPixmapCache, so this is the path
    the button really takes today: the reading fails, the function reports
    that it dropped nothing, and ``clear_ram`` finishes normally.
    """
    assert not hasattr(QPixmapCache, "totalUsed")
    assert rc._clear_pixmap_cache() == []
    assert rc.clear_ram(aggressive=True).action == "ram"


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------

def test_the_vram_cleanup_subtracts_the_two_readings_it_took(monkeypatch):
    torch = _fake_torch([900, 400])
    monkeypatch.setitem(sys.modules, "torch", torch)
    reclaim = rc.clear_vram()
    assert (reclaim.before, reclaim.after) == (900, 400)
    assert reclaim.freed == 500
    assert torch._calls["empty_cache"] == 1
    assert "torch.cuda.empty_cache()" in reclaim.details


def test_a_model_releaser_is_run_counted_and_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([100, 100]))
    monkeypatch.setattr(rc, "MODEL_RELEASERS",
                        [lambda: 2, lambda: None, lambda: 1])
    details = rc.clear_vram().details
    assert any("3 model reference(s) released" in line for line in details)


def test_a_releaser_that_raises_does_not_stop_empty_cache(monkeypatch):
    """A broken hook loses its own reclaim, never the whole button."""
    def broken():
        raise RuntimeError("model is in use")

    torch = _fake_torch([100, 50])
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(rc, "MODEL_RELEASERS", [broken, lambda: 1])
    reclaim = rc.clear_vram()
    assert torch._calls["empty_cache"] == 1
    assert any("1 model reference(s) released" in line
               for line in reclaim.details)


def test_a_failed_empty_cache_is_reported_as_freeing_nothing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch",
                        _fake_torch([100], raises="empty_cache"))
    reclaim = rc.clear_vram(release_models=False)
    assert reclaim.details == ()
    assert reclaim.freed == 0
    assert "Model references were kept" in reclaim.note


def test_keeping_the_models_says_why_in_the_note(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch([100, 80]))
    monkeypatch.setattr(rc, "MODEL_RELEASERS",
                        [lambda: pytest.fail("a pre-run cleanup released a "
                                             "model the run is about to use")])
    reclaim = rc.clear_vram(release_models=False)
    assert "about to use them" in reclaim.note
    assert "another process" in reclaim.note


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def test_the_pool_is_asked_to_retire_idle_threads_not_to_drop_work():
    from PySide6.QtCore import QThreadPool

    pool = QThreadPool.globalInstance()
    before = pool.expiryTimeout()
    said = rc._retire_idle_pool_threads()
    assert said and "idle threads retired" in said[0]
    assert pool.expiryTimeout() == before, \
        "the expiry timeout is restored, not left at zero"


def test_a_pool_that_will_not_answer_reports_nothing_rather_than_raising(
        monkeypatch):
    from PySide6.QtCore import QThreadPool

    monkeypatch.setattr(QThreadPool, "globalInstance",
                        staticmethod(lambda: None))
    assert rc._retire_idle_pool_threads() == []


def test_a_pool_that_refuses_the_new_timeout_is_not_a_crash(monkeypatch):
    from PySide6.QtCore import QThreadPool

    pool = QThreadPool.globalInstance()
    monkeypatch.setattr(type(pool), "expiryTimeout",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("gone")))
    assert rc._retire_idle_pool_threads() == []


def test_library_threads_come_down_to_the_floor_and_no_further():
    import cv2
    import torch

    before_torch = torch.get_num_threads()
    before_cv2 = cv2.getNumThreads()
    try:
        torch.set_num_threads(8)
        cv2.setNumThreads(8)
        said = rc._lower_library_threads()
        assert torch.get_num_threads() == rc.MIN_LIBRARY_THREADS
        assert cv2.getNumThreads() == rc.MIN_LIBRARY_THREADS
        assert any(line.startswith("torch threads 8") for line in said)
        assert any(line.startswith("OpenCV threads 8") for line in said)
        # Already at the floor: nothing to say and nothing to change.
        assert rc._lower_library_threads() == []
    finally:
        torch.set_num_threads(before_torch)
        cv2.setNumThreads(before_cv2)


def test_a_target_below_the_floor_is_raised_to_the_floor():
    import torch

    before = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        rc._lower_library_threads(target=1)
        assert torch.get_num_threads() == rc.MIN_LIBRARY_THREADS
    finally:
        torch.set_num_threads(before)


def test_a_library_that_will_not_say_how_many_threads_it_has(monkeypatch):
    """Neither reading is load-bearing enough to fail the button over."""
    import cv2
    import torch

    monkeypatch.setattr(torch, "get_num_threads",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    monkeypatch.setattr(cv2, "getNumThreads",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    assert rc._lower_library_threads() == []


def test_a_parked_thread_that_has_exited_is_released_and_counted(monkeypatch):
    """Released once it has actually exited — never terminated to get there."""
    from spacr.qt import bridge

    class Exited:
        def isRunning(self):
            return False

    class Stubborn:
        def isRunning(self):
            return True

    monkeypatch.setattr(bridge, "_PARKED_THREADS",
                        [(Exited(), object()), (Stubborn(), object())])
    reclaim = rc.clear_cpu()
    assert any("1 parked thread(s) released" in line
               for line in reclaim.details)
    assert "parked, not terminated" in reclaim.note
    assert len(bridge._PARKED_THREADS) == 1, \
        "the thread that is still running stays parked"


def test_a_cleanup_without_the_run_registry_still_retires_capacity(
        monkeypatch):
    """The registry is a source of NOTES, not the reason the button works."""
    from spacr.qt import bridge

    monkeypatch.setattr(bridge, "prune_parked_threads",
                        lambda: (_ for _ in ()).throw(RuntimeError("gone")))
    reclaim = rc.clear_cpu()
    assert "still running" not in reclaim.note
    assert any("thread pool" in line for line in reclaim.details)


def test_the_cpu_summary_counts_threads_not_bytes():
    reclaim = rc.Reclaim("cpu", before=9, after=4)
    assert reclaim.summary() == "CPU: 9 → 4 threads."
    assert rc.Reclaim("cpu", 1, 1).summary() == "CPU: 1 thread, unchanged."
    assert rc.Reclaim("cpu", 4, 4).summary() == "CPU: 4 threads, unchanged."


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def test_the_project_paths_are_real_directories_only(monkeypatch, tmp_path):
    """A remembered folder that has since been deleted is not a drive.

    Asked the way the disk report actually asks it: off the GUI thread,
    where stat-ing a remembered folder is the point rather than a freeze.
    On the GUI thread ``project_paths`` answers from the probe cache and
    would leave a folder it has not seen out of this run — which is
    ``tests/qt/test_the_disk_report_never_stats_on_the_gui_thread.py``,
    not this. The predicate is stood in for rather than a worker started,
    because nothing in this file starts a thread.
    """
    from spacr.qt import app as app_mod

    monkeypatch.setattr(rc, "_the_gui_thread_is_asking", lambda: False)

    gone = tmp_path / "deleted"
    here = tmp_path / "kept"
    here.mkdir()

    monkeypatch.setattr(app_mod, "APPS", [("cov_w3_7",)])
    monkeypatch.setattr(rc, "_LRU_CACHE_MODULES", rc._LRU_CACHE_MODULES)
    from spacr.qt import prefs

    monkeypatch.setattr(prefs, "get_last_source", lambda key: str(gone))
    monkeypatch.setattr(prefs, "get_recent_sources",
                        lambda key, limit=3: [str(here), "", None,
                                              str(here)])
    paths = rc.project_paths()
    assert str(here) in paths
    assert str(gone) not in paths
    assert paths.count(str(here)) == 1, "a folder listed twice is one drive"
    assert os.path.expanduser("~") in paths


def test_unreadable_recent_sources_do_not_cost_the_disk_check(monkeypatch):
    """The home and temp directories are reported even when prefs will not."""
    import tempfile

    from spacr.qt import prefs

    monkeypatch.setattr(prefs, "get_last_source",
                        lambda key: (_ for _ in ()).throw(
                            RuntimeError("settings are locked")))
    paths = rc.project_paths()
    assert os.path.expanduser("~") in paths
    assert os.path.realpath(tempfile.gettempdir()) in [
        os.path.realpath(p) for p in paths]


def test_a_temp_directory_that_cannot_be_named_is_skipped(monkeypatch):
    import tempfile

    monkeypatch.setattr(tempfile, "gettempdir",
                        lambda: (_ for _ in ()).throw(OSError("no TMPDIR")))
    assert os.path.expanduser("~") in rc.project_paths()


def test_a_folder_whose_drive_cannot_be_measured_is_counted_as_unreadable(
        monkeypatch, tmp_path):
    """``stat`` succeeds and ``disk_usage`` does not — a filesystem going away
    between the two calls, which is what an unplugged drive looks like."""
    import shutil as shutil_module

    monkeypatch.setattr(rc.shutil, "disk_usage",
                        lambda path: (_ for _ in ()).throw(
                            OSError("no such device")))
    report = rc.disk_report([str(tmp_path)])
    assert report.entries == ()
    assert report.note == "1 folder(s) could not be read."
    assert rc.shutil is shutil_module


# ---------------------------------------------------------------------------
# The modes, and the wiring that presses the buttons
# ---------------------------------------------------------------------------

def _record(monkeypatch, seen):
    """Stand the two cleanups down and record the arguments they were given.

    The cleanups themselves are measured elsewhere in this file; what a mode
    test is asking is WHICH of them a mode runs, and with what — the
    difference between ``aggressive=True`` and ``False`` is the whole
    difference between the two performance modes.
    """
    def _ram(**kwargs):
        seen["ram"] = kwargs
        return rc.Reclaim("ram")

    def _vram(**kwargs):
        seen["vram"] = kwargs
        return rc.Reclaim("vram")

    monkeypatch.setattr(rc, "clear_ram", _ram)
    monkeypatch.setattr(rc, "clear_vram", _vram)


@pytest.fixture
def in_mode(monkeypatch):
    """Put spaCR in one resource mode for the duration of a test."""
    from spacr.qt import preferences

    def choose(mode):
        monkeypatch.setattr(preferences, "get_spacr_mode", lambda: mode)
        return mode

    return choose


def test_the_mode_is_read_from_the_stored_preference():
    """``_mode`` is a lookup, not a cache — the dialog changes it live."""
    from spacr.qt import preferences

    assert rc._mode() == preferences.get_spacr_mode()


def test_a_preference_that_cannot_be_read_falls_back_to_balanced(monkeypatch):
    """Balanced is the safe answer: it cleans nothing on anybody's behalf."""
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_spacr_mode",
                        lambda: (_ for _ in ()).throw(RuntimeError("no store")))
    assert rc._mode() == "balanced"
    assert rc.run_launch_cleanup() == []
    assert rc.run_pre_run_cleanup() == []


def test_balanced_measures_nothing_at_launch(in_mode, monkeypatch):
    in_mode("balanced")
    monkeypatch.setattr(rc, "clear_ram",
                        lambda **k: pytest.fail("Balanced cleaned up anyway"))
    assert rc.run_launch_cleanup() == []


def test_performance_cleans_ram_and_vram_gently(in_mode, monkeypatch):
    """Gently: the expensive caches survive, and the CPU is not touched."""
    in_mode("performance")
    seen = {}
    _record(monkeypatch, seen)
    monkeypatch.setattr(rc, "clear_cpu",
                        lambda **k: pytest.fail("Performance touched the CPU"))
    results = rc.run_launch_cleanup()
    assert [r.action for r in results] == ["ram", "vram"]
    assert seen["ram"] == {"aggressive": False}
    assert seen["vram"] == {"release_models": True}


def test_extra_performance_cleans_everything_at_launch(in_mode, monkeypatch):
    in_mode("extra_performance")
    seen = {}
    _record(monkeypatch, seen)
    results = rc.run_launch_cleanup()
    assert [r.action for r in results] == ["ram", "vram", "cpu"]
    assert seen["ram"] == {"aggressive": True}


def test_the_pre_run_cleanup_keeps_the_models_the_run_is_about_to_load(
        in_mode, monkeypatch):
    in_mode("extra_performance")
    seen = {}
    _record(monkeypatch, seen)
    results = rc.run_pre_run_cleanup("mask_generator")
    assert [r.action for r in results] == ["ram", "vram"]
    assert seen["vram"] == {"release_models": False}


def test_no_pre_run_cleanup_while_another_run_is_in_flight(in_mode,
                                                           monkeypatch):
    """The caches it would drop are the ones the running job is reading."""
    from spacr.qt import bridge

    in_mode("extra_performance")
    monkeypatch.setattr(rc, "clear_ram",
                        lambda **k: pytest.fail("cleaned up mid-run"))
    handles = [types.SimpleNamespace(app_key="a"),
               types.SimpleNamespace(app_key="b")]
    monkeypatch.setattr(bridge, "registry",
                        lambda: types.SimpleNamespace(
                            active=lambda: handles))
    assert rc.run_pre_run_cleanup("c") == []


def test_a_registry_that_cannot_be_consulted_does_not_block_the_cleanup(
        in_mode, monkeypatch):
    in_mode("extra_performance")
    monkeypatch.setattr(rc, "clear_ram", lambda **k: rc.Reclaim("ram"))
    monkeypatch.setattr(rc, "clear_vram", lambda **k: rc.Reclaim("vram"))
    monkeypatch.setattr("spacr.qt.bridge.registry",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    assert [r.action for r in rc.run_pre_run_cleanup()] == ["ram", "vram"]


def test_only_a_new_run_triggers_the_pre_run_cleanup(in_mode, monkeypatch):
    """``changed`` fires for finishes too; a finish is not a new run."""
    from spacr.qt import bridge

    in_mode("extra_performance")
    cleaned = []
    monkeypatch.setattr(rc, "run_pre_run_cleanup",
                        lambda key="": cleaned.append(key))
    monkeypatch.setattr(rc, "_SEEN_RUNS", set())

    handles = [types.SimpleNamespace(app_key="mask_generator")]
    monkeypatch.setattr(bridge, "registry",
                        lambda: types.SimpleNamespace(
                            active=lambda: handles))

    rc._on_registry_changed()
    assert cleaned == ["mask_generator"]

    # The same run, reported again: already seen.
    rc._on_registry_changed()
    assert cleaned == ["mask_generator"]

    # It finishes, and a second one starts.
    handles[:] = [types.SimpleNamespace(app_key="measure")]
    rc._on_registry_changed()
    assert cleaned == ["mask_generator", "measure"]
    assert len(rc._SEEN_RUNS) == 1, \
        "a finished run is forgotten rather than accumulated"


def test_a_pre_run_cleanup_that_fails_does_not_break_starting_a_run(
        monkeypatch):
    """The run was cleaned for, the cleanup failed, and the run is still
    marked seen -- so the next registry change does not try the same
    failing cleanup again for the same run."""
    from spacr.qt import bridge

    attempted = []

    def _explode(key=""):
        attempted.append(key)
        raise RuntimeError("out of memory")

    monkeypatch.setattr(rc, "_SEEN_RUNS", set())
    monkeypatch.setattr(rc, "run_pre_run_cleanup", _explode)
    handles = [types.SimpleNamespace(app_key="x")]
    monkeypatch.setattr(bridge, "registry",
                        lambda: types.SimpleNamespace(
                            active=lambda: handles))

    rc._on_registry_changed()
    assert attempted == ["x"]
    assert len(rc._SEEN_RUNS) == 1

    rc._on_registry_changed()
    assert attempted == ["x"]


def test_an_unavailable_registry_leaves_the_hook_uninstalled(monkeypatch):
    monkeypatch.setattr(rc, "_INSTALLED", False)
    monkeypatch.setattr("spacr.qt.bridge.registry",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    assert rc.install_run_hook() is False
    assert rc._INSTALLED is False


def test_the_hook_is_connected_once_however_often_register_is_called(
        monkeypatch):
    """Forty launches in a test suite must not mean forty connections."""
    from spacr.qt import bridge

    monkeypatch.setattr(rc, "_INSTALLED", False)
    monkeypatch.setattr(rc, "_LAUNCH_DONE", False)
    monkeypatch.setattr(rc, "_SEEN_RUNS", set())
    launches = []
    monkeypatch.setattr(rc, "run_launch_cleanup", lambda: launches.append(1))

    real_registry = bridge.registry()
    connections = []
    monkeypatch.setattr(
        bridge, "registry",
        lambda: types.SimpleNamespace(
            changed=types.SimpleNamespace(
                connect=lambda slot: connections.append(slot)),
            active=real_registry.active))

    assert rc.register() is True
    assert rc.register() is True
    assert connections == [rc._on_registry_changed]
    assert launches == [1]


def test_a_failing_launch_cleanup_does_not_stop_the_application_starting(
        monkeypatch):
    monkeypatch.setattr(rc, "_INSTALLED", True)
    monkeypatch.setattr(rc, "_LAUNCH_DONE", False)
    monkeypatch.setattr(rc, "run_launch_cleanup",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert rc.register() is True


# ---------------------------------------------------------------------------
# The rest of the reporting, and the paths a broken cache takes
# ---------------------------------------------------------------------------

def test_the_summary_says_what_was_freed_what_grew_and_what_did_not_move():
    """Three outcomes, three sentences, and none of them reassuring prose."""
    freed = rc.Reclaim("ram", before=3 * 1024 ** 2, after=1024 ** 2)
    assert freed.summary() == "RAM: freed 2.0 MB."

    grew = rc.Reclaim("ram", before=1024 ** 2, after=3 * 1024 ** 2,
                      note="Caches were dropped.")
    assert grew.freed == 0
    assert grew.summary() == ("RAM: freed nothing — 2.0 MB more is in use "
                              "than before. Caches were dropped.")

    flat = rc.Reclaim("vram", before=5, after=5)
    assert flat.summary() == "VRAM: freed nothing measurable."


def test_a_name_that_cannot_be_fetched_is_skipped_not_fatal(monkeypatch):
    """``dir()`` lists a name; fetching it raises. The sweep goes on.

    A module that defines ``__getattr__`` (PEP 562) alongside ``__dir__`` can
    advertise a name it cannot produce — which is what a lazy re-export looks
    like once the thing it forwards to has gone.
    """
    from spacr.qt import iconset

    probe = types.ModuleType("spacr_cov_w3_7_probe")
    probe.__dir__ = lambda: ["a_name_that_is_not_there"]
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    monkeypatch.setattr(rc, "_LRU_CACHE_MODULES",
                        rc._LRU_CACHE_MODULES + (probe.__name__,))

    iconset.veil_color.cache_clear()
    iconset.veil_color("dark")
    assert any("veil_color" in line for line in rc._clear_lru_caches()), \
        "the module after the unfetchable name was never swept"


def test_a_value_that_cannot_be_read_does_not_stop_the_lru_sweep(monkeypatch):
    """One unreadable module attribute must not cost the whole cleanup."""
    from spacr.qt import iconset

    class Explodes:
        def __getattr__(self, name):
            raise RuntimeError("not readable")

    iconset.veil_color.cache_clear()
    iconset.veil_color("dark")
    monkeypatch.setattr(iconset, "_spacr_cov_probe", Explodes(),
                        raising=False)
    assert any("veil_color" in line for line in rc._clear_lru_caches())


def test_an_lru_that_refuses_to_report_its_size_is_skipped(monkeypatch):
    """A cache that will not say how much it holds is left alone -- and the
    sweep carries on to the caches after it.

    ``dir()`` is alphabetical, so the broken attribute is reached before
    ``veil_color``: a sweep that stopped at it would report nothing, and a
    sweep that never ran would report nothing either.
    """
    from spacr.qt import iconset

    class Broken:
        def cache_clear(self):
            raise AssertionError("cleared a cache it could not size")

        def cache_info(self):
            raise RuntimeError("no")

    iconset.veil_color.cache_clear()
    iconset.veil_color("dark")
    monkeypatch.setattr(iconset, "_spacr_cov_broken_cache", Broken(),
                        raising=False)

    cleared = rc._clear_lru_caches()
    assert not any("_spacr_cov_broken_cache" in line for line in cleared)
    assert any("veil_color" in line for line in cleared)


def test_a_thumbnail_cache_that_will_not_empty_is_left_and_not_counted(qtbot,
                                                                       monkeypatch):
    from spacr.qt.crop_thumbs import CropThumbnails

    holder = QWidget()
    qtbot.addWidget(holder)
    holder._thumbnails = CropThumbnails()
    holder._thumbnails.pixmap("/definitely/not/a/crop.png")
    monkeypatch.setattr(CropThumbnails, "clear",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("a screen is reading it")))
    assert rc._clear_thumbnail_caches() == []
    assert len(holder._thumbnails) == 1


def test_no_thumbnail_module_means_no_thumbnails_to_drop(monkeypatch):
    """Not imported is not populated — and importing it would allocate."""
    monkeypatch.delitem(sys.modules, "spacr.qt.crop_thumbs", raising=False)
    assert rc._clear_thumbnail_caches() == []


def test_no_application_means_no_widgets_to_sweep(monkeypatch):
    from PySide6 import QtWidgets

    monkeypatch.setattr(QtWidgets.QApplication, "instance",
                        staticmethod(lambda: None))
    assert rc._clear_thumbnail_caches() == []


# ---------------------------------------------------------------------------
# CPU, the quiet machine
# ---------------------------------------------------------------------------

def test_a_machine_with_nothing_to_retire_says_exactly_that(monkeypatch):
    monkeypatch.setattr(rc, "_retire_idle_pool_threads", list)
    monkeypatch.setattr(rc, "_lower_library_threads", lambda target=None: [])
    monkeypatch.setattr(rc, "_thread_count", lambda: 7)
    reclaim = rc.clear_cpu()
    assert reclaim.details == ()
    assert reclaim.note.endswith("There was no idle capacity to retire.")


def test_a_running_job_is_named_in_the_note_and_left_alone(monkeypatch):
    """Reading the registry is all this does — it never cancels anything."""
    from spacr.qt import bridge

    handles = [types.SimpleNamespace(app_key="measure")]
    monkeypatch.setattr(bridge, "registry",
                        lambda: types.SimpleNamespace(
                            active=lambda: handles))
    assert "1 spaCR job(s) are still running" in rc.clear_cpu().note
    assert handles == [handles[0]], "the job was left in the registry"


def test_a_qt_build_without_a_thread_pool_retires_nothing(monkeypatch):
    """``QThreadPool`` unimportable is "no idle capacity", not a crash."""
    monkeypatch.setitem(sys.modules, "PySide6.QtCore", None)
    assert rc._retire_idle_pool_threads() == []


# ---------------------------------------------------------------------------
# Disk, the rest of it
# ---------------------------------------------------------------------------

def test_a_folder_name_that_cannot_be_expanded_is_dropped(monkeypatch):
    """One unusable entry costs its own line, never the whole readout."""
    from spacr.qt import app as app_mod
    from spacr.qt import prefs

    monkeypatch.setattr(app_mod, "APPS", [("cov_w3_7",)])
    monkeypatch.setattr(prefs, "get_last_source", lambda key: "~broken")
    monkeypatch.setattr(prefs, "get_recent_sources", lambda key, limit=3: [])
    monkeypatch.setattr(os.path, "expanduser",
                        lambda text: (_ for _ in ()).throw(
                            RuntimeError("no such user"))
                        if text == "~broken" else text)
    assert rc.project_paths()  # the home and temp directories still land


def test_a_folder_that_is_gone_is_counted_rather_than_crashed_on(tmp_path):
    missing = tmp_path / "unplugged"
    report = rc.disk_report([str(missing), str(tmp_path)])
    assert [entry.path for entry in report.entries] == [str(tmp_path)]
    assert report.note == "1 folder(s) could not be read."


def test_two_folders_on_one_drive_are_one_line(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    report = rc.disk_report([str(tmp_path / "a"), str(tmp_path / "b")])
    assert len(report.entries) == 1
    assert report.entries[0].total > 0


def test_no_folders_at_all_says_how_to_get_a_reading():
    report = rc.disk_report([])
    assert report.entries == ()
    assert "No project folder is known yet" in report.summary()


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_a_registry_that_cannot_be_read_skips_the_pre_run_cleanup(monkeypatch):
    monkeypatch.setattr(rc, "run_pre_run_cleanup",
                        lambda key="": pytest.fail("cleaned up blind"))
    monkeypatch.setattr("spacr.qt.bridge.registry",
                        lambda: (_ for _ in ()).throw(RuntimeError("no")))
    rc._on_registry_changed()  # must not raise


def test_a_finished_run_alone_is_not_a_reason_to_clean_up(monkeypatch):
    from spacr.qt import bridge

    monkeypatch.setattr(rc, "_SEEN_RUNS", set())
    monkeypatch.setattr(rc, "run_pre_run_cleanup",
                        lambda key="": pytest.fail("cleaned up for a finish"))
    monkeypatch.setattr(bridge, "registry",
                        lambda: types.SimpleNamespace(active=list))
    rc._on_registry_changed()


def test_without_torch_there_is_no_vram_claim_to_make(monkeypatch):
    """No torch means no CUDA context means nothing this process can free."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    reclaim = rc.clear_vram()
    assert reclaim.measured is False
    assert reclaim.details == ()
    assert reclaim.note == ("torch is not loaded in this process, so it "
                            "holds no VRAM")


@pytest.mark.parametrize("action", rc.ACTIONS)
def test_every_button_can_say_what_it_is_about_to_do(action):
    """A confirmation that cannot name the action is not a confirmation."""
    title = rc.confirmation_title(action)
    text = rc.confirmation_text(action)
    assert title and title[0].isupper()
    assert len(text) > 80
