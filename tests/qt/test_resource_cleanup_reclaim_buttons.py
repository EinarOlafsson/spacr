"""The Clear RAM / Clear VRAM / Clear CPU buttons, one behaviour table each.

Each test walks a case table (``subtests``) through one unit of
:mod:`spacr.qt.resource_cleanup` and asserts the exact report it gives back:
the detail lines, the note, the measured endpoints, and what was left
untouched. Only the collaborator a case is about is stood in for (a torch
module, a process-size reading, a thread pool); everything around it is the
real function. Nothing here allocates VRAM or starts a thread.
"""
from __future__ import annotations

import io
import os
import sys
import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QPixmap, QPixmapCache  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt import resource_cleanup as rc  # noqa: E402

pytestmark = pytest.mark.qt

MB = 1024 * 1024


def _module(name, **attributes):
    """A throwaway module carrying ``attributes``."""
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _raise(exc):
    """A callable that raises ``exc`` whatever it is called with."""
    def _boom(*_args, **_kwargs):
        raise exc
    return _boom


def _fake_torch(reserved, *, available=True, initialised=True, raises=None):
    """A torch stand-in answering for a CUDA device this box must not touch.

    ``reserved`` is the sequence of ``memory_reserved`` readings (the last
    one repeats). ``raises`` names the call that fails: ``"reserved"`` or
    ``"empty_cache"``.
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


# ---------------------------------------------------------------------------
# human_bytes
# ---------------------------------------------------------------------------

def test_human_bytes_names_every_size_in_its_largest_whole_unit(subtests):
    """Whole bytes below 1 KB, one decimal above, and TB is the last unit."""
    cases = [
        ("zero", 0, "0 B"),
        ("negative clamps to zero", -5, "0 B"),
        ("just under a KB", 1023, "1023 B"),
        ("one KB", 1024, "1.0 KB"),
        ("a KB and a half", 1536, "1.5 KB"),
        ("two MB", 2 * MB, "2.0 MB"),
        ("three GB", 3 * 1024 ** 3, "3.0 GB"),
        ("one TB", 1024 ** 4, "1.0 TB"),
        ("8 EB stays in TB", 8 * 1024 ** 6, "8388608.0 TB"),
    ]
    for case, count, expected in cases:
        with subtests.test(msg=case):
            assert rc.human_bytes(count) == expected


# ---------------------------------------------------------------------------
# process_rss
# ---------------------------------------------------------------------------

def test_process_rss_reads_psutil_then_the_platform_fallback_then_zero(
        monkeypatch, subtests):
    """psutil when importable; ``/proc/self/statm`` or libproc otherwise; 0 when unreadable."""
    fake_psutil = _module("psutil", Process=lambda pid: types.SimpleNamespace(
        memory_info=lambda: types.SimpleNamespace(rss=777 + 0 * pid)))

    def statm(text):
        return lambda *a, **k: io.StringIO(text)

    cases = [
        ("psutil answers", dict(psutil=fake_psutil), 777),
        ("statm pages times page size", dict(
            psutil=None, platform="linux", open=statm("900 25 3 0 0 0 0\n"),
            sysconf=lambda name: 4096), 25 * 4096),
        ("macOS uses libproc", dict(
            psutil=None, platform="darwin", macos=lambda: 2048), 2048),
        ("statm unreadable", dict(
            psutil=None, platform="linux", open=_raise(OSError("no /proc"))), 0),
        ("page size unreadable", dict(
            psutil=None, platform="linux", open=statm("900 25\n"),
            sysconf=_raise(OSError("no"))), 0),
    ]
    for case, setup, expected in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setitem(sys.modules, "psutil", setup["psutil"])
            if "platform" in setup:
                patch.setattr(rc.sys, "platform", setup["platform"])
            if "open" in setup:
                patch.setattr(rc, "open", setup["open"], raising=False)
            if "sysconf" in setup:
                patch.setattr(os, "sysconf", setup["sysconf"], raising=False)
            patch.setattr(rc, "_macos_rss", setup.get(
                "macos", lambda: pytest.fail("libproc asked off macOS")))
            assert rc.process_rss() == expected

    with subtests.test(msg="an unreadable size makes Clear RAM unmeasured"):
        monkeypatch.setattr(rc, "process_rss", lambda: 0)
        reclaim = rc.clear_ram()
        assert reclaim.measured is False
        assert reclaim.note.startswith(
            "this process's memory use could not be read")
        assert reclaim.summary().startswith(
            "RAM: nothing to measure — this process's memory use could not "
            "be read")


# ---------------------------------------------------------------------------
# cuda_reserved
# ---------------------------------------------------------------------------

def test_cuda_reserved_answers_only_for_a_live_context(monkeypatch, subtests):
    """No torch, no device or a failing driver is ``None``; a live context is its bytes."""
    cases = [
        ("torch not imported", None, None),
        ("cuda hidden", _fake_torch([5], available=False), None),
        ("context never initialised", _fake_torch([5], initialised=False),
         None),
        ("driver raises", _fake_torch([5], raises="reserved"), None),
        ("live context", _fake_torch([6144]), 6144),
    ]
    for case, torch, expected in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if torch is None:
                patch.delitem(sys.modules, "torch", raising=False)
            else:
                patch.setitem(sys.modules, "torch", torch)
            assert rc.cuda_reserved() == expected
            if torch is None:
                assert "torch" not in sys.modules

    with subtests.test(msg="MPS is read and CUDA is never asked"):
        readings = iter([4096, 0])
        seen = []

        def driver_allocated_memory():
            seen.append("mps")
            try:
                return next(readings)
            except StopIteration:
                raise RuntimeError("the MPS context stopped") from None

        def forbidden():
            seen.append("cuda")
            raise AssertionError("MPS must not query CUDA")

        torch = _module(
            "torch",
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: True)),
            mps=types.SimpleNamespace(
                driver_allocated_memory=driver_allocated_memory),
            cuda=types.SimpleNamespace(is_available=forbidden,
                                       is_initialized=forbidden,
                                       memory_reserved=forbidden))
        monkeypatch.setitem(sys.modules, "torch", torch)
        assert [rc.cuda_reserved() for _ in range(3)] == [4096, 0, None]
        assert seen == ["mps", "mps", "mps"]

    cached = [
        ("cached bytes: torch not imported", None, None),
        ("cached bytes: a torch without cuda", _module("torch"), None),
        ("cached bytes: cuda hidden", _fake_torch([5], available=False),
         None),
    ]
    for case, torch, expected in cached:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if torch is None:
                patch.delitem(sys.modules, "torch", raising=False)
            else:
                patch.setitem(sys.modules, "torch", torch)
            assert rc._cuda_cached() is expected


# ---------------------------------------------------------------------------
# clear_ram (the mild form, and its report)
# ---------------------------------------------------------------------------

_ALLOCATOR_NOTE = ("The caches are gone; the allocator has not handed those "
                   "pages back to the OS, so the process size did not move.")
_NOTHING_NOTE = "Nothing was cached, so there was nothing to drop."
_QT_LIVE_NOTE = ("A full Python garbage collection was skipped while Qt "
                 "widgets were live.")


def test_clear_ram_drops_registered_dict_caches_and_reports_measured_rss(
        monkeypatch, subtests, qapp):
    """Exact details, freed bytes and note for each RSS/cache/collector combination."""

    class Stubborn(dict):
        def clear(self):
            raise RuntimeError("held open")

    def install(patch, caches, rss, collected, *, qt_live=False):
        holder = _module("spacr_test_cache_holder", **caches)
        patch.setitem(sys.modules, holder.__name__, holder)
        patch.delitem(sys.modules, "spacr_test_never_imported",
                      raising=False)
        patch.setattr(rc, "_DICT_CACHES", (
            (holder.__name__, "_A"), (holder.__name__, "_B"),
            (holder.__name__, "_MISSING"),
            ("spacr_test_never_imported", "_C")))
        readings = iter(rss)
        patch.setattr(rc, "process_rss", lambda: next(readings))
        if qt_live:
            patch.setattr(rc._gc, "collect",
                          lambda: pytest.fail("collected over live Qt"))
        else:
            patch.setitem(sys.modules, "PySide6.QtWidgets", None)
            patch.setattr(rc._gc, "collect", lambda: collected)
        return holder

    cases = [
        # case, caches, rss before/after, gc count, qt live,
        # expected details, freed, note
        ("caches dropped and RSS shrank",
         dict(_A={"a": 1, "b": 2}, _A_USED={"a": 1.0}, _B=Stubborn(x=1)),
         (200 * MB, 120 * MB), 0, False,
         ("spacr_test_cache_holder._A (2 entries)",), 80 * MB, ""),
        ("caches dropped but RSS stayed",
         dict(_A={"a": 1}, _B={"b": 2, "c": 3}),
         (100 * MB, 100 * MB), 0, False,
         ("spacr_test_cache_holder._A (1 entries)",
          "spacr_test_cache_holder._B (2 entries)"), 0, _ALLOCATOR_NOTE),
        ("only the collector found something",
         dict(_A={}, _B="not a dict"), (50 * MB, 60 * MB), 7, False,
         ("7 unreachable objects collected",), 0, _ALLOCATOR_NOTE),
        ("nothing at all", dict(_A={}), (50 * MB, 50 * MB), 0, False,
         (), 0, _NOTHING_NOTE),
        ("live Qt skips the collector", dict(_A={}), (50 * MB, 40 * MB),
         None, True, (), 10 * MB, f"{_NOTHING_NOTE} {_QT_LIVE_NOTE}"),
    ]
    for case, caches, rss, collected, qt_live, details, freed, note in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            holder = install(patch, caches, rss, collected, qt_live=qt_live)
            reclaim = rc.clear_ram()
            assert reclaim.action == "ram"
            assert reclaim.measured is True
            assert (reclaim.before, reclaim.after) == rss
            assert reclaim.details == details
            assert reclaim.freed == freed
            assert reclaim.note == note
            assert "spacr_test_never_imported" not in sys.modules
            if "_A_USED" in caches:
                assert holder._A == {} and holder._A_USED == {}
                assert dict(holder._B) == {"x": 1}

    with subtests.test(msg="the real crops caches are named and emptied"), \
            monkeypatch.context() as patch:
        import spacr.crops as crops

        patch.setattr(crops, "_FORMAT_CACHE", {"a": 1, "b": 2})
        patch.delitem(sys.modules, "spacr.qt.widgets.data_filter_panel",
                      raising=False)
        details = rc.clear_ram().details
        assert crops._FORMAT_CACHE == {}
        assert "spacr.crops._FORMAT_CACHE (2 entries)" in details
        assert "spacr.qt.widgets.data_filter_panel" not in sys.modules


# ---------------------------------------------------------------------------
# clear_ram (aggressive): LRU, thumbnail and pixmap caches
# ---------------------------------------------------------------------------

def test_aggressive_clear_ram_also_drops_icon_thumbnail_and_pixmap_caches(
        qtbot, monkeypatch, subtests):
    """Mild keeps the expensive caches; aggressive empties and names them, once."""
    from spacr.qt import iconset
    from spacr.qt.crop_thumbs import CropThumbnails

    monkeypatch.delitem(sys.modules, "spacr.qt.widgets.animation_zoom",
                        raising=False)
    iconset.veil_color.cache_clear()
    iconset.veil_color("dark")
    iconset.veil_color("light")

    first, second, impostor = QWidget(), QWidget(), QWidget()
    for widget in (first, second, impostor):
        qtbot.addWidget(widget)
    first._thumbs = CropThumbnails()
    first._thumbs.pixmap("/definitely/not/a/crop.png")
    second._thumbnails = CropThumbnails()
    second._thumbnails.pixmap("/definitely/not/a/crop1.png")
    second._thumbnails.pixmap("/definitely/not/a/crop2.png")
    impostor._thumbs = ["not", "a", "cache"]

    QPixmapCache.clear()
    pixmap = QPixmap(16, 16)
    pixmap.fill()
    assert QPixmapCache.insert("spacr-reclaim-test", pixmap)

    with subtests.test(msg="the mild form keeps them"):
        mild = rc.clear_ram(aggressive=False)
        assert iconset.veil_color.cache_info().currsize == 2
        assert (len(first._thumbs), len(second._thumbnails)) == (1, 2)
        assert QPixmapCache.find("spacr-reclaim-test") is not None
        assert not any("veil_color" in line or "thumbnail" in line
                       for line in mild.details)

    with subtests.test(msg="the aggressive form drops and names them"):
        dropped = rc.clear_ram(aggressive=True)
        assert "spacr.qt.iconset.veil_color (2 entries)" in dropped.details
        assert "2 thumbnail cache(s), 3 thumbnails" in dropped.details
        assert iconset.veil_color.cache_info().currsize == 0
        assert (len(first._thumbs), len(second._thumbnails)) == (0, 0)
        assert QPixmapCache.find("spacr-reclaim-test") is None
        assert impostor._thumbs == ["not", "a", "cache"]
        assert "spacr.qt.widgets.animation_zoom" not in sys.modules

    with subtests.test(msg="a second sweep has nothing of theirs to report"):
        again = rc.clear_ram(aggressive=True)
        assert not any("veil_color" in line or "thumbnail" in line
                       for line in again.details)

    with subtests.test(msg="PySide offers no pixmap size, so no line"):
        assert not hasattr(QPixmapCache, "totalUsed")
        assert rc._clear_pixmap_cache() == []


def test_the_lru_sweep_steps_over_attributes_it_cannot_read_or_size(
        monkeypatch, subtests):
    """An unfetchable name, an unreadable value or an unsizable cache costs only itself."""
    from spacr.qt import iconset

    class Explodes:
        def __getattr__(self, name):
            raise RuntimeError("not readable")

    class Unsizable:
        def cache_clear(self):
            raise AssertionError("cleared a cache it could not size")

        def cache_info(self):
            raise RuntimeError("no")

    probe = _module("spacr_reclaim_probe")
    probe.__dir__ = lambda: ["a_name_that_is_not_there"]

    def unfetchable(patch):
        patch.setitem(sys.modules, probe.__name__, probe)
        patch.setattr(rc, "_LRU_CACHE_MODULES",
                      (probe.__name__,) + rc._LRU_CACHE_MODULES)

    cases = [
        ("a listed name that cannot be fetched", unfetchable),
        ("a value whose attributes raise", lambda patch: patch.setattr(
            iconset, "_spacr_probe_explodes", Explodes(), raising=False)),
        ("a cache that will not report its size", lambda patch: patch.setattr(
            iconset, "_spacr_probe_unsizable", Unsizable(), raising=False)),
        ("a cache module that is not imported", lambda patch: patch.setattr(
            rc, "_LRU_CACHE_MODULES",
            ("spacr_never_imported_lru",) + rc._LRU_CACHE_MODULES)),
    ]
    for case, arrange in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            arrange(patch)
            iconset.veil_color.cache_clear()
            iconset.veil_color("dark")
            cleared = rc._clear_lru_caches()
            assert "spacr.qt.iconset.veil_color (1 entries)" in cleared
            assert not any("_spacr_probe" in line or "a_name_that" in line
                           for line in cleared)
            assert iconset.veil_color.cache_info().currsize == 0
            assert "spacr_never_imported_lru" not in sys.modules


def test_the_thumbnail_sweep_gives_up_quietly_when_it_cannot_reach_a_cache(
        qtbot, monkeypatch, subtests):
    """No module, no application, or a cache that refuses: ``[]`` and nothing emptied."""
    from PySide6 import QtWidgets

    from spacr.qt.crop_thumbs import CropThumbnails

    holder = QWidget()
    qtbot.addWidget(holder)
    holder._thumbnails = CropThumbnails()
    holder._thumbnails.pixmap("/definitely/not/a/crop.png")

    cases = [
        ("the cache refuses to clear", lambda patch: patch.setattr(
            CropThumbnails, "clear", _raise(RuntimeError("being read")))),
        ("the thumbnail module is not imported", lambda patch: patch.delitem(
            sys.modules, "spacr.qt.crop_thumbs", raising=False)),
        ("Qt widgets are not imported", lambda patch: patch.setitem(
            sys.modules, "PySide6.QtWidgets", None)),
        ("no application", lambda patch: patch.setattr(
            QtWidgets.QApplication, "instance", staticmethod(lambda: None))),
    ]
    for case, arrange in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            arrange(patch)
            assert rc._clear_thumbnail_caches() == []
            assert len(holder._thumbnails) == 1

    with subtests.test(msg="reachable, it is emptied and counted"):
        assert rc._clear_thumbnail_caches() == [
            "1 thumbnail cache(s), 1 thumbnails"]
        assert len(holder._thumbnails) == 0


# ---------------------------------------------------------------------------
# clear_vram
# ---------------------------------------------------------------------------

_VRAM_NOTE = "It cannot reclaim VRAM held by another process — nothing can."
_KEPT_NOTE = ("Model references were kept: this ran immediately before a run "
              "that is about to use them. " + _VRAM_NOTE)


def test_clear_vram_reports_the_two_readings_and_what_it_released(
        monkeypatch, subtests):
    """Endpoints, details and note for releasers, failures and the pre-run form."""
    def broken():
        raise RuntimeError("model is in use")

    cases = [
        # case, readings, releasers, release_models, raises,
        # (before, after), details, empty_cache calls, note
        ("empty_cache frees the difference", [900, 400], [], True, None,
         (900, 400), ("torch.cuda.empty_cache()",), 1, _VRAM_NOTE),
        ("releasers are counted together", [100, 100],
         [lambda: 2, lambda: None, lambda: 1], True, None, (100, 100),
         ("3 model reference(s) released", "torch.cuda.empty_cache()"), 1,
         _VRAM_NOTE),
        ("a raising releaser loses only itself", [100, 50],
         [broken, lambda: 1], True, None, (100, 50),
         ("1 model reference(s) released", "torch.cuda.empty_cache()"), 1,
         _VRAM_NOTE),
        ("a releaser that released nothing adds no line", [10, 10],
         [lambda: 0], True, None, (10, 10), ("torch.cuda.empty_cache()",), 1,
         _VRAM_NOTE),
        ("pre-run keeps the models", [100, 80],
         [lambda: pytest.fail("released a model the run will use")], False,
         None, (100, 80), ("torch.cuda.empty_cache()",), 1, _KEPT_NOTE),
        ("a failed empty_cache frees nothing", [100], [], False,
         "empty_cache", (100, 100), (), 1, _KEPT_NOTE),
    ]
    monkeypatch.delitem(sys.modules, "spacr.qt.annotate_engine",
                        raising=False)
    for (case, readings, releasers, release_models, raises, ends, details,
         empties, note) in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            torch = _fake_torch(readings, raises=raises)
            patch.setitem(sys.modules, "torch", torch)
            patch.setattr(rc, "MODEL_RELEASERS", list(releasers))
            reclaim = rc.clear_vram(release_models=release_models)
            assert reclaim.action == "vram"
            assert reclaim.measured is True
            assert (reclaim.before, reclaim.after) == ends
            assert reclaim.freed == ends[0] - ends[1]
            assert reclaim.details == details
            assert torch._calls["empty_cache"] == empties
            assert reclaim.note == note

    unmeasured = [
        ("no torch", None,
         "torch is not loaded in this process, so it holds no VRAM"),
        ("no initialised context", _fake_torch([5], initialised=False),
         "this process has no initialised CUDA context, so it holds no VRAM"),
        ("the driver will not answer", _fake_torch([5], raises="reserved"),
         "this process has no initialised CUDA context, so it holds no VRAM"),
    ]
    for case, torch, why in unmeasured:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if torch is None:
                patch.delitem(sys.modules, "torch", raising=False)
            else:
                patch.setitem(sys.modules, "torch", torch)
            patch.setattr(rc, "MODEL_RELEASERS",
                          [lambda: pytest.fail("released without a reading")])
            reclaim = rc.clear_vram()
            assert reclaim == rc.Reclaim("vram", 0, 0, (), note=why,
                                         measured=False)
            assert reclaim.summary() == f"VRAM: nothing to measure — {why}"

    with subtests.test(msg="an allocator that fails is never a reclaim"), \
            monkeypatch.context() as patch:
        from spacr import accelerator

        samples = iter([4096, None])
        patch.setattr(rc, "cuda_reserved", lambda: next(samples))
        patch.setattr(rc, "_torch_if_loaded", lambda: None)
        patch.setattr(accelerator, "empty_cache",
                      _raise(RuntimeError("allocator refused")))
        report = rc.clear_vram(release_models=False)
        assert (report.before, report.after, report.details) == (4096, 4096,
                                                                 ())
        assert report.note == _KEPT_NOTE


# ---------------------------------------------------------------------------
# clear_cpu and its two helpers
# ---------------------------------------------------------------------------

class _Pool:
    """A ``QThreadPool`` stand-in that records the expiry it is given."""

    def __init__(self, expiry, active=0, refuse=False):
        self.expiry = expiry
        self.active = active
        self.refuse = refuse
        self.set_calls = []

    def expiryTimeout(self):  # noqa: N802 - Qt spelling
        if self.refuse:
            raise RuntimeError("gone")
        return self.expiry

    def setExpiryTimeout(self, value):  # noqa: N802 - Qt spelling
        self.set_calls.append(value)
        self.expiry = value

    def activeThreadCount(self):  # noqa: N802 - Qt spelling
        return self.active


def test_idle_pool_threads_are_retired_by_a_zero_expiry_then_restored(
        monkeypatch, subtests, qapp):
    """The expiry is dropped to 0 and put back; queued work is never cleared."""
    from PySide6.QtCore import QThreadPool

    cases = [
        ("a set expiry is restored", _Pool(5000, active=3), [0, 5000],
         ["Qt thread pool: idle threads retired (3 still working)"]),
        ("an unset expiry gets Qt's default back", _Pool(0, active=0),
         [0, 30000],
         ["Qt thread pool: idle threads retired (0 still working)"]),
        ("a pool that refuses is not a crash", _Pool(5000, refuse=True), [],
         []),
        ("no pool", None, None, []),
    ]
    for case, pool, set_calls, expected in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(QThreadPool, "globalInstance",
                          staticmethod(lambda: pool))
            assert rc._retire_idle_pool_threads() == expected
            if pool is not None:
                assert pool.set_calls == set_calls

    with subtests.test(msg="the real pool keeps its timeout"):
        real = QThreadPool.globalInstance()
        before = real.expiryTimeout()
        said = rc._retire_idle_pool_threads()
        assert said == [f"Qt thread pool: idle threads retired "
                        f"({real.activeThreadCount()} still working)"]
        assert real.expiryTimeout() == before

    with subtests.test(msg="a Qt build without QtCore"), \
            monkeypatch.context() as patch:
        patch.setitem(sys.modules, "PySide6.QtCore", None)
        assert rc._retire_idle_pool_threads() == []


def test_library_threads_are_lowered_to_the_floor_only_when_loaded(
        monkeypatch, subtests):
    """torch and OpenCV come down to ``max(2, target)``; absent or stubborn ones are not reported."""
    assert rc.MIN_LIBRARY_THREADS == 2

    def torch_module(count, sets):
        return _module("torch", get_num_threads=lambda: count,
                       set_num_threads=sets.append)

    def cv2_module(count, sets, *, obeys=True):
        def get():
            return sets[-1] if (sets and obeys) else count
        return _module("cv2", getNumThreads=get, setNumThreads=sets.append)

    cases = [
        # case, torch (count) or None, cv2 (count, obeys) or None, target,
        # expected lines, expected torch sets, expected cv2 sets
        ("both loaded, default floor", 16, (12, True), None,
         ["torch threads 16 → 2", "OpenCV threads 12 → 2"], [2], [2]),
        ("both loaded, explicit target", 16, (12, True), 4,
         ["torch threads 16 → 4", "OpenCV threads 12 → 4"], [4], [4]),
        ("a target below the floor is raised to it", 8, (9, True), 1,
         ["torch threads 8 → 2", "OpenCV threads 9 → 2"], [2], [2]),
        ("already at the floor", 2, (2, True), None, [], [], []),
        ("only OpenCV loaded", None, (9, True), 1,
         ["OpenCV threads 9 → 2"], None, [2]),
        ("OpenCV ignores the request", None, (8, False), 2, [], None, [2]),
        ("neither loaded", None, None, None, [], None, None),
    ]
    for case, torch_count, cv2_spec, target, lines, torch_sets, cv_sets in \
            cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            t_sets, c_sets = [], []
            if torch_count is None:
                patch.delitem(sys.modules, "torch", raising=False)
            else:
                patch.setitem(sys.modules, "torch",
                              torch_module(torch_count, t_sets))
            if cv2_spec is None:
                patch.delitem(sys.modules, "cv2", raising=False)
            else:
                patch.setitem(sys.modules, "cv2",
                              cv2_module(cv2_spec[0], c_sets,
                                         obeys=cv2_spec[1]))
            assert rc._lower_library_threads(target) == lines
            if torch_sets is not None:
                assert t_sets == torch_sets
            else:
                assert "torch" not in sys.modules
            if cv_sets is not None:
                assert c_sets == cv_sets
            else:
                assert "cv2" not in sys.modules

    with subtests.test(msg="libraries that will not say their count"), \
            monkeypatch.context() as patch:
        patch.setitem(sys.modules, "torch", _module(
            "torch", get_num_threads=_raise(RuntimeError("no")),
            set_num_threads=lambda n: pytest.fail("set blind")))
        patch.setitem(sys.modules, "cv2", _module(
            "cv2", getNumThreads=_raise(RuntimeError("no")),
            setNumThreads=lambda n: pytest.fail("set blind")))
        assert rc._lower_library_threads() == []


def test_clear_cpu_releases_exited_parked_threads_and_names_what_it_left(
        monkeypatch, subtests):
    """Details and note for parked threads, running jobs, a missing registry, and an idle machine."""
    from spacr.qt import bridge

    class Thread:
        def __init__(self, running):
            self.running = running

        def isRunning(self):  # noqa: N802 - Qt spelling
            return self.running

    def registry_with(handles):
        return lambda: types.SimpleNamespace(active=lambda: handles)

    pool_line = "Qt thread pool: idle threads retired (0 still working)"
    parked_note = ("1 thread(s) have not exited yet; they are parked, not "
                   "terminated, and will be released when they finish.")
    cases = [
        # case, parked threads, registry, pool lines, lowered lines,
        # threads before/after, details, note, parked left
        ("an exited parked thread is released",
         [Thread(False), Thread(True)], registry_with([]), [pool_line], [],
         (9, 8), ("1 parked thread(s) released", pool_line), parked_note, 1),
        ("a running job is named and left alone", [],
         registry_with([types.SimpleNamespace(app_key="measure")]), [],
         ["torch threads 8 → 2"], (5, 5), ("torch threads 8 → 2",),
         "1 spaCR job(s) are still running and were left alone.", 0),
        ("a missing registry still retires capacity", [],
         _raise(RuntimeError("gone")), [pool_line], [], (5, 5), (pool_line,),
         "", 0),
        ("nothing to retire", [], registry_with([]), [], [], (7, 7), (),
         "There was no idle capacity to retire.", 0),
        ("threads went away without a detail", [], registry_with([]), [], [],
         (7, 6), (), "", 0),
    ]
    for (case, parked, registry, pool, lowered, counts, details, note,
         parked_left) in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            targets = []
            readings = iter(counts)
            patch.setattr(bridge, "_PARKED_THREADS",
                          [(thread, object()) for thread in parked])
            patch.setattr(bridge, "registry", registry)
            patch.setattr(rc, "_retire_idle_pool_threads", lambda: list(pool))
            patch.setattr(rc, "_lower_library_threads",
                          lambda target=None: targets.append(target)
                          or list(lowered))
            patch.setattr(rc, "_thread_count", lambda: next(readings))
            reclaim = rc.clear_cpu(target_threads=3)
            assert (reclaim.action, reclaim.before, reclaim.after) == (
                "cpu",) + counts
            assert reclaim.details == details
            assert reclaim.note == note
            assert targets == [3]
            assert len(bridge._PARKED_THREADS) == parked_left

    with subtests.test(msg="the thread count is the interpreter's"):
        import threading

        assert rc._thread_count() == threading.active_count()

    with subtests.test(msg="the CPU summary counts threads"):
        assert rc.Reclaim("cpu", 9, 4).summary() == "CPU: 9 → 4 threads."
        assert rc.Reclaim("cpu", 1, 1).summary() == (
            "CPU: 1 thread, unchanged.")
