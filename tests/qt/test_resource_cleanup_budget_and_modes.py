"""The live-cache budget sweep, the mode wiring, and the disk check.

One behaviour table per unit of :mod:`spacr.qt.resource_cleanup` that the
background sweep and the performance modes go through: how cache owners are
inventoried, how the two budget settings are resolved, whether a run counts
as active, what one sweep evicts and reports, how the periodic tick logs and
installs, which cleanups each mode runs, how a registry change triggers the
pre-run cleanup, and which folders and drives the disk check reports.
Collaborators that would touch the machine (preferences, the run registry,
the clock, the drives) are stood in for; every other call is real.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

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


class _Owner:
    """A cache owner speaking the two-method budget protocol."""

    def __init__(self, rows, *, dropped=True, inventory_error=None,
                 name="fake.owner"):
        self.__name__ = name
        self._rows = rows
        self._dropped = dropped
        self._inventory_error = inventory_error
        self.drop_calls = []

    def cache_budget_entries(self):
        if self._inventory_error is not None:
            raise self._inventory_error
        return list(self._rows)

    def drop_cache_budget_entry(self, key):
        self.drop_calls.append(key)
        if isinstance(self._dropped, Exception):
            raise self._dropped
        return self._dropped


@pytest.fixture
def budget_globals():
    """Restore the module's sweep globals whatever a test does to them."""
    timer = rc._BUDGET_TIMER
    pending = rc._BUDGET_SWEEP_PENDING
    yield
    rc._BUDGET_TIMER = timer
    rc._BUDGET_SWEEP_PENDING = pending


# ---------------------------------------------------------------------------
# _collect_budget_entries
# ---------------------------------------------------------------------------

def test_cache_inventory_lists_droppable_rows_and_names_what_it_cannot(
        subtests):
    """Labels, sizes and error lines for whole, broken, malformed and half owners."""
    class Halfway:
        __name__ = "fake.halfway"

        def cache_budget_entries(self):
            return [("k", 4 * MB, 0.0, False)]

    whole_halfway = Halfway()
    whole_halfway.drop_cache_budget_entry = lambda key: True

    cases = [
        # case, owners, expected (label, megabytes, last_used, in_use), errors
        ("an inventory that raises costs only its owner",
         [_Owner([], inventory_error=ValueError("db is closed")),
          _Owner([("k", 1024 * 1024, 100.0, 0)])],
         [("fake.owner['k']", 1.0, 100.0, False)],
         ["fake.owner: inventory failed (db is closed)"]),
        ("a malformed row is named by its ordinal",
         [_Owner([("good", 2 * MB, 100.0, False), ("short",),
                  ("also-good", 1024, 50.0, True)])],
         [("fake.owner['good']", 2.0, 100.0, False),
          ("fake.owner['also-good']", 1024 / MB, 50.0, True)],
         ["fake.owner entry 1: invalid (not enough values to unpack "
          "(expected 4, got 1))"]),
        ("a negative size counts as nothing",
         [_Owner([(7, -50, 1, 1)])], [("fake.owner[7]", 0.0, 1.0, True)], []),
        ("an owner with no eviction is not inventoried", [Halfway()], [],
         ["fake.halfway: cache-budget protocol is incomplete"]),
        ("the same owner with an eviction is", [whole_halfway],
         [("fake.halfway['k']", 4.0, 0.0, False)], []),
    ]
    for case, owners, expected, errors in cases:
        with subtests.test(msg=case):
            records, found_errors = rc._collect_budget_entries(owners)
            assert [(row.label, row.megabytes, row.last_used, row.in_use)
                    for row in records] == expected
            assert [row.token for row in records] == [
                f"cache-{index:08d}" for index in range(len(expected))]
            assert found_errors == errors

    with subtests.test(msg="a long key is elided at 160 characters"):
        records, errors = rc._collect_budget_entries(
            [_Owner([("k" * 500, 1024, 1.0, False)])])
        assert errors == []
        assert records[0].label == "fake.owner[" + ("'" + "k" * 156)[:157] \
            + "...]"
        assert len(records[0].label) == len("fake.owner[]") + 160

    with subtests.test(msg="each drop binds its own key"):
        owner = _Owner([("a", 1, 0.0, False), ("b", 1, 0.0, False)])
        records, _errors = rc._collect_budget_entries([owner])
        assert [row.drop() for row in reversed(records)] == [True, True]
        assert owner.drop_calls == ["b", "a"]


# ---------------------------------------------------------------------------
# _budget_values
# ---------------------------------------------------------------------------

def test_budget_values_read_only_the_missing_preference_and_clamp_at_zero(
        monkeypatch, subtests):
    """An explicit value wins; a missing one is the preference, or the shipped default."""
    from spacr.qt import preferences
    from spacr.qt.memory_budget import (
        DEFAULT_CACHE_CEILING_MB,
        DEFAULT_IDLE_MINUTES,
    )

    seen = []

    def idle():
        seen.append("idle")
        return 41.0

    def ceiling():
        seen.append("ceiling")
        return 4242

    monkeypatch.setattr(preferences, "get_idle_minutes", idle)
    monkeypatch.setattr(preferences, "get_cache_ceiling_mb", ceiling)
    cases = [
        ("both from preferences", (None, None), (41.0, 4242.0),
         ["idle", "ceiling"]),
        ("ceiling given", (None, 512), (41.0, 512.0), ["idle"]),
        ("idle given", (3, None), (3.0, 4242.0), ["ceiling"]),
        ("both given", (7, 9), (7.0, 9.0), []),
        ("negatives clamp to zero", (-5, -9), (0.0, 0.0), []),
    ]
    for case, args, expected, asked in cases:
        with subtests.test(msg=case):
            seen.clear()
            assert rc._budget_values(*args) == expected
            assert seen == asked

    unreadable = [
        ("neither given", (None, None),
         (float(DEFAULT_IDLE_MINUTES), float(DEFAULT_CACHE_CEILING_MB))),
        ("idle given", (9, None), (9.0, float(DEFAULT_CACHE_CEILING_MB))),
        ("ceiling given", (None, 77), (float(DEFAULT_IDLE_MINUTES), 77.0)),
    ]
    for case, args, expected in unreadable:
        with subtests.test(msg=f"unreadable preferences, {case}"), \
                monkeypatch.context() as patch:
            patch.setitem(sys.modules, "spacr.qt.preferences", None)
            assert rc._budget_values(*args) == expected


# ---------------------------------------------------------------------------
# _a_run_is_active
# ---------------------------------------------------------------------------

def test_a_run_counts_as_active_unless_the_registry_says_idle(
        monkeypatch, subtests):
    """No bridge or no registry is idle; an unreadable registry is busy."""
    def bridge(registry):
        return _module("spacr.qt.bridge", registry=registry)

    def answering(active):
        return lambda: SimpleNamespace(active=lambda: active)

    cases = [
        ("no bridge imported", None, False),
        ("a registry that is not callable", bridge("not-callable"), False),
        ("an idle registry", bridge(answering([])), False),
        ("a running job", bridge(answering(["a-run"])), True),
        ("a registry that raises", bridge(_raise(RuntimeError("mutating"))),
         True),
    ]
    for case, module, expected in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if module is None:
                patch.delitem(sys.modules, "spacr.qt.bridge", raising=False)
            else:
                patch.setitem(sys.modules, "spacr.qt.bridge", module)
            assert rc._a_run_is_active() is expected
            if module is None:
                assert "spacr.qt.bridge" not in sys.modules


# ---------------------------------------------------------------------------
# sweep_memory_budget
# ---------------------------------------------------------------------------

def test_one_budget_sweep_evicts_idle_rows_and_accounts_for_each_outcome(
        monkeypatch, subtests):
    """Dropped labels, freed MB, errors and completeness for each eviction outcome."""
    monkeypatch.setattr(rc, "_a_run_is_active", lambda: False)
    monkeypatch.setattr(rc, "_cuda_cached", lambda: None)
    monkeypatch.setattr(rc, "clear_vram",
                        lambda **k: pytest.fail("no pressure, no VRAM call"))

    four = [(f"k{index}", MB, float(index), False) for index in range(4)]
    cases = [
        # case, owner, max_entries, drop calls, dropped, before, after,
        # errors, retained, complete
        ("an eviction that throws is an error line",
         _Owner([("hot", 8 * MB, 0.0, False)],
                dropped=RuntimeError("still mapped")), 64, ["hot"], (), 8.0,
         8.0, ("fake.owner['hot']: eviction failed (still mapped)",), (),
         True),
        ("an owner that declines is attempted, not counted",
         _Owner([("pinned", 4 * MB, 0.0, False)], dropped=False), 64,
         ["pinned"], (), 4.0, 4.0, (), (), True),
        ("an owner that agrees is counted",
         _Owner([("pinned", 4 * MB, 0.0, False)]), 64, ["pinned"],
         ("fake.owner['pinned']",), 4.0, 0.0, (), (), True),
        ("a bounded pass stops and says it is incomplete", _Owner(four), 2,
         ["k0", "k1"], ("fake.owner['k0']", "fake.owner['k1']"), 4.0, 2.0,
         (), (), False),
        ("an unbounded pass takes all four", _Owner(four), 64,
         ["k0", "k1", "k2", "k3"],
         tuple(f"fake.owner['k{index}']" for index in range(4)), 4.0, 0.0,
         (), (), True),
        ("an in-use row is retained, never evicted",
         _Owner([("busy", 2 * MB, 0.0, True), ("idle", MB, 0.0, False)]), 64,
         ["idle"], ("fake.owner['idle']",), 3.0, 2.0, (),
         ("fake.owner['busy']",), True),
        ("a fresh row under the ceiling stays",
         _Owner([("fresh", MB, 10_000.0, False)],
                dropped=AssertionError("evicted a fresh row")), 64, [], (),
         1.0, 1.0, (), (), True),
    ]
    for (case, owner, max_entries, calls, dropped, before, after, errors,
         retained, complete) in cases:
        with subtests.test(msg=case):
            sweep = rc.sweep_memory_budget(
                now=10_000.0, idle_minutes=1, ceiling_mb=4096,
                headroom_short=False, max_entries=max_entries,
                owners=[owner])
            assert owner.drop_calls == calls
            assert sweep.dropped == dropped
            assert (sweep.before_mb, sweep.after_mb) == pytest.approx(
                (before, after))
            assert sweep.freed_mb == pytest.approx(before - after)
            assert sweep.errors == errors
            assert sweep.retained_in_use == retained
            assert sweep.complete is complete
            assert sweep.pressure is False
            assert (sweep.models_released, sweep.vram_freed) == (0, 0)

    with subtests.test(msg="unstated headroom is measured against the "
                           "preference"), monkeypatch.context() as patch:
        from spacr.qt import preferences

        floors = []
        patch.setattr(preferences, "get_headroom_mb",
                      lambda: floors.append(1) or 0)
        owner = _Owner([("fresh", MB, 10_000.0, False)])
        sweep = rc.sweep_memory_budget(now=10_000.0, idle_minutes=1,
                                       ceiling_mb=4096, owners=[owner])
        assert sweep.pressure is False
        assert sweep.complete is True
        assert len(floors) == 1


# ---------------------------------------------------------------------------
# _budget_tick
# ---------------------------------------------------------------------------

def test_the_budget_tick_clears_its_flag_and_logs_only_news(
        monkeypatch, caplog, subtests, budget_globals):
    """One DEBUG line for a sweep that freed something, its errors, or its failure."""
    freed = rc.BudgetSweep(before_mb=40.0, after_mb=15.0, dropped=("a", "b"),
                           models_released=2, vram_freed=3 * MB)
    cases = [
        ("a sweep that freed something", lambda: freed,
         ["memory budget: host RSS 40.0 -> 15.0 MiB; 2 cache entries, 2 "
          "model references and 3.0 MB device VRAM released"]),
        ("only VRAM was released", lambda: rc.BudgetSweep(vram_freed=1024),
         ["memory budget: host RSS 0.0 -> 0.0 MiB; 0 cache entries, 0 "
          "model references and 1.0 KB device VRAM released"]),
        ("a sweep with only errors", lambda: rc.BudgetSweep(
            errors=("screen: inventory failed (db is closed)", "x: y")),
         ["memory budget sweep: screen: inventory failed (db is closed); "
          "x: y"]),
        ("a sweep that did nothing", rc.BudgetSweep, []),
        ("a sweep that throws", _raise(RuntimeError("owner list mutated")),
         ["the live-cache budget sweep failed"]),
    ]
    for case, sweep, messages in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(rc, "sweep_memory_budget", sweep)
            rc._BUDGET_SWEEP_PENDING = True
            caplog.clear()
            with caplog.at_level(logging.DEBUG, logger=rc.LOG.name):
                rc._budget_tick()
            assert rc._BUDGET_SWEEP_PENDING is False
            records = [r for r in caplog.records if r.name == rc.LOG.name]
            assert [r.getMessage() for r in records] == messages
            assert {r.levelno for r in records} <= {logging.DEBUG}


# ---------------------------------------------------------------------------
# install_budget_sweep
# ---------------------------------------------------------------------------

def test_install_budget_sweep_keeps_a_running_timer_and_replaces_a_dead_one(
        monkeypatch, subtests, qapp, budget_globals):
    """A live timer is kept; stopped or deleted ones are replaced; no app or Qt is ``False``."""
    class Deleted:
        @staticmethod
        def isActive():  # noqa: N802 - Qt spelling
            raise RuntimeError("Internal C++ object already deleted.")

    running = SimpleNamespace(isActive=lambda: True)
    stopped = SimpleNamespace(isActive=lambda: False)

    with subtests.test(msg="a running timer is left exactly as it was"):
        rc._BUDGET_TIMER = running
        assert rc.install_budget_sweep() is True
        assert rc._BUDGET_TIMER is running

    for case, previous in (("a stopped timer", stopped),
                           ("a deleted wrapper", Deleted()),
                           ("no timer yet", None)):
        with subtests.test(msg=f"{case} is replaced by a running one"):
            rc._BUDGET_TIMER = previous
            try:
                assert rc.install_budget_sweep() is True
                timer = rc._BUDGET_TIMER
                assert timer is not previous
                assert timer.isActive() is True
                assert timer.isSingleShot() is False
                assert timer.interval() == rc.BUDGET_SWEEP_INTERVAL_MS == 5000
                assert timer.objectName() == "LiveCacheBudgetSweep"
                assert timer.parent() is qapp
            finally:
                if rc._BUDGET_TIMER not in (previous, None):
                    rc._BUDGET_TIMER.stop()
                    rc._BUDGET_TIMER.deleteLater()

    refusals = [
        ("no application yet", "PySide6.QtWidgets", _module(
            "PySide6.QtWidgets",
            QApplication=SimpleNamespace(instance=lambda: None))),
        ("QtCore cannot be imported", "PySide6.QtCore", None),
    ]
    for case, name, replacement in refusals:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            rc._BUDGET_TIMER = None
            patch.setitem(sys.modules, name, replacement)
            assert rc.install_budget_sweep() is False
            assert rc._BUDGET_TIMER is None


# ---------------------------------------------------------------------------
# run_launch_cleanup / run_pre_run_cleanup
# ---------------------------------------------------------------------------

def test_each_mode_runs_its_own_cleanups_with_its_own_arguments(
        monkeypatch, subtests, caplog):
    """Which cleanups a mode runs at launch and before a run, and with what."""
    from spacr.qt import bridge, preferences

    def registry_with(count):
        handles = [SimpleNamespace(app_key=str(i)) for i in range(count)]
        return lambda: SimpleNamespace(active=lambda: handles)

    cases = [
        # case, mode, call, registry, expected actions, ram kwargs,
        # vram kwargs
        ("balanced does nothing at launch", "balanced", "launch", None, [],
         None, None),
        ("performance is gentle at launch", "performance", "launch", None,
         ["ram", "vram"], {"aggressive": False}, {"release_models": True}),
        ("extra performance cleans everything at launch",
         "extra_performance", "launch", None, ["ram", "vram", "cpu"],
         {"aggressive": True}, {"release_models": True}),
        ("an unreadable mode is balanced", RuntimeError("no store"),
         "launch", None, [], None, None),
        ("pre-run keeps the models", "extra_performance", "pre",
         registry_with(1), ["ram", "vram"], {"aggressive": True},
         {"release_models": False}),
        ("pre-run with an unreadable registry still runs",
         "extra_performance", "pre", _raise(RuntimeError("no")),
         ["ram", "vram"], {"aggressive": True}, {"release_models": False}),
        ("no pre-run cleanup while another run is in flight",
         "extra_performance", "pre", registry_with(2), [], None, None),
        ("performance has no pre-run cleanup", "performance", "pre",
         registry_with(1), [], None, None),
        ("an unreadable mode has no pre-run cleanup",
         RuntimeError("no store"), "pre", registry_with(1), [], None, None),
    ]
    for case, mode, call, registry, actions, ram, vram in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            seen = {}
            if isinstance(mode, Exception):
                patch.setattr(preferences, "get_spacr_mode", _raise(mode))
            else:
                patch.setattr(preferences, "get_spacr_mode", lambda: mode)
            patch.setattr(rc, "clear_ram", lambda **k: seen.setdefault(
                "ram", k) and rc.Reclaim("ram"))
            patch.setattr(rc, "clear_vram", lambda **k: seen.setdefault(
                "vram", k) and rc.Reclaim("vram"))
            patch.setattr(rc, "clear_cpu", lambda **k: seen.setdefault(
                "cpu", k) is not None and rc.Reclaim("cpu"))
            if registry is not None:
                patch.setattr(bridge, "registry", registry)
            results = (rc.run_launch_cleanup() if call == "launch"
                       else rc.run_pre_run_cleanup("mask_generator"))
            assert [r.action for r in results] == actions
            assert seen.get("ram") == ram
            assert seen.get("vram") == vram
            assert ("cpu" in seen) is ("cpu" in actions)

    with subtests.test(msg="a cleanup that freed something is INFO"), \
            monkeypatch.context() as patch:
        patch.setattr(preferences, "get_spacr_mode",
                      lambda: "extra_performance")
        patch.setattr(bridge, "registry", registry_with(1))
        patch.setattr(rc, "clear_ram",
                      lambda **k: rc.Reclaim("ram", 2 * MB, MB))
        patch.setattr(rc, "clear_vram", lambda **k: rc.Reclaim("vram", 5, 5))
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=rc.LOG.name):
            rc.run_pre_run_cleanup("mask_generator")
        assert [(r.levelno, r.getMessage()) for r in caplog.records
                if r.name == rc.LOG.name] == [
            (logging.INFO, "before mask_generator: RAM: freed 1.0 MB."),
            (logging.DEBUG,
             "before mask_generator: VRAM: freed nothing measurable."),
        ]


# ---------------------------------------------------------------------------
# _on_registry_changed
# ---------------------------------------------------------------------------

def test_only_a_newly_registered_run_triggers_the_pre_run_cleanup(
        monkeypatch, subtests):
    """New runs are cleaned for once; finishes and repeats queue a budget sweep instead."""
    from spacr.qt import bridge

    handles = []
    cleaned, swept = [], []
    monkeypatch.setattr(rc, "_SEEN_RUNS", set())
    monkeypatch.setattr(bridge, "registry",
                        lambda: SimpleNamespace(active=lambda: list(handles)))
    monkeypatch.setattr(rc, "run_pre_run_cleanup",
                        lambda key="": cleaned.append(key))
    monkeypatch.setattr(rc, "_request_budget_sweep",
                        lambda: swept.append(len(swept)))

    first = SimpleNamespace(app_key="mask_generator")
    second = SimpleNamespace(app_key="measure")
    nameless = SimpleNamespace()
    steps = [
        # case, active handles, cleaned so far, sweeps so far, seen size
        ("nothing running queues a sweep", [], [], 1, 0),
        ("a new run is cleaned for", [first], ["mask_generator"], 1, 1),
        ("the same run again is not", [first], ["mask_generator"], 2, 1),
        ("it finishes and another starts", [second],
         ["mask_generator", "measure"], 2, 1),
        ("a run with no key is cleaned for by name ''", [second, nameless],
         ["mask_generator", "measure", ""], 2, 2),
        ("everything finished", [], ["mask_generator", "measure", ""], 3, 0),
    ]
    for case, active, expected_cleaned, sweeps, seen_size in steps:
        with subtests.test(msg=case):
            handles[:] = active
            rc._on_registry_changed()
            assert cleaned == expected_cleaned
            assert len(swept) == sweeps
            assert len(rc._SEEN_RUNS) == seen_size

    with subtests.test(msg="an unreadable registry does nothing at all"), \
            monkeypatch.context() as patch:
        patch.setattr(bridge, "registry", _raise(RuntimeError("no")))
        rc._on_registry_changed()
        assert (len(cleaned), len(swept)) == (3, 3)

    with subtests.test(msg="a failing cleanup still marks the run seen"), \
            monkeypatch.context() as patch:
        attempted = []

        def explode(key=""):
            attempted.append(key)
            raise RuntimeError("out of memory")

        patch.setattr(rc, "_SEEN_RUNS", set())
        patch.setattr(rc, "run_pre_run_cleanup", explode)
        handles[:] = [SimpleNamespace(app_key="x")]
        rc._on_registry_changed()
        rc._on_registry_changed()
        assert attempted == ["x"]
        assert len(rc._SEEN_RUNS) == 1


# ---------------------------------------------------------------------------
# project_paths
# ---------------------------------------------------------------------------

def test_project_paths_are_existing_remembered_folders_then_home_and_temp(
        monkeypatch, subtests, tmp_path):
    """Remembered folders that exist, once each, then home and temp; failures cost one entry."""
    from spacr.qt import app as app_mod
    from spacr.qt import prefs

    real_expanduser = os.path.expanduser
    home = os.path.abspath(real_expanduser("~"))
    temp = os.path.abspath(tempfile.gettempdir())
    gone = tmp_path / "deleted"
    kept = tmp_path / "kept"
    kept.mkdir()

    monkeypatch.setattr(rc, "_the_gui_thread_is_asking", lambda: False)
    monkeypatch.setattr(app_mod, "APPS", [("audit_app",)])

    def expanduser(text):
        if text == "~broken":
            raise RuntimeError("no such user")
        return real_expanduser(text)

    cases = [
        ("deleted, blank and repeated folders are dropped",
         dict(last=lambda key: str(gone),
              recent=lambda key, limit=3: [str(kept), "", None, str(kept)]),
         [str(kept), home, temp]),
        ("unreadable preferences still report home and temp",
         dict(last=_raise(RuntimeError("settings are locked"))),
         [home, temp]),
        ("a folder that cannot be expanded is dropped",
         dict(last=lambda key: "~broken",
              recent=lambda key, limit=3: [str(kept)],
              expanduser=expanduser),
         [str(kept), home, temp]),
        ("a temp directory that cannot be named is skipped",
         dict(last=lambda key: None, recent=lambda key, limit=3: [],
              gettempdir=_raise(OSError("no TMPDIR"))),
         [home]),
    ]
    for case, setup, expected in cases:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            patch.setattr(prefs, "get_last_source", setup["last"])
            patch.setattr(prefs, "get_recent_sources", setup.get(
                "recent", lambda key, limit=3: []))
            if "expanduser" in setup:
                patch.setattr(os.path, "expanduser", setup["expanduser"])
            if "gettempdir" in setup:
                patch.setattr(tempfile, "gettempdir", setup["gettempdir"])
            assert rc.project_paths() == expected


# ---------------------------------------------------------------------------
# disk_report
# ---------------------------------------------------------------------------

def test_disk_report_is_one_line_per_drive_and_counts_unreadable_folders(
        monkeypatch, subtests, tmp_path):
    """Missing and unmeasurable folders are counted; two folders on one drive are one line."""
    import shutil

    monkeypatch.setattr(rc, "_the_gui_thread_is_asking", lambda: False)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    usage = shutil.disk_usage(str(tmp_path))

    cases = [
        ("a folder that is gone is counted",
         [str(tmp_path / "unplugged"), str(tmp_path)], None, [str(tmp_path)],
         "1 folder(s) could not be read."),
        ("two folders on one drive are one line",
         [str(tmp_path / "a"), str(tmp_path / "b")], None,
         [str(tmp_path / "a")], ""),
        ("a drive that cannot be measured is counted", [str(tmp_path)],
         _raise(OSError("no such device")), [],
         "1 folder(s) could not be read."),
        ("nothing to read", [], None, [],
         "No project folder is known yet — open a module and choose a source "
         "folder, and this will report that drive."),
    ]
    gui = [(f"on the GUI thread: {case}", paths, usage_, expected, note)
           for case, paths, usage_, expected, note in cases[:2]]
    gui.append(("on the GUI thread: a folder listed twice is read once",
                [str(tmp_path), str(tmp_path)], None, [str(tmp_path)], ""))
    for case, paths, disk_usage, expected_paths, note in cases + gui:
        with subtests.test(msg=case), monkeypatch.context() as patch:
            if case.startswith("on the GUI thread"):
                patch.setattr(rc, "_the_gui_thread_is_asking", lambda: True)
            if disk_usage is not None:
                patch.setattr(rc.shutil, "disk_usage", disk_usage)
            report = rc.disk_report(paths)
            assert [entry.path for entry in report.entries] == expected_paths
            assert report.note == note
            assert [entry.total for entry in report.entries] == [
                usage.total] * len(expected_paths)
    assert rc.shutil is shutil
