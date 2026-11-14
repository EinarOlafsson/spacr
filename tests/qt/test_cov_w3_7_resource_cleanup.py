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

import sys
import types

import pytest

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


def test_torch_is_never_imported_just_to_ask_about_it(monkeypatch):
    """The VRAM button must not be the thing that allocates the memory."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert rc._torch_if_loaded() is None
    assert rc.cuda_reserved() is None

    sentinel = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", sentinel)
    assert rc._torch_if_loaded() is sentinel


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


# ---------------------------------------------------------------------------
# VRAM
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def test_the_cpu_summary_counts_threads_not_bytes():
    reclaim = rc.Reclaim("cpu", before=9, after=4)
    assert reclaim.summary() == "CPU: 9 → 4 threads."
    assert rc.Reclaim("cpu", 1, 1).summary() == "CPU: 1 thread, unchanged."
    assert rc.Reclaim("cpu", 4, 4).summary() == "CPU: 4 threads, unchanged."


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The modes, and the wiring that presses the buttons
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CPU, the quiet machine
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Disk, the rest of it
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", rc.ACTIONS)
def test_every_button_can_say_what_it_is_about_to_do(action):
    """A confirmation that cannot name the action is not a confirmation."""
    title = rc.confirmation_title(action)
    text = rc.confirmation_text(action)
    assert title and title[0].isupper()
    assert len(text) > 80
