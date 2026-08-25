"""Home when the things it reports on are unavailable.

Home is a dashboard over other people's state -- the run journal, the plate
queue, the GPU, the disk, the theme preferences -- and it is the first screen
the application shows. Every branch here is one of those readings failing.
None of them may keep the window from opening, and none may print a number it
did not measure: "n/a" and an empty list are answers, a traceback on launch is
not.
"""
from __future__ import annotations

import builtins
import os

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.widgets import home as home_mod
from spacr.qt.widgets.home import (
    HomePage, NewsPanel, QueuedPanel, RecentRunsPanel, RunningBanner,
    SystemPanel, TotalsPanel, _find_logo_pixmap, _fmt_elapsed, active_palette)

pytestmark = pytest.mark.qt

APPS = [("mask", "Mask", "Generate masks", "Core")]


def _block(name_fragment, monkeypatch):
    """Make ``import <name_fragment>...`` fail for the duration of a test."""
    real_import = builtins.__import__

    def _guarded(name, *args, **kwargs):
        if name_fragment in name:
            raise ImportError(f"{name} is unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded)


class _Gate:
    """A pause gate that is never paused."""

    @staticmethod
    def is_paused():
        return False


class _Handle:
    """A run handle with the surface Home reads."""

    def __init__(self, app_key="mask", running=True):
        self.app_key = app_key
        self.thread = None
        self.worker = None
        self.cancels = []
        self.supports_pause = False
        self.gate = _Gate()
        self.progress = None
        self.last_line = ""
        self._running = running

    def is_running(self):
        return self._running

    def request_cancel(self, why):
        self.cancels.append(why)

    def fraction(self):
        return None

    def elapsed(self):
        return 12.0


# ---------------------------------------------------------------------------
# Reading the theme and the assets
# ---------------------------------------------------------------------------

def test_a_theme_that_cannot_be_read_still_yields_a_palette(qapp, monkeypatch):
    """Home is the first screen shown, so it must open before preferences do.

    An unreadable preferences file would otherwise take the application down
    on launch. Dark is the fallback because a light page under an unset theme
    is the brighter mistake.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(preferences, "resolve_effective_theme", _explode)

    palette = active_palette()

    assert palette["bg"] == home_mod.palette_for("dark")["bg"]


def test_a_missing_logo_leaves_the_hero_without_one(qapp, monkeypatch):
    """The wordmark is drawn from a packaged file that may not be installed.

    A source checkout without the resources, or a trimmed wheel, must give a
    hero with no mark rather than a null pixmap scaled into the layout.
    """
    real_isfile = os.path.isfile

    def _no_logos(path):
        return (False if str(path).endswith(("logo_spacr.png",
                                             "logo_spacr_v1.png"))
                else real_isfile(path))

    monkeypatch.setattr(os.path, "isfile", _no_logos)

    assert _find_logo_pixmap() is None


def test_an_hour_long_run_is_reported_in_hours_and_minutes():
    """Past an hour, seconds are noise and the hour is the number that matters.

    A run reported as "5400s" is one a reader has to do arithmetic on before
    they can tell whether it is worth waiting for.
    """
    assert _fmt_elapsed(45) == "45s"
    assert _fmt_elapsed(125) == "2m 05s"
    assert _fmt_elapsed(5400) == "1h 30m"


# ---------------------------------------------------------------------------
# The running banner
# ---------------------------------------------------------------------------

@pytest.fixture
def banner(qtbot):
    widget = RunningBanner(lambda key: None, {"mask": "Mask"})
    qtbot.addWidget(widget)
    return widget


def test_a_banner_with_no_job_refreshes_to_nothing(banner):
    """The ticker fires on a hidden banner too, and there is nothing to read.

    Reading the handle regardless would raise on every tick after the job it
    was showing retired, and a stale subtitle would keep describing a run
    that has already finished.
    """
    banner.bind(_Handle())
    banner.refresh()
    assert banner._sub.text(), "a bound job puts something in the subtitle"
    before = banner._sub.text()

    banner.bind(None)
    banner.refresh()

    assert banner.isHidden()
    assert banner._sub.text() == before, "nothing was re-read"


def test_quitting_with_no_job_asks_nothing(banner, monkeypatch):
    """The button can outlive the job it was bound to.

    A click arriving after the run finished must not prompt about stopping a
    job that has already stopped -- a dialog about nothing is worse than no
    dialog.
    """
    import spacr.qt.shutdown as shutdown

    asked = []
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: asked.append(what))
    banner.bind(None)

    banner._on_quit()

    assert asked == []


def test_a_cancelled_quit_prompt_leaves_the_run_alone(banner, monkeypatch):
    """Choosing Cancel in the "how should this stop" prompt stops nothing.

    This is the one control on Home that can end somebody's analysis, so a
    dismissed dialog has to be exactly a no-op.
    """
    import spacr.qt.shutdown as shutdown

    handle = _Handle()
    banner.bind(handle)
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: shutdown.CANCEL)

    banner._on_quit()

    assert handle.cancels == []


def test_choosing_force_stops_the_job_outright(banner, monkeypatch):
    """Force still asks the worker to stop before the thread is taken away.

    A worker that IS still checking gets the chance to stop on its own terms
    in the moment before it is parked, which is the difference between a
    clean exit and a half-written file.
    """
    import spacr.qt.shutdown as shutdown

    handle = _Handle()
    banner.bind(handle)
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: shutdown.FORCE)

    banner._on_quit()

    assert handle.cancels == ["force quit from the Home screen"]


def test_a_graceful_quit_asks_the_run_to_stop_and_keeps_watching(
        banner, monkeypatch):
    """The watcher lives on the banner, not on the handle.

    The handle is retired the moment the job stops, and a timer parented to a
    dead object is a crash rather than a missed prompt.
    """
    import spacr.qt.shutdown as shutdown

    handle = _Handle()
    banner.bind(handle)
    monkeypatch.setattr(shutdown, "ask_how_to_quit",
                        lambda parent, what, detail: shutdown.GRACEFUL)

    banner._on_quit()

    assert handle.cancels == ["quit from the Home screen"]
    assert banner._quit_watcher.parent() is banner
    banner._quit_watcher.stop()


def test_force_stopping_a_job_with_no_thread_is_not_an_error(banner):
    """A handle whose thread has already gone has nothing left to park.

    The job finished between the prompt and the click, which is the good
    outcome; treating it as an error would report a failure to the user for
    getting what they asked for.
    """
    handle = _Handle()
    handle.thread = None

    RunningBanner._terminate(handle)

    assert handle.cancels == ["force quit from the Home screen"]


def test_force_stopping_survives_a_handle_that_refuses_to_cancel(banner,
                                                                 monkeypatch):
    """``request_cancel`` is a courtesy call and may fail on a dead handle.

    Force quit exists precisely for a job that is not responding, so a
    failure here must not stop the thread being parked -- which is the half
    that actually ends the run.
    """
    import spacr.qt.bridge as bridge

    class _Refusing(_Handle):
        def request_cancel(self, why):
            raise RuntimeError("the handle is already retired")

    drained = []
    monkeypatch.setattr(bridge, "drain_thread",
                        lambda thread, worker=None, timeout_ms=3000:
                            drained.append(thread))
    handle = _Refusing()
    handle.thread = object()

    RunningBanner._terminate(handle)

    assert drained == [handle.thread]


def test_force_stopping_a_thread_that_is_already_gone_is_not_an_error(
        banner, monkeypatch):
    """The drain call raises when the QThread's C++ half has been deleted.

    That is the job having finished, not a failure, so nothing is reported.
    """
    import spacr.qt.bridge as bridge

    def _gone(thread, worker=None, timeout_ms=3000):
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(bridge, "drain_thread", _gone)
    handle = _Handle()
    handle.thread = object()

    RunningBanner._terminate(handle)

    assert handle.cancels == ["force quit from the Home screen"]


# ---------------------------------------------------------------------------
# The aside panels, when what they report on cannot be read
# ---------------------------------------------------------------------------

def test_an_unreadable_plate_queue_shows_an_empty_queue(qtbot, monkeypatch):
    """A queue file that will not load is an empty queue on screen.

    Home must open without it; the queue is one panel of six and none of them
    is allowed to be the reason the window does not appear.
    """
    import spacr.qt.plate_queue as plate_queue

    def _explode(*args, **kwargs):
        raise OSError("the queue file is corrupt")

    monkeypatch.setattr(plate_queue, "PlateQueue", _explode)
    panel = QueuedPanel()
    qtbot.addWidget(panel)

    assert panel.queue_items() == []


def test_an_unreadable_journal_shows_no_recent_runs(qtbot, monkeypatch):
    """The run journal is thousands of manifests and any of them can be bad.

    An empty list is honest -- nothing could be read -- and it keeps the walk
    from taking down the page it was being drawn on.
    """
    import spacr.run_journal as run_journal

    def _explode(limit=4):
        raise OSError("the journal is unreadable")

    monkeypatch.setattr(run_journal, "recent_runs", _explode)
    panel = RecentRunsPanel()
    qtbot.addWidget(panel)

    assert panel.read() == []


def test_unreadable_journal_totals_are_reported_as_zeroes(qtbot, monkeypatch):
    """Every count is present and zero rather than the panel being absent.

    A panel with missing keys would raise while drawing; zeroes say "nothing
    could be counted", which is what happened.
    """
    import spacr.run_journal as run_journal

    def _explode():
        raise OSError("the journal is unreadable")

    monkeypatch.setattr(run_journal, "journal_totals", _explode)
    panel = TotalsPanel()
    qtbot.addWidget(panel)

    assert panel.read() == {"total_runs": 0, "mask_runs": 0,
                            "measure_runs": 0, "models_recorded": 0}


def test_a_machine_with_neither_nvml_nor_torch_reports_no_gpu(monkeypatch):
    """Two probes, and when both are unavailable the answer is "n/a".

    Reporting "0%" or "0.0 GB" would state a measurement of a device that was
    never found.
    """
    monkeypatch.setattr(home_mod, "_nvml", lambda: None)
    _block("torch", monkeypatch)

    assert SystemPanel.gpu_util() == "n/a"
    assert SystemPanel.gpu_vram() == "n/a"


def test_vram_falls_back_to_what_torch_has_allocated(monkeypatch):
    """Without NVML, torch still knows what this process is holding.

    It is not the card's total, and the panel says so by printing one number
    instead of the "used / total" pair NVML gives.
    """
    import torch

    monkeypatch.setattr(home_mod, "_nvml", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2_500_000_000)

    assert SystemPanel.gpu_vram() == "2.5 GB"


def test_a_disk_that_cannot_be_measured_reports_no_percentage(monkeypatch):
    """A home directory on a stale mount cannot be sized.

    "n/a" is the honest cell; a zero would read as an empty disk.
    """
    import shutil

    def _explode(path):
        raise OSError("the mount went away")

    monkeypatch.setattr(shutil, "disk_usage", _explode)

    assert SystemPanel.disk_used() == "n/a"


def test_replacing_the_news_content_releases_the_old_widget(qtbot):
    """The slot holds one widget, and the previous one has to go.

    Left parented it would keep painting under the new content and hold
    whatever it references for the life of the page.
    """
    panel = NewsPanel()
    qtbot.addWidget(panel)
    first = QLabel("release 1")
    second = QLabel("release 2")

    panel.set_content(first)
    panel.set_content(second)

    assert panel.content is second
    assert first.parent() is None


# ---------------------------------------------------------------------------
# The page itself
# ---------------------------------------------------------------------------

@pytest.fixture
def page(qtbot):
    widget = HomePage(APPS, lambda key: None)
    qtbot.addWidget(widget)
    return widget


def test_an_image_theme_paints_no_flat_page_fill(page, monkeypatch):
    """A wallpaper theme has a picture behind Home; a slab would hide it.

    The fill exists to replace the stylesheet background on the flat themes.
    Under an image theme there is nothing to replace.
    """
    import spacr.qt.preferences as preferences
    from spacr.qt.theme import IMAGE_THEMES

    page._ambient = None
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: sorted(IMAGE_THEMES)[0])

    assert page.page_fill() is None


def test_an_unreadable_theme_paints_no_page_fill(page, monkeypatch):
    """Without a theme there is no colour to fill with.

    Guessing one would paint a slab of the wrong colour over the whole page,
    which is the exact failure the fill exists to undo.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise OSError("preferences are unreadable")

    page._ambient = None
    monkeypatch.setattr(preferences, "resolve_effective_theme", _explode)

    assert page.page_fill() is None


def test_an_unreadable_theme_leaves_the_ambient_without_a_backdrop(monkeypatch):
    """Only the image themes have a wallpaper, and the lookup can fail.

    None means "paint over the flat window colour", which is what every other
    theme does anyway.
    """
    import spacr.qt.preferences as preferences

    def _explode(theme):
        raise OSError("the theme is unreadable")

    monkeypatch.setattr(preferences, "theme_background_path", _explode)

    assert HomePage._ambient_backdrop() is None


def test_nothing_is_discarded_when_the_ambient_module_is_absent(page,
                                                               monkeypatch):
    """If the import is what failed, nothing was ever constructed.

    Walking the children looking for a class that could not be imported is
    not possible, and there is nothing there to find.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    leftover = AmbientWidget(page)
    _block("ambient", monkeypatch)

    page._discard_ambient(leftover)

    assert leftover.parent() is page, "nothing was swept"


def test_an_aborted_ambient_install_leaves_no_live_child(page):
    """A widget parented before its wiring finished is still a live child.

    ``install_ambient`` parents first, so an installer that raises part way
    hands nothing back to unparent -- and an invisible leftover keeps its
    timer running behind every screen the user opens afterwards.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    leftover = AmbientWidget(page)

    page._discard_ambient(leftover)

    assert not [c for c in page.children() if isinstance(c, AmbientWidget)]


def test_a_leftover_that_refuses_to_be_freed_does_not_stop_the_sweep(page):
    """Both halves of the release are best-effort, and both can fail.

    A child whose C++ half is already gone raises on every call; the sweep
    has to reach the rest of the children regardless.
    """
    from spacr.qt.widgets.ambient import AmbientWidget

    class _Stubborn(AmbientWidget):
        def set_animating(self, on):
            raise RuntimeError("already deleted")

        def setParent(self, parent):
            raise RuntimeError("already deleted")

    stubborn = _Stubborn(page)
    ordinary = AmbientWidget(page)

    page._discard_ambient(stubborn)

    assert ordinary.parent() is None, "the sweep reached the rest"


def test_the_tabs_are_built_even_when_the_theme_cannot_be_read(qtbot,
                                                              monkeypatch):
    """Whether the theme is glass changes the tab styling, nothing more.

    Unreadable preferences must give the non-glass styling rather than a page
    with no tabs on it at all.
    """
    import spacr.qt.preferences as preferences

    real_resolve = preferences.resolve_effective_theme
    calls = {"n": 0}

    def _fails_once():
        calls["n"] += 1
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(preferences, "resolve_effective_theme", _fails_once)
    widget = HomePage(APPS, lambda key: None)
    qtbot.addWidget(widget)

    assert calls["n"] > 0
    assert widget._tabs.count() > 0


def test_unreadable_preferences_leave_the_panes_opaque(monkeypatch):
    """Opaque is what Home looked like before the transparency preference.

    A guessed alpha could put text on a background it cannot be read
    against, which is the one thing the preference's own floor prevents.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise OSError("preferences are unreadable")

    monkeypatch.setattr(preferences, "effective_pane_alpha", _explode)

    assert HomePage._pane_alpha() == 1.0


def test_a_build_with_no_importable_version_names_no_release(monkeypatch):
    """The News panel heads itself with a version only when there is one.

    "spaCR dev" says less than "News", and a package that cannot even be
    imported has no release to name.
    """
    _block("spacr", monkeypatch)

    assert HomePage._version() == ""


def test_closing_home_stops_its_ticker_even_when_the_registry_is_gone(page):
    """Releasing the registry connection may fail, and the ticker still stops.

    The run registry is process-wide and outlives no page in particular; a
    disconnect that raises must not leave a one-second timer running against
    a widget that is being torn down.
    """
    class _DeadSignal:
        @staticmethod
        def disconnect(_slot):
            raise RuntimeError("Internal C++ object already deleted.")

    class _DeadRegistry:
        changed = _DeadSignal()

    page._registry = _DeadRegistry()

    page.closeEvent(QCloseEvent())

    assert not page._ticker.isActive()


# ---------------------------------------------------------------------------
# Two pieces of Home that cannot run at all
# ---------------------------------------------------------------------------

def test_closing_home_shuts_down_its_journal_walk(qtbot, monkeypatch):
    """A journal walk must not outlive the page that asked for it.

    The walk reads thousands of manifests on a worker; a page closed while
    one is in flight leaves a thread running against a widget that is being
    torn down. ``closeEvent`` is meant to shut the runner down first.
    """
    widget = HomePage(APPS, lambda key: None)
    qtbot.addWidget(widget)
    stopped = []
    monkeypatch.setattr(widget._journal_jobs, "shutdown",
                        lambda: stopped.append(True))

    widget.closeEvent(QCloseEvent())

    assert stopped == [True]


def test_the_hero_labels_are_reachable_by_the_name_that_clears_them(page):
    """The masthead's type has to go transparent with the rest of the page.

    ``_clear_page_surfaces`` looks the hero up by object name to reach its
    labels; without that name the lookup finds nothing and a QLabel with no
    rule of its own takes the blanket window fill -- a black band across the
    masthead over the ambient backdrop.
    """
    hero = page.findChild(QWidget, "Hero")

    assert hero is not None
    assert hero.findChildren(QLabel)
