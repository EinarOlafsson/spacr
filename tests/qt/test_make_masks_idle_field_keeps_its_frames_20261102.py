"""An idle Make Masks screen never stops the spaCR field.

The maintainer, 2026-11-02: left on Make Masks with the spaCR field, the
field stutters or pauses every few seconds while nobody touches anything.

Measured with the real application on a private Xvfb/Openbox display at
1920x1080 and offscreen: paint timestamps of the window's field, a 5 ms
GUI-thread heartbeat, and the stacks of every thread sampled during each
gap. Two kinds of GUI step stopped the field on an idle screen.

* The hold (:class:`spacr.qt.widgets.ambient._HeldRows`). Once the field
  is calm it holds the rows over it, and the walk that finds what they
  cover renders every widget that paints into a probe the first time it
  meets it, all in one tick: 262-383 ms on the Mask screen and 238 ms on
  Make Masks with a folder open, 1.5 s after the field calmed. Most of it
  was single renders of containers with hidden pages that had never been
  shown, which ``QWidget.render`` lays out first (Make Masks' view tabs
  205 ms, the Mask screen 230-300 ms), and spin boxes (130-282 ms each
  the first time). Such containers are now walked as painting nothing of
  their own until their hidden pages are shown, spin and combo boxes are
  held whole like the other controls, and the remaining probes are
  rendered a few milliseconds per tick; the hold arms once the walk is
  complete.
* The collector's freeze (:mod:`spacr.qt.gc_policy`). The freezes after a
  screen build and after each background warm-up import collected
  everything not yet frozen first, 300 000-560 000 objects: 113-477 ms
  offscreen and 288-641 ms on the display, on Mask and Make Masks alike.
  They now freeze at once unless a window shown since the last freeze is
  gone or the collector was unfrozen, which is what that collection was
  there to catch.

Counted here on an idle screen: neither the hold's share of a field tick
while it arms nor the collector's freeze takes longer than two held frame
periods, and then, held and idle, the field delivers at least 90 % of the
frames its rate asks for.
"""
from __future__ import annotations

import gc
import time

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEventLoop, QObject, Qt, QTimer  # noqa: E402

pytestmark = pytest.mark.qt

#: The share of the frames asked for that the field must deliver.
DELIVERED = 0.90

#: Functions a warm-up import leaves alive for the next freeze: Make Masks'
#: warm-ups left 300 000-560 000 objects, mostly functions, dicts and code.
IMPORTED = 500_000

#: The longest GUI step allowed: two held field frame periods. A longer one
#: drops at least one whole held frame.
BUDGET_S = 2.0 / 20


class _Watch(QObject):
    """A 5 ms GUI heartbeat that records each gap and what the field asked."""

    def __init__(self, backdrop):
        """Watch ``backdrop``'s rate on every beat."""
        super().__init__()
        self._backdrop = backdrop
        self.gaps = []
        self.asked = 0.0
        self._last = None
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(5)
        self._timer.timeout.connect(self._beat)

    def start(self):
        """Start beating."""
        self._last = time.perf_counter()
        self._timer.start()

    def stop(self):
        """Stop beating."""
        self._timer.stop()

    def _beat(self):
        """Record the gap since the last beat and the frames it asked for."""
        now = time.perf_counter()
        gap = now - self._last
        self._last = now
        self.gaps.append(gap)
        self.asked += gap * self._backdrop._rate()


def _idle(seconds):
    """Run the real event loop for ``seconds`` without holding the GIL."""
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def _idle_until(done, timeout_s=60.0):
    """Run the event loop until ``done()`` or ``timeout_s``; whether it was."""
    deadline = time.perf_counter() + timeout_s
    while not done() and time.perf_counter() < deadline:
        _idle(0.1)
    return done()


def _a_folder_of_fields(folder):
    """Three 1024 x 1024 16-bit fields, the size of a real Make Masks image."""
    import imageio.v2 as imageio
    import numpy as np

    rng = np.random.default_rng(1102)
    folder.mkdir()
    for index in range(3):
        imageio.imwrite(folder / f"field_{index:02d}.tif",
                        rng.integers(500, 2500, (1024, 1024), dtype=np.uint16))
    return folder


@pytest.fixture
def opened(request, qtbot, qapp, monkeypatch, tmp_path):
    """The main window just opened on a module screen, with the spaCR field.

    The screen is ``request.param``, Make Masks unless a test names another;
    Make Masks opens a folder of fields, as it is left on screen.
    """
    from spacr.qt import preferences
    from spacr.qt.app import MainWindow, _freeze_what_survived
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets import ambient
    from spacr.qt.widgets.field_fade import invalidate_field_fade, uninstall_field_fade

    monkeypatch.setattr(preferences, "get_reduce_motion", lambda: False)
    monkeypatch.setattr(ambient, "_system_asks_for_stillness", lambda: False)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()
    apply_preferences_to_app(qapp)
    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1600, 1000)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    backdrop = window.window_backdrop()
    if not hasattr(backdrop, "_timer"):
        pytest.skip("the window backdrop is not the field in this setup")
    if backdrop.theme() != "data_art_impulse_lens":
        pytest.skip("the window backdrop is not the spaCR field here")
    _freeze_what_survived()
    qtbot.waitUntil(lambda: gc.get_freeze_count() > 0, timeout=20000)
    key = getattr(request, "param", "make_masks")
    window._on_nav_selected(key)
    for child in window.findChildren(QObject):
        if type(child).__name__ == "_TourOverlay":
            child.hide()
    if key == "make_masks":
        screen = window._screens[key]
        assert screen._open_folder(str(_a_folder_of_fields(tmp_path / "fields")))
        qtbot.waitUntil(lambda: not screen._loading
                        and screen._reader_worker is None
                        and screen._load_worker is None, timeout=30000)
    try:
        yield window, backdrop, key
    finally:
        backdrop._timer.stop()
        uninstall_field_fade(qapp)
        invalidate_field_fade()
        qapp.setFont(font)
        qapp.setPalette(palette)
        qapp.setStyleSheet(sheet)
        ambient._retire_shading_threads(timeout_s=3.0)


def _held_idle(backdrop, seconds=3.0):
    """Idle ``seconds`` with the field held; assert it kept moving.

    Counted in frames rather than in GUI steps: other idle work on the
    screen (a language pass over late captions, a deferred restyle) still
    takes 50-110 ms steps under a loaded test run, and they are not this
    test's to hold.
    """
    watch = _Watch(backdrop)
    before = backdrop.frames_painted
    watch.start()
    try:
        _idle(seconds)
    finally:
        watch.stop()
    painted = backdrop.frames_painted - before
    assert backdrop._held_rows.frame is not None, "the idle field let go of its hold"
    assert painted >= DELIVERED * watch.asked, (
        f"the held field painted {painted} of the {watch.asked:.0f} frames "
        f"its rate asked for in {seconds:.0f} idle seconds")


def _timed(monkeypatch, owner, name, steps):
    """Record into ``steps`` how long each call of ``owner.name`` takes."""
    original = getattr(owner, name)

    def timed(*args, **kwargs):
        """Time one call."""
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            steps.append(time.perf_counter() - started)

    monkeypatch.setattr(owner, name, timed)


@pytest.mark.parametrize("opened", ["mask", "make_masks"], indirect=True)
def test_the_hold_arms_without_stopping_the_field(opened, monkeypatch):
    """The first walk over a screen nobody has walked yet, each step timed.

    The first walk took 262-383 ms on Mask and 238 ms on Make Masks with a
    folder open, as one step of the field's tick.
    """
    from spacr.qt.widgets import ambient

    _window, backdrop, key = opened
    held = backdrop._held_rows
    steps = []
    _timed(monkeypatch, ambient._HeldRows, "tick", steps)
    held.stir()
    assert _idle_until(lambda: held.frame is not None), "the field never held"
    worst = max(steps)
    assert worst <= BUDGET_S, (
        f"the hold took {worst * 1000:.0f} ms of one field tick while it "
        f"armed on {key}; two held frames are {BUDGET_S * 1000:.0f} ms")
    assert _idle_until(lambda: _settled(_window, backdrop, key)), (
        f"{key} never finished opening")
    _held_idle(backdrop)


def _folds_prebuilt(screen) -> bool:
    """Whether Make Masks' idle build of its folded screens is over."""
    from spacr.qt.preferences import _screen_prewarm_allowed

    folded = getattr(screen, "_PREBUILD_FOLDED", ())
    if not folded or not _screen_prewarm_allowed()[0]:
        return True
    built = getattr(screen, "_fold_screens", {})
    laid_out = screen.__dict__.get("_laid_out_folds", set())
    return all(key in built and key in laid_out for key in folded)


def _settled(window, backdrop, key) -> bool:
    """Whether opening the screen is over: built, imported, frozen and held."""
    from spacr.qt import app as app_module
    from spacr.qt import gc_policy

    screen = window._screens.get(key)
    builder = getattr(screen, "__dict__", {}).get("_idle_prebuild")
    lock = getattr(app_module, "HEAVY_IMPORT_LOCK", None)
    return bool(screen is not None
                and (builder is None or not builder._work_left())
                and _folds_prebuilt(screen)
                and not (lock is not None and lock.locked())
                and not gc_policy._FREEZE_DUE.is_set()
                and backdrop._held_rows.frame is not None)


def _made_by_an_import(count):
    """``count`` live functions, as an imported module's contents are."""
    return [(lambda index=index: index) for index in range(count)]


def test_a_freeze_after_an_import_does_not_stop_the_field(opened, monkeypatch):
    """A warm-up import leaves its modules alive and asks for a freeze."""
    from spacr.qt import gc_policy
    from spacr.qt.app import _freeze_what_survived

    window, backdrop, key = opened
    assert _idle_until(lambda: _settled(window, backdrop, key)), (
        "Make Masks never finished opening")
    steps = []
    _timed(monkeypatch, gc_policy, "_collect_and_freeze", steps)
    imported = _made_by_an_import(IMPORTED)
    frozen_before = gc.get_freeze_count()
    _freeze_what_survived()
    assert _idle_until(
        lambda: gc.get_freeze_count() >= frozen_before + len(imported),
        timeout_s=20.0), "the freeze never happened"
    assert steps, "the freeze was not made by the collector's tick"
    assert max(steps) <= BUDGET_S, (
        f"the freeze took {max(steps) * 1000:.0f} ms of the GUI thread on an "
        f"idle Make Masks; two held frames are {BUDGET_S * 1000:.0f} ms")
    if backdrop._held_rows.frame is None:
        assert _idle_until(lambda: backdrop._held_rows.frame is not None)
    _held_idle(backdrop)
