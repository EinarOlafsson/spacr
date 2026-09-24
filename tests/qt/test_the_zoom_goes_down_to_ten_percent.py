"""413 -- the Zoom floor is 10 %, not 75 %.

The request: "i would like to change the font zoom floor from 75% to 10%".
One constant, ``preferences.FONT_SCALE_MIN``, is the floor for the
Preferences slider, the stored value and the Z + wheel gesture, and these
tests hold each of those paths to it.

A FLOOR THAT IS ONLY A NUMBER PROVES NOTHING. Three more floors sat further
down the pipeline: ``theme.font_px`` and the stylesheet's size table would
not go under 6 px, the close mark would not go under 12 px, and the live
gesture would not set text under 6 px. With only the constant changed, every
text role would have drawn at 6 px from 45 % down, so 10 % would have looked
exactly like 45 %. That is why the sizes checked here are the ones Qt
resolved for a real widget, not the number that was stored.

The floor kept is Qt's own: a font of at least one pixel, because
``QFont.setPixelSize(0)`` is refused with a warning and a style sheet's
``font-size: 0px`` falls back to the inherited size, which draws the text
LARGER rather than smaller.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import (QEvent, QPoint, QPointF, QSettings, Qt,  # noqa: E402
                            qInstallMessageHandler)
from PySide6.QtGui import QFontInfo, QKeyEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel, QSlider,  # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt import live_zoom  # noqa: E402
from spacr.qt import preferences as prefs  # noqa: E402

pytestmark = pytest.mark.qt

#: The request, stated once. Everything else reads the constant.
REQUESTED_FLOOR = 0.10

#: Where the floor was. Only the tests that prove text now goes below what
#: the old floor drew read it.
OLD_FLOOR = 0.75

#: Five percent a notch, from 100 % to the floor.
NOTCHES_FROM_100_TO_10 = 18


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _never_the_real_preferences():
    """Refuse to run against the preferences a person actually uses.

    Every test here stores a font scale, and a 10 % left in
    ``~/.config/spacr/qt.conf`` opens spaCR with text one pixel tall. The
    conftest's ``pytest_configure`` moves the store to a temporary
    directory before any ``QSettings`` exists; this checks that it did,
    rather than trusting it, because ``SPACR_TEST_KEEP_REAL_CONFIG`` turns
    that move off.
    """
    store = QSettings(prefs._ORG, prefs._APP).fileName()
    real = str(Path.home() / ".config" / "spacr")
    assert not store.startswith(real), (
        f"the preference store is the real one ({store}); refusing to write")
    yield store


@pytest.fixture
def no_restyle(monkeypatch):
    """Keep the gesture's settle from restyling the shared application.

    The settle rebuilds every live widget's sheet, which for the
    session-scoped ``qapp`` is every later test that measures a pixel. The
    live-zoom tests assert on the scale and on the fonts the gesture set,
    neither of which needs the rebuild.
    """
    applied = []
    monkeypatch.setattr(prefs, "apply_preferences_to_app",
                        lambda app=None: applied.append(app))
    return applied


@pytest.fixture
def page(qt_theme_applied):
    """A shown widget with a title, styled by the application stylesheet."""
    root = QWidget()
    layout = QVBoxLayout(root)
    title = QLabel("Plate 3")
    title.setObjectName("ScreenTitle")
    layout.addWidget(title)
    root.show()
    QApplication.processEvents()
    yield root, title
    root.hide()
    root.deleteLater()
    QApplication.processEvents()


@pytest.fixture
def qt_messages():
    """Every message Qt prints while the test runs, restoring the old handler."""
    seen = []
    previous = qInstallMessageHandler(
        lambda _mode, _context, text: seen.append(str(text)))
    try:
        yield seen
    finally:
        qInstallMessageHandler(previous)


@pytest.fixture
def main_window_at(qapp, qtbot, monkeypatch):
    """Build real MainWindows at a stored scale, and leave the app at 100 %.

    The restore is here and not left to the conftest: its teardown puts the
    stored number back, but the per-window sheets ``apply_preferences_to_app``
    wrote at 10 % stay on every live widget in the process. This fixture is
    torn down before ``qtbot`` closes the windows, so the rebuild runs on
    widgets that still exist.

    The backdrop and GL are off: neither is what this measures, and the
    machine running it may have no GPU.
    """
    monkeypatch.setenv("SPACR_NO_BACKDROP", "1")
    monkeypatch.setenv("SPACR_NO_GL", "1")
    windows = []

    def build(scale):
        from spacr.qt.app import MainWindow

        prefs.set_font_scale(scale)
        prefs.apply_preferences_to_app(qapp)
        window = MainWindow()
        qtbot.addWidget(window)
        windows.append(window)
        window.resize(1400, 900)
        window.show()
        _pump(qapp)
        assert window.open_module("mask") == "mask"
        _pump(qapp, 80)
        return window

    yield build
    if prefs.get_font_scale() != 1.0:
        prefs.set_font_scale(1.0)
        prefs.apply_preferences_to_app(qapp)
    for window in windows:
        window.close()
    _pump(qapp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pump(app, rounds: int = 40) -> None:
    for _ in range(rounds):
        app.processEvents()


def _wheel(notches: int):
    """One wheel event carrying ``notches`` detents, in Qt's 120ths."""
    return QWheelEvent(
        QPointF(10, 10), QPointF(10, 10), QPoint(0, 0),
        QPoint(0, int(notches * 120)), Qt.NoButton, Qt.NoModifier,
        Qt.NoScrollPhase, False)


def _z(kind):
    return QKeyEvent(kind, Qt.Key_Z, Qt.NoModifier, "z", False)


def _hold_z(zoom, widget) -> None:
    zoom.eventFilter(widget, _z(QEvent.KeyPress))
    assert zoom._held, "Z did not arm the gesture"


def _rendered_px(widget) -> int:
    """The pixel size Qt resolved for the widget's font."""
    return QFontInfo(widget.font()).pixelSize()


def _sheet_carrier(label, window):
    """The widget whose style sheet sets this label's font size, or None."""
    widget = label
    while widget is not None and widget is not window:
        if "font-size" in (widget.styleSheet() or ""):
            return widget
        widget = widget.parentWidget()
    return None


def _visible_label_sizes(window) -> dict:
    """Measure label fonts, identifying changing resource readings by meter.

    CPU/RAM values can change between snapshots. Their captions identify the
    readouts without dropping these labels from the font/visibility checks.
    """
    from spacr.qt.widgets.usage_bar import UsageBar

    sizes = {}
    for label in window.findChildren(QLabel):
        if label.isVisible() and label.text().strip():
            parent = label.parentWidget()
            key = (("UsageBarPercentage", parent._label.text())
                   if isinstance(parent, UsageBar) and label is parent._pct
                   else (label.objectName(), label.text()[:40]))
            sizes.setdefault(key, (_rendered_px(label),
                                   _sheet_carrier(label, window)))
    return sizes


def test_usage_readings_keep_their_identity_when_the_values_change(qtbot):
    from spacr.qt.widgets.usage_bar import UsageBar

    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    bars = [UsageBar(name) for name in ("CPU", "RAM")]
    for bar in bars:
        layout.addWidget(bar)
        bar.set_value(17)
    root.show()
    QApplication.processEvents()
    before = _visible_label_sizes(root)
    for bar, value in zip(bars, (91, 42)):
        bar.set_value(value)
    QApplication.processEvents()
    after = _visible_label_sizes(root)

    assert len(before) == len(after) == 4
    assert before.keys() == after.keys()
    assert {key: value[0] for key, value in before.items()} == {
        key: value[0] for key, value in after.items()}


# ---------------------------------------------------------------------------
# The floor itself, on each path that sets it
# ---------------------------------------------------------------------------

def test_the_floor_is_ten_percent():
    """The request, pinned once, so every test below can read the constant."""
    assert prefs.FONT_SCALE_MIN == pytest.approx(REQUESTED_FLOOR)


def test_the_preferences_slider_goes_down_to_ten(qtbot, qt_theme_applied):
    """The slider is the control a user reaches for, so its bottom is the floor.

    Ten, and from the constant: the range is ``int(FONT_SCALE_MIN * 100)``,
    so a floor typed into the dialog as well would be a second place to
    forget. ``int`` is safe at this value -- ``0.10 * 100`` is
    ``10.000000000000002`` -- and would not be at every value, which is why
    the literal is checked too.
    """
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    slider = dialog.findChild(QSlider, "FontScale")
    assert slider is not None, "the Zoom slider is no longer named FontScale"
    assert slider.minimum() == 10
    assert slider.minimum() == round(prefs.FONT_SCALE_MIN * 100)
    assert slider.maximum() == round(prefs.FONT_SCALE_MAX * 100)


def test_a_stored_ten_percent_reads_back_as_ten_percent_after_a_restart(
        qt_theme_applied, _never_the_real_preferences):
    """Read back by a separate Python process, because that is what a restart is.

    The clamp runs on the way in and on the way out, and a 75 % floor left
    on either side turns the 10 % somebody chose into 75 % the next time
    spaCR opens -- silently, because the slider would then show 75 as if it
    had been chosen. Reading it in this process would also go through the
    ``QSettings`` cache the write just filled, and prove less.

    The child is pointed at this test's sandboxed store with the same
    ``QSettings.setPath`` the conftest uses, and has to report that file
    back: a child that silently read some other store would otherwise pass
    or fail for reasons unrelated to the floor. Where the native store
    cannot be redirected (macOS, Windows) the conftest sandboxes it another
    way that a child does not inherit, so the check is skipped there rather
    than pointed at a real preference file.
    """
    prefs.set_font_scale(REQUESTED_FLOOR)
    written = QSettings(prefs._ORG, prefs._APP)
    written.sync()
    store = written.fileName()
    if sys.platform in ("darwin", "win32"):
        pytest.skip("a child process cannot be pointed at this platform's "
                    "sandboxed native store")
    root = str(Path(store).parent.parent)
    tree = str(Path(prefs.__file__).resolve().parents[2])
    probe = (
        "import sys\n"
        "from PySide6.QtCore import QSettings\n"
        "for fmt in (QSettings.NativeFormat, QSettings.IniFormat):\n"
        "    QSettings.setPath(fmt, QSettings.UserScope, sys.argv[1])\n"
        "from spacr.qt import preferences as p\n"
        "print(QSettings(p._ORG, p._APP).fileName())\n"
        "print(repr(p.get_font_scale()))\n"
        "print(p.__file__)\n")
    env = dict(os.environ, PYTHONPATH=tree, QT_QPA_PLATFORM="offscreen")
    done = subprocess.run([sys.executable, "-c", probe, root], env=env,
                          capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-2000:]
    child_store, scale, module = done.stdout.strip().splitlines()[-3:]
    assert child_store == store, (
        f"the restart read {child_store}, not the store written at {store}")
    assert module.startswith(tree), f"the restart imported {module}"
    assert float(scale) == pytest.approx(REQUESTED_FLOOR)
    assert prefs.get_font_scale() == pytest.approx(REQUESTED_FLOOR)


# ---------------------------------------------------------------------------
# The live gesture
# ---------------------------------------------------------------------------

def test_the_wheel_stops_at_ten_percent(page, no_restyle):
    """Down to the floor and no further, with the text shrinking all the way.

    Eighteen 5 % notches from 100 %, then the notches a user keeps turning
    after the text has stopped: they must neither overshoot the floor nor
    move the text again. The text is measured at each step because the
    gesture had its own 6 px floor, which would have stopped the title
    shrinking long before the scale stopped.
    """
    root, title = page
    prefs.set_font_scale(1.0)
    zoom = live_zoom.LiveZoomFilter()
    try:
        start_px = _rendered_px(title)
        _hold_z(zoom, root)
        seen = {}
        for _ in range(NOTCHES_FROM_100_TO_10):
            zoom.eventFilter(root, _wheel(-1))
            seen[round(zoom._live_scale, 2)] = _rendered_px(title)
        assert zoom._live_scale == pytest.approx(REQUESTED_FLOOR)
        at_floor = _rendered_px(title)

        for _ in range(6):
            zoom.eventFilter(root, _wheel(-1))
        assert zoom._live_scale == pytest.approx(prefs.FONT_SCALE_MIN)
        assert _rendered_px(title) == at_floor

        assert at_floor == max(1, round(start_px * REQUESTED_FLOOR))
        assert at_floor < seen[OLD_FLOOR], (
            f"{at_floor} px at 10 % is not below the {seen[OLD_FLOOR]} px "
            "the old floor drew")
    finally:
        zoom.settle()


def test_the_wheel_climbs_back_from_ten_percent_to_one_hundred(page,
                                                               no_restyle):
    """The way out of 10 % is the way in: eighteen notches up is 100 % again.

    A floor nobody can zoom back out of is a trap, and the gesture is the
    control a person at 10 % can still operate -- the Preferences text is
    one pixel tall.
    """
    root, _title = page
    prefs.set_font_scale(REQUESTED_FLOOR)
    zoom = live_zoom.LiveZoomFilter()
    try:
        _hold_z(zoom, root)
        for _ in range(NOTCHES_FROM_100_TO_10):
            zoom.eventFilter(root, _wheel(+1))
        assert zoom._live_scale == pytest.approx(1.0)
        zoom.eventFilter(root, _z(QEvent.KeyRelease))
        assert prefs.get_font_scale() == pytest.approx(1.0)
    finally:
        zoom.settle()


# ---------------------------------------------------------------------------
# No hidden floor
# ---------------------------------------------------------------------------

def test_text_at_ten_percent_renders_below_what_the_old_floor_drew(
        qt_theme_applied):
    """Measured on real labels under the real stylesheet, built from the preference.

    The sheet is built from ``get_font_scale()``, as
    ``apply_preferences_to_app`` builds it, so a clamp anywhere on the way
    shows up. Two roles are measured because a hidden floor flattens them:
    at a 6 px floor body text and a 30 px heading both drew at 6 px, and
    at 10 % they must still differ.
    """
    from spacr.qt.theme import FONT_SIZE, close_mark_font_px, font_px, stylesheet

    def rendered(scale):
        prefs.set_font_scale(scale)
        root = QWidget()
        root.setStyleSheet(stylesheet(font_scale=prefs.get_font_scale()))
        column = QVBoxLayout(root)
        body = QLabel("Nucleus channel")
        heading = QLabel("Mask Generation")
        heading.setObjectName("DisplayHeading")
        column.addWidget(body)
        column.addWidget(heading)
        root.show()
        QApplication.processEvents()
        sizes = (_rendered_px(body), _rendered_px(heading),
                 font_px("body"), close_mark_font_px())
        root.hide()
        root.deleteLater()
        QApplication.processEvents()
        return sizes

    old_body, old_heading, _old_font_px, _old_mark = rendered(OLD_FLOOR)
    body, heading, body_font_px, mark = rendered(REQUESTED_FLOOR)

    assert body < old_body and heading < old_heading
    assert body == max(1, round(FONT_SIZE["body"] * REQUESTED_FLOOR))
    assert heading == round(FONT_SIZE["display"] * REQUESTED_FLOOR)
    assert heading > body, "a floor flattened the heading onto body text"
    # The per-widget sheets and the close mark size through these two.
    assert body_font_px == body
    assert mark == max(1, round(body * 1.15))


# ---------------------------------------------------------------------------
# The application at 10 %
# ---------------------------------------------------------------------------

def test_a_main_window_survives_ten_percent_and_comes_back(
        main_window_at, qt_messages, qtbot, qapp, caplog):
    """Open spaCR with 10 % stored, use Mask and Preferences, wheel back to 100 %.

    SURVIVES means no exception (pytest-qt fails the test on one raised in
    a Qt callback, and the gesture's settle logs one it swallows, which
    ``caplog`` sees), and no "Pixel size <= 0" / "Point size <= 0" from Qt,
    which is what a zero-pixel font produces.

    COMES BACK includes self-styled chips, the run instruction and the
    first-run tour, not only labels inheriting the window stylesheet.
    Their effective sizes must match a window built at 100 %.
    """
    caplog.set_level(logging.ERROR, logger="spacr")

    reference = main_window_at(1.0)
    at_100 = _visible_label_sizes(reference)
    reference.close()
    _pump(qapp)

    window = main_window_at(REQUESTED_FLOOR)
    assert prefs.get_font_scale() == pytest.approx(REQUESTED_FLOOR)
    at_10 = _visible_label_sizes(window)

    rebuilt = [key for key in at_10 if key in at_100]
    assert len(rebuilt) >= 30, (
        f"only {len(rebuilt)} labels to compare; the test would prove little")
    not_smaller = [(key, at_100[key][0], at_10[key][0]) for key in rebuilt
                   if at_10[key][0] >= at_100[key][0]]
    assert not_smaller == [], f"did not shrink at 10 %: {not_smaller[:5]}"
    assert max(at_10[key][0] for key in rebuilt) <= 3

    dialog = prefs.PreferencesDialog(window)
    qtbot.addWidget(dialog)
    dialog.show()
    _pump(qapp)
    slider = dialog.findChild(QSlider, "FontScale")
    assert (slider.minimum(), slider.value()) == (10, 10)
    dialog.reject()
    _pump(qapp)

    zoom = live_zoom.LiveZoomFilter()
    _hold_z(zoom, window)
    for _ in range(NOTCHES_FROM_100_TO_10):
        zoom.eventFilter(window, _wheel(+1))
    zoom.eventFilter(window, _z(QEvent.KeyRelease))
    _pump(qapp, 80)
    assert prefs.get_font_scale() == pytest.approx(1.0)

    back = _visible_label_sizes(window)
    not_restored = [(key, at_100[key][0], back[key][0]) for key in rebuilt
                    if key in back and back[key][0] != at_100[key][0]]
    own_styles = [(key, back[key][1].styleSheet()[:240])
                  for key, _old, _new in not_restored if back[key][1] is not None]
    assert not_restored == [], (
        f"not back at 100 %: {not_restored}; nearest font sheets: {own_styles}")
    assert all(key in back for key in rebuilt), "labels vanished on the way"

    zero_size = [text for text in qt_messages if "size <= 0" in text.lower()]
    assert zero_size == [], zero_size[:5]
    assert [r.getMessage() for r in caplog.records
            if r.levelno >= logging.ERROR] == []


def test_console_text_obeys_ten_percent_and_recovers(qtbot, qt_theme_applied):
    from spacr.qt.widgets.console_panel import ConsolePanel, _StdoutBlock

    original_scale = prefs.get_font_scale()
    prefs.set_font_scale(1.0)
    console = ConsolePanel()
    qtbot.addWidget(console)
    console.append_stdout("Existing output must shrink and recover.")
    console.show()
    block = console.findChild(_StdoutBlock)
    assert block is not None
    QApplication.processEvents()
    full_pt = block.font().pointSize()
    full_px = _rendered_px(block)
    try:
        prefs.set_font_scale(REQUESTED_FLOOR)
        console.apply_zoom()
        QApplication.processEvents()
        assert block.font().pointSize() == max(1, round(full_pt * REQUESTED_FLOOR))
        assert 0 < _rendered_px(block) < full_px
        prefs.set_font_scale(1.0)
        console.apply_zoom()
        QApplication.processEvents()
        assert block.font().pointSize() == full_pt
        assert _rendered_px(block) == full_px
    finally:
        prefs.set_font_scale(original_scale)
        console.close()


@pytest.mark.parametrize("start", [REQUESTED_FLOOR, 2.0])
def test_setting_chips_and_the_group_button_follow_each_zoom(
        qtbot, qt_theme_applied, start):
    from spacr.qt.screens.settings_model import _ListEditor
    from spacr.qt.theme import font_px

    original_scale = prefs.get_font_scale()
    prefs.set_font_scale(start)
    prefs.apply_preferences_to_app(qt_theme_applied)
    editor = _ListEditor(default=[["one", "two"]], nested_capable=True)
    qtbot.addWidget(editor)
    editor.show()
    try:
        for scale in (start, 1.0, REQUESTED_FLOOR, 2.0, 1.0):
            prefs.set_font_scale(scale)
            prefs.apply_preferences_to_app(qt_theme_applied)
            _pump(qt_theme_applied)
            chips = editor.findChildren(QLabel, "SettingChipText")
            assert len(chips) == 2
            assert editor._footer.isVisible()
            for widget in [*chips, editor._footer]:
                assert _rendered_px(widget) == font_px(12, scale), (
                    widget.objectName(), scale, _rendered_px(widget))
    finally:
        prefs.set_font_scale(original_scale)
        prefs.apply_preferences_to_app(qt_theme_applied)
        editor.close()
