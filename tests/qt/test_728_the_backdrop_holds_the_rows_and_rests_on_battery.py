"""Item 728: an idle backdrop animates without repainting the rows over it,
and holds still on battery.

The native Mac run of 2026-10-25 found an idle spaCR at 90-110 % of a CPU
core, because Qt repainted every translucent settings row on every backdrop
frame (154-196 widget paints a frame). ``_HeldRows`` makes the backdrop
update only where nothing is drawn over it once the field is calm, and paint
the frame that was on screen when the hold began under everything else.

Pinned here:

* paint count: once held, the rows over the field are not repainted per
  frame (and, as a control, they are when the hold is switched off, so the
  counter can see them);
* visual equivalence: the card and its rows on screen while held are the
  pixels a per-frame paint of the held frame gives, also after a row
  repaints for its own reasons;
* a stirred field or a changed engine paints the whole field again;
* battery: the backdrop stops on battery and resumes on AC on every
  platform, with a mocked sensor, and the spaCR Reduce motion preference is
  left alone.
"""
from __future__ import annotations

import collections

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QObject, QRect               # noqa: E402
from PySide6.QtGui import QImage, QRegion                        # noqa: E402
from PySide6.QtWidgets import (QFrame, QLabel, QPushButton,      # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt.widgets import ambient                             # noqa: E402

pytestmark = pytest.mark.qt

THEME = "data_art_impulse_lens"
ROWS = 10


@pytest.fixture(autouse=True)
def _no_hub_left_behind(qapp):
    """Take down an application event hub these backdrops installed."""
    from spacr.qt import gil_priority

    before = gil_priority._application_event_hub(qapp, create=False)
    yield
    after = gil_priority._application_event_hub(qapp, create=False)
    if after is not None and after is not before:
        qapp.removeEventFilter(after)
        setattr(qapp, gil_priority._HUB_ATTRIBUTE, before)
        after.deleteLater()


class _Paints(QObject):
    """Counts paint events per widget class, everything but the backdrop."""

    def __init__(self):
        super().__init__()
        self.counts = collections.Counter()

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Paint and not isinstance(
                watched, ambient.AmbientWidget):
            self.counts[type(watched).__name__] += 1
        return False


@pytest.fixture
def screen(qtbot, monkeypatch):
    """A field backdrop under one translucent card of rows, shading on its
    own thread, with the hold quick to arm."""
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient, "_HOLD_SETTLE_S", 0.0)
    monkeypatch.setattr(ambient, "_HOLD_RECHECK_S", 0.2)
    host = QWidget()
    host.resize(640, 420)
    backdrop = ambient.AmbientWidget(host, theme=THEME, fps=60, seed=7)
    backdrop.follow_parent()
    card = QFrame(host)
    card.setObjectName("heldCard")
    card.setAttribute(ambient.Qt.WA_StyledBackground, True)
    card.setStyleSheet("#heldCard { background: rgba(28, 32, 44, 150);"
                       " border-radius: 10px; }"
                       "QLabel, QPushButton { background: transparent;"
                       " color: #dde; }")
    card.setGeometry(40, 40, 360, 340)
    rows = QVBoxLayout(card)
    labels = []
    for index in range(ROWS):
        label = QLabel(f"Setting {index}", card)
        rows.addWidget(label)
        rows.addWidget(QPushButton(f"Value {index}", card))
        labels.append(label)
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(lambda: backdrop._producer_box[0] is not None,
                    timeout=10000)
    qtbot.waitUntil(lambda: backdrop._last_frame is not None, timeout=10000)
    yield host, backdrop, card, labels
    backdrop.stop()


def _paints_per_frame(qtbot, qapp, backdrop, ms=1500):
    """Non-backdrop paints per backdrop frame over ``ms`` of running."""
    counter = _Paints()
    qapp.installEventFilter(counter)
    before = backdrop.frames_painted
    try:
        qtbot.wait(ms)
    finally:
        qapp.removeEventFilter(counter)
    frames = backdrop.frames_painted - before
    assert frames >= 5, f"the backdrop kept animating ({frames} frames)"
    return sum(counter.counts.values()) / frames, counter.counts


def test_a_held_field_does_not_repaint_the_rows_over_it(screen, qtbot, qapp):
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    assert held.region.contains(card.geometry().center())
    assert not held.live.isEmpty(), "the field still moves around the card"

    per_frame, counts = _paints_per_frame(qtbot, qapp, backdrop)
    assert per_frame <= 1.0, (
        f"{per_frame:.1f} widget paints per backdrop frame once idle: {counts}")
    assert backdrop._rate() <= ambient._HOLD_FPS, "an idle field is paced"
    held.stir()
    backdrop._on_tick()
    assert backdrop._rate() == backdrop._fps, "a stirred field runs at full rate"


def test_without_the_hold_every_row_repaints_each_frame(screen, qtbot, qapp,
                                                        monkeypatch):
    """The control: the counter does see the per-frame repaint the hold
    removes."""
    host, backdrop, card, labels = screen
    monkeypatch.setattr(ambient._HeldRows, "enabled", False)
    qtbot.wait(300)
    assert backdrop._held_rows.frame is None
    per_frame, counts = _paints_per_frame(qtbot, qapp, backdrop)
    assert per_frame >= 2 * ROWS, counts


def _on_screen(host) -> QImage:
    """What the window's backing store shows now, not a fresh render."""
    return host.screen().grabWindow(host.winId()).toImage().convertToFormat(
        QImage.Format_RGB32)


def _largest_difference(first: QImage, second: QImage, rect: QRect) -> int:
    """The largest channel difference between two images inside ``rect``."""
    import numpy as np

    def pixels(image):
        view = np.frombuffer(image.constBits(), np.uint8).reshape(
            image.height(), image.bytesPerLine() // 4, 4)
        return view[rect.top():rect.bottom() + 1,
                    rect.left():rect.right() + 1, :3].astype(int)

    return int(np.abs(pixels(first) - pixels(second)).max())


def test_held_rows_look_as_a_per_frame_paint_of_the_held_frame(
        screen, qtbot, monkeypatch):
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    start = backdrop.frames_painted
    qtbot.waitUntil(lambda: backdrop.frames_painted >= start + 6,
                    timeout=5000)
    held_frame = held.frame
    assert held_frame is not None and held_frame is not backdrop._last_frame
    labels[3].repaint()
    labels[6].update()
    qtbot.wait(50)
    shown = _on_screen(host)

    backdrop._timer.stop()
    producer = backdrop._producer_box[0]
    monkeypatch.setattr(producer, "latest", lambda: held_frame)
    held.release(fade=False)
    monkeypatch.setattr(ambient._HeldRows, "enabled", False)
    backdrop._last_frame = held_frame
    host.repaint()
    reference = _on_screen(host)

    assert _largest_difference(shown, reference, card.geometry()) == 0, (
        "the card and its rows look as they did under the per-frame paint")


def test_the_moving_field_around_the_rows_is_the_per_frame_paint(
        screen, qtbot, monkeypatch):
    """The live part, drawn a rectangle at a time, is pixel-exact."""
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    start = backdrop.frames_painted
    qtbot.waitUntil(lambda: backdrop.frames_painted >= start + 3,
                    timeout=5000)
    producer = backdrop._producer_box[0]
    newest = backdrop._last_frame
    monkeypatch.setattr(producer, "latest", lambda: newest)
    live = QRegion(held.live)
    assert live.rectCount() > 1
    backdrop.repaint(live)
    shown = _on_screen(host)

    monkeypatch.setattr(ambient._HeldRows, "enabled", False)
    held.release(fade=False)
    backdrop.repaint()
    reference = _on_screen(host)
    for part in live:
        assert _largest_difference(shown, reference, part) == 0, part


def test_pausing_a_held_field_paints_nothing_more(screen, qtbot, monkeypatch):
    """A run pausing the backdrop must not cost a frame, held or not
    (test_a_run_still_pauses_the_backdrop), and the rows keep the pixels
    under them."""
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    start = backdrop.frames_painted
    qtbot.waitUntil(lambda: backdrop.frames_painted >= start + 3,
                    timeout=5000)
    held_frame = held.frame
    backdrop.set_animating(False)
    paused = backdrop.frames_painted
    qtbot.wait(400)
    assert backdrop.frames_painted == paused, "a paused field paints nothing"
    assert held.frame is None and held.fade is None
    assert backdrop._still is not None and backdrop._still[1] is held_frame, (
        "the still is the frame under the rows")
    shown = _on_screen(host)
    labels[4].repaint()
    after = _on_screen(host)
    assert _largest_difference(shown, after, card.geometry()) == 0


def test_a_stirred_or_changed_field_is_painted_whole_again(screen, qtbot):
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    held.stir()
    assert held.frame is None and held.fade is not None, "a release fades"
    qtbot.waitUntil(lambda: held.fade is None, timeout=3000)

    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    backdrop.set_time(backdrop.time() + 5.0)
    assert held.frame is None and held.fade is None, (
        "an engine change shows at once, everywhere")


def test_a_run_pacing_the_field_skips_the_held_walk(screen, qtbot,
                                                    monkeypatch):
    """The once-a-second walk over the window is Python over every widget;
    while a run paces the field it is not repeated, and it resumes after."""
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    producer = backdrop._producer_box[0]
    walks = []
    covered = held._covered
    monkeypatch.setattr(held, "_covered",
                        lambda: walks.append(1) or covered())
    monkeypatch.setattr(backdrop, "_run_paced", True)
    held._walked = float("-inf")
    assert held.tick(producer) is not None
    assert walks == [], "the walk ran while a run paced the field"
    monkeypatch.setattr(backdrop, "_run_paced", False)
    held.tick(producer)
    assert walks == [1], "the walk comes back once the run ends"


def _sensor(plugged):
    reading = collections.namedtuple("sbattery",
                                     "percent secsleft power_plugged")
    return lambda: reading(55.0, 3600, plugged)


def _supplies(root, plugged, mains=True):
    """A Linux power-supply folder: a laptop battery, its charger, and a
    Bluetooth mouse's battery that is always discharging."""
    def supply(name, **attributes):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for key, value in attributes.items():
            (folder / key).write_text(value + "\n")

    supply("hid-90:9c:4a:08:56:fb-battery", type="Battery", scope="Device",
           status="Discharging")
    supply("BAT0", type="Battery", status="Discharging" if not plugged
           else "Charging")
    if mains:
        supply("AC", type="Mains", online="1" if plugged else "0")
    return root


def _plug(monkeypatch, tmp_path, plugged):
    """Answer ``plugged`` from every platform's battery source."""
    import psutil

    folder = tmp_path / ("ac" if plugged else "battery")
    monkeypatch.setattr(ambient, "_POWER_SUPPLY",
                        _supplies(folder, plugged))
    monkeypatch.setattr(psutil, "sensors_battery", _sensor(plugged))
    monkeypatch.setattr(ambient, "_BATTERY", [float("-inf"), False])


def test_linux_reads_the_system_battery_not_a_bluetooth_mouse(tmp_path):
    desktop = tmp_path / "desktop"
    desktop.mkdir()
    (desktop / "hid-mouse").mkdir()
    (desktop / "hid-mouse" / "type").write_text("Battery\n")
    (desktop / "hid-mouse" / "scope").write_text("Device\n")
    (desktop / "hid-mouse" / "status").write_text("Discharging\n")
    assert ambient._linux_on_battery(desktop) is False, (
        "a desktop with a Bluetooth mouse is not on battery")
    assert ambient._linux_on_battery(_supplies(tmp_path / "a", False)) is True
    assert ambient._linux_on_battery(_supplies(tmp_path / "b", True)) is False
    assert ambient._linux_on_battery(
        _supplies(tmp_path / "c", False, mains=False)) is True, (
            "no charger reported: the battery's own status decides")
    assert ambient._linux_on_battery(tmp_path / "missing") is False


def test_macos_and_windows_read_psutil(monkeypatch):
    import psutil

    monkeypatch.delenv("SPACR_BATTERY_PAUSE", raising=False)
    for platform in ("darwin", "win32"):
        monkeypatch.setattr(ambient.sys, "platform", platform)
        for plugged, expected in ((False, True), (True, False),
                                  (None, False)):
            monkeypatch.setattr(ambient, "_BATTERY", [float("-inf"), False])
            monkeypatch.setattr(psutil, "sensors_battery", _sensor(plugged))
            assert ambient._on_battery() is expected, (platform, plugged)

        monkeypatch.setattr(ambient, "_BATTERY", [float("-inf"), False])
        monkeypatch.setattr(psutil, "sensors_battery", lambda: None)
        assert ambient._on_battery() is False, "a desktop has no battery"

        monkeypatch.setattr(ambient, "_BATTERY", [float("-inf"), False])
        monkeypatch.setattr(psutil, "sensors_battery", _broken_sensor)
        assert ambient._on_battery() is False

    monkeypatch.setattr(ambient, "_BATTERY", [ambient.time.monotonic(), True])
    assert ambient._on_battery() is True, "a fresh answer is reused"
    monkeypatch.setenv("SPACR_BATTERY_PAUSE", "0")
    assert ambient._on_battery() is False, "the documented switch keeps it"


def _broken_sensor():
    raise OSError("no sensor")


def test_the_backdrop_rests_on_battery_and_resumes_on_ac(qtbot, monkeypatch,
                                                         tmp_path):
    from spacr.qt import preferences

    monkeypatch.delenv("SPACR_BATTERY_PAUSE", raising=False)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    monkeypatch.setattr(ambient, "_macos_asks_for_stillness", lambda: False)
    monkeypatch.setattr(ambient, "_BATTERY_TTL_S", 0.0)
    monkeypatch.setattr(ambient, "_MACOS_STILLNESS_RECHECK_MS", 50)
    _plug(monkeypatch, tmp_path, True)
    reduce_motion = preferences.get_reduce_motion()
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 240)
    backdrop = ambient.AmbientWidget(host, theme=THEME, seed=1)
    backdrop.follow_parent()
    host.show()
    qtbot.waitExposed(host)
    backdrop._sync_run_state()
    assert backdrop.is_running()

    _plug(monkeypatch, tmp_path, False)
    backdrop._on_tick()
    assert not backdrop.is_running() and not backdrop._should_run()
    assert preferences.get_reduce_motion() is reduce_motion, (
        "the spaCR Reduce motion preference is left alone")
    pending = backdrop._stillness_recheck
    assert pending is not None and pending.isActive(), "it asks again"

    _plug(monkeypatch, tmp_path, True)
    qtbot.waitUntil(backdrop.is_running, timeout=2000)
    backdrop.stop()


class _CanvasWithAMaskArray(QWidget):
    """A widget like Make Masks' canvas, whose ``mask`` attribute is image
    data and so hides ``QWidget.mask()``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        import numpy as np
        self.mask = np.zeros((4, 4), dtype=np.uint16)


def test_a_widget_whose_mask_attribute_is_data_does_not_break_the_hold(
        screen, qtbot):
    """GitHub #168: the hold's walk called ``widget.mask()`` and raised
    ``'numpy.ndarray' object is not callable`` on every tick when a widget
    kept its label image in ``self.mask``."""
    host, backdrop, card, labels = screen
    panel = QWidget(host)
    panel.setGeometry(420, 40, 180, 300)
    canvas = _CanvasWithAMaskArray(panel)
    canvas.setGeometry(10, 10, 80, 60)
    QLabel("beside it", panel).setGeometry(10, 90, 120, 20)
    panel.show()
    held = backdrop._held_rows
    held._own_paint(canvas)
    held._covered()
    held.stir()
    qtbot.wait(800)
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    assert held.region.contains(card.geometry().center())


_QT_NAMES_THE_WALK_READS = (
    "geometry", "children", "isWidgetType", "isWindow", "isHidden", "autoFillBackground",
    "testAttribute", "rect", "width", "height", "frameWidth", "frameShape", "render")


class _PanelHidingQtMethods(QFrame):
    """A framed panel whose attributes hide every Qt method the hold's walk
    reads, as Overview's ``geometry`` combo box once hid ``QWidget.geometry``
    (GitHub #162, #169)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Box)
        self.setLineWidth(2)
        self.setAttribute(ambient.Qt.WA_StyledBackground, True)
        for name in _QT_NAMES_THE_WALK_READS:
            setattr(self, name, object())


def test_attributes_hiding_qt_methods_never_break_the_hold_169(screen, qtbot):
    """Every Qt call the walk makes on a page's widget goes through the Qt
    class itself, so no instance attribute can stand in for it."""
    host, backdrop, card, labels = screen
    panel = _PanelHidingQtMethods(host)
    QFrame.setGeometry(panel, 420, 40, 180, 300)
    QLabel("inside it", panel).setGeometry(10, 10, 120, 20)
    leaf = _PanelHidingQtMethods(host)
    QFrame.setGeometry(leaf, 420, 350, 60, 40)
    QFrame.show(panel)
    QFrame.show(leaf)
    held = backdrop._held_rows
    held._own_paint(panel)
    held._own_paint(leaf)
    held._painted_tiles(leaf)
    held._paints_a_background(leaf)
    held._covered()
    held.stir()
    qtbot.wait(800)
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    assert held.region.contains(card.geometry().center())


def test_changed_and_cleared_card_growth_releases_the_held_frame(screen, qtbot, monkeypatch):
    host, backdrop, card, labels = screen
    held = backdrop._held_rows
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    assert not held.live.isEmpty()
    previous_stir = held._stirred
    rectangle = card.geometry()
    ambient._field_growing_regions_for_widget(host, (rectangle,))
    regions = backdrop._art_input._growing_regions
    assert regions and isinstance(regions, tuple)
    assert held.frame is None and held._stirred > previous_stir
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    qtbot.waitUntil(lambda: backdrop._rate() == 20, timeout=5000)
    frame, stirred = held.frame, held._stirred
    ambient._field_growing_regions_for_widget(host, (rectangle,))
    assert held.frame is frame and held._stirred == stirred
    assert backdrop._art_input._growing_regions == regions
    assert backdrop._rate() == 20
    ambient._field_growing_regions_for_widget(host)
    assert backdrop._art_input._growing_regions == ()
    assert held.frame is None and held._stirred > stirred
    qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
    qtbot.waitUntil(lambda: backdrop._rate() == 20, timeout=5000)
    frame, stirred = held.frame, held._stirred
    ambient._field_growing_regions_for_widget(host)
    assert held.frame is frame and held._stirred == stirred
    assert backdrop._rate() == 20

    from spacr.qt.widgets.home import _EasyHomeField

    assert card.isVisible()
    field = _EasyHomeField(host)
    field.resize(host.size())
    monkeypatch.setattr(field, '_motion_enabled', lambda: False)
    try:
        field.show()
        field.set_cards([('measure', 'Measure', 'Measure an existing dataset.', '')])
        completed = field.cards[0]
        assert completed.isVisible() and not completed._growing
        assert backdrop._art_input._growing_regions == ()
        qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
        qtbot.waitUntil(lambda: backdrop._rate() == 20, timeout=5000)
        stirred = held._stirred
        field.set_cards([])
        assert not completed.isVisible() and not field.cards
        assert backdrop._art_input._growing_regions == ()
        assert held.frame is None and held._stirred > stirred
        qtbot.waitUntil(lambda: held.frame is not None, timeout=5000)
        qtbot.waitUntil(lambda: backdrop._rate() == 20, timeout=5000)
        frame, stirred = held.frame, held._stirred
        field.set_cards([])
        assert held.frame is frame and held._stirred == stirred
        assert backdrop._rate() == 20
    finally:
        field.close()


class _DecorativeDimmer(QWidget):
    """Paint a real translucent layer over the field without a shading worker."""

    def paintEvent(self, event):
        from PySide6.QtGui import QColor, QPainter

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))
        painter.end()


def test_a_decorative_overlay_leaves_the_field_live_but_holds_its_child(qtbot):
    """Only the explicit overlay's own paint is omitted from the held region."""
    host = QWidget()
    host.resize(640, 416)
    qtbot.addWidget(host)
    backdrop = QWidget(host)
    backdrop.setGeometry(host.rect())
    overlay = _DecorativeDimmer(host)
    overlay.setGeometry(host.rect())
    card = QFrame(overlay)
    card.setGeometry(128, 96, 192, 128)
    card.setAutoFillBackground(True)
    host.show()
    qtbot.waitExposed(host)
    held = ambient._HeldRows(backdrop)
    whole_field = QRegion(backdrop.rect())
    assert held._covered() == whole_field
    overlay.setProperty("spacrAmbientTransparentOverlay", True)
    covered = held._covered()
    assert covered == QRegion(card.geometry())
    live = whole_field.subtracted(covered)
    assert not live.isEmpty()
    assert not live.contains(card.geometry().center())
    assert live.contains(ambient.QPoint(16, 16))
    overlay.setProperty("spacrAmbientTransparentOverlay", False)
    assert held._covered() == whole_field
