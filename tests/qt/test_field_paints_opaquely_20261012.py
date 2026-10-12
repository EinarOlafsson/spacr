"""The window no longer fills its background under the field on every frame.

A field frame repaints the backdrop, and the main window filled its whole
background under it first: 15-17 % of a frame's GUI-thread cost on Home and
Mask alike. The field always claimed ``WA_OpaquePaintEvent`` (it paints the
page colour or the wallpaper under the animation), but the application
stylesheet's polish clears that attribute from any widget a rule reaches,
so Qt treated the field as transparent. Since 2026-10-12 the field claims
it again after every polish and restyle, and only while it is true: square
corners and a backdrop picture without translucent pixels.

Checked here: the claim and when it is withheld; on a small scene and on the
real window, Home and Mask in dark, light and high contrast at pane opacity
0, 50 and 100 %, window grabs and the flushed backing store are identical
to the classic paint, also on the first frame after a show, a resize and a
screen switch, with the backing store poisoned beforehand so that any pixel
the field left unpainted shows. Home's tiles hand their hover to the page
through the application event hub, not a filter on each tile.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPalette, QPixmap  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt import preferences  # noqa: E402

pytestmark = pytest.mark.qt

MAGENTA = QColor(255, 0, 255)


def _pixmap(alpha: int, size=(64, 48)) -> QPixmap:
    image = QImage(*size, QImage.Format_ARGB32)
    image.fill(QColor(30, 60, 90, 255))
    if alpha < 255:
        image.setPixelColor(3, 5, QColor(30, 60, 90, alpha))
    return QPixmap.fromImage(image, Qt.NoOpaqueDetection)


def test_covers_opaquely_reads_the_alpha_of_the_picture(qapp):
    from spacr.qt.widgets.ambient import _covers_opaquely

    assert _covers_opaquely(None)
    plain = QImage(32, 32, QImage.Format_RGB32)
    plain.fill(QColor(1, 2, 3))
    assert not QPixmap.fromImage(plain).hasAlphaChannel()
    assert _covers_opaquely(QPixmap.fromImage(plain))
    full = _pixmap(255)
    assert full.hasAlphaChannel()
    assert _covers_opaquely(full)
    assert not _covers_opaquely(_pixmap(254))
    assert not _covers_opaquely(_pixmap(0))


def _styled_host(qtbot):
    """A host whose stylesheet reaches the field, as the application's does."""
    host = QWidget()
    host.setStyleSheet("QWidget { background: transparent; }")
    qtbot.addWidget(host)
    host.resize(160, 120)
    return host


def _field(host, **kwargs):
    from spacr.qt.widgets.ambient import AmbientWidget

    field = AmbientWidget(host, theme="blobs", seed=3, fps=1, **kwargs)
    field.set_animating(False)
    field.follow_parent()
    return field


def test_the_claim_survives_polish_and_restyle(qtbot):
    host = _styled_host(qtbot)
    field = _field(host)
    field.ensurePolished()
    assert not field.testAttribute(Qt.WA_OpaquePaintEvent), (
        "the stylesheet no longer clears the attribute; this test is stale")
    host.show()
    qtbot.waitExposed(host)
    assert field.testAttribute(Qt.WA_OpaquePaintEvent)
    host.setStyleSheet("QWidget { background: rgba(0, 0, 0, 0); }")
    assert field.testAttribute(Qt.WA_OpaquePaintEvent)
    for kind in (QEvent.FontChange, QEvent.ApplicationPaletteChange):
        field.changeEvent(QEvent(kind))
        assert field.testAttribute(Qt.WA_OpaquePaintEvent)


def test_the_claim_is_withheld_where_the_field_lets_through(qtbot):
    host = _styled_host(qtbot)
    rounded = _field(host, corner_radius=8)
    host.show()
    qtbot.waitExposed(host)
    assert not rounded.testAttribute(Qt.WA_OpaquePaintEvent)
    rounded.setStyleSheet("")
    assert not rounded.testAttribute(Qt.WA_OpaquePaintEvent)

    square = _field(host, backdrop=_pixmap(128))
    square.show()
    assert not square.testAttribute(Qt.WA_OpaquePaintEvent)
    square.set_backdrop(_pixmap(255))
    assert square.testAttribute(Qt.WA_OpaquePaintEvent)
    square.set_backdrop(_pixmap(0))
    assert not square.testAttribute(Qt.WA_OpaquePaintEvent)
    square.set_backdrop(None)
    assert square.testAttribute(Qt.WA_OpaquePaintEvent)


def _flushed(widget) -> QImage:
    """What the window's backing store holds, after pending paints."""
    for _turn in range(3):
        QApplication.processEvents()
    window = widget.window()
    image = window.screen().grabWindow(window.winId()).toImage()
    return image.convertToFormat(QImage.Format_RGB32)


def _grabbed(widget) -> QImage:
    return widget.window().grab().toImage().convertToFormat(
        QImage.Format_RGB32)


def _poison(window):
    """Cover the window in magenta, flush it, and take the cover away.

    The paint that follows starts from a backing store full of magenta, so a
    pixel the field does not paint stays magenta.
    """
    cover = getattr(window, "_test_cover", None)
    if cover is None:
        cover = window._test_cover = QWidget(window)
        cover.setAutoFillBackground(True)
        palette = cover.palette()
        palette.setColor(QPalette.Window, MAGENTA)
        cover.setPalette(palette)
    cover.setGeometry(window.rect())
    cover.raise_()
    cover.show()
    cover.repaint()
    QApplication.processEvents()
    return cover


def _magenta_in(image: QImage, field) -> int:
    offset = field.mapTo(field.window(), QPoint(0, 0))
    found = 0
    for y in range(offset.y(), offset.y() + field.height(), 7):
        for x in range(offset.x(), offset.x() + field.width(), 7):
            if image.pixelColor(x, y) == MAGENTA:
                found += 1
    return found


def _classic(field, on: bool) -> None:
    field.setAttribute(Qt.WA_OpaquePaintEvent, not on)


def _freeze(window, field) -> None:
    """Stop the field, so two paints of the window can be compared."""
    for child in window.findChildren(QWidget):
        if type(child).__name__ == "_TourOverlay":
            child.hide()
        usage = getattr(child, "_usage_timer", None)
        if usage is not None:
            usage.stop()
    field._timer.stop()
    producer = field._producer_box[0]
    if producer is not None:
        producer.stop()


def _compare(window, field, label, qtbot=None):
    """The first frame as it is now, then the window against itself painted
    classically.

    The first frame must leave no magenta (see :func:`_poison`) anywhere.
    Then, with the field stopped, an opaque paint must equal a classic one;
    on the real window labels and panels may still be settling after a
    resize or a screen switch, so a pair is taken only when two opaque
    paints around the classic one agree.
    """
    assert field.testAttribute(Qt.WA_OpaquePaintEvent), label
    first = _flushed(window)
    assert _magenta_in(first, field) == 0, label
    _freeze(window, field)
    for _attempt in range(8):
        window.repaint()
        flushed, grabbed = _flushed(window), _grabbed(window)
        _classic(field, True)
        window.repaint()
        classic_flushed, classic_grabbed = _flushed(window), _grabbed(window)
        _classic(field, False)
        window.repaint()
        if _flushed(window) == flushed and _grabbed(window) == grabbed:
            break
        assert qtbot is not None, label
        qtbot.wait(700)
    assert flushed == classic_flushed, label
    assert grabbed == classic_grabbed, label


def test_a_small_scene_is_identical_on_every_first_frame(qtbot):
    host = QWidget()
    host.setAutoFillBackground(True)
    palette = host.palette()
    palette.setColor(QPalette.Window, QColor(10, 200, 10))
    host.setPalette(palette)
    host.setStyleSheet("QWidget#cap { background: transparent; }")
    qtbot.addWidget(host)
    host.resize(200, 140)
    field = _field(host)
    caption = QWidget(host)
    caption.setObjectName("cap")
    caption.setGeometry(20, 20, 60, 30)
    host.show()
    qtbot.waitExposed(host)
    field.advance_frame(0.4)
    window = host.window()
    _compare(window, field, "steady")

    cover = _poison(window)
    cover.hide()
    _compare(window, field, "after a cover goes")

    field.hide()
    QApplication.processEvents()
    cover = _poison(window)
    cover.hide()
    field.show()
    _compare(window, field, "first frame after show")

    for size in ((260, 170), (150, 110)):
        cover = _poison(window)
        cover.hide()
        host.resize(*size)
        _compare(window, field, f"first frame after a resize to {size}")


@pytest.fixture
def themed(qapp, tmp_path, monkeypatch):
    """A private preference store; the application's look put back after."""
    from spacr.qt.widgets.field_fade import (invalidate_field_fade,
                                             uninstall_field_fade)

    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()

    def apply(theme, opacity):
        preferences.set_theme_choice(theme)
        preferences.set_pane_opacity(opacity)
        preferences.apply_preferences_to_app(qapp)

    yield apply
    uninstall_field_fade(qapp)
    invalidate_field_fade()
    qapp.setFont(font)
    qapp.setPalette(palette)
    qapp.setStyleSheet(sheet)


def _window(qtbot):
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: window.window_backdrop() is not None,
                    timeout=20000)
    field = window.window_backdrop()
    if not hasattr(field, "_card_tints"):
        pytest.skip("the window backdrop is not the field in this setup")
    return window, field


def _settle(window, field, key, qtbot):
    """Show ``key`` with the cards composited; :func:`_compare` freezes it."""
    field.stop()
    field.start()
    window._on_nav_selected(key)
    for child in window.findChildren(QWidget):
        if type(child).__name__ == "_TourOverlay":
            child.hide()
    tints = field._card_tints
    for _attempt in range(60):
        qtbot.waitUntil(lambda: tints.active, timeout=20000)
        qtbot.wait(300)
        if tints.active:
            break


@pytest.mark.parametrize("theme", ["dark", "light", "high_contrast"])
def test_home_and_mask_are_identical_to_the_classic_paint(themed, qtbot,
                                                          theme):
    themed(theme, 0.0)
    window, field = _window(qtbot)
    for opacity in (0.0, 0.5, 1.0):
        themed(theme, opacity)
        for key, other in (("__home__", "mask"), ("mask", "__home__")):
            label = (theme, opacity, key)
            _settle(window, field, key, qtbot)
            _compare(window, field, label + ("steady",), qtbot)
            cover = _poison(window)
            cover.hide()
            _compare(window, field, label + ("after a cover goes",), qtbot)
            if opacity == 0.5:
                for size in ((1300, 820), (1400, 900)):
                    cover = _poison(window)
                    cover.hide()
                    window.resize(*size)
                    _compare(window, field, label + ("resize", size), qtbot)
                cover = _poison(window)
                cover.hide()
                window._on_nav_selected(other)
                _compare(window, field, label + ("switched to", other),
                         qtbot)
                window._on_nav_selected(key)


@pytest.fixture
def _empty_journal(tmp_path, monkeypatch):
    from spacr import run_journal

    monkeypatch.setattr(run_journal, "runs_root", lambda: tmp_path)
    yield tmp_path


def _home(qtbot):
    from spacr.qt.widgets.home import HomePage

    apps = [("mask", "Mask", "Segment a plate", "Core"),
            ("measure", "Measure", "Measure objects", "Core")]
    page = HomePage(apps, lambda key: None)
    qtbot.addWidget(page)
    page.resize(1200, 800)
    page.show()
    qtbot.waitExposed(page)
    return page


def test_home_tiles_hand_their_hover_through_the_hub(qtbot, _empty_journal):
    import sys

    from spacr.qt.widgets.home import HomePage, _TileHintWatcher

    page = _home(qtbot)
    tiles = list(page._tile_hints)
    assert tiles
    watchers = page.findChildren(_TileHintWatcher)
    assert len(watchers) == 1
    shown = []
    page._hint_bar.show_module = lambda key, *_rest: shown.append(key)
    tile = tiles[0]
    QApplication.sendEvent(tile, QEvent(QEvent.Enter))
    assert shown == [page._tile_hints[tile][0]]
    released = []
    page._hint_bar.release = lambda: released.append(1)
    page._hint_bar.is_holding = lambda: False
    QApplication.sendEvent(tile, QEvent(QEvent.Leave))
    assert released == [1]
    QApplication.sendEvent(page, QEvent(QEvent.Enter))
    QApplication.sendEvent(page, QEvent(QEvent.Leave))
    assert shown == [page._tile_hints[tile][0]] and released == [1]

    asked = []
    hints = HomePage.eventFilter.__code__

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is hints:
            asked.append(1)

    sys.setprofile(profile)
    try:
        for each in tiles:
            each.repaint()
    finally:
        sys.setprofile(None)
    assert asked == [], f"{len(asked)} Home filter calls for one tile paint each"
