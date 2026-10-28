"""Shared popup software paint continuity; not compositor flash evidence."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QSize, Qt
from PySide6.QtGui import QMouseEvent, QRegion
from PySide6.QtWidgets import QApplication

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


class _PaintObserver(QObject):
    def __init__(self, dialog, card):
        super().__init__(dialog)
        self.dialog = dialog
        self.card = card
        self.armed = False
        self.paints = []
        self.lifecycle = []

    def eventFilter(self, watched, event):
        if self.armed:
            if event.type() == QEvent.Paint:
                self.paints.append((watched, watched.size(), self.card.geometry(),
                                    self.dialog.rect(), int(self.dialog.effectiveWinId()),
                                    QRegion(event.region())))
            elif watched is self.dialog and event.type() in (
                    QEvent.Hide, QEvent.Show, QEvent.WinIdChange):
                self.lifecycle.append(event.type())
        return False


def _painted_region(records, widget, size):
    region = QRegion()
    for row in records:
        if row[0] is widget and row[1] == size:
            region |= row[5]
    return region


@pytest.mark.parametrize('theme_name', ['light', 'dark'])
@pytest.mark.parametrize('opaque', [False, True])
def test_shared_popup_resizes_keep_software_coverage_and_native_identity(
        qapp, qtbot, private_ai_profile, theme_name, opaque):
    from spacr.qt import preferences, theme
    from spacr.qt.widgets import glass
    from spacr.qt.widgets.model_share_dialog import ShareDialog
    from spacr.qt.widgets.setup_card import SetupCard

    preferences.set_theme(theme_name)
    preferences.set_pane_opacity(0.6)
    preferences.set_popup_backdrop('off')
    old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
    theme.apply_qpalette(qapp, theme_name)
    qapp.setStyleSheet(theme.stylesheet(theme_name, preferences.get_font_scale()))
    installed = glass.install_glass_everywhere(qapp)
    dialog = ShareDialog('resize-continuity')
    qtbot.addWidget(dialog)
    dialog.setProperty(glass.SETTINGS_POPUP, opaque)
    dialog.resize(1000, 800)
    dialog.show()
    observer = None
    try:
        qtbot.waitExposed(dialog)
        cards = dialog.findChildren(SetupCard)
        assert len(cards) == 1
        card = cards[0]
        assert card.geometry() == dialog.rect()
        identity = int(dialog.effectiveWinId())
        assert identity and dialog.windowHandle().format().alphaBufferSize() >= 8
        edit = next(iter(dialog._edits.values()))
        edit.setText('keep this value during resize')
        dialog.activateWindow()
        edit.setFocus()
        qtbot.waitUntil(edit.hasFocus)
        observer = _PaintObserver(dialog, card)
        dialog.installEventFilter(observer)
        card.installEventFilter(observer)
        initial = dialog.size()
        observer.armed = True
        reference = None
        for width, height in ((initial.width() + step, initial.height() + step)
                              for step in (24, 48, 72, 48, 24, 0, 24, 48, 24, 0)):
            start = len(observer.paints)
            previous_rect = QRect(QPoint(), dialog.size())
            dialog.resize(width, height)
            newly_exposed = QRegion(dialog.rect()).subtracted(QRegion(previous_rect))
            qtbot.waitUntil(lambda: any(row[0] is card and row[1] == dialog.size()
                                      for row in observer.paints[start:])
                           and newly_exposed.subtracted(_painted_region(
                               observer.paints[start:], card, dialog.size())).isEmpty())
            current = observer.paints[start:]
            assert current and all(row[2] == row[3] and row[4] == identity for row in current)
            assert all(not row[5].isEmpty() for row in current)
            assert newly_exposed.subtracted(_painted_region(current, card, dialog.size())).isEmpty()
            assert not observer.lifecycle
            assert dialog.isVisible() and card.isVisible()
            assert edit.hasFocus() and edit.text() == 'keep this value during resize'
            observer.armed = False
            raster = dialog.grab().toImage()
            ratio = raster.devicePixelRatio()
            corner = raster.pixelColor(0, 0)
            interior = raster.pixelColor(round(8 * ratio), raster.height() // 2)
            assert corner.alpha() == 0
            assert interior.alpha() == 255 if opaque else 0 < interior.alpha() < 255
            if reference is None:
                reference = interior
            assert all(abs(left - right) <= 1 for left, right in zip(interior.getRgb(), reference.getRgb()))
            observer.armed = True
        assert len(observer.paints) >= 10
        assert dialog.size() == initial
    finally:
        if observer is not None:
            observer.armed = False
            dialog.removeEventFilter(observer)
            card.removeEventFilter(observer)
        dialog.close()
        if installed:
            glass.uninstall_glass_everywhere()
        qapp.setPalette(old_palette)
        qapp.setStyleSheet(old_sheet)


@pytest.mark.parametrize('theme_name', ['light', 'dark'])
@pytest.mark.parametrize('opaque,backdrop_name', [
    (False, 'off'), (False, 'blobs'), (True, 'off'),
], ids=['translucent', 'animated-translucent', 'opaque'])
def test_windows_edge_drag_paints_before_returning_to_the_mouse_queue(
        qapp, qtbot, private_ai_profile, monkeypatch,
        theme_name, opaque, backdrop_name):
    """An owned resize must submit its new surface before the next move."""
    from spacr.qt import preferences, theme
    from spacr.qt.widgets import glass
    from spacr.qt.widgets.model_share_dialog import ShareDialog
    from spacr.qt.widgets.setup_card import SetupCard

    preferences.set_theme(theme_name)
    preferences.set_pane_opacity(0.6)
    preferences._set_flow_graphics_backend('cpu')
    preferences.set_popup_backdrop(backdrop_name)
    old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
    theme.apply_qpalette(qapp, theme_name)
    qapp.setStyleSheet(theme.stylesheet(theme_name, preferences.get_font_scale()))
    installed = glass.install_glass_everywhere(qapp)
    dialog = ShareDialog('edge-resize-continuity')
    qtbot.addWidget(dialog)
    dialog.setProperty(glass.SETTINGS_POPUP, opaque)
    dialog.resize(1000, 800)
    dialog.show()
    observer = None
    original_platform = QApplication.platformName
    try:
        qtbot.waitExposed(dialog)
        card, = dialog.findChildren(SetupCard)
        backdrop = getattr(dialog, '_spacr_popup_backdrop', None)
        if backdrop_name != 'off':
            assert backdrop is not None and backdrop.isVisible()
            assert backdrop.is_running()
        identity = int(dialog.effectiveWinId())
        edit = next(iter(dialog._edits.values()))
        edit.setText('keep this value during edge drag')
        dialog.activateWindow()
        edit.setFocus()
        qtbot.waitUntil(edit.hasFocus)
        initial = dialog.size()
        dialog.setMinimumSize(initial)
        dialog.setMaximumSize(initial + QSize(72, 72))
        observer = _PaintObserver(dialog, card)
        dialog.installEventFilter(observer)
        card.installEventFilter(observer)
        if QApplication.platformName() != 'windows':
            monkeypatch.setattr(QApplication, 'platformName', staticmethod(lambda: 'windows'))
        origin = dialog.mapToGlobal(dialog.rect().bottomRight() - QPoint(1, 1))

        def send_mouse(kind, point, button, buttons):
            QApplication.sendEvent(dialog, QMouseEvent(
                kind, QPointF(dialog.mapFromGlobal(point)), QPointF(point),
                button, buttons, Qt.NoModifier))

        send_mouse(QEvent.MouseButtonPress, origin, Qt.LeftButton, Qt.LeftButton)
        assert dialog._spacr_resizer._grab is not None
        qapp.processEvents()
        observer.armed = True
        for step in (24, 24, 48, 72, 48, 24, 0, -24, -48, 96, 120):
            previous = QRect(dialog.rect())
            start = len(observer.paints)
            send_mouse(QEvent.MouseMove, origin + QPoint(step, step),
                       Qt.NoButton, Qt.LeftButton)
            bounded = min(72, max(0, step))
            assert dialog.size() == initial + QSize(bounded, bounded)
            current = observer.paints[start:]
            card_paints = [row for row in current if row[0] is card]
            if dialog.rect() == previous:
                assert not card_paints, 'stationary or clamped moves must not repaint synchronously'
                continue
            assert card_paints, 'the resize returned before painting the card'
            assert all(row[1] == dialog.size() and row[2] == row[3]
                       and row[4] == identity for row in card_paints)
            exposed = QRegion(dialog.rect()).subtracted(QRegion(previous))
            assert exposed.subtracted(_painted_region(
                current, card, dialog.size())).isEmpty()
            assert edit.hasFocus()
            assert edit.text() == 'keep this value during edge drag'
            if backdrop is not None:
                assert backdrop.geometry() == dialog.rect() and backdrop.is_running()
            assert not observer.lifecycle
        send_mouse(QEvent.MouseButtonRelease, origin + QPoint(120, 120),
                   Qt.LeftButton, Qt.NoButton)
        assert dialog._spacr_resizer._grab is None
        start = len(observer.paints)
        dialog.resize(initial)
        assert not [row for row in observer.paints[start:] if row[0] is card], (
            'ordinary programmatic resizing must retain deferred painting')
        qtbot.waitUntil(lambda: any(row[0] is card and row[1] == dialog.size()
                                  for row in observer.paints[start:]))
    finally:
        monkeypatch.setattr(QApplication, 'platformName', staticmethod(original_platform))
        if observer is not None:
            observer.armed = False
            dialog.removeEventFilter(observer)
            card.removeEventFilter(observer)
        dialog.close()
        if installed:
            glass.uninstall_glass_everywhere()
        qapp.setPalette(old_palette)
        qapp.setStyleSheet(old_sheet)
