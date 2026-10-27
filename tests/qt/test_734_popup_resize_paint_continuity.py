"""Shared popup software paint continuity; not compositor flash evidence."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint, QRect
from PySide6.QtGui import QRegion

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
