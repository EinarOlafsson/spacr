"""Use the real figure dialog to keep a tutorial's legend on the canvas."""


import capture_ui
from capture_ui_drv_a import modal_wait, watchdog_ms
from capture_ui_drv_c import Scenes


def fit_legend(app, window, queue, figure, capture, settle, name, *, scenes=None, next_name=None):
    """Put the legend of ``figure`` inside its canvas through the Figure settings dialog.

    Host: the clip ``name`` starts before the pointer clicks the settings button and
    ends in the dialog once the position is chosen; ``next_name`` (the scene that
    follows) starts there, before OK, and is ended by the caller.
    """
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    errors, accepted = [], []
    S = scenes if scenes is not None else Scenes(capture, settle)
    S.begin(name)

    def choose():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, FigureSettingsDialog):
                raise ValueError('The native Figure settings dialog did not open')
            watchdog = QTimer(dialog)
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(dialog.reject)
            watchdog.start(watchdog_ms(capture, 15000))
            capture_ui.dialog_geometry(capture, dialog, (1000, 1500),
                                       window.geometry().center() - dialog.rect().center())
            tabs = dialog.tabs
            title = figure.axes[0].get_title()[:18]
            matches = [i for i in range(tabs.count()) if tabs.tabText(i) == title]
            if len(matches) != 1:
                raise ValueError('The native figure axes tab is ambiguous')
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton, pos=tabs.tabBar().tabRect(matches[0]).center())
            settle(.2)
            page = tabs.currentWidget()
            boxes = [b for b in page.findChildren(QComboBox)
                     if b.findText('best') >= 0 and b.findText('upper right') >= 0]
            if len(boxes) != 1:
                raise ValueError('The native Legend position choice is ambiguous')
            choice = boxes[0]
            capture_ui.scroll_to(capture, page, choice)
            if capture_ui.live(capture):
                # The dialog starts at "best" even for an outside-anchored
                # legend. A real index change must invoke its apply handler.
                if choice.currentIndex() == choice.findText('best'):
                    capture_ui.select_combo(capture, choice, choice.findText('upper right'))
                capture_ui.select_combo(capture, choice, choice.findText('best'))
            else:
                choice.setFocus()
                QTest.keyClick(choice, Qt.Key_Home)
                QTest.keyClick(choice, Qt.Key_Down)
                QTest.keyClick(choice, Qt.Key_Home)
                QTest.keyClick(choice, Qt.Key_Tab)
            settle(.3)
            if choice.currentText() != 'best':
                raise ValueError('The native Legend choice did not change')
            S.end(name, choice)
            if next_name:
                S.begin(next_name, lead=.3)
            dialog.accepted.connect(lambda: accepted.append(True))
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    modal_wait(capture, 350, choose)
    QTest.mouseClick(queue._fig_settings_btn, Qt.LeftButton)
    settle(.5)
    if errors or not accepted:
        raise ValueError('; '.join(errors) or 'The real figure dialog was not accepted')
    bounds = figure.axes[0].get_legend().get_window_extent(figure.canvas.get_renderer())
    canvas = figure.bbox
    if not (canvas.x0 <= bounds.x0 <= bounds.x1 <= canvas.x1 and
            canvas.y0 <= bounds.y0 <= bounds.y1 <= canvas.y1):
        raise ValueError('The actual legend still exceeds the figure canvas')
    return dict(legend_inside_canvas=True, legend_bounds=list(bounds.bounds),
                canvas_bounds=list(canvas.bounds), actual_native_dialog=True)
