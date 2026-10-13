"""Use the real figure dialog to keep a tutorial's legend on the canvas."""


def fit_legend(app, window, queue, figure, capture, settle, name):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QComboBox, QDialogButtonBox
    from spacr.qt.widgets.figure_settings import FigureSettingsDialog

    errors, accepted = [], []

    def choose():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, FigureSettingsDialog):
                raise ValueError('The native Figure settings dialog did not open')
            watchdog = QTimer(dialog)
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(dialog.reject)
            watchdog.start(15000)
            dialog.resize(1000, 1500)
            dialog.move(window.geometry().center() - dialog.rect().center())
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
            page.ensureWidgetVisible(choice)
            choice.setFocus()
            QTest.keyClick(choice, Qt.Key_Home)
            QTest.keyClick(choice, Qt.Key_Down)
            QTest.keyClick(choice, Qt.Key_Home)
            QTest.keyClick(choice, Qt.Key_Tab)
            settle(.3)
            if choice.currentText() != 'best':
                raise ValueError('The native Legend choice did not change')
            capture(name)
            dialog.accepted.connect(lambda: accepted.append(True))
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(350, choose)
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
