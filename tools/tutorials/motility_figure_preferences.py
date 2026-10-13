"""Choose the app's existing Screen export mode through real Preferences."""


def choose_screen_export(app, window, screen, capture, settle):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialog, QTabWidget, QComboBox, QScrollArea, QDialogButtonBox
    from spacr.qt.preferences import get_figure_save_mode

    errors, accepted = [], []
    timer, watchdog = QTimer(window), QTimer(window)
    timer.setSingleShot(True)
    watchdog.setSingleShot(True)
    before = get_figure_save_mode()

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('An actual figure-export preference is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.2)

    def configure():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QDialog):
                raise ValueError('The real Preferences dialog did not open')
            dialog.accepted.connect(lambda: accepted.append(True))
            tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
            combo = dialog.findChild(QComboBox, 'FigureSaveMode')
            if tabs is None or combo is None:
                raise ValueError('The actual Figure save mode control is absent')
            dialog.resize(1800, 1600)
            pages = [i for i in range(tabs.count()) if tabs.widget(i).isAncestorOf(combo)]
            if len(pages) != 1 or not isinstance(tabs.widget(pages[0]), QScrollArea):
                raise ValueError('No unique scrollable Figures preference page')
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton, pos=tabs.tabBar().tabRect(pages[0]).center())
            tabs.widget(pages[0]).ensureWidgetVisible(combo)
            settle(.3)
            click(combo)
            QTest.keyClick(combo, Qt.Key_Home)
            index = combo.findData('screen')
            if index < 0:
                raise ValueError('The actual Screen export option is absent')
            for _ in range(index):
                QTest.keyClick(combo, Qt.Key_Down)
            QTest.keyClick(combo, Qt.Key_Return)
            settle(.2)
            if combo.currentData() != 'screen':
                raise ValueError('The real export option was not selected')
            capture('02_actual_screen_export_preference')
            box = dialog.findChild(QDialogButtonBox)
            click(box.button(QDialogButtonBox.Save))
        except Exception as error:
            errors.append(str(error))
            if isinstance(dialog, QDialog):
                dialog.reject()

    def abort():
        errors.append('The actual figure-export Preferences dialog timed out')
        if app.activeModalWidget() is not None:
            app.activeModalWidget().reject()

    timer.timeout.connect(configure)
    watchdog.timeout.connect(abort)
    timer.start(350)
    watchdog.start(20000)
    try:
        click(screen._btn_preferences)
    finally:
        timer.stop()
        watchdog.stop()
    if errors or not accepted or get_figure_save_mode() != 'screen':
        raise ValueError('; '.join(errors) or 'Screen export mode was not saved')
    return dict(before=before, after=get_figure_save_mode(), actual_gui_saved=True,
                scope='Private recorder preferences only; no desktop settings changed')
