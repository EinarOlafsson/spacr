"""Record the real Help search field and its generated results."""
import time


def record_help_search(app, window, capture, settle, name="08_help_search"):
    from capture_geometry import capture_rect
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from spacr.qt.help_search import field_of

    field = field_of(window)
    if field is None or not field.isVisible():
        raise RuntimeError("The Help search field is not visible")
    field.setFocus()
    QTest.keyClick(field, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(field, "Performance")
    deadline = time.monotonic() + 30
    while not field.results():
        if time.monotonic() >= deadline:
            raise TimeoutError("Help search did not return real Performance results")
        settle()
    settle()
    if not field.popup().isVisible():
        raise RuntimeError("Help search results are not visible")
    region = capture_rect(field.popup(), window)
    capture(name)
    QTest.keyClick(field, Qt.Key_Escape)
    field.clear()
    settle()
    return region


def record_performance(window, capture, settle):
    from PySide6.QtWidgets import QComboBox, QTabWidget

    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog(window)
    try:
        tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
        selector = dialog.findChild(QComboBox, "PerformanceLevel")
        if tabs is None or selector is None:
            raise RuntimeError("Preferences has no Performance selector")
        for index in range(tabs.count()):
            if tabs.widget(index).findChild(QComboBox, "PerformanceLevel") is selector:
                tabs.setCurrentIndex(index)
                break
        dialog.resize(1200, 1200)
        dialog.show()
        settle()
        if not selector.isVisible():
            raise RuntimeError("The Performance selector is not visible")
        capture("09_performance")
    finally:
        dialog.close()
        dialog.deleteLater()
        settle()
