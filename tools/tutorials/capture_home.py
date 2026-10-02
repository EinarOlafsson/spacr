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


def record_performance(window, capture, settle, *, dialog_size=(1200, 1200)):
    """Capture Performance and the current Appearance category navigation.

    :param window: owning recording window.
    :param capture: callback receiving each stable scene name.
    :param settle: callback that processes events until painting settles.
    :param dialog_size: recording dialog size in logical pixels.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QComboBox, QScrollArea, QTabWidget
    from spacr.qt.widgets.section import Section

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
        dialog.resize(*dialog_size)
        dialog.move(window.mapToGlobal(window.rect().center()) - dialog.rect().center())
        dialog.show()
        settle()
        if not selector.isVisible():
            raise RuntimeError("The Performance selector is not visible")
        capture("09_performance")
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        if "Theme" in titles or "Animation" in titles or "Appearance" not in titles:
            raise RuntimeError("Tutorial requires Theme and Animation inside Appearance")
        QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                         pos=tabs.tabBar().tabRect(titles.index("Appearance")).center())
        settle()
        page = tabs.currentWidget()
        categories = {section.title(): section for section in page.findChildren(Section)}
        if not {"THEME", "ANIMATION"} <= categories.keys():
            raise RuntimeError("Appearance is missing its Theme or Animation category")
        if any(categories[name].is_expanded() for name in ("THEME", "ANIMATION")):
            raise RuntimeError("Appearance categories must begin folded")
        if isinstance(page, QScrollArea):
            page.ensureWidgetVisible(categories["ANIMATION"]._header)
        settle()
        capture("10_appearance_categories")
        for title, frame in (("THEME", "11_appearance_theme"),
                             ("ANIMATION", "12_appearance_animation")):
            section = categories[title]
            if isinstance(page, QScrollArea):
                page.ensureWidgetVisible(section._header)
            QTest.mouseClick(section._header, Qt.LeftButton)
            settle()
            if not section.is_expanded():
                raise RuntimeError(f"The {title} category did not open")
            if isinstance(page, QScrollArea):
                page.ensureWidgetVisible(section._header)
            settle()
            controls = section.findChildren(QComboBox)
            if not controls or not any(control.isVisible() for control in controls):
                raise RuntimeError(f"The {title} controls are not visible")
            capture(frame)
            QTest.mouseClick(section._header, Qt.LeftButton)
            settle()

    finally:
        dialog.close()
        dialog.deleteLater()
        settle()
