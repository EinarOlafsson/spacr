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


def record_resize_panels(app, window, capture, settle, key="mask"):
    """Drag a module screen's settings handle wider and enlarge its right column's text.

    Moves the real splitter the way a drag does and applies the Ctrl + wheel
    column scale (item 529) through its own filter, then restores both.
    """
    from capture_geometry import capture_rect

    from spacr.qt.live_zoom import _COLUMN_FILTER_ATTRIBUTE

    screen = window._screens.get(key)
    body = getattr(screen, "_body_splitter", None)
    if body is None or not body.isVisible():
        raise RuntimeError(f"{key} has no visible settings splitter")
    # Start from the screen's own proportions (settings 1 : runtime 2), not
    # from whatever earlier recordings left in this private profile.
    total = sum(body.sizes())
    body.setSizes([total // 3, total - total // 3])
    settle(0.5)
    sizes = body.sizes()
    column = getattr(app, _COLUMN_FILTER_ATTRIBUTE, None)
    if column is None:
        raise RuntimeError("The Ctrl + wheel column text filter is not installed")
    scale = column.scale()
    try:
        body.moveSplitter(sizes[0] + 420, 1)
        column.set_scale(1.25)
        settle(1.2)
        handle = body.handle(1)
        runtime = getattr(screen, "_runtime_wrap", None)
        regions = {"handle": capture_rect(handle, window),
                   "column": capture_rect(runtime, window) if runtime is not None else None}
        capture("14_resize_panels")
    finally:
        column.set_scale(scale)
        body.setSizes(sizes)
        settle()
    return regions


def record_home_column(window, capture, settle):
    """Show the Home right-hand column, its Text size slider and its fold handle."""
    from capture_geometry import capture_rect
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    home = window._startup
    split = home._aside_split
    slider = home._homeAsideTextSlider
    if home._aside_collapsed():
        raise RuntimeError("The Home column starts folded")
    original = slider.value()
    regions = {}
    try:
        # The slider stays at its default: above 100 % the System panel
        # currently paints its old rows under the new ones (reported, not
        # patched here), and a recording must not show that defect.
        settle(1.2)
        regions["column"] = capture_rect(home._aside, window)
        regions["slider"] = capture_rect(home._aside_controls, window)
        capture("10_home_column")
    finally:
        slider.setValue(original)
        settle()
    index = split.indexOf(home._aside)
    handle = split.handle(index)
    regions["handle"] = capture_rect(handle, window)
    QTest.mouseClick(handle, Qt.LeftButton, pos=handle.rect().center())
    settle(1.2)
    if not home._aside_collapsed():
        raise RuntimeError("Clicking the Home column handle did not fold it")
    regions["folded_handle"] = capture_rect(split.handle(index), window)
    capture("11_column_folded")
    QTest.mouseClick(split.handle(index), Qt.LeftButton,
                     pos=split.handle(index).rect().center())
    settle(1.2)
    if home._aside_collapsed():
        raise RuntimeError("Clicking the handle again did not restore the Home column")
    return regions


def record_command_palette(window, capture, settle, text="Meas"):
    """Open the Ctrl+K command palette and filter it by typing."""
    from capture_geometry import capture_rect
    from PySide6.QtTest import QTest

    from spacr.qt.command_palette import CommandPalette

    palette = CommandPalette(window)
    try:
        palette.show()
        settle()
        QTest.keyClicks(palette._input, text)
        settle(1.0)
        if palette._list.count() == 0:
            raise RuntimeError("The command palette listed nothing for its filter")
        region = capture_rect(palette, window)
        capture("12_command_palette")
    finally:
        palette.close()
        palette.deleteLater()
        settle()
    return region


def record_preferences_page(window, capture, settle, object_name="PreferencesTabGeneral",
                            name="13_preferences_general"):
    """Show one Preferences tab; never the Modules tab with the alpha toggle."""
    from PySide6.QtWidgets import QTabWidget, QWidget

    from spacr.qt.preferences import PreferencesDialog

    if object_name == "PreferencesTabModules":
        raise ValueError("The Modules tab holds the alpha toggle; not recorded here")
    dialog = PreferencesDialog(window)
    try:
        tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
        page = dialog.findChild(QWidget, object_name)
        if tabs is None or page is None:
            raise RuntimeError(f"Preferences has no {object_name}")
        for index in range(tabs.count()):
            if tabs.widget(index).isAncestorOf(page):
                tabs.setCurrentIndex(index)
                break
        dialog.resize(1200, 1200)
        dialog.show()
        settle()
        if not page.isVisible():
            raise RuntimeError(f"{object_name} is not visible")
        capture(name)
    finally:
        dialog.close()
        dialog.deleteLater()
        settle()


def record_appearance_sections(window, capture, settle):
    """Open Appearance's folded Theme and Animation sections, one at a time (601)."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QScrollArea, QTabWidget, QWidget

    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.widgets.section import Section

    dialog = PreferencesDialog(window)
    try:
        tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
        page = dialog.findChild(QWidget, "PreferencesTabAppearance")
        for index in range(tabs.count()):
            if tabs.widget(index).isAncestorOf(page):
                tabs.setCurrentIndex(index)
                break
        dialog.resize(1200, 1500)
        dialog.show()
        settle()
        sections = {s.title().lower(): s for s in page.findChildren(Section)}
        for key in ("theme", "animation"):
            if key not in sections:
                raise RuntimeError(f"Appearance has no folded {key} section: {sorted(sections)}")
        scroll = page.parentWidget()
        while scroll is not None and not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        for key, name in (("theme", "13c_appearance_theme"), ("animation", "13d_appearance_animation")):
            other = sections["animation" if key == "theme" else "theme"]
            if other.is_expanded():
                QTest.mouseClick(other.header(), Qt.LeftButton)
                settle(0.5)
            section = sections[key]
            if not section.is_expanded():
                QTest.mouseClick(section.header(), Qt.LeftButton)
                settle(1)
            if not section.is_expanded():
                raise RuntimeError(f"Clicking {key} did not open it")
            if scroll is not None:
                scroll.ensureWidgetVisible(section.header(), 0, 0)
            settle(1)
            capture(name)
        for section in sections.values():
            if section.is_expanded():
                QTest.mouseClick(section.header(), Qt.LeftButton)
                settle(0.3)
    finally:
        dialog.close()
        dialog.deleteLater()
        settle()


def _open_preferences_tab(window, settle, object_name, height=1200):
    """Open Preferences on the tab holding ``object_name``; returns (dialog, page)."""
    from PySide6.QtWidgets import QTabWidget, QWidget

    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog(window)
    tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
    page = dialog.findChild(QWidget, object_name)
    if tabs is None or page is None:
        dialog.deleteLater()
        raise RuntimeError(f"Preferences has no {object_name}")
    for index in range(tabs.count()):
        if tabs.widget(index).isAncestorOf(page):
            tabs.setCurrentIndex(index)
            break
    dialog.resize(1200, height)
    dialog.show()
    settle()
    return dialog, page


def record_session_and_updates(window, capture, settle, name="13e_session_updates"):
    """Show Session restore and the update channel without the alpha toggle in view.

    Both rows live on the Modules tab, whose last row holds Show alpha
    features. The page is scrolled to the Session row and the dialog is made
    short enough that the toggle stays below the visible part of the page.
    """
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QScrollArea, QWidget

    dialog, page = _open_preferences_tab(window, settle, "PreferencesTabModules", 1200)
    try:
        session = dialog.findChild(QWidget, "RestoreLastSession")
        channel = dialog.findChild(QWidget, "UpdateChannel")
        # Module visibility holds the maturity toggles and Show alpha
        # features; keep its whole row out of view.
        toggle = dialog.findChild(QWidget, "ShowAlphaFeatures")
        if session is None or channel is None or toggle is None:
            raise RuntimeError("Preferences has no Session, Update channel or alpha row")
        scroll = page.parentWidget()
        while scroll is not None and not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        info = {}
        for _ in range(8):
            settle(0.5)
            top = session.mapTo(scroll.widget(), QPoint(0, 0)).y()
            scroll.verticalScrollBar().setValue(max(0, top - 60))
            settle(0.5)
            viewport = scroll.viewport()
            toggle_y = toggle.mapTo(viewport, QPoint(0, 0)).y()
            channel_bottom = channel.mapTo(viewport, QPoint(0, channel.height())).y()
            info = {"viewport_h": viewport.height(), "toggle_y": toggle_y,
                    "channel_bottom": channel_bottom, "dialog_h": dialog.height()}
            if toggle_y >= viewport.height() and channel_bottom <= viewport.height():
                capture(name)
                info["restore_last_session"] = session.isChecked()
                info["update_channel"] = channel.currentText()
                return info
            shrink = viewport.height() - toggle_y + 40
            if shrink <= 0 or dialog.height() - shrink < channel_bottom + 200:
                break
            dialog.resize(dialog.width(), dialog.height() - shrink)
        raise RuntimeError(f"Cannot show Session and Update channel without the alpha toggle: {info}")
    finally:
        dialog.close()
        dialog.deleteLater()
        settle()


def record_storage(window, capture, settle):
    """Storage tab at its defaults, then Prune now's list and question, cancelled (643)."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    dialog, page = _open_preferences_tab(window, settle, "PreferencesTabStorage", 1500)
    proof = {}
    try:
        storage = dialog._storage_page
        settle(2)
        capture("13f_storage")
        keep, cap = storage.spins["run_folders"]
        old = (keep.value(), cap.value())
        # A tight example limit for run folders, so Prune has something to list.
        keep.setValue(1)
        cap.setValue(0)
        settle(0.5)
        asked = []

        def answer():
            boxes = [w for w in QApplication.topLevelWidgets()
                     if isinstance(w, QMessageBox) and w.isVisible()]
            if not boxes:
                if len(asked) < 150:
                    asked.append(None)
                    QTimer.singleShot(200, answer)
                return
            box = boxes[0]
            settle(0.5)
            proof["question"] = [box.windowTitle(), box.text(), box.informativeText()]
            capture("13g_storage_prune")
            asked.append(box)
            cancel = [b for b in box.buttons() if box.buttonRole(b) == QMessageBox.RejectRole]
            (cancel[0] if cancel else box.escapeButton()).click()
        QTimer.singleShot(300, answer)
        storage.prune()
        import time
        deadline = time.monotonic() + 120
        while not any(asked) and time.monotonic() < deadline:
            settle(0.2)
        if not any(asked):
            raise RuntimeError("Prune now did not ask before deleting")
        settle(1)
        proof["cancelled"] = True
        keep.setValue(old[0])
        cap.setValue(old[1])
    finally:
        dialog.reject()
        dialog.deleteLater()
        settle()
    return proof
