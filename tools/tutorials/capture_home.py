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
        def resize():
            body.moveSplitter(sizes[0] + 420, 1)
            column.set_scale(1.25)
        live = getattr(capture, 'supports_live_actions', False)
        if not live:
            resize()
            settle(1.2)
        handle = body.handle(1)
        runtime = getattr(screen, "_runtime_wrap", None)
        regions = {"handle": capture_rect(handle, window),
                   "column": capture_rect(runtime, window) if runtime is not None else None}
        if live:
            capture("14_resize_panels", actions=[(1.0, 'Resize actual settings splitter and column text', resize)])
        else:
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


def record_home_navigation(app, window, capture, settle):
    """Refresh Home lesson chrome using actual tabs, dock rows and Ctrl+K."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton
    from spacr.qt import app as gui
    from spacr.qt.command_palette import CommandPalette

    home = window._startup
    tabs = home._tabs
    for index in range(1, tabs.count()):
        QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                         pos=tabs.tabBar().tabRect(index).center())
        settle()
        capture(f'{index:02d}_{tabs.tabText(index).lower()}')

    def select(key):
        rows = [row for row in window._sidebar._rows if row.key == key and row.isVisible()]
        if len(rows) != 1:
            raise RuntimeError(f"No unique visible sidebar route for {key}")
        QTest.mouseClick(rows[0], Qt.LeftButton, pos=rows[0].rect().center())
        settle(3)
        if key != '__home__' and window._stack.currentWidget() is not window._screens[key]:
            raise RuntimeError(f"Sidebar click did not open {key}")

    for key, _, _, _ in gui.tiled_apps():
        tiles = [button for button in home.findChildren(QAbstractButton)
                 if button.property('moduleAppKey') == key]
        for index in range(tabs.count()):
            choices = [tile for tile in tiles if tabs.widget(index).isAncestorOf(tile)]
            if choices:
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                 pos=tabs.tabBar().tabRect(index).center())
                tile = max(choices, key=lambda button: button.width() * button.height())
                QTest.mouseMove(tile, tile.rect().center())
                settle(.4)
                capture('home_module_' + key)
                break
        else:
            raise RuntimeError('Current Home has no visible tile for ' + key)
    from spacr.qt import preferences as prefs
    previous_backdrop, previous_density = prefs.get_ambient_animation(), prefs.get_ambient_density()
    prefs.set_ambient_animation('blobs')
    prefs.set_ambient_density(2.0)
    prefs.apply_preferences_to_app(app)
    window.setProperty('tutorialMaskBlobs', True)
    select('mask')
    capture('06_mask_host')
    capture('06b_mask_settings')
    capture('06c_mask_actions')
    focus = {'14_resize_panels': record_resize_panels(app, window, capture, settle)}
    prefs.set_ambient_animation(previous_backdrop)
    prefs.set_ambient_density(previous_density)
    prefs.apply_preferences_to_app(app)
    window.setProperty('tutorialMaskBlobs', False)
    select('__home__')
    focus.update(record_home_column(window, capture, settle))
    focus['08_help_search'] = record_help_search(app, window, capture, settle)
    errors = []

    def record_palette():
        visible = [widget for widget in app.topLevelWidgets()
                   if isinstance(widget, CommandPalette) and widget.isVisible()]
        palette = visible[0] if len(visible) == 1 else None
        try:
            if palette is None:
                raise RuntimeError("Actual Ctrl+K route did not open the command palette")
            QTest.keyClicks(palette._input, 'Meas')
            settle(1)
            if palette._list.count() == 0:
                raise RuntimeError("Actual command palette has no Measure result")
            from capture_geometry import capture_rect
            focus['12_command_palette'] = capture_rect(palette, window)
            capture('12_command_palette')
            QTest.keyClick(palette, Qt.Key_Escape)
            settle()
            if palette.isVisible():
                raise RuntimeError("Escape did not close the actual command palette")
        except Exception as exc:
            errors.append(str(exc))
        finally:
            if palette is not None and palette.isVisible():
                palette.reject()

    from PySide6.QtCore import QTimer
    watchdog = QTimer(window)
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: [widget.reject() for widget in app.topLevelWidgets()
                                     if isinstance(widget, CommandPalette) and widget.isVisible()])
    window.raise_()
    window.activateWindow()
    from PySide6.QtWidgets import QApplication
    QApplication.setActiveWindow(window)
    window.setFocus()
    settle(.4)
    QTimer.singleShot(350, record_palette)
    watchdog.start(max(12000, int(getattr(capture, 'maximum_live_clip_seconds', 0)*1000)+8000))
    QTest.keyClick(window, Qt.Key_K, Qt.ControlModifier)
    watchdog.stop()
    if errors or '12_command_palette' not in focus:
        raise RuntimeError('; '.join(errors) or "No actual Ctrl+K capture")
    return focus


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
    from PySide6.QtWidgets import QComboBox, QScrollArea, QTabWidget, QWidget

    from spacr.qt.preferences import PreferencesDialog
    from spacr.qt.widgets.section import Section

    dialog = PreferencesDialog(window)
    proof = {}
    try:
        tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
        page = dialog.findChild(QWidget, "PreferencesTabAppearance")
        for index in range(tabs.count()):
            if tabs.widget(index).isAncestorOf(page):
                tabs.setCurrentIndex(index)
                break
        dialog.resize(1200, 1800)
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
            if key == "animation":
                names = ("AmbientSpeed", "AmbientSize", "AmbientResolution",
                         "AmbientDensity", "AmbientGravityRadius", "FieldRipplesEnabled",
                         "FieldPopupWaveFrequency", "PopupBackdrop", "PopupBackdropSpeed",
                         "PopupBackdropSize", "PopupBackdropResolution", "PopupBackdropDensity",
                         "PopupBackdropDarkness")
                viewport = scroll.viewport()
                controls = {name: dialog.findChild(QWidget, name) for name in names}
                for control_name, control in controls.items():
                    if control is None or not control.isVisible():
                        raise RuntimeError(f"Animation control is not visible: {control_name}")
                    origin = control.mapTo(viewport, control.rect().topLeft())
                    if not viewport.rect().contains(origin) or not viewport.rect().contains(
                            control.mapTo(viewport, control.rect().bottomRight())):
                        raise RuntimeError(f"Animation control leaves the viewport: {control_name}")
                from spacr.qt.widgets.ambient import SPACEOUT_ONLY_THEMES
                choices = {}
                for control_name in ("AmbientTheme", "PopupBackdrop"):
                    box = dialog.findChild(QComboBox, control_name)
                    values = [box.itemData(i) for i in range(box.count())]
                    if set(values) & set(SPACEOUT_ONLY_THEMES):
                        raise RuntimeError("Ordinary tutorial includes a Spaceout-only backdrop")
                    choices[control_name] = values
                pairs = (("PopupBackdropSpeed", "AmbientSpeed"),
                         ("PopupBackdropSize", "AmbientSize"),
                         ("PopupBackdropResolution", "AmbientResolution"),
                         ("PopupBackdropDensity", "AmbientDensity"))
                independence = {}
                for settings_name, main_name in pairs:
                    settings, main = controls[settings_name], controls[main_name]
                    before, main_before = settings.value(), main.value()
                    settings.setFocus()
                    QTest.keyClick(settings, Qt.Key_Right)
                    if settings.value() == before or main.value() != main_before:
                        raise RuntimeError(f"Settings animation is not independent: {settings_name}")
                    independence[settings_name] = {"before": before, "edited": settings.value(),
                                                   "main_unchanged": main.value()}
                    settings.setValue(before)
                proof = {"all_named_controls_visible": list(names), "ordinary_choices": choices,
                         "settings_controls_independent": independence,
                         "field_ripples_separate_control": controls["FieldRipplesEnabled"].isChecked()}
                settle(.5)
            capture(name)
        for section in sections.values():
            if section.is_expanded():
                QTest.mouseClick(section.header(), Qt.LeftButton)
                settle(0.3)
    finally:
        dialog.close()
        dialog.deleteLater()
        settle()
    return proof


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


def record_spaceout_effects(window, capture, settle):
    """Capture all eight actual Spaceout switches and their reversible Apply path."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialogButtonBox, QScrollArea, QWidget
    from spacr.qt import preferences
    from spacr.qt.widgets.section import Section

    names = {"attractors": "SpaceoutFieldAttractors", "relaxation": "SpaceoutFieldRelaxation",
             "elastic_release": "SpaceoutFieldElasticRelease", "vortex": "SpaceoutFieldVortex",
             "density_pulses": "SpaceoutFieldDensityPulses", "density_waves": "SpaceoutFieldDensityWaves",
             "color_waves": "SpaceoutFieldColorWaves", "spirals": "SpaceoutFieldSpirals"}
    dialog, page = _open_preferences_tab(window, settle, "PreferencesTabAppearance", 1500)
    proof = {"saved_defaults": preferences._spaceout_field_effects(), "independent_controls": {}}
    try:
        sections = {section.title().casefold(): section for section in page.findChildren(Section)}
        section = sections["spacr field"]
        QTest.mouseClick(section.header(), Qt.LeftButton)
        settle(.5)
        if not section.is_expanded():
            raise RuntimeError("The Spaceout field section did not open through its header")
        scroll = page.parentWidget()
        while scroll is not None and not isinstance(scroll, QScrollArea):
            scroll = scroll.parentWidget()
        scroll.ensureWidgetVisible(section.header(), 0, 0)
        settle(.5)
        controls = {key: dialog.findChild(QWidget, name) for key, name in names.items()}
        for key, control in controls.items():
            if control is None or not control.isVisible() or not control.isChecked():
                raise RuntimeError(f"The default Spaceout effect is not visible and on: {key}")
            viewport = scroll.viewport()
            if not viewport.rect().contains(control.mapTo(viewport, control.rect().bottomRight())):
                raise RuntimeError(f"Spaceout effect leaves the actual viewport: {key}")
        capture("02_spaceout_field_effects")
        for key, control in controls.items():
            QTest.mouseClick(control, Qt.LeftButton)
            if control.isChecked() or any(not other.isChecked()
                                         for other_key, other in controls.items() if other_key != key):
                raise RuntimeError(f"Spaceout effect control is not independent: {key}")
            proof["independent_controls"][key] = {"off": True, "other_seven_unchanged": True}
            QTest.mouseClick(control, Qt.LeftButton)
        QTest.mouseClick(controls["vortex"], Qt.LeftButton)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Apply).click()
        settle(.6)
        question = dialog._apply_confirmation
        if question is None or not question.isVisible() or preferences._spaceout_field_effects()["vortex"]:
            raise RuntimeError("Spaceout Vortices edit was not previewed through real Apply")
        capture("03_spaceout_effect_apply_revert")
        next(button for button in question.buttons() if button.text() == "Revert").click()
        settle(.5)
        if preferences._spaceout_field_effects() != proof["saved_defaults"] or not dialog.isVisible():
            raise RuntimeError("Spaceout Revert failed to restore the exact eight-effect state")
        proof["apply_revert_verified"] = True
        proof["published"] = False
        proof["scope"] = "Current Spaceout companion; ordinary Home lesson stays alpha-off"
        return proof
    finally:
        dialog.reject()
        dialog.deleteLater()
        settle()


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


def record_preferences_apply(window, capture, settle):
    """Record the real preview question and verify Keep and Revert with Preferences open."""
    from capture_geometry import capture_rect
    from PySide6.QtWidgets import QDialogButtonBox, QMessageBox, QSlider
    from spacr.qt import preferences

    dialog, _page = _open_preferences_tab(window, settle, "PreferencesTabAppearance", 1500)
    original = preferences.get_pane_opacity()
    proof = {"original_opacity": original}
    try:
        slider = dialog.findChild(QSlider, "PaneOpacity")
        buttons = dialog.findChild(QDialogButtonBox)
        if slider is None or buttons is None:
            raise RuntimeError("Preferences has no opacity slider or buttons")
        preview = 55 if original != 0.55 else 60
        slider.setValue(preview)
        buttons.button(QDialogButtonBox.Apply).click()
        settle(1)
        question = dialog._apply_confirmation
        if not isinstance(question, QMessageBox) or not question.isVisible() or not dialog.isVisible():
            raise RuntimeError("Apply must show a separate question above visible Preferences")
        if abs(preferences.get_pane_opacity() - preview / 100) > 1e-9:
            raise RuntimeError("Apply did not preview the edited setting")
        proof["question"] = capture_rect(question, window)
        proof["preferences"] = capture_rect(dialog, window)
        capture("13h_preferences_apply")
        next(button for button in question.buttons() if button.text() == "Revert").click()
        settle(1)
        if preferences.get_pane_opacity() != original or not dialog.isVisible():
            raise RuntimeError("Revert must restore the saved value and keep Preferences open")
        proof["revert_verified"] = True
        slider.setValue(round(original * 100))
        buttons.button(QDialogButtonBox.Apply).click()
        settle(1)
        question = dialog._apply_confirmation
        next(button for button in question.buttons() if button.text() == "Keep").click()
        settle(1)
        if preferences.get_pane_opacity() != original or not dialog.isVisible():
            raise RuntimeError("Keep must preserve the applied value and keep Preferences open")
        proof["keep_verified"] = True
        capture("13i_preferences_kept")
        return proof
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
            cancel = [b for b in box.buttons()
                      if box.buttonRole(b) in (QMessageBox.RejectRole, QMessageBox.NoRole)]
            button = cancel[0] if cancel else box.escapeButton()
            if button is None:
                box.reject()
                proof['cancel_method'] = 'dismiss confirmation without accepting deletion'
            else:
                button.click()
                proof['cancel_method'] = 'native No or Cancel button'
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
