"""Record the alpha organism pages for lesson 86_alpha_organism_modules.

The maintainer's exception (2026-10-04): this one lesson turns Preferences >
Show alpha species on, on camera, tours every page that setting reveals, and
turns it off again. It runs only with ``--module alpha_organisms
--alpha-lesson 86_alpha_organism_modules`` (capture_policy.ALPHA_LESSONS);
Show alpha features stays off throughout. No analysis or external
application is started.
"""
from __future__ import annotations

import capture_ui
from capture_ui_drv_a import pick_list_row
from capture_ui_drv_e import guarded
from capture_ui_drv_f import prefs_toggle_live


@guarded
def record_alpha_organisms(app, window, stage, captures, capture, settle, write_json):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QDialogButtonBox, QScrollArea,
                                   QTabWidget, QWidget)

    from spacr import settings as spacr_settings
    from spacr.qt import preferences as prefs
    from spacr.qt.preferences import PreferencesDialog

    species = [key for entry in spacr_settings.ALPHA_SPECIES.values()
               for key in entry.get("apps", ())]
    if not species:
        raise RuntimeError("ALPHA_SPECIES lists no organism pages")
    if prefs._get_show_alpha_species() or prefs._get_show_alpha_features():
        raise RuntimeError("Start with Show alpha species and Show alpha features off")

    def click(widget):
        visible = widget.visibleRegion()
        if not widget.isEnabled() or visible.isEmpty():
            raise RuntimeError("A required control is unavailable")
        QTest.mouseClick(widget, Qt.LeftButton, pos=visible.boundingRect().center())
        settle(.6)

    def home():
        if not window._startup.isVisible():
            choices = [b for b in window.findChildren(QAbstractButton)
                       if b.property("navKey") == "__home__" and b.isVisible()]
            if choices:
                click(max(choices, key=lambda b: b.width() * b.height()))
            else:
                # Organism pages fill the window; the spaCR menu's Home entry
                # is the visible way back.
                bar = window.menuBar()
                top = [a for a in bar.actions() if a.text().replace("&", "") == "spaCR"]
                if len(top) != 1:
                    raise RuntimeError("No unique spaCR menu")
                QTest.mouseClick(bar, Qt.LeftButton, pos=bar.actionGeometry(top[0]).center())
                settle(.4)
                menu = top[0].menu()
                entry = [a for a in menu.actions() if a.text().replace("&", "") == "Home"]
                if len(entry) != 1 or not menu.isVisible():
                    raise RuntimeError("The spaCR menu has no Home entry")
                QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(entry[0]).center())
                settle(.8)
        if not window._startup.isVisible():
            raise RuntimeError("Home did not open")

    def assays_tab():
        home()
        from capture_policy import exclude_release_history
        exclude_release_history(window)
        tabs = window._startup._tabs
        for index in range(tabs.count()):
            # The Home tab that holds the organisms was called Assays; it is now Organism.
            if tabs.tabText(index).lower().startswith(("assays", "organism")):
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                 pos=tabs.tabBar().tabRect(index).center())
                settle(.6)
                return tabs.widget(index)
        raise RuntimeError("Home has no Assays/Organism tab: " + repr([tabs.tabText(i) for i in range(tabs.count())]))

    def tiles(key):
        return [b for b in window._startup.findChildren(QAbstractButton)
                if b.property("moduleAppKey") == key and b.isVisible()]

    def toggle_species(on, before, after):
        if capture_ui.live(capture):
            # Host: the spaCR menu opens Preferences and the pointer does the rest;
            # the clips are before/after/save when turning on, one clip when turning off.
            frames = (before, after, after + '_save') if on else (None, after, None)
            prefs_toggle_live(app, window, capture, settle, 'ShowAlphaSpecies', on, frames)
            settle(1.5)
            from capture_policy import exclude_release_history
            exclude_release_history(window)
            settle(.3)
            if prefs._get_show_alpha_species() != on:
                raise RuntimeError("Save did not store Show alpha species")
            if prefs._get_show_alpha_features():
                raise RuntimeError("Show alpha features must stay off")
            return
        dialog = PreferencesDialog(window)
        try:
            toggle = dialog.findChild(QWidget, "ShowAlphaSpecies")
            tabs = dialog.findChild(QTabWidget, "PreferencesTabs")
            if toggle is None or tabs is None:
                raise RuntimeError("Preferences has no Show alpha species toggle")
            for index in range(tabs.count()):
                if tabs.widget(index).isAncestorOf(toggle):
                    tabs.setCurrentIndex(index)
                    break
            dialog.resize(1500, 1500)
            dialog.show()
            settle(1)
            scroll = toggle.parentWidget()
            while scroll is not None and not isinstance(scroll, QScrollArea):
                scroll = scroll.parentWidget()
            if scroll is not None:
                scroll.ensureWidgetVisible(toggle, 0, 200)
                settle(.5)
            capture(before)
            click(toggle)
            if toggle.isChecked() != on:
                raise RuntimeError("The visible toggle did not change")
            capture(after)
            saves = [box.button(QDialogButtonBox.Save)
                     for box in dialog.findChildren(QDialogButtonBox)]
            saves = [b for b in saves if b is not None and b.isVisible()]
            if len(saves) != 1:
                raise RuntimeError("Preferences has no unique Save button")
            capture(after + "_save")
            click(saves[0])
            settle(1.5)
            # Saving rebuilds Home, News aside included; omit it again as every
            # current-version recording does (capture_policy).
            from capture_policy import exclude_release_history
            exclude_release_history(window)
            settle(.3)
        finally:
            if dialog.isVisible():
                dialog.close()
            dialog.deleteLater()
            settle(.5)
        if prefs._get_show_alpha_species() != on:
            raise RuntimeError("Save did not store Show alpha species")
        if prefs._get_show_alpha_features():
            raise RuntimeError("Show alpha features must stay off")

    with capture_ui.clip(capture, "00_home_assays_off"):
        assays_tab()
        if any(tiles(key) for key in species):
            raise RuntimeError("An alpha organism tile shows while the setting is off")
        capture_ui.wait(capture, 1.5, settle)
    toggle_species(True, "01_prefs_species_off", "02_prefs_species_on")
    with capture_ui.clip(capture, "03_home_assays_on"):
        assays_tab()
        shown = [key for key in species if tiles(key)]
        if shown != species:
            raise RuntimeError(f"Not every alpha organism tile appeared: {shown}")
        capture_ui.point_at(capture, *[max(tiles(key), key=lambda b: b.width() * b.height())
                                       for key in species], dwell=0.8)

    evidence = {"lesson": "86_alpha_organism_modules", "species": species,
                "alpha_features_on": False, "analysis_started": False,
                "external_application_started": False, "pages": {}}
    def open_page(key):
        home()
        exclude_release_history_now()
        every = [b for b in window._startup.findChildren(QAbstractButton)
                 if b.property("moduleAppKey") == key]
        tabs = window._startup._tabs
        for index in range(tabs.count()):
            choices = [t for t in every if tabs.widget(index).isAncestorOf(t)]
            if choices:
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                 pos=tabs.tabBar().tabRect(index).center())
                settle(.6)
                tile = max(choices, key=lambda b: b.width() * b.height())
                scroll = tile.parentWidget()
                while scroll is not None and not isinstance(scroll, QScrollArea):
                    scroll = scroll.parentWidget()
                if scroll is not None:
                    capture_ui.scroll_to(capture, scroll, tile)
                    settle(.3)
                click(tile)
                return
        raise RuntimeError(f"No Home tile for {key}")

    def exclude_release_history_now():
        from capture_policy import exclude_release_history
        exclude_release_history(window)

    for key in species:
        # Host: each clip starts before its actions (Home, tab and tile; the divider
        # drag and compartment ticks; Clear and the assay tiles).
        with capture_ui.clip(capture, f"{key}_page"):
            open_page(key)
            screen = window._screens.get(key)
            if screen is None or not screen.isVisible():
                raise RuntimeError(f"The {key} tile did not open its page")
            settle(.8)
            capture_ui.point_at(capture, screen._intro, dwell=1.5)
        with capture_ui.clip(capture, f"{key}_diagram"):
            handle = screen._splitter.handle(1)
            origin = handle.rect().center()
            QTest.mousePress(handle, Qt.LeftButton, pos=origin)
            QTest.mouseMove(handle, origin + QPoint(220, 0), delay=100)
            QTest.mouseRelease(handle, Qt.LeftButton, pos=origin + QPoint(220, 0))
            settle(.8)
            diagram = screen._diagram
            capture_ui.scroll_to(capture, screen._scroll, diagram)
            settle(.5)
            selector = diagram.selector
            picked = []
            for index in (0, 2):
                if index >= selector.count():
                    continue
                item = selector.item(index)
                pick_list_row(capture, selector, index)
                settle(.3)
                if item.checkState() != Qt.Checked:
                    raise RuntimeError(f"{key}: the compartment checkbox did not select")
                picked.append(item.text())
            capture_ui.point_at(capture, diagram.artwork, dwell=1.5)
        with capture_ui.clip(capture, f"{key}_modules"):
            click(diagram.clear_button)
            capture_ui.scroll_to_value(capture, screen._module_scroll, 0)
            settle(.4)
            capture_ui.point_at(capture, *list(screen._tiles)[:3], dwell=1.2)
        evidence["pages"][key] = {
            "compartments": [selector.item(i).text() for i in range(selector.count())],
            "selected": picked,
            "tiles": [{"key": t.property("organismModuleKey"), "enabled": t.isEnabled(),
                       "text": t.text() if hasattr(t, "text") else ""}
                      for t in screen._tiles]}

    toggle_species(False, "90_prefs_species_on_again", "91_prefs_species_off_again")
    with capture_ui.clip(capture, "92_home_assays_off_again"):
        assays_tab()
        if any(tiles(key) for key in species):
            raise RuntimeError("Alpha organism tiles remain after turning the setting off")
        capture_ui.wait(capture, 1.5, settle)
    evidence["accepted"] = True
    write_json(captures / "scientific_acceptance.json", evidence)
