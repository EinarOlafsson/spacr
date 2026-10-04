"""Record the alpha organism pages for lesson 86_alpha_organism_modules.

The maintainer's exception (2026-10-04): this one lesson turns Preferences >
Show alpha species on, on camera, tours every page that setting reveals, and
turns it off again. It runs only with ``--module alpha_organisms
--alpha-lesson 86_alpha_organism_modules`` (capture_policy.ALPHA_LESSONS);
Show alpha features stays off throughout. No analysis or external
application is started.
"""
from __future__ import annotations


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
            if tabs.tabText(index).lower().startswith("assays"):
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                 pos=tabs.tabBar().tabRect(index).center())
                settle(.6)
                return tabs.widget(index)
        raise RuntimeError("Home has no Assays tab")

    def tiles(key):
        return [b for b in window._startup.findChildren(QAbstractButton)
                if b.property("moduleAppKey") == key and b.isVisible()]

    def toggle_species(on, before, after):
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

    assays_tab()
    if any(tiles(key) for key in species):
        raise RuntimeError("An alpha organism tile shows while the setting is off")
    capture("00_home_assays_off")
    toggle_species(True, "01_prefs_species_off", "02_prefs_species_on")
    assays_tab()
    shown = [key for key in species if tiles(key)]
    if shown != species:
        raise RuntimeError(f"Not every alpha organism tile appeared: {shown}")
    capture("03_home_assays_on")

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
                    scroll.ensureWidgetVisible(tile)
                    settle(.3)
                click(tile)
                return
        raise RuntimeError(f"No Home tile for {key}")

    def exclude_release_history_now():
        from capture_policy import exclude_release_history
        exclude_release_history(window)

    for key in species:
        open_page(key)
        screen = window._screens.get(key)
        if screen is None or not screen.isVisible():
            raise RuntimeError(f"The {key} tile did not open its page")
        settle(.8)
        capture(f"{key}_page")
        handle = screen._splitter.handle(1)
        origin = handle.rect().center()
        QTest.mousePress(handle, Qt.LeftButton, pos=origin)
        QTest.mouseMove(handle, origin + QPoint(220, 0), delay=100)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=origin + QPoint(220, 0))
        settle(.8)
        diagram = screen._diagram
        screen._scroll.ensureWidgetVisible(diagram)
        settle(.5)
        selector = diagram.selector
        picked = []
        for index in (0, 2):
            if index >= selector.count():
                continue
            item = selector.item(index)
            selector.scrollToItem(item)
            region = selector.visualItemRect(item)
            QTest.mouseMove(selector.viewport(), region.center())
            QTest.mouseClick(selector.viewport(), Qt.LeftButton,
                             pos=QPoint(region.left() + 10, region.center().y()))
            settle(.3)
            if item.checkState() != Qt.Checked:
                raise RuntimeError(f"{key}: the compartment checkbox did not select")
            picked.append(item.text())
        capture(f"{key}_diagram")
        click(diagram.clear_button)
        screen._module_scroll.verticalScrollBar().setValue(0)
        settle(.4)
        capture(f"{key}_modules")
        evidence["pages"][key] = {
            "compartments": [selector.item(i).text() for i in range(selector.count())],
            "selected": picked,
            "tiles": [{"key": t.property("organismModuleKey"), "enabled": t.isEnabled(),
                       "text": t.text() if hasattr(t, "text") else ""}
                      for t in screen._tiles]}

    toggle_species(False, "90_prefs_species_on_again", "91_prefs_species_off_again")
    assays_tab()
    if any(tiles(key) for key in species):
        raise RuntimeError("Alpha organism tiles remain after turning the setting off")
    capture("92_home_assays_off_again")
    evidence["accepted"] = True
    write_json(captures / "scientific_acceptance.json", evidence)
