"""Record Mask generation's per-object settings table (item 592)."""


def record_object_table(app, window, screen, captures, capture, settle, write_json,
                        name="03b_object_table"):
    """Show every setting, scroll the per-object table into view, then restore.

    The table is the only layout of Mask's per-object settings: one column per
    object, one row per question, plus addable object-filter rows.
    """
    from capture_geometry import capture_rect
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QScrollArea

    from spacr.qt.settings_search import DISCLOSURE_NAME

    grid = getattr(screen, "_object_grid", None)
    if grid is None:
        raise RuntimeError("Mask generation has no per-object settings table")
    toggle = screen.findChild(QAbstractButton, DISCLOSURE_NAME)
    if toggle is None or not toggle.isVisible():
        raise RuntimeError("The Essentials / All settings switch is not visible")
    was_all = toggle.isChecked()
    if not was_all:
        QTest.mouseClick(toggle, Qt.LeftButton)
        settle(2)
    section = grid.parentWidget()
    while section is not None and type(section).__name__ != "Section":
        section = section.parentWidget()
    scroll = grid.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    if section is None or scroll is None:
        raise RuntimeError("Cannot find the per-object table's section and scroll area")
    top = section.mapTo(scroll.widget(), QPoint(0, 0)).y()
    scroll.verticalScrollBar().setValue(max(0, top - 8))
    settle(1.5)
    if not grid.isVisible():
        raise RuntimeError("The per-object table is not visible under All settings")
    region = capture_rect(section, window)
    capture(name)
    write_json(captures / "object_table.json", {
        "rect": region, "questions": len(grid.questions()),
        "shown_under": "All settings"})
    scroll.verticalScrollBar().setValue(0)
    if not was_all:
        QTest.mouseClick(toggle, Qt.LeftButton)
    settle(1.5)
    return region


def record_section(app, window, screen, captures, capture, settle, write_json,
                   title, name, *, all_settings=True):
    """Scroll one settings category into view (under All settings) and record it."""
    from capture_geometry import capture_rect
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QScrollArea

    from spacr.qt.settings_search import DISCLOSURE_NAME

    toggle = screen.findChild(QAbstractButton, DISCLOSURE_NAME)
    was_all = toggle.isChecked() if toggle is not None else True
    if all_settings and not was_all:
        QTest.mouseClick(toggle, Qt.LeftButton)
        settle(2)
    sections = list(getattr(screen, "_settings_sections", []))
    titles = [str(s.property("settingsCategorySource")) for s in sections]
    matches = [s for s, t in zip(sections, titles) if t.lower() == title.lower()]
    if not matches:
        write_json(captures / f"{name}_sections.json", titles)
        raise RuntimeError(f"No settings category {title!r}; see {name}_sections.json")
    section = matches[0]
    if hasattr(section, "set_expanded"):
        section.set_expanded(True)
    scroll = section.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    settle(1)
    top = section.mapTo(scroll.widget(), QPoint(0, 0)).y()
    scroll.verticalScrollBar().setValue(max(0, top - 8))
    settle(1.5)
    if not section.isVisible():
        raise RuntimeError(f"{title} is not visible")
    region = capture_rect(section, window)
    capture(name)
    write_json(captures / f"{name}.json", {"title": title, "rect": region, "sections": titles})
    scroll.verticalScrollBar().setValue(0)
    if all_settings and not was_all:
        QTest.mouseClick(toggle, Qt.LeftButton)
    settle(1.5)
    return region


def record_measure_qc(app, window, screen, captures, capture, settle, write_json,
                      name="09a_qc_popup", timeout=30):
    """Turn on Measure's QC switch, record its popup, then turn it off."""
    import time

    from capture_geometry import capture_rect
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    toggle = getattr(screen, "_seg_qc_toggle", None)
    dialog = getattr(screen, "_seg_qc_dialog", None)
    if toggle is None or dialog is None or not toggle.isVisible():
        raise RuntimeError("Measure has no visible QC switch")
    QTest.mouseClick(toggle, Qt.LeftButton)
    settle(1)
    if not toggle.isChecked():
        toggle.setChecked(True)
    deadline = time.monotonic() + timeout
    while not dialog.isVisible() and time.monotonic() < deadline:
        settle(0.2)
    settle(4)
    if not dialog.isVisible():
        raise RuntimeError("The QC popup did not open")
    regions = {"switch": capture_rect(toggle, window), "popup": capture_rect(dialog, window)}
    capture(name)
    write_json(captures / f"{name}.json", {"regions": regions,
               "banner_visible": not dialog.banner.isHidden()})
    dialog.close()
    settle(1)
    return regions
