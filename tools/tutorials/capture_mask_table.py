"""Record Mask generation's per-object settings table (item 592)."""


import capture_ui
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
    # Host: the clip opens before the switch to All settings, so the viewer
    # sees the click, the scroll down to the table and the pointer reading it.
    host_clip = capture_ui.live(capture) and capture.begin(name)
    if host_clip:
        capture_ui.wait(capture, 0.8, settle)
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
    capture_ui.scroll_to_value(capture, scroll, max(0, top - 8))
    settle(1.5)
    if not grid.isVisible():
        raise RuntimeError("The per-object table is not visible under All settings")
    region = capture_rect(section, window)
    if host_clip:
        _read_table(capture, grid)
        capture.end()
    elif not capture_ui.live(capture):
        capture(name)
    write_json(captures / "object_table.json", {
        "rect": region, "questions": len(grid.questions()),
        "shown_under": "All settings"})
    capture_ui.scroll_to_value(capture, scroll, 0)
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
    host_clip = capture_ui.live(capture) and capture.begin(name)
    if host_clip:
        capture_ui.wait(capture, 0.8, settle)
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
    if capture_ui.live(capture):
        import capture_home_live
        if hasattr(section, "is_expanded") and not section.is_expanded():
            capture_home_live.toggle_section(capture, section, True)
    elif hasattr(section, "set_expanded"):
        section.set_expanded(True)
    scroll = section.parentWidget()
    while scroll is not None and not isinstance(scroll, QScrollArea):
        scroll = scroll.parentWidget()
    settle(1)
    top = section.mapTo(scroll.widget(), QPoint(0, 0)).y()
    capture_ui.scroll_to_value(capture, scroll, max(0, top - 8))
    settle(1.5)
    if not section.isVisible():
        raise RuntimeError(f"{title} is not visible")
    region = capture_rect(section, window)
    if host_clip:
        _read_section(capture, section)
        capture.end()
    elif not capture_ui.live(capture):
        capture(name)
    write_json(captures / f"{name}.json", {"title": title, "rect": region, "sections": titles})
    capture_ui.scroll_to_value(capture, scroll, 0)
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
    host_clip = capture_ui.live(capture) and capture.begin(name)
    if host_clip:
        capture_ui.wait(capture, 0.8, settle)
    QTest.mouseClick(toggle, Qt.LeftButton)
    settle(1)
    if not toggle.isChecked():
        if capture_ui.live(capture):
            raise RuntimeError("Clicking the QC switch did not turn it on")
        toggle.setChecked(True)
    deadline = time.monotonic() + timeout
    while not dialog.isVisible() and time.monotonic() < deadline:
        settle(0.2)
    settle(4)
    if not dialog.isVisible():
        raise RuntimeError("The QC popup did not open")
    regions = {"switch": capture_rect(toggle, window), "popup": capture_rect(dialog, window)}
    if host_clip:
        _read_popup(capture, dialog)
        capture.end()
    elif not capture_ui.live(capture):
        capture(name)
    write_json(captures / f"{name}.json", {"regions": regions,
               "banner_visible": not dialog.banner.isHidden()})
    capture_ui.close_dialog(capture, dialog)
    settle(1)
    return regions


def _read_table(capture, grid, dwell=1.0):
    """Host: the pointer runs along the object columns, then down the questions."""
    from PySide6.QtCore import QPoint
    h = capture_ui._h()
    table = grid._table
    header = table.horizontalHeader()
    port = table.viewport()
    if header.isVisible():
        for col in range(min(table.model().columnCount(), 4)):
            x = header.sectionViewportPosition(col) + header.sectionSize(col) // 2
            h.hover(header, QPoint(x, header.height() // 2))
            h.pause(capture, dwell)
    for row in range(0, min(table.model().rowCount(), 8), 2):
        rect = table.visualRect(table.model().index(row, 0))
        if rect.isValid() and port.rect().contains(rect.center()):
            h.hover(port, rect.center())
            h.pause(capture, dwell * 0.8)


def _read_section(capture, section, dwell=1.0, limit=6):
    """Host: the pointer rests on the section's visible controls, top to bottom."""
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit
    found = []
    for kind in (QComboBox, QLineEdit, QAbstractSpinBox, QAbstractButton):
        found += [w for w in section.findChildren(kind) if w not in found]
    controls = [w for w in found
                if w.isVisible() and w.visibleRegion().boundingRect().height() >= w.height() // 2]
    controls.sort(key=lambda w: (w.mapTo(section, w.rect().topLeft()).y(),
                                 w.mapTo(section, w.rect().topLeft()).x()))
    step = max(1, len(controls) // limit)
    capture_ui.point_at(capture, *controls[::step][:limit], dwell=dwell)


def _read_popup(capture, dialog, dwell=1.2):
    """Host: the pointer moves into the popup and down its list."""
    from PySide6.QtWidgets import QAbstractItemView
    views = [v for v in dialog.findChildren(QAbstractItemView) if v.isVisible()]
    h = capture_ui._h()
    if not views:
        capture_ui.point_at(capture, dialog, dwell=dwell)
        return
    view = views[0]
    model = view.model()
    for row in range(min(model.rowCount(), 4)):
        rect = view.visualRect(model.index(row, 0))
        if rect.isValid() and view.viewport().rect().contains(rect.center()):
            h.hover(view.viewport(), rect.center())
            h.pause(capture, dwell)
