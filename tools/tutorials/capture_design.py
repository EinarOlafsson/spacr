"""Record genuine plate planning and export, not invented acquired results."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
import tempfile
import time


def record_design(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QLineEdit, QDialogButtonBox

    def fill(widget, value):
        widget.setFocus()
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, str(value))
        QTest.keyClick(widget, Qt.Key_Tab)
        settle(0.3)

    def choose(box, value, *, data=False):
        index = box.findData(value) if data else box.findText(value)
        if index < 0:
            raise RuntimeError(f'The actual selector has no {value!r}')
        box.setFocus()
        QTest.keyClick(box, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(box, Qt.Key_Down)
        QTest.keyClick(box, Qt.Key_Tab)
        settle(0.4)
        if (box.currentData() if data else box.currentText()) != value:
            raise RuntimeError('The actual keyboard selection did not take effect')

    def assignments():
        return {well.property('wellName'): {
                    'condition': well.property('wellCondition'),
                    'role': well.property('spacrWellRole')}
                for well in screen._well_labels if well.property('wellCondition')}

    def findings():
        return [label.text() for label in screen._findings_labels]

    def edit_count(name, value):
        table = screen._table
        rows = [i for i in range(table.rowCount()) if table.item(i, 0).text() == name]
        if len(rows) != 1:
            raise RuntimeError(f'Expected one actual condition named {name}')
        item = table.item(rows[0], 1)
        table.scrollToItem(item)
        QTest.mouseClick(table.viewport(), Qt.LeftButton, pos=table.visualItemRect(item).center())
        QTest.keyClick(table, Qt.Key_F2)
        settle(0.2)
        editors = [editor for editor in table.findChildren(QLineEdit) if editor.isVisible()]
        if len(editors) != 1:
            raise RuntimeError('F2 did not open the real replicate editor')
        fill(editors[0], value)
        if next(c.replicates for c in screen.conditions() if c.name == name) != value:
            raise RuntimeError('The actual replicate edit did not update the plan')

    # Use the header's own fit-to-contents gesture, not a hidden resize API.
    header = screen._table.horizontalHeader()
    for section in (1, 2):
        before = header.sectionSize(section)
        edge = header.sectionViewportPosition(section) + before - 1
        QTest.mouseDClick(header.viewport(), Qt.LeftButton,
                         pos=QPoint(edge, header.height() // 2))
        settle(0.3)
        print(f'Actual header fit, section {section}: {before} -> {header.sectionSize(section)}', flush=True)
    fill(screen._plate_id, 'tutorial_plan')
    if screen.design().wells_requested != 24:
        raise RuntimeError('The actual starting plan differs from the expected 6/6/12 example')
    capture('02_named_draft')
    choose(screen._format, 6, data=True)
    if assignments() or not any('only 6 usable' in text for text in findings()):
        raise RuntimeError('The actual capacity warning did not reject the overfull plan')
    capture('03_capacity_warning')
    choose(screen._format, 96, data=True)
    if len(assignments()) != 24:
        raise RuntimeError('Returning to 96 wells did not restore all planned replicates')
    capture('04_capacity_restored')
    edit_count('negative', 1)
    if not any('one well' in text for text in findings()):
        raise RuntimeError('The actual single-replicate warning did not appear')
    capture('05_single_replicate_warning')
    edit_count('negative', 6)
    if any('has one well' in text for text in findings()):
        raise RuntimeError('Restoring replicates did not clear its warning')
    choose(screen._layout_box, 'block')
    if not any('block layout' in text for text in findings()):
        raise RuntimeError('The actual block-position warning did not appear')
    capture('06_block_layout_warning')
    choose(screen._layout_box, 'random')
    choose(screen._edge, 'leave_empty', data=True)
    fill(screen._seed.lineEdit(), 42)
    original = assignments()
    if len(original) != 24 or screen.design().wells_available != 60:
        raise RuntimeError('The actual edge policy did not reserve the outer ring')
    for well in screen._well_labels:
        if well.property('wellCondition') and (well.row in {1, 8} or well.column in {1, 12}):
            raise RuntimeError('The plan assigns a well on the supposedly unused outer ring')
    capture('07_random_interior_seed42')
    fill(screen._seed.lineEdit(), 43)
    if assignments() == original:
        raise RuntimeError('A different seed did not change this actual layout')
    capture('08_seed43')
    fill(screen._seed.lineEdit(), 42)
    if assignments() != original:
        raise RuntimeError('The original seed did not reproduce the same displayed assignments')
    capture('09_seed42_restored')

    wells = {(well.row, well.column): well for well in screen._well_labels}
    start, end = wells[(2, 2)], wells[(3, 4)]
    QTest.mousePress(start, Qt.LeftButton, pos=start.rect().center())
    # The pressed well retains the mouse grab while the pointer crosses peers.
    QTest.mouseMove(start, start.mapFromGlobal(end.mapToGlobal(end.rect().center())), delay=100)
    QTest.mouseRelease(start, Qt.LeftButton, pos=start.mapFromGlobal(end.mapToGlobal(end.rect().center())))
    settle(0.5)
    expected = {(row, col) for row in (2, 3) for col in (2, 3, 4)}
    if screen.selected_wells() != expected or assignments() != original:
        raise RuntimeError('Actual well selection must select exactly the rectangle without assigning treatments')
    capture('10_selected_wells_not_reassigned')

    destination_root = stage / 'design_runs'
    destination_root.mkdir(exist_ok=True)
    from pathlib import Path
    destination = Path(tempfile.mkdtemp(prefix='plan-', dir=destination_root))
    errors, accepted = [], []
    def pick():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise RuntimeError('Export did not open its actual folder picker')
            timer = QTimer(dialog)
            timer.setSingleShot(True)
            timer.timeout.connect(dialog.reject)
            timer.start(12000)
            dialog.accepted.connect(lambda: accepted.append(True))
            fill(dialog.findChild(QLineEdit, 'fileNameEdit'), destination)
            capture('11_export_folder')
            QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open), Qt.LeftButton)
        except Exception as exc:
            errors.append(str(exc))
            if dialog is not None:
                dialog.reject()
    QTimer.singleShot(400, pick)
    QTest.mouseClick(screen._export, Qt.LeftButton)
    if errors or not accepted:
        raise RuntimeError('; '.join(errors) or 'Export was cancelled')
    deadline = time.monotonic() + timeout
    while screen.is_busy() or screen.active_jobs():
        if time.monotonic() >= deadline:
            raise TimeoutError('The real export did not finish')
        settle(0.1)
    settle(0.6)
    files = {name: destination / name for name in
             ('plate_map.csv', 'plate_map.json', 'plate_map_settings.json')}
    if not all(path.is_file() for path in files.values()) or not screen.status_text().startswith('Wrote '):
        raise RuntimeError('The actual export did not write all three expected files')
    with files['plate_map.csv'].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    exported = {row['well']: {'condition': row['condition'], 'role': row['role']} for row in rows}
    if len(rows) != 24 or exported != original or {row['plateID'] for row in rows} != {'tutorial_plan'}:
        raise RuntimeError('Exported well assignments differ from the displayed plan')
    design = json.loads(files['plate_map.json'].read_text())
    fragment = json.loads(files['plate_map_settings.json'].read_text())
    if design['seed'] != 42 or design['wells_available'] != 60 or fragment['expressible']:
        raise RuntimeError('The saved design or non-expressible random-layout warning changed')
    capture('12_real_export_complete')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'planned_not_acquired': True, 'design': asdict(screen.design()),
        'displayed_assignments': original, 'exported_well_count': len(rows),
        'selected_wells': screen.selected_well_names(), 'selection_reassigned_conditions': False,
        'seed42_43_42_restores_assignments': True, 'capacity_warning_then_recovery': True,
        'single_replicate_warning_then_recovery': True, 'block_warning_shown': True,
        'settings_fragment': fragment, 'destination': str(destination),
        'files': {name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'bytes': path.stat().st_size} for name, path in files.items()},
        'measured_treatment_effects': False, 'published': False,
    })
    print('Accepted real design/export: 24 planned wells, 60 usable, seed reproduced, exact CSV', flush=True)
