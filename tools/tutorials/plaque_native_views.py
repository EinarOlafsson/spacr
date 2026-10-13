"""Inspect actual plaque preview filters and read-only saved tables through Qt."""
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

from plaque_demo import require_preserved, verify_filtered, verify_tables
from replication_demo import digest


def finish_views(app, window, screen, panel, database, manifest, stage, captures,
                 capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest

    def fill(widget, value):
        widget.setFocus(); widget.selectAll()
        QTest.keyClicks(widget, str(value)); QTest.keyClick(widget, Qt.Key_Tab)
        settle(.3)

    # Free canvas space with the actual visible fold controls, not a fake
    # enlarged output panel. These operations do not touch input or results.
    for folder in (screen._console_folder, screen._usage_card.folder):
        if not folder.shut:
            QTest.mouseClick(folder.heading, Qt.LeftButton); settle(.3)
        if not folder.shut:
            raise ValueError('The actual Console/System heading did not fold')
    expected_image = np.asarray(Image.open(Path(manifest['root'])/'plate1_A01_control_1.tif'))
    if not np.array_equal(np.squeeze(panel._image), expected_image):
        raise ValueError('The preview loaded different input pixels')
    raw = panel._raw_masks['cell'].copy()
    if len(np.unique(raw[raw > 0])) != 4:
        raise ValueError('The current synthetic preview no longer has four objects')
    baseline = panel._masks['cell'].copy()
    proof = dict(input_pixels_checked=int(expected_image.size), synthetic=True,
                 model_accuracy_claim=False, batch_filters_claimed=False,
                 initial=verify_filtered(raw, baseline, 0))
    np.save(captures/'preview_raw.npy', raw)
    panel._view_mode.setFocus()
    QTest.keyClick(panel._view_mode, Qt.Key_Home)
    QTest.keyClick(panel._view_mode, Qt.Key_M)
    QTest.keyClick(panel._view_mode, Qt.Key_Tab)
    settle(.3)
    if panel._view_mode.currentText() != 'Masks':
        raise ValueError('The native preview did not switch to its Masks display mode')
    capture('16_readable_preview')
    panel.open_live_settings(); settle(.4)
    dialog = panel._live_settings_dialog
    dialog.resize(2400, 1280)
    dialog.move(window.mapToGlobal(QPoint(30, 270)))
    settle(.3)
    minimum = panel._compartment_widgets['cell']['min_area']
    if not minimum.isVisible() or minimum.value() != 0:
        raise ValueError('The real minimum-area control is not at the demonstrated baseline')
    before_hash = hashlib.sha256(raw.tobytes()).hexdigest()
    old_worker, old_token = panel._worker, panel._run_token
    _, areas = np.unique(raw[raw > 0], return_counts=True)
    cutoff = int(np.median(areas))+1
    capture('17_actual_filter_settings')
    fill(minimum, cutoff)
    proof['filtered'] = verify_filtered(raw, panel._masks['cell'], cutoff)
    if not 0 < proof['filtered']['objects'] < proof['initial']['objects']:
        raise ValueError('The actual filter has no selective visible effect')
    np.save(captures/'preview_filtered.npy', panel._masks['cell'])
    capture('18_actual_filter_changed')
    fill(minimum, 0)
    proof['restored'] = verify_filtered(raw, panel._masks['cell'], 0)
    if not np.array_equal(panel._masks['cell'], baseline):
        raise ValueError('Resetting the actual filter did not restore the exact result')
    if (panel._worker is not old_worker or panel._run_token != old_token or
            hashlib.sha256(panel._raw_masks['cell'].tobytes()).hexdigest() != before_hash):
        raise ValueError('The live filter reran or changed the underlying segmentation')
    proof.update(raw_mask_sha256=before_hash, segmentation_rerun=False, exact_restoration=True,
                 actual_status=panel._status.text())
    capture('19_actual_filter_restored')
    dialog.close(); settle(.3)
    # Plaque's reuse path writes tables, not a Matplotlib figure. Open those
    # exact outputs via the ordinary Help route, never inject a result grid.
    old_hash = digest(database)
    help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
    menu = help_action.menu()
    action = next(a for a in menu.actions() if a.text().replace('&', '') == 'Database browser')
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(help_action).center())
    settle(.2); capture('20_actual_database_help_route')
    QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(action).center())
    deadline = time.monotonic()+timeout
    while window._screens.get('db_browser') is None and time.monotonic()<deadline:
        settle(.1)
    browser = window._screens.get('db_browser')
    if browser is None or not browser.isVisible():
        raise ValueError('The actual Database Browser did not open')
    fill(browser._path_edit, database)
    QTest.mouseClick(browser._btn_open, Qt.LeftButton)

    def wait_jobs():
        deadline = time.monotonic()+timeout
        while browser.is_busy() or browser.active_jobs() or browser.queued_jobs():
            if time.monotonic()>deadline:
                raise TimeoutError('The actual read-only database query did not finish')
            settle(.1)
        settle(.2)
        if browser.last_error or browser.edit_mode_enabled():
            raise ValueError('The actual output table is not a successful read-only view')

    wait_jobs()
    if Path(browser.database_path()) != database or set(browser.tables()) != {'summary', 'stats', 'details'}:
        raise ValueError('The actual browser opened different output tables')
    tables = {}
    for index, name in enumerate(('summary', 'stats', 'details'), 21):
        items = browser._table_list.findItems(name, Qt.MatchExactly)
        if len(items) != 1:
            raise ValueError('The output table is not uniquely available')
        item = items[0]
        QTest.mouseClick(browser._table_list.viewport(), Qt.LeftButton,
                         pos=browser._table_list.visualItemRect(item).center())
        wait_jobs()
        if browser.current_table() != name or browser.row_count() != len(browser.preview_rows()):
            raise ValueError('The actual table view omitted rows in this small example')
        tables[name] = [dict(zip(browser.preview_columns(), row)) for row in browser.preview_rows()]
        capture(f'{index:02}_actual_{name}_table')
    proof['actual_visible_table_checks'] = verify_tables(tables, manifest['references'])
    proof['output_database_unchanged'] = digest(database) == old_hash
    if not proof['output_database_unchanged']:
        raise ValueError('Inspecting the saved result changed its bytes')
    write_json(captures/'actual_browser_tables.json', tables)
    require_preserved(manifest['files'])
    return proof
