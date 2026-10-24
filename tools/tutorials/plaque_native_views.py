"""Inspect actual plaque preview filters and read-only saved tables through Qt."""
import hashlib
from pathlib import Path
import time

import numpy as np
from PIL import Image

import capture_ui
from capture_ui_drv_a import type_text
from capture_ui_drv_c import Scenes
from plaque_demo import require_preserved, verify_filtered, verify_tables
from replication_demo import digest


def finish_views(app, window, screen, panel, database, manifest, stage, captures,
                 capture, settle, write_json, timeout, scenes=None):
    from PySide6.QtCore import Qt, QPoint
    from PySide6.QtTest import QTest

    # Host: each scene clip starts BEFORE its actions (shared with the caller).
    S = scenes if scenes is not None else Scenes(capture, settle)

    def fill(widget, value):
        if capture_ui.live(capture):
            type_text(capture, widget, value)
        else:
            widget.setFocus(); widget.selectAll()
            QTest.keyClicks(widget, str(value)); QTest.keyClick(widget, Qt.Key_Tab)
        settle(.3)

    S.begin('16_readable_preview')

    # Free canvas space with the actual visible fold controls, not a fake
    # enlarged output panel. These operations do not touch input or results.
    for folder in (screen._console_folder, screen._usage_card.folder):
        if not folder.shut:
            QTest.mouseClick(folder.heading, Qt.LeftButton); settle(.3)
        if not folder.shut:
            raise ValueError('The actual Console/System heading did not fold')
    expected = Path(manifest['root'])/'plate1_A01_control_1.tif'
    if panel.current_path() is None or Path(panel.current_path()).resolve() != expected.resolve():
        raise ValueError('The preview shows a different input image')
    result = panel._plaque_result
    if result is None or not result.get('count'):
        raise ValueError('The preview has no plaque result to inspect')
    views = [panel._view_selector.itemText(i) for i in range(panel._view_selector.count())]
    if views != ['Overlay', 'Masks', 'Flows', 'Cell probability']:
        raise ValueError('The preview result views changed: ' + str(views))
    proof = dict(image=str(expected), synthetic=True, model_accuracy_claim=False, batch_filters_claimed=False,
                 plaques=int(result['count']), views=views, status=panel._status.text())
    S.end('16_readable_preview', panel)

    # The four result views (lesson scene 8): Masks, Flows and Cell probability beside the Overlay.
    S.begin('18_actual_filter_changed')
    shown = []
    for name in ('Masks', 'Flows', 'Cell probability', 'Overlay'):
        capture_ui.select_combo(capture, panel._view_selector, panel._view_selector.findText(name))
        settle(1.2)
        if panel._view_selector.currentText() != name:
            raise ValueError('The preview did not switch to its ' + name + ' view')
        shown.append(name)
    proof['views_shown'] = shown
    S.end('18_actual_filter_changed', panel._view_selector)

    # The detection settings (lesson scene 9): Diameter, Flow threshold and Cell probability. Read only.
    S.begin('19_actual_filter_restored')
    from capture_plaque import open_preview_settings
    before_settings = {k: v for k, v in panel.current_settings().items() if k in ('diameter', 'flow_threshold', 'CP_prob')}
    dialog = open_preview_settings(capture, settle, panel); settle(.4)
    capture_ui.dialog_geometry(capture, dialog, (1100, 800), window.mapToGlobal(QPoint(60, 300)))
    settle(.3)
    if dialog.tabs.tabText(dialog.tabs.currentIndex()) != 'Plaque detection':
        raise ValueError('The preview settings did not open on the Plaque detection tab')
    for field in (panel._diameter, panel._flow, panel._cellprob):
        if not field.isVisible():
            raise ValueError('A Plaque detection setting is not visible')
        capture_ui.point_at(capture, field, dwell=0.8)
    after_settings = {k: v for k, v in panel.current_settings().items() if k in before_settings}
    if after_settings != before_settings:
        raise ValueError('Looking at the preview settings changed them')
    proof.update(detection_settings=after_settings, segmentation_rerun=False, actual_status=panel._status.text())
    S.end('19_actual_filter_restored', dialog)
    S.begin('20_actual_database_help_route')
    capture_ui.close_dialog(capture, dialog); settle(.3)
    # Plaque's reuse path writes tables, not a Matplotlib figure. Open those
    # exact outputs via the ordinary Help route, never inject a result grid.
    old_hash = digest(database)
    help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
    menu = help_action.menu()
    action = next(a for a in menu.actions() if a.text().replace('&', '').lower() == 'database browser')
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(help_action).center())
    settle(.2); S.end('20_actual_database_help_route', menu)
    S.begin('21_actual_summary_table')
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
    # The table list is filled after the open job settles; under load (live host) it can still be
    # empty at that moment. Wait for the tables, bounded, then check them.
    until = time.monotonic() + 60
    while not browser.tables() and time.monotonic() < until:
        settle(.2)
    wait_jobs()
    if Path(browser.database_path()) != database or not {'summary', 'stats', 'details'} <= set(browser.tables()) or set(browser.tables()) - {'summary', 'stats', 'details', 'per_image', 'per_plaque'}:
        raise ValueError('The actual browser opened different output tables '
                         f'(path={browser.database_path()!r} wanted {str(database)!r}, tables={sorted(browser.tables())})')
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
        S.end(f'{index:02}_actual_{name}_table', browser._table_list)
        if index < 23:
            S.begin(f'{index + 1:02}_actual_{("stats", "details")[index - 21]}_table')
    proof['actual_visible_table_checks'] = verify_tables(tables, manifest['references'])
    proof['output_database_unchanged'] = digest(database) == old_hash
    if not proof['output_database_unchanged']:
        raise ValueError('Inspecting the saved result changed its bytes')
    write_json(captures/'actual_browser_tables.json', tables)
    require_preserved(manifest['files'])
    return proof
