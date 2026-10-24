"""Record the current Host–Pathogen example through its native controls."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import capture_ui
from capture_ui_drv_a import Rec, search_typed, table_into_view
from capture_ui_drv_e import guarded
from capture_ui_drv_f import sweep_over


@guarded
def record_host_pathogen(app, window, stage, captures, capture, settle, write_json, timeout):
    """Download the supplied microscopy sample, preview it and run the analysis."""
    import pandas as pd
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QScrollArea

    deadline = time.monotonic() + timeout

    def wait_for(predicate, label):
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError(label)
            settle(.1)
        settle(.4)

    def expose(widget):
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                capture_ui.scroll_to(capture, parent, widget)
            parent = parent.parentWidget()
        settle(.2)

    def click(widget):
        expose(widget)
        visible = widget.visibleRegion()
        if not widget.isEnabled() or visible.isEmpty():
            raise RuntimeError('A required Host–Pathogen control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=visible.boundingRect().center())
        settle(.3)

    tabs = window._startup._tabs
    tiles = [button for button in window._startup.findChildren(QAbstractButton)
             if button.property('moduleAppKey') == 'toxoplasma']
    # Host: 00_home opens on the Home tabs and records the pointer choosing the
    # Toxoplasma tile; 01_test_data records the organism page, the Host-Pathogen
    # tile and the scroll to Load test data. Xvfb: the stills after the same steps.
    home_clip = None
    for index in range(tabs.count()):
        candidates = [tile for tile in tiles if tabs.widget(index).isAncestorOf(tile)]
        if candidates:
            home_clip = Rec(capture, '00_home', settle).start()
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                             pos=tabs.tabBar().tabRect(index).center())
            settle(.4)
            toxo_tile = max(candidates, key=lambda tile: tile.width() * tile.height())
            capture_ui.point_at(capture, toxo_tile, dwell=1.2)
            if not capture_ui.live(capture):
                home_clip.stop()                # Xvfb: the still of Home, before the tile is clicked
            click(toxo_tile)
            if capture_ui.live(capture):
                home_clip.stop()
            break
    else:
        raise RuntimeError('The Toxoplasma Home tile is absent')
    test_data_clip = Rec(capture, '01_test_data', settle).start()
    organism = window._screens['toxoplasma']
    tile = next(tile for tile in organism._tiles
                if tile.property('organismModuleKey') == 'host_pathogen')
    capture_ui.scroll_to(capture, organism._module_scroll, tile)
    settle(.4)
    click(tile)
    screen = window._screens['host_pathogen']
    if not screen.isVisible():
        raise RuntimeError('The actual organism tile did not open Host–Pathogen')
    button = screen._assay_example_button
    capture_ui.scroll_to(capture, screen._settings_scroll, button)
    settle(.4)
    capture_ui.point_at(capture, button, dwell=1.5)
    test_data_clip.stop()
    click(button)
    wait_for(lambda: button.isEnabled(), 'The native example download did not finish')
    settings = screen._settings_model.collect()
    root = Path(settings['src'])
    if not root.is_relative_to(stage) or not (root / 'example_manifest.json').is_file():
        raise RuntimeError('The downloaded example is outside the private recording stage')
    if (settings['hp_marker_channels'] != [1] or settings['hp_marker_thresholds'] != {}
            or settings['hp_parasite_table'] or settings['hp_count_column']):
        raise RuntimeError('The downloaded settings do not match the current real example')
    database = root / 'measurements/measurements.db'
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    write_json(captures / 'configured_settings.json', settings)
    bar = screen._settings_search
    previous = (bar.query(), bar.level(), bar.modified_only())
    if bar.modified_only():
        click(bar._modified)
    if bar.level() != 'all':
        click(bar._disclosure)
    for key, visual in [('src', '02_source'), ('hp_vacuole_table', '03_compartments'),
                        ('hp_marker_channels', '04_markers'),
                        ('hp_marker_thresholds', '05_thresholds'),
                        ('hp_parasite_table', '06_counts')]:
        # Host: the clip starts before the pointer clicks the search box.
        with Rec(capture, visual, settle) as scene:
            search_typed(capture, bar, key)
            settle(.4)
            if key not in bar.visible_keys():
                raise RuntimeError('The settings search did not expose ' + key)
            field = screen._settings_model._widgets[key]
            capture_ui.scroll_to(capture, screen._settings_scroll, field)
            settle(.2)
            scene.point(field, dwell=1.5)
    capture_ui.search_settings(capture, bar, previous[0])
    capture_ui.settings_level(capture, bar, previous[1])
    capture_ui.settings_modified_only(capture, bar, previous[2])
    if screen._settings_model.collect() != settings:
        raise RuntimeError('The settings walkthrough changed the example configuration')
    wait_for(lambda: getattr(screen, '_registry_preview', None) is not None,
             'The live preview control did not become available')
    host = screen._registry_preview
    with capture_ui.clip(capture, '07_preview'):
        click(host.toggle)
        wait_for(host.panel_is_built, 'The live preview did not open')
        panel = host.panel
        click(panel._run_btn)
        wait_for(lambda: panel._jobs.active_jobs() == 0, 'The real preview did not finish')
        if panel._result is None or panel._table.rowCount() == 0:
            raise RuntimeError('The preview did not produce vacuole measurements')
        item = panel._table.item(0, 0)
        if capture_ui.live(capture):
            table_into_view(capture, panel._table, 0)
        else:
            panel._table.scrollToItem(item)
        QTest.mouseClick(panel._table.viewport(), Qt.LeftButton,
                         pos=panel._table.visualItemRect(item).center())
        settle(.5)
        capture_ui.point_at(capture, panel._table, dwell=1.5)
    # Host: 08_run starts before Live is switched off and ends once Run has started.
    run_clip = Rec(capture, '08_run', settle).start()
    click(host.toggle)
    if not screen._actions_body.isVisible():
        click(screen._actions_heading)
    if screen._ai_switch.isChecked():
        click(screen._ai_switch)
    outcome = {'finished': False, 'ok': False, 'errors': []}
    expose(screen._btn_run)
    if not screen._btn_run.isEnabled() or screen._btn_run.visibleRegion().isEmpty():
        raise RuntimeError('The Run control is unavailable')
    QTest.mouseClick(screen._btn_run, Qt.LeftButton)
    worker = capture_ui.wait_worker(screen, settle)
    if worker is None:
        raise RuntimeError('The Run control did not start Host–Pathogen')
    worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
    worker.error.connect(lambda message: outcome['errors'].append(str(message)))
    capture_ui.wait(capture, 2.0, settle)
    run_clip.stop()
    wait_for(lambda: outcome['finished'] and not screen._worker_thread_is_running(),
             'The actual Host–Pathogen analysis did not finish')
    write_json(captures / 'run_outcome.json', outcome)
    if not outcome['ok'] or outcome['errors']:
        raise RuntimeError('The actual Host–Pathogen analysis reported failure')
    output = root / 'results/host_pathogen'
    tables = {path.stem: pd.read_csv(path) for path in output.glob('*.csv')}
    if len(tables['vacuoles']) != 97 or len(tables['cells']) != 164:
        raise RuntimeError('The output object counts differ from the supplied real example')
    if not tables['vacuoles']['parasite_count'].isna().all():
        raise RuntimeError('The example has no parasite counts; its output must remain unknown')
    if hashlib.sha256(database.read_bytes()).hexdigest() != before:
        raise RuntimeError('The analysis changed the input measurement database')
    # Host: the pointer takes the console from its end to the written tables.
    with capture_ui.clip(capture, '09_vacuoles'):
        capture_ui.console_to_end(capture, screen._console)
        settle(.5)
        sweep_over(capture, screen._console, dwell=1.2)
    with capture_ui.clip(capture, '10_summaries'):
        sweep_over(capture, screen._console, fractions=(0.3, 0.6, 0.9), dwell=1.2)
    with capture_ui.clip(capture, '11_denominators'):
        sweep_over(capture, screen._console, fractions=(0.9, 0.6, 0.3), dwell=1.2)
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'synthetic': False, 'actual_gui_run': True,
        'home_and_organism_tiles_clicked': True, 'test_data_button_clicked': True,
        'individual_parasite_counts_available': False, 'database_unchanged': True,
        'database_sha256': before, 'outputs': {
            path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'rows': len(tables[path.stem]) if path.suffix == '.csv' else None}
            for path in output.iterdir() if path.is_file()},
        'published': False})
