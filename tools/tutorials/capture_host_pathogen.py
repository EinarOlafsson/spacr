"""Record the current Host–Pathogen example through its native controls."""
from __future__ import annotations

import hashlib
from pathlib import Path
import time


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
                parent.ensureWidgetVisible(widget)
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
    for index in range(tabs.count()):
        candidates = [tile for tile in tiles if tabs.widget(index).isAncestorOf(tile)]
        if candidates:
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                             pos=tabs.tabBar().tabRect(index).center())
            settle(.4)
            capture('00_home')
            click(max(candidates, key=lambda tile: tile.width() * tile.height()))
            break
    else:
        raise RuntimeError('The Toxoplasma Home tile is absent')
    organism = window._screens['toxoplasma']
    tile = next(tile for tile in organism._tiles
                if tile.property('organismModuleKey') == 'host_pathogen')
    organism._module_scroll.ensureWidgetVisible(tile)
    settle(.4)
    click(tile)
    screen = window._screens['host_pathogen']
    if not screen.isVisible():
        raise RuntimeError('The actual organism tile did not open Host–Pathogen')
    button = screen._assay_example_button
    screen._settings_scroll.ensureWidgetVisible(button)
    settle(.4)
    capture('01_test_data')
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
        bar._input.setFocus()
        QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(bar._input, key)
        settle(.4)
        if key not in bar.visible_keys():
            raise RuntimeError('The settings search did not expose ' + key)
        field = screen._settings_model._widgets[key]
        screen._settings_scroll.ensureWidgetVisible(field)
        settle(.2)
        capture(visual)
    bar.set_query(previous[0])
    bar.set_level(previous[1])
    bar.set_modified_only(previous[2])
    if screen._settings_model.collect() != settings:
        raise RuntimeError('The settings walkthrough changed the example configuration')
    wait_for(lambda: getattr(screen, '_registry_preview', None) is not None,
             'The live preview control did not become available')
    host = screen._registry_preview
    click(host.toggle)
    wait_for(host.panel_is_built, 'The live preview did not open')
    panel = host.panel
    click(panel._run_btn)
    wait_for(lambda: panel._jobs.active_jobs() == 0, 'The real preview did not finish')
    if panel._result is None or panel._table.rowCount() == 0:
        raise RuntimeError('The preview did not produce vacuole measurements')
    item = panel._table.item(0, 0)
    panel._table.scrollToItem(item)
    QTest.mouseClick(panel._table.viewport(), Qt.LeftButton,
                     pos=panel._table.visualItemRect(item).center())
    settle(.5)
    capture('07_preview')
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
    worker = screen._worker
    if worker is None:
        raise RuntimeError('The Run control did not start Host–Pathogen')
    worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
    worker.error.connect(lambda message: outcome['errors'].append(str(message)))
    capture('08_run')
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
    screen._console.jump_to_the_end()
    settle(.5)
    capture('09_vacuoles')
    capture('10_summaries')
    capture('11_denominators')
    write_json(captures / 'scientific_acceptance.json', {
        'accepted': True, 'synthetic': False, 'actual_gui_run': True,
        'home_and_organism_tiles_clicked': True, 'test_data_button_clicked': True,
        'individual_parasite_counts_available': False, 'database_unchanged': True,
        'database_sha256': before, 'outputs': {
            path.name: {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'rows': len(tables[path.stem]) if path.suffix == '.csv' else None}
            for path in output.iterdir() if path.is_file()},
        'published': False})
