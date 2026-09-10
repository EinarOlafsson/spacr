"""Record a bounded synthetic Replication example without changing spaCR."""
from pathlib import Path
import time

from capture_acceptance import assess_pipeline
from replication_demo import prepare, digest, verify_files
from stage_lesson import read


def record_replication(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox

    manifest = prepare(stage)
    database = Path(manifest['database'])
    root = database.parent.parent
    proof = dict(accepted=False, synthetic=True, biological_effect_claim=False,
        new_download_button=False, actual_gui_run=False, app_source_modified=False,
        manifest=manifest, published=False)
    write_json(captures / 'scientific_acceptance.json', proof)
    settings = read(stage.parent / 'catalog/27_replication_settings.json')
    settings['src'] = str(root)
    # The private copy starts with only input tables, never old result files.
    if (root / 'results').exists():
        raise ValueError('A fresh tutorial run must not inherit old outputs')
    for key, value in settings.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('No real Replication setting for ' + key)
        settle(.05)
    collected = screen._settings_model.collect()
    if any(collected.get(key) != value for key, value in settings.items()):
        raise ValueError('The actual settings do not match the synthetic demonstration')
    write_json(captures / 'configured_settings.json', collected)
    capture('02_synthetic_inputs')
    bar = screen._settings_search
    old_query, old_level, old_modified = bar.query(), bar.level(), bar.modified_only()
    before = screen._settings_model.collect()
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    for index, key in enumerate(('src', 'vacuole_key', 'require_host_cell',
            'min_parasite_area', 'max_parasites_per_vacuole', 'non_power_of_two_warn',
            'pathogen_plate_metadata', 'save'), 3):
        bar._input.setFocus()
        QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(bar._input, key)
        settle(.4)
        if key not in bar.visible_keys():
            raise ValueError('The real settings search did not expose ' + key)
        field = screen._settings_model._widgets[key]
        screen._settings_scroll.ensureWidgetVisible(field)
        settle(.2)
        if not field.isVisible():
            raise ValueError('The intended setting is hidden: ' + key)
        capture(f'{index:02d}_setting_{key}')
    bar.set_query(old_query)
    bar.set_level(old_level)
    bar.set_modified_only(old_modified)
    settle(.3)
    if screen._settings_model.collect() != before:
        raise ValueError('A display-only settings tour changed the analysis')
    if screen._ai_switch.isChecked():
        QTest.mouseClick(screen._ai_switch, Qt.LeftButton)
    if screen._ai_switch.isChecked():
        raise ValueError('The demonstration must not submit external AI requests')
    outcome = dict(finished=False, ok=False, errors=[])

    def reject_prompt():
        for dialog in app.topLevelWidgets():
            if isinstance(dialog, QMessageBox) and dialog.isVisible():
                write_json(captures / 'unexpected_prompt.json', dict(title=dialog.windowTitle(), text=dialog.text()))
                dialog.reject()

    QTimer.singleShot(800, reject_prompt)
    QTest.mouseClick(screen._btn_run, Qt.LeftButton)
    worker = screen._worker
    if worker is None:
        raise ValueError('The native Run button did not start a worker')
    worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
    worker.error.connect(lambda message: outcome['errors'].append(str(message)))
    proof['actual_gui_run'] = True
    settle(.3)
    capture('11_actual_run')
    deadline = time.monotonic() + timeout
    while not outcome['finished'] or screen._worker_thread_is_running():
        if time.monotonic() > deadline:
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
            settle(3)
            raise TimeoutError('The bounded synthetic Replication run did not finish')
        settle(.1)
    settle(1)
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    write_json(captures / 'batch_outcome.json', outcome)
    write_json(captures / 'batch_console.json', blocks)
    queue = screen._figure_queue
    proof['pipeline'] = assess_pipeline(outcome, blocks, queue.count())
    output = root / 'results/analyze_replication'
    if output.is_dir():
        proof['output_checks'] = verify_files(output, manifest['expected'])
    proof['source_unchanged'] = digest(manifest['source']) == manifest['source_sha256']
    proof['private_database_unchanged'] = digest(database) == manifest['database_sha256']
    proof['figures_visible'] = screen._figures_card.isVisible()
    write_json(captures / 'scientific_acceptance.json', proof)
    screen._console.jump_to_the_end()
    settle(.3)
    capture('12_actual_completion')
    if (not proof['pipeline']['accepted'] or not proof.get('output_checks') or
            not proof['source_unchanged'] or not proof['private_database_unchanged'] or
            not proof['figures_visible']):
        raise ValueError('The real Replication output, visible figures or preservation checks failed')
    for index, pixmap in enumerate(queue.all_pixmaps()):
        if not pixmap.save(str(captures / f'figure_{index:02}.png'), 'PNG'):
            raise ValueError('Could not preserve an actual output figure')
        queue.show_index(index)
        settle(.3)
        capture(f'13_figure_{index:02}')
    proof['accepted'] = True
    proof['statistical_inference_validated'] = False
    proof['image_or_biological_validation'] = False
    write_json(captures / 'scientific_acceptance.json', proof)
    print('Synthetic Replication:', proof['output_checks'], 'figures', queue.count(), flush=True)
