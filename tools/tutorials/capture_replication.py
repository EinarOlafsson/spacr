"""Record a bounded synthetic Replication example without changing spaCR."""
from pathlib import Path
import time

from capture_acceptance import assess_pipeline
from replication_demo import prepare, digest, verify_files, verify_bars
from stage_lesson import read


def record_replication(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QMessageBox, QComboBox, QDialogButtonBox

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
    for block, _, _ in screen._console._pipeline_console_blocks():
        block.setFocus()
        QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    screen._console.jump_to_the_end()
    settle(.3)
    capture('12_actual_completion')
    if (not proof['pipeline']['accepted'] or not proof.get('output_checks') or
            not proof['source_unchanged'] or not proof['private_database_unchanged'] or
            not proof['figures_visible']):
        raise ValueError('The real Replication output, visible figures or preservation checks failed')
    def bar_data(figure):
        from matplotlib.container import BarContainer
        if len(figure.axes) != 1:
            raise ValueError('Expected one displayed distribution axes')
        axis = figure.axes[0]
        groups = [label.get_text() for label in axis.get_xticklabels()]
        series = [dict(bucket=container.get_label().replace('>', 'gt'),
                       rectangles=[dict(height=float(p.get_height()), bottom=float(p.get_y()))
                                   for p in container.patches])
                  for container in axis.containers if isinstance(container, BarContainer)]
        return groups, series

    csv_hashes = {str(path): digest(path) for path in output.glob('*.csv')}
    proof['figure_checks'] = []
    for index, pixmap in enumerate(queue.all_pixmaps()):
        if not pixmap.save(str(captures / f'figure_{index:02}.png'), 'PNG'):
            raise ValueError('Could not preserve an actual output figure')
        queue.show_index(index)
        settle(.3)
        capture(f'13_figure_{index:02}')
        figure = queue.figure_for(index)
        groups, series = bar_data(figure)
        write_json(captures / f'figure_data_{index:02}.json', dict(groups=groups, series=series))
        table = 'well_distribution.csv' if index == 0 else 'condition_summary.csv'
        checks = verify_bars(groups, series, manifest['expected'][table][1])
        errors, accepted = [], []

        def move_legend():
            from spacr.qt.widgets.figure_settings import FigureSettingsDialog
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, FigureSettingsDialog):
                    raise ValueError('The real Figure settings dialog did not open')
                watchdog = QTimer(dialog)
                watchdog.setSingleShot(True)
                watchdog.timeout.connect(dialog.reject)
                watchdog.start(15000)
                dialog.resize(1000, 1500)
                dialog.move(window.geometry().center() - dialog.rect().center())
                tabs = dialog.tabs
                axis_title = figure.axes[0].get_title()[:18]
                matches = [i for i in range(tabs.count()) if tabs.tabText(i) == axis_title]
                if len(matches) != 1:
                    raise ValueError('The figure axes tab is ambiguous')
                QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                                 pos=tabs.tabBar().tabRect(matches[0]).center())
                settle(.2)
                page = tabs.currentWidget()
                choices = [box for box in page.findChildren(QComboBox)
                           if box.findText('best') >= 0 and box.findText('upper right') >= 0]
                if len(choices) != 1:
                    raise ValueError('No unique native Legend position control')
                choice = choices[0]
                page.ensureWidgetVisible(choice)
                choice.setFocus()
                QTest.keyClick(choice, Qt.Key_Home)
                QTest.keyClick(choice, Qt.Key_Down)
                QTest.keyClick(choice, Qt.Key_Home)
                QTest.keyClick(choice, Qt.Key_Tab)
                settle(.5)
                if choice.currentText() != 'best':
                    raise ValueError('The native legend choice did not take effect')
                capture(f'14_legend_settings_{index:02}')
                dialog.accepted.connect(lambda: accepted.append(True))
                QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()

        QTimer.singleShot(350, move_legend)
        QTest.mouseClick(queue._fig_settings_btn, Qt.LeftButton)
        settle(.6)
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'The figure dialog was not accepted')
        after = bar_data(figure)
        if after != (groups, series):
            raise ValueError('Moving the legend changed plotted values or bar geometry')
        verify_bars(*after, manifest['expected'][table][1])
        legend = figure.axes[0].get_legend()
        bounds = legend.get_window_extent(figure.canvas.get_renderer())
        canvas = figure.bbox
        if not (canvas.x0 <= bounds.x0 <= bounds.x1 <= canvas.x1 and
                canvas.y0 <= bounds.y0 <= bounds.y1 <= canvas.y1):
            raise ValueError('The actual legend still leaves the figure canvas')
        capture(f'15_readable_legend_{index:02}')
        proof['figure_checks'].append(dict(index=index, groups=groups, series=series,
            geometry_fields_checked=checks, legend_inside_canvas=True,
            legend_bounds=list(bounds.bounds), canvas_bounds=list(canvas.bounds),
            data_unchanged_after_native_legend_control=True))
    if any(digest(path) != value for path, value in csv_hashes.items()):
        raise ValueError('Changing figure style altered an output data file')
    proof['csv_data_unchanged_after_styling'] = True
    proof['accepted'] = True
    proof['statistical_inference_validated'] = False
    proof['image_or_biological_validation'] = False
    write_json(captures / 'scientific_acceptance.json', proof)
    print('Synthetic Replication:', proof['output_checks'], 'figures', queue.count(), flush=True)
