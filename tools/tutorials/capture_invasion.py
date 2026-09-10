"""Record the actual Invasion UI on a new, explicitly synthetic input copy."""
from pathlib import Path
import time

from capture_acceptance import assess_pipeline
from invasion_demo import prepare, check_preserved, verify_output, verify_figure, verify_histograms, digest
from native_figure_legend import fit_legend
from stage_lesson import read


def record_invasion(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from matplotlib.container import BarContainer

    manifest = prepare(stage)
    root = Path(manifest['database']).parent.parent
    proof = dict(accepted=False, synthetic=True, biological_effect_claim=False,
        actual_gui_run=False, app_source_modified=False, manifest=manifest, published=False)
    write_json(captures/'scientific_acceptance.json', proof)
    settings = read(stage.parent/'catalog/26_invasion_settings.json')
    settings['stain_baseline_wells'] = settings.pop('control_wells')
    settings['src'] = str(root)
    # The actual GUI field has six decimals; record its representable value,
    # not an unrepresentable old script constant or a loosened readback check.
    settings['bimodality_cutoff'] = .555556
    for key, value in settings.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('No current Invasion setting: '+key)
        settle(.03)
    before = screen._settings_model.collect()
    write_json(captures/'configured_settings.json', before)
    mismatch = {key: dict(requested=value, actual=before.get(key))
                for key, value in settings.items() if before.get(key) != value}
    if mismatch:
        write_json(captures/'settings_mismatch.json', mismatch)
        raise ValueError('The native Invasion settings differ: '+str(mismatch))
    bar = screen._settings_search
    old_query, old_level, old_modified = bar.query(), bar.level(), bar.modified_only()
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    for index, key in enumerate(('src', 'outside_channel', 'total_channel',
        'intensity_statistic', 'stain_baseline_wells', 'control_quantile',
        'threshold_sensitivity', 'min_total_intensity', 'extracellular_class',
        'pathogen_plate_metadata', 'save'), 2):
        bar._input.setFocus()
        QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(bar._input, key)
        settle(.35)
        if key not in bar.visible_keys():
            raise ValueError('Settings search did not expose '+key)
        field = screen._settings_model._widgets[key]
        screen._settings_scroll.ensureWidgetVisible(field)
        settle(.2)
        if not field.isVisible():
            raise ValueError('The requested setting remains hidden: '+key)
        capture(f'{index:02}_setting_{key}')
    bar.set_query(old_query)
    bar.set_level(old_level)
    bar.set_modified_only(old_modified)
    if screen._settings_model.collect() != before:
        raise ValueError('The display tour changed analysis settings')
    if screen._ai_switch.isChecked():
        QTest.mouseClick(screen._ai_switch, Qt.LeftButton)
    if screen._ai_switch.isChecked():
        raise ValueError('No external AI request is part of this example')
    outcome = dict(finished=False, ok=False, errors=[])
    QTest.mouseClick(screen._btn_run, Qt.LeftButton)
    worker = screen._worker
    if worker is None:
        raise ValueError('The actual Run control did not launch a worker')
    worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
    worker.error.connect(lambda message: outcome['errors'].append(str(message)))
    proof['actual_gui_run'] = True
    settle(.3)
    capture('13_actual_run')
    deadline = time.monotonic()+timeout
    while not outcome['finished'] or screen._worker_thread_is_running():
        if time.monotonic() > deadline:
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
            settle(3)
            raise TimeoutError('The bounded synthetic Invasion run did not finish')
        settle(.1)
    settle(1)
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    write_json(captures/'batch_outcome.json', outcome)
    write_json(captures/'batch_console.json', blocks)
    queue = screen._figure_queue
    proof['pipeline'] = assess_pipeline(outcome, blocks, queue.count())
    output = root/'results/analyze_invasion'
    if output.is_dir():
        proof['output_checks'] = verify_output(output, manifest)
    check_preserved(manifest)
    proof.update(source_unchanged=True, private_database_unchanged=True,
        figures_visible=screen._figures_card.isVisible())
    write_json(captures/'scientific_acceptance.json', proof)
    for block, _, _ in screen._console._pipeline_console_blocks():
        block.setFocus()
        QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    screen._console.jump_to_the_end()
    settle(.3)
    capture('14_actual_completion')
    if not proof['pipeline']['accepted'] or not proof.get('output_checks') or not proof['figures_visible']:
        raise ValueError('The actual outputs, console or visible figures failed acceptance')
    proof['figure_checks'] = []
    csv_hashes = {str(path): digest(path) for path in output.glob('*.csv')}
    for index, pixmap in enumerate(queue.all_pixmaps()):
        if not pixmap.save(str(captures/f'figure_{index:02}.png'), 'PNG'):
            raise ValueError('Could not preserve the actual figure')
        queue.show_index(index)
        settle(.4)
        capture(f'15_figure_{index:02}')
        figure = queue.figure_for(index)
        if index < 2:
            axis = figure.axes[0]
            groups = [label.get_text() for label in axis.get_xticklabels()]
            series = [dict(bucket=item.get_label(), rectangles=[
                dict(height=float(p.get_height()), bottom=float(p.get_y())) for p in item.patches])
                for item in axis.containers if isinstance(item, BarContainer)]
            write_json(captures/f'figure_data_{index:02}.json', dict(groups=groups, series=series))
            filename = 'well_invasion.csv' if index == 0 else 'condition_summary.csv'
            checked = verify_figure(groups, series, manifest['expected']['tables'][filename][1])
            denominators = [text.get_text() for text in axis.texts if text.get_text().startswith('n=')]
            if denominators != [f"n={manifest['expected']['tables'][filename][1][key]['n_total']}" for key in groups]:
                raise ValueError('Actual bar denominator labels differ from counted objects')
            styled = fit_legend(app, window, queue, figure, capture, settle, f'16_legend_{index:02}')
            after = [dict(bucket=item.get_label(), rectangles=[
                dict(height=float(p.get_height()), bottom=float(p.get_y())) for p in item.patches])
                for item in axis.containers if isinstance(item, BarContainer)]
            if after != series or [text.get_text() for text in axis.get_xticklabels()] != groups:
                raise ValueError('Changing legend position changed actual plotted data')
            capture(f'17_readable_figure_{index:02}')
            proof['figure_checks'].append(dict(index=index, geometry_fields_checked=checked,
                groups=groups, series=series, denominators=denominators, styling=styled))
        else:
            panels = []
            for axis in figure.axes:
                title = axis.get_title().splitlines()
                panels.append(dict(well=title[0], denominator=int(title[1].split()[0].split('=')[1]),
                    bars=[dict(left=float(p.get_x()), width=float(p.get_width()),
                               height=float(p.get_height()), bottom=float(p.get_y())) for p in axis.patches],
                    thresholds=[list(map(float, line.get_xdata())) for line in axis.lines]))
            write_json(captures/'histogram_data.json', panels)
            proof['histogram_checks'] = verify_histograms(panels,
                manifest['expected']['tables']['parasite_calls.csv'][1], manifest['expected']['threshold'])
    if any(digest(path) != value for path, value in csv_hashes.items()):
        raise ValueError('Styling a figure changed an output CSV')
    proof['csv_unchanged_after_native_styling'] = True
    proof.update(accepted=True, statistical_inference_validated=False, image_or_biological_validation=False)
    write_json(captures/'scientific_acceptance.json', proof)
    print('Synthetic Invasion:', proof['output_checks'], 'figures', queue.count(), flush=True)
