"""Record the actual Invasion UI on a new, explicitly synthetic input copy."""
import capture_ui
from capture_ui_drv_a import search_typed
from capture_ui_drv_c import Scenes
from capture_ui_drv_e import guarded
from pathlib import Path

from capture_acceptance import assess_pipeline, observe_pipeline_run
from invasion_demo import prepare, check_preserved, verify_output, verify_figure, verify_histograms, digest
from native_figure_legend import fit_legend
from stage_lesson import read
from capture_settings import import_capture_settings as import_invasion_settings


def run_invasion_pipeline(screen, scenes, settle, timeout):
    """Keep the real Run and completion scenes across deferred or fast runs."""
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    def after_click():
        settle(.3)
        scenes.end('13_actual_run', screen._console)
        scenes.begin('14_actual_completion')

    scenes.begin('13_actual_run')
    try:
        return observe_pipeline_run(screen,
            lambda: QTest.mouseClick(screen._btn_run, Qt.LeftButton),
            settle, timeout, after_click=after_click)
    except TimeoutError:
        if screen._worker_thread_is_running():
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
            settle(3)
        raise


@guarded
def record_invasion(app, window, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from matplotlib.container import BarContainer

    manifest = prepare(stage)
    root = Path(manifest['database']).parent.parent
    proof = dict(accepted=False, synthetic=True, biological_effect_claim=False,
        actual_gui_run=False, app_source_modified=False, manifest=manifest, published=False)
    write_json(captures/'scientific_acceptance.json', proof)
    catalog = stage.parent/'catalog/26_invasion_settings.json'
    if not catalog.is_file():
        catalog = Path(__file__).resolve().parent/'authoring/catalog/26_invasion_settings.json'
    settings = read(catalog)
    settings['stain_baseline_wells'] = settings.pop('control_wells')
    settings['src'] = str(root)
    # The actual GUI field has six decimals; record its representable value,
    # not an unrepresentable old script constant or a loosened readback check.
    settings['bimodality_cutoff'] = .555556
    proof['settings_import'] = import_invasion_settings(
        app, screen, stage, settings, capture, settle)
    before = screen._settings_model.collect()
    write_json(captures/'configured_settings.json', before)
    mismatch = {key: dict(requested=value, actual=before.get(key))
                for key, value in settings.items() if before.get(key) != value}
    if mismatch:
        write_json(captures/'settings_mismatch.json', mismatch)
        raise ValueError('The native Invasion settings differ: '+str(mismatch))
    S = Scenes(capture, settle)
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
        S.begin(f'{index:02}_setting_{key}')      # host: before the pointer clicks the search box
        search_typed(capture, bar, key)
        settle(.35)
        if key not in bar.visible_keys():
            raise ValueError('Settings search did not expose '+key)
        field = screen._settings_model._widgets[key]
        # A list setting is now drawn by its row's own editor while the
        # registered backing editor stays hidden; the row is what the user sees.
        if field.isHidden() and field.parentWidget() is not None:
            field = field.parentWidget()
        capture_ui.scroll_to(capture, screen._settings_scroll, field)
        settle(.2)
        if not field.isVisible():
            raise ValueError('The requested setting remains hidden: '+key)
        S.end(f'{index:02}_setting_{key}', field)
    capture_ui.search_settings(capture, bar, old_query)
    capture_ui.settings_level(capture, bar, old_level)
    capture_ui.settings_modified_only(capture, bar, old_modified)
    if screen._settings_model.collect() != before:
        raise ValueError('The display tour changed analysis settings')
    if screen._ai_switch.isChecked():
        QTest.mouseClick(screen._ai_switch, Qt.LeftButton)
    if screen._ai_switch.isChecked():
        raise ValueError('No external AI request is part of this example')
    outcome = run_invasion_pipeline(screen, S, settle, timeout)
    proof['actual_gui_run'] = True
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
    capture_ui.console_end_all(capture, screen._console)
    if not capture_ui.live(capture):
        screen._console.jump_to_the_end()
    settle(.3)
    S.end('14_actual_completion', screen._console, screen._figures_card)
    if not proof['pipeline']['accepted'] or not proof.get('output_checks') or not proof['figures_visible']:
        raise ValueError('The actual outputs, console or visible figures failed acceptance')
    proof['figure_checks'] = []
    csv_hashes = {str(path): digest(path) for path in output.glob('*.csv')}
    for index, pixmap in enumerate(queue.all_pixmaps()):
        if not pixmap.save(str(captures/f'figure_{index:02}.png'), 'PNG'):
            raise ValueError('Could not preserve the actual figure')
        S.begin(f'15_figure_{index:02}')
        capture_ui.show_figure(capture, queue, index)
        settle(.4)
        S.end(f'15_figure_{index:02}', queue)
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
            styled = fit_legend(app, window, queue, figure, capture, settle, f'16_legend_{index:02}',
                                scenes=S, next_name=f'17_readable_figure_{index:02}')
            after = [dict(bucket=item.get_label(), rectangles=[
                dict(height=float(p.get_height()), bottom=float(p.get_y())) for p in item.patches])
                for item in axis.containers if isinstance(item, BarContainer)]
            if after != series or [text.get_text() for text in axis.get_xticklabels()] != groups:
                raise ValueError('Changing legend position changed actual plotted data')
            S.end(f'17_readable_figure_{index:02}', queue)
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
