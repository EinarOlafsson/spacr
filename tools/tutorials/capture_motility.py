"""Record Measure's real Motility page on private verified synthetic tracks."""
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from capture_acceptance import assess_pipeline
from replication_demo import digest


def record_motility(app, window, stage, captures, capture, settle, write_json, timeout,
                    *, probe_screen_export=False):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QFileDialog, QLineEdit,
                                  QDialogButtonBox, QMessageBox)
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    stage = Path(stage).resolve()
    prior = json.loads((stage / 'captures/timelapse_verified_explicit_relink/scientific_acceptance.json').read_text())
    source = Path(prior['source']).resolve()
    if not prior['accepted'] or not prior['synthetic'] or not source.is_relative_to(stage):
        raise ValueError('Motility requires the accepted private synthetic Timelapse source')
    files = [source / f'merged/plate1_A01_1_{i}.npy' for i in range(1, 9)]
    files.append(source / 'merged/.spacr_plane_layout.json')
    before = {str(p): digest(p) for p in files}
    runs = stage / 'motility_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='SYNTHETIC-verified-tracks-', dir=runs))
    (root / 'merged').mkdir()
    for path in files:
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError('Unexpected Timelapse source shape or link')
        if path.suffix == '.npy':
            arr = np.load(path, allow_pickle=False)
            if arr.shape != (256, 256, 4) or arr.dtype != np.uint16:
                raise ValueError('Unexpected verified Timelapse array')
        shutil.copy2(path, root / 'merged' / path.name)
        if digest(root / 'merged' / path.name) != before[str(path)]:
            raise ValueError('The private copy differs from the accepted source')
    copied = {str(p): digest(p) for p in (root / 'merged').iterdir() if p.is_file()}
    proof = dict(lesson='18_motility', accepted=False, synthetic=True,
                 source=str(root), source_manifest=before, copied_inputs=copied,
                 app_source_modified=False, published=False, actual_gui_run=False)
    write_json(captures / 'scientific_acceptance.json', proof)

    def click(widget, *, settle_after=True):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The actual Motility control is unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        if settle_after:
            settle(.25)

    choices = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey') == 'measure' or w.property('navKey') == 'measure')]
    if not choices:
        raise ValueError('No actual Home -> Measure control')
    click(max(choices, key=lambda w: w.width() * w.height()))
    host = window._screens['measure']
    capture('01_measure_host')
    folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'motility']
    if len(folds) != 1:
        raise ValueError('No unique actual Measure -> Motility fold')
    click(folds[0])
    settle(.8)
    screens = [w for w in window.findChildren(AppScreen) if w.isVisible() and w.app_key == 'motility']
    if len(screens) != 1:
        raise ValueError('The actual Motility page did not open')
    screen = screens[0]
    capture('02_motility_page')
    if probe_screen_export:
        from motility_figure_preferences import choose_screen_export
        proof['figure_export_preference'] = choose_screen_export(app, window, screen, capture, settle)
        settle(.5)
        screens = [w for w in window.findChildren(AppScreen) if w.isVisible() and w.app_key == 'motility']
        if len(screens) != 1:
            raise ValueError('Preferences did not retain the real Motility page')
        screen = screens[0]
    # Explicit arithmetic exercise, NOT an inferred acquisition calibration.
    # The batch form has numeric-only spinboxes: None is rejected. The real
    # preview can represent unknown units, and demonstrates that separately.
    proof['unknown_batch_calibration_probe'] = {}
    for key in ('pixels_per_um', 'seconds_per_frame'):
        widget = screen._settings_model._widgets[key]
        proof['unknown_batch_calibration_probe'][key] = dict(
            widget=type(widget).__name__, before=screen._settings_model.collect().get(key),
            none_accepted=screen._settings_model.set_value_for_key(key, None),
            after=screen._settings_model.collect().get(key))
    proof['calibration_scope'] = 'Hypothetical unit-conversion exercise: 2 px/um and 60 s/frame; not an acquired or validated physical scale.'
    requested = dict(src=str(root), channels=[0, 1], cell_channel=1, nucleus_channel=0,
                     pathogen_channel=None, tracked_object='cell', pixels_per_um=2.0,
                     seconds_per_frame=60, reuse_existing_measurements=False,
                     max_displacement=50, straightness_filter=False, n_jobs=1,
                     infection_intensity_qc_scope='none', infection_intensity_strategy='histogram',
                     infection_intensity_qc_graphs=True, motility_xlim=(-30, 30), motility_ylim=(-30, 30))
    if 'plot' in screen._settings_model._widgets:
        requested['plot'] = True
    for key, value in requested.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('No genuine Motility setting: ' + key)
    settings = screen._settings_model.collect()
    proof['configured_settings'] = settings
    proof['requested_settings'] = requested
    write_json(captures / 'scientific_acceptance.json', proof)
    if any(settings.get(k) != v for k, v in requested.items()):
        raise ValueError('Actual Motility settings differ: ' + repr({
            k: settings.get(k) for k, v in requested.items() if settings.get(k) != v}))
    for key in ('src', 'pixels_per_um', 'reuse_existing_measurements', 'infection_intensity_qc_graphs'):
        field = screen._settings_model._widgets[key]
        parents, widget = [], field.parentWidget()
        while widget is not None and widget is not screen:
            if callable(getattr(widget, 'is_expanded', None)) and callable(getattr(widget, 'header', None)):
                parents.append(widget)
            widget = widget.parentWidget()
        for section in reversed(parents):
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header())
                settle(.2)
                click(section.header())
        screen._settings_scroll.ensureWidgetVisible(field)
        settle(.2)
        if not field.isVisible() or field.visibleRegion().isEmpty():
            raise ValueError('A narrated Motility setting is not visible: ' + key)
        capture('03_actual_setting_' + key)
    if screen._ai_switch.isChecked():
        click(screen._ai_switch)
    if screen._console_folder.shut:
        click(screen._console_folder.heading)
    outcome = dict(finished=False, ok=False, errors=[])
    timer = QTimer(window)

    def reject_prompt():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                outcome['errors'].append(box.windowTitle() + ': ' + box.text())
                capture('04_unexpected_prompt')
                box.reject()

    timer.timeout.connect(reject_prompt)
    timer.start(300)
    try:
        click(screen._btn_run, settle_after=False)
        worker = screen._worker
        if worker is None:
            raise ValueError('The actual Run button did not launch a worker')
        worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
        worker.error.connect(lambda error: outcome['errors'].append(str(error)))
        proof['actual_gui_run'] = True
        capture('04_actual_running')
        deadline = time.monotonic() + timeout
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic() > deadline:
                click(screen._btn_stop)
                settle(2)
                raise TimeoutError('Bounded native Motility run timed out')
            settle(.1)
        settle(.5)
    finally:
        timer.stop()
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    write_json(captures / 'batch_console.json', blocks)
    proof['outcome'] = outcome
    proof['pipeline'] = assess_pipeline(outcome, blocks, screen._figure_queue.count(), requires_figure=False)
    proof['figure_count'] = screen._figure_queue.count()
    proof['figures_visible'] = screen._figures_card.isVisible()
    proof['outputs'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                        for p in root.rglob('*') if p.is_file()]
    proof['source_preserved'] = all(digest(p) == h for p, h in before.items())
    proof['copied_inputs_preserved'] = all(digest(p) == h for p, h in copied.items())
    if not proof['source_preserved'] or not proof['copied_inputs_preserved']:
        raise ValueError('The native Motility run changed input arrays')
    for block, _, _ in screen._console._pipeline_console_blocks():
        block.setFocus()
        QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    screen._console.jump_to_the_end()
    settle(.3)
    capture('05_actual_finished')
    for index, pixmap in enumerate(screen._figure_queue.all_pixmaps()):
        pixmap.save(str(captures / f'figure_{index:02}.png'), 'PNG')
        screen._figure_queue.show_index(index)
        settle(.2)
        capture(f'06_actual_figure_{index:02}')
    write_json(captures / 'scientific_acceptance.json', proof)
    if not proof['pipeline']['accepted']:
        return

    from motility_native_preview import record_preview
    proof['live_preview'] = record_preview(app, window, screen, root, captures,
                                          capture, settle, write_json, timeout)
    proof['next_gate'] = 'Independent measurement, unit, filter and plot checks before narration'
    write_json(captures / 'scientific_acceptance.json', proof)
