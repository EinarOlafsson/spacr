"""Record the genuine Measure -> Illumination fold on copied downloaded fields."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from capture_acceptance import assess_pipeline
from illumination_evidence import (prepare, digest, require_preserved,
                                   inspect_model, corrected_reference, verify_corrected)


def record_illumination(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QMessageBox
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widgets.fold_strip import FoldButton

    manifest = prepare(stage)
    merged = Path(manifest['merged'])
    model_path = merged.parent / 'illumination/illumination_model.npz'
    proof = dict(lesson='43_illumination', accepted=False, input_manifest=manifest,
                 actual_gui_run=False, app_source_modified=False, synthetic_images=False,
                 physical_illumination_validated=False, biological_validation=False,
                 published=False, runs=[])
    write_json(captures / 'scientific_acceptance.json', proof)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('A real Illumination control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    choices = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey') == 'measure' or w.property('navKey') == 'measure')]
    if not choices:
        raise ValueError('No real Home -> Measure route')
    click(max(choices, key=lambda w: w.width() * w.height()))
    settle(1)
    host = window._screens['measure']
    capture('01_measure_host')
    choices = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'illumination']
    if len(choices) != 1:
        raise ValueError('No unique Measure -> Illumination fold')
    click(choices[0])
    settle(1)
    screens = [w for w in window.findChildren(AppScreen) if w.isVisible() and w.app_key == 'illumination']
    if len(screens) != 1:
        raise ValueError('The actual Illumination form did not open')
    screen = screens[0]
    capture('02_illumination_form')
    requested = dict(src=str(merged), channels=[0], illumination_correction=False,
                     illumination_model='', illumination_estimator='polynomial',
                     illumination_degree=4, illumination_dark=0.0,
                     illumination_per_plate=True, illumination_max_fields=16,
                     illumination_qc=True, illumination_on_missing='error')
    for key, value in requested.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('No real Illumination setting for ' + key)
    settings = screen._settings_model.collect()
    for key, value in requested.items():
        if settings.get(key) != value and not (value == '' and settings.get(key) is None):
            raise ValueError('The actual setting differs: ' + key)
    write_json(captures / 'configured_settings.json', settings)
    capture('03_downloaded_field_settings')
    if screen._ai_switch.isChecked():
        click(screen._ai_switch)
    if screen._ai_switch.isChecked():
        raise ValueError('External AI requests must remain off')

    def run(name):
        outcome = dict(finished=False, ok=False, errors=[])

        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box, QMessageBox) and box.isVisible():
                    outcome['errors'].append(box.windowTitle() + ': ' + box.text())
                    capture(name + '_prompt')
                    box.reject()

        timer = QTimer(window)
        timer.timeout.connect(reject_prompt)
        timer.start(300)
        try:
            click(screen._btn_run)
            worker = screen._worker
            if worker is None:
                raise ValueError('The native Run button did not start a worker')
            worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
            worker.error.connect(lambda error: outcome['errors'].append(str(error)))
            proof['actual_gui_run'] = True
            capture(name + '_running')
            deadline = time.monotonic() + timeout
            while not outcome['finished'] or screen._worker_thread_is_running():
                if time.monotonic() > deadline:
                    click(screen._btn_stop)
                    settle(3)
                    raise TimeoutError('The bounded native Illumination run did not finish')
                settle(.1)
            settle(.8)
        finally:
            timer.stop()
        blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
        report = assess_pipeline(outcome, blocks, screen._figure_queue.count(), requires_figure=False)
        for block, _, _ in screen._console._pipeline_console_blocks():
            block.setFocus()
            QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
        screen._console.jump_to_the_end()
        settle(.3)
        capture(name + '_finished')
        proof['runs'].append(dict(name=name, outcome=outcome, pipeline=report, console=blocks))
        write_json(captures / 'scientific_acceptance.json', proof)
        if not report['accepted']:
            raise ValueError('Native Illumination run failed: ' + str(report['reasons']))
        return blocks

    if model_path.exists():
        raise ValueError('Fresh inputs unexpectedly have a saved model')
    off_console = run('04_correction_off')
    if model_path.exists() or 'illumination correction is OFF' not in '\n'.join(off_console):
        raise ValueError('Correction-off was not the explicitly reported no-op')
    proof['off_no_model_and_explicit_console_notice'] = True

    # Folded settings pages expose their real category headers, not the
    # top-level module's search bar. Expand the actual ancestors of each field.
    def expose(key):
        field = screen._settings_model._widgets[key]
        parents, widget = [], field.parentWidget()
        sections = {id(s) for s in screen._settings_sections}
        while widget is not None and widget is not screen:
            if id(widget) in sections:
                parents.append(widget)
            widget = widget.parentWidget()
        for section in reversed(parents):
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header())
                settle(.2)
                click(section.header())
        screen._settings_scroll.ensureWidgetVisible(field)
        settle(.2)
        if not field.isVisible():
            raise ValueError('The intended field remains hidden: ' + key)
        return field

    for key in ('src', 'channels', 'illumination_correction', 'illumination_estimator',
                'illumination_degree', 'illumination_max_fields', 'illumination_dark',
                'illumination_qc', 'illumination_on_missing'):
        field = expose(key)
        if key == 'illumination_correction':
            click(field)
            if screen._settings_model.collect()[key] is not True:
                raise ValueError('The native correction switch did not enable correction')
        capture('05_setting_' + key)
    settle(.3)
    run('06_correction_on')
    plane, model_checks = inspect_model(model_path, merged)
    proof['saved_model_checks'] = model_checks
    proof['saved_model_sha256'] = digest(model_path)

    # Offline audit of the REAL saved model through its public API, after the
    # GUI run. This is not substituted output or a simulated application run.
    from spacr.illumination import IlluminationModel, IlluminationCorrector
    actual_model = IlluminationModel.load(str(model_path))
    corrector = IlluminationCorrector(actual_model, verbose=False)
    proof['offline_pixel_audit'] = []
    for row in manifest['files'][:2]:
        image = np.array(np.load(row['copy'], mmap_mode='r')[..., :1], copy=True)
        original = image.copy()
        expected = corrected_reference(image, plane)
        actual = corrector(image, SimpleNamespace(file_name=Path(row['copy']).name, channels=[0]))
        checked = verify_corrected(actual, expected)
        if not np.array_equal(image, original):
            raise ValueError('The public correction API changed its input array')
        proof['offline_pixel_audit'].append(dict(file=Path(row['copy']).name, **checked))
    proof['input_files_preserved'] = require_preserved(manifest['files'])
    proof['qc_files'] = [dict(path=str(p), sha256=digest(p)) for p in
                         sorted(model_path.parent.glob('illumination_qc*.png'))]
    queue = screen._figure_queue
    proof['figure_count'] = queue.count()
    proof['figures_visible'] = screen._figures_card.isVisible()
    for index, pixmap in enumerate(queue.all_pixmaps()):
        pixmap.save(str(captures / f'figure_{index:02}.png'), 'PNG')
        queue.show_index(index)
        settle(.3)
        capture(f'07_qc_figure_{index:02}')
    proof['model_and_pixel_checks_passed'] = True
    write_json(captures / 'scientific_acceptance.json', proof)
    if len(proof['qc_files']) != 1:
        raise ValueError('Expected exactly one real plate QC image')

    # The application saves QC but does not currently enqueue it. Show the
    # ACTUAL exported PNG in the ordinary system viewer, clearly separate from
    # spaCR, as the Diagnostics tutorial already does. No injected GUI panels.
    import subprocess
    from capture_diagnostics import PrivateDesktop
    desktop = PrivateDesktop(Path(stage))
    qc = Path(proof['qc_files'][0]['path'])
    viewer = None
    try:
        viewer = subprocess.Popen(['eog', '--new-instance', str(qc)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        wid, title = desktop.find(qc.name, settle)
        desktop.show(wid)
        settle(1.5)
        if viewer.poll() is not None:
            raise ValueError('The real system image viewer exited before inspection')
        capture('08_exported_qc_system_viewer', desktop=True)
        proof['qc_system_viewer'] = dict(window_title=title, file=str(qc),
                                        sha256=digest(qc), native_external_window=True,
                                        embedded_spacr_panel=False)
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try:
                viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                viewer.kill()
                viewer.wait(timeout=5)
        desktop.x.XMapRaised(desktop.display, int(window.winId()))
        desktop.x.XFlush(desktop.display)
        desktop.close()
    settle(.5)
    capture('09_return_to_illumination')
    if digest(qc) != proof['qc_files'][0]['sha256']:
        raise ValueError('Viewing the exported QC changed the image')
    if digest(model_path) != proof['saved_model_sha256']:
        raise ValueError('The saved model changed during inspection')
    require_preserved(manifest['files'])
    proof['accepted'] = True
    proof['qc_viewing_requires_exported_file'] = not bool(queue.count() and proof['figures_visible'])
    write_json(captures / 'scientific_acceptance.json', proof)


def main():
    """Isolate the image viewer's desktop, bus and settings before recording."""
    import os
    import subprocess
    import sys
    import tempfile
    from stage_lesson import DEFAULT_STAGE
    stage = DEFAULT_STAGE.resolve()
    root = stage / 'desktop/illumination_native_qc'
    env = dict(os.environ)
    for key, name in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                      ('XDG_CACHE_HOME', 'cache')]:
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    runtime = root / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    env['XDG_RUNTIME_DIR'] = tempfile.mkdtemp(prefix='capture-', dir=runtime)
    env.update(SPACR_TUTORIAL_PRIVATE_DESKTOP='1', GIO_USE_VFS='local',
               GVFS_DISABLE_FUSE='1', GSETTINGS_BACKEND='memory', GTK_USE_PORTAL='0',
               QT_QPA_PLATFORMTHEME='', XDG_CURRENT_DESKTOP='SPACR_TUTORIAL',
               NO_AT_BRIDGE='1', GDK_SCALE='2', GDK_DPI_SCALE='1')
    command = ['xvfb-run', '-a', '-s', '-screen 0 3840x2160x24 -nolisten tcp',
               'dbus-run-session', '--', sys.executable,
               str(Path(__file__).with_name('capture_refresh.py')), '--module', 'illumination',
               '--stage', str(stage), '--capture-name', 'illumination_exported_qc_viewer',
               '--platform', 'xcb', '--timeout', '240']
    return subprocess.run(command, env=env, timeout=360, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
