"""Record the genuine Measure -> Illumination fold on copied downloaded fields."""
import capture_ui
import capture_ui_drv_e as drv
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from capture_acceptance import assess_pipeline
from illumination_evidence import (prepare, digest, require_preserved,
                                   inspect_model, corrected_reference, verify_corrected)


def _viewer_window(tree, pid, query):
    """Identify the launched, mapped Xwayland viewer using its actual properties."""
    import re
    for wid in re.findall(r'^\s*(0x[0-9a-fA-F]+)\s', tree, re.MULTILINE):
        properties = query(['xprop', '-id', wid, '_NET_WM_PID',
                            '_NET_WM_NAME', 'WM_NAME', 'WM_CLASS'])
        owner = re.search(r'^_NET_WM_PID\([^)]*\)\s*=\s*(\d+)', properties, re.MULTILINE)
        if owner is None or int(owner.group(1)) != pid:
            continue
        if not re.search(r'^WM_CLASS\([^)]*\)\s*=.*"[Ee]og"', properties, re.MULTILINE):
            continue
        geometry = query(['xwininfo', '-id', wid])
        if 'Map State: IsViewable' not in geometry:
            continue
        for name in ('_NET_WM_NAME', 'WM_NAME'):
            title = re.search(r'^' + name + r'\([^)]*\)\s*=\s*"(.*)"\s*$',
                              properties, re.MULTILINE)
            if title is not None and title.group(1):
                return wid, title.group(1)
    return None


@drv.guarded
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

    live = capture_ui.live(capture)
    # Host: every scene clip opens before the first action of the scene.
    rec = drv.scene(capture, settle, '01_measure_host')
    choices = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
               (w.property('moduleAppKey') == 'measure' or w.property('navKey') == 'measure')]
    if not choices:
        raise ValueError('No real Home -> Measure route')
    click(max(choices, key=lambda w: w.width() * w.height()))
    settle(1)
    host = window._screens['measure']
    folds = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'illumination']
    rec.point(*folds)
    rec.stop()
    # The clip of the Source scene opens before the click on the Illumination fold.
    rec = drv.scene(capture, settle, '05_setting_src')
    choices = [w for w in host.findChildren(FoldButton) if w.isVisible() and w.app_key == 'illumination']
    if len(choices) != 1:
        raise ValueError('No unique Measure -> Illumination fold')
    click(choices[0])
    settle(1)
    screens = [w for w in window.findChildren(AppScreen) if w.isVisible() and w.app_key == 'illumination']
    if len(screens) != 1:
        raise ValueError('The actual Illumination form did not open')
    screen = screens[0]
    drv.snap(capture, '02_illumination_form')
    requested = dict(src=str(merged), channels=[0], illumination_correction=False,
                     illumination_model='', illumination_estimator='polynomial',
                     illumination_degree=4, illumination_dark=0.0,
                     illumination_per_plate=True, illumination_max_fields=16,
                     illumination_qc=True, illumination_on_missing='error')

    def fill_keys(keys):
        for key in keys:
            if not capture_ui.fill_setting(capture, screen, key, requested[key]):
                raise ValueError('No real Illumination setting for ' + key)

    def check_settings(expected):
        settings = screen._settings_model.collect()
        for key, value in expected.items():
            if settings.get(key) != value and not (value == '' and settings.get(key) is None):
                raise ValueError('The actual setting differs: ' + key)
        write_json(captures / 'configured_settings.json', settings)
        drv.snap(capture, '03_downloaded_field_settings')

    def switch_ai_off():
        if screen._ai_switch.isChecked():
            click(screen._ai_switch)
        if screen._ai_switch.isChecked():
            raise ValueError('External AI requests must remain off')

    def run(name, finished):
        outcome = dict(finished=False, ok=False, errors=[])

        def reject_prompt():
            for box in app.topLevelWidgets():
                if isinstance(box, QMessageBox) and box.isVisible():
                    outcome['errors'].append(box.windowTitle() + ': ' + box.text())
                    drv.snap(capture, name + '_prompt')
                    capture_ui.reject_dialog(capture, box)

        timer = QTimer(window)
        timer.timeout.connect(reject_prompt)
        timer.start(300)
        try:
            click(screen._btn_run)
            worker = capture_ui.wait_worker(screen, settle)
            if worker is None:
                raise ValueError('The native Run button did not start a worker')
            worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
            worker.error.connect(lambda error: outcome['errors'].append(str(error)))
            proof['actual_gui_run'] = True
            drv.snap(capture, name + '_running')
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
        capture_ui.console_end_all(capture, screen._console)
        if not capture_ui.live(capture):
            screen._console.jump_to_the_end()
        settle(.3)
        finished.stop()
        proof['runs'].append(dict(name=name, outcome=outcome, pipeline=report, console=blocks))
        write_json(captures / 'scientific_acceptance.json', proof)
        if not report['accepted']:
            raise ValueError('Native Illumination run failed: ' + str(report['reasons']))
        return blocks

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
                capture_ui.scroll_to(capture, screen._settings_scroll, section.header())
                settle(.2)
                click(section.header())
        capture_ui.scroll_to(capture, screen._settings_scroll, field)
        settle(.2)
        if not field.isVisible():
            raise ValueError('The intended field remains hidden: ' + key)
        return field

    # Host: the setting is set inside the clip of the scene that names it.
    scene_fills = {'illumination_estimator': ['illumination_estimator'],
                   'illumination_degree': ['illumination_degree'],
                   'illumination_max_fields': ['illumination_per_plate', 'illumination_max_fields'],
                   'illumination_dark': ['illumination_dark'],
                   'illumination_qc': ['illumination_qc'],
                   'illumination_on_missing': ['illumination_on_missing']}

    def setting_scene(key, rec):
        """Show one setting; the host first fills it. ``rec`` stops where the still was taken."""
        if live:
            fill_keys(scene_fills.get(key, [key] if key in ('src', 'channels') else []))
        field = expose(key)
        if key == 'illumination_correction':
            click(field)
            if screen._settings_model.collect()[key] is not True:
                raise ValueError('The native correction switch did not enable correction')
        rec.point(field)
        rec.stop()

    if live:
        setting_scene('src', rec)
        setting_scene('channels', drv.scene(capture, settle, '05_setting_channels'))
        # Not narrated on their own: the switch starts off and no model is set.
        fill_keys(['illumination_correction', 'illumination_model'])
    else:
        fill_keys(requested)
        check_settings(requested)
    switch_ai_off()

    if model_path.exists():
        raise ValueError('Fresh inputs unexpectedly have a saved model')
    off_console = run('04_correction_off', drv.scene(capture, settle, '04_correction_off_finished'))
    if model_path.exists() or 'illumination correction is OFF' not in '\n'.join(off_console):
        raise ValueError('Correction-off was not the explicitly reported no-op')
    proof['off_no_model_and_explicit_console_notice'] = True

    for key in (('src', 'channels') if not live else ()) + (
            'illumination_correction', 'illumination_estimator',
            'illumination_degree', 'illumination_max_fields', 'illumination_dark',
            'illumination_qc', 'illumination_on_missing'):
        setting_scene(key, drv.scene(capture, settle, '05_setting_' + key))
    settle(.3)
    if live:
        check_settings(dict(requested, illumination_correction=True))
    on_finished = drv.scene(capture, settle, '06_correction_on_finished')
    run('06_correction_on', on_finished)
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
        capture_ui.show_figure(capture, queue, index)
        settle(.3)
        drv.snap(capture, f'07_qc_figure_{index:02}')
    proof['model_and_pixel_checks_passed'] = True
    write_json(captures / 'scientific_acceptance.json', proof)
    if len(proof['qc_files']) != 1:
        raise ValueError('Expected exactly one real plate QC image')

    # The application saves QC but does not currently enqueue it. Show the
    # ACTUAL exported PNG in the ordinary system viewer, clearly separate from
    # spaCR, as the Diagnostics tutorial already does. No injected GUI panels.
    import subprocess
    from capture_diagnostics import PrivateDesktop
    host = capture_ui.live(capture)
    # In the capture host the whole GNOME session is private: the viewer is an
    # ordinary window the shell places; nothing is moved or raised by code.
    desktop = None if host else PrivateDesktop(Path(stage))
    qc = Path(proof['qc_files'][0]['path'])
    viewer = None
    viewer_log = None
    # Host: the clip opens before the viewer is launched. No spaCR route opens
    # the exported PNG, so the launch itself is a subprocess, as before.
    viewer_rec = drv.scene(capture, settle, '08_exported_qc_system_viewer', desktop=True)
    return_rec = drv.Rec(capture, '09_return_to_illumination', settle)
    try:
        import os as _os
        viewer_env = dict(_os.environ)
        if capture_ui.live(capture):
            viewer_env['GDK_BACKEND'] = 'x11'
            viewer_env.pop('WAYLAND_DISPLAY', None)
        viewer_log = (captures / 'qc_viewer_stderr.log').open('w')
        viewer_command = ['eog', '--new-instance']
        if host:
            viewer_command.append('--display=' + viewer_env['DISPLAY'])
        viewer_command.append(str(qc))
        viewer = subprocess.Popen(viewer_command, env=viewer_env,
                                  stdout=subprocess.DEVNULL, stderr=viewer_log)
        if host:
            from concurrent.futures import ThreadPoolExecutor

            deadline, found = time.monotonic() + 25, None
            probe = dict(tree='', errors=[])

            def query(command):
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        raise TimeoutError('Viewer detection deadline expired')
                    return subprocess.run(
                        command, env=viewer_env, text=True, check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=min(2, remaining)).stdout
                except (OSError, subprocess.SubprocessError, TimeoutError) as error:
                    probe['errors'].append(dict(command=command,
                                                error=type(error).__name__ + ': ' + str(error),
                                                stderr=str(getattr(error, 'stderr', '') or '')))
                    raise

            def inspect_viewer():
                # Only subprocess/text operations run here; Qt stays on its main thread.
                probe['tree'] = query(['xwininfo', '-root', '-tree'])
                return _viewer_window(probe['tree'], viewer.pid, query)

            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    while found is None and time.monotonic() < deadline:
                        future = executor.submit(inspect_viewer)
                        while not future.done():
                            settle(.02)
                        found = future.result()
                        if found is None:
                            if viewer.poll() is not None:
                                break
                            settle(.3)
            finally:
                write_json(captures / 'qc_viewer_window_probe.json',
                           dict(command=viewer_command, pid=viewer.pid, display=viewer_env['DISPLAY'],
                                gdk_backend=viewer_env['GDK_BACKEND'], x_window=found,
                                x_window_tree=probe['tree'], errors=probe['errors'],
                                exit_code=viewer.poll()))
            if found is None:
                raise ValueError('The launched system viewer has no mapped Xwayland window; '
                                 'see qc_viewer_stderr.log and qc_viewer_window_probe.json')
            _, title = found
        else:
            wid, title = desktop.find(qc.name, settle)
            desktop.show(wid)
        settle(1.5)
        if viewer.poll() is not None:
            raise ValueError('The real system image viewer exited before inspection')
        viewer_rec.point(window, dwell=1.5)
        viewer_rec.stop()
        proof['qc_system_viewer'] = dict(window_title=title, file=str(qc),
                                        sha256=digest(qc), native_external_window=True,
                                        embedded_spacr_panel=False)
        if capture_ui.live(capture):
            # Close the viewer the way a user does (Ctrl+W); spaCR is underneath.
            return_rec.start()
            capture.host.adapter.call('chord', ['Control_L'], 'w')
            deadline = time.monotonic() + 10
            while viewer.poll() is None and time.monotonic() < deadline:
                settle(.2)
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()
            try:
                viewer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                viewer.kill()
                viewer.wait(timeout=5)
        if viewer_log is not None:
            viewer_log.close()
        if desktop is not None:
            desktop.x.XMapRaised(desktop.display, int(window.winId()))
            desktop.x.XFlush(desktop.display)
            desktop.close()
    settle(.5)
    return_rec.stop()
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
