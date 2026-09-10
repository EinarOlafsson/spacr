"""Record the current Mask fold and real synthetic demo without repairing the app."""
from pathlib import Path
import tempfile
import time

from capture_acceptance import assess_pipeline
from replication_demo import digest


def record_timelapse(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QFileDialog, QLineEdit,
                                  QDialogButtonBox, QMessageBox)
    from spacr.qt.widgets.fold_strip import FoldButton

    old = Path(stage).parent / 'synthetic/timelapse'
    original = {str(p): digest(p) for p in old.rglob('*') if p.is_file()}
    runs = Path(stage) / 'timelapse_runs'
    runs.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='SYNTHETIC-current-demo-', dir=runs))
    proof = dict(lesson='17_timelapse', accepted=False, synthetic=True,
                 source=str(root), original_files=original, app_source_modified=False,
                 actual_gui_run=False, published=False)
    write_json(captures / 'scientific_acceptance.json', proof)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The native Timelapse control is unavailable: ' + widget.objectName())
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.3)

    home = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
            (w.property('moduleAppKey') == 'mask' or w.property('navKey') == 'mask')]
    if not home:
        raise ValueError('No native Home -> Mask route')
    click(max(home, key=lambda w: w.width() * w.height()))
    screen = window._screens['mask']
    capture('01_mask_host')
    folds = [w for w in screen.findChildren(FoldButton) if w.isVisible() and w.app_key == 'timelapse']
    if len(folds) != 1:
        raise ValueError('No unique actual Mask -> Timelapse fold')
    if not folds[0].isChecked():
        click(folds[0])
    if screen._settings_model.collect().get('timelapse') is not True:
        raise ValueError('The actual Timelapse fold did not enable tracking')
    capture('02_timelapse_fold')

    action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
    help_menu = action.menu()
    demos_action = next(a for a in help_menu.actions() if a.text().replace('&', '') == 'Demos')
    from spacr.qt.first_run import find_menu
    menu = find_menu(window, 'Demos')
    if menu is None:
        raise ValueError('No actual Help -> Demos submenu')
    demo = next(a for a in menu.actions() if a.text().replace('&', '') == 'Timelapse demo…')
    QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                     pos=window.menuBar().actionGeometry(action).center())
    settle(.3)
    QTest.mouseMove(help_menu, help_menu.actionGeometry(demos_action).center())
    settle(.5)
    if not menu.isVisible():
        raise ValueError('The actual Help -> Demos submenu did not open')
    capture('03_actual_demo_menu')
    accepted, errors = [], []
    timer, watchdog = QTimer(window), QTimer(window)
    timer.setSingleShot(True)
    watchdog.setSingleShot(True)

    def destination():
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise ValueError('The Demos action did not open its real directory picker')
            dialog.accepted.connect(lambda: accepted.append(True))
            dialog.resize(1400, 950)
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            click(edit)
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(root))
            capture('04_fresh_synthetic_destination')
            box = dialog.findChild(QDialogButtonBox)
            buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
            if len(buttons) != 1:
                raise ValueError('No unique real directory selection button')
            click(buttons[0])
        except Exception as error:
            errors.append(str(error))
            if dialog is not None:
                dialog.reject()

    def abort():
        errors.append('The real demo directory picker timed out')
        if app.activeModalWidget() is not None:
            app.activeModalWidget().reject()

    timer.timeout.connect(destination)
    watchdog.timeout.connect(abort)
    timer.start(400)
    watchdog.start(20000)
    try:
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(demo).center())
    finally:
        timer.stop()
        watchdog.stop()
    if errors or not accepted:
        raise ValueError('; '.join(errors) or 'The native demo directory was not accepted')
    settle(.7)
    # A changed channel layout legitimately rebuilds the complete Mask form.
    # Never read or click the superseded screen retained by the recorder.
    previous_screen = screen
    screen = window._screens['mask']
    proof['demo_rebuilt_form'] = screen is not previous_screen
    if not screen.isVisible():
        raise ValueError('The current Mask form is not the visible demo screen')
    if not (root / 'settings_timelapse.csv').is_file():
        raise ValueError('The actual Demos action did not generate its settings file')
    inputs = sorted(root.glob('*.tif'))
    if len(inputs) != 16:
        raise ValueError('The native demo no longer contains eight frames / two channels')
    proof['generated_inputs'] = [dict(path=str(p), sha256=digest(p)) for p in inputs]
    proof['actual_demo_action'] = True
    proof['demo_settings'] = screen._settings_model.collect()
    write_json(captures / 'scientific_acceptance.json', proof)
    capture('05_demo_loaded')
    if (proof['demo_settings'].get('timelapse') is not True or
            proof['demo_settings'].get('src') != str(root)):
        raise ValueError('The actual demo did not populate Mask with tracking enabled: ' +
                         repr({k: proof['demo_settings'].get(k) for k in ('src', 'timelapse')}))
    requested = dict(plot=True, batch_size=1, keep_original_images=True,
                     timelapse_displacement=50)
    if 'n_jobs' in screen._settings_model._widgets:
        requested['n_jobs'] = 1
    for key, value in requested.items():
        if not screen._settings_model.set_value_for_key(key, value):
            raise ValueError('Missing genuine setting: ' + key)
    configured = screen._settings_model.collect()
    if any(configured.get(k) != v for k, v in requested.items()):
        raise ValueError('A bounded/Plot setting did not stick')
    write_json(captures / 'configured_settings.json', configured)
    bar = screen._settings_search
    previous_search = (bar.query(), bar.level(), bar.modified_only())
    if bar.modified_only():
        click(bar._modified)
    if bar.level() != 'all':
        click(bar._disclosure)
    bar.set_query('')
    for key in ('timelapse_frame_limits', 'timelapse_objects', 'timelapse_mode',
                'timelapse_displacement', 'keep_original_images', 'plot'):
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
            raise ValueError('The actual Timelapse setting is not visible: ' + key)
        from capture_geometry import capture_rect
        proof.setdefault('setting_rectangles', {})[key] = capture_rect(field, window)
        capture('05_setting_' + key)
    bar.set_query(previous_search[0])
    bar.set_level(previous_search[1])
    bar.set_modified_only(previous_search[2])
    if screen._settings_model.collect() != configured:
        raise ValueError('The settings tour changed the real demo configuration')
    if screen._ai_switch.isChecked():
        click(screen._ai_switch)
    if screen._console_folder.shut:
        click(screen._console_folder.heading)
    outcome = dict(finished=False, ok=False, errors=[])
    prompt_guard = QTimer(window)

    def reject_prompt():
        for box in app.topLevelWidgets():
            if isinstance(box, QMessageBox) and box.isVisible():
                outcome['errors'].append(box.windowTitle() + ': ' + box.text())
                capture('06_unexpected_pipeline_prompt')
                box.reject()

    prompt_guard.timeout.connect(reject_prompt)
    prompt_guard.start(400)
    try:
        click(screen._btn_run)
        worker = screen._worker
        if worker is None:
            raise ValueError('The actual Timelapse Run button did not launch a worker')
        worker.finished.connect(lambda ok: outcome.update(finished=True, ok=bool(ok)))
        worker.error.connect(lambda message: outcome['errors'].append(str(message)))
        proof['actual_gui_run'] = True
        capture('06_actual_running')
        deadline = time.monotonic() + timeout
        while not outcome['finished'] or screen._worker_thread_is_running():
            if time.monotonic() > deadline:
                click(screen._btn_stop)
                settle(3)
                raise TimeoutError('Bounded native Timelapse run exceeded its time limit')
            settle(.1)
        settle(.5)
    finally:
        prompt_guard.stop()
    blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
    write_json(captures / 'batch_console.json', blocks)
    proof['outcome'] = outcome
    proof['pipeline'] = assess_pipeline(outcome, blocks, screen._figure_queue.count())
    proof['figure_count'] = screen._figure_queue.count()
    proof['figures_visible'] = screen._figures_card.isVisible()
    proof['outputs'] = [dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p))
                        for p in root.rglob('*') if p.is_file()]
    proof['original_files_preserved'] = all(digest(p) == h for p, h in original.items())
    if not proof['original_files_preserved']:
        raise ValueError('An original tutorial file changed')
    for block, _, _ in screen._console._pipeline_console_blocks():
        block.setFocus()
        QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
    screen._console.jump_to_the_end()
    settle(.3)
    capture('07_actual_finished')
    for i, pixmap in enumerate(screen._figure_queue.all_pixmaps()):
        pixmap.save(str(captures / f'figure_{i:02}.png'), 'PNG')
        screen._figure_queue.show_index(i)
        settle(.2)
        capture(f'08_actual_figure_{i:02}')
    # Successful execution alone is not independent track validation.
    proof['accepted'] = False
    proof['next_gate'] = 'Independent input/track/output and live-preview checks'
    write_json(captures / 'scientific_acceptance.json', proof)
    if proof['pipeline']['accepted'] and proof['figures_visible']:
        from timelapse_native_preview import record_preview
        proof['live_preview'] = record_preview(app, window, screen, root, captures,
                                              capture, settle, write_json, timeout)
        proof['next_gate'] = 'Editorial review of the measured workflow and remaining application warnings'
        write_json(captures / 'scientific_acceptance.json', proof)
