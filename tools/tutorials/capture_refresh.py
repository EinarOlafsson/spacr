#!/usr/bin/env python3
"""Capture the current, unmodified GUI for the tutorial refresh.

Use one process per module. Configuration and downloaded data are isolated from
the desktop user's files, while model/download caches can still be reused. No
version strings, pipeline functions, download callbacks or results are mocked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from capture_acceptance import assess_pipeline

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path('/mnt/firecuda2/Claude/toxoplasma_projects/tutorials')
DEFAULT_STAGE = WORKSPACE / 'refresh_2026-09-09'


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', default='home')
    parser.add_argument('--stage', type=Path, default=DEFAULT_STAGE)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--preview-variants', action='store_true')
    parser.add_argument('--platform', choices=('offscreen', 'xcb'), default='offscreen')
    parser.add_argument('--run', action='store_true', help='Record a bounded Plot-enabled real pipeline run')
    parser.add_argument('--ai-controls', action='store_true', help='Show the AI toggle and an UNSENT example question')
    parser.add_argument('--settings-tour', action='store_true', help='Show bounded Regression or Classify choices through real settings searches')
    parser.add_argument('--annotation-tour', action='store_true', help='Record actual crop labelling and view changes in a new example column')
    parser.add_argument('--mask-editor-tour', action='store_true', help='Record actual reversible mask-editing gestures on private real-data copies')
    parser.add_argument('--editor-detect', action='store_true', help='Also run actual Cellpose once on the small recropped example')
    parser.add_argument('--font-scale', type=float, default=1.5, help='Use the actual app font preference for the recording')
    parser.add_argument('--capture-name', help='Preserve earlier accepted frames in a separate capture directory')
    parser.add_argument('--test-data-route', choices=('load', 'stream'), default='load', help='Choose the real Annotate/Classify test-data route')
    parser.add_argument('--diagnostics-from', type=Path, help='Existing private tutorial regression project to inspect')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.preview_variants and not args.preview:
        parser.error('--preview-variants requires --preview')
    if args.settings_tour and (args.module not in {'regression', 'classify_merged', 'umap'} or not args.run):
        parser.error('--settings-tour requires --module regression/classify_merged/umap --run')
    if args.annotation_tour and args.module != 'annotate':
        parser.error('--annotation-tour requires --module annotate')
    if args.mask_editor_tour and args.module != 'make_masks':
        parser.error('--mask-editor-tour requires --module make_masks')
    if args.editor_detect and not args.mask_editor_tour:
        parser.error('--editor-detect requires --mask-editor-tour')
    if args.capture_name and (Path(args.capture_name).name != args.capture_name or args.capture_name in {'.', '..'}):
        parser.error('--capture-name must be one directory name')
    if args.module == 'regression_diagnostics' and args.diagnostics_from is None:
        parser.error('Regression Diagnostics needs an already completed --diagnostics-from project')
    stage = args.stage.resolve()
    stage.mkdir(parents=True, exist_ok=True)
    # The current downloader intentionally uses Path.home(), not the older
    # SPACR_EXAMPLE_DATA override. Bind ONLY its cache into this recording's
    # private directory. The app, its UI callback and downloader stay genuine,
    # and neither a user's cached experiments nor their HOME are changed.
    if os.environ.get('SPACR_TUTORIAL_CACHE_ISOLATED') != '1':
        private_cache = stage / 'example_data'
        private_cache.mkdir(parents=True, exist_ok=True)
        destination = Path.home() / '.cache/spacr/example_data'
        private_runs = stage / 'runs'
        private_runs.mkdir(parents=True, exist_ok=True)
        runs_destination = Path.home() / '.spacr/runs'
        os.environ['SPACR_TUTORIAL_CACHE_ISOLATED'] = '1'
        os.execvp('bwrap', ['bwrap', '--die-with-parent', '--bind', '/', '/',
                          '--dev-bind', '/dev', '/dev',
                          '--bind', str(private_cache), str(destination),
                          '--bind', str(private_runs), str(runs_destination), '--',
                          sys.executable, str(Path(__file__).resolve()),
                          *sys.argv[1:]])
    for key, value in {
        'QT_QPA_PLATFORM': args.platform, 'QT_SCALE_FACTOR': '1',
        'QT_AUTO_SCREEN_SCALE_FACTOR': '0', 'QT_FONT_DPI': '96',
        'SPACR_LANGUAGE': 'en', 'XDG_CONFIG_HOME': str(stage / 'config' / args.module),
        'SPACR_EXAMPLE_DATA': str(stage / 'example_data'),
        'SPACR_LOG_DIR': str(stage / 'logs'),
        'MPLCONFIGDIR': str(stage / 'mpl'),
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(REPO))
    from PySide6.QtCore import QPoint, Qt, QTimer
    from PySide6.QtGui import QPainter
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QAbstractButton, QDialog, QLabel, QMenu, QMessageBox, QTabWidget
    from shiboken6 import isValid
    import spacr
    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt import app as gui
    from spacr.qt.first_run import mark_tour_seen
    from spacr.qt.walkthrough import mark_seen
    from spacr.qt.preferences import (apply_preferences_to_app, set_preload_policy,
                                      set_theme, set_font_scale)
    from spacr.qt.widgets.fold_strip import folded_modules

    # A private Xvfb recording cannot capture a portal/GTK dialog in another
    # desktop process. Use Qt's genuine file dialog, with identical operations.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs)
    app = QApplication.instance() or QApplication([])
    mark_tour_seen()
    for key, *_ in gui.APPS:
        mark_seen(key)
    for key in folded_modules():
        mark_seen(key)
    set_preload_policy('on_demand')
    set_theme('dark')
    set_font_scale(args.font_scale)
    if args.module == 'regression':
        from spacr.qt.preferences import set_figure_format
        set_figure_format('png')
    apply_preferences_to_app(app)
    window = gui.MainWindow()
    window.apply_dock_mode('locked')
    window.resize(3840, 2160)
    window.show()
    captures = stage / 'captures' / (args.capture_name or args.module)
    captures.mkdir(parents=True, exist_ok=True)
    write_json(captures / 'provenance.json', {'module': args.module,
               'completed_capture': False, 'status': 'capture_in_progress'})
    frames = {}

    def settle(seconds=0.6):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            app.processEvents()
            time.sleep(0.02)

    def rect(widget):
        if not widget.isVisible():
            return None
        point = widget.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
        x, y = max(0, point.x()), max(0, point.y())
        right = min(window.width(), point.x() + widget.width())
        bottom = min(window.height(), point.y() + widget.height())
        if right <= x or bottom <= y:
            return None
        return [x, y, right - x, bottom - y]

    def capture(name, *, desktop=False):
        pixmap = app.primaryScreen().grabWindow(0) if desktop else window.grab()
        if (pixmap.width(), pixmap.height()) != (3840, 2160):
            raise RuntimeError(f'Unexpected capture size {pixmap.size()}')
        painter = QPainter(pixmap)
        dialogs = [] if desktop else [w for w in app.topLevelWidgets()
                   if isinstance(w, (QDialog, QMenu)) and w.isVisible()]
        for dialog in dialogs:
            origin = dialog.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
            painter.drawPixmap(origin, dialog.grab())
        painter.end()
        path = captures / f'{name}.png'
        if not pixmap.save(str(path), 'PNG'):
            raise RuntimeError(f'Cannot save {path}')
        buttons = []
        for widget in window.findChildren(QAbstractButton):
            geometry = rect(widget)
            if geometry is not None:
                buttons.append({'text': widget.text(), 'name': widget.objectName(),
                                'tooltip': widget.toolTip(), 'rect': geometry,
                                'enabled': widget.isEnabled(),
                                'nav_key': widget.property('navKey'),
                                'module_key': widget.property('moduleAppKey')})
        frames[name] = {'image': path.name,
                        'capture_surface': 'private_desktop' if desktop else 'application_window',
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'buttons': buttons,
                        'dialogs': [{'title': d.windowTitle(), 'rect': rect(d),
                                     'labels': [l.text() for l in d.findChildren(QLabel)]}
                                    for d in dialogs]}
        write_json(captures / 'frames.json', frames)
        print(f'captured {args.module}/{name}', flush=True)

    settle(2)
    apps = gui.tiled_apps(gui.visible_apps())
    inventory = {
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                                          text=True).strip(),
        'version': spacr.__version__,
        'registry': gui.APPS,
        'home_bands': gui.home_bands(apps),
        'home_categories': gui.home_categories(apps),
        'core': [r for r in apps if r[3] == gui.SECTION_CORE],
        'folds': folded_modules(),
        'folded_children': gui.folded_children(),
    }
    write_json(stage / 'runtime_inventory.json', inventory)
    capture('00_home')
    if args.module == 'home':
        home = window._startup
        tabs = home._tabs
        for index in range(1, tabs.count()):
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                             pos=tabs.tabBar().tabRect(index).center())
            settle()
            capture(f'{index:02d}_{tabs.tabText(index).lower()}')
        tabs.setCurrentIndex(1)
        window._on_nav_selected('classify_merged')
        settle(3)
        capture('05_classify_host')
        window._on_nav_selected('mask')
        settle(3)
        capture('06_mask_host')
        window._on_nav_selected('__home__')
        settle()
        help_menu = next(a.menu() for a in window.menuBar().actions()
                         if a.text().replace('&', '') == 'Help')
        help_menu.popup(window.menuBar().mapToGlobal(QPoint(110, 30)))
        settle()
        capture('07_help')
        help_menu.hide()
    if args.module != 'home':
        host_key = {'import_images': 'foreign', 'regression_diagnostics': 'regression'}.get(args.module, args.module)
        window._on_nav_selected(host_key)
        deadline = time.monotonic() + 60
        while window._screens.get(host_key) is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{args.module} did not open')
            settle(0.1)
        settle(2)
        screen = window._screens[host_key]
        capture('01_module')
        if args.mask_editor_tour:
            from capture_make_masks import record_editor
            record_editor(app, window, screen, stage, captures, capture,
                          settle, write_json, args.timeout, detect=args.editor_detect)
        if args.module == 'import_images':
            from capture_image_import import record_import
            screen = record_import(app, window, screen, stage, captures,
                                   capture, settle, write_json, args.timeout)
        if args.module == 'regression_diagnostics':
            from capture_diagnostics import record_diagnostics
            record_diagnostics(window, screen, stage, args.diagnostics_from,
                               captures, capture, settle, write_json)
        if args.download:
            def visible_test_data_buttons():
                return [w for w in screen.findChildren(QAbstractButton)
                        if w.isVisible() and w.isEnabled()
                        and w.text().replace('…', '').strip() == 'Load test data']

            buttons = visible_test_data_buttons()
            if not buttons and args.module == 'classify_merged':
                from copy import deepcopy
                from capture_settings import require_unchanged_settings
                before_disclosure = deepcopy(screen._settings_model.collect())
                bar = screen._settings_search
                if bar.level() != 'all':
                    QTest.mouseClick(bar._disclosure, Qt.LeftButton)
                    settle()
                    capture('01_all_settings')
                require_unchanged_settings(before_disclosure, screen._settings_model.collect())
                buttons = visible_test_data_buttons()
            if len(buttons) != 1:
                raise RuntimeError(f'Expected one visible test-data control, got {len(buttons)}')
            button = buttons[0]
            if hasattr(screen, '_settings_scroll'):
                screen._settings_scroll.ensureWidgetVisible(button)
                settle()
            already_cached = any((stage / 'example_data/plate1').glob('*.tif'))
            loading_frame = '02_cached_load' if already_cached else '02_download'
            QTimer.singleShot(800, lambda: capture(loading_frame))
            choice = {}
            if args.module == 'map_barcodes':
                from capture_sequencing import schedule_picker
                sequence_choice = schedule_picker(app, captures, capture, settle,
                                                  write_json, args.timeout)
            if args.module in {'annotate', 'classify_merged'}:
                def choose_test_data():
                    from spacr.qt.widgets.test_data_chooser import TestDataChooser
                    dialogs = [d for d in app.topLevelWidgets()
                               if isinstance(d, TestDataChooser) and d.isVisible()]
                    if len(dialogs) != 1:
                        choice['error'] = 'Expected exactly one test-data chooser'
                        for d in dialogs:
                            d.reject()
                        return
                    dialog = dialogs[0]
                    try:
                        options = {str(b.property('routeKey')): b for b in dialog.findChildren(QAbstractButton)
                                   if b.property('routeKey')}
                        for route in ('load', 'stream'):
                            QTest.mouseMove(options[route])
                            settle()
                            capture(f'02_data_choice_{route}')
                        QTest.mouseClick(options[args.test_data_route], Qt.LeftButton)
                        if dialog.chosen != args.test_data_route:
                            raise RuntimeError('The actual test-data route was not selected')
                        choice['selected_route'] = dialog.chosen
                    except Exception as error:
                        choice['error'] = str(error)
                        dialog.reject()
                QTimer.singleShot(1200, choose_test_data)
            QTest.mouseClick(button, Qt.LeftButton)
            if args.module == 'map_barcodes' and (
                    sequence_choice.get('error') or not sequence_choice.get('paired_read_identities_match')):
                raise RuntimeError(f'The archive example was not verified: {sequence_choice}')
            if args.module in {'annotate', 'classify_merged'}:
                write_json(captures / 'test_data_choice.json', choice)
                if choice.get('error') or choice.get('selected_route') != args.test_data_route:
                    raise RuntimeError(f'Test-data selection failed: {choice}')
            deadline = time.monotonic() + args.timeout
            while isValid(button) and not button.isEnabled():
                if time.monotonic() > deadline:
                    raise TimeoutError('Download did not finish')
                settle(0.2)
            settle(2)
            # Applying the example settings may replace the entire screen.
            # Never keep driving the detached pre-download screen: its preview
            # can run successfully while the recording shows another widget.
            screen = window._screens[args.module]
            if not screen.isVisible():
                raise RuntimeError('The current module screen is not visible after loading data')
            capture('03_data_ready')
            if hasattr(screen, '_settings_model'):
                settings = screen._settings_model.collect()
                write_json(captures / 'settings.json', settings)
            if args.module == 'mask':
                images = list((stage / 'example_data/plate1').glob('*.tif'))
                if not images:
                    raise RuntimeError('The UI did not download any real images')
                write_json(captures / 'dataset.json', {
                    'image_count': len(images), 'images': [p.name for p in images],
                    'bytes': sum(p.stat().st_size for p in images)})
        if args.preview and args.module == 'measure':
            import numpy as np
            panel = screen._measure_preview
            screen._preview_switch.setChecked(True)
            settle()
            deadline = time.monotonic() + args.timeout
            while panel._data is None or panel._loads_in_flight:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Measure preview did not load: {panel._status.text()}')
                settle(0.2)
            if not panel._crops:
                raise RuntimeError(f'Measure preview has no visible crops: {panel._status.text()}')
            capture('04_live_crops')
            QTest.mouseClick(panel._settings_btn, Qt.LeftButton)
            settle()
            dialog = panel._crop_settings_dialog
            dialog.resize(1350, 1350)
            dialog.move(window.mapToGlobal(QPoint(60, 110)))
            tabs = dialog.findChild(QTabWidget)
            settle()
            capture('05_crop_general')
            tabs.setCurrentIndex(1)
            settle()
            capture('06_crop_options')
            # The archived example has normalization off and its low-valued
            # channels are nearly black. Show the actual crop control making
            # them inspectable; this is not brightness editing of an image.
            if not panel._normalise.isChecked():
                QTest.mouseClick(panel._normalise, Qt.LeftButton)
                settle()
                deadline = time.monotonic() + args.timeout
                while panel._loads_in_flight:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Normalised crop preview did not finish')
                    settle(0.2)
            capture('06_normalised_crops')
            tabs.setCurrentIndex(2)
            settle()
            capture('07_crops_before')
            def crop_rows():
                return [{'label': int(c['label']), 'area': int(c['area']),
                         'category': c.get('category'), 'included': bool(c.get('included', True))}
                        for c in panel._crops]
            before = crop_rows()
            original = panel._min_sizes['cell'].value()
            threshold = int(np.median([c['area'] for c in before])) + 1
            source_hash = hashlib.sha256(panel._data.tobytes()).hexdigest()
            if args.preview_variants:
                if panel._propagate_btn.isChecked():
                    QTest.mouseClick(panel._propagate_btn, Qt.LeftButton)
                def set_area(value):
                    control = panel._min_sizes['cell']
                    control.setFocus()
                    QTest.keyClick(control, Qt.Key_A, Qt.ControlModifier)
                    QTest.keyClicks(control, str(value))
                    QTest.keyClick(control, Qt.Key_Enter)
                    settle(0.3)
                    deadline = time.monotonic() + args.timeout
                    while panel._loads_in_flight:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Live recropping did not finish')
                        settle(0.2)
                    settle()
                main_before = screen._settings_model.collect()['cell_min_size']
                set_area(threshold)
                after = crop_rows()
                capture('08_crops_filtered')
                if not 0 < len(after) < len(before):
                    raise RuntimeError('The live filter did not visibly reduce the nonempty crop grid')
                if screen._settings_model.collect()['cell_min_size'] != main_before:
                    raise RuntimeError('Preview changed batch settings while propagation was off')
                QTest.mouseClick(panel._propagate_btn, Qt.LeftButton)
                settle()
                if screen._settings_model.collect()['cell_min_size'] != threshold:
                    raise RuntimeError('Propagate settings did not reach the real batch form')
                capture('09_crops_propagated')
                set_area(original)
                restored = crop_rows()
                if restored != before:
                    raise RuntimeError('Restoring the filter did not restore the same crops')
                QTest.mouseClick(panel._propagate_btn, Qt.LeftButton)
                capture('10_crops_restored')
                if hashlib.sha256(panel._data.tobytes()).hexdigest() != source_hash:
                    raise RuntimeError('Filtering unexpectedly modified the loaded array')
                write_json(captures / 'live_variants.json', {
                    'source': panel._data_path, 'source_sha256': source_hash,
                    'shape': list(panel._data.shape), 'minimum_area_before': original,
                    'minimum_area_after': threshold, 'before': before, 'after': after,
                    'restored': restored, 'source_unchanged': True,
                    'propagation_off_preserved_batch': True, 'propagation_on_updated_batch': True})
            dialog.close()
            settle()
        elif args.preview:
            if args.module != 'mask':
                raise ValueError('The current preview capture supports Mask and Measure')
            panel = screen._live_preview
            screen._preview_switch.setChecked(True)
            settle()
            if not panel.isVisible():
                ancestors = []
                ancestor = panel
                while ancestor is not None:
                    ancestors.append({'class': type(ancestor).__name__,
                                      'name': ancestor.objectName(),
                                      'visible': ancestor.isVisible(),
                                      'hidden': ancestor.isHidden(),
                                      'size': [ancestor.width(), ancestor.height()]})
                    ancestor = ancestor.parentWidget()
                write_json(captures / 'preview_visibility.json', ancestors)
                capture('04_preview_visibility_failure')
                raise RuntimeError('Live preview is not visible after enabling Live: '
                                   f'checked={screen._preview_switch.isChecked()}, '
                                   f'card={screen._preview_card_attr}')
            deadline = time.monotonic() + 90
            while getattr(panel, '_image', None) is None:
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Preview image not loaded: {panel._status.text()}')
                settle(0.1)
            settle(1)
            capture('04_live_image')
            panel.open_live_settings()
            settle(1)
            dialog = panel._live_settings_dialog
            dialog.resize(1650, 1100)
            dialog.move(window.mapToGlobal(QPoint(50, 120)))
            settle()
            capture('05_live_settings')
            dialog.close()
            settle()
            QTest.mouseClick(panel._run_btn, Qt.LeftButton)
            preview_error = []
            panel._worker.finished_masks.connect(
                lambda masks, error, token: preview_error.append(error) if error else None)
            settle(0.5)
            capture('06_preview_running')
            deadline = time.monotonic() + args.timeout
            while not panel._raw_masks or (panel._worker and panel._worker.isRunning()):
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Preview not finished: {panel._status.text()}')
                # Thread completion can precede delivery of the GUI's queued
                # result signal. A stopped thread alone is not a failed run.
                if preview_error:
                    raise RuntimeError(f'Preview failed: {preview_error[0]}')
                settle(0.2)
            settle(1)
            capture('07_preview_result')
            import numpy as np
            outputs = {}
            for name, mask in panel._raw_masks.items():
                output = captures / f'preview_{name}.npy'
                np.save(output, mask)
                outputs[name] = {'shape': list(mask.shape),
                                 'objects': int(np.count_nonzero(np.unique(mask))),
                                 'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
            if not any(item['objects'] for item in outputs.values()):
                raise RuntimeError('Preview completed but detected no objects')
            write_json(captures / 'preview_outputs.json', outputs)
            if args.preview_variants:
                panel.open_live_settings()
                settle()
                dialog = panel._live_settings_dialog
                dialog.resize(1650, 1100)
                dialog.move(window.mapToGlobal(QPoint(50, 120)))
                settle()
                minimum = panel._compartment_widgets['cell']['min_area']
                if not minimum.isVisible():
                    raise RuntimeError('Cannot demonstrate a hidden minimum-area control')
                original = minimum.value()
                raw = panel._raw_masks['cell']
                labels, areas = np.unique(raw[raw > 0], return_counts=True)
                cutoff = int(np.median(areas)) + 1
                before = int(np.count_nonzero(np.unique(panel._masks['cell'])))
                raw_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                worker_before = panel._worker
                capture('08_filters_before')
                minimum.setFocus()
                minimum.selectAll()
                QTest.keyClicks(minimum, str(cutoff))
                QTest.keyClick(minimum, Qt.Key_Tab)
                settle(1)
                after = int(np.count_nonzero(np.unique(panel._masks['cell'])))
                if not 0 < after < before:
                    raise RuntimeError(f'Live area filter had no demonstrated selective effect: {before} -> {after}')
                if panel._worker is not worker_before or hashlib.sha256(panel._raw_masks['cell'].tobytes()).hexdigest() != raw_hash:
                    raise RuntimeError('The filter changed/recomputed the raw segmentation')
                capture('09_filters_after')
                minimum.setValue(original)
                settle()
                restored = int(np.count_nonzero(np.unique(panel._masks['cell'])))
                if restored != before:
                    raise RuntimeError('Restoring the area threshold did not restore the original result')
                capture('10_filters_restored')
                available_models = [panel._model_box.itemText(i) for i in range(panel._model_box.count())]
                old_diameter = panel._diameter.value()
                panel._diameter.setFocus()
                panel._diameter.selectAll()
                QTest.keyClicks(panel._diameter, str(old_diameter * 2))
                QTest.keyClick(panel._diameter, Qt.Key_Tab)
                capture('11_model_diameter')
                QTest.mouseClick(dialog._run_btn, Qt.LeftButton)
                settle(0.3)
                deadline = time.monotonic() + args.timeout
                while panel._worker and panel._worker.isRunning():
                    if time.monotonic() > deadline:
                        raise TimeoutError('Model-option comparison did not finish')
                    settle(0.2)
                settle()
                variant = panel._raw_masks['cell']
                if hashlib.sha256(variant.tobytes()).hexdigest() == raw_hash:
                    raise RuntimeError('The model-option comparison did not change the segmentation')
                capture('12_model_result')
                np.save(captures / 'preview_cell_diameter_variant.npy', variant)
                write_json(captures / 'live_variants.json', {
                    'filter': {'minimum_area': cutoff, 'original_minimum_area': original,
                               'before': before, 'after': after, 'restored': restored,
                               'raw_mask_unchanged': True, 'model_rerun': False},
                    'model_option': {'model': panel._model_box.currentText(),
                                     'available_models': available_models,
                                     'diameter_before': old_diameter,
                                     'diameter_after': panel._diameter.value(),
                                     'objects_after': int(np.count_nonzero(np.unique(variant))),
                                     'array_sha256': hashlib.sha256(variant.tobytes()).hexdigest(),
                                     'rerun_completed': True},
                })
                dialog.close()
        if args.run:
            model = screen._settings_model
            if args.module == 'measure' and model.collect().get('normalize'):
                # Show the real scientific caveat, then choose No. Crops may
                # be brightened for inspection without teaching irreversible
                # per-crop intensity rescaling as a default for saved data.
                warning_seen = []
                def decline_crop_warning():
                    for box in app.topLevelWidgets():
                        if isinstance(box, QMessageBox) and box.isVisible():
                            if box.windowTitle() != 'Check the crop settings':
                                box.reject()
                                return
                            box.setMinimumWidth(1000)
                            box.resize(1100, 1250)
                            settle(0.2)
                            capture('19_crop_normalization_warning')
                            warning_seen.append(box.text())
                            QTest.mouseClick(box.button(QMessageBox.No), Qt.LeftButton)
                QTimer.singleShot(500, decline_crop_warning)
                QTest.mouseClick(screen._btn_run, Qt.LeftButton)
                if not warning_seen or getattr(screen, '_worker', None) is not None:
                    raise RuntimeError('The crop warning did not cancel the trial before running')
                write_json(captures / 'crop_normalization_warning.json', {
                    'shown': warning_seen[0], 'choice': 'No', 'pipeline_started': False})
            presets = {'mask': {'test_mode': True, 'test_images': 2, 'batch_size': 2,
                       'n_jobs': 2, 'randomize': False, 'plot': True,
                       'examples_to_plot': 1, 'cell_diameter': 30,
                       'nucleus_diameter': 30, 'pathogen_diameter': 15},
                       'measure': {'test_mode': True, 'test_nr': 1, 'n_jobs': 1,
                                   'plot': True, 'save_measurements': True, 'save_png': True,
                                   'normalize': False},
                       'map_barcodes': {'n_jobs': 2, 'chunk_size': 1000,
                                        'test': False, 'mode': 'paired', 'save_h5': True},
                       'umap': {'src': str(Path.home() / '.cache/spacr/example_data/plate1'),
                                'row_limit': 400, 'n_jobs': 2, 'random_seed': 42,
                                'n_neighbors': 15, 'min_dist': 0.1, 'min_samples': 5,
                                'remove_cluster_noise': False, 'image_nr': 12,
                                'img_zoom': 0.3, 'plot_images': True, 'plot_points': True,
                                'save_figure': True},
                       'classify_merged': {
                           'classifier_family': 'cv', 'dataset_mode': 'annotation',
                           'classes': {
                               'infected_1': {'column': 'infected', 'value': 1},
                               'infected_2': {'column': 'infected', 'value': 2}},
                           'model_type': 'resnet18', 'epochs': 1, 'batch_size': 8,
                           'image_size': 128, 'n_jobs': 2, 'init_weights': False,
                           'tensorboard': False, 'augment': False,
                           'gradient_accumulation': False, 'train': True, 'test': True,
                           'generate_training_dataset': True,
                           'apply_model_to_dataset': False, 'generate_full_dataset': False,
                           'plot': True, 'cv_group_by': 'well', 'random_seed': 42,
                           'evaluation_fail_on_leakage': True,
                           'leakage_audit_train_test': True,
                           'leakage_hash_content': True, 'leakage_require_identity': True}}
            if args.module == 'regression':
                run_parent = stage / 'regression_runs'
                run_parent.mkdir(parents=True, exist_ok=True)
                # Never inherit the app's saved project output path. The
                # downloaded score/count CSVs stay intact in the private cache.
                destination = tempfile.mkdtemp(prefix='example-', dir=run_parent)
                presets['regression'] = {
                    'src': destination, 'inference': 'nonparametric',
                    'analysis_unit': 'well', 'guide_permutations': 199,
                    'guide_min_wells': [2], 'guide_permutation_seed': 0,
                    'level': 'both', 'annotation_source': 'none',
                }
            if args.module not in presets:
                raise ValueError('No bounded recording preset for this module')
            bounded = presets[args.module]
            for key, value in bounded.items():
                if not model.set_value_for_key(key, value):
                    raise RuntimeError(f'Cannot configure the real {key} control')
            settings = model.collect()
            for key, value in bounded.items():
                if settings.get(key) != value:
                    raise RuntimeError(f'The UI did not retain {key}={value}')
            write_json(captures / 'batch_settings.json', settings)
            if args.settings_tour:
                from capture_settings import record_settings
                record_settings(screen, captures, capture, settle, write_json)
            if getattr(screen, '_preview_switch', None) is not None:
                screen._preview_switch.setChecked(False)
            settle()
            capture('20_batch_settings')
            def reject_unexpected_prompt():
                for box in app.topLevelWidgets():
                    if isinstance(box, QMessageBox) and box.isVisible():
                        capture('21_unexpected_run_prompt')
                        write_json(captures / 'unexpected_prompt.json', {
                            'title': box.windowTitle(), 'text': box.text(), 'accepted': False})
                        box.reject()
            QTimer.singleShot(1000, reject_unexpected_prompt)
            QTest.mouseClick(screen._btn_run, Qt.LeftButton)
            worker = getattr(screen, '_worker', None)
            if worker is None:
                capture('21_batch_not_started')
                raise RuntimeError('The Run button did not start a pipeline worker')
            outcome = {'finished': False, 'ok': False, 'errors': []}
            def finished(ok):
                outcome.update(finished=True, ok=bool(ok))
            worker.finished.connect(finished)
            worker.error.connect(lambda text: outcome['errors'].append(str(text)))
            settle(1)
            capture('21_batch_running')
            deadline = time.monotonic() + args.timeout
            next_frame = time.monotonic() + 20
            while not outcome['finished'] or screen._worker_thread_is_running():
                if time.monotonic() > deadline:
                    QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
                    settle(3)
                    raise TimeoutError('Bounded pipeline exceeded the recording time limit')
                if time.monotonic() >= next_frame:
                    capture('22_batch_progress')
                    next_frame = time.monotonic() + 30
                settle(0.2)
            settle(2)
            write_json(captures / 'batch_outcome.json', outcome)
            blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
            write_json(captures / 'batch_console.json', blocks)
            write_json(captures / 'settings_after_run.json', screen._settings_model.collect())
            for block, _, _ in screen._console._pipeline_console_blocks():
                # Real text selection/navigation, including an internally
                # scrollable console block, not a replacement transcript.
                block.setFocus()
                QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
            screen._console.jump_to_the_end()
            settle()
            capture('23_batch_finished')
            if not outcome['ok'] or outcome['errors']:
                raise RuntimeError('The real pipeline failed; see batch_outcome.json')
            queue = screen._figure_queue
            requires_figure = args.module != 'map_barcodes'
            if requires_figure and queue.count() < 1:
                raise RuntimeError('Plot was enabled but the run produced no inspectable figure')
            figures = []
            for index, pixmap in enumerate(queue.all_pixmaps()):
                path = captures / f'batch_figure_{index:02d}.png'
                if not pixmap.save(str(path), 'PNG'):
                    raise RuntimeError(f'Could not preserve figure {index}')
                figures.append({'image': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            write_json(captures / 'batch_figures.json', figures)
            acceptance = assess_pipeline(outcome, blocks, len(figures), requires_figure=requires_figure)
            write_json(captures / 'batch_acceptance.json', acceptance)
            if not acceptance['accepted']:
                raise RuntimeError('Recording is not a successful complete example: '
                                   + '; '.join(acceptance['reasons']))
            if queue.count():
                queue.show_index(queue.count() - 1)
                settle()
                capture('24_batch_figure')
            if args.module == 'map_barcodes':
                from capture_sequencing import inspect_mapping
                write_json(captures / 'mapping_outputs.json',
                           inspect_mapping(Path(settings['src']), sequence_choice['run'], 10000))
            if args.settings_tour and args.module == 'regression':
                from capture_settings import record_results
                record_results(screen, captures, capture, settle, write_json)
            if args.settings_tour and args.module == 'umap':
                from capture_umap import record_explorer
                record_explorer(screen, captures, capture, settle, write_json)
        if args.annotation_tour:
            from capture_annotate import record_annotation
            record_annotation(app, window, screen, stage, captures, capture,
                              settle, write_json, args.timeout)
        if args.ai_controls:
            # Show the genuine control and a draft; never submit a provider
            # request or imply that an AI response was generated.
            if not hasattr(screen, '_ai_switch'):
                raise RuntimeError('This screen has no AI console control')
            screen._ai_switch.setChecked(True)
            settle()
            screen._console._input.setPlainText(
                'Explain the latest results and suggest which settings I should inspect. Do not change anything.')
            capture('30_ai_unsent_question')
            write_json(captures / 'ai_demo.json', {'prompt_submitted': False,
                       'response_generated': False, 'toggle_enabled': screen._ai_switch.isChecked()})
            screen._console._input.clear()
            screen._ai_switch.setChecked(False)
        if args.run:
            # Opening result tabs can launch a reader after the main worker
            # finished. Do not accept a lesson that then logs a late failure.
            settle(2)
            blocks = [text for _, _, text in screen._console._pipeline_console_blocks()]
            write_json(captures / 'batch_console_after_tour.json', blocks)
            final_acceptance = assess_pipeline(outcome, blocks, screen._figure_queue.count(),
                                                requires_figure=requires_figure)
            write_json(captures / 'batch_acceptance.json', final_acceptance)
            if not final_acceptance['accepted']:
                raise RuntimeError('The completed result tour exposed an error: '
                                   + '; '.join(final_acceptance['reasons']))
            if args.module == 'classify_merged':
                from capture_classify import inspect_database_split
                generated = [line.partition('Generated Train set: ')[2]
                             for block in blocks for line in block.splitlines()
                             if line.startswith('Generated Train set: ')]
                if len(generated) != 1:
                    raise RuntimeError('Cannot identify the actual generated classifier dataset')
                source = Path(settings['src'][0])
                proof = inspect_database_split(source / 'measurements/measurements.db',
                                               Path(generated[0]).parent)
                write_json(captures / 'scientific_acceptance.json', proof)
                if not proof['accepted']:
                    raise RuntimeError(proof['reason'])
    write_json(captures / 'provenance.json', {'commit': inventory['commit'],
               'version': inventory['version'], 'module': args.module,
               'download_requested': args.download, 'dataset_cache': str(stage / 'example_data'),
               'app_source_modified': False, 'cache_isolated_with_bind_mount': True,
               'completed_capture': True})
    window.close()
    settle(0.2)
    app.quit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
