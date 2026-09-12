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
    parser.add_argument('--settings-tour', action='store_true', help='Show bounded analysis choices through real settings searches')
    parser.add_argument('--annotation-tour', action='store_true', help='Record actual crop labelling and view changes in a new example column')
    parser.add_argument('--mask-editor-tour', action='store_true', help='Record actual reversible mask-editing gestures on private real-data copies')
    parser.add_argument('--editor-detect', action='store_true', help='Also run actual Cellpose once on the small recropped example')
    parser.add_argument('--font-scale', type=float, default=1.5, help='Use the actual app font preference for the recording')
    parser.add_argument('--capture-name', help='Preserve earlier accepted frames in a separate capture directory')
    parser.add_argument('--methods-review-export', action='store_true', help='Record unchanged Methods export plus explicit review findings; never approve its draft')
    parser.add_argument('--hit-list-companion', action='store_true', help='Explicit external Hit List window after showing the hidden native panel; not a shortcut fix')
    parser.add_argument('--graph-review-handoff', action='store_true', help='Verify native graphs and explicitly record the broken annotation handoff, without repairing it')
    parser.add_argument('--mask-bounded-recapture', action='store_true', help='Use explicit 0.4 flow thresholds in a fresh two-field Mask recording; not a quality claim')
    parser.add_argument('--mask-saved-plots', action='store_true', help='Show verified per-file API overlays in the external image viewer; no model rerun')
    parser.add_argument('--cellpose-training-review', action='store_true', help='Show the native source-import defect and a separately verified API training figure; never start GUI training')
    parser.add_argument('--plaque-zoo-model', action='store_true', help='Try the actual plaque Model Zoo download and preview in private staging')
    parser.add_argument('--motility-screen-export-probe', action='store_true', help='Diagnose the real Screen PDF export preference; not a production tutorial workaround')
    parser.add_argument('--manager-execute', action='store_true', help='Demonstrate confirmed cleanup/archive on the independently verified private Data Manager clone')
    parser.add_argument('--test-data-route', choices=('load', 'stream'), default='load', help='Choose the real Annotate/Classify test-data route')
    parser.add_argument('--classifier-family', choices=('cv', 'ml'), default='cv', help='Choose the real merged Classify workflow')
    parser.add_argument('--classifier-existing-split', type=Path, help='Reuse the explicitly prepared, metadata-verified tutorial split; never rebuild it from legacy filenames')
    parser.add_argument('--classify-overview', action='store_true', help='Record only native family choices and nested Classify navigation; never start a model')
    parser.add_argument('--model-zoo-inventory', action='store_true', help='Record actual Model Zoo inventory/provenance only; no download, training or benchmark')
    parser.add_argument('--measure-full-example', action='store_true', help='Measure the sixteen downloaded fields in normal mode, not redirected test mode')
    parser.add_argument('--measure-preview-controls', action='store_true', help='Record only visible Measure field/channel controls, restoring saved-crop normalization before exit')
    parser.add_argument('--anndata-api-introduction', action='store_true', help='Record only the AnnData GUI route/settings before the separately verified API workaround')
    parser.add_argument('--barcode-saved-plots', action='store_true', help='Show independently verified Barcode QC PNGs in the actual external viewer; no claim of GUI figure repair')
    parser.add_argument('--activation-saved-plots', action='store_true', help='Show independently verified Activation PNG grids in the real external viewer; does not certify GUI figures')
    parser.add_argument('--napari-reopen-each-edit', action='store_true', help='Record the explicit close/reopen-between-imports workflow; does not certify repeated edits in one viewer')
    parser.add_argument('--diagnostics-from', type=Path, help='Existing private tutorial regression project to inspect')
    parser.add_argument('--evaluation-from', type=Path, help='Private prepared known-overlap classifier evaluation bundle')
    parser.add_argument('--sweep-from', type=Path, help='Replay this verified private two-trial sweep without refitting')
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.model_zoo_inventory and (args.module != 'model_zoo' or args.run or args.download or args.preview):
        parser.error('--model-zoo-inventory requires model_zoo without run/download/preview')
    if args.preview_variants and not args.preview:
        parser.error('--preview-variants requires --preview')
    if args.hit_list_companion and args.module != 'hit_list':
        parser.error('--hit-list-companion requires --module hit_list')
    if args.graph_review_handoff and args.module != 'graph_builder':
        parser.error('--graph-review-handoff requires --module graph_builder')
    if args.methods_review_export and (args.module != 'methods_export' or args.run or args.download):
        parser.error('--methods-review-export requires methods_export without a run or download')
    if args.mask_bounded_recapture and (args.module != 'mask' or not args.run or not args.download):
        parser.error('--mask-bounded-recapture requires mask with --download and --run')
    if args.mask_saved_plots and (args.module != 'mask' or args.run or args.download or args.preview):
        parser.error('--mask-saved-plots requires mask without a new run, download or preview')
    if args.cellpose_training_review and (args.module != 'train_cellpose' or args.run or args.download or args.preview):
        parser.error('--cellpose-training-review requires train_cellpose without a new run, download or preview')
    if args.measure_full_example and (args.module != 'measure' or not args.run):
        parser.error('--measure-full-example requires --module measure --run')
    if args.measure_preview_controls and (args.module != 'measure' or not args.download or args.preview or args.run):
        parser.error('--measure-preview-controls requires --module measure --download without --preview/--run')
    if args.classifier_existing_split and (args.module != 'classify_merged' or args.classifier_family != 'cv' or not args.run):
        parser.error('--classifier-existing-split requires --module classify_merged --classifier-family cv --run')
    if args.classify_overview and (args.module != 'classify_merged' or args.run or args.download or args.classifier_existing_split):
        parser.error('--classify-overview requires classify_merged without a run or download')
    if args.anndata_api_introduction and args.module != 'anndata_export':
        parser.error('--anndata-api-introduction requires --module anndata_export')
    if args.barcode_saved_plots and args.module != 'barcode_qc':
        parser.error('--barcode-saved-plots requires --module barcode_qc')
    if args.activation_saved_plots and args.module != 'activation':
        parser.error('--activation-saved-plots requires --module activation')
    if args.napari_reopen_each_edit and args.module != 'napari_bridge':
        parser.error('--napari-reopen-each-edit requires --module napari_bridge')
    if args.settings_tour and (args.module not in {'regression', 'classify_merged', 'umap', 'recruitment'} or not args.run):
        parser.error('--settings-tour requires --module regression/classify_merged/umap/recruitment --run')
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
        queue_binds = []
        manager_binds = []
        if args.module in ('lineage', 'image_scatter'):
            from lineage_data import prepare
            state = stage / (args.module + '_state') / f'{args.capture_name or args.module}.json'
            if state.exists():
                raise RuntimeError('Use a new capture name for a fresh private Lineage example')
            prepared = prepare(stage, run_directory=args.module + '_runs',
                               copy_format_markers=args.module == 'image_scatter')
            write_json(state, prepared)
            private_cache = Path(prepared['cache'])
        if args.module == 'pipeline_graph':
            from pipeline_graph_data import prepare
            state = stage / 'pipeline_graph_state' / f'{args.capture_name or args.module}.json'
            if state.exists():
                raise RuntimeError('Use a new capture name for a fresh private Pipeline Graph example')
            prepared = prepare(stage)
            write_json(state, prepared)
            manager_binds = ['--ro-bind', prepared['source'], prepared['original_readonly'],
                             '--bind', prepared['clone'], prepared['source']]
        if args.module == 'project_browser':
            from project_browser_data import prepare
            state = stage / 'project_browser_state' / f'{args.capture_name or args.module}.json'
            if state.exists():
                raise RuntimeError('Use a new capture name for a fresh private Project Browser example')
            prepared = prepare(stage)
            write_json(state, prepared)
            manager_binds = ['--ro-bind', prepared['source'], prepared['original_readonly'],
                             '--bind', prepared['clone'], prepared['source']]
        if args.module == 'data_manager':
            from manager_data import prepare
            state = stage / 'data_manager_state' / f'{args.capture_name or args.module}.json'
            if state.exists():
                raise RuntimeError('Use a new capture name for a fresh private Data Manager project')
            prepared = prepare(stage)
            write_json(state, prepared)
            # Keep the original readable at an immutable alias, then shadow
            # only its original pathname with the verified disposable copy.
            # Existing artifact/project paths remain valid without rewriting
            # their registry or confusing a copied project with a new run.
            manager_binds = ['--ro-bind', prepared['source'], prepared['original_readonly'],
                             '--bind', prepared['clone'], prepared['source']]
        if args.module == 'queue':
            # Queue persists outside XDG_CONFIG_HOME. Isolate the whole app
            # state directory before constructing Home/Queue, without changing
            # Path.home or replacing the application's persistence function.
            queue_state = stage / 'queue_state' / (args.capture_name or 'queue')
            queue_state.mkdir(parents=True, exist_ok=True)
            if (queue_state / 'queue.json').exists():
                raise RuntimeError('Use a new capture name for a fresh private queue')
            queue_binds = ['--bind', str(queue_state), str(Path.home() / '.spacr')]
        os.environ['SPACR_TUTORIAL_CACHE_ISOLATED'] = '1'
        os.execvp('bwrap', ['bwrap', '--die-with-parent', '--bind', '/', '/',
                          *(['--unshare-net'] if args.module == 'distributed_jobs' else []),
                          '--dev-bind', '/dev', '/dev',
                          *queue_binds,
                          *manager_binds,
                          '--bind', str(private_cache), str(destination),
                          '--bind', str(private_runs), str(runs_destination), '--',
                          sys.executable, str(Path(__file__).resolve()),
                          *sys.argv[1:]])
    for key, value in {
        'QT_QPA_PLATFORM': args.platform, 'QT_SCALE_FACTOR': '1',
        'QT_AUTO_SCREEN_SCALE_FACTOR': '0', 'QT_FONT_DPI': '96',
        'SPACR_LANGUAGE': 'en', 'XDG_CONFIG_HOME': str(stage / 'config' /
            ((args.capture_name or args.module) if args.module in ('project_browser', 'lineage', 'image_scatter', 'motility', 'classifier_evaluation') else args.module)),
        'SPACR_EXAMPLE_DATA': str(stage / 'example_data'),
        'SPACR_LOG_DIR': str(stage / 'logs'),
        'MPLCONFIGDIR': str(stage / 'mpl'),
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
    }.items():
        os.environ[key] = value
    if args.module == 'distributed_jobs':
        remote_state = stage / 'distributed_state' / (args.capture_name or args.module)
        remote_state.mkdir(parents=True, exist_ok=True)
        if (remote_state / 'profiles.json').exists() or (remote_state / 'jobs.json').exists():
            raise RuntimeError('Use a fresh capture name for isolated distributed profiles')
        os.environ['SPACR_REMOTE_STATE_DIR'] = str(remote_state)
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
    if args.module in ('regression', 'queue', 'train_cellpose'):
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
        from capture_geometry import capture_rect
        return capture_rect(widget, window)

    def capture(name, *, desktop=False):
        pixmap = app.primaryScreen().grabWindow(0) if desktop else window.grab()
        if (pixmap.width(), pixmap.height()) != (3840, 2160):
            raise RuntimeError(f'Unexpected capture size {pixmap.size()}')
        painter = QPainter(pixmap)
        dialogs = [] if desktop else [w for w in app.topLevelWidgets()
                   if isinstance(w, (QDialog, QMenu)) and w.isVisible()]
        from capture_geometry import foreground_dialogs
        dialogs = foreground_dialogs(dialogs, app.activeModalWidget(), app.activePopupWidget())
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
    if args.module == 'db_browser':
        # The retained Database narration is still accurate. Capture its
        # current Help route and real controls without pre-opening it through
        # the private navigation slot or regenerating any voice track.
        from capture_database import record_database
        screen = record_database(app, window, None, stage, captures, capture,
                                 settle, write_json, args.timeout)
    elif args.module == 'distributed_jobs':
        from capture_distributed import record_distributed
        record_distributed(app, window, stage, captures, capture, settle, write_json)
    elif args.module == 'curate':
        from capture_curate import record_curate
        record_curate(app, window, stage, captures, capture,
                      settle, write_json, args.timeout)
    elif args.module == 'cellpose_masks':
        from capture_cellpose_masks import record_apply
        record_apply(app, window, stage, captures, capture,
                     settle, write_json, args.timeout)
    elif args.module == 'train_cellpose':
        if args.cellpose_training_review:
            from capture_cellpose_training_review import record
            record(app, window, stage, captures, capture, settle, write_json, args.timeout)
        else:
            from capture_cellpose_training import record_training
            record_training(app, window, stage, captures, capture,
                            settle, write_json, args.timeout)
    elif args.module == 'napari_bridge':
        from capture_napari import record_napari
        record_napari(app, window, stage, captures, capture,
                      settle, write_json, args.timeout,
                      reopen_each_edit=args.napari_reopen_each_edit)
    elif args.module == 'explain_cv':
        from capture_explain_cv import record_explain
        record_explain(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'profiler':
        from capture_profiler import record_profiler
        record_profiler(app, window, stage, captures, capture,
                        settle, write_json, args.timeout)
    elif args.module == 'parameter_sweep':
        from capture_parameter_sweep import record_sweep
        record_sweep(app, window, stage, captures, capture,
                     settle, write_json, args.timeout, existing=args.sweep_from)
    elif args.module == 'feature_dict':
        from capture_feature_dictionary import record_dictionary
        record_dictionary(app, window, stage, captures, capture,
                          settle, write_json, args.timeout)
    elif args.module == 'queue':
        from capture_plate_queue import record_queue
        record_queue(app, window, stage, captures, capture,
                     settle, write_json, args.timeout)
    elif args.module == 'batch':
        from capture_batch_runner import record_batch
        record_batch(app, window, stage, captures, capture,
                     settle, write_json, args.timeout)
    elif args.module == 'run_history':
        from capture_run_history import record_history
        record_history(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'run_compare':
        from capture_run_compare import record
        record(app, window, stage, captures, capture, settle, write_json, args.timeout)
    elif args.module == 'data_manager':
        from capture_data_manager import record_manager
        record_manager(app, window, stage, captures, capture,
                       settle, write_json, args.timeout, execute=args.manager_execute)
    elif args.module == 'project_browser':
        from capture_project_browser import record_browser
        record_browser(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'pipeline_graph':
        from capture_pipeline_graph import record_graph
        record_graph(app, window, stage, captures, capture,
                     settle, write_json, args.timeout)
    elif args.module == 'layer_viewer':
        from capture_layer_viewer import record_layers
        record_layers(app, window, stage, captures, capture,
                      settle, write_json, args.timeout)
    elif args.module == 'lineage':
        from capture_lineage import record_lineage
        record_lineage(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'volcano_explorer':
        from capture_volcano import record_volcano
        record_volcano(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'hit_list':
        from capture_hit_list import record_hits
        record_hits(app, window, stage, captures, capture,
                    settle, write_json, args.timeout, companion=args.hit_list_companion)
    elif args.module == 'methods_export':
        from capture_methods import record_methods
        record_methods(app, window, stage, captures, capture,
                       settle, write_json, args.timeout, review_export=args.methods_review_export)
    elif args.module == 'timelapse':
        from capture_timelapse import record_timelapse
        record_timelapse(app, window, stage, captures, capture,
                        settle, write_json, args.timeout)
    elif args.module == 'motility':
        from capture_motility import record_motility
        record_motility(app, window, stage, captures, capture,
                       settle, write_json, args.timeout,
                       probe_screen_export=args.motility_screen_export_probe)
    elif args.module == 'classifier_evaluation':
        from capture_evaluation import record_evaluation
        if args.evaluation_from is None:
            raise ValueError('The evaluation inspector requires --evaluation-from')
        record_evaluation(app, window, stage, captures, capture, settle, write_json,
                          args.timeout, args.evaluation_from)
    elif args.module == 'illumination':
        from capture_illumination import record_illumination
        record_illumination(app, window, stage, captures, capture,
                            settle, write_json, args.timeout)
    elif args.module == 'dose_response':
        from capture_dose_response import record_dose_response
        record_dose_response(app, window, stage, captures, capture,
                             settle, write_json, args.timeout)
    elif args.module == 'barcode_qc':
        from capture_barcode_qc import record_barcode_qc
        record_barcode_qc(app, window, stage, captures, capture,
                      settle, write_json, args.timeout, saved_plots=args.barcode_saved_plots)
    elif args.module == 'activation':
        from capture_activation import record_activation
        record_activation(app, window, stage, captures, capture,
                          settle, write_json, args.timeout, saved_plots=args.activation_saved_plots)
    elif args.module == 'image_scatter':
        from capture_image_scatter import record_scatter
        record_scatter(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'anndata_export':
        from capture_anndata import record_anndata
        record_anndata(app, window, stage, captures, capture,
                      settle, write_json, args.timeout, route_only=args.anndata_api_introduction)
    elif args.module == 'pca':
        from capture_pca import record_pca
        record_pca(app, window, stage, captures, capture,
                   settle, write_json, args.timeout)
    elif args.module == 'trellis':
        from capture_trellis import record_trellis
        record_trellis(app, window, stage, captures, capture,
                       settle, write_json, args.timeout)
    elif args.module == 'tabulate':
        from capture_tabulate import record_tabulate
        record_tabulate(app, window, stage, captures, capture,
                        settle, write_json, args.timeout)
    elif args.module == 'outliers':
        from capture_outliers import record_outliers
        record_outliers(app, window, stage, captures, capture,
                        settle, write_json, args.timeout)
    elif args.module == 'feature_explorer':
        from capture_feature_explorer import record_explorer
        record_explorer(app, window, stage, captures, capture,
                        settle, write_json, args.timeout)
    elif args.module == 'embeddings':
        from capture_embeddings import record_screen
        record_screen(app, window, stage, captures, capture, settle, write_json)
    elif args.module == 'ops':
        from capture_ops import record_screen
        record_screen(app, window, stage, captures, capture, settle, write_json)
    elif args.classify_overview:
        from capture_classify_overview import record_overview
        record_overview(app, window, captures, capture, settle, write_json)
    elif args.module != 'home':
        host_key = {'import_images': 'foreign', 'convert': 'foreign',
                    'agreement': 'annotate',
                    'external_masks': 'foreign', 'model_zoo': 'make_masks',
                    'train_compare': 'classify_merged',
                    'plate_view': 'graph_builder',
                    'control_chart': 'qc_dashboard',
                    'regression_diagnostics': 'regression'}.get(args.module, args.module)
        if args.module == 'report':
            # Report no longer has a Home tile. Record the actual Help menu
            # entry, rather than calling the navigation slot off camera.
            help_actions = [action for action in window.menuBar().actions()
                            if action.text().replace('&', '') == 'Help']
            if len(help_actions) != 1 or help_actions[0].menu() is None:
                raise RuntimeError('The current application has no unique Help menu')
            menu = help_actions[0].menu()
            choices = [action for action in menu.actions()
                       if action.text().replace('&', '') == 'Report']
            if len(choices) != 1 or not choices[0].isEnabled():
                raise RuntimeError('The current Help menu has no usable Report action')
            QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                             pos=window.menuBar().actionGeometry(help_actions[0]).center())
            settle(0.3)
            if not menu.isVisible():
                raise RuntimeError('The actual Help menu did not open')
            capture('00a_help_report_menu')
            QTest.mouseClick(menu, Qt.LeftButton,
                             pos=menu.actionGeometry(choices[0]).center())
        else:
            window._on_nav_selected(host_key)
        deadline = time.monotonic() + 60
        while window._screens.get(host_key) is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{args.module} did not open')
            settle(0.1)
        settle(2)
        screen = window._screens[host_key]
        capture('01_module')
        if args.module == 'agreement':
            from capture_agreement import record_agreement
            record_agreement(app, window, screen, stage, captures, capture,
                             settle, write_json, args.timeout)
        if args.module == 'replication':
            from capture_replication import record_replication
            record_replication(app, window, screen, stage, captures, capture,
                               settle, write_json, args.timeout)
        if args.module == 'invasion':
            from capture_invasion import record_invasion
            record_invasion(app, window, screen, stage, captures, capture,
                            settle, write_json, args.timeout)
        if args.module == 'analyze_plaques':
            from capture_plaque import record_plaque
            record_plaque(app, window, screen, stage, captures, capture,
                          settle, write_json, args.timeout, use_zoo_model=args.plaque_zoo_model)
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
        if args.module == 'foreign':
            from capture_foreign import record_foreign
            record_foreign(app, window, screen, stage, captures, capture,
                           settle, write_json, args.timeout)
        if args.module == 'graph_builder':
            from capture_graph import record_graph
            record_graph(app, window, screen, stage, captures, capture,
                         settle, write_json, args.timeout,
                         review_handoff=args.graph_review_handoff)
        if args.module == 'qc_dashboard':
            from capture_qc import record_qc
            record_qc(app, window, screen, stage, captures, capture,
                      settle, write_json, args.timeout)
        if args.module == 'experiment_design':
            from capture_design import record_design
            record_design(app, window, screen, stage, captures, capture,
                          settle, write_json, args.timeout)
        if args.module == 'power':
            from capture_power import record_power
            record_power(app, window, screen, stage, captures, capture,
                         settle, write_json, args.timeout)
        if args.module == 'gate_editor':
            from capture_gates import record_gates
            record_gates(app, window, screen, stage, captures, capture,
                         settle, write_json, args.timeout)
        if args.module == 'align':
            from capture_align import record_align
            record_align(app, window, screen, stage, captures, capture,
                         settle, write_json, args.timeout)
        if args.module == 'convert':
            from capture_converter import record_converter
            record_converter(app, window, screen, stage, captures, capture,
                             settle, write_json, args.timeout)
        if args.module == 'external_masks':
            from capture_external_masks import record_external_masks
            record_external_masks(app, window, screen, stage, captures, capture,
                                  settle, write_json, args.timeout)
        if args.module == 'model_zoo':
            if args.model_zoo_inventory:
                from capture_model_inventory import record_inventory
                record_inventory(app, window, screen, stage, captures, capture,
                                 settle, write_json, args.timeout)
            else:
                from capture_model_zoo import record_model_zoo
                record_model_zoo(app, window, screen, stage, captures, capture,
                                 settle, write_json, args.timeout)
        if args.module == 'train_compare':
            from capture_training_runs import record_training_runs
            record_training_runs(app, window, screen, stage, captures, capture,
                                 settle, write_json, args.timeout)
        if args.module == 'report':
            from capture_report import record_report
            record_report(app, window, screen, stage, captures, capture,
                          settle, write_json, args.timeout)
        if args.module == 'plate_view':
            from capture_plate_retention import record_plate_retention
            record_plate_retention(app, window, screen, stage, captures, capture,
                                   settle, write_json, args.timeout)
        if args.module == 'control_chart':
            from capture_control_chart import record_control_chart
            record_control_chart(app, window, screen, stage, captures, capture,
                                 settle, write_json, args.timeout)
        if args.mask_saved_plots:
            from capture_mask_saved_plots import record
            record(app, window, stage, captures, capture, settle, write_json)
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
        if args.measure_preview_controls:
            from capture_measure_controls import record_controls
            record_controls(app, window, screen, captures, capture, settle,
                            write_json, args.timeout)
        if args.preview and args.module == 'measure':
            import numpy as np
            # Native panel resizing, not screenshot enlargement: leave enough
            # vertical space to see the live crop categories and their images.
            if screen._usage_card.body.isVisible():
                QTest.mouseClick(screen._usage_card.title_label, Qt.LeftButton)
            screen._runtime_splitter.setSizes([1400, 300])
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
            settle(2)  # Crop metadata can arrive before the queued thumbnail paint.
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
            # The current dialog can exceed the desktop because its Crop modes
            # list includes every organelle. Move the genuine window, never
            # hide/reparent controls or paint a replacement panel. Keep evidence
            # of this limitation rather than imply the dialog fits normally.
            def expose_dialog_control(control):
                origin = control.mapToGlobal(QPoint(0, 0))
                available = app.primaryScreen().availableGeometry()
                if not available.contains(control.mapToGlobal(control.rect().bottomRight())) or origin.y() < available.top():
                    dialog.move(dialog.x(), dialog.y() + available.center().y() - origin.y())
                    settle()
                if not available.contains(control.mapToGlobal(control.rect().center())):
                    raise RuntimeError('Crop control remains outside the actual desktop')
                return {'dialog_position': [dialog.x(), dialog.y()],
                        'dialog_size': [dialog.width(), dialog.height()],
                        'control_screen_y': control.mapToGlobal(QPoint(0, 0)).y()}
            normalization_geometry = expose_dialog_control(panel._normalise)
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
            dialog.move(window.mapToGlobal(QPoint(60, 110)))
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
                propagation_geometry = expose_dialog_control(panel._propagate_btn)
                QTest.mouseClick(panel._propagate_btn, Qt.LeftButton)
                settle()
                if screen._settings_model.collect()['cell_min_size'] != threshold:
                    raise RuntimeError('Propagate settings did not reach the real batch form')
                capture('09_crops_propagated')
                dialog.move(window.mapToGlobal(QPoint(60, 110)))
                set_area(original)
                restored = crop_rows()
                if restored != before:
                    raise RuntimeError('Restoring the filter did not restore the same crops')
                expose_dialog_control(panel._propagate_btn)
                QTest.mouseClick(panel._propagate_btn, Qt.LeftButton)
                dialog.move(window.mapToGlobal(QPoint(60, 110)))
                capture('10_crops_restored')
                if hashlib.sha256(panel._data.tobytes()).hexdigest() != source_hash:
                    raise RuntimeError('Filtering unexpectedly modified the loaded array')
                write_json(captures / 'live_variants.json', {
                    'source': panel._data_path, 'source_sha256': source_hash,
                    'shape': list(panel._data.shape), 'minimum_area_before': original,
                    'minimum_area_after': threshold, 'before': before, 'after': after,
                    'restored': restored, 'source_unchanged': True,
                    'propagation_off_preserved_batch': True, 'propagation_on_updated_batch': True})
                write_json(captures / 'preview_dialog_framing.json', {
                    'normalization': normalization_geometry,
                    'propagation': propagation_geometry,
                    'actual_window_moved': True, 'application_layout_fixed': False,
                    'controls_hidden_or_reparented': False})
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
            if args.measure_full_example:
                presets['measure']['test_mode'] = False
            if args.mask_bounded_recapture:
                # The downloaded settings use 100, which the real console says
                # disables flow filtering. Show explicit bounded example values
                # instead, never silently teach 100 as a recommended default.
                presets['mask'].update(cell_flow_threshold=.4,
                    nucleus_flow_threshold=.4, pathogen_flow_threshold=.4)
            if args.module == 'classify_merged' and args.classifier_family == 'ml':
                from capture_classify_ml import bounded_settings
                presets['classify_merged'] = bounded_settings()
                write_json(captures / 'scientific_acceptance.json', {
                    'accepted': False, 'reason': 'Actual ML output identities not yet checked'})
            if args.classifier_existing_split:
                from classify_split_evidence import inspect_inputs
                input_proof = inspect_inputs(args.classifier_existing_split)
                write_json(captures / 'canonical_input_checks.json', input_proof)
                from capture_classify_existing import choose_existing_folder
                choose_existing_folder(app, screen, args.classifier_existing_split,
                                       capture, settle)
                write_json(captures / 'scientific_acceptance.json', {
                    'accepted': False, 'reason': 'The prepared split has not completed a verified native run'})
                presets['classify_merged'].pop('gradient_accumulation', None)
                presets['classify_merged'].update(
                    src=[str(args.classifier_existing_split.resolve())],
                    generate_training_dataset=False, n_jobs=0, val_split=0.5,
                    test_split=0.5, train_channels=['r', 'g', 'b'])
            if args.module == 'recruitment':
                from recruitment_data import prepare_subset
                source = WORKSPACE.parent / 'test_datasets/spacr/tutorials'
                run_parent = stage / 'recruitment_runs'
                run_parent.mkdir(parents=True, exist_ok=True)
                destination = Path(tempfile.mkdtemp(prefix='example-', dir=run_parent)) / 'project'
                write_json(captures / 'scientific_acceptance.json', {
                    'accepted': False, 'reason': 'Real Recruitment outputs not yet independently checked',
                    'published': False})
                manifest = prepare_subset(source, destination)
                write_json(captures / 'input_manifest.json', manifest)
                recorded = json.loads((REPO / 'tools/tutorials/authoring/catalog/25_recruitment_settings.json').read_text())
                # These old mask-plane controls no longer exist in Recruitment.
                # Keep the recorded raw-channel and biological metadata, without
                # treating those annotations as independently verified biology.
                for key in ('cell_mask_dim', 'nucleus_mask_dim', 'pathogen_mask_dim'):
                    recorded.pop(key, None)
                recorded.update(src=str(destination), channel_dims=[1],
                                cell_intensity_range=None, nucleus_intensity_range=[-1, 65536],
                                pathogen_intensity_range=[-1, 65536], plot=True,
                                plot_control=False, plot_nr=0,
                                cell_plate_metadata=[['c1', 'c2', 'c3']],
                                treatment_plate_metadata=[['c1', 'c2', 'c3']])
                presets['recruitment'] = recorded
            if args.module not in presets:
                raise ValueError('No bounded recording preset for this module')
            bounded = presets[args.module]
            for key, value in bounded.items():
                if not model.set_value_for_key(key, value):
                    raise RuntimeError(f'Cannot configure the real {key} control')
            settings = model.collect()
            write_json(captures / 'configured_settings.json', settings)
            for key, value in bounded.items():
                if settings.get(key) != value:
                    raise RuntimeError(f'The UI did not retain {key}={value}')
            write_json(captures / 'batch_settings.json', settings)
            if args.settings_tour:
                from capture_settings import record_settings
                record_settings(screen, captures, capture, settle, write_json,
                    extra_keys=('src', 'generate_training_dataset', 'val_split', 'train_channels')
                    if args.classifier_existing_split else ())
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
            if args.module == 'measure':
                from capture_settings import require_unchanged_settings
                before_tour = screen._settings_model.collect()
                if screen._usage_card.body.isVisible():
                    QTest.mouseClick(screen._usage_card.title_label, Qt.LeftButton)
                screen._runtime_splitter.setSizes([1400, 300])
                for index in (0, 1, 2):
                    queue.show_index(index)
                    settle()
                    capture(f'25_measure_figure_{index:02d}')
                screen._runtime_splitter.setSizes([300, 1400])
                screen._console._split.setSizes([1200, 100])
                for block, _, _ in screen._console._pipeline_console_blocks():
                    block.setFocus()
                    QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
                screen._console.jump_to_the_end()
                settle()
                capture('26_measure_console_complete')
                require_unchanged_settings(before_tour, screen._settings_model.collect())
                write_json(captures / 'readable_results_tour.json', {
                    'display_only': True, 'settings_unchanged': True,
                    'figure_indices': [0, 1, 2], 'console_complete_shown': True})
            if args.module == 'recruitment':
                # Preserve the genuine overlay and every calculated chart.
                # The archived masks are NOT asserted to be the postprocessed
                # masks behind the stored measurement rows.
                if 'Failed to plot images with outlines' in '\n'.join(blocks):
                    raise RuntimeError('The actual Recruitment overlay failed')
                if queue.count() < 5:
                    raise RuntimeError('Recruitment did not produce its overlay and four charts')
                width = sum(screen._body_splitter.sizes())
                screen._body_splitter.setSizes([width // 4, width - width // 4])
                screen._settings_scroll.horizontalScrollBar().setValue(0)
                if screen._usage_card.body.isVisible():
                    QTest.mouseClick(screen._usage_card.title_label, Qt.LeftButton)
                screen._runtime_splitter.setSizes([1200, 450])
                for index in range(queue.count()):
                    queue.show_index(index)
                    settle()
                    capture(f'25_recruitment_figure_{index:02d}')
                screen._runtime_splitter.setSizes([300, 1350])
                screen._console._split.setSizes([1200, 100])
                for block, _, _ in screen._console._pipeline_console_blocks():
                    block.setFocus()
                    QTest.keyClick(block, Qt.Key_End, Qt.ControlModifier)
                screen._console.jump_to_the_end()
                settle()
                capture('26_recruitment_console_counts')
                screen._runtime_splitter.setSizes([1200, 450])
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
            if args.classifier_existing_split:
                from classify_split_evidence import inspect_finished, inspect_metrics
                proof = inspect_finished(args.classifier_existing_split)
                proof['independent_test_metrics'] = inspect_metrics(args.classifier_existing_split)
                write_json(captures / 'scientific_acceptance.json', proof)
            elif args.module == 'classify_merged' and args.classifier_family == 'cv':
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
            if args.module == 'classify_merged' and args.classifier_family == 'ml':
                from capture_classify_ml import inspect_output
                sources = settings['src']
                source = Path(sources[0] if isinstance(sources, list) else sources)
                outputs = list((source / 'results/random_forest').glob('*/results.csv'))
                if len(outputs) != 1:
                    raise RuntimeError(f'Expected one fresh actual ML result, found {outputs}')
                proof = inspect_output(source / 'measurements/measurements.db', outputs[0],
                                       requested_fraction=settings['test_size'])
                write_json(captures / 'scientific_acceptance.json', proof)
                if not proof['accepted']:
                    raise RuntimeError('; '.join(proof['reasons']))
            if args.module == 'recruitment':
                from recruitment_evidence import inspect_results
                from recruitment_data import _sha256
                originals = [(Path(manifest['source_database']), manifest['source_database_sha256'])]
                originals += [(Path(row['source']), row['sha256']) for row in manifest['arrays']]
                unchanged = {str(path): _sha256(path) == digest for path, digest in originals}
                write_json(captures / 'source_preservation_after_run.json', unchanged)
                if not all(unchanged.values()):
                    raise RuntimeError('An original Recruitment input changed during the run')
                proof = inspect_results(Path(settings['src']), settings)
                write_json(captures / 'scientific_acceptance.json', proof)
                if not proof['accepted']:
                    raise RuntimeError(proof['reason'])
    write_json(captures / 'provenance.json', {'commit': inventory['commit'],
               'version': inventory['version'], 'module': args.module,
               'download_requested': args.download, 'dataset_cache': (
                   json.loads((stage / (args.module + '_state') / f'{args.capture_name or args.module}.json').read_text())['cache']
                   if args.module in ('lineage', 'image_scatter') else str(stage / 'example_data')),
               'app_source_modified': False, 'cache_isolated_with_bind_mount': True,
               'completed_capture': True})
    window.close()
    settle(0.2)
    app.quit()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
