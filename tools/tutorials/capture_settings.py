"""Record real settings searches and result tabs without changing the analysis.

These are recorder actions, not application patches. Every search must expose
the named real setting, and the complete settings mapping must survive the tour.
"""
from __future__ import annotations

import sys
import hashlib
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parent))   # tests load drivers by path

import capture_ui
from capture_ui_drv_a import Rec, search_typed



def _same_setting(actual, wanted):
    """Equal as the editor holds it: a class rule value is typed text, so 1 and '1' are the same value."""
    if isinstance(wanted, dict) and isinstance(actual, dict) and set(wanted) == set(actual):
        return all(_same_setting(actual[k], wanted[k]) for k in wanted)
    if isinstance(wanted, (int, float)) and isinstance(actual, str) and not isinstance(wanted, bool):
        return actual.strip() == str(wanted)
    return actual == wanted

def import_capture_settings(app, screen, stage, settings, capture, settle, *,
                            filename="invasion_settings.csv"):
    """Import the prepared configuration through the actual settings picker."""
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialog, QFileDialog, QLineEdit, QDialogButtonBox
    from spacr.cli import write_settings_file

    window, app_key = screen.window(), screen.app_key
    path = _Path(stage) / filename
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError('The private capture settings destination must be fresh')
    write_settings_file(settings, path)
    if not screen._btn_import.isVisible() or not screen._btn_import.isEnabled():
        raise ValueError('The actual Import settings button is not available')
    accepted, errors = [], []
    active = True
    watchdog = QTimer(window)
    watchdog.setSingleShot(True)

    def fail(message):
        errors.append(message)
        dialog = app.activeModalWidget()
        if isinstance(dialog, QDialog):
            dialog.reject()

    def choose():
        if not active:
            return
        dialog = app.activeModalWidget()
        try:
            if not isinstance(dialog, QFileDialog):
                raise ValueError('The real Import settings file picker did not open')
            edit = dialog.findChild(QLineEdit, 'fileNameEdit')
            if edit is None:
                raise ValueError('The actual settings picker has no filename field')
            dialog.accepted.connect(lambda: accepted.append(True))
            capture_ui.focus(capture, edit)
            QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(edit, str(path))
            capture_ui.accept_file_dialog(capture, dialog,
                dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Open))
        except Exception as error:
            fail(str(error))

    watchdog.timeout.connect(lambda: fail('The actual Import settings picker timed out'))
    capture_ui.when_modal(capture, 350, choose, timeout=60)
    watchdog.start(90000)
    try:
        QTest.mouseClick(screen._btn_import, Qt.LeftButton)
    finally:
        active = False
        watchdog.stop()
        watchdog.deleteLater()
    settle(.3)
    if errors or not accepted:
        raise ValueError('; '.join(errors) or 'The settings picker was not accepted')
    current = getattr(window, '_screens', {}).get(app_key, screen)
    actual = current._settings_model.collect()
    mismatch = {key: dict(requested=value, actual=actual.get(key))
                for key, value in settings.items() if not _same_setting(actual.get(key), value)}
    if mismatch:
        raise ValueError('The genuine settings import did not apply the requested values: ' + str(mismatch))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), actual_file_picker=True,
                requested_settings_read_back=len(settings))


def require_unchanged_settings(before, after):
    if before != after:
        changed = sorted(key for key in before.keys() | after.keys()
                         if before.get(key) != after.get(key))
        raise RuntimeError(f'A display-only tutorial tour changed settings: {changed}')


def record_settings(screen, captures, capture, settle, write_json, *, extra_keys=()):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    keys_by_module = {
        'regression': ('paired_data', 'inference', 'level', 'analysis_unit',
                       'guide_min_wells', 'guide_permutations',
                       'guide_permutation_seed', 'regression_backend',
                       'annotation_source'),
        'classify_merged': ('classifier_family', 'dataset_mode', 'classes',
                            'image_source', 'model_type', 'epochs', 'batch_size',
                            'image_size', 'n_jobs', 'init_weights', 'cv_group_by',
                            'test', 'plot', 'apply_model_to_dataset'),
        'umap': ('src', 'tables', 'reduction_method', 'row_limit',
                 'n_neighbors', 'min_dist', 'random_seed', 'clustering', 'plot_images'),
        'recruitment': ('src', 'target', 'channel_of_interest', 'channel_dims',
                       'cell_chann_dim', 'pathogen_plate_metadata',
                       'cell_size_range', 'nucleus_size_range', 'pathogen_size_range',
                       'nuclei_limit', 'pathogen_limit', 'cells_per_well',
                       'cell_intensity_range', 'nucleus_intensity_range',
                       'pathogen_intensity_range', 'plot', 'plot_nr'),
    }
    if screen.app_key not in keys_by_module:
        raise ValueError('No measured settings tour is configured for this module')
    before = screen._settings_model.collect()
    if screen.app_key == 'classify_merged' and before.get('classifier_family') == 'ml':
        keys_by_module['classify_merged'] = (
            'classifier_family', 'dataset_mode', 'classes', 'model_type_ml',
            'channel_of_interest', 'n_estimators', 'test_size', 'cv_group_by',
            'cross_validation', 'n_repeats', 'prune_features', 'plot')
    splitter = screen._body_splitter
    sizes = splitter.sizes()
    # The same resize the user can make: leave both live panels visible.
    capture_ui.splitter_sizes(capture, splitter, [sum(sizes) // 2, sum(sizes) - sum(sizes) // 2])
    bar = screen._settings_search
    old_query, old_level, old_modified = bar.query(), bar.level(), bar.modified_only()
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    observations = []
    try:
        for key in dict.fromkeys((*extra_keys, *keys_by_module[screen.app_key])):
            # Host: the clip starts before the pointer clicks the search box,
            # so the viewer sees the name typed and the field it finds.
            with Rec(capture, f'19_setting_{key}', settle) as scene:
                search_typed(capture, bar, key)
                settle()
                if key not in bar.visible_keys():
                    raise RuntimeError(f'Search did not expose {key}')
                field = screen._settings_model._widgets[key]
                capture_ui.scroll_to(capture, screen._settings_scroll, field)
                if screen._settings_scroll.horizontalScrollBar().value():
                    capture_ui.scroll_to_value(capture, screen._settings_scroll, 0, 'horizontal')
                settle()
                if not field.isVisible():
                    raise RuntimeError(f'The searched {key} field is hidden')
                scene.point(field, dwell=1.5)
            observations.append({'query': key, 'visible_keys': bar.visible_keys(),
                                 'value': screen._settings_model.collect().get(key)})
    finally:
        # Put the form back the way it was found, through the same controls.
        capture_ui.search_settings(capture, bar, old_query)
        capture_ui.settings_level(capture, bar, old_level)
        capture_ui.settings_modified_only(capture, bar, old_modified)
        capture_ui.scroll_to_value(capture, screen._settings_scroll, 0)
        capture_ui.scroll_to_value(capture, screen._settings_scroll, 0, 'horizontal')
        settle()
    after = screen._settings_model.collect()
    require_unchanged_settings(before, after)
    write_json(captures / 'settings_tour.json', {
        'display_only': True, 'settings_unchanged': True,
        'before_splitter_sizes': sizes, 'after_splitter_sizes': splitter.sizes(),
        'observations': observations})


def record_results(screen, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    tabs = screen._results_tabs
    if tabs is None:
        raise RuntimeError('There are no real result tabs to demonstrate')
    before = screen._settings_model.collect()
    # The wider form was useful for setup. Give the real result surfaces
    # their space back for inspection, using the application's splitter.
    width = sum(screen._body_splitter.sizes())
    observations = []
    for title in ('Runs', 'Results', 'Measurements', 'Cells'):
        matches = [i for i in range(tabs.count()) if tabs.tabText(i) == title]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one {title} tab')
        index = matches[0]
        # Host: the first clip also shows the splitter dragged to give the
        # result tabs their space; every clip shows the tab clicked.
        with Rec(capture, f'25_result_{title.lower()}', settle):
            if not observations:
                capture_ui.splitter_sizes(capture, screen._body_splitter, [width // 4, width - width // 4])
                settle()
            QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                             pos=tabs.tabBar().tabRect(index).center())
            settle(1)
            if tabs.currentIndex() != index:
                raise RuntimeError(f'The {title} tab was not selected')
        observations.append({'title': title, 'selected_index': index})
    results_index = next(i for i in range(tabs.count()) if tabs.tabText(i) == 'Results')
    QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                     pos=tabs.tabBar().tabRect(results_index).center())
    settle()
    detail_tabs = screen._results_panel.tabs
    detail_observations = []
    # Family selection adds suffixes to these labels. Resolve the actual
    # public result surface, then click its real tab; never guess by text.
    for title, page in (('p-values', screen._results_panel.p_values),
                        ('Q-Q', screen._results_panel.qq),
                        ('Coefficients', screen._results_panel.table)):
        index = detail_tabs.indexOf(page)
        if index < 0:
            raise RuntimeError(f'The result detail surface is not mounted: {title}')
        with Rec(capture, '26_detail_' + title.lower().replace('-', '_'), settle):
            QTest.mouseClick(detail_tabs.tabBar(), Qt.LeftButton,
                             pos=detail_tabs.tabBar().tabRect(index).center())
            settle()
            if detail_tabs.currentIndex() != index:
                raise RuntimeError(f'Detail tab did not open: {title}')
        detail_observations.append({'surface': title, 'displayed_label': detail_tabs.tabText(index)})
    require_unchanged_settings(before, screen._settings_model.collect())
    write_json(captures / 'results_tour.json', {
        'settings_unchanged': True, 'tabs': observations, 'detail_tabs': detail_observations,
        'new_analysis_requested': False,
        'measurement_database_attached': any(
            row.get('database') for row in before.get('paired_data', []))})
