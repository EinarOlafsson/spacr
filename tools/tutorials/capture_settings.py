"""Record real settings searches and result tabs without changing the analysis.

These are recorder actions, not application patches. Every search must expose
the named real setting, and the complete settings mapping must survive the tour.
"""
from __future__ import annotations


def require_unchanged_settings(before, after):
    if before != after:
        changed = sorted(key for key in before.keys() | after.keys()
                         if before.get(key) != after.get(key))
        raise RuntimeError(f'A display-only tutorial tour changed settings: {changed}')


def record_settings(screen, captures, capture, settle, write_json):
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
    splitter.setSizes([sum(sizes) // 2, sum(sizes) - sum(sizes) // 2])
    bar = screen._settings_search
    old_query, old_level, old_modified = bar.query(), bar.level(), bar.modified_only()
    if bar.modified_only():
        QTest.mouseClick(bar._modified, Qt.LeftButton)
    if bar.level() != 'all':
        QTest.mouseClick(bar._disclosure, Qt.LeftButton)
    observations = []
    try:
        for key in keys_by_module[screen.app_key]:
            bar._input.setFocus()
            QTest.keyClick(bar._input, Qt.Key_A, Qt.ControlModifier)
            QTest.keyClicks(bar._input, key)
            settle()
            if key not in bar.visible_keys():
                raise RuntimeError(f'Search did not expose {key}')
            field = screen._settings_model._widgets[key]
            screen._settings_scroll.ensureWidgetVisible(field)
            screen._settings_scroll.horizontalScrollBar().setValue(0)
            settle()
            if not field.isVisible():
                raise RuntimeError(f'The searched {key} field is hidden')
            capture(f'19_setting_{key}')
            observations.append({'query': key, 'visible_keys': bar.visible_keys(),
                                 'value': screen._settings_model.collect().get(key)})
    finally:
        bar.set_query(old_query)
        bar.set_level(old_level)
        bar.set_modified_only(old_modified)
        screen._settings_scroll.verticalScrollBar().setValue(0)
        screen._settings_scroll.horizontalScrollBar().setValue(0)
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
    screen._body_splitter.setSizes([width // 4, width - width // 4])
    settle()
    observations = []
    for title in ('Runs', 'Results', 'Measurements', 'Cells'):
        matches = [i for i in range(tabs.count()) if tabs.tabText(i) == title]
        if len(matches) != 1:
            raise RuntimeError(f'Expected exactly one {title} tab')
        index = matches[0]
        QTest.mouseClick(tabs.tabBar(), Qt.LeftButton,
                         pos=tabs.tabBar().tabRect(index).center())
        settle(1)
        if tabs.currentIndex() != index:
            raise RuntimeError(f'The {title} tab was not selected')
        capture(f'25_result_{title.lower()}')
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
        QTest.mouseClick(detail_tabs.tabBar(), Qt.LeftButton,
                         pos=detail_tabs.tabBar().tabRect(index).center())
        settle()
        if detail_tabs.currentIndex() != index:
            raise RuntimeError(f'Detail tab did not open: {title}')
        capture('26_detail_' + title.lower().replace('-', '_'))
        detail_observations.append({'surface': title, 'displayed_label': detail_tabs.tabText(index)})
    require_unchanged_settings(before, screen._settings_model.collect())
    write_json(captures / 'results_tour.json', {
        'settings_unchanged': True, 'tabs': observations, 'detail_tabs': detail_observations,
        'new_analysis_requested': False,
        'measurement_database_attached': any(
            row.get('database') for row in before.get('paired_data', []))})
