"""Record the Alpha lesson: every control behind Preferences > Show alpha features.

The only recording that turns Show alpha features on (brief EXCEPTION,
2026-10-04). It turns it on in Preferences on camera, visits each screen that
holds alpha controls and shows them, then turns it off again. Show alpha
species stays off throughout. Nothing is run or downloaded: the frames show
where each control lives and what it is set to, not results.

Each group is recorded independently; a group whose control cannot be shown
is reported in ``alpha_tour.json`` instead of stopping the whole tour.
"""
from __future__ import annotations

import time

#: (frame, module, settings keys to expose, widget object names to show).
SETTINGS_GROUPS = (
    ('02_mask_preprocessing', 'mask',
     ('unmix', 'n2v_denoise', 'illumination_correction',
      'psf_operation', 'enhance_clahe'), ()),
    ('04_mask_quality', 'mask',
     ('robustness_report', 'real_object_classifier', 'image_qc_classifier'), ()),
    ('05_mask_automation', 'mask',
     ('mask_parallel', 'watch_folder', 'microscope_feedback', 'cloud_profile'), ()),
    ('07_measure_assays', 'measure',
     ('viability', 'cell_cycle', 'wound_closure', 'confluency'), ()),
    ('07b_measure_time', 'measure', ('bleach_correction', 'time_to_event'), ()),
    ('08_measure_outputs', 'measure',
     ('profiling', 'cellprofiler_pipeline', 'measure_gpu', 'measurement_backend',
      'intensity_calibration'), ()),
    ('08b_anndata_format', 'anndata_export', ('anndata_format',), ()),
    ('12_plaque_growth', 'analyze_plaques',
     ('colony_counting', 'plaque_estimate_growth', 'plaque_growth_reference_um',
      'plaque_growth_reference_hours'), ()),
    ('13_activation_counterfactuals', 'activation',
     ('counterfactuals', 'counterfactual_crops', 'counterfactual_target'),
     ('ActivationCounterfactualViewer',)),
    ('19_ops_spot_detectors', 'ops', ('ops_spot_detector',), ()),
)

#: (frame, module, widget object names that must be visible).
WIDGET_GROUPS = (
    ('09_make_masks_tools', 'make_masks',
     ('MakeMasksBlindToggle', 'MakeMasksRoisButton', 'MakeMasksUncertaintyButton',
      'MakeMasksSam2Button')),
    ('10_workbench_virtual_stain', 'train_cellpose', ('CellposeWorkbenchVirtualStain',)),
    ('11_annotate_tools', 'annotate', ('AnnotateBlindToggle', 'AnnotateFieldQCButton')),
    ('14_embeddings_tools', 'embeddings',
     ('EmbeddingsFoundationPicker', 'EmbeddingsDinoPretrainButton', 'EmbeddingsWellMilButton')),
    ('15_control_chart', 'control_chart',
     ('ControlChartAnomaly', 'ControlChartChemistry')),
    ('16_report_archive', 'report', ('ReportArchivePackage', 'ReportZenodoDeposit')),
    ('17_run_history_export', 'run_history', ('RunHistoryExportWorkflow',)),
    ('19b_map_barcodes_spatial', 'map_barcodes', ('MapBarcodesSpatialToggle',)),
    ('20_convert_barcodes', 'convert', ('ConvertPlateBarcodeLinkage',)),
    ('20b_power_arrayed', 'power', ('PowerArrayedPlanner',)),
    ('21_test_data', 'feature_explorer', ('FeatureExplorerTestDataButton',)),
)


#: Groups whose screen shows its rows through the settings search instead.
SEARCH_INSTEAD = {'13_activation_counterfactuals': 'counterfactual'}


def record_alpha_features(app, window, stage, captures, capture, settle, write_json):
    from PySide6.QtCore import QObject, QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import (QAbstractButton, QDialogButtonBox, QTabWidget,
                                   QWidget)

    from spacr.qt import preferences as prefs

    report = {'groups': {}, 'alpha_species_on': False}

    def rect_of(widget):
        top_left = widget.mapToGlobal(QPoint(0, 0)) - window.mapToGlobal(QPoint(0, 0))
        return [top_left.x(), top_left.y(), widget.width(), widget.height()]

    def open_module(key):
        window._on_nav_selected(key)
        deadline = time.monotonic() + 90
        while window._screens.get(key) is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{key} did not open')
            settle(0.1)
        settle(2)
        return window._screens[key]

    def preferences_toggle(on, frame):
        dialog = prefs.PreferencesDialog(window)
        try:
            toggle = dialog.findChild(QAbstractButton, 'ShowAlphaFutureFeatures')
            species = dialog.findChild(QAbstractButton, 'ShowAlphaSpecies')
            tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
            if toggle is None or tabs is None:
                raise RuntimeError('Preferences has no Show alpha features toggle')
            for index in range(tabs.count()):
                if tabs.widget(index).isAncestorOf(toggle):
                    tabs.setCurrentIndex(index)
                    break
            dialog.resize(1200, 1200)
            dialog.move(window.mapToGlobal(window.rect().center()) - dialog.rect().center())
            dialog.show()
            settle(0.8)
            if toggle.isChecked() != on:
                QTest.mouseClick(toggle, Qt.LeftButton,
                                 pos=toggle.visibleRegion().boundingRect().center())
                settle(0.8)
            if toggle.isChecked() != on:
                raise RuntimeError('The Show alpha features toggle did not change')
            if species is not None and species.isChecked():
                raise RuntimeError('Show alpha species must stay off')
            report['groups'][frame] = {'toggle': rect_of(toggle), 'dialog': rect_of(dialog)}
            capture(frame)
            box = dialog.findChild(QDialogButtonBox)
            ok = box.button(QDialogButtonBox.Save) or box.button(QDialogButtonBox.Ok)
            QTest.mouseClick(ok, Qt.LeftButton)
            settle(1.5)
        finally:
            dialog.close()
            dialog.deleteLater()
            settle(0.5)
        if prefs._get_show_alpha_features() != on:
            raise RuntimeError('Saving Preferences did not apply Show alpha features')

    def show_all_settings(screen):
        bar = screen._settings_search
        if bar.modified_only():
            QTest.mouseClick(bar._modified, Qt.LeftButton)
        if bar.level() != 'all':
            QTest.mouseClick(bar._disclosure, Qt.LeftButton)
        bar.set_query('')
        settle(0.6)

    def expose(screen, key):
        widget = screen._settings_model._widgets.get(key)
        if widget is None:
            raise RuntimeError(f'No {key} setting on {screen.app_key}')
        if widget.isHidden() and widget.parentWidget() is not None:
            widget = widget.parentWidget()
        parents, parent = [], widget.parentWidget()
        while parent is not None and parent is not screen:
            if hasattr(parent, 'is_expanded') and hasattr(parent, 'header'):
                parents.append(parent)
            parent = parent.parentWidget()
        for section in reversed(parents):
            if not section.is_expanded():
                screen._settings_scroll.ensureWidgetVisible(section.header())
                settle(0.2)
                section.header().click()
                settle(0.4)
            if not section.is_expanded():
                raise RuntimeError(f'{section.title()} did not open for {key}')
        return widget

    def time_switch(screen):
        switch = screen.dimension_switch('t') if hasattr(screen, 'dimension_switch') else None
        if switch is not None and switch.isVisible() and not switch.isChecked():
            QTest.mouseClick(switch, Qt.LeftButton)
            settle(1.0)

    # 1. Turn the preference on, on camera.
    preferences_toggle(True, '01_prefs_alpha_on')

    def settings_group(frame, module, keys, widgets):
        screen = open_module(module)
        show_all_settings(screen)
        if frame in ('06_timelapse', '07b_measure_time'):
            time_switch(screen)
            if module == 'mask':
                # Timelapse itself is a stored setting (no control of its own
                # once Time is on); a timelapse settings file sets it.
                screen.apply_settings_dict({'timelapse': True})
                settle(1.0)
        shown = []
        for key in keys:
            try:
                shown.append((key, expose(screen, key)))
            except RuntimeError as error:
                report['groups'].setdefault(frame, {}).setdefault('missing', []).append(str(error))
        if not shown:
            raise RuntimeError(f'No alpha setting of {frame} could be shown')
        # Opening a category can close its siblings; reopen the first key's.
        first = expose(screen, shown[0][0])
        if not first.isVisible() and frame in SEARCH_INSTEAD:
            screen._settings_search.set_query(SEARCH_INSTEAD[frame])
            settle(1.0)
        screen._settings_scroll.ensureWidgetVisible(first, 0, 300)
        screen._settings_scroll.horizontalScrollBar().setValue(0)
        settle(0.6)
        for name in widgets:
            for widget in window.findChildren(QWidget, name):
                if widget.isVisible():
                    report['groups'].setdefault(frame, {}).setdefault('widgets', {})[name] = rect_of(widget)
        report['groups'].setdefault(frame, {})['settings'] = {
            key: rect_of(widget) for key, widget in shown if widget.isVisible()}
        capture(frame)

    def widget_group(frame, module, names):
        screen = open_module(module)
        found = {}
        for name in names:
            matches = [w for w in window.findChildren(QObject, name)
                       if isinstance(w, QWidget) and w.isVisible()]
            if matches:
                found[name] = rect_of(matches[0])
        report['groups'][frame] = {'widgets': found,
                                   'missing': sorted(set(names) - set(found))}
        if not found:
            raise RuntimeError(f'No alpha control of {frame} is visible')
        capture(frame)

    groups = list(SETTINGS_GROUPS[:3]) + [('06_timelapse', 'mask',
               ('timelapse_mode', 'timelapse_lineage', 'timelapse_events'), ())] + list(SETTINGS_GROUPS[3:])
    for frame, module, keys, widgets in groups:
        try:
            settings_group(frame, module, keys, widgets)
        except Exception as error:  # report and keep touring
            report['groups'].setdefault(frame, {})['error'] = repr(error)
    # A genuine empty-folder watch exposes the card without inventing results.
    frame = '05b_watch_live_plate'
    watching = None
    try:
        watching = open_module('mask')
        source = stage / 'alpha_watch_empty'
        source.mkdir(exist_ok=False)
        before = watching._settings_model.collect()
        watching.apply_settings_dict({
            **before, 'src': str(source), 'watch_folder': True,
            'watch_pipeline': 'mask', 'watch_poll_seconds': 0.1,
            'watch_settle_seconds': 0.1, 'watch_idle_minutes': 0.2,
            'metadata_type': 'cellvoyager', 'channels': [0, 1, 2],
            'cell_channel': 0, 'nucleus_channel': 1, 'pathogen_channel': 2,
            'batch_size': 1, 'n_jobs': 1,
            'timelapse': False, 'z_stack': False, 't_stack': False,
            'microscope_feedback': False, 'plot': False,
        })
        settle(0.5)
        watching = window._screens['mask']
        applied = watching._settings_model.collect()
        if (not applied.get('watch_folder') or applied.get('src') != str(source)
                or applied.get('cell_channel') != 0):
            raise RuntimeError('The current Mask form did not accept the watch settings')
        QTest.mouseClick(watching._btn_run, Qt.LeftButton)
        ledger = source / 'spacr_watch/watch_ledger.json'
        deadline = time.monotonic() + 30
        while not ledger.exists() and watching._worker_thread_is_running():
            if time.monotonic() > deadline:
                raise TimeoutError('The empty-folder watch did not start')
            settle(0.1)
        if not ledger.exists() or not watching._watch_live_plate.isVisible():
            raise RuntimeError('The real watch did not expose its Live plate card')
        settle(0.5)
        report['groups'][frame] = {
            'card': rect_of(watching._watch_live_plate),
            'empty_acquisition_folder': True,
            'summary': watching._watch_live_plate._summary.text(),
            'ledger': str(ledger),
        }
        capture(frame)
        while watching._worker_thread_is_running():
            if time.monotonic() > deadline:
                raise TimeoutError('The empty-folder watch did not finish')
            settle(0.1)
        watching.apply_settings_dict(before)
        watching = None
    except Exception as error:
        report['groups'].setdefault(frame, {})['error'] = repr(error)
    finally:
        if watching is not None and watching._worker_thread_is_running():
            watching._request_cooperative_stop()
            deadline = time.monotonic() + 30
            while watching._worker_thread_is_running():
                if time.monotonic() > deadline:
                    raise TimeoutError('The private watch did not stop')
                settle(0.1)
    for frame, module, names in WIDGET_GROUPS:
        try:
            widget_group(frame, module, names)
        except Exception as error:
            report['groups'].setdefault(frame, {})['error'] = repr(error)
    # Make Masks' folded Model Zoo lists the alpha backends and models
    # (DINOCell among them) once Scan lists the catalogue (642, 404).
    frame = '09b_model_zoo_alpha_models'
    try:
        from spacr.qt.screens.model_zoo import _stem_version
        from spacr.settings import _is_alpha

        masks = open_module('make_masks')
        masks.open_folded('model_zoo')
        settle(2)
        zoo = masks.folded_screen('model_zoo')
        QTest.mouseClick(zoo._btn_scan, Qt.LeftButton)
        deadline = time.monotonic() + 120
        while zoo._table.rowCount() == 0 and time.monotonic() < deadline:
            settle(0.5)
        settle(2)
        table = zoo._table
        rows, names = [], []
        for group, (stem, pairs) in enumerate(zoo._groups):
            row = zoo._row_of_group(group)
            if row is None or table.isRowHidden(row):
                continue
            keys = [entry.key for _label, entry in pairs]
            if any(_is_alpha('models', key) or _is_alpha('models', _stem_version(entry)[0])
                   for key, (_label, entry) in zip(keys, pairs)):
                rows.append(row)
                names.append(stem)
        if 'dinocell' not in names:
            raise RuntimeError(f'DINOCell is not among the alpha rows: {names}')
        backends = sorted(r for r, n in zip(rows, names) if '_' not in n)
        table.scrollToItem(table.item(backends[0], 0), table.ScrollHint.PositionAtTop)
        settle(1)
        top = table.visualItemRect(table.item(backends[0], 0))
        bottom = table.visualItemRect(table.item(backends[-1], 0))
        viewport = table.viewport()
        origin = viewport.mapToGlobal(QPoint(0, top.top())) - window.mapToGlobal(QPoint(0, 0))
        report['groups'][frame] = {
            'alpha_rows': names, 'focus_rows': [table.item(r, 0).text() for r in backends],
            'focus': [origin.x(), origin.y(), viewport.width(), bottom.bottom() - top.top() + 1],
            'table': rect_of(table)}
        capture(frame)
    except Exception as error:
        report['groups'].setdefault(frame, {})['error'] = repr(error)
    # Preferences tabs that only exist with alpha on: Notifications, plugins.
    try:
        dialog = prefs.PreferencesDialog(window)
        tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
        for name, frame in (('NotifyTabHelp', '18_prefs_notifications'),
                            ('PluginCatalogueHelp', '18b_prefs_plugins')):
            label = dialog.findChild(QWidget, name)
            if label is None:
                report['groups'].setdefault(frame, {})['error'] = f'no {name}'
                continue
            for index in range(tabs.count()):
                if tabs.widget(index).isAncestorOf(label):
                    tabs.setCurrentIndex(index)
                    break
            dialog.resize(1200, 1200)
            dialog.move(window.mapToGlobal(window.rect().center()) - dialog.rect().center())
            dialog.show()
            settle(0.8)
            capture(frame)
        dialog.close()
        dialog.deleteLater()
        settle(0.5)
    except Exception as error:
        report['groups'].setdefault('18_prefs_notifications', {})['error'] = repr(error)
    write_json(captures / 'alpha_tour.json', report)
    # Turn it off again, over the last screen.
    try:
        preferences_toggle(False, '22_prefs_alpha_off')
    except Exception as error:
        report['groups'].setdefault('22_prefs_alpha_off', {})['error'] = repr(error)
    report['alpha_species_on'] = bool(getattr(prefs, '_get_show_alpha_species', lambda: False)())
    write_json(captures / 'alpha_tour.json', report)
