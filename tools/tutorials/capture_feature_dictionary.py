"""Record the actual Help dictionary, without pretending it runs an analysis."""
from __future__ import annotations

from pathlib import Path
import time

import capture_ui
import capture_ui_drv_d as drv


def verify_detail(doc, text, key):
    if doc is None or doc.key != key:
        raise ValueError('The actual detail pane shows a different feature')
    for field in ('description', 'unit', 'module', 'computed_by'):
        value = getattr(doc, field)
        if value and value not in text:
            raise ValueError(f'The actual detail pane omitted {field}')
    return {'key': doc.key, 'title': doc.title, 'unit': doc.unit,
            'module': doc.module, 'computed_by': doc.computed_by,
            'object_type_count': len(doc.object_types), 'detail_characters': len(text)}


def record_dictionary(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QAbstractScrollArea, QComboBox
    from spacr.qt.widgets.feature_dictionary import FeatureDictionaryDialog

    deadline = time.monotonic() + timeout
    acceptance = Path(captures) / 'dictionary_acceptance.json'
    proof = {'lesson': '62_feature_dictionary', 'accepted': False,
             'published': False, 'dataset_loaded': False, 'analysis_run': False,
             'export_performed': False, 'ai_request_sent': False,
             'scope': 'Built-in definitions and genuine Help controls, not project feature availability'}
    dialog = None

    def tick():
        if time.monotonic() > deadline:
            raise TimeoutError('The bounded Feature Dictionary capture timed out')

    def click(widget):
        tick()
        if not widget.isVisible() or not widget.isEnabled():
            raise RuntimeError('The requested dictionary control is not usable')
        target = widget.viewport() if isinstance(widget, QAbstractScrollArea) else widget
        QTest.mouseClick(target, Qt.LeftButton)
        settle(.15)

    def fill(widget, text):
        if capture_ui.live(capture):
            if text:
                capture_ui.set_text(capture, widget, text, commit=False)
            else:
                h = capture_ui._h()
                h.click(widget)
                h.key(widget, Qt.Key_A, Qt.ControlModifier)
                h.key(widget, Qt.Key_Backspace)
            settle(.3)
            return
        click(widget)
        QTest.keyClick(widget, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClicks(widget, text)
        if not text:
            QTest.keyClick(widget, Qt.Key_Backspace)
        settle(.3)

    def combo(widget, text):
        index = widget.findText(text)
        if index < 0:
            raise RuntimeError('The actual dictionary filter has no requested option')
        click(widget)
        view = widget.view()
        if not view.isVisible():
            raise RuntimeError('The actual dictionary dropdown did not open')
        QTest.keyClick(view, Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(view, Qt.Key_Down)
        QTest.keyClick(view, Qt.Key_Return)
        settle(.3)
        if widget.currentText() != text:
            raise RuntimeError('The actual dictionary filter selected a different option')

    def scroll_detail(panel, *, bottom):
        bar = panel._detail.verticalScrollBar()
        before = bar.value()
        target = bar.maximum() if bottom else bar.minimum()
        if capture_ui.live(capture):
            capture_ui.scroll_to_value(capture, panel._detail, target)
            close = abs(bar.value() - target) <= max(2, bar.singleStep())
        else:
            QTest.keyClick(bar, Qt.Key_End if bottom else Qt.Key_Home)
            close = bar.value() == target
        settle(.2)
        if bar.maximum() <= 0:
            # The whole entry fits in the detail pane (a 2560 px host window shows it all), so there is nothing to
            # scroll and everything the narration points at is already visible. Record that instead of failing.
            proof.setdefault('scroll_checks', []).append({'before': before, 'after': bar.value(), 'maximum': 0,
                                                          'bottom': bottom, 'nothing_to_scroll': True})
            return
        if not close:
            raise ValueError('The native detail scrollbar did not reach its requested position '
                             '(value %d, target %d, maximum %d, step %d)' % (bar.value(), target, bar.maximum(), bar.singleStep()))
        proof.setdefault('scroll_checks', []).append({'before': before,
            'after': bar.value(), 'maximum': bar.maximum(), 'bottom': bottom})

    try:
        help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = help_action.menu()
        choices = [a for a in menu.actions() if a.text().replace('&', '') == 'Feature Dictionary…']
        if len(choices) != 1:
            raise RuntimeError('The real Help menu has no unique dictionary action')
        scene1 = drv.begin(capture, settle, '01_help_dictionary')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton,
                         pos=window.menuBar().actionGeometry(help_action).center())
        settle(.2)
        if not menu.isVisible():
            raise RuntimeError('The actual Help menu did not open')
        capture_ui.wait(capture, 1.5, settle)
        drv.snap(capture, '01_help_dictionary')
        drv.finish(capture, scene1)
        scene2 = drv.begin(capture, settle, '02_actual_dictionary')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(choices[0]).center())
        settle(.6)
        dialogs = [w for w in app.topLevelWidgets() if isinstance(w, FeatureDictionaryDialog) and w.isVisible()]
        if len(dialogs) != 1:
            raise RuntimeError('The real non-modal dictionary did not open')
        dialog = dialogs[0]
        if not capture_ui.live(capture):
            dialog.resize(3000, 1850)
            dialog.move(window.geometry().center() - dialog.rect().center())
        settle(.3)
        panel = dialog.panel
        proof['dialog_modal'] = dialog.isModal()
        proof['visible_dropdowns'] = [w.objectName() for w in dialog.findChildren(QComboBox) if w.isVisible()]
        proof['visible_buttons'] = [w.text() for w in dialog.findChildren(QAbstractButton) if w.isVisible()]
        proof['initial_features'] = len(panel.result_keys())
        capture_ui.point_at(capture, panel._search, panel._concept, panel._object)
        drv.snap(capture, '02_actual_dictionary')
        drv.finish(capture, scene2)
        proof['states'] = {}

        def snapshot(name, key, act=None):
            started = drv.begin(capture, settle, name)
            if act is not None:
                act()
            detail = panel.detail_text()
            data = verify_detail(panel.current_doc(), detail, key)
            data.update(query=panel._search.text(), concept=panel._concept.currentText(),
                        object_filter=panel._object.currentText(), result_count=len(panel.result_keys()),
                        detail_text=detail)
            proof['states'][name] = data
            capture_ui.point_at(capture, panel._detail)
            drv.snap(capture, name)
            drv.finish(capture, started)

        snapshot('03_search_by_idea', 'area', act=lambda: fill(panel._search, 'size'))
        def filters():
            fill(panel._search, '')
            combo(panel._concept, 'intensity')
            combo(panel._object, 'cell')
            if not panel.result_keys():
                raise ValueError('The actual intensity/cell filters returned no features')
            if any('cell' not in hit.doc.object_types or 'intensity' not in hit.doc.concepts for hit in panel._hits):
                raise ValueError('The actual filtered definitions do not satisfy both selected filters')
        with capture_ui.clip(capture, '04_actual_concept_object_filters'):
            filters()
            capture_ui.point_at(capture, panel._concept, panel._object)
        proof['filtered_results'] = len(panel.result_keys())
        def by_name():
            combo(panel._concept, 'Any concept')
            combo(panel._object, 'Any object')
            fill(panel._search, 'cell_channel_1_mean_intensity')
        snapshot('05_measurement_name', 'mean_intensity', act=by_name)
        snapshot('06_computing_function', 'mean_intensity', act=lambda: scroll_detail(panel, bottom=True))
        def area_top():
            fill(panel._search, 'cell_area')
            scroll_detail(panel, bottom=False)
        snapshot('07_conditional_area_units', 'area', act=area_top)
        snapshot('08_area_caveats', 'area', act=lambda: scroll_detail(panel, bottom=True))
        # A failed lookup is not a claim that a feature cannot exist in data.
        with capture_ui.clip(capture, '09_unknown_search'):
            fill(panel._search, 'zzzz_not_a_feature_zzzz')
            if panel.result_keys() or panel.current_doc() is not None:
                raise ValueError('The unmatched lookup retained a stale feature definition')
            proof['unmatched_lookup'] = panel.detail_text()
        snapshot('10_known_lookup_restored', 'area', act=lambda: fill(panel._search, 'cell_area'))
        copy_clip = drv.begin(capture, settle, '11_actual_text_copy')
        click(panel._detail)
        QTest.keyClick(panel._detail, Qt.Key_A, Qt.ControlModifier)
        QTest.keyClick(panel._detail, Qt.Key_C, Qt.ControlModifier)
        settle(.2)
        selected = panel._detail.textCursor().selection().toPlainText()
        clipboard = app.clipboard().text()
        proof['copy_diagnostic'] = {'selection_characters': len(selected),
                                    'clipboard_characters': len(clipboard)}
        if not selected or clipboard != selected:
            raise ValueError('The actual Copy gesture did not copy the selected definition')
        proof['private_xvfb_clipboard_copy'] = {'characters': len(clipboard), 'exact_selection': True}
        capture_ui.wait(capture, 1.0, settle)
        drv.snap(capture, '11_actual_text_copy')
        drv.finish(capture, copy_clip)
        proof['accepted'] = True
    except Exception as error:
        proof['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(acceptance, proof)
        if dialog is not None:
            capture_ui.close_dialog(capture, dialog)
            settle(.1)
