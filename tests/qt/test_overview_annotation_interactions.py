"""Real annotation and comparison actions retain physical source identities."""
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QPushButton

from spacr import tabular
from spacr.qt.i18n import tr
from spacr.qt.widgets import qc_field_browser as browser


@pytest.fixture
def overview(qtbot, tmp_path):
    source = tmp_path / 'images'
    for plate, wells, count in [('plate1', ['A01', 'A02', 'B01', 'B02'], 7),
                               ('plate2', ['A01'], 2)]:
        for well in wells:
            for number in range(1, count + 1):
                folder = source / plate / well / f'field{number}'
                folder.mkdir(parents=True)
                tifffile.imwrite(folder / 'ch1.tif',
                                 np.arange(72, dtype=np.uint16).reshape(8, 9) + number)
    dialog = browser._FieldOverview(settings={'source': str(source), 'geometry': (2, 2),
                                              'channels': [1], 'normalize': False})
    runners = (dialog._index_jobs, dialog._image_jobs, dialog._export_jobs,
               dialog._thumbnail_jobs, dialog._montage_jobs, dialog._field_thumbnail_jobs)

    def retire(widget):
        widget._cancel()
        qtbot.waitUntil(lambda: all(r.active_jobs() == 0 for r in runners), timeout=10000)
        for runner in runners:
            runner.shutdown(timeout_ms=3000)

    qtbot.addWidget(dialog, before_close_func=retire)
    dialog._indexed(browser._overview_index(source))
    qtbot.waitUntil(lambda: dialog._normalization_ready and not dialog._index_jobs.is_busy(),
                    timeout=10000)
    return dialog


def _click(dialog, caption):
    button = next(b for b in dialog.findChildren(QPushButton) if b.text() == tr(caption))
    button.click()


def _choose(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def _idle(dialog, qtbot):
    qtbot.waitUntil(lambda: not dialog._export_jobs.is_busy(), timeout=10000)


@pytest.mark.parametrize('scope,expected', [('column', {'A02', 'B02'}),
                                           ('wells', {'A01', 'B02'})])
def test_column_and_selected_well_annotations_can_be_removed_without_touching_other_plates(
        overview, scope, expected):
    dialog = overview
    plate = dialog.plates.currentData()
    other = next(f.plate for f in dialog._fields if f.plate != plate)
    dialog._conditions[(other, 'A01')] = 'other plate'

    def select():
        _choose(dialog.annotation_scope, scope)
        if scope == 'column':
            _choose(dialog.columns, 2)
        else:
            for well in expected:
                dialog.wells.item(*dialog._well_positions[well]).setSelected(True)

    select()
    dialog.annotation_text.setText(' Drug A ')
    _click(dialog, 'Apply annotation')
    assert dialog._conditions == {(other, 'A01'): 'other plate',
                                  **{(plate, well): 'Drug A' for well in expected}}
    for well in expected:
        assert 'Drug A' in dialog.wells.item(*dialog._well_positions[well]).toolTip()
    select()
    dialog.annotation_text.setText('   ')
    _click(dialog, 'Apply annotation')
    assert dialog._conditions == {(other, 'A01'): 'other plate'}


@pytest.mark.parametrize('scope', ['row', 'column', 'wells'])
def test_annotation_without_a_selected_scope_preserves_existing_conditions(overview, scope):
    dialog = overview
    previous = {(dialog.plates.currentData(), 'A01'): 'keep'}
    dialog._conditions = dict(previous)
    _choose(dialog.annotation_scope, scope)
    dialog.annotation_text.setText('must not spread')
    _click(dialog, 'Apply annotation')
    assert dialog._conditions == previous
    assert 'Choose a row, column or well selection' in dialog.status.text()


@pytest.mark.parametrize('operation,caption', [('_save_annotations', 'Save annotations'),
                                             ('_load_annotations', 'Load annotations')])
def test_annotation_file_dialog_cancel_starts_no_work(overview, monkeypatch, operation, caption):
    dialog = overview
    before = {(dialog.plates.currentData(), 'A01'): 'keep'}
    dialog._conditions = dict(before)
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: ('', ''))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: ('', ''))
    _click(dialog, caption)
    assert dialog._conditions == before
    assert dialog._export_jobs.active_jobs() == 0


def test_annotation_actions_refuse_to_open_files_while_an_export_is_running(
        overview, monkeypatch, qtbot):
    dialog = overview
    entered, release = threading.Event(), threading.Event()

    def running_export():
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                        lambda *a, **k: pytest.fail('busy save opened a file chooser'))
    monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                        lambda *a, **k: pytest.fail('busy load opened a file chooser'))
    dialog._export_jobs.submit(running_export, lambda _: None)
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        for caption in ('Save annotations', 'Load annotations'):
            _click(dialog, caption)
            assert 'Wait for the current export' in dialog.status.text()
        assert dialog._export_jobs.is_busy()
    finally:
        release.set()
        _idle(dialog, qtbot)


@pytest.mark.parametrize('invalid', ['schema', 'duplicate'])
def test_annotation_import_rejects_invalid_tables_atomically(overview, tmp_path, monkeypatch, qtbot, invalid):
    import pandas as pd
    dialog = overview
    plate = dialog.plates.currentData()
    before = {(plate, 'B02'): 'keep'}
    dialog._conditions = dict(before)
    records = [{'plate': plate, 'well': 'A01', 'condition': 'new'}]
    table = pd.DataFrame(records + records if invalid == 'duplicate' else records)
    if invalid == 'schema':
        table = table.rename(columns={'condition': 'wrong_header'})
    path = tmp_path / 'invalid.csv'
    tabular.write_table(table, path, canonicalise=False)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(path), ''))
    _click(dialog, 'Load annotations')
    _idle(dialog, qtbot)
    assert dialog._conditions == before
    expected = 'plate, well and condition columns' if invalid == 'schema' else 'duplicated or outside this source'
    assert expected in dialog.status.text()


@pytest.mark.parametrize('action', ['cancel', 'rescan'])
def test_late_annotation_import_cannot_replace_conditions_after_navigation_generation_changes(
        overview, tmp_path, monkeypatch, qtbot, action):
    import pandas as pd
    dialog = overview
    plate = dialog.plates.currentData()
    before = {(plate, 'A01'): 'before'}
    dialog._conditions = dict(before)
    path = tmp_path / 'late.csv'
    tabular.write_table(pd.DataFrame([{'plate': plate, 'well': 'A01', 'condition': 'late'}]),
                        path, canonicalise=False)
    entered, release = threading.Event(), threading.Event()
    original = tabular.read_table

    def delayed_read(*args, **kwargs):
        result = original(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(tabular, 'read_table', delayed_read)
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(path), ''))
    _click(dialog, 'Load annotations')
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        if action == 'cancel':
            _click(dialog, 'Cancel loading or export')
        else:
            dialog.source.setText(str(tmp_path / 'images' / 'plate2'))
            dialog._scan()
    finally:
        release.set()
    _idle(dialog, qtbot)
    qtbot.waitUntil(lambda: not dialog._index_jobs.is_busy(), timeout=10000)
    assert dialog._conditions == before


def test_comparison_refuses_no_selection_and_more_than_one_page_limit_without_losing_selection(overview):
    dialog = overview
    _click(dialog, 'Compare selected fields')
    assert 'Select fields by clicking' in dialog.status.text()
    assert dialog._montage_jobs.active_jobs() == 0
    for index in range(dialog.fields.count()):
        dialog.fields.item(index).setSelected(True)
    assert len(dialog._selected_fields) == 24
    _click(dialog, 'Next fields')
    dialog.fields.item(0).setSelected(True)
    identities = set(dialog._selected_fields)
    assert len(identities) == 25
    _click(dialog, 'Compare selected fields')
    assert 'Compare up to 24 fields' in dialog.status.text()
    assert set(dialog._selected_fields) == identities
    assert dialog._montage_jobs.active_jobs() == 0


def test_comparison_validates_display_choices_before_starting_a_worker(overview):
    dialog = overview
    dialog.fields.item(0).setSelected(True)
    before = set(dialog._selected_fields)
    dialog.channels.setText('not a channel')
    _click(dialog, 'Compare selected fields')
    assert 'nonnegative channel indices' in dialog.status.text()
    assert set(dialog._selected_fields) == before
    assert dialog._montage_jobs.active_jobs() == 0


def test_comparison_reports_an_unreadable_selected_source_and_keeps_the_readable_field(
        overview, qtbot):
    dialog = overview
    first, second = dialog.fields.item(0), dialog.fields.item(1)
    first.setSelected(True)
    second.setSelected(True)
    record = first.data(Qt.UserRole)
    for _, path in record.files:
        Path(path).unlink()
    before = set(dialog._selected_fields)
    _click(dialog, 'Compare selected fields')
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy() and dialog.view._item is not None,
                    timeout=10000)
    assert len(before) == 2 and set(dialog._selected_fields) == before
    assert '2 fields; 1 unreadable previews' in dialog.status.text()
    assert not dialog.view._item.pixmap().isNull()


@pytest.mark.parametrize('action', ['cancel', 'back'])
def test_late_comparison_cannot_replace_navigation_or_cancelled_view(overview, monkeypatch, qtbot, action):
    dialog = overview
    dialog.fields.item(0).setSelected(True)
    before = set(dialog._selected_fields)
    entered, release = threading.Event(), threading.Event()
    original = browser._overview_montage

    def delayed_montage(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(browser, '_overview_montage', delayed_montage)
    _click(dialog, 'Compare selected fields')
    try:
        qtbot.waitUntil(entered.is_set, timeout=10000)
        if action == 'cancel':
            _click(dialog, 'Cancel loading or export')
        else:
            dialog._back()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not dialog._montage_jobs.is_busy(), timeout=10000)
    assert dialog._comparison_signature is None
    assert dialog.view._item is None
    assert set(dialog._selected_fields) == before
