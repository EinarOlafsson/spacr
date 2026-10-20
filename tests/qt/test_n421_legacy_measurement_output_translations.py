"""Retained hand-drawn measurement outputs use the active reviewed language."""
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import measure
from spacr.qt import i18n
from spacr.qt.screens.measure_inputs import MeasureInputsScreen
from spacr.qt.widgets import measure_input_table


@pytest.fixture
def swedish(monkeypatch):
    monkeypatch.setattr(i18n, 'current_language', lambda: 'sv')


@pytest.fixture
def window(qtbot, swedish):
    widget = MeasureInputsScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


def test_stage_messages_reach_the_real_log_in_swedish(window, monkeypatch, tmp_path):
    image = tmp_path / 'field_C1.tif'
    mask = tmp_path / 'field_cell_mask.tif'
    tifffile.imwrite(image, np.arange(64, dtype=np.uint16).reshape(8, 8))
    labels = np.zeros((8, 8), dtype=np.uint16)
    labels[2:6, 2:6] = 1
    tifffile.imwrite(mask, labels)
    table = measure.FieldTable(
        rows=[measure.FieldRow(label='field', channels={0: str(image)},
                               masks={'cell': str(mask)}, well='A01', field=1)],
        n_channels=1, roles=('cell',), plate='drawn')
    calls = []
    monkeypatch.setattr(measure, 'measure_crop', lambda settings: calls.append(settings))
    destination = tmp_path / 'result {unchanged}'

    result = measure.measure_from_field_table(
        table, {'save_png': False}, dst=str(destination), progress=window.progress.emit)

    assert len(calls) == 1
    assert calls[0] == result['settings']
    assert Path(result['db_path']) == destination / 'measurements' / 'measurements.db'
    assert result['merged']
    assert all(Path(path).is_file() for path in result['merged'])
    assert list((destination / 'merged').glob('*.npy'))
    assert window._log.toPlainText().splitlines() == [
        'Skriver de sammanslagna arrayerna för 1 fält...',
        'Skrev 1 fält. Mäter dem nu; detta är Measure-modulens egen körning.',
    ]


def test_result_failure_messages_preserve_payloads_and_signal_order(window, tmp_path):
    path = str(tmp_path / 'mätningar {original}' / 'measurements.db')
    reason = 'I/O {source}: /not/replaced/ä.tif'
    result = {'stems': ['first', 'second'], 'db_path': path, 'db_exists': False}
    finished = []
    window.run_finished.connect(finished.append)

    window._on_done(None)
    window._on_done(result)
    window._runner.job_failed.emit(reason)

    assert finished[0] is None
    assert finished[1] is result
    assert finished[2] is None
    assert window.result() is None
    assert window.inputs.isEnabled()
    assert result == {'stems': ['first', 'second'], 'db_path': path, 'db_exists': False}
    assert window._log.toPlainText().splitlines() == [
        'Körningen gav inget resultat.',
        'Mätte 2 fält.',
        f'Databas: {path}',
        'VARNING: databasen finns inte på den förväntade platsen.',
        f'Körningen avbröts: {reason}',
    ]


def test_scan_failure_writes_are_localized_without_changing_scan_flow(
        qtbot, swedish, monkeypatch, tmp_path):
    widget = measure_input_table.MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    trouble = 'unreadable {folder}/å'
    failure = 'permission {original}: /readonly'
    seen = []
    original = widget._regex_status.setText

    def record(text):
        original(text)
        seen.append(widget._regex_status.text())

    monkeypatch.setattr(widget._regex_status, 'setText', record)
    monkeypatch.setattr(measure_input_table, '_walk', lambda paths: ([], trouble))
    widget.add_dropped([str(tmp_path / 'absent')])
    widget._scanner.job_failed.emit(failure)

    assert widget.table().rows == []
    assert widget.unassigned() == []
    assert not widget.is_scanning()
    assert len(seen) == 3
    assert seen[0] == 'Granskar det som släpptes…'
    assert seen[1] == f'En del av det som släpptes kunde inte läsas: {trouble}'
    assert seen[2] == f'Det som släpptes kunde inte läsas: {failure}'


@pytest.mark.parametrize(('language', 'caption'), [
    ('sv', 'Granskar det som släpptes…'),
    ('is', 'Skoða það sem var dregið inn...'),
    ('de', 'Abgelegte Dateien werden geprüft...'),
    ('fr', 'Examen des fichiers déposés…'),
    ('es', 'Examinando los archivos arrastrados…'),
    ('pt', 'Inspecionando os arquivos arrastados...'),
    ('zh_CN', '正在检查拖入的内容…'),
    ('hi', 'छोड़ी गई फ़ाइलें देखी जा रही हैं...'),
    ('ko', '드롭한 항목을 살펴보는 중...'),
])
def test_drop_scan_initial_status_uses_the_active_language(
        qtbot, monkeypatch, language, caption):
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    widget = measure_input_table.MeasureInputTable(threaded=False)
    qtbot.addWidget(widget)
    submitted = []
    monkeypatch.setattr(widget._scanner, 'submit',
                        lambda work, done: submitted.append((work, done)) or True)

    widget.add_dropped(['/not-accessed/by-this-test'])

    assert len(submitted) == 1
    assert widget._regex_status.text() == caption
    assert widget.table().rows == []
    assert widget.unassigned() == []
