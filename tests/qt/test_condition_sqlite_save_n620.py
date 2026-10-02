"""Physical condition saves preserve original tables and commit provenance atomically."""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from PySide6.QtWidgets import QDialog

from spacr.condition_annotations import (
    PROVENANCE_TABLE,
    new_definition,
    save_annotated_table,
    table_identity,
)
from spacr.qt.screens.graph_builder import GraphBuilderScreen
from spacr.qt.widgets.condition_annotation_dialog import ConditionAnnotationDialog


@pytest.fixture
def screen(qtbot, tmp_path):
    path = tmp_path / 'measurements.db'
    frame = pd.DataFrame({'wellID': ['A1', 'B1', 'B1'], 'value': [3., 5., 5.]})
    with sqlite3.connect(path) as db:
        frame.to_sql('samples', db, index=False)
    widget = GraphBuilderScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.load_path(str(path), 'samples')
    definition = new_definition(widget._frame, widget._condition_source)
    definition['conditions'] = [dict(name='treated', metadata_column='wellID', include='A', exclude='',
                                     manual_rows=[table_identity(widget._frame)[2][2]])]
    widget.apply_condition_definition(definition)
    return widget


def test_new_physical_table_picker_editable_roundtrip(screen, tmp_path):
    source = screen._condition_source['path']
    screen.save_annotated_table('samples_annotated')
    assert screen._table_picker.currentText() == 'samples_annotated'
    assert screen._frame.condition.fillna('').tolist() == ['treated', '', 'treated']
    assert screen._condition_definition['source']['table'] == 'samples_annotated'
    with sqlite3.connect(source) as db:
        assert 'condition' not in pd.read_sql_query('SELECT * FROM samples', db)
        payload = json.loads(db.execute(f'SELECT payload_json FROM {PROVENANCE_TABLE}').fetchone()[0])
        assert payload['source']['table'] == 'samples'
    saved = tmp_path / 'materialized-chart.json'
    screen.save_chart(saved)
    screen.load_path(source, 'samples')
    screen.load_chart(saved)
    assert screen._frame.condition.fillna('').tolist() == ['treated', '', 'treated']
    definition = screen._condition_definition
    definition['column'] = 'second label'
    screen.apply_condition_definition(definition)
    screen.save_annotated_table('revised')
    assert 'second label' in screen._frame and 'condition' not in screen._frame


@pytest.mark.parametrize('collision', ['samples', 'samples_view', 'derived_name', '_spacr_reserved'])
def test_collisions_never_overwrite_source(screen, collision):
    source = screen._condition_source['path']
    with sqlite3.connect(source) as db:
        db.execute('CREATE VIEW samples_view AS SELECT * FROM samples')
    from spacr.derived_tables import sidecar_path
    sidecar_path(source).write_text(json.dumps({'source': str(Path(source).resolve()), 'definitions': {'derived_name': {}}}))
    with pytest.raises(ValueError):
        save_annotated_table(source, collision, screen._frame, screen._condition_definition, screen._condition_source)
    with sqlite3.connect(source) as db:
        assert db.execute('SELECT count(*) FROM samples').fetchone()[0] == 3
        assert not db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (PROVENANCE_TABLE,)).fetchone()


def test_failed_receipt_rolls_back_new_table_and_rows(screen):
    source = screen._condition_source['path']
    with sqlite3.connect(source) as db:
        db.execute(f'CREATE TABLE {PROVENANCE_TABLE} (table_name TEXT PRIMARY KEY, payload_json TEXT NOT NULL)')
        db.execute(f"CREATE TRIGGER receipt_failure BEFORE INSERT ON {PROVENANCE_TABLE} BEGIN SELECT RAISE(ABORT, 'receipt failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='receipt failure'):
        save_annotated_table(source, 'new_table', screen._frame, screen._condition_definition, screen._condition_source)
    with sqlite3.connect(source) as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='new_table'").fetchone()
        assert db.execute(f'SELECT count(*) FROM {PROVENANCE_TABLE}').fetchone()[0] == 0


def test_changed_materialized_labels_are_not_overwritten(screen):
    source = screen._condition_source['path']
    screen.save_annotated_table('saved')
    with sqlite3.connect(source) as db:
        db.execute("UPDATE saved SET condition='edited' WHERE wellID='A1'")
    screen.load_path(source, 'saved')
    assert screen._frame.condition.iloc[0] == 'edited'
    assert screen._condition_definition is None
    assert 'changed' in screen._source.text()


def test_dialog_result_reused_and_source_switch_rejected(screen, monkeypatch):
    installed = []
    def accepted(dialog):
        dialog.refresh_preview()
        installed.append(dialog.result_frame)
        return QDialog.Accepted
    monkeypatch.setattr(ConditionAnnotationDialog, 'exec', accepted)
    screen.open_condition_dialog()
    assert screen._frame is installed[0]
    def switched(dialog):
        dialog.refresh_preview()
        screen.set_frame(pd.DataFrame({'different': [1, 2]}))
        return QDialog.Accepted
    monkeypatch.setattr(ConditionAnnotationDialog, 'exec', switched)
    screen.open_condition_dialog()
    assert 'different' in screen._frame and 'condition' not in screen._frame
    assert 'changed while conditions' in screen._source.text()
