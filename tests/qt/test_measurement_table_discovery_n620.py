"""Graph/Gate discovery omits private receipts without hiding user tables."""
import sqlite3

import pandas as pd
import pytest

from spacr.condition_annotations import (
    PROVENANCE_TABLE,
    apply_conditions,
    new_definition,
    save_annotated_table,
    source_context,
)
from spacr.derived_tables import default_definition, save_definition, schemas
from spacr.qt.screens.graph_builder import GraphBuilderScreen, table_names


@pytest.fixture
def annotated_database(tmp_path):
    path = str(tmp_path / 'measurements.db')
    frame = pd.DataFrame({'plateID': ['p1'], 'rowID': ['r1'], 'columnID': ['c1'],
                          'fieldID': ['f1'], 'object_label': [1], 'area': [100.]})
    with sqlite3.connect(path) as db:
        frame.to_sql('cell', db, index=False)
        for name in ('_spacr_condition_annotations_backup', '_spacr_user_measurements', 'ordinary_data'):
            frame.to_sql(name, db, index=False)
    source = source_context(path, 'cell')
    definition = new_definition(frame, source)
    definition['conditions'] = [dict(name='control', metadata_column='plateID', include='p1',
                                     exclude='', manual_rows=[])]
    save_annotated_table(path, 'annotated_cells', apply_conditions(frame, definition, source), definition, source)
    save_definition(path, default_definition(path, ['cell'], name='Named derived cells'))
    return path


def test_discovery_hides_only_receipt_and_keeps_schema_collision_guard(annotated_database):
    names = table_names(annotated_database)
    assert PROVENANCE_TABLE not in names
    assert {'cell', 'annotated_cells', '_spacr_condition_annotations_backup',
            '_spacr_user_measurements', 'ordinary_data', 'Named derived cells'} <= set(names)
    assert names[0] == 'cell'
    assert PROVENANCE_TABLE in schemas(annotated_database)
    definition = default_definition(annotated_database, ['cell'], name=PROVENANCE_TABLE)
    with pytest.raises(ValueError, match='different from every source table'):
        save_definition(annotated_database, definition)


@pytest.mark.parametrize('kind', ['graph', 'gate'])
def test_both_real_pickers_offer_saved_annotations_without_receipts(qtbot, annotated_database, kind):
    if kind == 'graph':
        screen = GraphBuilderScreen(threaded=False)
    else:
        from spacr.qt.screens.gate_editor import GateEditorScreen
        screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(annotated_database, 'annotated_cells')
    assert screen._table_picker.findText(PROVENANCE_TABLE) == -1
    assert screen._table_picker.findText('annotated_cells') >= 0
    assert screen._table_picker.findText('_spacr_condition_annotations_backup') >= 0
    assert screen._table_picker.findText('Named derived cells') >= 0
    assert screen._frame.condition.iloc[0] == 'control'
    screen.close()
