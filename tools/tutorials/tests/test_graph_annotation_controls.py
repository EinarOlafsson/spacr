"""The lesson recipe is entered through real controls, never an injected spec."""
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_graph_annotations import record_annotations


@pytest.mark.parametrize('real_example', [False, True])
def test_real_controls_apply_and_check_every_row(tmp_path, real_example):
    from PySide6.QtWidgets import QApplication

    from spacr.qt.screens.graph_builder import GraphBuilderScreen

    app = QApplication.instance() or QApplication([])
    if real_example:
        archive = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials/examples/Graph_Builder_real_measurements.zip'
        with zipfile.ZipFile(archive) as bundle:
            names = [name for name in bundle.namelist() if name.endswith('/measurements.db') or name == 'measurements.db']
            assert len(names) == 1
            database = tmp_path / 'measurements.db'
            database.write_bytes(bundle.read(names[0]))
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
            frame = pd.read_sql_query('SELECT * FROM cell', connection)
    else:
        frame = pd.DataFrame({'columnID': ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c10', None, 'c1'],
                              'rowID': ['r1', 'r4', 'r5', 'r6', 'r7', 'r9', 'r10', 'r1', 'r1', None],
                              'value': list(range(10))}, index=[3] * 10)
    before = frame.copy(deep=True)
    screen = GraphBuilderScreen(threaded=real_example)
    screen.set_frame(frame)
    screen.resize(1200, 850)
    screen.show()
    visuals = []
    workers = []

    def settle(seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)  # Release the GIL for the real Python preview worker.

    def capture(visual):
        visuals.append(visual)
        dialog = app.activeModalWidget()
        if dialog is not None:
            workers.extend(thread for thread, _ in dialog._jobs._jobs.values())

    try:
        proof = record_annotations(app, screen, capture, settle, 20)
        assert visuals == ['13_genotype_exact_values', '14_replicate_rule_boxes',
                           '15_composed_condition_preview', '16_annotations_applied']
        settle(0.1)
        for thread in workers:
            try:
                assert not thread.isRunning()
            except RuntimeError:  # Qt already deleted the retired thread.
                pass
        assert frame.equals(before)
        assert proof['rows_checked'] == len(frame)
        assert proof['counts']['replicate'] == {'replicate 1': int(frame.rowID.isin(['r1', 'r4', 'r5', 'r6', 'r7', 'r9', 'r10']).sum())}
        assert set(proof['counts']['condition']) <= {'WildType_replicate 1', 'mutant_replicate 1'}
        assert proof['actual_dialog_controls']
        assert proof['illustrative_labels_only'] and not proof['published']
        assert screen._condition_definition['version'] == 3
        assert screen._condition_definition['columns'][-1] == {
            'column': 'condition', 'kind': 'template', 'parts': [
                {'kind': 'column', 'column': 'genotype'}, {'kind': 'text', 'text': '_'},
                {'kind': 'column', 'column': 'replicate'}]}
        if real_example:
            assert len(frame) == 2341
            assert set(frame.columnID) == {'c1', 'c2'}
            assert set(frame.rowID) == {'r5', 'r12'}
            assert proof['counts']['genotype'] == {'WildType': 2341}
            assert proof['combined_missing_rows'] == int((frame.rowID == 'r12').sum())
        else:
            assert proof['counts']['genotype'] == {'WildType': 4, 'mutant': 4}
            assert proof['combined_missing_rows'] == 3
    finally:
        screen.close()
        screen.deleteLater()
        app.processEvents()
