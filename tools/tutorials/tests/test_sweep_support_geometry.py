"""Full visibility uses a real table and independent saved-value acceptance."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_parameter_sweep import reveal_support, support_geometry
from sweep_evidence import SUPPORT_COLUMNS, check_support_geometry


@pytest.fixture
def support_table():
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QTableWidget, QTableWidgetItem

    app = QApplication.instance() or QApplication([])
    headers = ['trial_id'] + [f'setting_{i}' for i in range(12)] + list(SUPPORT_COLUMNS) + ['n_rows_prepared']
    table = QTableWidget(2, len(headers))
    table.setHorizontalHeaderLabels(headers)
    saved = [dict(trial_id='a', n_rows_fitted_grna=2955, n_wells_grna=620,
                  n_rows_fitted_gene=2700, n_wells_gene=600),
             dict(trial_id='b', n_rows_fitted_grna=2800, n_wells_grna=610,
                  n_rows_fitted_gene=0, n_wells_gene=0)]
    for row, values in enumerate(saved):
        for column, name in enumerate(headers):
            table.setItem(row, column, QTableWidgetItem(str(values.get(name, 'setting'))))
    table.resize(1100, 240)
    table.resizeColumnsToContents()
    table.show()
    QTest.qWait(50)

    def settle(seconds):
        QTest.qWait(round(seconds * 1000))

    yield table, saved, settle
    table.close()
    app.processEvents()


@pytest.mark.parametrize('pixel_scroll', [False, True])
def test_actual_scroll_reveals_all_four_counts_for_both_trials(support_table, pixel_scroll):
    from PySide6.QtWidgets import QAbstractItemView
    table, saved, settle = support_table
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel if pixel_scroll
                                  else QAbstractItemView.ScrollPerItem)
    with pytest.raises(ValueError):
        check_support_geometry(support_geometry(table, table), saved)
    before = [[table.item(r, c).text() for c in range(table.columnCount())] for r in range(2)]
    reveal_support(table, settle)
    evidence = support_geometry(table, table)
    assert check_support_geometry(evidence, saved)['cells'] == 8
    assert table.horizontalScrollBar().value() > 0
    assert before == [[table.item(r, c).text() for c in range(table.columnCount())] for r in range(2)]
    assert all(not table.isColumnHidden(c) for c in range(table.columnCount()))


@pytest.mark.parametrize('case', ['wrong_frame', 'missing_header', 'hidden_header', 'clipped_header',
    'missing_cell', 'duplicate_cell', 'wrong_trial', 'wrong_value', 'clipped_cell',
    'hidden_cell', 'outside_window', 'empty_cell', 'nan_geometry'])
def test_independent_geometry_verifier_rejects_corruption(support_table, case):
    table, saved, settle = support_table
    reveal_support(table, settle)
    evidence = support_geometry(table, table)
    assert check_support_geometry(evidence, saved)['passed']
    bad = deepcopy(evidence)
    if case == 'wrong_frame': bad['frame'] = '08_actual_sweep_finished'
    elif case == 'missing_header': bad['headers'].pop()
    elif case == 'hidden_header': bad['headers'][0]['visible'] = False
    elif case == 'clipped_header': bad['headers'][0]['rect'][0] = -1
    elif case == 'missing_cell': bad['cells'].pop()
    elif case == 'duplicate_cell': bad['cells'][-1] = bad['cells'][0]
    elif case == 'wrong_trial': bad['cells'][0]['trial_id'] = 'different'
    elif case == 'wrong_value': bad['cells'][0]['value'] = '1'
    elif case == 'clipped_cell': bad['cells'][0]['rect'][1] = -1
    elif case == 'hidden_cell': bad['cells'][0]['visible'] = False
    elif case == 'outside_window': bad['window'][2] = 100
    elif case == 'empty_cell': bad['cells'][0]['rect'][2] = 0
    else: bad['cells'][0]['rect'][0] = float('nan')
    with pytest.raises(ValueError):
        check_support_geometry(bad, saved)


@pytest.mark.parametrize('hide', ['column', 'row', 'header', 'narrow'])
def test_actual_visibility_changes_are_not_certified(support_table, hide):
    table, saved, settle = support_table
    reveal_support(table, settle)
    assert check_support_geometry(support_geometry(table, table), saved)['passed']
    if hide == 'column': table.setColumnHidden(13, True)
    elif hide == 'row': table.setRowHidden(1, True)
    elif hide == 'header': table.horizontalHeader().hide()
    else: table.resize(150, 240)
    settle(.05)
    with pytest.raises(ValueError):
        check_support_geometry(support_geometry(table, table), saved)
