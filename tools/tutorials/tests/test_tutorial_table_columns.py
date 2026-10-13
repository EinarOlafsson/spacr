from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def test_real_header_drag_preserves_values_and_exposes_headings():
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication,QHeaderView,QTableWidget,QTableWidgetItem
    from tutorial_table_columns import resize_visible_columns
    app=QApplication.instance() or QApplication([])
    table=QTableWidget(2,3);table.setHorizontalHeaderLabels(['source_object_identity','predicted_class','remaining'])
    for row in range(2):
        for col in range(3):table.setItem(row,col,QTableWidgetItem(f'object_{row}_{col}'))
    table.horizontalHeader().setDefaultSectionSize(45)
    table.horizontalHeader().setSectionResizeMode(2,QHeaderView.Stretch)
    table.resize(1100,400);table.show();table.activateWindow();QTest.qWait(100)
    def settle(seconds):QTest.qWait(round(seconds*1000))
    try:
        changes=resize_visible_columns(table,settle)
        assert len(changes)==2
        assert all(row['after']>row['before'] for row in changes)
        for col in range(2):
            assert table.columnWidth(col)>table.horizontalHeader().fontMetrics().horizontalAdvance(table.horizontalHeaderItem(col).text())
        assert [[table.item(r,c).text() for c in range(3)] for r in range(2)]==[
            ['object_0_0','object_0_1','object_0_2'],['object_1_0','object_1_1','object_1_2']]
    finally:table.close()
