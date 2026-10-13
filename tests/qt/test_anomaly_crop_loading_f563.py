"""Control Charts loads PNG links and renders scored outlier crops."""
import hashlib
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip('PySide6')
from PIL import Image

from spacr.qt.screens.control_chart import ControlChartScreen, _read_control_chart_table


def test_loaded_object_table_scores_and_displays_its_crops(qtbot, tmp_path):
    rng = np.random.default_rng(4)
    rows = []
    for well in range(1, 9):
        for label in range(1, 9):
            rows.append(dict(
                plateID='plate1', rowID='r1', columnID=f'c{well}', fieldID='f1',
                object_label=label, condition='neg' if well <= 4 else 'compound',
                feature_0=float(rng.normal() + (4 if well == 8 else 0)),
                feature_1=float(rng.normal() + (5 if well == 8 else 0))))
    objects = pd.DataFrame(rows)
    crop = tmp_path / 'example.png'
    Image.new('RGB', (12, 12), (30, 120, 210)).save(crop)
    crops = objects[['plateID', 'rowID', 'columnID', 'fieldID']].copy()
    crops['cell_id'] = ['o' + str(value) for value in objects['object_label']]
    crops['png_path'] = str(crop)
    db = tmp_path / 'measurements.db'
    with sqlite3.connect(db) as connection:
        objects.to_sql('cell', connection, index=False)
        crops.to_sql('png_list', connection, index=False)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    screen = ControlChartScreen(threaded=False)
    qtbot.addWidget(screen)
    screen.load_path(str(db), 'cell')
    assert len(screen._frame) == len(objects)
    assert screen._frame['png_path'].eq(str(crop)).all()
    screen._value.setCurrentText('feature_0')
    screen._control_column.setCurrentText('condition')
    screen._on_control_column('condition')
    screen._negative.setCurrentText('neg')
    screen._anomaly_method.setCurrentIndex(1)
    screen._anomaly_score.setChecked(True)
    assert screen._anomaly is not None, screen.anomaly_summary.text()
    assert len(screen._anomaly.cells) == len(objects)
    assert screen._anomaly.cells['png_path'].eq(str(crop)).all()
    assert any(axis.images for axis in screen.anomaly_figure.axes)
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before
    screen.close()


def test_csv_and_nonobject_tables_keep_their_supplied_rows(tmp_path):
    frame = pd.DataFrame({'measurement': [3, 2], 'png_path': ['chosen.png', '']})
    csv = tmp_path / 'external.csv'
    frame.to_csv(csv, index=False)
    loaded = _read_control_chart_table(str(csv), None)
    pd.testing.assert_frame_equal(loaded, pd.read_csv(csv))
    db = tmp_path / 'custom.db'
    with sqlite3.connect(db) as connection:
        frame.to_sql('custom', connection, index=False)
    pd.testing.assert_frame_equal(_read_control_chart_table(str(db), 'custom'), frame)
