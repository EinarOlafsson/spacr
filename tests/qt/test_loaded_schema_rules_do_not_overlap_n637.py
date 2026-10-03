"""After Load schema the replaced condition boxes never draw over the loaded ones."""
import json
import os

import pandas as pd
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication

from spacr import condition_annotations as backend
from spacr.qt.widgets import condition_annotation_dialog as ui


def _schema(path):
    rules = [{'name': name, 'metadata_column': 'columnID', 'include': '', 'exclude': '',
              'manual_rows': [], 'match_mode': 'values', 'include_values': values, 'exclude_values': []}
             for name, values in (('WildType', ['c1', 'c2']), ('mutant', ['c4']))]
    path.write_text(json.dumps({'format': 'spacr.annotation-schema', 'version': 1, 'recipe_version': 3,
                                'manual_rows': 'excluded',
                                'columns': [{'column': 'genotype', 'kind': 'rules', 'conditions': rules}]}))


def _visible_boxes(dialog):
    return [box for box in dialog.findChildren(ui.ConditionBox) if box.isVisibleTo(dialog)]


def _global_rect(widget):
    return QRect(widget.mapTo(widget.window(), widget.rect().topLeft()), widget.size())


def test_loaded_schema_rows_have_disjoint_geometry(qtbot, tmp_path, monkeypatch):
    path = tmp_path / 'genotype.json'
    _schema(path)
    monkeypatch.setattr(ui.QFileDialog, 'getOpenFileName', lambda *_a, **_k: (str(path), 'JSON'))
    frame = pd.DataFrame({'object_label': [1, 2, 3, 4], 'columnID': ['c1', 'c2', 'c4', 'c7']})
    dialog = ui.ConditionAnnotationDialog(frame, backend.source_context(), threaded=False)
    qtbot.addWidget(dialog)
    dialog.resize(1400, 1000)
    dialog.show()
    dialog.output_column.setText('genotype')
    dialog.add_box()
    QApplication.processEvents()
    assert dialog.boxes and dialog.boxes[0].match_mode.currentData() == 'criteria'
    qtbot.mouseClick(dialog.load_schema_button, Qt.LeftButton)
    QApplication.processEvents()
    boxes = _visible_boxes(dialog)
    assert boxes == dialog.boxes
    assert [box.name.text() for box in boxes] == ['WildType', 'mutant']
    rects = [_global_rect(box) for box in boxes]
    for index, rect in enumerate(rects):
        for other in rects[index + 1:]:
            assert not rect.intersects(other), (rect, other)
    for box in boxes:
        rows = [w for w in box.findChildren(ui.QWidget) if w.parentWidget() is box and w.isVisibleTo(box)
                and w.height() > 0]
        for index, widget in enumerate(rows):
            for other in rows[index + 1:]:
                assert not widget.geometry().intersects(other.geometry()), (widget, other)
    out = os.environ.get('SPACR_637_GRAB')
    if out:
        dialog.grab().save(out)
    dialog.reject()
