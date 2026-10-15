"""Exercise actual offscreen table fallback before/after the caption correction."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from PySide6.QtWidgets import QApplication
from spacr.qt import path_probe
from spacr.qt.widgets import file_list as fl

app = QApplication.instance() or QApplication([])
source = subprocess.check_output([
    'git', 'show', '348c63b842a18663a364a746f3b277bcef5efdc8:spacr/qt/widgets/file_list.py'], cwd=ROOT)
owner = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)
             and node.name == 'PairedFileTableWidget')
method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
              and node.name == 'attach_database')
namespace = dict(vars(fl))
exec(compile(ast.Module(body=[method], type_ignores=[]), '<pinned-before-method>', 'exec'), namespace)
before_method = namespace['attach_database']
after_method = fl.PairedFileTableWidget.attach_database


def run(implementation):
    fl.PairedFileTableWidget.attach_database = implementation
    scores = ['/screen/plate1_scores.csv']
    counts = ['/screen/plate1_counts.csv']
    mismatch = '/screen/plate2/measurements/measurements.db'
    for path in scores + counts + [mismatch]:
        path_probe.prime(path, True)
    table = fl.PairedFileTableWidget(fl.suggest_file_pairs(scores, counts))
    message = table.attach_database(mismatch)
    rows = table.get_value()
    assert len(rows) == 1 and rows[0]['plate'] == 'plate1'
    assert rows[0]['database'] == mismatch
    assert table.status.text().startswith(message)
    anchor = table._anchor_for_row(0)
    assert anchor == {'plate': 'plate1', 'score': scores[0], 'count': counts[0]}
    table.close()
    table.deleteLater()
    return {'message': message, 'rows': rows, 'anchor': anchor}


try:
    before = run(before_method)
    after = run(after_method)
finally:
    fl.PairedFileTableWidget.attach_database = after_method
assert before['rows'] == after['rows'] and before['anchor'] == after['anchor']
assert "nothing in the file's path named a plate" in before['message']
assert 'plate2' in before['rows'][0]['database']
assert 'Filename matching did not attach the database to a plate row.' in after['message']
assert "nothing in the file's path named a plate" not in after['message']
print(json.dumps({'before': before, 'after': after,
                  'same_attachment_and_anchor': True,
                  'scope': 'Actual Qt table offscreen; no native desktop or filesystem access acceptance.'}, indent=2))
