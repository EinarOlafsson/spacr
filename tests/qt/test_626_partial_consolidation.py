"""A partial image copy must never silently replace the organizer's input."""
import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from spacr import folder_consolidation as fc
from spacr.qt.widgets import organize_for_measure as ofm


@pytest.mark.parametrize('action', ['sort_button', 'auto_button', 'teach_button', 'detect_button'])
def test_failed_copy_preserves_working_state_and_aborts_every_action(qtbot, qt_theme_applied,
                                                                    tmp_path, monkeypatch, action):
    source = tmp_path / 'experiment'
    for channel in (1, 2):
        folder = source / f'c{channel}'
        folder.mkdir(parents=True)
        for field in (1, 2):
            tifffile.imwrite(folder / f'f{field}_c{channel}.tif',
                             np.full((8, 9), field + channel, dtype=np.uint16))
    originals = {path: path.read_bytes() for path in source.rglob('*.tif')}
    dialog = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    dialog._set_view('text', remember=False)
    if action != 'detect_button':
        existing = tmp_path / 'existing.tif'
        tifffile.imwrite(existing, np.ones((8, 9), dtype=np.uint16))
        if not dialog._channel_columns():
            dialog.add_column('channel')
        dialog.add_files(dialog._channel_columns()[0], [str(existing)])
    dialog.source_edit.setText(str(source))
    dialog.consolidate_check.setChecked(True)
    dialog.metadata_field.set_value('custom')
    dialog.custom_edit.setText(r'(?:c\d+_)?(?P<wellID>f\d+)_(?P<chanID>c\d+)\.tif$')
    dialog.show()
    before = (copy.deepcopy(dialog.rows), copy.deepcopy(dialog.row_keys),
              copy.deepcopy(dialog.columns), dialog.table.rowCount(), dialog.table.columnCount())
    real_copy = fc.shutil.copy2

    def failed_copy(src, dst):
        if Path(src).name == 'f2_c2.tif':
            Path(dst).write_bytes(b'incomplete copy')
            raise OSError('deliberate disk failure')
        return real_copy(src, dst)

    def forbidden(*args, **kwargs):
        pytest.fail('Downstream sorting, teaching or detection ran after an incomplete copy')

    with monkeypatch.context() as patch:
        patch.setattr(fc.shutil, 'copy2', failed_copy)
        patch.setattr(ofm.cs, 'parse_names', forbidden)
        patch.setattr(ofm.cs, 'infer_regex', forbidden)
        patch.setattr(ofm.cs, '_teach_step', forbidden)
        patch.setattr(dialog, '_rematch', forbidden)
        qtbot.mouseClick(getattr(dialog, action), Qt.LeftButton)

    output = tmp_path / 'experiment_renamed'
    manifest = output / 'rename_manifest.csv'
    with manifest.open(newline='', encoding='utf-8') as handle:
        receipt = list(csv.DictReader(handle))
    assert [item['status'] for item in receipt].count('error') == 1
    assert [item['status'] for item in receipt].count('copied') == 3
    successful = {output / item['new_filename']: (output / item['new_filename']).read_bytes()
                  for item in receipt if item['status'] == 'copied'}
    assert len(list(output.glob('*.tif'))) == 3
    assert dialog.status.isVisible()
    assert 'Consolidation failed' in dialog.status.text()
    assert '1 file(s) could not be copied; see the manifest.' in dialog.status.text()
    assert str(manifest) in dialog.status.text()
    assert QApplication.overrideCursor() is None
    assert dialog.source_edit.text() == str(source)
    assert dialog.consolidate_check.isChecked()
    assert (dialog.rows, dialog.row_keys, dialog.columns,
            dialog.table.rowCount(), dialog.table.columnCount()) == before
    assert all(path.read_bytes() == data for path, data in originals.items())

    # Fixing the copy problem and retrying uses a new complete output folder;
    # partial output and its diagnostic manifest remain available for inspection.
    qtbot.mouseClick(dialog.sort_button, Qt.LeftButton)
    assert dialog.source_edit.text() == str(tmp_path / 'experiment_renamed_2')
    assert not dialog.consolidate_check.isChecked()
    assert 'Consolidation failed' not in dialog.status.text()
    assert len(dialog.rows) == 2
    paths = [path for row in dialog.rows for path in row if path]
    assert len(paths) == len(set(paths)) == 4
    assert not dialog._incomplete_rows()
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert all(path.read_bytes() == data for path, data in successful.items())
    assert manifest.exists()


def test_checked_consolidation_without_nested_images_is_a_noop(qtbot, qt_theme_applied,
                                                               tmp_path, monkeypatch):
    source = tmp_path / 'flat'
    source.mkdir()
    tifffile.imwrite(source / 'f1_c1.tif', np.ones((8, 9), dtype=np.uint16))
    dialog = ofm.OrganizeForMeasureDialog()
    qtbot.addWidget(dialog)
    dialog.source_edit.setText(str(source))
    dialog.consolidate_check.setChecked(True)
    monkeypatch.setattr(fc, 'consolidate_folder', lambda *a, **k: pytest.fail('Unexpected copy'))
    assert dialog._consolidate(str(source)) == str(source)
    assert dialog.source_edit.text() == str(source)
    assert dialog.consolidate_check.isChecked()
