"""The actual Share button preserves an existing export after failed writes."""
import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from spacr.qt import recipes


@pytest.mark.parametrize('failure', ['serialization', 'replacement'])
def test_share_button_preserves_existing_export_and_can_retry(qtbot, tmp_path, monkeypatch, failure):
    monkeypatch.setenv(recipes.RECIPE_DIR_ENV, str(tmp_path / 'store'))
    source = Path(recipes.save_recipe(recipes.Recipe('Shared', 'mask', settings={'n_jobs': 3})))
    source_bytes = source.read_bytes()
    screen = QWidget()
    screen.app_key = 'mask'
    qtbot.addWidget(screen)
    dialog = recipes.RecipeDialog(screen)
    qtbot.addWidget(dialog)
    target = tmp_path / 'shared.json'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', staticmethod(lambda *a, **k: (str(target), '')))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', staticmethod(lambda parent, title, text: warnings.append((title, text))))
    assert dialog._btn_export.isEnabled()
    dialog._btn_export.click()
    original = target.read_bytes()
    assert recipes.load_recipe(str(target)).settings == {'n_jobs': 3}
    selected = dialog.selected()
    selected.settings = {'n_jobs': 7}
    real_replace = os.replace
    blocked = tmp_path / 'blocked-destination'
    blocked.mkdir()
    sentinel = blocked / 'keep.bin'
    sentinel.write_bytes(b'preserve blocking-directory bytes')
    if failure == 'serialization':
        cyclic = []
        cyclic.append(cyclic)
        selected.settings['invalid'] = cyclic
    else:
        def refuse_commit(src, dst):
            if Path(dst) == target:
                return real_replace(src, blocked)
            return real_replace(src, dst)
        monkeypatch.setattr(os, 'replace', refuse_commit)
    for attempt in range(2):
        dialog._btn_export.click()
        assert len(warnings) == attempt + 1
        assert warnings[-1][0] == 'Could not write the file'
        assert target.read_bytes() == original
        assert recipes.load_recipe(str(target)).settings == {'n_jobs': 3}
        assert sentinel.read_bytes() == b'preserve blocking-directory bytes'
        assert not list(tmp_path.glob('.shared.json.*.tmp'))
        assert source.read_bytes() == source_bytes
        assert selected.path == str(source)
    selected.settings = {'n_jobs': 7}
    monkeypatch.setattr(os, 'replace', real_replace)
    dialog._btn_export.click()
    assert recipes.load_recipe(str(target)).settings == {'n_jobs': 7}
    assert len(warnings) == 2
    assert source.read_bytes() == source_bytes
    assert selected.path == str(source)
    assert not list(tmp_path.glob('.shared.json.*.tmp'))
