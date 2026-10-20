"""Failed rename writes retain the selected profile's saved identity."""
from pathlib import Path

import pytest

from spacr.qt import recipes


def test_failed_rename_retains_identity_and_retries(tmp_path, monkeypatch):
    monkeypatch.setenv(recipes.RECIPE_DIR_ENV, str(tmp_path))
    recipe = recipes.Recipe('Original plate', 'mask', settings={'n_jobs': 3})
    original = Path(recipes.save_recipe(recipe))
    saved_bytes = original.read_bytes()
    cyclic = []
    cyclic.append(cyclic)
    recipe.settings = {'cyclic': cyclic}

    for _ in range(2):
        with pytest.raises(ValueError, match='Circular reference'):
            recipes._rename_recipe(recipe, 'Renamed plate')
        assert recipe.name == 'Original plate'
        assert recipe.path == str(original)
        assert original.read_bytes() == saved_bytes
        assert list(original.parent.iterdir()) == [original]

    recipe.settings = {'n_jobs': 7}
    renamed = Path(recipes._rename_recipe(recipe, 'Renamed plate'))
    assert recipe.name == 'Renamed plate'
    assert recipe.path == str(renamed)
    assert not original.exists()
    assert recipes.load_recipe(str(renamed)).settings == {'n_jobs': 7}
    assert list(renamed.parent.iterdir()) == [renamed]


def test_failed_rename_commit_retains_identity_and_destination(tmp_path, monkeypatch):
    monkeypatch.setenv(recipes.RECIPE_DIR_ENV, str(tmp_path))
    recipe = recipes.Recipe('Original plate', 'mask', settings={'n_jobs': 3})
    original = Path(recipes.save_recipe(recipe))
    saved_bytes = original.read_bytes()
    writer = recipes._atomic_write_text
    target = original.with_name('renamed-plate.json')

    def refuse_commit(path, text):
        path.mkdir()
        (path / 'keep.bin').write_bytes(b'preserve destination')
        return writer(path, text)

    monkeypatch.setattr(recipes, '_atomic_write_text', refuse_commit)
    with pytest.raises(OSError):
        recipes._rename_recipe(recipe, 'Renamed plate')
    assert recipe.name == 'Original plate'
    assert recipe.path == str(original)
    assert original.read_bytes() == saved_bytes
    assert (target / 'keep.bin').read_bytes() == b'preserve destination'
    assert set(original.parent.iterdir()) == {original, target}
