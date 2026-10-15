"""Failed template serialization preserves the previous complete profile."""
from pathlib import Path

import pytest

from spacr.qt import recipes


@pytest.mark.parametrize('exported', [False, True])
def test_failed_profile_serialization_keeps_saved_bytes_and_can_retry(tmp_path, monkeypatch, exported):
    monkeypatch.setenv(recipes.RECIPE_DIR_ENV, str(tmp_path / 'store'))
    directory = str(tmp_path / 'export') if exported else None
    recipe = recipes.Recipe('Plate A', 'mask', settings={'n_jobs': 3})
    saved = Path(recipes.save_recipe(recipe, directory=directory))
    original = saved.read_bytes()
    cyclic = []
    cyclic.append(cyclic)
    recipe.settings = {'n_jobs': 7, 'invalid': cyclic}
    for attempt in range(2):
        with pytest.raises(ValueError, match='Circular reference'):
            recipes.save_recipe(recipe, directory=directory)
        assert saved.read_bytes() == original
        assert recipes.load_recipe(str(saved)).settings == {'n_jobs': 3}
        assert recipe.path == str(saved)
        assert list(saved.parent.iterdir()) == [saved]
    recipe.settings = {'n_jobs': 7}
    assert recipes.save_recipe(recipe, directory=directory) == str(saved)
    assert recipes.load_recipe(str(saved)).settings == {'n_jobs': 7}
    assert list(saved.parent.iterdir()) == [saved]


def test_failed_profile_commit_keeps_blocking_directory_and_prior_path(tmp_path):
    target = tmp_path / 'plate-a.json'
    target.mkdir()
    sentinel = target / 'keep.bin'
    sentinel.write_bytes(b'preserve existing destination contents')
    recipe = recipes.Recipe('Plate A', 'mask', settings={'n_jobs': 3}, path='previous.json')
    with pytest.raises(OSError):
        recipes.save_recipe(recipe, directory=str(tmp_path))
    assert sentinel.read_bytes() == b'preserve existing destination contents'
    assert recipe.path == 'previous.json'
    assert list(tmp_path.iterdir()) == [target]
