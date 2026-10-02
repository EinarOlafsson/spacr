"""Offline acceptance requires fresh real mask-directory and merged artifacts."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def checker():
    """Load the shipped standalone checker without importing application modules."""
    source = Path(__file__).parents[1] / 'packaging/offline/offline_mask_check.py'
    spec = importlib.util.spec_from_file_location('offline_mask_acceptance_f587', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data(folder):
    """Create the bundled-settings contract and an unchanged source marker."""
    (folder / 'settings').mkdir(parents=True)
    (folder / 'settings/gen_mask_settings.csv').write_text('Key,Value\ncell_model_name,cpsam\n')
    (folder / 'input.tif').write_bytes(b'original acquisition marker')
    return folder


def _array(root, relative):
    """Write a small genuine NPY artifact in the requested output directory."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ones((3, 3), dtype=np.uint16))


@pytest.mark.parametrize('outputs,expected', [
    (['merged/field.npy'], 1),
    (['not_a_mask_folder/field.npy', 'merged/field.npy'], 1),
    (['cell_mask_stack/field.npy'], 1),
    (['cell_mask_stack/field.npy', 'merged/field.npy'], 0),
])
def test_checker_requires_actual_mask_directory(checker, monkeypatch, tmp_path, outputs, expected):
    data = _data(tmp_path / 'bundle')
    work = tmp_path / 'scratch'
    work.mkdir()

    def run(command):
        """Simulate only the child command's exact fresh output artifacts."""
        root = Path(next(item[4:] for item in command if item.startswith('src=')))
        for relative in outputs:
            _array(root, relative)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(checker.subprocess, 'run', run)
    assert checker.main(['--data', str(data), '--work', str(work)]) == expected
    assert not list(work.iterdir())
    assert (data / 'input.tif').read_bytes() == b'original acquisition marker'


def test_stale_copied_outputs_cannot_satisfy_acceptance(checker, monkeypatch, tmp_path):
    data = _data(tmp_path / 'bundle')
    for relative in ('cell_mask_stack/old.npy', 'merged/old.npy'):
        _array(data, relative)
    before = {str(path.relative_to(data)): path.read_bytes()
              for path in data.rglob('*') if path.is_file()}
    monkeypatch.setattr(checker.subprocess, 'run', lambda command: SimpleNamespace(returncode=0))
    assert checker.main(['--data', str(data)]) == 1
    assert {str(path.relative_to(data)): path.read_bytes()
            for path in data.rglob('*') if path.is_file()} == before


def test_child_failure_is_not_hidden_by_written_outputs(checker, monkeypatch, tmp_path):
    data = _data(tmp_path / 'bundle')

    def run(command):
        """Write valid outputs but return a failing pipeline exit code."""
        root = Path(next(item[4:] for item in command if item.startswith('src=')))
        _array(root, 'cell_mask_stack/field.npy')
        _array(root, 'merged/field.npy')
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(checker.subprocess, 'run', run)
    assert checker.main(['--data', str(data)]) == 1
