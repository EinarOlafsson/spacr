"""Repeated and refused model-cache moves preserve the active effective location."""

import pytest

from spacr import image_quality, model_zoo as zoo, run_journal as journal
from tests.test_n644_relocated_spacr_model_cache import cache


def test_repeated_model_cache_moves_keep_runtime_on_latest_destination(cache, tmp_path, monkeypatch):
    home, entry, _raw, data = cache
    original = home / 'models'
    original.mkdir()
    (original / entry.name).write_bytes(data)
    first = journal._relocate_cache('models', tmp_path / 'first-drive')
    second = journal._relocate_cache('models', tmp_path / 'second-drive')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('repeated move downloaded weights'))
    assert zoo._model_cache_root() == second
    assert zoo._ensure_model_file(entry.key) == second / entry.name
    assert not original.exists() and not first.exists()
    assert (second / entry.name).read_bytes() == data


def test_failed_model_relocation_keeps_original_runtime_cache_and_bytes(cache, tmp_path, monkeypatch):
    home, entry, _raw, data = cache
    original = home / 'models'
    original.mkdir()
    (original / entry.name).write_bytes(data)
    def fail_metadata(*args):
        raise OSError('controlled cache metadata failure')
    monkeypatch.setattr(journal, '_atomic_write_text', fail_metadata)
    with pytest.raises(OSError, match='controlled cache metadata failure'):
        journal._relocate_cache('models', tmp_path / 'other-drive')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('refused move downloaded weights'))
    assert zoo._model_cache_root() == original
    assert zoo._ensure_model_file(entry.key) == original / entry.name
    assert (original / entry.name).read_bytes() == data
    assert not (tmp_path / 'other-drive/spacr-models').exists()


def test_on_demand_quality_classifier_cache_follows_same_model_relocation(cache, tmp_path):
    home, _entry, _raw, data = cache
    original = home / 'models'
    original.mkdir()
    filename = f'image_qc_classifier_v{image_quality._QC_MODEL_VERSION}.pt'
    (original / filename).write_bytes(data)
    relocated = journal._relocate_cache('models', tmp_path / 'other-drive')
    assert image_quality._builtin_qc_model_path() == relocated / filename
    assert image_quality._builtin_qc_model_path().read_bytes() == data
    assert not original.exists()
