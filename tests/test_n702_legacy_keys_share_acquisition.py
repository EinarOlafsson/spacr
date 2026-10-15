"""Bare catalogue keys reuse shared cache/lock policies and never fuzzy-substitute."""
import hashlib
from pathlib import Path

import pytest

from spacr import model_zoo as zoo, plaque_papers, submodules


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    payload = b'exact legacy key checkpoint'
    mirror = tmp_path / 'private-mirror.pt'
    mirror.write_bytes(payload)
    entry = zoo.ModelEntry(key='exact-model-key', name='published-model.pt', kind='cellpose',
                           uri=mirror.as_uri(), sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    monkeypatch.setattr(zoo, '_planning_model_entries', lambda: [entry])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.delenv('SPACR_MODEL_DOWNLOAD', raising=False)
    return entry, payload


@pytest.mark.parametrize('consumer', ['training', 'plaque', 'figure', 'detector'])
def test_actual_bare_key_consumers_reuse_verified_cache_on_repeated_calls(checkpoint, monkeypatch, tmp_path, consumer):
    from dataclasses import replace
    entry, payload = checkpoint
    if consumer == 'detector':
        entry = replace(entry, kind='detector')
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    fetch = zoo.fetch
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0].key)
        return fetch(*args, **kwargs)
    monkeypatch.setattr(zoo, 'fetch', counted)
    def resolve():
        if consumer == 'training':
            return submodules._resolve_training_base(entry.key)
        if consumer == 'plaque':
            return submodules._resolve_plaque_model({'plaque_model': entry.key})
        if consumer == 'detector':
            return submodules._resolve_well_detector({'well_detection': entry.key})
        return plaque_papers._zoo_path(entry.key, tmp_path / 'legacy-cache', kind='cellpose')[0]
    first = resolve()
    second = resolve()
    assert first == second
    assert Path(first).read_bytes() == payload
    assert calls == [entry.key]


def test_figure_key_does_not_use_fuzzy_catalogue_resolution(checkpoint, monkeypatch, tmp_path):
    monkeypatch.setattr(zoo, 'resolve', lambda *a, **k: pytest.fail('consumer used fuzzy resolver'))
    with pytest.raises(zoo.ModelUnreadable):
        plaque_papers._zoo_path('exact-model', tmp_path, kind='cellpose')


def test_figure_model_kind_is_checked_before_acquisition(checkpoint, monkeypatch, tmp_path):
    entry, _payload = checkpoint
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('wrong kind fetched'))
    with pytest.raises(zoo.ModelZooError):
        plaque_papers._zoo_path(entry.key, tmp_path, kind='detector')


def test_plaque_key_preview_uses_only_pure_planning_index(checkpoint, monkeypatch):
    entry, _payload = checkpoint
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: pytest.fail('preview discovered providers'))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('preview acquired weights'))
    with pytest.raises(submodules.ModelZooMissing):
        submodules._resolve_plaque_model({'plaque_model': entry.key}, fetch=False)


@pytest.mark.parametrize('consumer', ['training', 'plaque', 'figure', 'detector'])
def test_shared_legacy_acquisition_honors_relocated_model_cache(
        checkpoint, monkeypatch, tmp_path, consumer):
    from dataclasses import replace
    entry, payload = checkpoint
    if consumer == 'detector':
        entry = replace(entry, kind='detector')
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    destination = tmp_path / 'relocated-models'
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(destination))
    if consumer == 'training':
        result = submodules._resolve_training_base(entry.key)
    elif consumer == 'plaque':
        result = submodules._resolve_plaque_model({'plaque_model': entry.key})
    elif consumer == 'detector':
        result = submodules._resolve_well_detector({'well_detection': entry.key})
    else:
        result = plaque_papers._zoo_path(entry.key, tmp_path / 'obsolete-cache', kind='cellpose')[0]
    assert Path(result).parent == destination
    assert Path(result).read_bytes() == payload
    assert not (tmp_path / 'cache' / 'models').exists()
