"""Legacy consumer paths reuse exact shared acquisition rather than stock/fuzzy fallback."""
import hashlib
from pathlib import Path

import pytest

from spacr import model_zoo as zoo, plaque_papers, submodules


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    payload = b'exact native mirror legacy checkpoint'
    mirror = tmp_path / 'mirror'
    mirror.mkdir()
    source = mirror / 'published-r7.CP_model'
    source.write_bytes(payload)
    entry = zoo.ModelEntry(key='legacy-r7', name=source.name, kind='cellpose',
                           sha256=hashlib.sha256(payload).hexdigest(), version='7', uri=source.as_uri())
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [entry])
    monkeypatch.setattr(zoo, '_planning_model_entries', lambda: [entry])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.delenv('SPACR_MODEL_DOWNLOAD', raising=False)
    return entry, payload


@pytest.mark.parametrize('requested', ['/old/laptop/published-r7.CP_model',
                                     r'C:\Users\lab\published-r7.CP_model'])
@pytest.mark.parametrize('consumer', ['training', 'plaque', 'figure'])
def test_actual_legacy_consumers_fetch_only_exact_missing_basename(checkpoint, tmp_path, requested, consumer):
    entry, payload = checkpoint
    if consumer == 'training':
        resolved = submodules._resolve_training_base(requested)
    elif consumer == 'plaque':
        resolved = submodules._resolve_plaque_model({'plaque_model': requested})
    else:
        resolved, identity = plaque_papers._zoo_path(requested, tmp_path / 'old-cache')
        assert entry.sha256 in identity
    assert Path(resolved).read_bytes() == payload
    assert Path(resolved).name == entry.name


def test_actual_detector_fetches_missing_exact_published_filename(checkpoint, monkeypatch):
    from dataclasses import replace
    entry, payload = checkpoint
    detector = replace(entry, kind='detector')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [detector])
    resolved = submodules._resolve_well_detector({'well_detection': '/old/' + entry.name})
    assert Path(resolved).read_bytes() == payload


def test_preview_of_missing_legacy_path_never_fetches(checkpoint, monkeypatch):
    entry, _payload = checkpoint
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('preview acquired weights'))
    with pytest.raises(submodules.ModelZooMissing):
        submodules._resolve_plaque_model({'plaque_model': '/old/' + entry.name}, fetch=False)


def test_unknown_training_path_is_refused_before_cellpose_can_use_stock(checkpoint):
    with pytest.raises(submodules.ModelZooMissing):
        submodules._resolve_training_base('/old/published-r6.CP_model')


def test_wrong_model_kind_is_not_loaded_as_a_detector(checkpoint):
    entry, _payload = checkpoint
    with pytest.raises(zoo.ModelZooError):
        submodules._resolve_well_detector({'well_detection': '/old/' + entry.name})


def test_supplied_existing_custom_training_file_keeps_priority(checkpoint, tmp_path, monkeypatch):
    custom = tmp_path / 'custom.CP_model'
    custom.write_bytes(b'user custom weights')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: pytest.fail('custom file discovered models'))
    assert submodules._resolve_training_base(str(custom)) == str(custom)
    assert custom.read_bytes() == b'user custom weights'
