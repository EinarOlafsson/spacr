"""Only configured trusted families participate in bounded startup warm-up."""
import hashlib

import pytest

from tools.web_mask_generator import registry, service, masks


@pytest.fixture
def provisioned(tmp_path, monkeypatch):
    families = []
    paths = {}
    for family, default, keys in [
        ('toxoplasma_plaque', 'toxoplasma_plaque_v4', ['toxoplasma_plaque_v3', 'toxoplasma_plaque_v4']),
        ('cell_from_hoechst', 'cell_from_hoechst_v2', ['cell_from_hoechst_v1', 'cell_from_hoechst_v2']),
        ('toxoplasma_well_detector', 'toxoplasma_well_detector_v3', ['toxoplasma_well_detector_v3']),
    ]:
        versions = []
        for key in keys:
            path = tmp_path / key
            path.write_bytes(('controlled local fixture ' + key).encode())
            paths[key] = path
            versions.append({'key': key, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        families.append({'family': family, 'default': default, 'versions': versions})
    monkeypatch.setattr(registry, 'model_families', lambda: families)
    monkeypatch.setattr(service, 'model_families', lambda: families)
    calls = []
    monkeypatch.setattr(service, '_load_model', lambda path, digest:
                        calls.append(('plaque', path, digest)))
    monkeypatch.setattr(masks, '_load_mask_model', lambda path, digest, role:
                        calls.append((role, path, digest)))
    return paths, calls


def test_nonplaque_only_startup_uses_its_exact_provisioned_mask(provisioned):
    paths, calls = provisioned
    key = 'cell_from_hoechst_v1'
    instance = service.ModelService({key: paths[key]})
    assert instance.preload_default() == key
    assert calls == [('cell', paths[key], hashlib.sha256(paths[key].read_bytes()).hexdigest())]


def test_detector_only_startup_does_not_require_or_construct_plaque(provisioned):
    paths, calls = provisioned
    key = 'toxoplasma_well_detector_v3'
    assert service.ModelService({key: paths[key]}).preload_default() is None
    assert calls == []


def test_empty_service_does_not_construct_models(provisioned):
    _, calls = provisioned
    assert service.ModelService({}).preload_default() is None
    assert calls == []


def test_current_plaque_default_wins_and_only_one_model_is_constructed(provisioned):
    paths, calls = provisioned
    instance = service.ModelService(dict(reversed(list(paths.items()))))
    assert instance.preload_default() == 'toxoplasma_plaque_v4'
    path = paths['toxoplasma_plaque_v4']
    assert calls == [('plaque', path, hashlib.sha256(path.read_bytes()).hexdigest())]


def test_nondefault_configured_segmentation_is_warmed_without_unavailable_default(provisioned):
    paths, calls = provisioned
    key = 'toxoplasma_plaque_v3'
    assert service.ModelService({key: paths[key]}).preload_default() == key
    assert calls == [('plaque', paths[key], hashlib.sha256(paths[key].read_bytes()).hexdigest())]


@pytest.mark.parametrize('reverse', [False, True])
def test_mask_selection_is_deterministic_and_bounded(provisioned, reverse):
    paths, calls = provisioned
    keys = ['cell_from_hoechst_v1', 'cell_from_hoechst_v2']
    if reverse:
        keys.reverse()
    assert service.ModelService({key: paths[key] for key in keys}).preload_default() == 'cell_from_hoechst_v1'
    assert len(calls) == 1
    assert calls[0][1] == paths['cell_from_hoechst_v1']


@pytest.mark.parametrize('failure', ['unknown', 'missing', 'checksum'])
def test_all_configured_entries_validate_before_any_model_is_constructed(provisioned, failure, tmp_path):
    paths, calls = provisioned
    configured = {'toxoplasma_plaque_v4': paths['toxoplasma_plaque_v4']}
    if failure == 'unknown':
        configured['foreign_model'] = paths['cell_from_hoechst_v1']
    else:
        key = 'cell_from_hoechst_v1'
        configured[key] = paths[key]
        if failure == 'missing':
            paths[key].unlink()
        else:
            paths[key].write_bytes(b'wrong checkpoint identity')
    with pytest.raises(ValueError):
        service.ModelService(configured).preload_default()
    assert calls == []


def test_warmup_does_not_make_unprovisioned_request_default_available(provisioned):
    paths, calls = provisioned
    key = 'cell_from_hoechst_v1'
    instance = service.ModelService({key: paths[key]})
    assert instance.preload_default() == key
    with pytest.raises(ValueError, match='not provisioned'):
        registry.resolve_trusted_model('cell_from_hoechst', instance.models)
    assert len(calls) == 1
