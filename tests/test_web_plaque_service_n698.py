"""Private service boundary and artifact lifetime, without hosted/model claims."""
import hashlib
from pathlib import Path
import types
import sys

import numpy as np
import pytest
import tifffile

from tools.web_mask_generator import plaque, service
from tests.refused_access import only_the_owner_can_open


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / "upload.tif"
    tifffile.imwrite(source, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"})
    weights = tmp_path / "weights"
    weights.write_bytes(b"controlled test weights")
    metadata = {"key": "toxoplasma_plaque_v4", "version": "4", "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}
    monkeypatch.setattr(service, "resolve_trusted_model", lambda *a, **kw: {"checkpoint": weights, "metadata": metadata})
    runner_calls = []
    def runner(output, **kwargs):
        runner_calls.append(kwargs)
        output.write_bytes(b"completed result")
        return output
    monkeypatch.setattr(service, "run_plaque_download", runner)
    instance = service.PlaqueService({"toxoplasma_plaque_v4": weights}, job_parent=tmp_path / "private")
    request = {"family": "toxoplasma_plaque", "consent_upload": True, "settings": {"diameter": 52}}
    return instance, source, request, runner_calls


def test_private_copy_is_cleaned_after_successful_delivery(setup):
    instance, source, request, calls = setup
    before = source.read_bytes()
    with instance.job(source, request) as result:
        output = result["result_zip"]
        staged_input = calls[0]["input_file"]
        assert output.read_bytes() == b"completed result"
        assert staged_input.read_bytes() == before
        assert only_the_owner_can_open(output.parent)
        assert calls[0]["model_id"] == "toxoplasma_plaque_v4"
        assert calls[0]["effective_settings"] == {"diameter": 52}
    assert not output.exists() and not staged_input.exists()
    assert not list(instance.job_parent.iterdir())
    assert source.read_bytes() == before


@pytest.mark.parametrize("where", ["inference", "consumer"])
def test_failure_and_consumer_exception_also_remove_private_upload(setup, monkeypatch, where):
    instance, source, request, _ = setup
    if where == "inference":
        def fail(*a, **kw):
            raise RuntimeError("backend failed")
        monkeypatch.setattr(service, "run_plaque_download", fail)
    with pytest.raises(RuntimeError):
        with instance.job(source, request):
            raise RuntimeError("client transfer failed")
    assert not list(instance.job_parent.iterdir())
    assert source.is_file()


@pytest.mark.parametrize("changes", [{"consent_upload": False}, {"consent_upload": "yes"}, {"family": "unknown"}, {"checkpoint": "/client/weights"}, {"selection": {"max_decoded_bytes": 2**40}}, {"selection": []}])
def test_untrusted_requests_cannot_choose_paths_or_limits(setup, changes):
    instance, source, request, calls = setup
    request.update(changes)
    with pytest.raises(ValueError):
        with instance.job(source, request):
            pytest.fail("invalid request was served")
    assert calls == []
    assert not list(instance.job_parent.iterdir())


def test_growing_upload_is_refused_and_partial_copy_is_removed(setup, monkeypatch):
    instance, source, request, calls = setup
    copy = service._copy_upload
    def grow(source, target, limit):
        copy(source, target, limit)
        with target.open("ab") as handle:
            handle.truncate(service.DEFAULT_MAX_FILE_BYTES + 1)
    monkeypatch.setattr(service, "_copy_upload", grow)
    with pytest.raises(ValueError, match="file size"):
        with instance.job(source, request):
            pytest.fail("oversized growing upload served")
    assert not list(instance.job_parent.iterdir())
    assert calls == []


def test_preload_is_explicit_and_uses_exact_registry_digest(setup, monkeypatch):
    instance, _, _, _ = setup
    calls = []
    monkeypatch.setattr(service, "_load_model", lambda path, digest: calls.append((path, digest)))
    assert calls == []
    assert instance.preload_default() == "toxoplasma_plaque_v4"
    checkpoint = instance.models["toxoplasma_plaque_v4"]
    assert calls == [(checkpoint, hashlib.sha256(checkpoint.read_bytes()).hexdigest())]
    with pytest.raises(TypeError):
        instance.models["other"] = checkpoint


def test_digest_cache_never_reuses_a_model_for_different_weight_bytes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "spacr.submodules", types.SimpleNamespace(_plaque_cellpose_model=lambda path: calls.append(path) or object()))
    path = tmp_path / "weights"
    first = plaque._load_model(path, "aa" * 32)
    assert plaque._load_model(path, "aa" * 32) is first
    assert plaque._load_model(path, "bb" * 32) is not first
    assert calls == [str(path), str(path)]


def test_stream_copy_cannot_write_beyond_its_own_limit(tmp_path):
    source = tmp_path / "large"
    source.write_bytes(b"123456789")
    target = tmp_path / "bounded"
    with pytest.raises(ValueError, match="file size"):
        service._copy_upload(source, target, 8)
    assert target.stat().st_size <= 8


@pytest.mark.parametrize('family', ['toxoplasma_plaque', 'toxoplasma_well_detector', *service.OBJECT_ROLES])
def test_all_eight_families_dispatch_only_their_trusted_adapter(setup, monkeypatch, family):
    instance, source, request, _ = setup
    called = []
    def runner(kind):
        def run(output, **kwargs):
            called.append((kind, kwargs))
            output.write_bytes(b'result')
            return output
        return run
    monkeypatch.setattr(service, 'run_plaque_download', runner('plaque'))
    monkeypatch.setattr(service, 'run_well_download', runner('detector'))
    monkeypatch.setattr(service, 'run_mask_download', runner('masks'))
    resolved = []
    def resolve(selected_family, paths, **kwargs):
        resolved.append((selected_family, kwargs))
        checkpoint = next(iter(paths.values()))
        return {'checkpoint': checkpoint, 'metadata': {'key': family + '_v1', 'version': '1', 'sha256': 'ab' * 32}}
    monkeypatch.setattr(service, 'resolve_trusted_model', resolve)
    request.update(family=family, model_key=family + '_v1')
    with instance.job(source, request) as artifact:
        assert artifact['model']['key'] == family + '_v1'
        assert artifact['result_zip'].read_bytes() == b'result'
    assert resolved == [(family, {'model_key': family + '_v1'})]
    expected = 'plaque' if family == 'toxoplasma_plaque' else 'detector' if family == 'toxoplasma_well_detector' else 'masks'
    assert called[0][0] == expected
    assert called[0][1].get('family') == (family if expected == 'masks' else None)
    assert not list(instance.job_parent.iterdir())
