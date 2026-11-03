"""A community model the user downloaded resolves where no fetch may happen.

Reported 2026-11-03: ``toxoplasma_plaque_v7`` exists only in the community
catalogue. The Model Zoo listed and downloaded it, but the plaque preview,
which resolves with ``fetch=False``, answered "no existing checkpoint or exact
model-zoo identity matches", and its list of identities stopped at v6: the
planning index held the bundled rows only, and a new process forgot the
community rows until the next fetch.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from spacr import model_zoo as mz
from spacr import submodules as sm

PAYLOAD = b"a small stand-in for the community plaque checkpoint"
ROW = {
    "key": "toxoplasma_plaque_vtest",
    "name": "cpsam_plaque_vtest_e50",
    "kind": "cellpose",
    "uri": "https://example.invalid/cpsam_plaque_vtest_e50",
    "sha256": hashlib.sha256(PAYLOAD).hexdigest(),
    "version": "1",
    "preprocess": {"well_resize": {"side": 1500, "max_factor": 1500},
                   "thresholds": {"cellprob": 0.0, "flow": 0.4, "min_size": 0},
                   "use_bfloat16": False},
}


@pytest.fixture
def community(tmp_path, monkeypatch):
    """A fetched community catalogue with one row and its checkpoint in the cache."""
    cache = tmp_path / "models"
    monkeypatch.setenv("SPACR_MODEL_CACHE", str(cache))
    monkeypatch.delenv("SPACR_MODEL_DOWNLOAD", raising=False)
    monkeypatch.delenv(mz.CATALOGUE_ENV_VAR, raising=False)
    monkeypatch.setattr(mz, "_shared_catalogue_file",
                        lambda: cache / ".shared-catalogue.json")
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_CACHE",
                        {"fetched_at": 0.0, "entries": ()})
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_PERSISTED",
                        {"stamp": None, "records": ()}, raising=False)
    monkeypatch.setattr(mz, "fetch", lambda *a, **k: pytest.fail(
        "a downloaded community model was fetched again"))
    served = tmp_path / "catalogue.json"
    served.write_text(json.dumps({"models": [ROW]}), encoding="utf-8")
    assert [e.key for e in mz.shared_catalogue(
        uri=served.as_uri(), force=True, block=True)] == [ROW["key"]]
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / ROW["name"]
    checkpoint.write_bytes(PAYLOAD)
    return checkpoint


def _new_process(monkeypatch):
    """Forget everything this process read from the network."""
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_CACHE",
                        {"fetched_at": 0.0, "entries": ()})
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_PERSISTED",
                        {"stamp": None, "records": ()}, raising=False)


def _no_network(monkeypatch):
    """Fail the test on any catalogue fetch."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail(
        "a planning resolution went to the network"))
    monkeypatch.setattr(mz, "_refresh_shared_catalogue_in_background",
                        lambda *a: pytest.fail("a background catalogue fetch"))


def test_the_preview_resolves_a_downloaded_community_model(community, monkeypatch):
    _no_network(monkeypatch)
    settings = {"plaque_model": ROW["key"]}
    assert sm._resolve_plaque_model(dict(settings), fetch=False) == str(community)


def test_a_new_process_still_resolves_it_and_its_preprocessing(community, monkeypatch):
    _new_process(monkeypatch)
    _no_network(monkeypatch)
    assert sm._resolve_plaque_model({"plaque_model": ROW["key"]},
                                    fetch=False) == str(community)
    declared = mz._model_preprocess(ROW["key"])
    assert declared["well_resize"] == {"side": 1500, "max_factor": 1500}
    assert mz._model_preprocess(str(community)) == declared
    assert mz._model_precision_kwargs(str(community), {})["use_bfloat16"] is False


def test_the_run_resolves_it_with_downloads_off(community, monkeypatch):
    _new_process(monkeypatch)
    _no_network(monkeypatch)
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "off")
    assert sm._resolve_plaque_model({"plaque_model": ROW["key"]},
                                    fetch=True) == str(community)


def test_the_run_resolves_it_when_the_catalogue_host_is_down(community, monkeypatch):
    _new_process(monkeypatch)
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    monkeypatch.setattr(mz, "REMOTE_CATALOGUE_URI",
                        (community.parent / "absent.json").as_uri())
    assert sm._resolve_plaque_model({"plaque_model": ROW["key"]},
                                    fetch=True) == str(community)


def test_changed_bytes_are_never_taken_for_the_published_model(community, monkeypatch):
    _new_process(monkeypatch)
    _no_network(monkeypatch)
    community.write_bytes(b"not the published checkpoint")
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "off")
    with pytest.raises(mz.ModelUnreadable):
        sm._resolve_plaque_model({"plaque_model": ROW["key"]}, fetch=True)


def test_the_diagnostic_lists_the_identities_the_resolver_accepts(community, monkeypatch):
    _new_process(monkeypatch)
    _no_network(monkeypatch)
    with pytest.raises(mz.ModelZooError) as error:
        sm._resolve_plaque_model({"plaque_model": "toxoplasma_plaque_absent"},
                                 fetch=False)
    message = str(error.value)
    assert f"{ROW['key']}@{ROW['version']!r} [sha256={ROW['sha256']}]" in message
    assert "toxoplasma_plaque_v6@" in message
    assert mz._model_resolution_context(
        "x", setting="plaque_model", kinds=("detector",)).count(ROW["key"]) == 0
