"""The community catalogue: additive, verifiable, and never fatal.

Before this existed, a model reached other spaCR users only by someone editing
``BUNDLED_REMOTE_MODELS`` in the source and cutting a release. That is a high
price for a row of metadata, and it is why the zoo shipped with one entry.

Three properties carry the design, and each is a way this could go wrong:
it must not be able to REDEFINE a shipped model, it must not turn a network
problem into a module that will not open, and a row without a checksum must
still be unusable without an explicit override.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from spacr import model_zoo as mz


@pytest.fixture(autouse=True)
def _no_cache():
    """The cache is module state; a stale one would make these pass in the
    wrong order and fail alone."""
    mz._SHARED_CATALOGUE_CACHE.update(fetched_at=0.0, entries=())
    yield
    mz._SHARED_CATALOGUE_CACHE.update(fetched_at=0.0, entries=())


def _serve(tmp_path, payload):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(payload))
    return path.as_uri()


def test_an_unreachable_catalogue_is_empty_rather_than_an_exception():
    """A laptop on a train must still open the module.

    The list is useful without the shared half -- bundled and local models are
    still there -- so a fetch failure costs a shorter list and a log line.
    """
    assert mz.shared_catalogue(uri="https://invalid.invalid/x.json",
                               timeout=1) == ()


def test_a_corrupt_row_is_dropped_without_taking_the_others(tmp_path):
    """One bad row must not cost the whole catalogue.

    The rows come from other people. A catalogue that is all-or-nothing means
    one contributor's typo removes every other contributor's model.
    """
    uri = _serve(tmp_path, {"models": [
        {"nothing": "usable"},                      # no name and no key
        {"key": "good", "name": "good.CP_model", "kind": "cellpose",
         "uri": "https://example.invalid/good.CP_model", "sha256": "a" * 64},
    ]})
    entries = mz.shared_catalogue(uri=uri)
    assert [e.key for e in entries] == ["good"]


def test_the_shared_catalogue_cannot_redefine_a_shipped_model(tmp_path):
    """THE SAFETY PROPERTY. It may add; it may not replace.

    The shared catalogue is edited by people other than whoever is running the
    code. If a row there could shadow a shipped key, changing which weights a
    published pipeline segments with would need no access to the user's machine
    at all -- just an edit to a shared file.
    """
    shipped = mz._entry_from_mapping(mz.BUNDLED_REMOTE_MODELS[1])
    uri = _serve(tmp_path, {"models": [
        {"key": shipped.key, "name": shipped.name, "kind": "cellpose",
         "uri": "https://example.invalid/impostor", "sha256": "b" * 64},
    ]})
    mz.shared_catalogue(uri=uri)          # prime the cache with the impostor

    entries = mz.catalogue(remote=True)
    matching = [e for e in entries if e.key == shipped.key]
    assert len(matching) == 1, "the impostor must not appear alongside it"
    assert "impostor" not in matching[0].uri, (
        "a shared row replaced a shipped model's location")


def test_a_shared_row_without_a_checksum_is_still_unverifiable(tmp_path):
    """A row is a claim about where a file lives. The hash is what makes it a
    claim about WHICH file. Without one, fetch must still refuse."""
    uri = _serve(tmp_path, {"models": [
        {"key": "unhashed", "name": "unhashed.CP_model", "kind": "cellpose",
         "uri": "https://example.invalid/unhashed.CP_model"},
    ]})
    entry = mz.shared_catalogue(uri=uri)[0]
    assert not entry.sha256
    assert any("cannot be verified" in n for n in entry.notes), (
        "an unverifiable entry must say so on the entry itself")


def test_a_second_call_is_served_from_cache(tmp_path):
    """Opening a module repeatedly must not be a request per open."""
    uri = _serve(tmp_path, {"models": [
        {"key": "k", "name": "k.CP_model", "kind": "cellpose",
         "uri": "https://example.invalid/k", "sha256": "c" * 64}]})
    first = mz.shared_catalogue(uri=uri)
    stamp = mz._SHARED_CATALOGUE_CACHE["fetched_at"]
    second = mz.shared_catalogue(uri=uri)
    assert [e.key for e in first] == [e.key for e in second]
    assert mz._SHARED_CATALOGUE_CACHE["fetched_at"] == stamp, "re-fetched"


def test_publish_model_refuses_a_kind_the_registry_would_reject(tmp_path):
    """Fail at publish, not at the far end.

    ModelEntry validates kind, so a bad one would otherwise be discovered by
    whoever tried to USE the published model.
    """
    checkpoint = tmp_path / "m.CP_model"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValueError, match="kind must be one of"):
        mz.publish_model(checkpoint, "someone/repo", key="k", kind="nonsense")


def test_publish_model_refuses_a_path_that_is_not_a_file(tmp_path):
    with pytest.raises(mz.ModelZooError, match="is not a file"):
        mz.publish_model(tmp_path / "absent", "someone/repo", key="k")


def test_publish_model_names_its_optional_client_when_missing(monkeypatch,
                                                               tmp_path):
    """Publishing without the Hub client says how to enable the operation."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(
            ImportError, match="pip install huggingface_hub") as caught:
        mz.publish_model(tmp_path / "model.pt", "someone/repo", key="k")

    assert isinstance(caught.value.__cause__, ImportError)


def test_publish_model_with_a_client_reaches_input_validation(monkeypatch,
                                                               tmp_path):
    """The positive import path proceeds past dependency discovery."""
    fake = types.ModuleType("huggingface_hub")
    fake.HfApi = object
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)

    with pytest.raises(mz.ModelZooError, match="is not a file"):
        mz.publish_model(tmp_path / "absent", "someone/repo", key="k")


def test_the_retired_plaque_model_is_not_offered():
    """310-style retirement: filtered from the listing, kept on disk.

    ``toxo_plaque_cyto`` recalls 0.631 on the literature set -- it misses about
    a third of the plaques -- and published no checksum, so its row in the
    picker carried a Download button that could never succeed: fetch refuses an
    entry it cannot verify.
    """
    names = {e.name for e in mz.catalogue(remote=True)}
    assert not any(n in mz.RETIRED_MODEL_NAMES for n in names), (
        "a retired model is still offered")


def test_retirement_survives_the_model_still_being_on_disk():
    """The filter has to beat LOCAL DISCOVERY, not only the remote list.

    The checkpoint ships inside the package, so discover_local finds it as a
    real file on this machine. Dropping only the remote entry left it in the
    listing anyway -- which is what happened on the first attempt.
    """
    entries = mz.catalogue(remote=True, include_bundled=True)
    assert all(e.name not in mz.RETIRED_MODEL_NAMES for e in entries)


def test_the_retired_model_is_still_reachable_for_reproducibility():
    """Retired from the menu, not deleted from disk.

    A run recorded against it has to stay reproducible: removing the weights
    would silently change what re-running an old analysis produces, which is
    worse than a model nobody should newly pick. plaque_model='bundled' is the
    documented way back to it.
    """
    import os

    from spacr.submodules import _resolve_plaque_model

    path = _resolve_plaque_model({"plaque_model": "bundled"})
    assert path.endswith(".CP_model")
    assert os.path.isfile(path), "the retired checkpoint must still be there"
