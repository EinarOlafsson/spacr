"""Model pins, curated cards and the shared cache: the edges no run reaches.

Run 38971984143 named these as uncovered in :mod:`spacr.model_zoo`: a model
pin that is malformed or names two catalogue rows, a curated card that is
missing or publishes fewer counts, a pinned file that is gone, the offline
catalogue's duplicate rows, and the download cache's lock and versioned
copies. Each test drives one edge through the function that owns it and
checks what it returns or refuses. No test reaches the network.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from copy import deepcopy
from types import SimpleNamespace

import pytest

from spacr import model_cards
from spacr import model_zoo as mz

_BOUND = "toxoplasma_plaque_v1"


def _raw(key=_BOUND):
    """The bundled declaration for ``key``."""
    return next(row for row in mz.BUNDLED_REMOTE_MODELS if row.get("key") == key)


def _pinned(raw, setting="cellpose_model"):
    """Settings that pin ``setting`` to the catalogue row ``raw`` by its key."""
    return {setting: raw["key"], "model_metadata": {"schema_version": 1, "settings": {
        setting: mz._model_reference_from_record(raw["key"], raw)}}}


def _record(**changes):
    """One well-formed model_metadata record, with ``changes`` applied."""
    record = {"requested_value": "/models/cells.pt", "key": "cells", "version": "1",
              "version_status": "declared", "sha256": "a" * 64, "kind": "cellpose",
              "filename": "cells.pt"}
    record.update(changes)
    return record


# -- curated cards -------------------------------------------------------------

def test_a_missing_curated_card_leaves_the_declared_measurements(monkeypatch, tmp_path):
    raw = _raw()
    monkeypatch.setattr(model_cards, "_CARD_DIR", tmp_path)
    entry = mz._builtin_entry_from_mapping(raw)
    assert entry.metrics == mz._entry_from_mapping(raw).metrics
    assert "scorecard_provenance" not in entry.metrics


def test_counts_a_card_does_not_publish_are_left_out_not_invented(monkeypatch, tmp_path):
    raw = _raw()
    binding = mz._CURATED_SCORECARD_BINDINGS[_BOUND]
    card = json.loads((model_cards._CARD_DIR / binding["resource"]).read_bytes())
    del card["scores"]["counts"]["this"]["fn"]
    payload = json.dumps(card).encode("utf-8")
    (tmp_path / binding["resource"]).write_bytes(payload)
    monkeypatch.setattr(model_cards, "_CARD_DIR", tmp_path)
    monkeypatch.setitem(mz._CURATED_SCORECARD_BINDINGS, _BOUND, dict(
        binding, resource_sha256=hashlib.sha256(payload).hexdigest()))
    metrics = mz._builtin_entry_from_mapping(raw).metrics
    assert metrics["true_positives"] == card["scores"]["counts"]["this"]["tp"]
    assert metrics["false_positives"] == card["scores"]["counts"]["this"]["fp"]
    assert "false_negatives" not in metrics


def test_a_scorecard_row_without_a_stock_number_shows_only_the_measurement():
    metrics = mz._builtin_entry_from_mapping(_raw()).metrics
    metrics = deepcopy(metrics)
    metrics["aji"]["vanilla"] = None
    lines = mz._curated_scorecard_lines(metrics)
    aji = next(line for line in lines if line.startswith("AJI"))
    assert aji == f"AJI: {metrics['aji']['finetuned']:.4f}"
    assert "stock" in next(line for line in lines if line.startswith("Dice"))


def test_the_scorecard_export_defaults_to_every_builtin_declaration():
    rows = mz.scorecard_export_rows()
    keys = {row["key"] for row in rows}
    assert keys == {row.get("key") or row.get("name") for row in mz.BUNDLED_REMOTE_MODELS}
    assert any(row["key"] == _BOUND and row["metric"] for row in rows)


# -- recorded identities ------------------------------------------------------

def test_a_key_naming_two_catalogue_identities_is_refused(monkeypatch):
    raw = _raw()
    twin = dict(raw, sha256="b" * 64)
    monkeypatch.setattr(mz, "BUNDLED_REMOTE_MODELS", [raw, dict(raw), twin])
    with pytest.raises(mz.ModelZooError, match="one exact catalogue identity"):
        mz._capture_model_metadata("ops", {"cellpose_model": raw["key"]})


def test_a_key_listed_twice_with_one_identity_is_pinned_once(monkeypatch):
    raw = _raw()
    monkeypatch.setattr(mz, "BUNDLED_REMOTE_MODELS", [raw, dict(raw)])
    captured = mz._capture_model_metadata("ops", {"cellpose_model": raw["key"]})
    assert captured["model_metadata"]["settings"]["cellpose_model"] == \
        mz._model_reference_from_record(raw["key"], raw)


@pytest.mark.parametrize("settings, field", [
    (["not", "a", "mapping"], "settings"),
    ({"model_metadata": {"schema_version": 1, "settings": {"": _record()}}}, "model_metadata.settings['']"),
    ({"model": "/models/cells.pt", "model_metadata": {"schema_version": 1, "settings": {
        "model": _record(key=7)}}}, "model_metadata.settings['model'].key"),
])
def test_malformed_model_metadata_names_the_field_that_is_wrong(settings, field):
    with pytest.raises(mz.ModelZooError, match=field.replace("[", r"\[").replace("]", r"\]")):
        mz._validate_model_metadata_structure(settings)


def test_one_accepted_kind_may_be_named_without_a_tuple():
    settings = {"model": "/models/cells.pt", "model_metadata": {"schema_version": 1,
                                                                "settings": {"model": _record()}}}
    checked = mz._validate_model_metadata_structure(settings, model_kinds={"model": "cellpose"})
    assert checked["settings"]["model"]["kind"] == "cellpose"
    with pytest.raises(mz.ModelZooError, match="kind"):
        mz._validate_model_metadata_structure(settings, model_kinds={"model": "classifier"})


def test_a_catalogue_row_must_be_a_mapping_with_a_text_version():
    with pytest.raises(mz.ModelZooError, match="catalogue"):
        mz._model_reference_from_record("cells", ["cells"])
    with pytest.raises(mz.ModelZooError, match="catalogue.version"):
        mz._model_reference_from_record("cells", dict(_raw(), version=7))


def test_a_reference_that_is_not_a_mapping_matches_nothing():
    with pytest.raises(mz.ModelZooError, match="model_metadata"):
        mz._match_model_reference(("cells",))


def test_a_verified_reference_to_a_file_that_is_gone_is_refused(tmp_path):
    raw = _raw()
    entry = mz._entry_from_mapping(raw)
    with pytest.raises(mz.ModelZooError, match="sha256="):
        mz._verified_model_reference(entry, tmp_path / raw["name"])


def test_a_pinned_model_the_cache_cannot_supply_is_refused(monkeypatch):
    raw = _raw()
    monkeypatch.setattr(mz, "_ensure_catalogue_model_file", lambda *a, **k: None)
    with pytest.raises(mz.ModelZooError, match=raw["sha256"]):
        mz._resolve_pinned_model_file(_pinned(raw), "cellpose_model", download=False)


def test_a_pinned_download_that_left_no_file_is_refused(monkeypatch, tmp_path):
    raw = _raw()
    missing = tmp_path / raw["name"]
    monkeypatch.setattr(mz, "_ensure_catalogue_model_file", lambda *a, **k: str(missing))
    with pytest.raises(mz.ModelZooError, match=str(missing)):
        mz._resolve_pinned_model_file(_pinned(raw), "cellpose_model", download=True)
    assert mz._resolve_pinned_model_file(_pinned(raw), "cellpose_model", download=False) == missing


def test_the_offline_catalogue_lists_each_identity_once_and_no_retired_name(monkeypatch):
    raw = _raw()
    retired = dict(raw, key="retired_model", name=next(iter(mz.RETIRED_MODEL_NAMES)))
    monkeypatch.setattr(mz, "BUNDLED_REMOTE_MODELS", [raw, dict(raw), retired])
    monkeypatch.setitem(mz._SHARED_CATALOGUE_CACHE, "entries", [])
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "0")
    monkeypatch.delenv(mz.CATALOGUE_ENV_VAR, raising=False)
    entries = mz._resolution_model_entries(download=True)
    assert [(entry.key, entry.name) for entry in entries] == [(raw["key"], raw["name"])]


# -- the shared download cache --------------------------------------------------

def test_a_lock_heartbeat_stops_when_its_lock_file_is_gone(tmp_path):
    stop = threading.Event()
    worker = threading.Thread(target=mz._refresh_model_download_lock,
                              args=(tmp_path / "gone.lock", stop, 0.001))
    worker.start()
    worker.join(5)
    assert not worker.is_alive()
    assert not stop.is_set()


def test_a_missing_lock_has_no_signature(tmp_path):
    assert mz._model_lock_signature(tmp_path / "gone.lock") is None
    lock = tmp_path / "held.lock"
    lock.write_bytes(b"1")
    assert mz._model_lock_signature(lock)[3] == 1


def _cache(monkeypatch, tmp_path, payload=b"verified weights"):
    """Point the zoo at an empty cache and a fetch that writes ``payload``."""
    cache = tmp_path / "models"
    fetched = []

    def fetch(entry, destination):
        fetched.append(entry.key)
        target = destination / entry.name
        target.write_bytes(payload)
        return target

    monkeypatch.setattr(mz, "_model_cache_root", lambda: cache)
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    monkeypatch.setattr(mz, "fetch", fetch)
    monkeypatch.delenv("SPACR_MODEL_DOWNLOAD", raising=False)
    return cache, fetched


def test_a_model_without_a_checksum_is_downloaded_rather_than_matched(monkeypatch, tmp_path):
    cache, fetched = _cache(monkeypatch, tmp_path)
    entry = mz.ModelEntry(key="cells_r1", name="cells_r1.CP_model", kind="cellpose",
                          uri="https://example.invalid/cells_r1", sha256="")
    path = mz._ensure_catalogue_model_file(entry)
    assert fetched == ["cells_r1"]
    assert path == cache / "cells_r1.CP_model"


def test_a_versioned_copy_with_other_bytes_is_not_taken_for_the_model(monkeypatch, tmp_path):
    payload = b"verified weights"
    cache, fetched = _cache(monkeypatch, tmp_path, payload)
    cache.mkdir()
    (cache / "cells_r1_v2.CP_model").write_bytes(b"another model's weights")
    entry = mz.ModelEntry(key="cells_r1", name="cells_r1.CP_model", kind="cellpose",
                          uri="https://example.invalid/cells_r1",
                          sha256=hashlib.sha256(payload).hexdigest())
    path = mz._ensure_catalogue_model_file(entry)
    assert fetched == ["cells_r1"]
    assert path.read_bytes() == payload
    assert (cache / "cells_r1_v2.CP_model").read_bytes() == b"another model's weights"


def test_a_claim_whose_rename_fails_after_its_reservation_vanished_still_fails(monkeypatch, tmp_path):
    temp = tmp_path / "download.part"
    temp.write_bytes(b"weights")
    real_replace = os.replace

    def refuse(source, target, *args, **kwargs):
        if os.fspath(source) != str(temp):
            return real_replace(source, target, *args, **kwargs)
        os.unlink(target)
        raise PermissionError("the cache became read-only")

    monkeypatch.setattr(mz.os, "replace", refuse)
    with pytest.raises(PermissionError, match="read-only"):
        mz._claim(temp, tmp_path, "cells.CP_model")
    monkeypatch.setattr(mz.os, "replace", real_replace)
    assert temp.read_bytes() == b"weights"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["download.part"]
