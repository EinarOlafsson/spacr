"""toxoplasma_plaque_v9 is the default plaque model; V8 epoch 50 is toxoplasma_plaque_v8.

The maintainer, 2026-11-10: "why is the plaque model v6 on the spacr website,
nightly and main readme? and why is v8 not available in the model zoo? add V8,
make v9 default." The V8 epoch-50 checkpoint had been published under the key
``toxoplasma_plaque_v7``; it is ``toxoplasma_plaque_v8`` now, and settings
files or recorded pins that name the old key still resolve to the same file.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from spacr import model_zoo as mz

V8_SHA = "b4f4a9e9d026f44c597c2a865cd35ce66cbde4293cc173935720c34daa1dec73"
V9_SHA = "8f58d3744fd7b3bcf0fc7cf9f9ecc44030da03b911163f70af689ca9804b8f35"


def _record(key):
    return next(r for r in mz.BUNDLED_REMOTE_MODELS if r["key"] == key)


def _plaque_family():
    return sorted((r for r in mz.BUNDLED_REMOTE_MODELS
                   if r["key"].startswith("toxoplasma_plaque_v")),
                  key=lambda r: int(r["key"].rsplit("v", 1)[1]), reverse=True)


@pytest.fixture
def offline_cache(tmp_path, monkeypatch):
    """An empty model cache with downloads off and no community rows."""
    monkeypatch.setenv("SPACR_MODEL_CACHE", str(tmp_path / "models"))
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "off")
    monkeypatch.delenv(mz.CATALOGUE_ENV_VAR, raising=False)
    monkeypatch.setattr(mz, "_known_shared_entries", lambda: ())
    monkeypatch.setattr(mz, "_known_shared_records", lambda: ())
    monkeypatch.setattr(mz, "_on_the_qt_gui_thread", lambda: False)
    return tmp_path / "models"


def test_every_plaque_default_names_v9():
    from spacr import plaque_papers, submodules
    from spacr.settings import get_analyze_plaque_settings

    assert submodules.DEFAULT_PLAQUE_MODEL == "toxoplasma_plaque_v9"
    assert plaque_papers.DEFAULT_SEGMENTER == "toxoplasma_plaque_v9"
    assert get_analyze_plaque_settings({})["plaque_model"] == "toxoplasma_plaque_v9"
    assert submodules._requested_plaque_model({}) == "toxoplasma_plaque_v9"


def test_the_zoo_opens_the_plaque_family_on_v9():
    family = _plaque_family()
    pairs = [(r["key"], SimpleNamespace(key=r["key"])) for r in family]
    assert family[mz.default_version_index(pairs)]["key"] == "toxoplasma_plaque_v9"
    v9 = _record("toxoplasma_plaque_v9")
    assert v9["sha256"] == V9_SHA
    assert "toxoplasma_plaque_v9" not in mz._CANDIDATE_MODEL_KEYS
    assert "candidate" not in v9["display_name"].lower()
    assert "default" in v9["display_name"].lower()
    assert not any("Candidate;" in note for note in v9["notes"])


def test_v6_is_no_longer_called_the_default_and_says_when_it_stopped():
    v6 = _record("toxoplasma_plaque_v6")
    assert "default" not in v6["display_name"].lower()
    assert v6["notes"][0].startswith("The default plaque model 2026-10-25 to 2026-11-10; "
                                     "toxoplasma_plaque_v9 replaced it by the maintainer's choice")


def test_v8_epoch_50_is_listed_as_v8_and_no_record_is_keyed_v7():
    keys = [r["key"] for r in mz.BUNDLED_REMOTE_MODELS]
    assert "toxoplasma_plaque_v7" not in keys
    v8 = _record("toxoplasma_plaque_v8")
    assert v8["sha256"] == V8_SHA and v8["name"] == "cpsam_plaque_v8_e50"
    assert "V8 epoch 50" in v8["display_name"]
    assert "toxoplasma_plaque_v8" in mz._CURATED_SCORECARD_BINDINGS
    assert "toxoplasma_plaque_v7" not in mz._CURATED_SCORECARD_BINDINGS
    entry = mz._builtin_entry_from_mapping(v8)
    assert mz.scorecard_state(entry) == "reviewed"


def test_the_old_key_names_the_same_weights():
    assert mz._canonical_key("toxoplasma_plaque_v7") == "toxoplasma_plaque_v8"
    assert mz._canonical_key(" toxoplasma_plaque_v7 ") == "toxoplasma_plaque_v8"
    assert mz._canonical_key("toxoplasma_plaque_v6") == "toxoplasma_plaque_v6"
    assert mz._canonical_key(None) is None
    entries = [mz._builtin_entry_from_mapping(r) for r in mz.BUNDLED_REMOTE_MODELS]
    old = mz.resolve("toxoplasma_plaque_v7", entries)
    assert old.key == "toxoplasma_plaque_v8" and old.sha256 == V8_SHA
    assert mz._model_preprocess("toxoplasma_plaque_v7") == _record("toxoplasma_plaque_v8")["preprocess"]
    assert mz._model_precision_kwargs("toxoplasma_plaque_v7", {})["use_bfloat16"] is False


def test_an_old_settings_file_resolves_to_the_v8_checkpoint(offline_cache):
    from spacr import submodules

    old = mz._ensure_model_file("toxoplasma_plaque_v7", kinds=("cellpose",), download=False)
    new = mz._ensure_model_file("toxoplasma_plaque_v8", kinds=("cellpose",), download=False)
    assert old is not None and old == new
    assert old.parent == offline_cache and "cpsam_plaque_v8_e50" in old.name
    with pytest.raises(submodules.ModelZooMissing):
        submodules._resolve_plaque_model({"plaque_model": "toxoplasma_plaque_v7"}, fetch=False)


def test_a_run_recorded_with_the_old_key_keeps_its_pin(offline_cache):
    module = SimpleNamespace(key="analyze_plaques")
    pinned = mz._capture_model_metadata(module, {"plaque_model": "toxoplasma_plaque_v7"})
    pin = pinned["model_metadata"]["settings"]["plaque_model"]
    assert pin["key"] == "toxoplasma_plaque_v8" and pin["sha256"] == V8_SHA
    assert pin["requested_value"] == "toxoplasma_plaque_v7"
    recorded = {"plaque_model": "toxoplasma_plaque_v7", "model_metadata": {
        "schema_version": 1, "settings": {"plaque_model": dict(pin, key="toxoplasma_plaque_v7")}}}
    assert mz._validate_model_metadata(recorded) is not None
    path = mz._resolve_pinned_model_file(recorded, "plaque_model", kinds=("cellpose",),
                                         download=False)
    assert path == mz._ensure_model_file("toxoplasma_plaque_v8", download=False)


def test_a_pin_under_the_old_key_with_other_bytes_is_refused(offline_cache):
    module = SimpleNamespace(key="analyze_plaques")
    pin = mz._capture_model_metadata(
        module, {"plaque_model": "toxoplasma_plaque_v8"})["model_metadata"]["settings"]["plaque_model"]
    forged = dict(pin, key="toxoplasma_plaque_v7", sha256=V9_SHA)
    with pytest.raises(mz.ModelZooError):
        mz._match_model_reference(forged)


def test_a_community_row_under_the_old_key_is_not_listed_twice(monkeypatch):
    v8 = _record("toxoplasma_plaque_v8")
    community = mz._entry_from_mapping(dict(v8, key="toxoplasma_plaque_v7"))
    monkeypatch.setattr(mz, "shared_catalogue", lambda **_: [community])
    monkeypatch.setattr(mz, "bioimageio_entries", lambda *a, **k: [])
    listed = [e for e in mz.catalogue(include_bundled=False, include_plugins=False)
              if e.name == "cpsam_plaque_v8_e50"]
    assert [e.key for e in listed] == ["toxoplasma_plaque_v8"]
    extra = mz._unseen_entries([mz._entry_from_mapping(v8)], [community])
    assert extra == []


def test_the_v8_and_v9_cards_carry_their_keys():
    from spacr import model_cards

    model_cards._cards.cache_clear()
    assert model_cards.repo_of_key("toxoplasma_plaque_v8") == (
        "einarolafsson/toxoplasma-plaque-segmentation-cpsam-v8-e50")
    assert "toxoplasma_plaque_v7" not in model_cards.card_keys()
    card = model_cards._cards()["toxoplasma_plaque_v9"]
    assert card["status"] == "default"
    assert card["about"]["status"].startswith("Default plaque model since 2026-11-10")
