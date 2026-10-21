"""Accepted exact-checkpoint cards reach builtin descriptions without rescoring."""
from copy import deepcopy
import json

import pytest

from spacr import model_cards, model_zoo as zoo


_KEYS = tuple(json.loads(path.read_text(encoding="utf-8"))["key"] for path in sorted(model_cards._CARD_DIR.glob("*.json")))
_FIELDS = (("f1", "object_f1"), ("boundary_f1_2px", "boundary_f1"),
           ("precision", "precision"), ("recall", "recall"), ("aji", "aji"), ("dice", "dice"))


def _construct(raw):
    constructor = getattr(zoo, "_builtin_entry_from_mapping", zoo._entry_from_mapping)
    return constructor(raw)


def _raw(key="toxoplasma_pv_v4"):
    return next(record for record in zoo.BUNDLED_REMOTE_MODELS if record["key"] == key)


@pytest.mark.parametrize("key", _KEYS)
def test_exact_builtin_preserves_all_six_approved_matched_scores_and_dataset(monkeypatch, key):
    from spacr.qt import i18n
    monkeypatch.setattr(i18n, "current_language", lambda: "en")
    raw = _raw(key)
    before = deepcopy(raw)
    card = model_cards._cards()[key]
    scores = card["scores"]
    rows = {row["key"]: row for row in scores["rows"]}
    entry = _construct(raw)
    for field, source in _FIELDS:
        assert entry.metrics[field]["finetuned"] == rows[source]["this"]
        assert entry.metrics[field]["vanilla"] == rows[source]["stock"]
        assert entry.metrics[field]["delta"] == rows[source]["this"] - rows[source]["stock"]
    assert entry.metrics["n_fields"] == scores["holdout"]["n_fields"]
    assert entry.metrics["n_objects"] == scores["holdout"]["n_objects"]
    assert entry.metrics["historical_scorecards"][0]["metrics"] == before["metrics"]
    assert entry.metrics["scorecard_provenance"]["boundary_tolerance_px"] == 2
    assert entry.metrics["scorecard_provenance"]["stock_model"] == scores["stock_model"]
    assert "boundary_f1_1px" not in entry.metrics
    tooltip = zoo.scorecard_html(entry)
    assert tooltip.count("<tr>") == 7
    for label in ("Boundary F1 (2 px)", "Precision", "Recall", "AJI", "Dice"):
        assert label in tooltip
    assert scores["holdout"].get("set", "gt_rescore_20261011") in tooltip
    assert str(scores["holdout"]["n_fields"]) in tooltip
    assert "Boundary F1 (2 px)" in "\n".join(entry.scorecard_lines())
    assert "Boundary F1 (2 px)" in entry.describe()
    assert entry.sha256 == raw["sha256"] and entry.name == raw["name"]
    assert raw == before


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_card_binding_survives_checkout_line_endings(monkeypatch, tmp_path, newline):
    raw = _raw()
    resource = zoo._CURATED_SCORECARD_BINDINGS[raw["key"]]["resource"]
    payload = (model_cards._CARD_DIR / resource).read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / resource).write_bytes(payload.replace(b"\n", newline))
    monkeypatch.setattr(model_cards, "_CARD_DIR", tmp_path)
    entry = _construct(raw)
    assert entry.metrics["scorecard_provenance"]["boundary_tolerance_px"] == 2
    tampered = payload.replace(b"\n", newline).replace(b'"', b"'", 1)
    (tmp_path / resource).write_bytes(tampered)
    assert "scorecard_provenance" not in _construct(raw).metrics


@pytest.mark.parametrize("change", ["custom", "local", "sha", "uri", "unknown"])
def test_other_entries_keep_their_own_measurements(monkeypatch, change):
    raw = deepcopy(_raw())
    raw["metrics"] = {"f1": {"finetuned": 0.125, "vanilla": 0.25}, "holdout": "own-dataset"}
    if change == "local":
        raw.update(source="local", path="/custom/checkpoint")
    elif change == "sha":
        raw["sha256"] = "0" * 64
        monkeypatch.setattr(zoo, "BUNDLED_REMOTE_MODELS", (raw,))
    elif change == "uri":
        raw["uri"] = "https://example.invalid/different-checkpoint"
        monkeypatch.setattr(zoo, "BUNDLED_REMOTE_MODELS", (raw,))
    elif change == "unknown":
        raw["key"] = "unapproved"
        monkeypatch.setattr(zoo, "BUNDLED_REMOTE_MODELS", (raw,))
    before = deepcopy(raw)
    assert _construct(raw).metrics == before["metrics"]
    assert zoo._entry_from_mapping(raw).metrics == before["metrics"]
    assert raw == before


def test_changed_packaged_card_is_not_attached_to_approved_weights(monkeypatch, tmp_path):
    raw = _raw()
    repo = raw["repo_id"].split("/")[-1]
    card = deepcopy(model_cards._cards()[raw["key"]])
    card["scores"]["rows"][0]["this"] = 0.999
    (tmp_path / (repo + ".json")).write_text(json.dumps(card), encoding="utf-8")
    monkeypatch.setattr(model_cards, "_CARD_DIR", tmp_path)
    assert _construct(raw).metrics == raw["metrics"]


def test_generic_one_pixel_boundary_and_description_are_unchanged():
    raw = deepcopy(_raw())
    raw["source"] = "local"
    raw["metrics"] = {"boundary_f1_1px": {"finetuned": 0.5, "vanilla": 0.25}, "note": "custom"}
    entry = zoo._entry_from_mapping(raw)
    assert entry.metrics == raw["metrics"]
    assert entry.scorecard_lines() == ["Boundary F1 0.500 (+0.250 vs stock)"]
    assert "boundary_f1_1px" in entry.describe()
    assert "Boundary F1 (2 px)" not in zoo.scorecard_html(entry)


def test_serialized_curated_flag_and_provenance_cannot_activate_custom_formatting():
    raw = deepcopy(_raw())
    raw["_curated_card"] = True
    raw["metrics"] = {"f1": {"finetuned": 0.125, "vanilla": 0.25},
                      "scorecard_provenance": {"schema": "spacr-approved-curated-card-v1"}}
    entry = zoo._entry_from_mapping(raw)
    assert not entry._curated_card
    assert entry.metrics == raw["metrics"]
    assert "Boundary F1 (2 px)" not in zoo.scorecard_html(entry)
    assert entry.scorecard_lines() == ["F1 0.125 (-0.125 vs stock)"]


def test_verified_copy_preserves_bound_card_and_generic_identity_equality():
    from dataclasses import replace
    raw = _raw()
    bound = _construct(raw)
    verified = replace(bound, path="/private/verified-checkpoint", verified=True)
    assert verified._curated_card and "Boundary F1 (2 px)" in verified.describe()
    assert bound == zoo._entry_from_mapping(raw)
    assert "_curated_card" not in repr(bound)


def test_every_pinned_digest_is_of_the_committed_lf_bytes():
    """Line-ending normalisation is a no-op where the checkout keeps LF: every pin is of the raw file."""
    import hashlib

    for key, binding in zoo._CURATED_SCORECARD_BINDINGS.items():
        raw = (model_cards._CARD_DIR / binding["resource"]).read_bytes()
        normalised = raw.replace(b"\r\n", b"\n")
        assert hashlib.sha256(normalised).hexdigest() == binding["resource_sha256"], key
        if b"\r" not in raw:
            assert hashlib.sha256(raw).hexdigest() == binding["resource_sha256"], key
