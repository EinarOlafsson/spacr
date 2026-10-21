"""Documentation uses the same exact-checkpoint scorecard as the builtin UI."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from spacr import model_cards, model_zoo
from tools import build_model_cards, build_score_cards

ROOT = Path(__file__).resolve().parents[1]
KEYS = tuple(model_cards._cards())
METRICS = ("f1", "boundary_f1_2px", "precision", "recall", "aji", "dice")


def raw(key="toxoplasma_pv_v4"):
    return next(row for row in model_zoo.BUNDLED_REMOTE_MODELS if row["key"] == key)


def visuals():
    spec = importlib.util.spec_from_file_location(
        "f370_readme_visuals", ROOT / "packaging/generate_readme_visuals.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", KEYS)
def test_generator_uses_all_exact_builtin_measurements_and_provenance(key):
    declaration = raw(key)
    before = deepcopy(declaration)
    expected = model_zoo._builtin_entry_from_mapping(declaration)
    record = build_score_cards.rescored(declaration)
    assert record["metrics"] == expected.metrics
    assert list(record["scorecard_pages"]) == model_cards._cards()[key]["pages"]
    assert all(record["metrics"][field] == expected.metrics[field] for field in METRICS)
    assert record["metrics"]["historical_scorecards"][0]["metrics"] == before["metrics"]
    assert record["metrics"]["scorecard_provenance"]["boundary_tolerance_px"] == 2
    assert "boundary_f1_1px" not in record["metrics"]
    assert declaration == before
    record["metrics"]["historical_scorecards"][0]["metrics"]["f1"] = "changed copy"
    assert declaration == before


@pytest.mark.parametrize("kind", ["same-key-copy", "different-checkpoint", "different-uri", "local"])
def test_generator_never_attaches_approved_scores_by_key_alone(kind):
    declaration = deepcopy(raw())
    declaration["metrics"] = {"f1": {"finetuned": .1, "vanilla": .2}, "holdout": "custom"}
    if kind == "different-checkpoint":
        declaration["sha256"] = "0" * 64
    elif kind == "different-uri":
        declaration["uri"] = "https://example.invalid/different-checkpoint"
    elif kind == "local":
        declaration.update(source="local", path="/custom/checkpoint")
    before = deepcopy(declaration)
    assert build_score_cards.rescored(declaration) == before
    assert declaration == before


def test_generator_refuses_changed_packaged_card(monkeypatch, tmp_path):
    declaration = raw()
    card = deepcopy(model_cards._cards()[declaration["key"]])
    card["scores"]["rows"][0]["this"] = .999
    resource = model_zoo._CURATED_SCORECARD_BINDINGS[declaration["key"]]["resource"]
    import json
    (tmp_path / resource).write_text(json.dumps(card), encoding="utf-8")
    monkeypatch.setattr(model_cards, "_CARD_DIR", tmp_path)
    assert build_score_cards.rescored(declaration) == declaration


def test_markdown_scorecard_names_two_pixel_metric_and_exact_current_counts():
    record = next(row for row in build_model_cards.published_models() if row["key"] == "toxoplasma_pv_v4")
    table = "\n".join(build_model_cards._scorecard(record["metrics"]))
    assert "| Boundary F1 (2 px) | 0.8053 | 0.2229 |" in table
    assert "209 scored fields" in table
    assert "1 px" not in table
    assert len([line for line in table.splitlines() if line.startswith("| ")]) == 8


def test_api_sections_use_approved_card_pages_not_absent_root_exports(monkeypatch):
    module = visuals()
    records = build_model_cards.published_models()
    approved = [row for row in records if row["key"] in KEYS]
    monkeypatch.setattr(module, "_published_models", lambda: approved)
    text = module._model_zoo_sections()
    for entry in approved:
        repo = entry["repo_id"]
        for page in ("page1", "page2", "page3"):
            assert f"https://huggingface.co/{repo}/resolve/main/cards/{page}.png" in text
        assert f"https://huggingface.co/{repo}/resolve/main/scorecard.png" not in text
        assert f"https://huggingface.co/{repo}/blob/main/scorecard.csv" not in text
    assert "Boundary F1 (2 px)" in text
    assert "209 scored fields" in text
    assert "boundary_f1_1px" not in text


def test_both_generators_keep_raw_declaration_identity_until_binding():
    readme_records = {row["key"]: row for row in visuals()._published_models()}
    card_records = {row["key"]: row for row in build_model_cards.published_models()}
    for key in KEYS:
        expected = model_zoo._builtin_entry_from_mapping(raw(key)).metrics
        assert readme_records[key]["metrics"] == expected
        assert card_records[key]["metrics"] == expected


def test_archived_metrics_do_not_become_current_glossary_claims():
    record = build_score_cards.rescored(raw())
    assert "mAP" in record["metrics"]["historical_scorecards"][0]["metrics"]
    current = "\n".join(build_model_cards.how_to_read(record))
    assert "**AP**" not in current
    assert "**mAP**" not in current
    assert "**F1**" in current and "**AJI**" in current
