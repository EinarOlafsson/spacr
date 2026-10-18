"""Item 370: every published model shows a scorecard state, and one export.

Each zoo row spaCR publishes is in one of four scorecard states (reviewed,
held-out, recorded at publication, not yet evaluated). A row that is not
reviewed says so in its tooltip and card instead of showing nothing, and
``docs/models/scorecards.csv`` carries every published scorecard with the
stock model's value on the same fields. The card JSON, the zoo entries, the
CSV and the SVG score cards must agree.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
from dataclasses import replace
from pathlib import Path

import pytest

from spacr import model_zoo as zoo

ROOT = Path(__file__).resolve().parents[1]
STATES = {"reviewed", "held_out", "recorded", "not_evaluated"}


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def cards():
    return _load("spacr_model_cards_f370_states", "tools/build_model_cards.py")


def _record(key):
    return next(r for r in zoo.BUNDLED_REMOTE_MODELS if r.get("key") == key)


def _rows(entry):
    return {row["metric"]: row for row in zoo.scorecard_export_rows([entry])}


def test_every_builtin_row_has_a_known_state():
    for record in zoo.BUNDLED_REMOTE_MODELS:
        entry = zoo._builtin_entry_from_mapping(record)
        assert zoo.scorecard_state(entry) in STATES, entry.key


def test_every_published_model_carries_a_measurement(cards):
    for record in cards.published_models():
        entry = cards.bound_entry(record)
        assert zoo.scorecard_state(entry) != "not_evaluated", entry.key
        assert zoo.scorecard_html(entry) or zoo.scorecard_state_text(entry), entry.key


def test_the_nine_reviewed_cards_stay_reviewed():
    reviewed = {zoo._builtin_entry_from_mapping(r).key for r in zoo.BUNDLED_REMOTE_MODELS
                if zoo.scorecard_state(zoo._builtin_entry_from_mapping(r)) == "reviewed"}
    assert reviewed == {f"toxoplasma_pv_v{i}" for i in range(1, 6)} | {
        f"toxoplasma_plaque_v{i}" for i in range(1, 6)}


def test_an_unevaluated_spacr_model_says_so_and_a_stock_model_does_not():
    entry = replace(zoo._builtin_entry_from_mapping(_record("toxoplasma_from_hoechst_v1")),
                    metrics={})
    assert zoo.scorecard_state(entry) == "not_evaluated"
    text = zoo.scorecard_state_text(entry)
    assert "not yet evaluated" in text
    assert text in zoo._scorecard_state_html(entry)
    rows = zoo.scorecard_export_rows([entry])
    assert len(rows) == 1 and rows[0]["state"] == "not_evaluated" and rows[0]["metric"] == ""
    stock = zoo.stock_cellpose_entries()[0]
    assert zoo.scorecard_state_text(stock) == ""
    assert zoo._scorecard_state_html(stock) == ""


def test_recorded_rows_name_their_state_and_reviewed_rows_do_not():
    recorded = zoo._builtin_entry_from_mapping(_record("cell_from_hoechst_v1"))
    assert zoo.scorecard_state(recorded) == "recorded"
    assert "recorded with the model" in zoo.scorecard_state_text(recorded)
    reviewed = zoo._builtin_entry_from_mapping(_record("toxoplasma_pv_v4"))
    assert zoo.scorecard_state_text(reviewed) == ""


def test_the_gui_tooltip_carries_the_state():
    from spacr.qt.screens.model_zoo import _tooltip_for

    recorded = zoo._builtin_entry_from_mapping(_record("live_cell_v1"))
    assert zoo._scorecard_state_html(recorded) in _tooltip_for(recorded)
    empty = replace(recorded, metrics={})
    tip = _tooltip_for(empty)
    assert tip.startswith(zoo.scorecard_state_text(empty))


def test_reviewed_export_rows_equal_the_packaged_card_json():
    names = {"f1": "object_f1", "boundary_f1_2px": "boundary_f1", "precision": "precision",
             "recall": "recall", "aji": "aji", "dice": "dice"}
    paths = sorted((ROOT / "spacr/resources/model_cards").glob("*.json"))
    assert len(paths) == 10
    for path in paths:
        card = json.loads(path.read_text(encoding="utf-8"))
        entry = zoo._builtin_entry_from_mapping(_record(card["key"]))
        rows = _rows(entry)
        assert set(rows) == set(names), card["key"]
        source = {row["key"]: row for row in card["scores"]["rows"]}
        for metric, key in names.items():
            assert rows[metric]["this_model"] == source[key]["this"], (card["key"], metric)
            assert rows[metric]["stock"] == source[key]["stock"], (card["key"], metric)
            assert rows[metric]["baseline"] == card["scores"]["stock_model"]
            assert rows[metric]["n_fields"] == card["scores"]["holdout"]["n_fields"]
        assert rows["boundary_f1_2px"]["boundary_tolerance_px"] == 2


def test_export_agrees_with_the_svg_score_cards(cards):
    scores = cards.score_cards()
    labels = {"F1": "f1", "AJI": "aji", "Dice": "dice", "mAP50": "mAP50",
              "mAP50-95": "mAP50_95"}
    checked = 0
    for record in cards.published_models():
        rows = _rows(cards.bound_entry(record))
        for label, mine, stock in scores.scores(record)["rows"]:
            metric = labels.get(label)
            row = rows.get(metric)
            if row is None or not isinstance(row["this_model"], float):
                continue
            assert row["this_model"] == pytest.approx(mine, abs=5e-4), (record["key"], label)
            if stock is not None and isinstance(row["stock"], float):
                assert row["stock"] == pytest.approx(stock, abs=5e-4), (record["key"], label)
            checked += 1
    assert checked >= 40


def test_recorded_stock_values_match_the_versus_stock_sentence():
    for key, stock in (("nuclei_from_cellmask_v1", 0.201), ("toxoplasma_from_hoechst_v1", 0.002)):
        rows = _rows(zoo._builtin_entry_from_mapping(_record(key)))
        assert rows["f1"]["stock"] == stock
        assert f"{stock:.3f}" in _record(key)["versus_stock"]


def test_the_published_csv_is_current_and_matches_the_index(cards):
    text = (cards.CARD_DIR / cards.SCORECARD_CSV).read_text(encoding="utf-8")
    assert text == cards.render_scorecard_csv()
    rows = list(csv.DictReader(io.StringIO(text)))
    published = {cards.key_of(r): r for r in cards.published_models()}
    assert {row["key"] for row in rows} == set(published)
    index = (cards.CARD_DIR / cards.INDEX_NAME).read_text(encoding="utf-8")
    for row in rows:
        assert row["state"] in STATES
        line = next(line for line in index.splitlines()
                    if line.startswith(f"| [{row['key']}]("))
        assert line.rstrip(" |").endswith(cards.STATE_WORDS[row["state"]]), row["key"]
