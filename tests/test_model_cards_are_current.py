"""The model cards in docs/models match the zoo records they are built from.

Every number on a card comes from ``BUNDLED_REMOTE_MODELS``; a re-score in
the zoo must fail this test until ``python tools/build_model_cards.py`` is
run again.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _cards():
    path = ROOT / "tools" / "build_model_cards.py"
    spec = importlib.util.spec_from_file_location("spacr_model_cards", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_card_is_current():
    cards = _cards()
    assert cards.stale_files() == [], (
        "docs/models is stale; run python tools/build_model_cards.py")


def test_every_published_model_has_a_card_with_its_numbers():
    cards = _cards()
    for entry in cards.published_models():
        key = cards.key_of(entry)
        text = (cards.CARD_DIR / f"{key}.md").read_text(encoding="utf-8")
        versus = str(entry.get("versus_stock") or "").strip().rstrip(".")
        assert versus[1:] in text, key
        assert str(entry["sha256"]) in text, key
        assert "## How to read these scores" in text, key


def test_how_to_read_fits_the_card():
    cards = _cards()
    by_key = {cards.key_of(e): e for e in cards.published_models()}
    detector = "\n".join(cards.how_to_read(by_key["toxoplasma_well_detector_v2"]))
    assert "mAP50" in detector and "**F1**" not in detector
    segmenter = "\n".join(cards.how_to_read(by_key["toxoplasma_pv_v4"]))
    for term in ("**Precision**", "**Recall**", "**F1**", "**IoU**"):
        assert term in segmenter, term
    assert "**SD**" in segmenter
    single = "\n".join(cards.how_to_read(by_key["toxoplasma_from_cellmask_v1"]))
    assert "**SD**" not in single


def test_card_links_resolve():
    cards = _cards()
    for path in cards.CARD_DIR.glob("*.md"):
        for target in re.findall(r"\]\(([^)#:]+\.md)\)",
                                 path.read_text(encoding="utf-8")):
            assert (path.parent / target).is_file(), (path.name, target)


def test_family_defaults_match_the_zoo_markers():
    cards = _cards()
    defaults = {cards.key_of(e) for e in cards.default_records()}
    for entry in cards.published_models():
        if cards.marker_of(entry):
            assert cards.key_of(entry) not in defaults or all(
                cards.marker_of(other) for other in cards.published_models()
                if cards.family_and_version(cards.key_of(other))[0]
                == cards.family_and_version(cards.key_of(entry))[0])
