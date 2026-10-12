"""Every published model has an SVG score card, and the READMEs show them.

The maintainer asked on 2026-10-12 for visual score cards: one graphic per
model with its main hold-out score, bars against stock Cellpose-SAM (or the
earlier detector) and the hold-out set. ``tools/build_score_cards.py`` draws
them from ``BUNDLED_REMOTE_MODELS``; ``tools/build_model_cards.py`` writes
them to ``docs/models/cards/<key>.svg``.
"""
from __future__ import annotations

import importlib.util
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SVG = "{http://www.w3.org/2000/svg}"


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cards():
    return _load("spacr_model_cards_for_svg", "tools/build_model_cards.py")


def test_every_published_model_has_a_current_score_card():
    cards = _cards()
    scores = cards.score_cards()
    for family in cards.families():
        for entry in family["versions"]:
            key = cards.key_of(entry)
            path = cards.CARD_DIR / scores.card_name(key)
            assert path.is_file(), key
            text = path.read_text(encoding="utf-8")
            assert text == scores.render_svg(entry, family["status"][key]), (
                f"{path.name} is stale; run python tools/build_model_cards.py")
            root = ET.fromstring(text)
            assert root.tag == f"{SVG}svg"
            assert root.find(f"{SVG}title") is not None, key
            assert root.find(f"{SVG}desc") is not None, key
            assert "prefers-color-scheme: dark" in text, key


def test_the_card_numbers_come_from_the_zoo_record():
    cards = _cards()
    scores = cards.score_cards()
    by_key = {cards.key_of(e): e for e in cards.published_models()}
    pv = scores.scores(by_key["toxoplasma_pv_v4"])
    f1 = by_key["toxoplasma_pv_v4"]["metrics"]["f1"]
    assert pv["rows"][0] == ("F1", f1["finetuned"], f1["vanilla"])
    assert pv["baseline"] == "Stock cpsam_v2"
    detector = scores.scores(by_key["toxoplasma_well_detector_v2"])
    assert detector["rows"][0][0] == "mAP50"
    assert detector["baseline"].startswith("v1")
    hoechst = scores.scores(by_key["toxoplasma_from_hoechst_v1"])
    assert hoechst["rows"][0][0] == "F1" and hoechst["rows"][0][2] is not None
    for entry in cards.default_records():
        card = scores.scores(entry)
        assert card["rows"], cards.key_of(entry)
        head = scores.headline(entry, card)
        svg = scores.render_svg(entry)
        assert f">{head[1]:.3f}<" in svg, cards.key_of(entry)


def test_every_markdown_card_opens_with_its_score_card():
    cards = _cards()
    scores = cards.score_cards()
    for entry in cards.published_models():
        key = cards.key_of(entry)
        text = (cards.CARD_DIR / cards.card_file(key)).read_text(encoding="utf-8")
        head = text.split("## ", 1)[0]
        assert f"]({scores.card_name(key)})" in head, key


def test_every_readme_embeds_the_default_score_cards():
    generator = _load("spacr_readme_visuals_for_svg",
                      "packaging/generate_readme_visuals.py")
    cards = generator.build_model_cards
    defaults = [cards.key_of(e) for e in cards.default_records()]
    for readme in generator.README_PATHS:
        text = readme.read_text(encoding="utf-8")
        block = text.partition(generator.MODEL_ZOO_BEGIN)[2]
        block = block.partition(generator.MODEL_ZOO_END)[0]
        prefix = generator._card_prefix(readme)
        for key in defaults:
            image = f"{prefix}cards/{key}.svg"
            assert f"image:: {image}" in block, (readme.name, key)
            assert (readme.parent / image).resolve().is_file(), (
                readme.name, image)
            assert re.search(rf"\|zoo_card_{key}\|", block), (readme.name, key)
