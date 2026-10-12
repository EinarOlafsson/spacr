"""Every published model has an SVG score card, and the READMEs show them.

The maintainer asked on 2026-10-12 for visual score cards: one graphic per
model with its main hold-out score, bars against stock Cellpose-SAM (or the
earlier detector) and the hold-out set. ``tools/build_score_cards.py`` draws
them from ``BUNDLED_REMOTE_MODELS``; ``tools/build_model_cards.py`` writes
them to ``docs/models/cards/<key>.svg``. Later that day the maintainer
rejected the first cards for their brown surface and ragged grid: every card
now has the same size, the same slots and the same number of bar rows, on
spaCR's dark teal.
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
            expected = scores.render_svg(entry, family["status"][key])
            assert (scores.without_font_data(text)
                    == scores.without_font_data(expected)), (
                f"{path.name} is stale; run python tools/build_model_cards.py")
            root = ET.fromstring(text)
            assert root.tag == f"{SVG}svg"
            assert root.find(f"{SVG}title") is not None, key
            assert root.find(f"{SVG}desc") is not None, key
            assert scores.PALETTE["card"] in text, key


def _rows_drawn(text):
    """Bar rows in one card: a model bar, or an em dash for a row the
    record has no number for."""
    bars = len(re.findall(r'<path class="m"', text))
    empty = len(re.findall(r'class="i3" font-size="12.5"[^>]*>\u2014<', text))
    return bars + empty


def test_every_card_has_the_same_size_and_the_same_rows():
    cards = _cards()
    scores = cards.score_cards()
    sizes = set()
    for entry in cards.published_models():
        key = cards.key_of(entry)
        text = (cards.CARD_DIR / scores.card_name(key)).read_text(
            encoding="utf-8")
        root = ET.fromstring(text)
        sizes.add((root.get("width"), root.get("height"), root.get("viewBox")))
        assert len(scores.scores(entry)["rows"]) == scores.ROWS, key
        assert _rows_drawn(text) == scores.ROWS, key
        card = re.search(r'<rect class="card" x="[^"]+" y="[^"]+"'
                         r' width="([^"]+)" height="([^"]+)"', text)
        assert card and card.groups() == (f"{scores.CARD_W - 1}",
                                          f"{scores.CARD_H - 1}"), key
    assert sizes == {(str(scores.WIDTH), str(scores.HEIGHT),
                      f"0 0 {scores.WIDTH} {scores.HEIGHT}")}


def test_the_card_face_is_spacr_teal_with_no_warm_grey():
    cards = _cards()
    scores = cards.score_cards()

    def warm(colour):
        r, _g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        return r > b

    assert warm("#1a1a19") and warm("#fcfcfb"), "the rejected palette"

    for entry in cards.published_models():
        text = (cards.CARD_DIR / scores.card_name(cards.key_of(entry))
                ).read_text(encoding="utf-8")
        for colour in set(re.findall(r"#[0-9a-fA-F]{6}\b", text)):
            assert not warm(colour), (cards.key_of(entry), colour)


def test_the_cards_embed_open_sans_for_every_character_they_draw():
    """2026-10-12: titles and the version tag in Open Sans Regular, all other
    text in Open Sans Light, embedded because an SVG shown through <img>
    cannot load a web font."""
    import base64
    import io
    from html import unescape

    from fontTools.ttLib import TTFont

    cards = _cards()
    scores = cards.score_cards()
    names = [scores.card_name(cards.key_of(e)) for e in cards.published_models()]
    for name in names + [scores.HEADING_NAME]:
        text = (cards.CARD_DIR / name).read_text(encoding="utf-8")
        faces = dict(re.findall(
            r"font-weight:(\d+);src:url\(data:font/woff;base64,([^)]+)\)", text))
        drawn = {}
        for weight, value in re.findall(
                r'<text [^>]*font-weight="(\d+)"[^>]*>([^<]*)</text>', text):
            drawn.setdefault(weight, set()).update(unescape(value))
        assert set(drawn) <= {"300", "400"}, name
        assert set(drawn) == set(faces), name
        for weight, characters in drawn.items():
            font = TTFont(io.BytesIO(base64.b64decode(faces[weight])))
            family = font["name"].getDebugName(1)
            assert family.startswith("Open Sans") and (
                font["OS/2"].usWeightClass == int(weight)), (name, weight)
            missing = {c for c in characters - {" "}
                       if ord(c) not in font.getBestCmap()}
            assert not missing, (name, weight, missing)
        assert len(text) < 40_000, (name, len(text))


def test_the_grid_heading_is_current():
    cards = _cards()
    scores = cards.score_cards()
    path = cards.CARD_DIR / scores.HEADING_NAME
    assert (scores.without_font_data(path.read_text(encoding="utf-8"))
            == scores.without_font_data(scores.render_heading()))


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
        assert all(mine is not None for _l, mine, _t in card["rows"]), (
            cards.key_of(entry))
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
        grid = block.strip().partition("\n\n")[0]
        assert grid.startswith("| |zoo_heading|"), readme.name
        assert "`" not in grid, (readme.name, "a text link in the card grid")
        heading = cards.score_cards().HEADING_NAME
        assert f"image:: {prefix}{heading}" in block, readme.name
        assert block.count(":width: 49.5%") == len(defaults), readme.name
