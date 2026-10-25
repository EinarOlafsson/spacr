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


def test_score_text_and_every_plotted_value_precede_the_picture():
    from markdown_it import MarkdownIt
    from tests.test_readme_panel_text_destinations import Article

    cards = _cards()
    scores = cards.score_cards()
    renderer = MarkdownIt("commonmark").enable("table")
    for family in cards.families():
        for entry in family["versions"]:
            key = cards.key_of(entry)
            markdown = (cards.CARD_DIR / f"{key}.md").read_text(encoding="utf-8")
            before_picture = markdown.split("![", 1)[0]
            article = Article()
            article.feed(renderer.render(before_picture))
            assert article.text.startswith(scores.title_of(entry)), key
            plotted = scores.scores(entry)
            assert scores._version(key) in article.text
            for label, mine, theirs in plotted["rows"]:
                assert label in article.text, (key, label)
                for value in (mine, theirs):
                    if value is not None:
                        assert f"{value:.3f}" in article.text, (key, value)
            if plotted["holdout"]:
                assert plotted["holdout"] in article.text, key
            assert "table" in article.tags and "img" not in article.tags
    index = Article()
    index.feed(renderer.render((cards.CARD_DIR / cards.INDEX_NAME).read_text(
        encoding="utf-8")))
    assert index.text.startswith(scores.HEADING_TEXT)


def test_family_defaults_match_the_zoo_markers():
    cards = _cards()
    defaults = {cards.key_of(e) for e in cards.default_records()}
    for entry in cards.published_models():
        if cards.marker_of(entry):
            assert cards.key_of(entry) not in defaults or all(
                cards.marker_of(other) for other in cards.published_models()
                if cards.family_and_version(cards.key_of(other))[0]
                == cards.family_and_version(cards.key_of(entry))[0])


def test_the_app_the_readme_and_the_cards_open_a_family_on_the_same_version():
    """2026-10-12: the maintainer made toxoplasma_well_detector_v3 the
    well-detector default in both the README and the app, and asked that the
    app's default_version_index and the README/website rule agree."""
    from types import SimpleNamespace

    from spacr.model_zoo import default_version_index

    cards = _cards()
    for family in cards.families():
        pairs = [(cards.key_of(e), SimpleNamespace(key=cards.key_of(e)))
                 for e in family["versions"]]
        assert family["versions"][default_version_index(pairs)] is family["default"]
    defaults = {cards.key_of(e) for e in cards.default_records()}
    assert "toxoplasma_well_detector_v3" in defaults
    assert {"toxoplasma_pv_v4", "toxoplasma_plaque_v6"} <= defaults
    # Stock Cellpose-SAM stays the label-free default (2026-10-20).
    assert "live_cell_v1" not in defaults


def test_a_candidate_never_becomes_a_family_default():
    from types import SimpleNamespace

    from spacr import model_zoo

    pairs = [("v9", SimpleNamespace(key="not_a_published_key_v9")),
             ("v1", SimpleNamespace(key="not_a_published_key_v1"))]
    original = model_zoo._CANDIDATE_MODEL_KEYS
    model_zoo._CANDIDATE_MODEL_KEYS = frozenset({"not_a_published_key_v9"})
    try:
        assert model_zoo.default_version_index(pairs) == 1
    finally:
        model_zoo._CANDIDATE_MODEL_KEYS = original
