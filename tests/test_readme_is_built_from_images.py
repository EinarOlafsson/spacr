"""The README is images and code blocks, built from its structured source.

The maintainer's decision of 2026-10-12: everything in the GitHub README that
is not a code block is an image set in Open Sans (Regular for headings, Light
for body text), and code blocks stay RST code blocks so they can be copied.
``tools/build_readme_images.py`` builds the English README from
``packaging/readme_source.yaml`` and each translation from
``docs/i18n/readme/source/<lang>.yaml``. These tests hold the committed
READMEs and images to that source.

None of them needs the Noto fonts except the last, which redraws the English
images when the fonts are cached and skips otherwise: each SVG records the
SHA-256 of the block it was drawn from, so staleness is decided without
drawing anything.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool():
    from tests.readme_source_words import readme_tool

    return readme_tool()


LANGUAGES = ("en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
_IMAGE = re.compile(
    r"(?m)^\.\. (?:\|(?P<name>[^|]+)\| )?image:: (?P<src>\S+)\n"
    r"(?P<options>(?:   :[a-z]+:[^\n]*\n?)*)")


def _readme(language: str) -> Path:
    return _tool().readme_path(language)


#: An image in the README's raw-HTML form: the phone drawing as ``<img>``,
#: the desktop drawing as the ``<source>`` before it.
_HTML_IMAGE = re.compile(
    r'(?:<source [^>]*srcset="(?P<desktop>[^"]+)">)?<img src="(?P<src>[^"]+)"'
    r'(?: width="[^"]*")?(?: align="[^"]*")? alt="(?P<alt>[^"]*)">')


def _images(text: str) -> list[dict]:
    """Every image the README shows, in either layout, with its alt text."""
    from html import unescape

    found = []
    for match in _IMAGE.finditer(text):
        options = dict(re.findall(r"   :([a-z]+): ?([^\n]*)",
                                  match.group("options")))
        found.append({"src": match.group("src"), **options})
    for match in _HTML_IMAGE.finditer(text):
        alt = unescape(match.group("alt"))
        found.append({"src": unescape(match.group("src")), "alt": alt})
        if match.group("desktop"):
            found.append({"src": unescape(match.group("desktop")),
                          "alt": alt})
    return found


def test_every_language_has_a_source_and_the_tool_knows_them_all():
    tool = _tool()
    assert tuple(tool.LANGUAGES) == LANGUAGES
    for language in LANGUAGES:
        assert tool.source_path(language).is_file(), language
        assert set(tool.FONT_COVERAGE) == set(LANGUAGES)


@pytest.mark.parametrize("language", LANGUAGES[1:])
def test_each_translation_has_the_english_block_structure(language):
    assert _tool().check_structure(language) == []


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_readme_holds_only_images_code_badges_and_markers(language):
    """No prose, list, heading or table reaches the README as text."""
    tool = _tool()
    text = _readme(language).read_text(encoding="utf-8")
    for block in tool.parse_rst(text):
        kind = block["kind"]
        if kind == "marker":
            assert block["name"] in (*tool.GENERATED_BLOCKS, "citing-papers")
            inner = tool.parse_rst(block["content"])
            assert {item["kind"] for item in inner} <= {"raw"}, (
                f"{language}: the {block['name']} block carries text: "
                f"{[item for item in inner if item['kind'] != 'raw'][:1]}")
        else:
            assert kind in ("raw", "code"), (
                f"{_readme(language).name} carries a {kind} as text: "
                f"{block.get('text', block)!r:.120}")


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_readme_image_has_alt_text_and_the_svg_says_the_same(language):
    tool = _tool()
    path = _readme(language)
    images = [image for image in _images(path.read_text(encoding="utf-8"))
              if "readme_images/" in image["src"]]
    assert len(images) > 40
    for image in images:
        alt = image.get("alt", "").strip()
        assert alt, f"{image['src']} has no alt text"
        svg = (path.parent / image["src"]).resolve()
        assert svg.is_file(), image["src"]
        desc = re.search(r"<desc id=\"d\">(.*?)</desc>",
                         svg.read_text(encoding="utf-8"), re.S).group(1)
        from html import unescape

        assert tool._alt(unescape(desc)) == alt, image["src"]


@pytest.mark.parametrize("language", LANGUAGES)
def test_panel_alt_text_is_the_full_text_of_its_source_blocks(language):
    """Every prose block's text is in some image's alt, word for word."""
    tool = _tool()
    alts = " ".join(image.get("alt", "") for image in _images(
        _readme(language).read_text(encoding="utf-8")))
    for block in tool.load_source(language):
        texts = []
        if block["kind"] in ("paragraph", "heading"):
            texts = [block["text"]]
        elif block["kind"] == "list":
            texts = block["items"]
        for text in texts:
            plain = tool.plain(text)
            if tool.only_links(text):
                for label, _target in tool.links_of(text):
                    assert tool._alt(label) in alts
            else:
                assert tool._alt(plain) in alts, (language, plain[:80])


def test_the_readmes_and_images_are_current():
    """README text equals what the sources build; every SVG is up to date."""
    problems = _tool().check()
    assert not problems, (
        "the README or its images are stale; run "
        "`python tools/build_readme_images.py`:\n" + "\n".join(problems[:20]))


def test_no_stray_images_are_left_behind():
    tool = _tool()
    for language in LANGUAGES:
        text = _readme(language).read_text(encoding="utf-8")
        used = set(re.findall(r"readme_images/[^/\s]+/([^/\s]+)\.svg", text))
        on_disk = {path.stem for path in
                   (tool.IMAGE_DIR / language).glob("*.svg")}
        assert on_disk == used, (language, sorted(on_disk ^ used)[:10])


@pytest.mark.parametrize("language", LANGUAGES)
def test_each_image_embeds_the_fonts_its_script_needs(language):
    """Open Sans everywhere; Noto for Devanagari, Hangul and Han."""
    tool = _tool()
    families = set()
    for path in (tool.IMAGE_DIR / language).glob("*.svg"):
        text = path.read_text(encoding="utf-8")
        families |= set(re.findall(r"font-family:'(spacr-[a-z]+)'", text))
        assert "url(data:font/woff2;base64," in text, path.name
        assert "@import" not in text and "href=\"http" not in text
    assert "spacr-os" in families
    assert ("spacr-script" in families) == (language in tool.SCRIPT_FONT)


def test_light_body_and_regular_headings():
    """The two weights the decision names, and nothing heavier."""
    tool = _tool()
    weights = set()
    for path in (tool.IMAGE_DIR / "en").glob("*.svg"):
        weights |= set(re.findall(r'font-weight="(\d+)"',
                                  path.read_text(encoding="utf-8")))
    assert weights == {"300", "400"}


def test_one_text_size_at_true_display_size_on_phone_and_desktop():
    """2026-10-13, read on a phone: the image text was smaller than the code
    blocks. Every image is drawn per layout at its display size (one SVG
    unit per CSS pixel: 358 px phone column, 830 px desktop column) and
    every word is 13.6 px, GitHub's code-block text, with section titles at
    16 px."""
    tool = _tool()
    assert (tool.PHONE, tool.DESKTOP) == (358, 830)
    assert (tool.BODY, tool.TITLE) == (13.6, 16)
    sizes = set()
    for path in (tool.IMAGE_DIR / "en").glob("*.svg"):
        text = path.read_text(encoding="utf-8")
        width = float(re.search(r'<svg [^>]*width="([\d.]+)"', text).group(1))
        layout = tool.DESKTOP if ".desktop" in path.name else tool.PHONE
        assert width <= layout, path.name
        assert re.search(rf'viewBox="0 0 {width:g} ', text), path.name
        sizes |= set(re.findall(r'font-size="([\d.]+)"', text))
    assert sizes == {"13.6", "16"}, sizes


def test_icon_rows_are_equal_cells_spanning_the_row():
    """Module, data and installer icons: one row each, equal cells, the
    same icon size in every row."""
    tool = _tool()
    text = _readme("en").read_text(encoding="utf-8")
    rows = [re.findall(r"\|([A-Za-z_]+)\|", line)
            for line in text.splitlines()
            if re.match(r"\| \|(Module_|Data|Installer)", line)]
    assert rows
    for names in rows:
        widths = set()
        for name in names:
            found = re.search(rf"\.\. \|{name}\| raw:: html\n\n   "
                              r'<a href="[^"]+"><img src="([^"]+)" '
                              r'width="([\d.]+)%"', text)
            assert found, name
            assert (ROOT / found.group(1)).is_file(), found.group(1)
            widths.add(float(found.group(2)))
        assert len(widths) == 1, names
        assert 98 <= widths.pop() * len(names) < 100, names


def test_the_palette_is_the_score_cards_and_never_warm():
    tool = _tool()
    cards = (ROOT / "docs" / "models" / "cards" / "heading.svg").read_text(
        encoding="utf-8").lower()
    assert tool.PALETTE["surface"] in cards
    for value in tool.PALETTE.values():
        red, green, blue = (int(value[i:i + 2], 16) for i in (1, 3, 5))
        assert not (red > blue + 16 and red > green), (
            f"{value} is a warm colour")


def test_a_single_link_makes_the_whole_image_a_link_and_several_become_buttons():
    tool = _tool()
    blocks = [
        {"kind": "heading", "level": 2, "text": "Guides"},
        {"kind": "paragraph", "text": "See the `guide <docs/a.rst>`_."},
        {"kind": "paragraph",
         "text": "Read `one <docs/b.rst>`_ and `two <docs/c.rst>`_."},
        {"kind": "list", "items": ["`x <docs/x.rst>`_ -- first",
                                   "`y <docs/y.rst>`_ -- second"]},
        {"kind": "code", "lang": "bash", "text": "spacr"},
    ]
    units = tool.plan(blocks)
    assert [unit["kind"] for unit in units] == ["panel", "panel", "rows",
                                                "code"]
    assert units[0]["targets"] == ["docs/a.rst"]
    images: dict = {}
    markup = tool._panel_markup("en", "guides-2", units[1]["blocks"], images,
                                render=False)
    assert "<a href" not in markup.split("\n\n")[0]
    assert markup.count('href="docs/b.rst"') == 1
    assert markup.count('href="docs/c.rst"') == 1


def test_inline_markup_reaches_the_alt_text_as_plain_words():
    tool = _tool()
    text = ("Use **Load test data…** with ``spacr-download`` -- see the "
            "`guide <docs/a.rst>`_ and `Citing spaCR`_, *et al.*")
    assert tool.plain(text) == ("Use Load test data… with spacr-download -- "
                                "see the guide and Citing spaCR, et al.")


def test_chinese_wraps_between_characters_and_keeps_punctuation_attached():
    tool = _tool()
    style = tool.Style(17, 300, "#ffffff")
    units = tool._break_units([("模型。然后", style)])
    texts = ["".join(piece.text for piece in unit) for unit in units]
    assert "型。" in texts and not any(text.startswith("。")
                                         for text in texts)


@pytest.mark.skipif(not _tool().fonts_available(),
                    reason="the pinned Noto fonts are not cached here")
def test_the_english_images_redraw_byte_for_byte():
    pytest.importorskip("uharfbuzz", reason="drawing the README images needs uharfbuzz")
    tool = _tool()
    output = tool.compose("en")
    stale = [name for name, svg in output.images.items()
             if (tool.IMAGE_DIR / "en" / f"{name}.svg").read_text(
                 encoding="utf-8") != svg]
    assert not stale, stale[:10]
