"""Every text picture opens its own faithful, structured, copyable words."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from tests.readme_source_words import readme_tool


class Article(HTMLParser):
    def __init__(self):
        super().__init__()
        self.words = []
        self.links = []
        self.pictures = {}
        self.target = None
        self.tags = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        self.tags.append(tag)
        if tag == "a":
            self.target = values.get("href")
            self.links.append(self.target)
        if tag == "img":
            self.pictures[values["src"]] = self.target

    def handle_endtag(self, tag):
        if tag == "a":
            self.target = None

    def handle_data(self, data):
        self.words.append(data)

    @property
    def text(self):
        return " ".join("".join(self.words).split())


def rendered(markdown):
    from markdown_it import MarkdownIt

    result = Article()
    result.feed(MarkdownIt("commonmark").enable("table").render(markdown))
    return result


@pytest.mark.parametrize("language", readme_tool().LANGUAGES)
def test_every_panel_opens_rendered_words_before_any_other_content(language):
    tool = readme_tool()
    output = tool.compose(language, render=False)
    assert len(output.pages) == 68
    pictures = Article()
    pictures.feed(tool.readme_path(language).read_text(encoding="utf-8"))
    for name, expected in output.pages.items():
        source = tool._image_reference(language, name)
        destination = pictures.pictures[source]
        assert destination, (language, name)
        path = (tool.readme_path(language).parent / destination).resolve()
        assert path == tool.TEXT_DIR / language / f"{name}.md"
        actual = path.read_text(encoding="utf-8")
        article = rendered(actual)
        assert article.text and article.text == rendered(expected).text
        assert not any(tag in article.tags for tag in ("img", "script", "iframe"))
        for link in article.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc:
                continue
            target = (path.parent / unquote(parsed.path)).resolve()
            assert target.is_file(), (path, link)
    assert "table" in rendered(output.pages["generated-hardware-1"]).tags


@pytest.mark.parametrize("language", readme_tool().LANGUAGES)
def test_authoring_words_keep_their_order_in_each_rendered_article(language):
    tool = readme_tool()
    english = tool.load_source("en")
    local = tool.load_source(language)
    units = tool.plan(english)
    names = tool.image_ids(units, english)
    for unit, name in zip(units, names):
        if unit["kind"] not in ("panel", "rows"):
            continue
        blocks = [local[index] for index in unit["span"]]
        expected = []
        if unit["kind"] == "rows":
            expected = [(f"{name}-{index + 1}", [item])
                        for index, item in enumerate(blocks[0]["items"])]
        else:
            expected = [(name, [text for block in blocks for text in
                                (block["items"] if block["kind"] == "list"
                                 else [block["text"]])])]
        for route, texts in expected:
            page = tool.TEXT_DIR / language / f"{route}.md"
            article = rendered(page.read_text(encoding="utf-8"))
            assert article.text == " ".join(tool.plain(text) for text in texts)


def test_markdown_preserves_inline_syntax_tables_and_relative_fragments():
    tool = readme_tool()
    blocks = [
        {"kind": "heading", "level": 2, "text": "A [literal] heading"},
        {"kind": "paragraph", "text": "Use **strong**, *emphasis* and ``a`b``. "
         "Read `details <docs/source/installer_guide.rst#container-images>`_ "
         "or `remote <https://example.org/?x=1&y=2>`_."},
        {"kind": "table", "header_rows": 1, "rows": [["A | B", "Value"],
         ["a_b", "1 < 2"]], "legend": "A **complete** legend."},
    ]
    article = rendered(tool.readable_page(blocks, "fr"))
    assert article.text.startswith("A [literal] heading Use strong, emphasis and a`b.")
    assert article.text.endswith("A | B Value a_b 1 < 2 A complete legend.")
    assert {"h2", "strong", "em", "code", "table", "th", "td"} <= set(article.tags)
    assert article.links == ["../../source/installer_guide.rst#container-images",
                             "https://example.org/?x=1&y=2"]
    assert "../../source/installer_guide.rst#container-images" in tool.readable_page(
        [{"kind": "paragraph", "text": "`details "
          "<../../../docs/source/installer_guide.rst#container-images>`_"}],
        "fr", generated=True)


def test_missing_stale_and_obsolete_pages_fail_the_currency_check(tmp_path, monkeypatch):
    tool = readme_tool()
    monkeypatch.setattr(tool, "TEXT_DIR", tmp_path)
    output = tool.compose("en", render=False)
    tool._write_pages("en", output.pages)
    directory = tmp_path / "en"
    (directory / "spacr.md").unlink()
    (directory / "model-zoo.md").write_text("Wrong text\n", encoding="utf-8")
    (directory / "retired.md").write_text("Old text\n", encoding="utf-8")
    monkeypatch.setattr(tool, "COVER_SOURCE", tool.ROOT / tool.COVER_SOURCE)
    monkeypatch.setattr(tool, "ROOT", Path("/"))
    problems = tool.check(["en"])
    assert any("spacr.md is missing" in problem for problem in problems)
    assert any("model-zoo.md is stale" in problem for problem in problems)
    assert any("retired.md is obsolete" in problem for problem in problems)
