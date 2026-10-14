"""Verify the actual setup metadata renders on PyPI without losing content."""
import io
import re
import runpy
from html.parser import HTMLParser
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def metadata(monkeypatch):
    captured = {}
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("setuptools.setup", lambda **kwargs: captured.update(kwargs))
    namespace = runpy.run_path(str(ROOT / "setup.py"))
    return captured, namespace["pypi_readme"]


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "img":
            self.images.append(values)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


def test_packaged_readme_renders_all_original_images(metadata):
    renderer = pytest.importorskip("readme_renderer.rst")
    render = renderer.render
    from docutils.core import publish_parts

    packaged, _ = metadata
    source = (ROOT / "README.rst").read_text(encoding="utf-8")
    warnings = io.StringIO()
    html = render(packaged["long_description"], stream=warnings)
    assert html is not None, warnings.getvalue()
    original, converted = Images(), Images()
    writer = renderer.Writer()
    writer.translator_class = renderer.ReadMeHTMLTranslator
    original.feed(publish_parts(source, writer=writer, settings_overrides={
        **renderer.SETTINGS, "raw_enabled": True,
    })["fragment"])
    converted.feed(html)
    assert len(converted.images) == len(original.images) > 30
    assert [i.get("alt") for i in converted.images] == [i.get("alt") for i in original.images]
    assert all(i["src"].startswith("https://") for i in converted.images)
    assert all(re.match(r"(?:https?://|mailto:|#)", link) for link in converted.links)
    restored = packaged["long_description"].replace(
        "https://raw.githubusercontent.com/EinarOlafsson/spacr/main/", "").replace(
        "https://github.com/EinarOlafsson/spacr/blob/main/", "")
    original_normalized = source.replace(
        "https://raw.githubusercontent.com/EinarOlafsson/spacr/main/", "").replace(
        "https://github.com/EinarOlafsson/spacr/blob/main/", "")
    # Only GitHub's raw HTML changes shape for PyPI: icon cells and the
    # phone/desktop pictures become RST images (the phone drawing). Every
    # code block and every line outside the raw HTML stays intact.
    def kept(text):
        text = re.sub(r"(?m)^\.\. (?:\|[^|\n]+\| )?raw:: html\n\n(?:   [^\n]*\n?)+",
                      "", text)
        text = re.sub(r"(?m)^\.\. (?:\|[^|\n]+\| )?image::[^\n]*\n(?:   :[^\n]*\n?)*",
                      "", text)
        return [line for line in text.splitlines()
                if line.strip() and not line.startswith("| ")]
    assert kept(restored) == kept(original_normalized)
    assert "raw:: html" not in packaged["long_description"]
    assert {i["alt"]: i["src"].split("/main/")[-1] for i in converted.images} == {
        i["alt"]: i["src"].split("/main/")[-1] for i in original.images}


def test_pypi_keeps_each_readable_panel_destination(metadata):
    from docutils.core import publish_parts
    from tests.readme_source_words import readme_tool

    packaged, _ = metadata
    html = publish_parts(packaged["long_description"], writer_name="html5")["fragment"]
    article = Images()
    article.feed(html)
    destinations = {
        "https://github.com/EinarOlafsson/spacr/blob/main/"
        f"docs/readme_text/en/{name}.md"
        for name in readme_tool().compose("en", render=False).pages
    }
    assert len(destinations) == 65
    assert destinations <= set(article.links)


def test_conversion_handles_figures_links_and_preserves_external_urls(metadata):
    _, convert = metadata
    source = """.. figure:: ./docs/picture.png
   :target: docs/guide.rst#example

`Guide <docs/guide.rst>`_ `Email <mailto:someone@example.org>`_
`Anchor <#example>`_ `Remote <https://example.org/page?q=a&b=c>`_
.. |badge| image:: https://example.org/image.svg?q=a&b=c
"""
    converted = convert(source)
    assert "main/docs/picture.png" in converted
    assert "blob/main/docs/guide.rst#example" in converted
    assert "blob/main/docs/guide.rst>`_" in converted
    for value in ("mailto:someone@example.org", "<#example>",
                  "https://example.org/page?q=a&b=c", "https://example.org/image.svg?q=a&b=c"):
        assert value in converted
    assert convert(converted) == converted


def test_conversion_preserves_escaped_tile_labels_and_destinations(metadata):
    _, convert = metadata
    source = '''.. |Module_sample| raw:: html

   <a href="https://example.org/api?a=1&amp;b=2"><img src="icons/sample.png" width="16.5%" align="middle" alt="Open the A &amp; &quot;B&quot; API"></a>
'''
    converted = convert(source)
    assert "raw:: html" not in converted
    assert ':alt: Open the A & "B" API' in converted
    assert ':target: https://example.org/api?a=1&b=2' in converted
    assert 'image:: https://raw.githubusercontent.com/EinarOlafsson/spacr/main/icons/sample.png' in converted
    assert ':width: 16.5%' in converted
    assert convert(converted) == converted
