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
    render = pytest.importorskip("readme_renderer.rst").render

    packaged, _ = metadata
    source = (ROOT / "README.rst").read_text(encoding="utf-8")
    warnings = io.StringIO()
    html = render(packaged["long_description"], stream=warnings)
    assert html is not None, warnings.getvalue()
    original, converted = Images(), Images()
    original.feed(render(source))
    converted.feed(html)
    assert len(converted.images) == len(original.images) > 30
    assert [i.get("alt") for i in converted.images] == [i.get("alt") for i in original.images]
    assert all(i["src"].startswith("https://") for i in converted.images)
    assert all(re.match(r"(?:https?://|mailto:|#)", link) for link in converted.links)
    restored = packaged["long_description"].replace(
        "https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/", "").replace(
        "https://github.com/EinarOlafsson/spacr/blob/nightly/", "")
    original_normalized = source.replace(
        "https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/", "").replace(
        "https://github.com/EinarOlafsson/spacr/blob/nightly/", "")
    assert restored == original_normalized


def test_conversion_handles_figures_links_and_preserves_external_urls(metadata):
    _, convert = metadata
    source = """.. figure:: ./docs/picture.png
   :target: docs/guide.rst#example

`Guide <docs/guide.rst>`_ `Email <mailto:someone@example.org>`_
`Anchor <#example>`_ `Remote <https://example.org/page?q=a&b=c>`_
.. |badge| image:: https://example.org/image.svg?q=a&b=c
"""
    converted = convert(source)
    assert "nightly/docs/picture.png" in converted
    assert "blob/nightly/docs/guide.rst#example" in converted
    assert "blob/nightly/docs/guide.rst>`_" in converted
    for value in ("mailto:someone@example.org", "<#example>",
                  "https://example.org/page?q=a&b=c", "https://example.org/image.svg?q=a&b=c"):
        assert value in converted
    assert convert(converted) == converted
