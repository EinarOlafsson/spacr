"""PyPI shows the README's rows as GitHub does, after readme_renderer's sanitiser.

The 1.5.1.5 page on PyPI ran the README's images together. GitHub sets each
``<picture>`` row in its own ``<div>``, but the PyPI description turned every
single image into a bare inline link. The language picker sat beside the
spaCR panel, the hardware heading beside its table and the cover beside the
first buttons, all in phone drawings on a desktop page. Each row is now a
line block (a ``<div>``) in a desktop and a phone drawing, chosen by PyPI's
own ``hide-below-desktop`` and ``hide-on-desktop`` classes. Each image's
width is its share of the column it was drawn for.
"""
import io
import os
import re
import runpy
import tarfile
import zipfile
from email import message_from_string
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pytest

pytest.importorskip("readme_renderer")

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = os.environ.get("SPACR_PYPI_README_REF", "main")
CLASSES = {"hide-below-desktop": "desktop", "hide-on-desktop": "phone"}
VOID = ("img", "br", "hr", "col", "input")


@pytest.fixture(scope="module")
def packaged():
    import setuptools

    captured = {}
    original, cwd = setuptools.setup, Path.cwd()
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    try:
        os.chdir(ROOT)
        namespace = runpy.run_path(str(ROOT / "setup.py"))
    finally:
        setuptools.setup = original
        os.chdir(cwd)
    artifact_dir = os.environ.get("SPACR_PYPI_DIST_DIR")
    if artifact_dir is not None:
        directory = Path(artifact_dir)
        wheels = list(directory.glob("*.whl"))
        sources = list(directory.glob("*.tar.gz"))
        assert len(wheels) == len(sources) == 1, directory
        with zipfile.ZipFile(wheels[0]) as wheel:
            names = [name for name in wheel.namelist()
                     if name.endswith(".dist-info/METADATA")]
            assert len(names) == 1
            wheel_metadata = message_from_string(wheel.read(names[0]).decode("utf-8"))
        with tarfile.open(sources[0], "r:gz") as source:
            names = [name for name in source.getnames()
                     if name.count("/") == 1 and name.endswith("/PKG-INFO")]
            assert len(names) == 1
            source_metadata = message_from_string(
                source.extractfile(names[0]).read().decode("utf-8"))
        for metadata in (wheel_metadata, source_metadata):
            assert metadata["Description-Content-Type"] == "text/x-rst"
            assert metadata.get_payload() == captured["long_description"]
        captured["long_description"] = wheel_metadata.get_payload()
    return captured, namespace


@pytest.fixture(scope="module")
def rendered(packaged):
    from readme_renderer.rst import render

    captured, _ = packaged
    warnings = io.StringIO()
    html = render(captured["long_description"], stream=warnings)
    assert html is not None, warnings.getvalue()
    return html


class Layout(HTMLParser):
    """Each top-level block of the description and the images inside it."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.blocks = []
        self.loose = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if self.depth == 0:
            if tag in ("img", "a"):
                self.loose.append((tag, values))
            self.blocks.append({"tag": tag, "class": values.get("class"),
                                "images": []})
        if tag == "img":
            self.blocks[-1]["images"].append(values)
        if tag not in VOID:
            self.depth += 1

    def handle_endtag(self, tag):
        if tag not in VOID:
            self.depth -= 1


def _layout(rendered):
    layout = Layout()
    layout.feed(rendered)
    return layout


def _readme_rows():
    source = (ROOT / "README.rst").read_text(encoding="utf-8")
    rows = [re.findall(
        r'<picture><source [^>]*srcset="([^"]+)"[^>]*><img src="([^"]+)" '
        r'alt="([^"]*)"></picture>', row)
        for row in re.findall(r"(?m)^   <div>(.*)</div>$", source)]
    return [row for row in rows if row]


def test_no_image_or_link_floats_loose_between_blocks(rendered):
    assert _layout(rendered).loose == []


def test_every_readme_row_is_a_desktop_and_a_phone_line_block(rendered):
    drawn = [block for block in _layout(rendered).blocks
             if block["images"] and all(image.get("class") in CLASSES
                                        for image in block["images"])]
    rows = _readme_rows()
    assert len(drawn) == 2 * len(rows) > 60
    for index, row in enumerate(rows):
        desktop, phone = drawn[2 * index], drawn[2 * index + 1]
        assert desktop["tag"] == phone["tag"] == "div"
        assert set(desktop["class"].split()) == {"line-block", "hide-below-desktop"}
        assert set(phone["class"].split()) == {"line-block", "hide-on-desktop"}
        for block, name, column in ((desktop, "desktop", 0), (phone, "phone", 1)):
            assert {CLASSES[image["class"]] for image in block["images"]} == {name}
            assert [image["src"].split(f"/{REFERENCE}/")[-1]
                    for image in block["images"]] == [item[column] for item in row]
            assert [image["alt"] for image in block["images"]] == [
                unescape(item[2]) for item in row]


def test_each_image_is_a_share_of_its_column_and_rows_never_overflow(rendered):
    one_line_rows = 0
    for block in _layout(rendered).blocks:
        images = [image for image in block["images"]
                  if image.get("class") in CLASSES]
        if not images:
            continue
        widths = [image.get("width", "") for image in images]
        assert all(re.fullmatch(r"\d+(?:\.\d+)?%", width) for width in widths), widths
        shares = [float(width[:-1]) for width in widths]
        assert all(0 < share <= 100 for share in shares)
        if len(images) > 1 and sum(shares) <= 100:
            one_line_rows += 1
    assert one_line_rows >= 3


def test_a_full_width_panel_fills_the_column_in_both_layouts(rendered):
    panel = "docs/readme_images/en/spacr"
    widths = {image["class"]: image["width"]
              for block in _layout(rendered).blocks for image in block["images"]
              if image.get("class") in CLASSES
              and image["src"].split(f"/{REFERENCE}/")[-1] in (f"{panel}.svg",
                                                       f"{panel}.desktop.svg")}
    assert widths == {"hide-below-desktop": "100%", "hide-on-desktop": "100%"}


def test_a_code_block_has_space_after_it_before_the_next_panel(rendered):
    following = re.findall(
        r"</pre>\s*(?:</div>\s*)?<div class=\"line-block\">\s*"
        r"<div class=\"line\">(.*?)</div>", rendered, re.S)
    assert len(following) >= 5
    assert all(re.fullmatch(r"\s*(?:<br\s*/?>)?\s*", content)
               for content in following), following


def test_a_wheel_built_from_the_sdist_reuses_its_description(packaged, tmp_path):
    captured, namespace = packaged
    description = captured["long_description"]
    (tmp_path / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\nName: spacr\nVersion: 0\n"
        "Description-Content-Type: text/x-rst\n\n" + description,
        encoding="utf-8")
    readme = (ROOT / "README.rst").read_text(encoding="utf-8")
    assert namespace["long_description_for"](readme, tmp_path) == description
    assert namespace["long_description_for"](readme, ROOT) == description


def test_responsive_rows_hide_contact_links_with_their_images(rendered):
    layout = _layout(rendered)
    rows = [block for block in layout.blocks
            if any(image.get("alt") == "olafsson@med.umich.com"
                   for image in block["images"])]
    assert len(rows) == 2
    for row in rows:
        image_classes = {image["class"] for image in row["images"]}
        assert len(image_classes) == 1
        assert image_classes.pop() in row["class"].split()
    assert 'href="mailto:olafsson%40med.umich.com"' in rendered
