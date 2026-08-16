"""The README's three installer downloads are drawn platform icons.

What these tests protect is not "an image directive exists". It is the set of
ways an icon row silently stops working:

* white line art on a transparent background disappears on GitHub's white
  light-mode page, and nobody notices because the maintainer reads in dark
  mode;
* an icon hotlinked from a CDN dies the day that host moves;
* one glyph drawn solid beside two drawn in outline turns a row of equal
  choices into a recommendation;
* the release helper that bumps the download links every release stops
  recognising the block and either fails the release or leaves stale links.

Each of those is asserted on the rendered or measured effect.
"""

from __future__ import annotations

import importlib.util
import io
import re
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.rst"
ICON_DIR = ROOT / "spacr" / "resources" / "icons" / "platforms"

BEGIN = ".. spacr-installer-links-begin"
END = ".. spacr-installer-links-end"

#: The published assets keep the spelling they were published under, which is
#: why ``tests/test_the_name_is_spaCR.py`` exempts ``releases/download/``
#: lines: renaming them in text makes the front page 404.
ASSET_URL = "https://github.com/EinarOlafsson/spacr/releases/download/v{version}/SpaCR-{version}-{fragment}"  # noqa: E501

#: ``artwork stem -> the release-asset fragment its link must point at``
PLATFORM_ASSETS = {
    "windows": "Windows-Online-Setup.exe",
    "macos": "macOS-Universal-Online.pkg",
    "linux": "Linux-x86_64-Online.run",
}

#: github/markup renders ``.rst`` with docutils configured like this. Raw HTML
#: is off, which is why the README cannot use the ``<picture>`` +
#: ``prefers-color-scheme`` trick that a Markdown README would use for "white".
GITHUB_RST_SETTINGS = {
    "cloak_email_addresses": True,
    "file_insertion_enabled": False,
    "raw_enabled": False,
    "strip_comments": True,
    "doctitle_xform": True,
    "report_level": 2,
    "syntax_highlight": "short",
    "input_encoding": "utf-8",
    "halt_level": 5,
}

#: GitHub's two README page colours
LIGHT_PAGE = (255, 255, 255)
DARK_PAGE = (13, 17, 23)


def _release_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_release_helper", ROOT / "packaging" / "release.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _block() -> str:
    text = README.read_text(encoding="utf-8")
    start = text.find(BEGIN)
    end = text.find(END)
    assert start >= 0 and end > start, "README lost its installer-link markers"
    return text[start:end + len(END)]


def _render_readme(text: str) -> tuple[str, str]:
    """Return ``(html, docutils messages)`` for GitHub's own RST settings."""
    from docutils.core import publish_parts

    warnings = io.StringIO()
    settings = dict(GITHUB_RST_SETTINGS, warning_stream=warnings)
    parts = publish_parts(
        source=text, writer_name="html", settings_overrides=settings)
    # ``html_body`` only -- ``whole`` carries docutils' default stylesheet,
    # which mentions every class name it can style.
    return parts["html_body"], warnings.getvalue()


def _linked_images(html: str) -> dict[str, str]:
    """Map every ``<img>`` source to the href of the link wrapping it."""
    found = {}
    for anchor in re.finditer(
            r'<a\b[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<inner>.*?)</a>',
            html, re.DOTALL):
        for image in re.finditer(r'<img\b[^>]*src="(?P<src>[^"]+)"',
                                 anchor.group("inner")):
            found[image.group("src")] = anchor.group("href")
    return found


def _relative_luminance(rgb) -> float:
    channels = []
    for value in rgb:
        srgb = value / 255.0
        channels.append(srgb / 12.92 if srgb <= 0.04045
                        else ((srgb + 0.055) / 1.055) ** 2.4)
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(first, second) -> float:
    light, dark = sorted((_relative_luminance(first),
                          _relative_luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def _icon(stem: str):
    from PIL import Image

    return Image.open(ICON_DIR / f"{stem}.png").convert("RGBA")


def _opaque_colours(image) -> Counter:
    """Count the RGB of every fully opaque pixel."""
    return Counter(pixel[:3] for pixel in image.getdata() if pixel[3] == 255)


# ------------------------------------------------------------------ the block

@pytest.mark.parametrize("stem", sorted(PLATFORM_ASSETS))
def test_each_download_is_an_icon_linking_to_its_installer(stem):
    """GitHub renders the row as three linked images, one per platform."""
    html, messages = _render_readme(README.read_text(encoding="utf-8"))
    assert "system-message" not in html, (
        f"README.rst does not render cleanly:\n{messages}")

    links = _linked_images(html)
    sources = [src for src in links if src.endswith(f"/platforms/{stem}.png")]
    assert len(sources) == 1, (
        f"expected exactly one {stem} platform icon, rendered {sources}")
    href = links[sources[0]]
    assert PLATFORM_ASSETS[stem] in href, (
        f"the {stem} icon links to {href}, not to its installer")
    assert href.startswith(
        "https://github.com/EinarOlafsson/spacr/releases/download/"), href


def test_the_icons_are_committed_here_and_not_hotlinked():
    """Every image in the block is served from this repository."""
    block = _block()
    urls = re.findall(r"image:: (\S+)", block)
    assert len(urls) == len(PLATFORM_ASSETS)
    prefix = (
        "https://raw.githubusercontent.com/EinarOlafsson/spacr/main"
        "/spacr/resources/icons/platforms/"
    )
    for url in urls:
        assert url.startswith(prefix), f"{url} is hotlinked from elsewhere"
        committed = ICON_DIR / url[len(prefix):]
        assert committed.is_file(), f"{url} has no committed artwork"


def test_no_platform_is_left_as_a_bare_text_link():
    """The three downloads are icons, not the list of text links they were."""
    block = _block()
    assert "download spaCR" not in block
    for fragment in PLATFORM_ASSETS.values():
        assert block.count(fragment) == 1, (
            f"{fragment} should appear exactly once, as an image target")


# ------------------------------------------------------------------ the art

@pytest.mark.parametrize("stem", sorted(PLATFORM_ASSETS))
def test_the_icon_carries_its_own_background_into_a_white_page(stem):
    """White art on transparency is invisible on GitHub's light-mode page.

    reStructuredText on GitHub cannot switch artwork by theme, so each glyph
    must sit on an opaque chip that contrasts with a white page as well as
    with a dark one.
    """
    image = _icon(stem)
    colours = _opaque_colours(image)
    assert sum(colours.values()) / (image.width * image.height) > 0.90, (
        f"{stem}.png is mostly transparent, so its white art has no backing")

    chip = colours.most_common(1)[0][0]
    assert _contrast(chip, LIGHT_PAGE) >= 3.0, (
        f"{stem}.png's chip {chip} is invisible on a white README page")
    assert _contrast(chip, DARK_PAGE) >= 1.05, (
        f"{stem}.png's chip {chip} vanishes into a dark README page")


@pytest.mark.parametrize("stem", sorted(PLATFORM_ASSETS))
def test_the_glyph_is_white_and_nothing_else_is_painted(stem):
    """One ink, one chip.

    Every opaque pixel has to be the chip, pure white, or an antialiased blend
    of exactly those two. A second colour anywhere -- a grey substituted for
    "white", a coloured platform mark -- breaks that line.
    """
    image = _icon(stem)
    colours = _opaque_colours(image)
    chip = colours.most_common(1)[0][0]
    assert (255, 255, 255) in colours, f"{stem}.png has no pure white artwork"
    assert _contrast((255, 255, 255), chip) >= 4.5

    for colour in colours:
        for channel in range(3):
            blend = (colour[0] - chip[0]) / (255.0 - chip[0])
            expected = chip[channel] + blend * (255 - chip[channel])
            assert abs(colour[channel] - expected) <= 3, (
                f"{stem}.png paints {colour}, which is not white on {chip}")


def test_the_three_icons_read_as_one_row_of_equal_choices():
    """Same canvas, and no platform drawn heavier than the others."""
    generator = _generator_module()
    sizes = set()
    coverage = {}
    for stem in PLATFORM_ASSETS:
        image = _icon(stem)
        sizes.add(image.size)
        white = sum(1 for pixel in image.getdata()
                    if pixel[3] == 255 and min(pixel[:3]) > 200)
        coverage[stem] = white / float(image.width * image.height)

    assert len(sizes) == 1, f"the icons are different sizes: {sizes}"
    for stem, seen in coverage.items():
        assert generator.COVERAGE_LO <= seen <= generator.COVERAGE_HI, (
            f"{stem}.png covers {seen:.3f} of its tile, outside the shared "
            f"band {generator.COVERAGE_LO:.3f}..{generator.COVERAGE_HI:.3f}")
    spread = max(coverage.values()) - min(coverage.values())
    assert spread < 0.02, f"one platform shouts louder than the rest: {coverage}"


def _generator_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_platform_icons",
        ROOT / "packaging" / "generate_platform_icons.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------- the release

def test_the_release_helper_still_bumps_the_icon_block(tmp_path):
    """The next release must be able to move all three links forward."""
    helper = _release_module()
    working = tmp_path / "README.rst"
    working.write_text(README.read_text(encoding="utf-8"), encoding="utf-8")

    updated = helper._updated_readme_text(working, "9.9.9")
    start = updated.find(BEGIN)
    block = updated[start:updated.find(END) + len(END)]

    for fragment in PLATFORM_ASSETS.values():
        expected = ASSET_URL.format(version="9.9.9", fragment=fragment)
        assert expected in block, f"{expected} is not in the bumped block"
    assert "1.5.0" not in block, "a stale version survived the bump"
    assert block.count("/platforms/") == 3, "the artwork links were rewritten"
    assert updated[:start] == README.read_text(encoding="utf-8")[:start]


def test_a_block_advertising_two_versions_is_refused(tmp_path):
    """Half-bumped links are a release that ships the wrong installer."""
    helper = _release_module()
    working = tmp_path / "README.rst"
    text = README.read_text(encoding="utf-8")
    stale = ASSET_URL.format(version="1.4.0", fragment=PLATFORM_ASSETS["linux"])
    current = re.search(
        r"https://github\.com/EinarOlafsson/spacr/releases/download/\S+"
        + re.escape(PLATFORM_ASSETS["linux"]), text)
    assert current, "the README lost its Linux installer link"
    working.write_text(text.replace(current.group(0), stale), encoding="utf-8")

    with pytest.raises(ValueError, match="more than one installer version"):
        helper._updated_readme_text(working, "9.9.9")
