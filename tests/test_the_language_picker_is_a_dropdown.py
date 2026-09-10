"""The language picker is one control, and the menu lives where it renders.

WHAT THIS FILE PINS, and why it is a rule rather than a preference.

The maintainer asked for the ten language links to be a dropdown menu instead
of a row of links (instruction 361). On GitHub a dropdown means one thing --
``<details>`` with a ``<summary>`` -- and that element cannot exist in
``README.rst``. GitHub renders reStructuredText through docutils with
github/markup's settings, and those settings disable raw HTML, so:

* ``.. raw:: html`` is refused. Locally it renders as a "raw directive
  disabled" system message; on github.com the whole block is printed as an
  escaped ``<pre>``, which was checked on a real rendered page (dask/dask's
  ``docs/source/index.rst``, fetched through the contents API with
  ``Accept: application/vnd.github.html`` -- the renderer the site uses).
* ``<details>`` typed straight into RST is escaped to visible
  ``&lt;details&gt;`` text, because docutils does not treat it as HTML.

So the menu moved to ``docs/i18n/readme/README.md``, which is Markdown and
does render it: posting that exact page to ``api.github.com/markdown`` with
``mode=gfm`` returns ``<details open="">`` with the language table inside.
Every README carries a single link to it, localized and naming the language
the reader is currently in.

The tests below therefore assert the two halves of that arrangement: that no
``.rst`` README tries the mechanism GitHub strips, and that the ten READMEs
and the one Markdown page are exactly what the generator writes. The second
half matters because the picker is generated from two directions -- the
visuals generator writes the block, and the documentation i18n build rewrites
the translated READMEs around it -- and a disagreement between them would put
the English word "Languages:" back on the Swedish page.
"""

from __future__ import annotations

import functools
import importlib.util
import io
import re
import sys
from pathlib import Path

import pytest

from tests.test_readme_installer_icons import GITHUB_RST_SETTINGS

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
README = ROOT / "README.rst"
README_DIR = ROOT / "docs" / "i18n" / "readme"
PICKER_PAGE = README_DIR / "README.md"
LOCALIZED = sorted(README_DIR.glob("README.*.rst"))
ALL_READMES = [README, *LOCALIZED]

BEGIN = ".. spacr-language-picker-begin"
END = ".. spacr-language-picker-end"


@functools.lru_cache(maxsize=1)
def _generator():
    spec = importlib.util.spec_from_file_location(
        "spacr_readme_visuals", ROOT / "packaging" / "generate_readme_visuals.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(text: str) -> "tuple[str, str]":
    """Return ``(html, messages)`` for github/markup's own docutils settings."""
    from docutils.core import publish_parts

    warnings = io.StringIO()
    parts = publish_parts(
        source=text, writer_name="html",
        settings_overrides=dict(GITHUB_RST_SETTINGS, warning_stream=warnings))
    return parts["html_body"], warnings.getvalue()


def _code(path: Path) -> str:
    match = re.fullmatch(r"README\.(?P<language>[^.]+)\.rst", path.name)
    return match.group("language") if match else "en"


def _block(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.find(BEGIN)
    end = text.find(END)
    assert start >= 0 and end > start, f"{path} lost its language-picker markers"
    return text[start:end + len(END)]


# --------------------------------------------- what GitHub actually permits

def test_a_details_menu_in_reStructuredText_is_thrown_away_by_github():
    """The measurement that decides where the menu can live.

    Both spellings are tried, because both are what someone reaching for a
    dropdown in a ``.rst`` file would write. Neither produces a menu.
    """
    raw_directive = (
        "spaCR\n=====\n\n.. raw:: html\n\n"
        "   <details>\n   <summary>Language</summary>\n   </details>\n\n"
        "Body.\n"
    )
    html, messages = _render(raw_directive)
    assert "<details" not in html
    assert "raw" in messages and "disabled" in messages
    assert "system-message" in html, (
        "a raw:: html block is refused, and the refusal is visible on the page")

    inline = (
        "spaCR\n=====\n\n<details><summary>Language</summary>\n\n"
        "`Svenska <docs/i18n/readme/README.sv.rst>`_\n\n</details>\n"
    )
    html, _messages = _render(inline)
    assert "<details" not in html
    assert "&lt;details&gt;" in html, (
        "inline HTML in RST is escaped to visible text, not rendered")


@pytest.mark.parametrize("path", ALL_READMES, ids=lambda path: path.name)
def test_no_rst_readme_reaches_for_the_mechanism_github_strips(path: Path):
    """A future "make it a real dropdown" edit must fail here, not on GitHub."""
    text = path.read_text(encoding="utf-8")
    for stripped in ("<details", "<summary", ".. raw:: html", "<script"):
        assert stripped not in text, (
            f"{path.name} uses {stripped!r}; GitHub renders .rst through "
            "docutils with raw HTML disabled and prints it as text")


# ------------------------------------------------------------- the control

def test_the_front_page_shows_one_language_link_not_ten():
    """Rendered by docutils exactly as github/markup configures it."""
    html, messages = _render(README.read_text(encoding="utf-8"))
    assert "system-message" not in html, messages

    links = re.findall(r'<a class="reference external" href="([^"]+)">', html)
    picker = [href for href in links if "i18n/readme/README" in href]
    assert picker == ["docs/i18n/readme/README.md"], (
        f"the front page should carry one language control, it has {picker}")
    assert not re.search(r"README\.[a-z_]+\.rst", html), (
        "the nine side-by-side language links are what this item removed")


@pytest.mark.parametrize("path", ALL_READMES, ids=lambda path: path.name)
def test_every_readme_carries_the_generated_picker_for_its_own_language(
        path: Path):
    """One control per README, and the committed one is the generated one.

    The link names the language the reader is already in -- the ordinary
    language-switcher label -- and the word before it is localized, so a
    reader who cannot read "Svenska" still meets "Språk".
    """
    generator = _generator()
    code = _code(path)
    line = generator._language_picker_line(code)
    assert _block(path) == f"{BEGIN}\n\n{line}\n\n{END}", (
        f"{path.name} is not what packaging/generate_readme_visuals.py writes; "
        "re-run it rather than editing the README")
    assert line.startswith(f"{generator.LANGUAGE_PICKER_LABELS[code]}:")
    target = "docs/i18n/readme/README.md" if code == "en" else "README.md"
    assert f"<{target}>`_" in line
    assert (path.parent / target).is_file()


def test_the_picker_label_matches_the_documentation_builder():
    """Two generators write this label; they may not disagree.

    ``packaging/generate_readme_visuals.py`` writes the block, and
    ``tools/build_documentation_i18n.py`` relabels it when a translated README
    is rebuilt from the English source. If the tables drift, one of the two
    runs silently reverts the other's work.
    """
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from build_documentation_i18n import LANGUAGE_PICKER_LABELS as builder

    generator = _generator()
    ours = dict(generator.LANGUAGE_PICKER_LABELS)
    assert ours.pop("en") == "Languages"
    assert ours == builder


def test_the_picker_line_is_never_sent_through_a_translation_model():
    """The guard that keeps the control's RST intact across an i18n rebuild.

    ``translatable_blocks`` holds any line starting ``Languages:`` out of the
    model. That is why the English line keeps that prefix even though the
    rendered word is localized afterwards: a picker that goes through a
    translation model comes back with its delimiters rearranged.
    """
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from build_documentation_i18n import rebuild_document, translatable_blocks

    source = f"{_block(README)}\n\nTranslate this sentence.\n"
    blocks, layout = translatable_blocks(source)
    assert blocks == ["Translate this sentence."]
    assert rebuild_document(layout, blocks).startswith(_block(README))


# ---------------------------------------------------------------- the menu

def test_the_menu_is_a_details_element_on_a_markdown_page():
    """``<details>`` renders in Markdown, which is the whole reason for it.

    Checked against GitHub itself on 2026-09-02 by posting this page to
    ``api.github.com/markdown`` with ``mode=gfm``: it returns
    ``<details open="">`` wrapping the table. Only the shape is asserted here,
    because a test may not depend on the network.
    """
    text = PICKER_PAGE.read_text(encoding="utf-8")
    # One menu. The page's closing paragraph also NAMES ``<details>`` while
    # explaining why the menu is here rather than on the front page, so the
    # opening tag is counted in the spelling only the element uses.
    assert text.count("<details open>") == 1, (
        "the page has nothing on it but the menu; a reader who followed a "
        "link to get here should not need a second click to see the list")
    assert text.count("</details>") == 1
    assert text.count("<summary>") == text.count("</summary>") == 1
    assert "🌐" in text


def test_the_menu_offers_every_shipped_language_and_each_link_resolves():
    from spacr.qt.i18n import LANGUAGES

    text = PICKER_PAGE.read_text(encoding="utf-8")
    menu = text.split("<details open>")[1].split("</details>")[0]
    targets = dict(re.findall(r"\[([^\]]+)\]\(([^)]+)\)", menu))
    assert list(targets) == [language.native_name for language in LANGUAGES]
    for name, target in targets.items():
        assert (PICKER_PAGE.parent / target).resolve().is_file(), (name, target)


def test_the_menu_says_which_translations_nobody_has_read():
    """Instruction 316: the claim has to be honest per locale.

    Offering ten languages with nothing said about them is itself a claim
    that all ten are equally good. Three were sampled by a fluent speaker;
    six are machine drafts nobody who speaks them has checked.
    """
    generator = _generator()
    from spacr.qt.i18n import LANGUAGES

    text = PICKER_PAGE.read_text(encoding="utf-8")
    menu = text.split("<details open>")[1].split("</details>")[0]
    rows = {
        re.search(r"\[([^\]]+)\]", line).group(1): line
        for line in menu.splitlines()
        if line.startswith("| [")
    }
    assert set(generator.SPOT_CHECKED_LOCALES) == {"sv", "de", "is"}
    for language in LANGUAGES:
        row = rows[language.native_name]
        if language.code == "en":
            assert "Source text" in row
        elif language.code in generator.SPOT_CHECKED_LOCALES:
            assert "A fluent speaker read a sample" in row
        else:
            assert "No fluent-speaker review" in row


def test_the_page_and_the_readmes_are_regenerated_not_hand_written(tmp_path):
    """Running the generator again must change nothing."""
    generator = _generator()
    assert PICKER_PAGE.read_text(encoding="utf-8") == \
        generator._language_picker_page()
    working = tmp_path / "README.sv.rst"
    source = (README_DIR / "README.sv.rst").read_text(encoding="utf-8")
    stale = source.replace(
        "Språk: `🌐 Svenska ▾ <README.md>`_", "Languages: stale")
    assert stale != source, "the Swedish picker line is not what this expects"
    working.write_text(stale, encoding="utf-8")
    assert generator._write_the_language_picker(working)
    assert working.read_text(encoding="utf-8") == source
