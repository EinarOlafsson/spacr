"""Translated Sphinx user guides: catalogs, staleness guard and selector."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_guide_i18n as guide  # noqa: E402

SCRIPT = ROOT / "docs" / "source" / "_static" / "api_i18n.js"
LANGUAGES = guide.catalog_languages()


def _po_files(language):
    return sorted((guide.LOCALE_DIR / language / "LC_MESSAGES").glob("*.po"))


# -- message validation -----------------------------------------------------

def test_problems_reject_changed_literals_roles_urls_and_ui_names():
    msgid = ("Press **Save mask** and read ``masks/``; see :func:`spacr.io.save` "
             "and the `tutorial <tutorials/#lesson=14_make_masks>`_.")
    good = ("Tryck på **Spara mask** och läs ``masks/``; se :func:`spacr.io.save` "
            "och `handledningen <tutorials/#lesson=14_make_masks>`_.")
    glossary = {"Save mask": "Spara mask"}
    assert guide.message_problems(msgid, good, glossary) == []
    assert guide.message_problems(msgid, good.replace("``masks/``", "``maskar/``"), glossary)
    assert guide.message_problems(msgid, good.replace("spacr.io.save", "spacr.io.load"), glossary)
    assert guide.message_problems(msgid, good.replace("lesson=14", "lesson=15"), glossary)
    assert guide.message_problems(msgid, good.replace("Spara mask", "Spara masken"), glossary)
    assert guide.message_problems(msgid, "", glossary) == ["empty translation"]


def test_ui_names_split_menu_paths():
    assert guide.ui_names("Open **Home → Tools → Make Masks** and **Undo**") == [
        "Home", "Tools", "Make Masks", "Undo"]


# -- staleness guard ---------------------------------------------------------

def _write_pot(directory: Path, domain: str, messages):
    from babel.messages import pofile
    from babel.messages.catalog import Catalog

    catalog = Catalog(project="fixture")
    for message in messages:
        catalog.add(message)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{domain}.pot").open("wb") as stream:
        pofile.write_po(stream, catalog)


def test_changed_english_becomes_stale_and_is_not_published(tmp_path):
    pytest.importorskip("babel")
    pot = tmp_path / "pot"
    locale = tmp_path / "locale"
    _write_pot(pot, "make_masks", ["Press **Save mask**.", "Unchanged."])
    guide.update_language("sv", pot, locale)
    work = tmp_path / "work.json"
    guide.export_worklist("sv", work, locale_dir=locale)
    rows = json.loads(work.read_text())
    strings = tmp_path / "strings.json"
    strings.write_text(json.dumps(["Tryck på **Spara mask**.", "Oförändrad."]))
    applied, rejected = guide.import_worklist("sv", work, strings, locale_dir=locale)
    assert (applied, rejected) == (2, [])
    assert len(rows) == 2

    # English changes (the pending Make Masks re-layout is exactly this).
    _write_pot(pot, "make_masks", ["Press **Save mask** or Ctrl+S.", "Unchanged."])
    report = guide.audit(pot, ["sv"], locale)["languages"]["sv"]
    assert report["pages"]["make_masks"]["translated"] == 1
    assert report["pages"]["make_masks"]["missing"] == 1
    assert report["pages"]["make_masks"]["stale"] == 1

    guide.update_language("sv", pot, locale)
    catalog = guide.read_catalog(locale / "sv/LC_MESSAGES/make_masks.po")
    changed = catalog.get("Press **Save mask** or Ctrl+S.")
    assert changed is not None and changed.fuzzy          # hint only
    assert catalog.get("Unchanged.").string == "Oförändrad."
    guide.export_worklist("sv", work, locale_dir=locale)
    pending = json.loads(work.read_text())
    assert [row["msgid"] for row in pending] == ["Press **Save mask** or Ctrl+S."]
    assert pending[0]["hint"] == "Tryck på **Spara mask**."


def test_import_refuses_a_translation_for_an_old_msgid(tmp_path):
    pot = tmp_path / "pot"
    locale = tmp_path / "locale"
    _write_pot(pot, "page", ["Old text."])
    guide.update_language("sv", pot, locale)
    work = tmp_path / "work.json"
    work.write_text(json.dumps([{"domain": "page", "msgid": "Gone text.",
                                 "msgstr": "Borta."}]))
    applied, rejected = guide.import_worklist("sv", work, locale_dir=locale)
    assert applied == 0 and rejected and "stale msgid" in rejected[0]


def test_sphinx_renders_a_changed_message_in_english_and_marks_it(tmp_path):
    """The guard the published pages rely on: gettext falls back per message."""
    pytest.importorskip("sphinx")
    source = tmp_path / "src"
    source.mkdir()
    (source / "conf.py").write_text(textwrap.dedent("""
        project = 'fixture'
        locale_dirs = ['locale/']
        gettext_compact = False
        translation_progress_classes = True
    """))
    (source / "index.rst").write_text(
        "Title edited\n============\n\nFirst paragraph.\n\n"
        "Second paragraph, now edited.\n")
    messages = source / "locale/sv/LC_MESSAGES"
    messages.mkdir(parents=True)
    (messages / "index.po").write_text(textwrap.dedent('''
        msgid ""
        msgstr ""
        "Content-Type: text/plain; charset=utf-8\\n"

        msgid "Title"
        msgstr "Rubrik"

        msgid "First paragraph."
        msgstr "Första stycket."

        msgid "Second paragraph."
        msgstr "Andra stycket."
    '''), encoding="utf-8")
    output = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", "-b", "html", "-D", "language=sv",
         str(source), str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-4000:]
    page = (output / "index.html").read_text(encoding="utf-8")
    assert "Första stycket." in page
    assert "Andra stycket." not in page and "Rubrik" not in page
    # The stylesheet flags exactly these text blocks (p and headings).
    assert re.search(r'<h1 class="untranslated">Title edited', page)
    assert re.search(r'class="[^"]*untranslated[^"]*">Second paragraph, now edited\.', page)


# -- committed catalogs ------------------------------------------------------

def test_catalog_languages_are_known_and_labelled():
    assert set(LANGUAGES) <= set(guide.LANGUAGES)
    script = SCRIPT.read_text(encoding="utf-8")
    for language in LANGUAGES:
        assert set(guide.BANNERS[language]) >= {"label", "review", "fallback", "original"}
        assert re.search(rf'\b{language}: "', script), language
        files = _po_files(language)
        assert files, language
        for path in files:
            assert guide.catalog_review_kind(path) == guide.REVIEW_KIND, path
        assert (guide.GLOSSARY_DIR / f"{language}.json").is_file()


@pytest.mark.parametrize("language", LANGUAGES)
def test_published_translations_keep_markup_and_app_ui_names(language):
    pytest.importorskip("babel")
    glossary = guide.load_glossary(language)
    failures = []
    for path in _po_files(language):
        for message in guide.read_catalog(path):
            if not message.id or not message.string or message.fuzzy:
                continue
            problems = guide.message_problems(message.id, message.string, glossary)
            if problems:
                failures.append(f"{path.name}: {message.id[:50]!r}: {problems}")
    assert not failures, "\n".join(failures[:20])


def test_glossary_matches_the_runtime_catalogs():
    pytest.importorskip("PySide6")
    from spacr.qt.i18n import _exact_translation
    from spacr.qt.i18n_catalogs import en, setting_label

    keys = {label: key for key, label in reversed(list(en.SETTING_LABELS.items()))
            if "." not in key}
    for language in LANGUAGES:
        for english, translated in guide.load_glossary(language).items():
            shown = _exact_translation(english, language)
            if not shown and english in keys:
                shown = setting_label(keys[english], english, language)
            assert shown == translated, (language, english)


def test_english_only_pages_have_no_catalogs():
    for language in LANGUAGES:
        names = {path.stem for path in _po_files(language)}
        assert not names & guide.ENGLISH_ONLY_PAGES


# -- Sphinx wiring -----------------------------------------------------------

def test_conf_wires_guides_only_mode():
    conf = (ROOT / "docs/source/conf.py").read_text(encoding="utf-8")
    assert "locale_dirs = ['../i18n/guides/']" in conf
    assert "translation_progress_classes = True" in conf
    assert "'data-guide-languages'" in conf
    assert "extensions.remove('autoapi.extension')" in conf
    assert "'i18n/**', 'deck/**'" in conf


def test_english_inventory_reader(tmp_path):
    body = ("spacr.io.save py:function 1 api/spacr/io/index.html#$ -\n"
            "api/index std:doc -1 api/index.html API reference\n")
    path = tmp_path / "objects.inv"
    path.write_bytes(b"# Sphinx inventory version 2\n# Project: x\n# Version: 1\n"
                     b"# The remainder of this file is compressed using zlib.\n"
                     + zlib.compress(body.encode()))
    inventory = guide.read_inventory(path)
    assert inventory[("py:function", "spacr.io.save")] == \
        "api/spacr/io/index.html#spacr.io.save"
    assert inventory[("std:doc", "api/index")] == "api/index.html"


# -- selector ----------------------------------------------------------------

def _guide_page(language: str, depth: str, harness: str) -> bytes:
    return f"""<!doctype html>
<html lang="{language}"><head><meta charset="utf-8"><title>guide</title>
<script src="{depth}_static/api_i18n.js" data-api-catalog-version="unit"
  data-guide-languages="sv de"></script>
<script>window.addEventListener("DOMContentLoaded", () => {{ {harness} }});</script>
</head><body><main><article role="main"><h1>Make Masks</h1><p>Body</p>
</article></main></body></html>""".encode()


def test_guide_selector_moves_between_language_trees():
    from tests.test_api_i18n_frontend import CHROME, _dump_dom, _server

    if not CHROME:
        pytest.skip("Chrome/Chromium is required for the browser test")
    report = """
setTimeout(() => {
  const select = document.querySelector('.spacr-guide-language select');
  const values = select ? [...select.options].map((o) => o.value).join(',') : '';
  document.body.dataset.result = [document.documentElement.lang, location.pathname,
    select && select.value, values].join('|');
}, 300);
"""
    script = SCRIPT.read_bytes()
    files = {
        "/site/_static/api_i18n.js": script,
        "/site/sv/_static/api_i18n.js": script,
        "/site/make_masks.html": _guide_page("en", "", report),
        "/site/sv/make_masks.html": _guide_page("sv", "", report),
    }
    with _server(files) as (base, _requests):
        english = _dump_dom(f"{base}/site/make_masks.html")
        redirected = _dump_dom(f"{base}/site/make_masks.html?lang=sv")
        swedish = _dump_dom(f"{base}/site/sv/make_masks.html")
    assert 'data-result="en|/site/make_masks.html|en|en,sv,de"' in english
    assert 'data-result="sv|/site/sv/make_masks.html|sv|en,sv,de"' in redirected
    assert 'data-result="sv|/site/sv/make_masks.html|sv|en,sv,de"' in swedish


def test_guide_selector_targets_are_computed_from_the_script_location():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "function guideTarget(current, target)" in script
    assert 'new URL("../", scriptUrl)' in script
    assert "setupGuideSelector(apiArticle)" in script
    assert 'safeStorageSet(select.value)' in script
