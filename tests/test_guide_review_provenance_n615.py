"""Guide reviews retain actual authorship across import, update and rendering."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import build_guide_i18n as guide  # noqa: E402


def _template(path, messages):
    from babel.messages.catalog import Catalog
    from babel.messages.pofile import write_po
    path.mkdir(parents=True, exist_ok=True)
    catalog = Catalog(project="review fixture")
    for source in messages:
        catalog.add(source)
    with (path / "page.pot").open("wb") as stream:
        write_po(stream, catalog)


def _import(tmp_path, locale, rows, reviewer="codex"):
    work = tmp_path / "work.json"
    work.write_text(json.dumps([
        {"domain": "page", "msgid": source, "msgstr": target}
        for source, target in rows
    ]))
    return guide.import_worklist("sv", work, locale_dir=locale, reviewer=reviewer)


def test_mixed_review_keeps_existing_translation_and_survives_update(tmp_path):
    pot, locale = tmp_path / "pot", tmp_path / "locale"
    _template(pot, ["Previous paragraph.", "New paragraph."])
    guide.update_language("sv", pot, locale)
    assert _import(tmp_path, locale, [("Previous paragraph.", "Föregående stycke.")],
                   "claude-opus-5.5") == (1, [])
    path = locale / "sv/LC_MESSAGES/page.po"
    original = guide.read_catalog(path).get("Previous paragraph.")
    old_comments = list(original.user_comments)
    assert _import(tmp_path, locale, [("New paragraph.", "Nytt stycke.")]) == (1, [])
    kind = guide.catalog_review_kind(path)
    assert "Claude Opus 5.5 / Codex" in kind
    catalog = guide.read_catalog(path)
    assert catalog.get("Previous paragraph.").string == "Föregående stycke."
    assert catalog.get("Previous paragraph.").user_comments == old_comments
    assert all("Codex" not in c for c in old_comments)
    assert any("Codex" in c for c in catalog.get("New paragraph.").user_comments)
    translator = catalog.last_translator
    _template(pot, ["Previous paragraph.", "New paragraph.", "Later paragraph."])
    guide.update_language("sv", pot, locale)
    assert guide.catalog_review_kind(path) == kind
    assert guide.read_catalog(path).last_translator == translator
    report = guide.audit(pot, ["sv"], locale)["languages"]["sv"]
    assert report["review_kinds"]["page"] == kind
    assert not report["label_missing"]
    assert report["pages"]["page"]["missing"] == 1


def test_new_codex_page_does_not_claim_claude_review(tmp_path, monkeypatch):
    pot, locale = tmp_path / "pot", tmp_path / "locale"
    _template(pot, ["New paragraph."])
    guide.update_language("sv", pot, locale)
    assert _import(tmp_path, locale, [("New paragraph.", "Nytt stycke.")]) == (1, [])
    path = locale / "sv/LC_MESSAGES/page.po"
    assert guide.catalog_review_kind(path) == (
        "AI technical review (Codex), no native-speaker signoff")
    assert "Claude" not in path.read_text()
    monkeypatch.setattr(guide, "LOCALE_DIR", locale)
    context = {"body": "<p>Nytt stycke.</p>"}
    guide._page_context(SimpleNamespace(config=SimpleNamespace(language="sv")),
                        "page", "page.html", context, object())
    assert "Codex" in context["body"]
    assert "Claude" not in context["body"]
    assert "ingen granskning av modersmålstalare" in context["body"]


def test_unknown_reviewer_and_false_header_do_not_pass(tmp_path):
    pot, locale = tmp_path / "pot", tmp_path / "locale"
    _template(pot, ["New paragraph."])
    guide.update_language("sv", pot, locale)
    path = locale / "sv/LC_MESSAGES/page.po"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Unsupported guide reviewer"):
        _import(tmp_path, locale, [("New paragraph.", "Nytt stycke.")], "native-speaker")
    assert path.read_bytes() == original
    path.write_text(path.read_text().replace(guide.REVIEW_KIND, "Human approved"))
    assert guide.audit(pot, ["sv"], locale)["languages"]["sv"]["label_missing"] == ["page"]
    with pytest.raises(ValueError, match="Unsupported guide review label"):
        _import(tmp_path, locale, [("New paragraph.", "Nytt stycke.")])


def test_rejected_literal_change_adds_no_review_claim(tmp_path):
    pot, locale = tmp_path / "pot", tmp_path / "locale"
    source = "Read ``data.csv``."
    _template(pot, [source])
    guide.update_language("sv", pot, locale)
    path = locale / "sv/LC_MESSAGES/page.po"
    kind = guide.catalog_review_kind(path)
    applied, rejected = _import(tmp_path, locale, [(source, "Läs ``other.csv``.")])
    assert applied == 0 and rejected
    assert guide.catalog_review_kind(path) == kind
    assert not guide.read_catalog(path).get(source).string


@pytest.mark.parametrize("caption, expected", [
    ("Markerade bilder ({count})", "Markerade bilder"),
    ("已勾选的图像（{count}）", "已勾选的图像"),
    ("({count}) selected images", None),
    ("Selected {count} images ({count})", None),
])
def test_counted_guide_name_uses_only_exact_button_prefix(monkeypatch, caption, expected):
    from spacr.qt import i18n

    def exact(source, language):
        assert source == "Checked images ({count})"
        return caption

    monkeypatch.setattr(i18n, "_exact_translation", exact)
    assert guide.runtime_ui_name("Checked images", "sv") == expected
