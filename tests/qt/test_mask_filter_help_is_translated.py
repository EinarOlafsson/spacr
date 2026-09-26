"""418: published mask-bound help reaches the real runtime in all nine locales.

Reviewed evidence alone is not publication. These checks use the public
source-bound lookup and the settings tooltip formatter, without building a
Mask screen or starting a preview worker. Higher slots reuse the primary
tooltip with its channel identifier rewritten, not its prose renumbered.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
from pathlib import Path
import re

import pytest

pytest.importorskip("PySide6")

from spacr.object_roles import setting_label as english_label
from spacr.organelle_types import organelle_role
from spacr.qt import i18n, i18n_catalogs
from spacr.qt.screens.settings_model import format_tooltip
from spacr.settings import tooltips

pytestmark = pytest.mark.qt

LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
ROLES = ("cell", "nucleus", "pathogen", "organelle", "organelleb",
         "organellec", "organelled")
ROOT = Path(__file__).resolve().parents[2]
REVIEW_NAME = "2026-09-15-mask-mean-bounds.json"
CHANNEL = re.compile(r"\b(?:cell|nucleus|pathogen|organelle[a-z]*)_channel\b")


@pytest.fixture(autouse=True)
def _restored_language(monkeypatch, language):
    """Select the real runtime language without writing user preferences."""
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    assert i18n.current_language() == language
    # monkeypatch restores the prior environment even when an assertion fails.
    yield


@pytest.fixture(scope="module")
def reviewed():
    """Only the small 418 evidence files, not a whole-catalog/model audit."""
    records = {}
    for language in LANGUAGES:
        path = ROOT / "docs/i18n/reviewed/runtime" / language / REVIEW_NAME
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["language"] == language
        for record in payload["records"]:
            identity = (language, record["table"], record["key"])
            assert identity not in records
            records[identity] = record
    return records


def _body_source(key):
    # The documented formatter removes the type prefix and collapses source
    # whitespace before looking up the exact scientific paragraph.
    body = re.sub(r"^\s*\([^)]*\)\s*[-–:]?\s*", "", tooltips[key])
    return " ".join(body.split())


def _assert_published(reviewed, language, table, key, source, translated):
    assert translated and translated != source, (language, table, key)
    record = reviewed.get((language, table, key))
    if record is not None:
        assert record["source"] == source, (language, table, key)
        assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        assert translated == record["translation"], (
            f"{language}/{table}/{key}: the runtime has not published the "
            "current source-bound 418 review"
        )


def _assert_rendered(key, label, body):
    # HTML escaping is presentation, not a translation change. Read the real
    # formatter using the active language, including its current-source body.
    rendered = format_tooltip(tooltips[key], "mask", key)
    assert f"<b>{escape(label)}</b>" in rendered
    assert escape(body) in rendered


@pytest.mark.parametrize("language", LANGUAGES)
def test_mask_bounds_publish_current_labels_and_own_channel_help(language, reviewed):
    # The maintainer, 2026-09-25 (item 511): cell, nucleus and pathogen's
    # mean bounds are object_filters rows now, and their reviewed records
    # were deleted with the settings; the organelle slots keep theirs.
    for role in ROLES:
        if role in ("cell", "nucleus", "pathogen"):
            for bound in ("min", "max"):
                for table in ("setting_labels", "setting_tooltips"):
                    assert (language, table,
                            f"{role}_{bound}_intensity") not in reviewed
            continue
        for bound in ("min", "max"):
            key = f"{role}_{bound}_intensity"
            label_source = english_label(key)
            body_source = _body_source(key)
            label = i18n_catalogs.setting_label(key, label_source, language, "mask")
            body = i18n_catalogs.setting_tooltip(key, body_source, language, "mask")
            _assert_published(reviewed, language, "setting_labels", key,
                              label_source, label)
            _assert_published(reviewed, language, "setting_tooltips", key,
                              body_source, body)
            assert set(CHANNEL.findall(body)) == {f"{role}_channel"}
            _assert_rendered(key, label, body)


@pytest.mark.parametrize("language", LANGUAGES)
def test_live_preview_bound_captions_publish_current_translation(language, reviewed):
    from spacr.qt.widgets.live_preview import COMPARTMENT_FIELDS

    captions = {suffix: caption for suffix, caption, *_rest in COMPARTMENT_FIELDS
                if suffix in ("min_intensity", "max_intensity")}
    assert captions == {"min_intensity": "Min intensity", "max_intensity": "Max intensity"}
    for source in captions.values():
        translated = i18n_catalogs.ui_text(source, language)
        _assert_published(reviewed, language, "ui", source, source, translated)
        assert i18n.tr(source) == translated


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("slot", (2, 702))
def test_numbered_slots_keep_current_help_and_their_own_channel(language, slot, reviewed):
    role = organelle_role(slot)
    for bound in ("min", "max"):
        key = f"{role}_{bound}_intensity"
        source = _body_source(key)
        label_source = english_label(key)
        body = i18n_catalogs.setting_tooltip(key, source, language, "mask")
        label = i18n_catalogs.setting_label(key, label_source, language, "mask")
        _assert_published(reviewed, language, "setting_tooltips", key, source, body)
        _assert_published(reviewed, language, "setting_labels", key, label_source, label)
        assert set(CHANNEL.findall(body)) == {f"{role}_channel"}
        assert re.search(rf"(?<!\d){slot}(?!\d)", label)

        if slot == 702:
            # Above the four materialized slots, the runtime adapter borrows
            # the source-bound primary record and substitutes the key prefix.
            primary = f"organelle_{bound}_intensity"
            primary_body = i18n_catalogs.setting_tooltip(
                primary, _body_source(primary), language, "mask")
            primary_label = i18n_catalogs.setting_label(
                primary, english_label(primary), language, "mask")
            _assert_published(reviewed, language, "setting_tooltips", primary,
                              _body_source(primary), primary_body)
            _assert_published(reviewed, language, "setting_labels", primary,
                              english_label(primary), primary_label)
            assert body == primary_body.replace("organelle_", f"{role}_")
            assert label == re.sub(r"(?<!\d)1(?!\d)", str(slot), primary_label, count=1)
            if language == "zh_CN":
                # The first review accidentally numbered primary prose. Its
                # slot-702 alias then described slot 1 despite the right key.
                assert "细胞器" in body
                assert "细胞器 1" not in body

        _assert_rendered(key, label, body)
        # Positive lookups above pair with stale-source rejection below:
        # similar setting names must not receive obsolete translated help.
        assert i18n_catalogs.setting_tooltip(key, source + " changed", language) is None
        assert i18n_catalogs.setting_label(key, label_source + " changed", language) is None
