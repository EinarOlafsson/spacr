"""Numbered background switches use current prose and their own channel."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pytest

from spacr.object_roles import setting_label as english_label
from spacr.organelle_types import organelle_role
from spacr.qt import i18n_catalogs as catalogs
from spacr.settings import tooltips

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = catalogs.CATALOG_LANGUAGES


def _source(key):
    return " ".join(re.sub(r"^\s*\([^)]*\)\s*[-–:]?\s*", "", tooltips[key]).split())


@pytest.mark.parametrize("language", LANGUAGES)
def test_every_numbered_switch_has_its_own_translated_label_and_floor(language):
    # Include materialized slots, the first alias, a letter rollover and the
    # registry's upper bound. No GUI, preview worker or model is started.
    for number in (1, 2, 3, 4, 5, 26, 27, 702):
        role = organelle_role(number)
        key = f"remove_background_{role}"
        source = _source(key)
        label = catalogs.setting_label(key, english_label(key), language)
        body = catalogs.setting_tooltip(key, source, language)
        assert label and label != english_label(key), (language, key)
        assert body and body != source, (language, key)
        assert re.search(rf"(?<!\d){number}(?!\d)", label)
        assert re.search(rf"(?<!\d){number}(?!\d)", body)
        assert set(re.findall(r"(?<![A-Za-z0-9_])organelle[a-z]*_background(?![A-Za-z0-9_])", body)) == {
            f"{role}_background"}
        assert "False" in body
        assert catalogs.setting_label(key, english_label(key) + " changed", language) is None
        assert catalogs.setting_tooltip(key, source + " changed", language) is None


@pytest.mark.parametrize("language", LANGUAGES)
def test_reviewed_switch_records_are_published_and_source_bound(language):
    path = ROOT / "docs/i18n/reviewed/runtime" / language / "2026-09-21-organelle-background.json"
    records = json.loads(path.read_text())["records"]
    assert len(records) == 8
    for record in records:
        key = record["key"]
        is_label = record["table"] == "setting_labels"
        source = english_label(key) if is_label else _source(key)
        assert record["source"] == source
        assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        lookup = catalogs.setting_label if is_label else catalogs.setting_tooltip
        assert lookup(key, source, language) == record["translation"]


def test_dynamic_switches_refuse_stale_translation_and_missing_template(monkeypatch):
    from spacr.qt.i18n_catalogs import de, en

    key = "remove_background_organellezz"
    source = _source(key)
    assert catalogs.setting_tooltip(key, source, "de")
    with monkeypatch.context() as patch:
        patch.setitem(de.SOURCE_HASHES,
                      ("SETTING_TOOLTIPS", "remove_background_organelleb"), "stale")
        assert catalogs.setting_tooltip(key, source, "de") is None
    with monkeypatch.context() as patch:
        patch.delitem(en.SETTING_TOOLTIPS, "remove_background_organelleb")
        assert catalogs.setting_tooltip(key, source, "de") is None
    assert catalogs.setting_tooltip(key, source, "en") is None
    assert catalogs.setting_tooltip("remove_background_not_a_role", source, "de") is None


def test_catalogs_keep_four_switches_and_translate_higher_slots_at_runtime():
    from spacr.qt.i18n_catalogs import en

    expected = {f"remove_background_{organelle_role(i)}" for i in range(1, 5)}
    for table in (en.SETTING_LABELS, en.SETTING_TOOLTIPS):
        assert {key for key in table if key.startswith("remove_background_organelle")} == expected
