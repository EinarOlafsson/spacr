"""The three structured fit summaries must be translated, not merely hashed."""

import hashlib
import json
import re
from importlib import import_module
from pathlib import Path
from string import Formatter

import pytest

LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")
SOURCES = (
    "{name}: pooled EC50 {ec50}{unit} ({low}–{high}) across {used} of {plates} plates, I² {spread}",
    "{name}: selectivity index {index} ({low}–{high}), host EC50 {host} over response EC50 {response}",
    "{name}: {model} excess over {cells} combination wells, max {max} at {dose_a} + {dose_b}, min {min}; {synergistic} synergistic, {antagonistic} antagonistic",
)


def _fields(text):
    return sorted(field for _, field, _, _ in Formatter().parse(text) if field is not None)


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("source", SOURCES)
def test_fit_summary_is_translated_and_retains_its_data_fields(language, source):
    english = import_module("spacr.qt.i18n_catalogs.en")
    catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
    assert source in english.UI_SOURCES
    translated = catalog.UI[source]
    assert translated != source, f"{language}: structured summary remains English"
    assert catalog.SOURCE_HASHES["UI", source] == hashlib.sha256(source.encode()).hexdigest()
    assert _fields(translated) == _fields(source)
    assert translated.count("EC50") == source.count("EC50")
    # Spacing before the superscript does not change the heterogeneity statistic.
    assert len(re.findall(r"\bI\s*²", translated)) == source.count("I²")


@pytest.mark.parametrize("language", LANGUAGES)
def test_fit_summaries_keep_the_source_bound_technical_review(language):
    path = (Path(__file__).resolve().parents[2] / "docs/i18n/reviewed/runtime"
            / language / "2026-09-15-dose-response-summary-repair.json")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    assert {record["source"] for record in records} == set(SOURCES)
    catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
    for record in records:
        assert record["table"] == "ui"
        assert record["key"] == record["source"]
        assert record["source_sha256"] == hashlib.sha256(record["source"].encode()).hexdigest()
        assert catalog.UI[record["source"]] == record["translation"]
