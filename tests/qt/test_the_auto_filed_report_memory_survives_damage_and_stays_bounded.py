"""The memory of automatically filed reports survives damage and stays bounded.

Pinned behaviour of :mod:`spacr.qt.ai.settings`'s auto-filed store:

* a stored value that is not JSON, or JSON that is not an object, reads
  as "nothing filed yet" rather than raising;
* remembering an empty fingerprint stores nothing;
* beyond :data:`AUTO_FILED_LIMIT` the oldest fingerprints are forgotten
  and the newest are kept.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402

from spacr.qt.ai import settings as ai_settings  # noqa: E402

pytestmark = pytest.mark.qt


def _store():
    return QSettings("spacr", "qt")


@pytest.mark.parametrize("raw", ["{not json", "[1, 2, 3]", "\"a string\""])
def test_a_damaged_store_reads_as_nothing_filed(qapp, raw):
    _store().setValue("ai/auto_filed_reports", raw)

    assert ai_settings.auto_filed_url("abc123") == ""


def test_a_damaged_store_is_replaced_by_the_next_report(qapp):
    _store().setValue("ai/auto_filed_reports", "{not json")

    ai_settings.remember_auto_filed("abc123", "https://example.org/issues/7")

    assert ai_settings.auto_filed_url("abc123") == (
        "https://example.org/issues/7")


def test_an_empty_fingerprint_is_not_remembered(qapp):
    ai_settings.remember_auto_filed("", "https://example.org/issues/1")

    assert _store().value("ai/auto_filed_reports", "") == ""
    assert ai_settings.auto_filed_url("") == ""


def test_the_oldest_reports_are_forgotten_beyond_the_limit(qapp, monkeypatch):
    monkeypatch.setattr(ai_settings, "AUTO_FILED_LIMIT", 3)

    for n in range(5):
        ai_settings.remember_auto_filed(f"fp{n}", f"https://example.org/{n}")

    stored = json.loads(_store().value("ai/auto_filed_reports"))
    assert list(stored) == ["fp2", "fp3", "fp4"]
    assert ai_settings.auto_filed_url("fp0") == ""
    assert ai_settings.auto_filed_url("fp1") == ""
    assert ai_settings.auto_filed_url("fp4") == "https://example.org/4"
