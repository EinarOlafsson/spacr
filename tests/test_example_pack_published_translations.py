"""317: published example-pack notices and API prose match reviewed meanings."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

#: Item 710: the API catalogs are assembled for the session, not committed.
pytestmark = pytest.mark.usefixtures("api_catalogs")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_documentation_i18n as api

from tests import locale_cases


@pytest.fixture(scope="module")
def current_documents():
    return api.public_docstrings()


def _example_pack_api_publishes_reviewed_import_semantics_case(language, current_documents):
    """One case of :func:`test_example_pack_api_publishes_reviewed_import_semantics`."""
    path = (ROOT / "docs/i18n/reviewed/api" / language /
            "2026-09-16-example-settings-pack.json")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    assert len(records) == 6
    symbols = json.loads((api.API_DIR / f"{language}.json").read_text(
        encoding="utf-8"))["symbols"]
    for record in records:
        symbol, _, index = record["label"].rpartition("#")
        source = current_documents[symbol]
        blocks, _ = api.translatable_blocks(source)
        assert blocks[int(index)] == record["source"]
        assert api._source_hash(blocks[int(index)]) == record["source_sha256"]
        assert symbols[symbol]["source_sha256"] == api._source_hash(source)
        published, _ = api.translatable_blocks(symbols[symbol]["text"])
        assert len(published) == len(blocks)
        assert published[int(index)] == record["translation"]


def test_example_pack_api_publishes_reviewed_import_semantics(subtests, current_documents):
    for label, case in locale_cases.cases(("language", tuple(api.MODEL_SPECS))):
        with subtests.test(**label):
            _example_pack_api_publishes_reviewed_import_semantics_case(current_documents=current_documents, **case)


def _example_pack_runtime_publishes_reviewed_counts_and_losses_case(language):
    """One case of :func:`test_example_pack_runtime_publishes_reviewed_counts_and_losses`."""
    from spacr.qt import i18n
    from spacr.qt.i18n_catalogs import en

    path = (ROOT / "docs/i18n/reviewed/runtime" / language /
            "2026-09-16-example-settings-pack.json")
    review = json.loads(path.read_text(encoding="utf-8"))
    records, retired = review["records"], review["retired_records"]
    assert len(records) == 5
    assert {record["source"] for record in retired} == {
        "Demo dataset loaded with its settings. Press Live Preview to see one field, or Run to process the plate.",
        "Demo dataset loaded without a settings pack; using defaults. Press Live Preview to see one field, or Run to process the plate.",
    }
    for record in retired:
        assert record["source"] not in en.UI_SOURCES
        assert api._source_hash(record["source"]) == record["source_sha256"]
        assert record["translation"]
    catalog = importlib.import_module(f"spacr.qt.i18n_catalogs.{language}")
    for record in records:
        assert record["table"] == "ui"
        source = record["source"]
        assert source in en.UI_SOURCES
        assert api._source_hash(source) == record["source_sha256"]
        assert catalog.UI[source] == record["translation"]
        assert i18n._exact_translation(source, language) == record["translation"]


def test_example_pack_runtime_publishes_reviewed_counts_and_losses(subtests):
    for label, case in locale_cases.cases(("language", tuple(api.MODEL_SPECS))):
        with subtests.test(**label):
            _example_pack_runtime_publishes_reviewed_counts_and_losses_case(**case)
