"""317: published example-pack notices and API prose match reviewed meanings."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_documentation_i18n as api


@pytest.fixture(scope="module")
def current_documents():
    return api.public_docstrings()


@pytest.mark.parametrize("language", tuple(api.MODEL_SPECS))
def test_example_pack_api_publishes_reviewed_import_semantics(
        language, current_documents):
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


@pytest.mark.parametrize("language", tuple(api.MODEL_SPECS))
def test_example_pack_runtime_publishes_reviewed_counts_and_losses(language):
    from spacr.qt import i18n
    from spacr.qt.i18n_catalogs import en

    path = (ROOT / "docs/i18n/reviewed/runtime" / language /
            "2026-09-16-example-settings-pack.json")
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    assert len(records) == 7
    catalog = importlib.import_module(f"spacr.qt.i18n_catalogs.{language}")
    for record in records:
        assert record["table"] == "ui"
        source = record["source"]
        assert source in en.UI_SOURCES
        assert api._source_hash(source) == record["source_sha256"]
        assert catalog.UI[source] == record["translation"]
        assert i18n._exact_translation(source, language) == record["translation"]
