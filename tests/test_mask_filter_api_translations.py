"""Mask mean-bound API pages publish their source-bound semantic corrections."""
from __future__ import annotations

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
def test_mask_filter_api_publishes_reviewed_mean_and_bound_semantics(
        language, current_documents):
    """Passing syntax alone cannot detect mean/limit/optional-filter omissions."""
    review_path = (ROOT / "docs/i18n/reviewed/api" / language /
                   "2026-09-16-mask-mean-bounds.json")
    records = json.loads(review_path.read_text(encoding="utf-8"))["records"]
    assert records, "The semantic review must not silently become an empty scan"
    symbols = json.loads((api.API_DIR / f"{language}.json").read_text(
        encoding="utf-8"))["symbols"]
    for record in records:
        key, _separator, index_text = record["label"].rpartition("#")
        index = int(index_text)
        source = current_documents[key]
        sources, _layout = api.translatable_blocks(source)
        assert sources[index] == record["source"], record["label"]
        assert api._source_hash(sources[index]) == record["source_sha256"]
        assert symbols[key]["source_sha256"] == api._source_hash(source)
        published, _layout = api.translatable_blocks(symbols[key]["text"])
        assert len(published) == len(sources), (language, key)
        assert published[index] == record["translation"], (
            language, record["label"], published[index])
