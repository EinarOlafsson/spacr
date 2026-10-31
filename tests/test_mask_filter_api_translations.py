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
    calibrated = json.loads((ROOT / "docs/i18n/reviewed/api" / language /
                             "2026-10-24-calibrated-batch-segmentation.json").read_text(
                                 encoding="utf-8"))["records"]
    for record in records:
        assert api._source_hash(record["source"]) == record["source_sha256"]
        if record["label"] in ("spacr.pipeline_v2.stream_masks_from_stack#3",
                               "spacr.pipeline_v2.stream_masks_from_stack#5"):
            [current] = [row for row in calibrated if row["label"] == record["label"]]
            if record["label"].endswith("#3"):
                expected = record["source"].replace(
                    " The first selected channel", " Calibrated sessions instead require explicit "
                    "in-range positions. The first selected channel", 1)
            else:
                expected = record["source"] + (
                    " Calibrated batches refuse above ``64 * 1024**2`` packed input "
                    "or padded model-input values.")
            assert current["source"] == expected
            record = current
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
