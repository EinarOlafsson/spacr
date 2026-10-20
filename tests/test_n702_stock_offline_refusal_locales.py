"""The actual offline stock refusal uses each installed reviewed translation."""
import json
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import utils as utilities
from tests.test_n702_stock_offline_boundary import native_cache as native_cache
from tests.test_n702_unknown_cellpose_model_refusal import isolated as isolated


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "fr", "pt", "hi", "ko", "is"])
def test_offline_stock_constructor_refusal_uses_selected_language(native_cache, monkeypatch, language):
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "off")
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs: pytest.fail("Uncached offline model reached constructor"))
    review = (Path(__file__).resolve().parents[1] / "docs/i18n/reviewed/runtime" / language
              / "2026-10-15-cellpose-stock-offline-refusal.json")
    record = json.loads(review.read_text(encoding="utf-8"))["records"][0]
    with pytest.raises(zoo.ModelUnreadable) as error:
        utilities._choose_model("cpsam", device="cpu", object_type="cell")
    assert record["translation"] != record["source"]
    assert str(error.value) == record["translation"].format(
        model=repr("cpsam"), path=str(native_cache / "cpsam"))
    assert not native_cache.exists()
