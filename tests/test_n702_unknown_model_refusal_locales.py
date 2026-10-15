"""Actual unresolved-model refusals use the exact installed reviewed targets."""
import json
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import utils as utilities
from tests.test_n702_unknown_cellpose_model_refusal import isolated as isolated


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "fr", "pt", "hi", "ko", "is"])
def test_actual_unknown_model_refusal_is_localized(isolated, monkeypatch, language):
    requested = "unpublished-cluster-model.CP_model"
    path = (Path(__file__).parents[1] / "docs/i18n/reviewed/runtime" / language
            / "2026-10-15-cellpose-unknown-model-refusal.json")
    record = json.loads(path.read_text())["records"][0]
    expected = record["translation"].format(name=repr(requested))
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs:
                        pytest.fail("Unknown model reached constructor"))
    with pytest.raises(zoo.ModelZooError) as error:
        utilities._choose_model(requested, device="cpu", object_type="cell")
    assert str(error.value) == expected
    assert expected != record["source"].format(name=repr(requested))
