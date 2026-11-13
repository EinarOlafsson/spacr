"""Actual unresolved-model refusals use the exact installed reviewed targets."""
import json
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import utils as utilities
from tests.test_n702_unknown_cellpose_model_refusal import isolated as isolated

from tests import locale_cases


def _actual_unknown_model_refusal_is_localized_case(isolated, monkeypatch, language):
    """One case of :func:`test_actual_unknown_model_refusal_is_localized`."""
    requested = "unpublished-cluster-model.CP_model"
    path = (Path(__file__).parents[1] / "docs/i18n/reviewed/runtime" / language
            / "2026-10-15-cellpose-unknown-model-refusal.json")
    record = json.loads(path.read_text(encoding="utf-8"))["records"][0]
    expected = record["translation"].format(name=repr(requested))
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs:
                        pytest.fail("Unknown model reached constructor"))
    with pytest.raises(zoo.ModelZooError) as error:
        utilities._choose_model(requested, device="cpu", object_type="cell")
    first, _, context = str(error.value).partition("\n")
    assert first == expected
    assert "cell_model_name" in context and repr(requested) in context
    assert not context.startswith("Model setting ")
    assert expected != record["source"].format(name=repr(requested))


def test_actual_unknown_model_refusal_is_localized(subtests, isolated, monkeypatch):
    for label, case in locale_cases.cases(("language", ["sv", "de", "es", "zh_CN", "fr", "pt", "hi", "ko", "is"])):
        with subtests.test(**label), locale_cases.monkeypatch_scope() as case_monkeypatch:
            _actual_unknown_model_refusal_is_localized_case(isolated=isolated, monkeypatch=case_monkeypatch, **case)
