"""Plaque inference uses the weight precision recorded with its scorecard."""
from types import SimpleNamespace

import pytest

from spacr import model_zoo


@pytest.mark.parametrize("settings", [{"gpu": True}, {"gpu": True, "use_bfloat16": True},
                                      {"gpu": False, "use_bfloat16": False}])
def test_declared_float32_keeps_device_and_does_not_mutate_caller(settings):
    records = [{"key": "plaque_test", "name": "weights", "preprocess": {"use_bfloat16": False}}]
    original = dict(settings)
    result = model_zoo._model_precision_kwargs("/models/weights", settings, records)
    assert result == dict(settings, use_bfloat16=False)
    assert settings == original


def test_undeclared_models_keep_the_accelerator_precision():
    settings = {"gpu": True, "device": "cuda:0"}
    assert model_zoo._model_precision_kwargs("unlisted", settings, []) == settings


@pytest.mark.parametrize("factory", ["assay", "figure"])
def test_assay_and_figure_constructors_honor_the_same_record(monkeypatch, factory, tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"checkpoint")
    from spacr import accelerator
    monkeypatch.setattr(accelerator, "cellpose_kwargs", lambda: {"gpu": True, "device": "cuda:0"})
    monkeypatch.setattr(model_zoo, "_model_preprocess", lambda *a, **k: {"use_bfloat16": False})
    calls = []
    marker = SimpleNamespace(eval=lambda *a, **k: None)
    if factory == "assay":
        from spacr import submodules
        monkeypatch.setattr(submodules.cp_models, "CellposeModel", lambda **kw: calls.append(kw) or marker)
        assert submodules._plaque_cellpose_model(str(checkpoint)) is marker
    else:
        from cellpose import models

        from spacr import plaque_papers
        monkeypatch.setattr(models, "CellposeModel", lambda **kw: calls.append(kw) or marker)
        assert callable(plaque_papers._cellpose_segmenter(str(checkpoint)))
    assert calls == [{"pretrained_model": str(checkpoint), "device": None,
                      "gpu": True, "use_bfloat16": False}]
