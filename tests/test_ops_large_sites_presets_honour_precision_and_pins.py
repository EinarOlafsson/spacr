"""The ops_large_sites presets build models with the run's precision and pins."""
import spacr.utils as utils
import spacr.ops_large_sites as ols


def test_preset_models_pass_the_whole_settings_to_the_model_builder(monkeypatch):
    seen = {}

    def fake_build(name, *, settings=None, key=None, **kwargs):
        seen[key] = (name, dict(settings), kwargs)
        return key

    monkeypatch.setattr(utils, "build_cellpose_model", fake_build)
    settings = {"seg_nucleus_model": "cpsam", "cell_model": "cpsam",
                "mask_inference_precision": "fp32",
                "model_metadata": {"settings": {"cell_model": {"sha256": "x"}}}}
    assert ols.preset_models(settings) == ("seg_nucleus_model", "cell_model")
    for key, kind in (("seg_nucleus_model", "nucleus"), ("cell_model", "cell")):
        name, passed, kwargs = seen[key]
        assert name == "cpsam"
        assert passed["mask_inference_precision"] == "fp32"
        assert passed["model_metadata"] == settings["model_metadata"]
        assert kwargs["object_type"] == kind
