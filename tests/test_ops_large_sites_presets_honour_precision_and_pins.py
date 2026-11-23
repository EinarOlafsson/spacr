"""The ops_large_sites presets build models with the run's precision and pins."""
import spacr.ops_engine as engine
import spacr.utils as utils
import spacr.ops_large_sites as ols


def test_preset_models_pass_the_whole_settings_to_the_model_builder(monkeypatch):
    seen = {}
    monkeypatch.setattr(utils, "_resolve_cellpose_pretrained",
                        lambda name, kind, setting=None: name)
    monkeypatch.setattr(engine, "_cellpose_model",
                        lambda settings, gpu, key: seen.setdefault(key, dict(settings)) and key)
    settings = {"seg_nucleus_model": "cpsam", "cell_model": "cpsam",
                "mask_inference_precision": "fp32",
                "model_metadata": {"settings": {"cell_model": {"sha256": "x"}}}}
    ols.preset_models(settings)
    for key in ("seg_nucleus_model", "cell_model"):
        assert seen[key]["mask_inference_precision"] == "fp32"
        assert seen[key]["model_metadata"] == settings["model_metadata"]
