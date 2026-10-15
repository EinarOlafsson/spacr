"""Managed SDK launch compatibility without opening a server or loading weights."""
import inspect
import json

import pytest


gr = pytest.importorskip("gradio")

from tools.web_mask_generator import app


@pytest.mark.parametrize("version", ["6.17.3", "6.30.0", "6.17.4", "6.17.30", "6.17.3+local"])
@pytest.mark.parametrize("zero_gpu", [False, True])
def test_only_exact_managed_6173_omits_the_unavailable_history_keyword(monkeypatch, version, zero_gpu):
    calls = []

    class Demo:
        def launch(self, **kwargs):
            calls.append(("launch", kwargs))

    def build(paths, **kwargs):
        calls.append(("build", paths, kwargs))
        return Demo()

    monkeypatch.setattr(gr, "__version__", version)
    monkeypatch.setattr(app, "build_app", build)
    models = {"toxoplasma_plaque_v4": "/operator-only/checkpoint"}
    examples = [{"family": "fixture", "revision": "immutable-fixture"}]
    monkeypatch.setenv("SPACR_WEB_MODELS_JSON", json.dumps(models))
    monkeypatch.setenv("SPACR_WEB_EXAMPLES_JSON", json.dumps(examples))
    monkeypatch.setenv("SPACR_WEB_ZERO_GPU", "1" if zero_gpu else "0")
    app.main()
    assert calls[0] == ("build", models, {"preload": zero_gpu, "zero_gpu": zero_gpu,
                                        "approved_examples": examples})
    expected = {"share": False, "server_name": "0.0.0.0", "max_file_size": "64mb",
                "show_error": False, "theme": "base", "strict_cors": True}
    if version != "6.17.3":
        expected["run_history"] = False
    assert calls[1:] == [("launch", expected)]


def test_actual_managed_6173_constructs_all_routes_and_accepts_launch_signature_without_server(monkeypatch):
    if gr.__version__ != "6.17.3":
        pytest.skip("Requires the exact optional managed SDK candidate")
    launches = []
    created = []
    launch_signature = inspect.signature(gr.Blocks.launch)
    assert "run_history" not in launch_signature.parameters
    original_build = app.build_app

    def build(*args, **kwargs):
        demo = original_build(*args, **kwargs)
        created.append(demo)
        return demo

    def launch(self, *args, **kwargs):
        launch_signature.bind(self, *args, **kwargs)
        launches.append(kwargs)
        assert self.is_running is False

    monkeypatch.setattr(app, "build_app", build)
    monkeypatch.setattr(gr.Blocks, "launch", launch)
    monkeypatch.setenv("SPACR_WEB_MODELS_JSON", "{}")
    monkeypatch.setenv("SPACR_WEB_ZERO_GPU", "0")
    monkeypatch.delenv("SPACR_WEB_EXAMPLES_JSON", raising=False)
    try:
        app.main()
        assert len(created) == len(launches) == 1
        demo = created[0]
        config = demo.get_config_file()
        tabs = [component for component in config["components"] if component["type"] == "tabitem"]
        assert len(tabs) == 8
        assert {fn.api_name for fn in demo.fns.values() if fn.api_visibility == "public"} == {
            "models", "segment_plaque", "detect_wells", "segment_toxoplasma_pv", "segment_live_cell",
            "segment_nuclei_from_cellmask", "segment_cell_from_hoechst", "segment_toxoplasma_from_cellmask",
            "segment_toxoplasma_from_hoechst"}
        inference = [fn for fn in demo.fns.values() if fn.api_name != "models" and fn.api_visibility == "public"]
        assert all(len(fn.inputs) == 24 for fn in inference)
        assert demo._queue.max_size == 8
        assert launches == [{"share": False, "server_name": "0.0.0.0", "max_file_size": "64mb",
                             "show_error": False, "theme": "base", "strict_cors": True}]
        assert not demo.is_running
    finally:
        for demo in created:
            demo.close()
