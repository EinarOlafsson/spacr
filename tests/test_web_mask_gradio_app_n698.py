"""Actual optional Gradio graph and callback contracts, no listening server."""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pytest
import tifffile

pytest.importorskip("gradio")

from tools.web_mask_generator import app, results


@pytest.fixture
def demo(tmp_path, monkeypatch):
    calls = []
    class Controller:
        def __init__(self, *a, **kw):
            pass
        def preload_default(self):
            calls.append("preload")
        @contextmanager
        def job(self, path, request):
            calls.append(request)
            with tempfile.TemporaryDirectory(dir=tmp_path) as directory:
                output = Path(directory) / "result.zip"
                image = tifffile.imread(path)
                mask = np.zeros(image.shape[-2:], dtype=np.uint32)
                mask[1:3, 2:4] = 9
                results.build_result_zip(output, input_file=path, image=image, image_axes="YX", masks=mask,
                    model_id="toxoplasma_plaque_v4", model_version="4", weights_sha256="ab" * 32,
                    settings=request["settings"], unavailable={"flows": "no model used", "cell_probability": "no model used"})
                yield {"result_zip": output}
            assert not output.exists()
    monkeypatch.setattr(app, "PlaqueService", Controller)
    result = app.build_app({}, job_parent=tmp_path)
    result.GRADIO_CACHE = str(tmp_path / "gradio-cache")
    yield result, calls
    result.close()


def function(demo, name):
    return next(item.fn for item in demo.fns.values() if item.api_name == name)


def test_actual_app_builds_eight_class_tabs_without_server_or_model_start(demo):
    actual, calls = demo
    assert calls == []
    assert actual.is_running is False
    config = actual.get_config_file()
    tabs = [component for component in config["components"] if component["type"] == "tabitem"]
    assert len(tabs) == 8
    assert {item["family"] for item in function(actual, "models")()} == set(app.FAMILY_LABELS)
    assert actual._queue.max_size == 8


def test_actual_callback_caches_result_before_private_job_is_removed(demo, tmp_path):
    actual, calls = demo
    image = tmp_path / "input.tif"
    tifffile.imwrite(image, np.arange(35, dtype=np.uint16).reshape(5, 7), metadata={"axes": "YX"})
    cached, preview = function(actual, "segment_plaque")(
        str(image), "toxoplasma_plaque_v4", True, 0, 0, 0, [0], 0, 150, 52, False, 0.4, -1)
    assert Path(cached).is_file()
    with zipfile.ZipFile(cached) as archive:
        assert archive.read("input/input.tif") == image.read_bytes()
    assert preview.shape == (5, 7, 3) and preview.dtype == np.uint8
    assert np.any(np.all(preview == [255, 70, 70], axis=-1))
    assert calls[-1]["settings"] == {"diameter": 52, "flow_threshold": 0.4, "CP_prob": -1}
    assert calls[-1]["selection"] == {"series": 0, "indices": {}, "channels": [0], "page": 0, "pdf_dpi": 150}


def test_missing_input_and_implicit_scale_do_not_enter_backend(demo, tmp_path):
    actual, calls = demo
    callback = function(actual, "segment_plaque")
    with pytest.raises(Exception, match="Choose an image"):
        callback(None, "key", True, 0, 0, 0, [0], 0, 150, 52, False, 0.4, 0)
    image = tmp_path / "input.tif"
    tifffile.imwrite(image, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"})
    with pytest.raises(Exception, match="resolved diameter"):
        callback(str(image), "key", True, 0, 0, 0, [0], 0, 150, None, False, 0.4, 0)
    assert calls == []


def test_refused_consent_and_bad_series_never_enter_backend(demo, tmp_path):
    actual, calls = demo
    image = tmp_path / "input.tif"
    tifffile.imwrite(image, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"})
    callback = function(actual, "segment_plaque")
    with pytest.raises(Exception, match="consent"):
        callback(str(image), "key", False, 0, 0, 0, [0], 0, 150, 52, False, 0.4, 0)
    with pytest.raises(Exception, match="out of range"):
        callback(str(image), "key", True, -1, 0, 0, [0], 0, 150, 52, False, 0.4, 0)
    assert calls == []


def test_real_upload_choice_callbacks_expose_channels_and_handle_pdf(demo, tmp_path):
    actual, _ = demo
    callback = next(item.fn for item in actual.fns.values() if item.fn and item.fn.__name__ == "choices")
    empty, text = callback(None, 0)
    assert empty.value == [] and text == ""
    source = tmp_path / "image.tif"
    tifffile.imwrite(source, np.ones((2, 5, 7), dtype=np.uint16), metadata={"axes": "CYX"}, photometric="minisblack")
    dropdown, text = callback(str(source), 0)
    assert [value for _, value in dropdown.choices] == [0, 1]
    assert dropdown.value == []
    assert dropdown.multiselect is True
    assert "7 × 5" in text
    with pytest.raises(Exception, match="out of range"):
        callback(str(source), 1)
    from PIL import Image
    pdf = tmp_path / "page.pdf"
    Image.new("RGB", (5, 7)).save(pdf, "PDF")
    dropdown, text = callback(str(pdf), 0)
    assert [value for _, value in dropdown.choices] == [0, 1, 2]
    assert "PDF" in text


def test_preview_handles_constant_and_multichannel_pixels(tmp_path):
    source = tmp_path / "image.tif"
    image = np.ones((2, 5, 7), dtype=np.uint16)
    tifffile.imwrite(source, image, metadata={"axes": "CYX"}, photometric="minisblack")
    output = tmp_path / "result.zip"
    results.build_result_zip(output, input_file=source, image=image, image_axes="CYX", masks=np.zeros((5, 7), dtype=np.uint32),
        model_id="fixture", model_version="one", weights_sha256="ab" * 32,
        settings={}, unavailable={"flows": "none", "cell_probability": "none"})
    preview = app._preview(output)
    assert preview.shape == (5, 7, 3)
    assert not preview.any()


def test_actual_spaces_wrapper_constructs_without_launch_or_model_loading(demo):
    pytest.importorskip("spaces")
    _, calls = demo
    actual = app.build_app({}, zero_gpu=True)
    try:
        assert actual.is_running is False
        assert function(actual, "segment_plaque") is not None
        assert calls == []
    finally:
        actual.close()


@pytest.mark.parametrize("enabled", [False, True])
def test_launch_policy_is_explicit_and_never_enables_a_public_share(monkeypatch, enabled):
    calls = []
    class Demo:
        def launch(self, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(app, "build_app", lambda paths, **kwargs: calls.append((paths, kwargs)) or Demo())
    monkeypatch.setenv("SPACR_WEB_MODELS_JSON", '{"model":"/operator/weights"}')
    monkeypatch.setenv("SPACR_WEB_ZERO_GPU", "1" if enabled else "0")
    app.main()
    assert calls[0] == ({"model": "/operator/weights"}, {"preload": enabled, "zero_gpu": enabled})
    assert calls[1]["share"] is False
    assert calls[1]["max_file_size"] == "64mb"
    assert calls[1]["run_history"] is False
    assert calls[1]["strict_cors"] is True


def test_model_version_changes_update_the_shown_provenance(demo):
    actual, _ = demo
    callback = next(item.fn for item in actual.fns.values() if item.fn and item.fn.__name__ == "show_card")
    entries = callback.__defaults__[0]
    key = next(iter(entries))
    shown = callback(key)
    assert entries[key]["display_name"] in shown
    assert entries[key]["source_url"] in shown


def test_operator_preload_is_only_an_explicit_build_choice(demo):
    _, calls = demo
    actual = app.build_app({}, preload=True)
    try:
        assert calls == ["preload"]
        assert actual.is_running is False
    finally:
        actual.close()


def test_space_file_entry_imports_source_checkout_and_calls_only_main(monkeypatch):
    import runpy
    import sys
    import types
    calls = []
    monkeypatch.setitem(sys.modules, "tools.web_mask_generator.app", types.SimpleNamespace(main=lambda: calls.append("entry")))
    runpy.run_path(str(Path(__file__).parents[1] / "tools/web_mask_generator/space_app.py"), run_name="__main__")
    assert calls == ["entry"]
