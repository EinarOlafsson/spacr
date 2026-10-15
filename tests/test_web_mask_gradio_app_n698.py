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
    with pytest.raises(Exception, match="finite whole number"):
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


@pytest.mark.parametrize("position", [3, 4, 5, 7, 8])
@pytest.mark.parametrize("invalid", [0.5, float("nan"), float("inf"), True])
def test_fractional_nonfinite_and_bool_indices_never_reach_gpu(demo, tmp_path, monkeypatch, position, invalid):
    import sys
    import types
    original, calls = demo
    gpu_calls = []
    def gpu(**options):
        def decorate(function):
            def wrapped(*args, **kwargs):
                gpu_calls.append(True)
                return function(*args, **kwargs)
            return wrapped
        return decorate
    monkeypatch.setitem(sys.modules, "spaces", types.SimpleNamespace(GPU=gpu))
    actual = app.build_app({}, zero_gpu=True)
    source = tmp_path / "input.tif"
    tifffile.imwrite(source, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"})
    values = [str(source), "toxoplasma_plaque_v4", True, 0, 0, 0, [0], 0, 150, 52, False, 0.4, 0]
    values[position] = invalid
    try:
        import gradio as gr
        endpoint = next(item for item in actual.fns.values() if item.api_name == "segment_plaque")
        with pytest.raises((ValueError, gr.Error)):
            values[position] = endpoint.inputs[position].preprocess(invalid)
            endpoint.fn(*values)
        assert gpu_calls == []
        assert calls == []
    finally:
        actual.close()


def test_header_choice_callback_rejects_fractional_series_without_truncation(demo, tmp_path):
    actual, _ = demo
    source = tmp_path / "image.tif"
    tifffile.imwrite(source, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"})
    callback = next(item.fn for item in actual.fns.values() if item.fn and item.fn.__name__ == "choices")
    with pytest.raises(ValueError, match="finite whole number"):
        callback(str(source), 0.5)


def test_exact_integer_rest_floats_keep_the_explicit_selection(demo, tmp_path):
    actual, calls = demo
    source = tmp_path / "input.tif"
    tifffile.imwrite(source, np.ones((5, 7), dtype=np.uint16), metadata={"axes": "YX"})
    endpoint = next(item for item in actual.fns.values() if item.api_name == "segment_plaque")
    values = [str(source), "toxoplasma_plaque_v4", True, 0.0, 0.0, 0.0, [0.0], 0.0, 150.0, 52, False, 0.4, 0]
    for position in (3, 4, 5, 7, 8):
        values[position] = endpoint.inputs[position].preprocess(values[position])
    cached, _ = endpoint.fn(*values)
    assert Path(cached).is_file()
    selection = calls[-1]["selection"]
    assert selection == {"series": 0, "indices": {}, "channels": [0], "page": 0, "pdf_dpi": 150}
    assert all(type(selection[name]) is int for name in ("series", "page", "pdf_dpi"))
    assert type(selection["channels"][0]) is int


@pytest.mark.parametrize('family', list(app.FAMILY_LABELS))
def test_each_actual_endpoint_captures_its_own_family_and_settings(demo, tmp_path, family):
    actual, calls = demo
    source = tmp_path / 'image.tif'
    tifffile.imwrite(source, np.arange(35, dtype=np.uint16).reshape(5, 7), metadata={'axes': 'YX'})
    endpoint = 'segment_plaque' if family == 'toxoplasma_plaque' else 'detect_wells' if family == 'toxoplasma_well_detector' else 'segment_' + family
    callback = function(actual, endpoint)
    callback(str(source), family + '_v1', True, 0, 0, 0, [0], 0, 150, 32, False, 0.6, -2, 40,
             '{"background": 123, "remove_border_objects": true}', 0.4, 0.8)
    request = calls[-1]
    assert request['family'] == family and request['model_key'] == family + '_v1'
    if family == 'toxoplasma_plaque':
        assert request['settings'] == {'diameter': 32, 'flow_threshold': 0.6, 'CP_prob': -2}
    elif family == 'toxoplasma_well_detector':
        assert request['settings'] == {'confidence': 0.4, 'min_axis_ratio': 0.8}
    else:
        assert request['settings'] == {'diameter': 32, 'flow_threshold': 0.6, 'CP_prob': -2, 'magnification': 40, 'background': 123, 'remove_border_objects': True}


def test_generic_blank_auto_and_invalid_settings_remain_distinct(demo, tmp_path):
    actual, calls = demo
    source = tmp_path / 'image.tif'
    tifffile.imwrite(source, np.ones((5, 7), dtype=np.uint16), metadata={'axes': 'YX'})
    callback = function(actual, 'segment_live_cell')
    args = [str(source), 'live_cell_v1', True, 0, 0, 0, [0], 0, 150, None, False, 0.4, 0]
    callback(*args)
    assert calls[-1]['settings']['diameter'] is None
    args[10] = True
    callback(*args)
    assert calls[-1]['settings']['diameter'] == 'auto'
    before = len(calls)
    for invalid in ('{invalid', '[]', '{"checkpoint": "/untrusted"}'):
        with pytest.raises(Exception):
            callback(*args, 20, invalid)
    assert len(calls) == before
    with pytest.raises(Exception, match='one channel'):
        callback(*[args[i] if i != 6 else [0, 1] for i in range(len(args))])
    assert len(calls) == before


def test_detector_preview_uses_actual_boxes_and_no_mask(tmp_path):
    from tools.web_mask_generator.detector import build_well_result_zip
    from spacr.plaque import Well
    source = tmp_path / 'image.tif'
    image = np.zeros((5, 7), dtype=np.uint8)
    tifffile.imwrite(source, image)
    target = tmp_path / 'detector.zip'
    build_well_result_zip(target, input_file=source, image=image, image_axes='YX', wells=[Well(1, 1, 5, 4, 0.9)],
                          model_id='detector', model_version='1', weights_sha256='ab' * 32, settings={})
    overlay = app._preview(target)
    assert np.all(overlay[1, 1:5] == [255, 70, 70])
    assert np.all(overlay[3, 1:5] == [255, 70, 70])
    assert np.all(overlay[2, 2:4] == 0)
    with zipfile.ZipFile(target) as archive:
        assert 'mask.tif' not in archive.namelist()
