"""Actual managed SDK preprocessing and pre-GPU channel guards, no server/models."""
from contextlib import contextmanager
import functools
import sys
import types

import numpy as np
import pytest
import tifffile

gr = pytest.importorskip("gradio")

from tools.web_mask_generator import app


class GPUEntry(RuntimeError):
    pass


@pytest.fixture
def graph(monkeypatch):
    calls = []

    class Controller:
        def __init__(self, *args, **kwargs):
            pass

        @contextmanager
        def job(self, *args, **kwargs):
            pytest.fail('channel guard reached model controller')
            yield

    def gpu(**options):
        assert options == {'duration': app.gpu_seconds}

        def decorate(function):
            @functools.wraps(function)
            def sentinel(*args, **kwargs):
                calls.append((args, kwargs))
                raise GPUEntry('valid request reached guarded GPU boundary')
            return sentinel
        return decorate

    monkeypatch.setattr(app, 'PlaqueService', Controller)
    monkeypatch.setitem(sys.modules, 'spaces', types.SimpleNamespace(GPU=gpu))
    demo = app.build_app({}, zero_gpu=True)
    assert not demo.is_running
    try:
        yield demo, calls
    finally:
        demo.close()


def route(demo, name):
    return next(item for item in demo.fns.values() if item.api_name == name)


def invoke(function, image, channels, *, accepted=True):
    return function(str(image), 'nuclei_from_cellmask_v1', accepted,
                    0, 0, 0, channels, 0, 150, 32, False, 0.4, 0)


def test_managed_empty_channel_choices_accept_explicit_api_channel(graph):
    demo, calls = graph
    channels = route(demo, 'segment_nuclei_from_cellmask').inputs[6]
    assert channels.choices == []
    assert channels.preprocess([0]) == [0]
    assert calls == []


def test_native_header_range_refuses_before_gpu_entry(graph, tmp_path):
    demo, calls = graph
    image = tmp_path / 'one-channel.tif'
    tifffile.imwrite(image, np.ones((5, 7), dtype=np.uint16), metadata={'axes': 'YX'})
    with pytest.raises(gr.Error, match='within the declared range'):
        invoke(route(demo, 'segment_nuclei_from_cellmask').fn, image, [1])
    assert calls == []


@pytest.mark.parametrize('channels', [[0.5], [True], ['0'], [-1]])
def test_noninteger_and_negative_channels_refuse_before_gpu(graph, tmp_path, channels):
    demo, calls = graph
    image = tmp_path / 'one-channel.tif'
    tifffile.imwrite(image, np.ones((5, 7), dtype=np.uint16), metadata={'axes': 'YX'})
    with pytest.raises((gr.Error, ValueError), match='finite whole number'):
        invoke(route(demo, 'segment_nuclei_from_cellmask').fn, image, channels)
    assert calls == []


@pytest.mark.parametrize('endpoint,channels,count', [
    ('segment_nuclei_from_cellmask', [0, 1], 2),
    ('segment_cell_from_hoechst', [0, 1, 2], 3),
    ('segment_plaque', [0, 1, 2, 3], 4),
    ('detect_wells', [0, 1], 3),
])
def test_family_channel_limits_refuse_before_gpu(graph, tmp_path, endpoint, channels, count):
    demo, calls = graph
    image = tmp_path / 'multiple-channels.tif'
    tifffile.imwrite(image, np.ones((count, 5, 7), dtype=np.uint16), metadata={'axes': 'CYX'}, photometric='minisblack')
    with pytest.raises(gr.Error, match='Choose one channel'):
        invoke(route(demo, endpoint).fn, image, channels)
    assert calls == []


def test_duplicate_channels_refuse_before_gpu(graph, tmp_path):
    demo, calls = graph
    image = tmp_path / 'one-channel.tif'
    tifffile.imwrite(image, np.ones((5, 7), dtype=np.uint16), metadata={'axes': 'YX'})
    with pytest.raises(gr.Error, match='distinct'):
        invoke(route(demo, 'segment_plaque').fn, image, [0, 0])
    assert calls == []


def test_valid_native_header_selection_reaches_only_gpu_sentinel(graph, tmp_path):
    demo, calls = graph
    image = tmp_path / 'one-channel.tif'
    tifffile.imwrite(image, np.ones((5, 7), dtype=np.uint16), metadata={'axes': 'YX'})
    with pytest.raises(GPUEntry):
        invoke(route(demo, 'segment_nuclei_from_cellmask').fn, image, [0])
    assert len(calls) == 1
    assert calls[0][0][1]['selection'] == {'series': 0, 'indices': {}, 'channels': [0], 'page': 0, 'pdf_dpi': 150}


def test_all_eight_routes_keep_file_preprocessing_and_24_inputs(graph):
    demo, calls = graph
    routes = [item for item in demo.fns.values() if item.api_visibility == 'public' and item.api_name != 'models']
    assert len(routes) == 8
    assert all(len(item.inputs) == 24 and item.preprocess is True for item in routes)
    assert all(isinstance(item.inputs[0], gr.File) and item.inputs[0].type == 'filepath' for item in routes)
    assert all(item.inputs[2].value is False and item.inputs[0].interactive is False for item in routes)
    assert calls == []


def test_dynamic_ui_channel_choices_retain_api_compatibility(graph, tmp_path):
    demo, calls = graph
    callback = next(item.fn for item in demo.fns.values() if item.fn and item.fn.__name__ == 'choices')
    empty, _ = callback(None, 0)
    assert empty.allow_custom_value is True and empty.value == []
    image = tmp_path / 'two-channels.tif'
    tifffile.imwrite(image, np.ones((2, 5, 7), dtype=np.uint16), metadata={'axes': 'CYX'}, photometric='minisblack')
    choices, _ = callback(str(image), 0)
    assert choices.allow_custom_value is True
    assert choices.choices == [('Channel 1', 0), ('Channel 2', 1)]
    assert choices.preprocess([1]) == [1]
    assert calls == []


def test_gpu_reservation_scales_with_upload_and_stays_bounded(tmp_path):
    small = tmp_path / 'small.tif'
    small.write_bytes(b'0' * 1000)
    large = tmp_path / 'large.tif'
    large.write_bytes(b'0' * 64_000_000)
    assert app.gpu_seconds(str(small), {}) == 40
    assert app.gpu_seconds(str(large), {}) == 120
    assert app.gpu_seconds(str(tmp_path / 'missing.tif'), {}) == 40
