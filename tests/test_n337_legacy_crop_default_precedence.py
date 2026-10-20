"""Legacy crop channels survive real generic settings defaults and file loading."""
import csv
import json

import numpy as np
import pytest


def resolved_settings(boundary, source, tmp_path):
    from spacr.cli import resolve_module, resolve_settings
    from spacr.settings import get_measure_crop_settings

    if boundary == 'factory':
        return get_measure_crop_settings(dict(source))
    path = tmp_path / ('settings.' + boundary)
    if boundary == 'json':
        path.write_text(json.dumps(source), encoding="utf-8")
    else:
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Key', 'Value'])
            writer.writerows((key, 'None' if value is None else value)
                             for key, value in source.items())
    if boundary == 'csv' and 'png_channel_mapping' in source and source['png_channel_mapping'] is None:
        from spacr.cli import load_settings_file

        assert load_settings_file(str(path))['png_channel_mapping'] == 'None'
    before = path.read_bytes()
    resolved = resolve_settings(resolve_module('measure'), str(path))
    assert path.read_bytes() == before
    return resolved


@pytest.mark.parametrize('boundary', ['factory', 'csv', 'json'])
@pytest.mark.parametrize('dims, expected', [([1, 3, 0], [10, 180, 40]), ([1, 3], [0, 180, 40]), ([3], [180])])
def test_legacy_only_crop_channels_reach_native_rgb(boundary, dims, expected, tmp_path):
    from spacr.crops import build_png_channels, resolve_png_channel_mapping

    settings = resolved_settings(boundary, {'png_dims': dims}, tmp_path)
    raw = np.broadcast_to(np.array([10, 40, 90, 180], dtype=np.uint8), (3, 4, 4))
    result = build_png_channels(raw, resolve_png_channel_mapping(settings))
    assert result[1, 1].tolist() == expected
    assert settings['png_dims'] == dims


@pytest.mark.parametrize('boundary', ['factory', 'csv', 'json'])
def test_explicit_current_mapping_retains_precedence(boundary, tmp_path):
    from spacr.crops import resolve_png_channel_mapping

    current = {'r': 2, 'g': 1, 'b': 0}
    resolved = resolved_settings(boundary, {'png_dims': [1, 3, 0], 'png_channel_mapping': current}, tmp_path)
    assert resolve_png_channel_mapping(resolved) == current


@pytest.mark.parametrize('boundary', ['factory', 'csv', 'json'])
def test_explicit_none_mapping_keeps_loader_semantics(boundary, tmp_path):
    from spacr.crops import CropError, resolve_png_channel_mapping

    resolved = resolved_settings(boundary, {'png_dims': [1, 3, 0], 'png_channel_mapping': None}, tmp_path)
    if boundary == 'csv':
        assert resolved['png_channel_mapping'] == 'None'
        with pytest.raises(CropError, match='must be a dict'):
            resolve_png_channel_mapping(resolved)
    else:
        assert resolved['png_channel_mapping'] is None
        assert resolve_png_channel_mapping(resolved) == {'r': 0, 'g': 3, 'b': 1}


@pytest.mark.parametrize('boundary', ['factory', 'csv', 'json'])
@pytest.mark.parametrize('legacy', [None, []])
def test_unspecified_legacy_mapping_preserves_stock_default(boundary, legacy, tmp_path):
    from spacr.crops import resolve_png_channel_mapping

    resolved = resolved_settings(boundary, {'png_dims': legacy}, tmp_path)
    assert resolve_png_channel_mapping(resolved) == {'r': 2, 'g': 1, 'b': 0}


def test_absent_legacy_mapping_preserves_stock_default(tmp_path):
    from spacr.crops import resolve_png_channel_mapping

    assert resolve_png_channel_mapping(resolved_settings('factory', {}, tmp_path)) == {'r': 2, 'g': 1, 'b': 0}


def test_factory_parses_existing_string_list_before_legacy_conversion(tmp_path):
    from spacr.crops import resolve_png_channel_mapping

    assert resolve_png_channel_mapping(resolved_settings('factory', {'png_dims': '[1, 3, 0]'}, tmp_path)) == {'r': 0, 'g': 3, 'b': 1}
