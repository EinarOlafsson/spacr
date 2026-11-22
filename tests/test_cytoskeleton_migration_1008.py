"""Schema 958 preserves historical whole-image cytoskeleton measurements."""
import json

import numpy as np
import pytest

from spacr import measure as M
from spacr import settings as S


def _saved_overview_settings(tmp_path, version, explicit=None):
    """Read a real settings JSON through the shared CLI migration path."""
    from spacr.cli import _in_its_own_era, load_settings_file

    payload = {} if version is None else {S.SETTINGS_VERSION_KEY: version}
    if explicit is not None:
        payload['cytoskeleton_object_support'] = explicit
    path = tmp_path / 'overview.json'
    path.write_text(json.dumps(payload))
    defaults = S.get_overview_default_settings()
    loaded = _in_its_own_era(load_settings_file(str(path)), known=defaults)
    defaults.update(loaded)
    return defaults


@pytest.mark.parametrize('version', [None, 950, 955, 956, 957])
def test_old_saved_overview_files_keep_legacy_counts(tmp_path, version):
    """Unstamped and pre-958 files receive the previous numerical default."""
    settings = _saved_overview_settings(tmp_path, version)
    assert settings['cytoskeleton_object_support'] is False
    assert S.SETTINGS_VERSION_KEY not in settings
    assert S.DEFAULT_CHANGES[958] == {'cytoskeleton_object_support': False}


@pytest.mark.parametrize('version,explicit,expected', [(958, None, True),
    (950, True, True), (958, False, False)])
def test_current_files_and_explicit_choices_use_the_requested_semantics(
        tmp_path, version, explicit, expected):
    """The current stamp uses corrected counts and an explicit choice always wins."""
    assert _saved_overview_settings(tmp_path, version, explicit)[
        'cytoskeleton_object_support'] is expected
    assert S.get_overview_default_settings()['cytoskeleton_object_support'] is True
    assert 'cytoskeleton_object_support' not in S.get_measure_crop_settings()
    assert S.SETTINGS_SCHEMA_VERSION >= 958


@pytest.mark.parametrize('version,expected', [(950, 23), (958, 3)])
def test_saved_settings_reach_the_actual_overview_measurement(
        tmp_path, monkeypatch, version, expected):
    """The actual Overview caller propagates the migrated mode into the shared helper."""
    from spacr.qt.widgets import qc_field_browser as browser

    mask = np.zeros((180, 200), bool)
    mask[70:90, 80:100] = True
    image = np.full(mask.shape, np.nan)
    image[mask] = np.random.default_rng(1008).integers(1, 1000, mask.sum())
    settings = _saved_overview_settings(tmp_path, version)
    for key in tuple(settings):
        if key.startswith('overview_measure_') and isinstance(settings[key], bool):
            settings[key] = False
    settings['overview_measure_fibres'] = True

    def raw_planes(field, channels, **options):
        """Supply acquired pixels at the existing reader seam for one synthetic field."""
        yield 0, str(tmp_path / 'field.npy'), image

    monkeypatch.setattr(browser, '_overview_raw_planes', raw_planes)
    field = browser._OverviewField('plate1', 'A01', 'A', 1, '1',
                                    (('', str(tmp_path / 'field.npy')),))
    row = browser._overview_measurement(field, (0,), measurements=settings)
    assert row['status'] == 'ok', row['error']
    assert row['channel_0_fibres_skeleton_length'] == expected
    assert row['channel_0_fibres_skeleton_density'] == expected / mask.sum()


def test_hidden_compatibility_value_survives_overview_import_and_collect(tmp_path):
    """The existing hidden-value mechanism keeps replay mode without a GUI toggle."""
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import SettingsWidgets, _APP_HIDDEN_KEYS
    from types import SimpleNamespace

    model = SettingsWidgets('overview')
    assert 'cytoskeleton_object_support' in _APP_HIDDEN_KEYS['overview']
    host = SimpleNamespace(_settings_model=model)
    old = AppScreen._in_its_own_era(host, {S.SETTINGS_VERSION_KEY: 950})
    assert old['cytoskeleton_object_support'] is False
    assert model.set_hidden_value('cytoskeleton_object_support', False)
    assert model.collect()['cytoskeleton_object_support'] is False
    assert 'cytoskeleton_object_support' not in model._widgets


def test_schema958_runtime_arrivals_are_exact_canonical_sources(monkeypatch):
    """Bind all new runtime and compatibility-setting prose before locale review."""
    import ast
    import hashlib
    import importlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / 'tools'))
    builder = importlib.import_module('build_i18n_catalogs')
    canonical = builder.canonical_sources()
    contract = json.loads((root / 'tests/data/release_contracts/1008c_runtime_arrivals_2026-11-21.json').read_text())
    assert contract['schema'] == 1
    rows = contract['sources']
    tree = ast.parse((root / 'spacr/measure.py').read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'measure_cytoskeleton')
    messages = {node.args[0].value for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'tr' and node.args and isinstance(node.args[0], ast.Constant)}
    expected = {('ui', message) for message in messages} | {
        ('setting_labels', 'cytoskeleton_object_support'),
        ('setting_tooltips', 'cytoskeleton_object_support')}
    assert len(rows) == len(expected) == 4
    assert {(row['table'], row['key']) for row in rows} == expected
    for row in rows:
        values = canonical[row['table']]
        if isinstance(values, dict):
            assert values[row['key']] == row['source']
        else:
            assert row['key'] == row['source'] and row['source'] in values
        assert hashlib.sha256(row['source'].encode()).hexdigest() == row['source_sha256']
