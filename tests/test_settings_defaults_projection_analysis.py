"""Static settings projections must produce honest consumer documentation.

Fixtures below exercise the analyzers only; none executes a scientific module
or supplies image-classifier acceptance evidence.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = {'net_gate', 'focus_floor'}
QC_KEYS = {'image_qc_classifier', 'image_qc_classifier_model',
           'image_qc_classifier_labels', 'image_qc_classifier_threshold'}
PROJECTION = '''
DEFAULTS = {'net_gate': False, 'focus_floor': 4}
def resolve(settings):
    policy = {key: settings.get(key, default)
              for key, default in DEFAULTS.items()}
    return policy
def apply(settings):
    policy = resolve(settings)
    return policy['net_gate']
'''


@pytest.fixture(params=['build_setting_consumer_map', 'settings_flow'])
def analyser(request):
    spec = importlib.util.spec_from_file_location(
        request.param, ROOT / 'tools' / f'{request.param}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _analyse_fixture(analyser, source, tmp_path, monkeypatch):
    """Run a verifier-only source fixture through one real analyzer."""
    tree = ast.parse(source)
    if hasattr(analyser, 'Reads'):
        tables = analyser._module_tables(tree)
        monkeypatch.setattr(analyser.Reads, 'helpers', frozenset(
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and analyser._returns_settings(node, tables)))
        visitor = analyser.Reads(KEYS, 'spacr.fixture')
        visitor.visit(tree)
        return [(hit['key'], hit['qualname'], hit['form'])
                for hit in visitor.hits]
    package = tmp_path / 'spacr'
    package.mkdir()
    (package / 'fixture.py').write_text(source, encoding='utf-8')
    monkeypatch.setattr(analyser, 'ROOT', tmp_path)
    monkeypatch.setattr(analyser, 'PACKAGE', package)
    monkeypatch.setattr(analyser, '_tooltips', lambda: dict.fromkeys(KEYS, 'Fixture'))
    data = analyser.analyse()
    return [(key, hit['function'].removeprefix('spacr.fixture.'), hit['form'])
            for key, hits in data['reads'].items() for hit in hits]


def test_defaults_projection_records_each_lookup_and_returned_policy(
        analyser, tmp_path, monkeypatch):
    hits = _analyse_fixture(analyser, PROJECTION, tmp_path, monkeypatch)
    assert {(key, 'resolve', 'get-dynamic') for key in KEYS} <= set(hits)
    assert ('net_gate', 'apply', 'subscript') in hits
    assert len(hits) == 3


def test_unrelated_dictionary_projection_is_not_a_settings_factory(
        analyser, tmp_path, monkeypatch):
    source = PROJECTION.replace('def resolve(settings):', 'def resolve(metadata):')
    source = source.replace('settings.get(key, default)', 'metadata.get(key, default)')
    assert _analyse_fixture(analyser, source, tmp_path, monkeypatch) == []


@pytest.mark.parametrize('expression', [
    '{key: metadata.get(key, default) for key, default in DEFAULTS.items()}',
    '{default: settings.get(key, default) for key, default in DEFAULTS.items()}',
    '{key: settings.get(other, default) for key, default in DEFAULTS.items()}',
    '{key: settings.get(key, other) for key, default in DEFAULTS.items()}',
    '{key: settings.get(key, default) for key, default in OTHER.items()}',
    '{key: settings.get(key, default) for key, default in DEFAULTS.values()}',
    '{key: settings.get(key, default) for key, default in DEFAULTS.items() if key}',
    '{key: default for key, default in DEFAULTS.items()}',
])
def test_projection_requires_the_complete_matching_settings_shape(
        analyser, expression):
    tree = ast.parse(PROJECTION)
    assert not analyser._projected_settings(
        ast.parse(expression, mode='eval').body, analyser._module_tables(tree))


def test_comprehension_key_does_not_leak_to_an_unrelated_later_lookup(
        analyser, tmp_path, monkeypatch):
    source = PROJECTION.split('def apply')[0].replace(
        'def resolve(settings):', 'def resolve(settings, key):').replace(
        'return policy', 'return settings.get(key)')
    hits = _analyse_fixture(analyser, source, tmp_path, monkeypatch)
    assert set(hits) == {(key, 'resolve', 'get-dynamic') for key in KEYS}
    assert len(hits) == len(KEYS)


def test_actual_qc_defaults_have_four_public_consumers(analyser):
    source = ROOT / 'spacr' / 'image_quality.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    if hasattr(analyser, 'Reads'):
        visitor = analyser.Reads(QC_KEYS, 'spacr.image_quality')
        visitor.visit(tree)
        consumers = {key: [hit for hit in visitor.hits if hit['key'] == key]
                     for key in QC_KEYS}
        targets = analyser.resolve_targets(consumers)
        assert targets == {key: {'module': 'spacr.image_quality',
                                'symbol': 'quality_policy', 'exact': True}
                           for key in QC_KEYS}
        readers = {key: {hit['qualname'] for hit in hits}
                   for key, hits in consumers.items()}
    else:
        data = analyser.analyse()
        readers = {key: {hit['function'].removeprefix('spacr.image_quality.')
                         for hit in data['reads'].get(key, [])}
                   for key in QC_KEYS}
    assert all('quality_policy' in hits for hits in readers.values())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'quality_policy')
    assert analyser._returns_settings(function, analyser._module_tables(tree))
