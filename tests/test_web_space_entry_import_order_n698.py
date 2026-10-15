"""Actual Space entrypoint import order with isolated optional-framework doubles."""
import importlib
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


@pytest.fixture
def imports(monkeypatch):
    monkeypatch.setattr(sys, 'path', list(sys.path))
    events = []
    app = ModuleType('tools.web_mask_generator.app')
    def main():
        events.append('main-called')
    def lookup(name):
        if name == 'main':
            events.append('app-import')
            return main
        raise AttributeError(name)
    app.__getattr__ = lookup
    monkeypatch.setitem(sys.modules, 'tools.web_mask_generator.app', app)
    original = importlib.import_module
    def import_module(name, *args, **kwargs):
        if name == 'spaces':
            events.append('spaces-import')
            return ModuleType('spaces')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, 'import_module', import_module)
    return events


def entry(name='isolated_web_space_entry'):
    source = Path(__file__).parents[1] / 'tools/web_mask_generator/space_app.py'
    return runpy.run_path(str(source), run_name=name)


def test_explicit_zerogpu_initializes_spaces_before_app_import(imports, monkeypatch):
    monkeypatch.setenv('SPACR_WEB_ZERO_GPU', '1')
    entry()
    assert imports[0] == 'spaces-import'
    assert imports.count('spaces-import') == 1
    assert imports.index('spaces-import') < imports.index('app-import')
    assert 'main-called' not in imports


@pytest.mark.parametrize('setting', [None, '0', 'off', 'false', 'yes'])
def test_cpu_entry_keeps_spaces_optional_and_does_not_launch(imports, monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv('SPACR_WEB_ZERO_GPU', raising=False)
    else:
        monkeypatch.setenv('SPACR_WEB_ZERO_GPU', setting)
    entry()
    assert 'spaces-import' not in imports
    assert 'app-import' in imports
    assert 'main-called' not in imports


def test_explicit_zerogpu_missing_framework_refuses_before_app(imports, monkeypatch):
    monkeypatch.setenv('SPACR_WEB_ZERO_GPU', '1')
    def missing(name):
        assert name == 'spaces'
        raise ModuleNotFoundError('controlled missing optional spaces package')
    monkeypatch.setattr(importlib, 'import_module', missing)
    with pytest.raises(ModuleNotFoundError, match='missing optional spaces'):
        entry()
    assert imports == []


@pytest.mark.parametrize('setting', ['0', '1'])
def test_main_guard_still_launches_only_when_run_as_main(imports, monkeypatch, setting):
    monkeypatch.setenv('SPACR_WEB_ZERO_GPU', setting)
    entry('__main__')
    assert imports.count('main-called') == 1
    assert imports[-1] == 'main-called'
