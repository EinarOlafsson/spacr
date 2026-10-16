"""Compatibility must preserve the guard against heavyweight top-level imports."""
from importlib.metadata import distribution
import sys

import pytest

from tests.stdlib_inventory import stdlib_names


@pytest.mark.parametrize('old_python', [False, True])
def test_stdlib_inventory_accepts_core_and_extension_modules_but_not_extras(monkeypatch, old_python):
    if old_python:
        monkeypatch.delattr(sys, 'stdlib_module_names', raising=False)
    names = stdlib_names()
    assert {'__future__', 'sys', 'os', 'json', 'hashlib', 'ctypes', 'dataclasses', 'pathlib', 'math', 'winreg', 'msvcrt'} <= names
    assert distribution('pytest').metadata['Name'].lower() == 'pytest'
    assert not {'numpy', 'torch', 'pytest', 'spacr'} & names


def test_modern_stdlib_inventory_is_returned_without_fallback_additions(monkeypatch):
    import tests.stdlib_inventory as inventory

    published = frozenset({'sys', 'os', 'modern_only'})
    monkeypatch.setattr(sys, 'stdlib_module_names', published, raising=False)

    def forbid_fallback(*args, **kwargs):
        raise AssertionError('The published stdlib inventory must not scan package paths')

    monkeypatch.setattr(inventory.pkgutil, 'iter_modules', forbid_fallback)
    assert stdlib_names() == set(published)
    assert sys.stdlib_module_names is published
