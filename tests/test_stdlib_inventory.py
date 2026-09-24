"""Compatibility must preserve the guard against heavyweight top-level imports."""
import sys

import pytest

from tests.stdlib_inventory import stdlib_names


@pytest.mark.parametrize('old_python', [False, True])
def test_stdlib_inventory_accepts_core_and_extension_modules_but_not_extras(monkeypatch, old_python):
    if old_python:
        monkeypatch.delattr(sys, 'stdlib_module_names', raising=False)
    names = stdlib_names()
    assert {'__future__', 'sys', 'os', 'json', 'hashlib', 'ctypes', 'dataclasses', 'pathlib', 'math', 'winreg'} <= names
    assert not {'numpy', 'torch', 'pytest', 'spacr'} & names
