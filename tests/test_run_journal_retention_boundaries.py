"""Portable journal startup and cache refusals preserve existing state."""
import builtins
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest

from spacr import errors, logging_util
from spacr import run_journal as journal


def test_journal_import_without_fork_hooks_keeps_a_working_independent_registry(tmp_path, monkeypatch):
    portable_os = ModuleType('os')
    portable_os.__dict__.update(vars(os))
    portable_os.__dict__.pop('register_at_fork', None)
    private_errors = ModuleType('spacr.errors')
    private_errors.__dict__.update(vars(errors))
    private_errors._FINALIZE_LISTENERS = []
    actual_import = builtins.__import__

    def portable_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'os' and level == 0:
            return portable_os
        if name == 'errors' and level == 1:
            return private_errors
        return actual_import(name, globals, locals, fromlist, level)

    isolated = ModuleType('spacr._journal_without_fork_hooks')
    isolated.__package__ = 'spacr'
    isolated.__file__ = journal.__file__
    isolated.__dict__['__builtins__'] = {**vars(builtins), '__import__': portable_import}
    monkeypatch.setitem(sys.modules, isolated.__name__, isolated)
    monkeypatch.setattr(logging_util, '_spacr_home', lambda: tmp_path)
    original_registry = set(journal._ACTIVE_RUN_DIRS)
    original_fork_hook = getattr(os, 'register_at_fork', None)
    original_listeners = list(errors._FINALIZE_LISTENERS)

    exec(compile(Path(journal.__file__).read_text(), journal.__file__, 'exec'), isolated.__dict__)

    assert isolated.current_run() is None
    assert isolated._ACTIVE_RUNS_LOCK is not journal._ACTIVE_RUNS_LOCK
    with isolated._ACTIVE_RUNS_LOCK:
        isolated._ACTIVE_RUN_DIRS.add(str(tmp_path / 'active-run'))
    assert str(tmp_path / 'active-run') in isolated._protected_storage()
    assert journal._ACTIVE_RUN_DIRS == original_registry
    assert getattr(os, 'register_at_fork', None) is original_fork_hook
    assert private_errors._FINALIZE_LISTENERS == [isolated._note_ledger_on_the_open_run]
    assert errors._FINALIZE_LISTENERS == original_listeners
    assert list(tmp_path.iterdir()) == []


def test_unknown_cache_key_cannot_move_files_or_write_a_location_record(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    locations = home / 'cache_locations.json'
    locations.write_text('{"TORCH_HOME":"/previous/cache"}\n')
    original = locations.read_bytes()
    target = tmp_path / 'destination'
    monkeypatch.setattr(journal, '_spacr_home', lambda: home)
    monkeypatch.setattr(journal, '_cache_folder',
                        lambda _key: pytest.fail('unknown cache resolved a folder'))

    with pytest.raises(ValueError, match='no setting to move it with'):
        journal._relocate_cache('not-a-cache', target)

    assert not target.exists()
    assert locations.read_bytes() == original
