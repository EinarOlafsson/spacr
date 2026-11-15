"""Quieting launch noise must never cost a warning that mattered.

``spacr.qt`` installs a Qt message handler and a warnings filter at launch.
Both are deliberately narrow, and both have to survive the environments they
run in: PySide6 may be absent (a headless ``spacr-run`` importing the
package for its constants), and the logging subsystem may be unusable at the
moment a Qt warning arrives. In every one of those cases the requirement is
the same -- the line still reaches stderr, and installing the quieter never
raises.

The warnings filter carries a second requirement: it is re-asserted on every
launch, and ``warnings.filterwarnings`` prepends unconditionally, so a
second call must find its own rule and do nothing rather than grow
``warnings.filters`` without bound.
"""

from __future__ import annotations

import logging
import warnings

import pytest

pytest.importorskip("PySide6")

import spacr.qt as qtpkg                      # noqa: E402


def test_re_asserting_the_library_filter_does_not_grow_the_filter_list(
        monkeypatch):
    """Called once per launch, so a second call has to be a no-op."""
    monkeypatch.setattr(warnings, "filters", [])

    qtpkg._quiet_library_warnings()
    after_first = list(warnings.filters)
    assert len(after_first) == len(qtpkg._LIBRARY_NOISE)

    qtpkg._quiet_library_warnings()

    assert warnings.filters == after_first


def test_nonpositive_inotify_limit_prints_dynamic_recovery_not_a_bad_limit(
        monkeypatch, capsys):
    from pathlib import Path

    monkeypatch.setattr(qtpkg.sys, 'platform', 'linux')
    monkeypatch.setattr(qtpkg, '_SAID_IT_ONCE', False)
    monkeypatch.setattr(Path, 'read_text', lambda self, **kwargs: '0')
    qtpkg._explain_the_inotify_line()
    first = capsys.readouterr().err
    assert 'sysctl -n fs.inotify.max_user_watches' in first
    assert 'spacr_watch_limit * 2' in first
    assert 'max_user_watches=0' not in first
    qtpkg._explain_the_inotify_line()
    assert capsys.readouterr().err == ''


@pytest.mark.parametrize('broken_spec', [False, True])
def test_qt_dependency_directories_keep_only_real_xcb_bundles(
        tmp_path, monkeypatch, broken_spec):
    import importlib.util
    from types import SimpleNamespace

    bundle = tmp_path / 'pyside'
    lib = bundle / 'Qt/lib'
    lib.mkdir(parents=True)
    (lib / 'libQt6XcbQpa.so.6').touch()
    prefix = tmp_path / 'prefix'
    (prefix / 'lib').mkdir(parents=True)
    (prefix / 'lib/libQt6XcbQpa.so.6').touch()

    def find(name):
        assert name == 'PySide6'
        if broken_spec:
            raise ValueError('module has no spec')
        return SimpleNamespace(submodule_search_locations=[str(bundle), str(tmp_path / 'absent')])

    monkeypatch.setattr(importlib.util, 'find_spec', find)
    monkeypatch.setattr(qtpkg.sys, 'prefix', str(prefix))
    assert qtpkg._qt_library_dirs() == ([str(lib)] if not broken_spec else []) + [str(prefix / 'lib')]


def test_loader_cache_retries_bad_candidates_and_keeps_sonames(monkeypatch):
    import os
    import shutil
    import subprocess
    from types import SimpleNamespace

    seen = []
    monkeypatch.setattr(os.path, 'isfile', lambda path: path in ('/sbin/ldconfig', '/usr/sbin/ldconfig'))
    monkeypatch.setattr(shutil, 'which', lambda name: None)

    def listing(argv, **kwargs):
        seen.append(argv)
        assert kwargs == dict(capture_output=True, text=True, timeout=10, check=False)
        if len(seen) == 1:
            raise OSError('loader cache unavailable')
        return SimpleNamespace(stdout='header\nlibxcb.so.1 (libc6) => /lib/libxcb.so.1\n')

    monkeypatch.setattr(subprocess, 'run', listing)
    assert qtpkg._loader_cache() == frozenset({'libxcb.so.1'})
    assert seen == [['/sbin/ldconfig', '-p'], ['/usr/sbin/ldconfig', '-p']]
    monkeypatch.setattr(subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout='no libraries'))
    assert qtpkg._loader_cache() is None


def test_failed_x11_relaunch_restores_existing_search_path(monkeypatch):
    import os

    env = {'LD_LIBRARY_PATH': '/prior'}
    calls = []
    monkeypatch.setattr(qtpkg, '_conda_library_dirs', lambda environ: ['/owned/lib'])
    monkeypatch.setattr(os.path, 'isfile', lambda path: path == '/owned/lib/libxcb.so')

    def denied(executable, argv):
        calls.append((executable, list(argv), dict(env)))
        raise OSError('exec refused')

    assert not qtpkg._relaunch_with_conda_x11_libraries([('libxcb.so',)], env, execv=denied)
    assert env == {'LD_LIBRARY_PATH': '/prior'}
    assert len(calls) == 1
    assert calls[0][2]['LD_LIBRARY_PATH'] == '/owned/lib:/prior'
    assert calls[0][2][qtpkg._X11_RELAUNCH_FLAG] == '1'


def test_x11_probe_failure_does_not_abort_startup(monkeypatch):
    attempted = []

    def unavailable(environ):
        raise RuntimeError('dependency probe unavailable')

    monkeypatch.setattr(qtpkg, '_missing_x11_libraries', unavailable)
    assert qtpkg._check_the_x11_libraries({'QT_QPA_PLATFORM': 'xcb'}, linux=True, execv=lambda *args: attempted.append(args)) == 0
    assert attempted == []


def test_complete_x11_dependencies_start_without_relaunch_or_warning(
        monkeypatch, caplog, capsys):
    examined, relaunched = [], []
    env = {'QT_QPA_PLATFORM': 'xcb', 'DISPLAY': ':owned'}
    monkeypatch.setattr(qtpkg, '_missing_x11_libraries', lambda current: examined.append(current) or [])
    monkeypatch.setattr(qtpkg, '_relaunch_with_conda_x11_libraries', lambda *args, **kwargs: relaunched.append(args))
    with caplog.at_level(logging.INFO, logger='spacr.qt'):
        assert qtpkg._check_the_x11_libraries(env, linux=True) == 0
    assert examined == [env]
    assert relaunched == []
    assert 'known environment problem' not in caplog.text
    assert capsys.readouterr().err == ''
    assert env == {'QT_QPA_PLATFORM': 'xcb', 'DISPLAY': ':owned'}
