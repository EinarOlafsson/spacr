"""Backend resolution and the sentences a greyed-out entry has to carry.

Every branch here decides what a user reads on a control they cannot press, so
the wrong branch is a user who installs a package that changes nothing, or who
never learns a package would have helped. The awkward combinations -- a wired-up
backend that is not installed, an installed backend with no pip command, a probe
that raises -- are the ones no real environment happens to produce, so they are
produced here.
"""
from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from spacr import regression_backends as backends
from spacr.regression_backends import (DEFAULT_REGRESSION_BACKEND,
                                       REGRESSION_BACKENDS, backend_status,
                                       backend_install_offer,
                                       cuda_present_without_importing_torch,
                                       package_installed,
                                       resolve_backend_name)


def test_a_blank_backend_setting_means_the_default():
    """Whitespace in the settings CSV resolves to the default, not an error.

    A settings sheet with an empty backend cell is the commonest way this is
    reached; refusing it would stop a run over a cell nobody meant to fill.
    """
    assert resolve_backend_name('   ') == DEFAULT_REGRESSION_BACKEND
    assert resolve_backend_name('') == DEFAULT_REGRESSION_BACKEND


def test_a_backend_label_typed_in_any_case_still_resolves():
    """Case-insensitive matching covers the label as well as the key.

    The label is what the dropdown shows, so it is what gets copied into a
    settings file by hand -- usually with the capitalisation changed.
    """
    assert resolve_backend_name('STATSMODELS (CPU)') == 'statsmodels'
    assert resolve_backend_name('PyFixest') == 'pyfixest'
    assert resolve_backend_name('  Glum (CPU)  ') == 'glum'


def test_a_probe_that_raises_reports_the_package_as_absent(monkeypatch):
    """``find_spec`` blowing up is answered "not installed", not re-raised.

    A broken entry on sys.path makes find_spec raise; a settings panel that
    crashed while drawing a dropdown row would be unusable for every backend,
    not just the broken one.
    """
    def boom(name):
        raise ValueError('bad path entry')

    monkeypatch.setattr(importlib.util, 'find_spec', boom)
    assert package_installed('pyfixest') is False
    assert package_installed('') is True


def test_a_torch_whose_cuda_probe_raises_counts_as_no_device(monkeypatch):
    """An already-imported torch that cannot answer is treated as CPU-only.

    torch raises here when the driver and the runtime disagree. The point of
    this probe is to decide a dropdown row without paying for an import, so it
    must not turn a broken CUDA install into a crash in the settings panel.
    """
    fake_cuda = types.SimpleNamespace()

    def boom():
        raise RuntimeError('CUDA driver version is insufficient')

    fake_cuda.is_available = boom
    monkeypatch.setitem(sys.modules, 'torch',
                        types.SimpleNamespace(cuda=fake_cuda))
    assert cuda_present_without_importing_torch() is False


def test_with_torch_unimported_the_driver_nodes_are_what_is_consulted(
        monkeypatch):
    """No torch in sys.modules falls back to looking for the device files.

    That fallback is the whole reason this function exists: deciding GPU
    availability must not import torch, which costs seconds on every panel.
    """
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    monkeypatch.setattr(backends.os.path, 'exists', lambda path: False)
    assert cuda_present_without_importing_torch() is False

    monkeypatch.setattr(backends.os.path, 'exists',
                        lambda path: path == '/dev/nvidia0')
    assert cuda_present_without_importing_torch() is True


def test_a_wired_up_backend_that_is_not_installed_says_both_things(
        monkeypatch):
    """With the family unchosen, the row names the types AND the install state.

    Either fact alone leaves the reader stuck: the types say which choice would
    make the row selectable, the install state says whether that choice would be
    enough on this machine.
    """
    monkeypatch.setattr(backends, 'package_installed', lambda name: False)
    status = backend_status('pyfixest', regression_type=None)

    assert status['enabled'] is False
    assert 'not installed' in status['short_reason']
    assert 'not installed' in status['reason']
    assert 'ols' in status['short_reason']


def test_a_backend_with_no_install_command_just_says_it_is_not_installed(
        monkeypatch):
    """A missing package with nothing to pip reads "not installed", not blank.

    A row that names a package the reader cannot install must still say the
    package is absent; an empty clause reads as "installed" to a skimmer.
    """
    spec = dict(REGRESSION_BACKENDS['pyfixest'])
    spec['pip'] = None
    monkeypatch.setitem(REGRESSION_BACKENDS, 'pyfixest', spec)
    monkeypatch.setattr(backends, 'package_installed', lambda name: False)

    status = backend_status('pyfixest', regression_type=None)
    assert status['enabled'] is False
    assert 'This backend is not installed.' in status['reason']
    assert 'None' not in status['reason']


def test_an_installed_but_unwired_backend_is_offered_nothing_to_install(
        monkeypatch):
    """Installing more cannot help a backend spaCR routes no fit through.

    Offering an install command there would send the user to fix their
    environment for a limitation that is entirely spaCR's.
    """
    monkeypatch.setattr(backends, 'cuda_present_without_importing_torch',
                        lambda: True)
    monkeypatch.setattr(backends, 'package_installed', lambda name: True)

    offer = backend_install_offer('gpytorch', regression_type='mixed')
    assert offer.action == 'impossible'
    assert 'routes no fit through it yet' in offer.message


def test_an_unwired_backend_that_IS_installed_does_not_advertise_a_pip_command(
        monkeypatch):
    """"not wired up; installed" is the whole truth, and adding a pip line
    to it would be advice that fixes nothing.

    Four backends are declared and not implemented -- pymer4, cuml, numpyro,
    gpytorch. A reader who already has one of them installed and sees
    "pip install numpyro" beside it will run it, watch pip report the
    requirement is already satisfied, and come back no wiser. The install
    clause is added only when installing would actually change the row.
    """
    monkeypatch.setattr(backends, 'package_installed', lambda name: True)

    status = backend_status('numpyro', regression_type=None)

    assert status['enabled'] is False
    assert status['short_reason'].endswith('not wired up; installed')
    assert 'pip install' not in status['short_reason']


def test_an_unwired_backend_that_is_absent_does_say_how_to_get_it(monkeypatch):
    """The counterpart, so the omission above is about the install state.

    Not wired up AND not installed is two separate obstacles, and the row
    names the one the reader can act on.
    """
    monkeypatch.setattr(backends, 'package_installed', lambda name: False)

    status = backend_status('numpyro', regression_type=None)

    assert 'not wired up' in status['short_reason']
    assert 'not installed' in status['short_reason']
    assert 'pip install numpyro' in status['short_reason']


def test_an_unwired_backend_with_nothing_to_install_says_only_that(monkeypatch):
    """No pip command means no clause, rather than a dangling dash.

    pymer4 is the real case: it needs R, rpy2 and lme4, so `pip install
    pymer4` succeeds and the backend still does not run. A row ending in
    "— None" would be worse than one that stops.
    """
    spec = dict(REGRESSION_BACKENDS['numpyro'])
    spec['pip'] = None
    monkeypatch.setitem(REGRESSION_BACKENDS, 'numpyro', spec)
    monkeypatch.setattr(backends, 'package_installed', lambda name: False)

    status = backend_status('numpyro', regression_type=None)

    assert status['short_reason'].endswith('not wired up')
    assert 'None' not in status['short_reason']
