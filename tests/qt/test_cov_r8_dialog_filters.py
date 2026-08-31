"""`install_the_dialog_filters` -- three installers, isolated from each other.

The function had no test of any kind, and the promise in its docstring
is the interesting part: "a failure in one filter does not prevent the
remaining filters from being installed". That is the branch the measured
gap list pointed at, and it is the one that matters -- these run during
launch, and one optional decoration that cannot install must not take
the dialogs, the glass or the translation layer down with it.

The return value is the evidence: it names the installers that actually
ran, so a caller (and a test) can tell a silent skip from a success.
"""
from __future__ import annotations

import pytest

from spacr.qt import app as app_module

pytestmark = pytest.mark.qt


def test_every_registered_installer_runs_and_is_named(qapp, monkeypatch):
    called = []

    class _Fake:
        def __init__(self, name):
            self.name = name

        def __call__(self, app):
            called.append((self.name, app))

    import sys
    import types

    module = types.ModuleType("spacr_fake_filters")
    module.one = _Fake("one")
    module.two = _Fake("two")
    monkeypatch.setitem(sys.modules, "spacr_fake_filters", module)
    monkeypatch.setattr(app_module, "_DIALOG_FILTERS", (
        ("spacr_fake_filters", "one"),
        ("spacr_fake_filters", "two"),
    ))

    installed = app_module.install_the_dialog_filters(qapp)
    assert installed == ("one", "two")
    assert [name for name, _ in called] == ["one", "two"]
    assert all(a is qapp for _, a in called), "the app was not passed through"


def test_one_installer_that_raises_does_not_stop_the_others(qapp,
                                                            monkeypatch,
                                                            caplog):
    """THE ISOLATION THE DOCSTRING PROMISES.

    These run during launch. A decoration that cannot install must not
    cost the dialogs, the glass or the translation layer.
    """
    import sys
    import types

    ran = []
    module = types.ModuleType("spacr_fake_filters")
    module.good = lambda app: ran.append("good")

    def bad(_app):
        raise RuntimeError("this filter cannot install here")

    module.bad = bad
    module.also_good = lambda app: ran.append("also_good")
    monkeypatch.setitem(sys.modules, "spacr_fake_filters", module)
    monkeypatch.setattr(app_module, "_DIALOG_FILTERS", (
        ("spacr_fake_filters", "good"),
        ("spacr_fake_filters", "bad"),
        ("spacr_fake_filters", "also_good"),
    ))

    with caplog.at_level("DEBUG"):
        installed = app_module.install_the_dialog_filters(qapp)

    assert ran == ["good", "also_good"], "a later installer was skipped"
    assert installed == ("good", "also_good"), (
        "the failed installer was reported as installed")
    assert "bad" in caplog.text


def test_a_module_that_will_not_import_is_survived(qapp, monkeypatch,
                                                   caplog):
    """The import is inside the same guard as the call.

    A filter whose module is missing entirely -- an optional extra that
    was not installed -- is the ordinary way this fails in the wild.
    """
    monkeypatch.setattr(app_module, "_DIALOG_FILTERS", (
        ("spacr.this_module_does_not_exist", "whatever"),
    ))
    with caplog.at_level("DEBUG"):
        assert app_module.install_the_dialog_filters(qapp) == ()


def test_a_missing_function_in_a_real_module_is_survived(qapp, monkeypatch):
    """`getattr` is inside the guard too, not only the import."""
    monkeypatch.setattr(app_module, "_DIALOG_FILTERS", (
        ("spacr.qt.dialogs", "no_such_installer"),
    ))
    assert app_module.install_the_dialog_filters(qapp) == ()


def test_an_empty_registry_installs_nothing_and_says_so(qapp, monkeypatch):
    monkeypatch.setattr(app_module, "_DIALOG_FILTERS", ())
    assert app_module.install_the_dialog_filters(qapp) == ()


def test_the_real_registry_names_three_installers_that_exist():
    """The registry itself, so a rename cannot silently empty it.

    Every entry is resolved without being CALLED -- calling them installs
    real filters on the live QApplication for the rest of the session.
    """
    import importlib

    assert app_module._DIALOG_FILTERS, "the registry is empty"
    for module_name, function_name in app_module._DIALOG_FILTERS:
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name)), (
            f"{module_name}.{function_name} is not callable")
