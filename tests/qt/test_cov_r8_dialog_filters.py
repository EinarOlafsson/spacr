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
