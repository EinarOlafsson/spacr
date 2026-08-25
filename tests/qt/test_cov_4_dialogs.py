"""Installing the dialog detacher without an application is refused, not
crashed on.

``detach_all_dialogs`` runs during start-up, before the caller can be sure a
QApplication exists. Returning False for "no application" keeps the failure
visible to the caller and, crucially, leaves the module-level filter slot
untouched so a later call with a real application still installs.
"""
from __future__ import annotations

import pytest

from spacr.qt import dialogs


@pytest.fixture(autouse=True)
def _restore_detacher():
    """The detacher is module state shared with every other Qt test."""
    saved = (dialogs._DETACHER, dialogs._DETACHED_APP)
    yield
    dialogs._DETACHER, dialogs._DETACHED_APP = saved


def test_no_application_means_nothing_was_installed():
    """The caller is told the detacher did not go in."""
    assert dialogs.detach_all_dialogs(None) is False


def test_a_refused_install_does_not_consume_the_filter_slot(qapp):
    """A later call with a real application must still be able to install."""
    dialogs._DETACHER = None
    dialogs._DETACHED_APP = None
    assert dialogs.detach_all_dialogs(None) is False
    assert dialogs._DETACHER is None
    assert dialogs._DETACHED_APP is None
    try:
        assert dialogs.detach_all_dialogs(qapp) is True
        assert dialogs._DETACHED_APP is qapp
        # Idempotent per application: the second call is a no-op.
        assert dialogs.detach_all_dialogs(qapp) is False
    finally:
        if dialogs._DETACHER is not None:
            qapp.removeEventFilter(dialogs._DETACHER)
        dialogs._DETACHER = None
        dialogs._DETACHED_APP = None
