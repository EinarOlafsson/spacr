"""The dictionary's honest answers, and its seatbelts around a dying window.

Two groups. The renderer must not invent a definition it does not have and
must not print a row with nothing in it -- an empty labelled row reads as
"this feature has no unit", which is a claim rather than a silence. And the
application-wide event filter must survive being handed an object that is
already being destroyed: it runs for every event in the process, so an
exception there is not a lost menu, it is a lost window.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QTimer  # noqa: E402
from PySide6.QtWidgets import QMenu  # noqa: E402

from spacr.qt.widgets import feature_dictionary as fd  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def menus(qapp):
    """Record context menus instead of opening one, and leave nothing behind.

    Autouse for the same reason the Qt suite's copy is: the default runner is
    ``QMenu.exec``, which in a headless run has nobody to click it. The one
    test that drives the real runner installs its own way out first.
    """
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    yield shown
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


# ---------------------------------------------------------------------------
# is this a measurements table at all
# ---------------------------------------------------------------------------

def test_a_view_with_no_model_is_not_a_measurements_table(qapp):
    """A view whose model is gone must answer no, not raise.

    The filter reaches this while deciding whether an unrecognised column is
    worth a menu item; the view it was handed can be mid-teardown.
    """
    assert fd._table_looks_measured(None) is False


# ---------------------------------------------------------------------------
# the default menu runner
# ---------------------------------------------------------------------------

def test_the_default_runner_opens_a_live_menu_that_can_be_clicked(qapp):
    """The shipped runner has to show a real, interactive menu.

    Every other test in the suite replaces it with a recorder, so nothing
    otherwise proves the production path opens a menu whose action can fire.
    The scheduled trigger stands in for the user's click.
    """
    menu = QMenu()
    action = menu.addAction(fd.CONTEXT_ACTION_TEXT)
    fired: list = []
    action.triggered.connect(lambda checked=False: fired.append(True))

    def click_it():
        action.trigger()
        menu.close()

    QTimer.singleShot(0, click_it)
    fd._default_menu_runner(menu, QPoint(10, 10))

    assert fired == [True]
    assert menu.isVisible() is False


# ---------------------------------------------------------------------------
# liveness checks around a wrapper that is going away
# ---------------------------------------------------------------------------

def test_without_shiboken_every_object_is_treated_as_live(monkeypatch):
    """No shiboken means no way to ask, and refusing everything loses the feature.

    ``_still_alive`` gates the whole context-menu filter. Answering "dead" for
    a perfectly good widget would silently switch the feature off on any
    install where the import is unavailable.
    """
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "shiboken6" or name.startswith("shiboken6."):
            raise ImportError("No module named 'shiboken6'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)

    assert fd._still_alive(object()) is True
    # None is still nothing, with or without shiboken.
    assert fd._still_alive(None) is False


def test_a_liveness_check_that_itself_fails_treats_the_object_as_live(monkeypatch):
    """``isValid`` raising is not evidence the object is dead.

    Same rule as the missing import: the check is a seatbelt, and a seatbelt
    that fastens itself shut is worse than none.
    """
    import shiboken6

    def explode(_wrapped):
        raise RuntimeError("shiboken is unhappy")

    monkeypatch.setattr(shiboken6, "isValid", explode)

    assert fd._still_alive(object()) is True
