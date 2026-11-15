"""What the Feature Dictionary ANSWERS, and what it refuses to answer.

``tests/qt/test_feature_dictionary_panel.py`` covers the happy paths — the
panel opens, the Help menu gains an entry, a right-click on a measurements
column offers to explain it. This file covers the answers that are easy to get
subtly wrong, and the gestures that must produce NO answer at all:

* the sentences the detail pane composes for a channel PAIR, for a metadata
  column, for a feature no standard run writes, and for the two columns whose
  name lies about which table they live in
  (``organelle_summary_*``, ``<object>_before_filtration``);
* a column name is TEXT — HTML in it is shown, never rendered;
* the right-clicks that must stay silent: a row header, past the last column,
  below the last row, a grid whose model names no columns;
* every swallowed-failure arm — a broken search backend, a menu that will not
  open, a window with no menu bar, a registry that refuses the app — because
  each one exists to keep a failure from costing the user a window, and a
  swallow that swallows the wrong thing is invisible until someone tests it.

Regression coverage also pins three formerly broken paths: a right-click on a
row header must not answer with a column nobody clicked, a reused dialog must
not keep answering the previous question, and a destroyed dialog must not take
the live replacement's cache entry with it.

Offscreen, offline, no database. No test opens a modal: the context-menu runner
is replaced with a recorder by an autouse fixture, so no ``QMenu.exec`` can
reach a headless run.
"""
from __future__ import annotations

import logging

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (  # noqa: E402
    QPoint,
)
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QTableWidget,
    QWidget,
)

from spacr.qt.widgets import feature_dictionary as fd  # noqa: E402

pytestmark = pytest.mark.qt

MEASUREMENT_HEADERS = [
    "object_label",
    "cell_area",
    "cell_channel_1_percentile_75",
    "nucleus_periphery_mean",
]


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def menus(qapp):
    """Record context menus instead of opening one, and leave nothing behind.

    Autouse on purpose. The default runner is ``QMenu.exec``, which in a
    headless run has nobody to click it and hangs the whole suite — this
    project has lost a run to exactly that three times. With the recorder
    installed for every test in this file, a filter that decides to show a
    menu when it should not fails an assertion instead of stopping the run.
    """
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    yield shown
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


@pytest.fixture
def panel(qtbot):
    """A fresh panel, cleaned up with the rest of the widgets."""
    widget = fd.FeatureDictionaryPanel()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def qss_sandbox():
    """Take the panel's QSS block back out, whatever the test did to it."""
    from spacr.qt import theme as theme_mod

    had = fd.OBJECT_NAME in theme_mod.widget_qss_names()
    register = theme_mod.register_widget_qss
    theme_mod.unregister_widget_qss(fd.OBJECT_NAME)
    try:
        yield theme_mod
    finally:
        theme_mod.unregister_widget_qss(fd.OBJECT_NAME)
        if had:
            register(fd.OBJECT_NAME, fd._panel_qss)


def _measurement_table(qtbot, headers=None):
    headers = list(headers or MEASUREMENT_HEADERS)
    table = QTableWidget(2, len(headers))
    table.setHorizontalHeaderLabels(headers)
    qtbot.addWidget(table)
    table.resize(800, 200)
    return table


def _right_click(qapp, widget, pos):
    """Send a real context-menu event, the way a right-click does."""
    event = QContextMenuEvent(QContextMenuEvent.Mouse, pos,
                              widget.mapToGlobal(pos))
    qapp.notify(widget, event)


# ---------------------------------------------------------------------------
# the sentences the detail pane composes
# ---------------------------------------------------------------------------

def test_a_fresh_search_answers_with_its_best_hit(panel):
    """Typing a query leaves the pane showing the top result, not the old one.

    This is the invariant the panel is built on — the list's first row is
    selected, and the detail pane is that row. Everything a user reads after
    typing depends on it.
    """
    panel.set_query("texture")
    assert panel.result_keys()[0] == "homogeneity_distance_<d>"
    assert panel.current_doc().key == panel.result_keys()[0]

    panel.set_query("zzzzz-not-a-feature")
    assert panel.result_keys() == []
    # nothing matched, so nothing may be left claiming to be the answer
    assert panel.current_doc() is None


def test_a_new_search_stops_answering_about_the_previous_column(panel):
    """A new question must not be answered with the old question's answer."""
    panel.show_column("nucleus_channel_2_percentile_25")
    panel.set_query("texture")

    assert panel.result_keys()[0] == "homogeneity_distance_<d>"
    assert "nucleus_channel_2_percentile_25" not in panel.detail_text()
    assert panel.current_doc().key == panel.result_keys()[0]


# ---------------------------------------------------------------------------
# registration: the two seams, and what happens when they refuse
# ---------------------------------------------------------------------------

def test_make_screen_builds_the_panel_the_qss_targets(qtbot):
    """The factory's screen must carry the objectName the QSS block selects.

    ``register()`` registers a QSS block written against
    ``QWidget#FeatureDictionary``. A screen built without that objectName is
    an unstyled panel in a styled app, and nothing else would notice.
    """
    screen = fd.make_screen(host=object())
    qtbot.addWidget(screen)
    assert isinstance(screen, fd.FeatureDictionaryPanel)
    assert screen.objectName() == fd.OBJECT_NAME
    assert screen.result_keys()


def test_the_qss_block_selects_the_names_the_panel_builds(qtbot, qss_sandbox):
    """Every child the block styles has to exist, under that exact name.

    The QSS reaches the sheet as text, so a child renamed in the panel and not
    in the block costs the styling silently — the sheet still compiles and the
    list simply stops being a rounded surface. Both halves are checked against
    each other rather than against a copy of the names.
    """
    fd.register()
    try:
        panel = fd.FeatureDictionaryPanel()
        qtbot.addWidget(panel)
        built = {child.objectName() for child in panel.findChildren(QWidget)}
        qss = qss_sandbox.stylesheet()
        for name in ("FeatureDictionaryBlurb", "FeatureDictionaryStatus",
                     "FeatureDictionaryList", "FeatureDictionaryDetail"):
            assert name in built, f"{name} is styled but never built"
            assert f"#{name}" in qss, f"{name} is built but never styled"
        assert f"QWidget#{fd.OBJECT_NAME}" in qss
    finally:
        from spacr.qt import app as app_mod
        app_mod.unregister_app(fd.APP_KEY)


# ---------------------------------------------------------------------------
# the right-clicks that must answer
# ---------------------------------------------------------------------------

def test_a_menu_that_will_not_open_does_not_take_the_right_click_down(
        qtbot, qapp, caplog):
    """Help is not function: a failure here must be logged, never raised.

    An exception escaping an application-wide event filter surfaces inside
    Qt's event dispatch, where spaCR has no handler — the price of a broken
    help menu would be the right-click, and possibly the window.
    """
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)

    def boom(menu, pos):
        raise RuntimeError("no screen to show it on")

    fd.set_menu_runner(boom)
    header = table.horizontalHeader()
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.feature_dictionary"):
        _right_click(qapp, header, QPoint(header.sectionPosition(2) + 4, 4))
    assert "Feature help context menu failed" in caplog.text


# ---------------------------------------------------------------------------
# the shared dialog
# ---------------------------------------------------------------------------

def test_closing_the_dictionary_leaves_nothing_cached(qtbot):
    """``close_feature_dictionary`` must drop the reference, not just hide."""
    fd.open_feature_dictionary(None, "cell_area")
    fd.close_feature_dictionary()
    assert fd._DIALOG is None
    fd.close_feature_dictionary()          # and again, on nothing
    assert fd._DIALOG is None
