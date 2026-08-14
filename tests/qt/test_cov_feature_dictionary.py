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
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    Qt,
)
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QMainWindow,
    QTableView,
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


def _texts(menu) -> list[str]:
    return [action.text() for action in menu.actions()]


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


class _HeaderModel(QAbstractTableModel):
    """A table model whose column names are exactly what it is handed.

    ``None`` for a column means the model does not name that column at all —
    which neither ``QTableWidget`` nor ``QStandardItemModel`` can express (both
    fall back to the section number), and which the dictionary has to survive.
    """

    def __init__(self, headers):
        super().__init__()
        self._headers = list(headers)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return "1"
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self._headers[section]


def _model_table(qtbot, headers):
    table = QTableView()
    model = _HeaderModel(headers)
    table.setModel(model)
    table._keep_model = model          # the view does not own it
    qtbot.addWidget(table)
    table.resize(800, 200)
    return table


class _TableWithNoHeaderAccessor(QTableWidget):
    """A view that refuses to hand its header over from Python.

    Stands in for a part-destructed view, which is what the ``try`` in
    ``_menu_family`` is there for: the right-click still has to work.
    """

    def horizontalHeader(self):
        raise RuntimeError("horizontalHeader is gone")


# ---------------------------------------------------------------------------
# the sentences the detail pane composes
# ---------------------------------------------------------------------------

def test_a_colocalisation_column_names_both_of_its_channels(panel):
    """A pair column is about TWO channels, and must say which two.

    ``cell_channel_0_channel_2_M1_correlation_85`` is the one shape of column
    where reading the generic feature is not enough: M1 is asymmetric, so
    "channels 0 and 2" is load-bearing — swap them and the number means a
    different thing. The pane must resolve the pair from the column name AND
    say that this feature exists once per channel pair.
    """
    panel.show_column("cell_channel_0_channel_2_M1_correlation_85")
    text = panel.detail_text()
    assert panel.current_doc().key == "M1_correlation_<t>"
    assert "This column is channels 0 and 2." in text
    assert "One column per channel PAIR" in text
    # the concrete threshold out of the name, not the template
    assert "percentile 85" in text
    assert "fraction in [0, 1]" in text


def test_a_metadata_column_is_not_reported_as_a_measurement(panel):
    """``object_label`` is a key, not a number to model on.

    The pane has to say so in both places a user reads: the Objects line
    ("not a per-object measurement") and the family gloss that warns against
    feeding identifiers to a model. It must also NOT claim the column lives in
    "the <object> table" — metadata is written to every table.
    """
    panel.show_column("object_label")
    text = panel.detail_text()
    assert panel.current_doc().key == "object_label"
    assert "Not a per-object measurement." in text
    assert "never feed these to a model as features" in text
    assert "none (an identifier)" in text
    # "This column" is the label of the object-table line, and of nothing
    # else on a column with no channel.
    assert "This column" not in text


def test_a_feature_no_standard_run_writes_says_so(panel):
    """Empty ``object_types`` must read as "nobody writes this", not as blank.

    ``skeleton_length`` only appears when the optional cytoskeleton analysis
    ran. Rendering an empty object list as an empty string would read as "this
    exists everywhere"; the pane has to point at the note instead.
    """
    panel.show_column("cell_skeleton_length")
    text = panel.detail_text()
    assert panel.current_doc().key == "skeleton_length"
    assert "Not written for any object type by a standard run" in text


def test_an_organelle_summary_column_does_not_claim_the_organelle_table(panel):
    """The name says organelle; the column is NOT in the organelle table.

    ``organelle_summary_organelle_count`` is one row per PARENT object and
    lives in ``<parent>_organelle_summary``. Telling a user it is in the
    organelle table sends them to write a query that returns nothing, so the
    "This column — the <object> table" line has to stay off this one and the
    note has to be the answer instead.
    """
    panel.show_column("organelle_summary_organelle_count")
    text = panel.detail_text()
    assert panel.current_doc().key == "organelle_summary_organelle_count"
    # the object-table line is the one that must not be composed
    assert "This column" not in text
    assert "NOT in the organelle table" in text
    assert "organelle_summary tables" in text


def test_a_pivoted_count_column_does_not_claim_the_cell_table(panel):
    """``cell_before_filtration`` is per FIELD, in ``pivoted_counts``.

    Same trap as above from the other direction: this one carries a real
    object prefix, so the object-table line would be composed confidently and
    would be wrong.
    """
    panel.show_column("cell_before_filtration")
    text = panel.detail_text()
    assert panel.current_doc().key == "before_filtration"
    assert "This column" not in text
    assert "pivoted_counts table only" in text


def test_html_in_a_column_name_is_shown_not_rendered(panel):
    """A column name is data. It reaches a QTextBrowser and must be escaped.

    Column names come out of whatever CSV or database the user loaded, so a
    name containing markup is possible and must be displayed literally. If the
    escape were dropped the tags would be interpreted and the plain text would
    read ``cell_evil_area`` — the user would be shown a name that is not the
    one in their table.
    """
    panel.show_column("cell_<b>evil</b>_area")
    assert "cell_<b>evil</b>_area" in panel.detail_text()
    assert panel.current_doc() is None


def test_opening_the_panel_on_a_column_answers_that_column(qtbot):
    """``FeatureDictionaryPanel(column=...)`` must not need a second call.

    The screen factory and the dialog both hand the column in at
    construction; if the constructor only ran the plain search, the panel
    would open on the whole dictionary and the user's question would be lost.
    """
    widget = fd.FeatureDictionaryPanel(
        column="nucleus_channel_2_percentile_25")
    qtbot.addWidget(widget)
    assert widget.current_doc().key == "percentile_<p>"
    text = widget.detail_text()
    assert "nucleus_channel_2_percentile_25" in text
    assert "This column is channel 2." in text


def test_picking_another_feature_stops_answering_about_the_column(panel):
    """Moving off the column must drop the column's channel and table.

    The pane is pinned to a concrete column after a "What is this?", and the
    pin is what makes it say "channel 2" and "the nucleus table". Click a
    different feature and those two facts are no longer true of anything on
    screen — carrying them over would attribute the previous column's channel
    to a feature the user is only browsing.
    """
    panel.show_column("nucleus_channel_2_percentile_25")
    assert "This column is channel 2." in panel.detail_text()

    other = panel.result_keys().index("std_intensity")
    panel._list.setCurrentRow(other)

    text = panel.detail_text()
    assert panel.current_doc().key == "std_intensity"
    assert "nucleus_channel_2_percentile_25" not in text
    assert "This column is channel 2." not in text
    assert "the nucleus table" not in text

    # Qt clears the current row whenever the list is repopulated; the pane
    # keeps the last thing it was told to show rather than blanking.
    panel._list.setCurrentRow(-1)
    assert panel.current_doc().key == "std_intensity"
    assert panel.detail_text() == text


def test_a_column_lookup_survives_a_broken_search_backend(panel, monkeypatch):
    """A search that raises must not cost the user the answer they asked for.

    ``_refresh`` swallows the failure so the panel stays usable, and the
    column lookup then has NO list to select from — the fallback that resolves
    the key directly is the only thing that still answers the question. If it
    were dropped, a right-click would open an empty dictionary.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(fd, "search_features", boom)
    panel.show_column("cell_area")

    assert panel.result_keys() == []
    assert panel.current_doc().key == "area"
    # The generic feature, not the pinned column: the fallback renders the
    # doc without the parsed entry.
    assert "Area" in panel.detail_text()


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


def test_register_reports_failure_when_the_registry_refuses(monkeypatch,
                                                            qss_sandbox):
    """A registry that will not take the app must not stop the GUI starting.

    ``register()`` runs from ``spacr.qt.run`` before the main window exists.
    An exception escaping it would take the whole launch down for a panel that
    is also reachable from the Help menu — so the failure is reported as
    ``False``, not raised, and the QSS half still registers.
    """
    from spacr.qt import app as app_mod

    def boom(*args, **kwargs):
        raise RuntimeError("registry full")

    monkeypatch.setattr(app_mod, "register_app", boom)
    assert fd.register() is False
    assert not any(row[0] == fd.APP_KEY for row in app_mod.APPS)
    # the second seam is independent and still ran
    assert fd.OBJECT_NAME in qss_sandbox.widget_qss_names()


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


def test_register_succeeds_even_when_the_theme_refuses(monkeypatch,
                                                       qss_sandbox):
    """A QSS block is decoration; losing it must not fail the registration.

    ``register()``'s return value is "is the app row there?" — answering
    ``False`` because a stylesheet block failed would make the caller think
    the app is missing when it is present and usable.
    """
    from spacr.qt import app as app_mod

    def boom(*args, **kwargs):
        raise RuntimeError("no theme")

    monkeypatch.setattr(qss_sandbox, "register_widget_qss", boom)
    try:
        assert fd.register() is True
        assert any(row[0] == fd.APP_KEY for row in app_mod.APPS)
        assert fd.OBJECT_NAME not in qss_sandbox.widget_qss_names()
    finally:
        app_mod.unregister_app(fd.APP_KEY)


# ---------------------------------------------------------------------------
# the Help menu
# ---------------------------------------------------------------------------

def test_the_help_entry_goes_above_the_separator(qtbot):
    """Placement is the point: with the explanations, not with the tools.

    A plain ``addAction`` would drop it at the bottom, under "Check for
    updates…" — the tools half of the menu. The entry has to land before the
    first separator.
    """
    window = QMainWindow()
    qtbot.addWidget(window)
    menu = window.menuBar().addMenu("&Help")
    menu.addAction("About")
    menu.addSeparator()
    menu.addAction("Check for updates…")

    action = fd.install_help_action(window)
    assert action is not None
    texts = [a.text() for a in menu.actions()]
    assert texts.index(fd.HELP_ACTION_TEXT) == 1
    assert menu.actions()[2].isSeparator()


def test_a_window_with_no_menu_bar_still_gets_the_context_menu_route(qtbot,
                                                                     qapp):
    """The two hooks are independent, and the second is the one that scales.

    ``install_window_hooks`` is called on whatever window a launch produced.
    A window with no menu bar at all (a bare widget in a test, a stripped
    window) must not lose the right-click route as well — that route is what
    reaches the eleven table screens.
    """
    plain = QWidget()
    qtbot.addWidget(plain)
    fd.install_window_hooks(plain)
    assert fd._FILTER is not None


def test_a_window_with_nowhere_to_put_the_entry_is_left_alone(qtbot):
    """Two shapes of "no Help menu here", both no-ops rather than errors.

    A window whose ``menuBar()`` answers None (it has not been built yet) and
    a window with a menu bar that has no Help menu. Neither may raise: this
    runs during ``MainWindow.__init__``.
    """
    class _NoBar(QMainWindow):
        def menuBar(self):
            return None

    window = _NoBar()
    qtbot.addWidget(window)
    assert fd.install_help_action(window) is None

    other = QMainWindow()
    qtbot.addWidget(other)
    other.menuBar().addMenu("&File").addAction("Open")
    assert fd.install_help_action(other) is None


def test_a_menu_that_cannot_be_read_does_not_hide_the_help_menu(qtbot):
    """One unreadable menu must cost that menu, not the Help entry.

    ``_find_menu`` walks the menu bar's children, and a QMenu wrapper whose
    C++ object has gone raises RuntimeError on ``title()`` — the failure this
    module was written around. Skipping it and carrying on is what keeps the
    Help entry installable; letting it out would take the whole window down.
    """
    from PySide6.QtWidgets import QMenu

    class _Unreadable(QMenu):
        def title(self):
            raise RuntimeError(
                "Internal C++ object (PySide6.QtWidgets.QMenu) already "
                "deleted")

    window = QMainWindow()
    qtbot.addWidget(window)
    bar = window.menuBar()
    bar.addMenu(_Unreadable("&Tools", bar))
    help_menu = bar.addMenu("&Help")
    help_menu.addAction("About")

    assert fd.install_help_action(window) is not None
    assert fd.HELP_ACTION_TEXT in [a.text() for a in help_menu.actions()]


def test_installing_the_hooks_twice_leaves_one_help_entry(qtbot, qapp):
    """``shortcuts.install`` may run again on a window it already wired.

    A second entry in the Help menu is the visible symptom; the invisible one
    is a second lambda holding the window alive.
    """
    window = QMainWindow()
    qtbot.addWidget(window)
    menu = window.menuBar().addMenu("&Help")
    menu.addAction("About")

    fd.install_window_hooks(window)
    fd.install_window_hooks(window)
    texts = [a.text() for a in menu.actions()]
    assert texts.count(fd.HELP_ACTION_TEXT) == 1


def test_a_broken_help_hook_does_not_cost_the_context_menu(monkeypatch, qtbot):
    """One hook failing must not skip the other.

    Both live in ``install_window_hooks`` under separate ``try`` blocks for
    this reason; merging them would mean a Help menu that cannot take the
    action silently removes "What is this?" from every table in the app.
    """
    def boom(*args, **kwargs):
        raise RuntimeError("no help menu")

    monkeypatch.setattr(fd, "install_help_action", boom)
    window = QMainWindow()
    qtbot.addWidget(window)
    fd.install_window_hooks(window)
    assert fd._FILTER is not None


def test_a_broken_filter_hook_does_not_cost_the_help_entry(monkeypatch, qtbot):
    """And the same the other way round."""
    def boom(*args, **kwargs):
        raise RuntimeError("no application")

    monkeypatch.setattr(fd, "install_context_menu_filter", boom)
    window = QMainWindow()
    qtbot.addWidget(window)
    menu = window.menuBar().addMenu("&Help")
    menu.addAction("About")
    fd.install_window_hooks(window)
    assert fd.HELP_ACTION_TEXT in [a.text() for a in menu.actions()]


# ---------------------------------------------------------------------------
# the right-clicks that must stay silent
# ---------------------------------------------------------------------------

def test_right_clicking_a_row_header_offers_nothing(qtbot, qapp, menus):
    """A vertical header names rows, not columns: there is nothing to explain.

    The row header is a place people right-click to select a row, so the
    gesture is ordinary and the answer must be silence.
    """
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)
    header = table.verticalHeader()
    # the resolver itself is right
    assert fd.column_name_at(header, QPoint(4, 4)) is None
    _right_click(qapp, header, QPoint(4, 4))
    assert menus == []


def test_a_widget_inside_a_cell_is_answered_by_the_table_behind_it(
        qtbot, qapp, menus):
    """Tables with embedded widgets must still explain their columns.

    A right-click on a cell widget reaches the widget, not the view, so the
    resolver finds no model on it. The event then propagates to the viewport,
    which is where the answer comes from — the filter has to stay silent on
    the first pass for the second one to happen.
    """
    from PySide6.QtWidgets import QLabel

    table = _measurement_table(qtbot)
    label = QLabel("42")
    table.setCellWidget(0, 1, label)
    table.visualRect(table.model().index(0, 1))   # place the cell widget
    fd.install_context_menu_filter(qapp)

    # nothing on the widget itself: it has no model
    assert fd.column_name_at(label, QPoint(2, 2)) is None
    # the pixel the user right-clicked, read in the viewport's coordinates
    under_cursor = fd.column_name_at(
        table.viewport(), label.mapTo(table.viewport(), QPoint(2, 2)))
    assert under_cursor == "cell_area"

    _right_click(qapp, label, QPoint(2, 2))
    assert [_texts(menu) for menu in menus] == [[fd.CONTEXT_ACTION_TEXT]]
    assert under_cursor in menus[0].actions()[0].statusTip()


def test_a_table_that_has_no_model_yet_offers_nothing(qtbot, qapp, menus):
    """A results view before its query has run has no columns to explain."""
    table = QTableView()
    qtbot.addWidget(table)
    table.resize(400, 200)
    fd.install_context_menu_filter(qapp)
    assert fd.column_name_at(table.viewport(), QPoint(20, 20)) is None
    _right_click(qapp, table.viewport(), QPoint(20, 20))
    assert menus == []


def test_a_table_carrying_its_own_actions_keeps_its_own_menu(qtbot, qapp,
                                                              menus):
    """``ActionsContextMenu`` is a claim too, not only the custom signal.

    A screen that adds QActions to its table gets Qt's own menu of them. This
    filter appending "What is this?" to a menu it did not build would be the
    same intrusion the CustomContextMenu guard exists to prevent.
    """
    table = _measurement_table(qtbot)
    table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
    fd.install_context_menu_filter(qapp)
    rect = table.visualRect(table.model().index(0, 1))
    _right_click(qapp, table.viewport(), rect.center())
    assert menus == []


def test_right_clicking_past_the_last_column_offers_nothing(qtbot, qapp,
                                                            menus):
    """The empty stretch to the right of the last header section."""
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)
    header = table.horizontalHeader()
    far = QPoint(header.length() + 60, 4)
    assert fd.column_name_at(header, far) is None
    _right_click(qapp, header, far)
    assert menus == []


def test_right_clicking_below_the_last_row_offers_nothing(qtbot, qapp, menus):
    """Empty viewport is not a column, even in a measurements table."""
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)
    below = QPoint(20, table.viewport().height() - 4)
    assert fd.column_name_at(table.viewport(), below) is None
    _right_click(qapp, table.viewport(), below)
    assert menus == []


def test_a_grid_whose_model_names_no_columns_is_left_alone(qtbot, qapp, menus):
    """No column name means no question to answer.

    A model that returns ``None`` from ``headerData`` is a real shape — the
    dictionary must read that as "nothing to explain" rather than stringifying
    it into the column name ``"None"``.
    """
    table = _model_table(qtbot, [None, None, None])
    fd.install_context_menu_filter(qapp)
    rect = table.visualRect(table.model().index(0, 1))
    assert fd.column_name_at(table.viewport(), rect.center()) is None
    _right_click(qapp, table.viewport(), rect.center())
    assert menus == []


def test_a_grid_with_unnamed_columns_is_still_not_a_measurements_table(
        qtbot, qapp, menus):
    """Unnamed columns must be skipped when deciding, not counted.

    ``_table_looks_measured`` is what keeps "What is this?" off somebody
    else's grid. A ``None`` header in the middle of it must not be parsed as a
    column name — counting it would be one step towards the threshold of three
    that opens the menu on a grid that has nothing to do with spaCR.
    """
    table = _model_table(qtbot, [None, "Source", "Plate", "Filename"])
    fd.install_context_menu_filter(qapp)
    rect = table.visualRect(table.model().index(0, 1))
    assert fd.column_name_at(table.viewport(), rect.center()) == "Source"
    _right_click(qapp, table.viewport(), rect.center())
    assert menus == []


def test_a_context_menu_event_for_a_non_widget_is_ignored(qapp, menus):
    """The filter sits on the QApplication and sees every object's events."""
    filt = fd.install_context_menu_filter(qapp)
    event = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(1, 1),
                              QPoint(1, 1))
    assert filt.eventFilter(QObject(), event) is False
    assert filt.eventFilter(QObject(), QEvent(QEvent.Type.User)) is False
    assert menus == []


# ---------------------------------------------------------------------------
# the right-clicks that must answer
# ---------------------------------------------------------------------------

def test_an_unrecognised_cell_in_a_measurements_table_still_gets_the_menu(
        qtbot, qapp, menus):
    """"I do not know" is a useful answer next to columns that are features.

    The cell gesture, not the header one: the model has to be found through
    the viewport's parent for the "is this a measurements table?" question to
    be answerable at all.
    """
    table = _measurement_table(
        qtbot, MEASUREMENT_HEADERS + ["someones_custom_column"])
    fd.install_context_menu_filter(qapp)
    rect = table.visualRect(table.model().index(0, 4))
    _right_click(qapp, table.viewport(), rect.center())

    assert [_texts(menu) for menu in menus] == [[fd.CONTEXT_ACTION_TEXT]]
    menus[0].actions()[0].trigger()
    assert "Not in the dictionary" in fd._DIALOG.panel.detail_text()


def test_a_view_that_will_not_hand_over_its_header_still_answers(qtbot, qapp,
                                                                 menus):
    """A part-destructed view costs the menu nothing.

    ``_menu_family`` reaches for the header and the viewport to see whether
    the table claimed its own context menu. Either accessor can raise on a
    view that is on its way out, and a right-click must not raise with it.
    """
    table = _TableWithNoHeaderAccessor(2, len(MEASUREMENT_HEADERS))
    table.setHorizontalHeaderLabels(MEASUREMENT_HEADERS)
    qtbot.addWidget(table)
    table.resize(800, 200)
    fd.install_context_menu_filter(qapp)

    rect = table.visualRect(table.model().index(0, 1))
    _right_click(qapp, table.viewport(), rect.center())
    assert [_texts(menu) for menu in menus] == [[fd.CONTEXT_ACTION_TEXT]]


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


def test_the_filter_is_removed_once_and_only_once(qapp):
    """Idempotent teardown: a second removal must report there was nothing."""
    fd.install_context_menu_filter(qapp)
    assert fd.remove_context_menu_filter(qapp) is True
    assert fd.remove_context_menu_filter(qapp) is False
    assert fd._FILTER is None


# ---------------------------------------------------------------------------
# the shared dialog
# ---------------------------------------------------------------------------

def test_the_dialog_is_forgotten_when_its_parent_window_dies(qtbot, qapp):
    """A destroyed dialog must not be handed out again.

    The dialog is parented to the window that opened it, so closing that
    window destroys it. The cached reference would then be a wrapper around a
    dead C++ object, and the next "What is this?" would raise RuntimeError
    inside a right-click instead of opening a window.
    """
    window = QMainWindow()
    qtbot.addWidget(window)
    dialog = fd.open_feature_dictionary(window, "cell_area")
    assert dialog.panel.current_doc().key == "area"

    window.close()
    window.deleteLater()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    assert fd._DIALOG is None

    fresh = fd.open_feature_dictionary(None, "nucleus_channel_0_blur")
    assert fresh is not dialog
    assert fresh.panel.current_doc().key == "blur"


def test_closing_the_dictionary_leaves_nothing_cached(qtbot):
    """``close_feature_dictionary`` must drop the reference, not just hide."""
    fd.open_feature_dictionary(None, "cell_area")
    fd.close_feature_dictionary()
    assert fd._DIALOG is None
    fd.close_feature_dictionary()          # and again, on nothing
    assert fd._DIALOG is None


def test_an_old_dialog_dying_does_not_forget_the_live_one(qapp):
    """One dictionary window, however the previous one was disposed of."""
    first = fd.open_feature_dictionary(None, "cell_area")
    fd.close_feature_dictionary()               # posts deleteLater on `first`
    second = fd.open_feature_dictionary(None, "cell_area")
    assert second is not first

    qapp.sendPostedEvents(None, QEvent.DeferredDelete)   # `first` dies here
    try:
        assert fd._DIALOG is second
        assert fd.open_feature_dictionary(None, "cell_area") is second
    finally:
        second.close()
        second.deleteLater()
