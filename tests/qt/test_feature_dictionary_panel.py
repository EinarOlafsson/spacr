"""The in-app feature dictionary: the panel, and the two ways into it.

The dictionary itself is tested in ``tests/test_feature_dict_search.py``.
What is tested here is that a user can REACH it — from the Help menu and by
right-clicking a column in a results table — and that what they then read is
about the column they clicked rather than about the feature in general.

Registration is done and undone by a fixture. ``spacr.qt.app.APPS`` is a
process-global list that a dozen other test modules count tiles, sidebar rows
and translated phrases from, so a registration leaked out of this file would
be a failure somewhere else entirely.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import QTableWidget, QWidget  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

from spacr.qt.widgets import feature_dictionary as fd  # noqa: E402

pytestmark = pytest.mark.qt

MEASUREMENT_HEADERS = [
    "object_label",
    "cell_area",
    "cell_channel_1_percentile_75",
    "nucleus_periphery_mean",
]


@pytest.fixture
def panel(qtbot):
    """A fresh panel, cleaned up with the rest of the widgets."""
    widget = fd.FeatureDictionaryPanel()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def registered():
    """Register the app + QSS for one test, then take them back out."""
    from spacr.qt import app as app_mod
    from spacr.qt import theme as theme_mod

    fd.register()
    try:
        yield app_mod
    finally:
        app_mod.unregister_app(fd.APP_KEY)
        theme_mod.unregister_widget_qss(fd.OBJECT_NAME)


@pytest.fixture(autouse=True)
def _no_leaked_dialog_or_filter(qapp):
    """Neither the shared dialog nor the app-wide filter may outlive a test."""
    yield
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


def _measurement_table(qtbot, headers=None):
    table = QTableWidget(2, len(headers or MEASUREMENT_HEADERS))
    table.setHorizontalHeaderLabels(headers or MEASUREMENT_HEADERS)
    qtbot.addWidget(table)
    table.resize(800, 200)
    return table


def _right_click(qapp, widget, pos):
    """Send a real context-menu event, the way a right-click does."""
    event = QContextMenuEvent(QContextMenuEvent.Mouse, pos,
                              widget.mapToGlobal(pos))
    qapp.notify(widget, event)


# ---------------------------------------------------------------------------
# the panel
# ---------------------------------------------------------------------------

def test_panel_opens_on_the_whole_dictionary(panel):
    from spacr.feature_dict import feature_docs
    assert len(panel.result_keys()) == len(feature_docs())
    assert panel.current_doc() is not None


def test_typing_a_concept_narrows_to_that_concept(panel):
    panel.set_query("texture")
    keys = panel.result_keys()
    assert keys[0] == "homogeneity_distance_<d>"
    assert "mean_intensity" not in keys[:3]


def test_typing_a_column_name_selects_that_feature(panel):
    panel.set_query("cell_channel_1_percentile_75")
    assert panel.current_doc().key == "percentile_<p>"


def test_show_column_explains_that_column_not_just_its_feature(panel):
    """A user clicked one column; the answer must be about that column."""
    panel.show_column("nucleus_channel_2_percentile_25")
    assert panel.current_doc().key == "percentile_<p>"
    text = panel.detail_text()
    assert "nucleus_channel_2_percentile_25" in text
    # the concrete parameter, not the template
    assert "25th percentile" in text
    # the concrete channel and the concrete table
    assert "channel 2" in text
    assert "the nucleus table" in text
    # and the four things the panel is required to show
    assert "native image intensity units" in text          # unit
    assert "spacr.measure" in text                          # module
    assert "intensity" in text                              # family
    assert "cell, nucleus, pathogen, organelle, cytoplasm" in text


def test_show_column_says_where_a_feature_is_NOT_written(panel):
    """The fact no column name carries."""
    panel.show_column("nucleus_channel_0_periphery_mean")
    text = panel.detail_text()
    assert "nucleus, pathogen, organelle" in text
    assert "NOT for cell, cytoplasm" in text


def test_show_column_reports_an_unknown_column_as_unknown(panel):
    """No invention: the panel must not pick the nearest-looking entry."""
    panel.show_column("cell_channel_1_flurb")
    text = panel.detail_text()
    assert "Not in the dictionary" in text
    assert "not guessed at" in text
    assert panel.current_doc() is None


def test_a_composed_name_nobody_has_written_still_resolves(panel):
    """The panel inherits the resolver's composition, not a literal table."""
    panel.show_column("cytoplasm_channel_9_percentile_5")
    assert panel.current_doc().key == "percentile_<p>"
    assert "channel 9" in panel.detail_text()


def test_the_object_filter_hides_features_that_object_does_not_have(panel):
    panel.set_query("periphery")
    assert "periphery_mean" in panel.result_keys()
    panel._object.setCurrentIndex(panel._object.findData("cell"))
    assert "periphery_mean" not in panel.result_keys()
    panel._object.setCurrentIndex(panel._object.findData("nucleus"))
    assert "periphery_mean" in panel.result_keys()


def test_a_search_that_matches_nothing_says_so_and_suggests_a_way_in(panel):
    panel.set_query("zzzzz-not-a-feature")
    assert panel.result_keys() == []
    assert "No feature matches" in panel.detail_text()


def test_the_panel_emits_the_key_it_selected(panel, qtbot):
    with qtbot.waitSignal(panel.feature_selected) as caught:
        panel.show_column("cell_area")
    assert caught.args == ["area"]


# ---------------------------------------------------------------------------
# hook 1: the app registry and the theme, through their seams
# ---------------------------------------------------------------------------

def test_register_adds_the_app_through_the_seam(registered):
    app_mod = registered
    rows = [row for row in app_mod.APPS if row[0] == fd.APP_KEY]
    assert len(rows) == 1
    assert rows[0][1] == fd.APP_NAME
    assert rows[0][3] == app_mod.SECTION_EXPLORE
    # A screen factory, so the panel is the screen rather than a settings form
    assert app_mod.registered_factory(fd.APP_KEY) is fd.make_screen


def test_registering_makes_the_explore_section_appear(registered):
    app_mod = registered
    assert app_mod.SECTION_EXPLORE in app_mod.SECTIONS
    assert app_mod.SECTION_EXPLORE in app_mod.SECTION_NOTES


def test_register_is_idempotent(registered):
    app_mod = registered
    fd.register()
    fd.register()
    assert sum(1 for row in app_mod.APPS if row[0] == fd.APP_KEY) == 1


def test_the_panel_qss_reaches_the_stylesheet(registered, qapp):
    from spacr.qt.theme import stylesheet, widget_qss_names
    assert fd.OBJECT_NAME in widget_qss_names()
    qss = stylesheet()
    assert "FeatureDictionaryDetail" in qss
    assert f"QWidget#{fd.OBJECT_NAME}" in qss


def test_nothing_is_registered_until_register_is_called():
    """Importing the module must not mutate a global registry.

    ``tests/qt/test_all_module_smoke.py`` imports every module in the package;
    if this one registered at import time it would add an app to every test
    session that happens to import it, in an order nobody controls.
    """
    from spacr.qt import app as app_mod
    assert not any(row[0] == fd.APP_KEY for row in app_mod.APPS)


def test_the_module_is_declared_in_the_self_registering_list():
    """The wiring that makes `register()` run in a real launch."""
    from spacr.qt import SELF_REGISTERING_MODULES
    assert "spacr.qt.widgets.feature_dictionary" in SELF_REGISTERING_MODULES


# ---------------------------------------------------------------------------
# hook 2: the Help menu
# ---------------------------------------------------------------------------

def test_the_help_menu_gains_the_dictionary(qtbot):
    from PySide6.QtWidgets import QMainWindow
    window = QMainWindow()
    qtbot.addWidget(window)
    bar = window.menuBar()
    help_menu = bar.addMenu("&Help")
    help_menu.addAction("About")

    action = fd.install_help_action(window)
    assert action is not None
    assert fd.HELP_ACTION_TEXT in [a.text() for a in help_menu.actions()]
    assert action.statusTip()


def test_the_help_action_is_not_added_twice(qtbot):
    from PySide6.QtWidgets import QMainWindow
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&Help").addAction("About")
    assert fd.install_help_action(window) is not None
    assert fd.install_help_action(window) is None
    menu = fd._find_menu(window, "Help")
    assert [a.text() for a in menu.actions()].count(fd.HELP_ACTION_TEXT) == 1


def test_a_window_with_no_help_menu_is_left_alone(qtbot):
    from PySide6.QtWidgets import QMainWindow
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&File")
    assert fd.install_help_action(window) is None


def test_the_help_action_opens_the_dictionary(qtbot):
    from PySide6.QtWidgets import QMainWindow
    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&Help").addAction("About")
    action = fd.install_help_action(window)
    action.trigger()
    assert fd._DIALOG is not None
    assert fd._DIALOG.panel.result_keys()


def test_the_shortcuts_module_installs_the_hooks_on_a_real_window(qtbot):
    """The seam that runs in a real launch, exercised on the shipped window."""
    from PySide6.QtWidgets import QMainWindow
    from spacr.qt import shortcuts

    window = QMainWindow()
    qtbot.addWidget(window)
    window.menuBar().addMenu("&Help").addAction("About")
    shortcuts.install(window)

    menu = fd._find_menu(window, "Help")
    assert fd.HELP_ACTION_TEXT in [a.text() for a in menu.actions()]
    assert fd._FILTER is not None


# ---------------------------------------------------------------------------
# hook 3: "What is this?" on a results table
# ---------------------------------------------------------------------------

def test_right_clicking_a_column_header_offers_to_explain_it(qtbot, qapp):
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)

    shown: list[list[str]] = []
    fd.set_menu_runner(lambda menu, pos: shown.append(
        [a.text() for a in menu.actions()]))

    header = table.horizontalHeader()
    assert fd.column_name_at(
        header, QPoint(header.sectionPosition(2) + 4, 4)
    ) == "cell_channel_1_percentile_75"
    _right_click(qapp, header, QPoint(header.sectionPosition(2) + 4, 4))
    assert shown == [[fd.CONTEXT_ACTION_TEXT]]


def test_the_menu_item_opens_the_dictionary_on_that_column(qtbot, qapp):
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)

    fd.set_menu_runner(lambda menu, pos:
                       [a.trigger() for a in menu.actions()])
    header = table.horizontalHeader()
    _right_click(qapp, header, QPoint(header.sectionPosition(3) + 4, 4))

    assert fd._DIALOG is not None
    panel = fd._DIALOG.panel
    assert panel.current_doc().key == "periphery_mean"
    assert "nucleus_periphery_mean" in panel.detail_text()


def test_right_clicking_a_cell_explains_that_cell_s_column(qtbot, qapp):
    """Right-clicking the number, not the header, is the commoner gesture."""
    table = _measurement_table(qtbot)
    fd.install_context_menu_filter(qapp)
    fd.set_menu_runner(lambda menu, pos:
                       [a.trigger() for a in menu.actions()])

    rect = table.visualRect(table.model().index(0, 1))
    _right_click(qapp, table.viewport(), rect.center())
    assert fd._DIALOG is not None
    assert fd._DIALOG.panel.current_doc().key == "area"


def test_a_table_that_is_not_measurements_is_left_alone(qtbot, qapp):
    """The ingestion grid, the settings diff, the run list: not our business."""
    table = _measurement_table(qtbot, ["Source", "Plate", "Well", "Filename"])
    fd.install_context_menu_filter(qapp)
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))

    header = table.horizontalHeader()
    _right_click(qapp, header, QPoint(4, 4))
    assert shown == []


def test_an_unrecognised_column_in_a_measurements_table_still_gets_the_menu(
        qtbot, qapp):
    """"I do not know" is a useful answer when the neighbours are features."""
    table = _measurement_table(
        qtbot, MEASUREMENT_HEADERS + ["someones_custom_column"])
    fd.install_context_menu_filter(qapp)
    fd.set_menu_runner(lambda menu, pos:
                       [a.trigger() for a in menu.actions()])

    header = table.horizontalHeader()
    _right_click(qapp, header, QPoint(header.sectionPosition(4) + 4, 4))
    assert fd._DIALOG is not None
    assert "Not in the dictionary" in fd._DIALOG.panel.detail_text()


def test_the_filter_stands_aside_for_a_table_with_its_own_menu(qtbot, qapp):
    """Adopting a context menu later must silently win over this one."""
    table = _measurement_table(qtbot)
    table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
    fd.install_context_menu_filter(qapp)
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))

    header = table.horizontalHeader()
    _right_click(qapp, header, QPoint(header.sectionPosition(2) + 4, 4))
    assert shown == []


def test_right_clicking_something_that_is_not_a_table_does_nothing(qtbot, qapp):
    plain = QWidget()
    qtbot.addWidget(plain)
    plain.resize(120, 80)
    fd.install_context_menu_filter(qapp)
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    _right_click(qapp, plain, QPoint(5, 5))
    assert shown == []


def test_installing_the_filter_twice_installs_one_filter(qapp):
    first = fd.install_context_menu_filter(qapp)
    second = fd.install_context_menu_filter(qapp)
    assert first is second


def test_the_dialog_is_reused_rather_than_stacked(qtbot):
    first = fd.open_feature_dictionary(None, "cell_area")
    second = fd.open_feature_dictionary(None, "nucleus_channel_0_blur")
    assert first is second
    assert second.panel.current_doc().key == "blur"
