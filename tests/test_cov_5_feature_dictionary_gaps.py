"""The dictionary against the tables, menus and windows it does not own.

Right-clicking a column is help bolted onto grids the dictionary did not
build, through an application-wide event filter that sees every event in the
process. Two rules follow: it must recognise a column only where a column is
really under the pointer, and it must never take a right-click — or the
window — down with it. The renderer's job is the same discipline in words: a
sentence that would be a guess is not printed.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPoint, Qt          # noqa: E402
from PySide6.QtWidgets import (QMainWindow, QTableWidget,       # noqa: E402
                               QTableWidgetItem, QWidget)
from PySide6.QtGui import QContextMenuEvent                     # noqa: E402

from spacr.feature_dict import CHANNEL_PAIR, FeatureDoc         # noqa: E402
from spacr.qt.widgets import feature_dictionary as fd           # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def menus(qapp):
    """Record context menus rather than opening one nobody can click."""
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    yield shown
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


def _doc(**overrides) -> FeatureDoc:
    fields = dict(
        key="cell_area", title="Area", kind="measurement", family="shape",
        concepts=(), description="How many pixels the object covers.",
        unit="px²", computed_by="regionprops", module="spacr.measure",
        object_types=("cell",), channel_scope="none", written_when=None,
        notes=None, examples=(),
    )
    fields.update(overrides)
    return FeatureDoc(**fields)


def _table(qtbot, columns):
    table = QTableWidget(2, len(columns))
    table.setHorizontalHeaderLabels(list(columns))
    for row in range(2):
        for col in range(len(columns)):
            table.setItem(row, col, QTableWidgetItem("1"))
    qtbot.addWidget(table)
    table.resize(600, 200)
    table.show()
    return table


# ---------------------------------------------------------------------------
# The sentences the detail pane will and will not say
# ---------------------------------------------------------------------------

def test_a_two_channel_feature_says_both_channels_are_being_compared():
    said = fd._channel_sentence(CHANNEL_PAIR)

    assert "channel PAIR" in said
    assert "two channels" in said


def test_a_metadata_row_is_not_called_a_missing_measurement():
    """"Not written for any object type" would read as a broken pipeline."""
    assert fd._objects_sentence(_doc(kind="metadata", object_types=())) == (
        "Not a per-object measurement.")
    assert "see the note below" in fd._objects_sentence(
        _doc(kind="measurement", object_types=()))


def test_a_colocalisation_column_names_the_pair_it_was_measured_from(qapp):
    """Which two channels is the whole content of a colocalisation number."""
    from spacr.feature_dict import parse_column

    entry = parse_column("pathogen_channel_0_channel_2_M1_correlation_85")
    assert (entry.channel, entry.channel_2) == (0, 2)
    html = fd._doc_html(_doc(key=entry.key, channel_scope=CHANNEL_PAIR), entry)

    assert "channels 0 and 2" in html


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

def test_a_panel_opened_on_a_column_shows_that_column(qtbot):
    panel = fd.FeatureDictionaryPanel(column="cell_area")
    qtbot.addWidget(panel)

    assert (panel.current_doc().key if panel.current_doc() else None) == "area"
    assert "cell table" in panel.detail_text()


def test_a_column_the_search_never_surfaced_is_still_explained(qtbot):
    """A concrete column always resolves; the free-text list may not hold it."""
    panel = fd.FeatureDictionaryPanel()
    qtbot.addWidget(panel)
    panel.set_query("no feature has this in its name at all")
    assert panel._hits == []

    panel._select_key("area")

    assert (panel.current_doc().key if panel.current_doc() else None) == "area"


def test_a_key_that_is_not_a_feature_leaves_the_pane_alone(qtbot):
    panel = fd.FeatureDictionaryPanel()
    qtbot.addWidget(panel)
    panel._select_key("area")
    before = (panel.current_doc().key if panel.current_doc() else None)

    panel._select_key("this_is_not_a_spacr_column_at_all_%%%")

    assert (panel.current_doc().key if panel.current_doc() else None) == before


def test_a_search_that_raises_empties_the_list_instead_of_the_window(
        qtbot, monkeypatch):
    """The search runs on every keystroke; one bad query must not kill it."""
    panel = fd.FeatureDictionaryPanel()
    qtbot.addWidget(panel)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("the feature index is corrupt")

    monkeypatch.setattr(fd, "search_features", refuse)

    panel.set_query("area")

    assert panel._hits == []
    assert "No feature matches" in panel._detail.toHtml()


def test_moving_off_the_pinned_column_unpins_the_detail_pane(qtbot):
    """The channel sentence belongs to one column, not to the whole feature."""
    panel = fd.FeatureDictionaryPanel(column="cell_channel_1_mean_intensity")
    qtbot.addWidget(panel)
    assert panel._column == "cell_channel_1_mean_intensity"
    assert "channel 1" in panel.detail_text()

    other = next(index for index, hit in enumerate(panel._hits)
                 if hit.doc.key != "mean_intensity")
    panel._on_row_changed(other)

    assert panel._column is None, "the pane is still pinned to a column"
    assert "channel 1" not in panel.detail_text()

    # And a row that IS the pinned column keeps the pin.
    panel.show_column("cell_channel_1_mean_intensity")
    same = next(index for index, hit in enumerate(panel._hits)
                if hit.doc.key == "mean_intensity")
    panel._on_row_changed(same)
    assert panel._column == "cell_channel_1_mean_intensity"


def test_a_selection_change_to_no_row_changes_nothing(qtbot):
    panel = fd.FeatureDictionaryPanel(column="cell_area")
    qtbot.addWidget(panel)
    before = (panel.current_doc().key if panel.current_doc() else None)

    panel._on_row_changed(-1)

    assert (panel.current_doc().key if panel.current_doc() else None) == before


# ---------------------------------------------------------------------------
# The shared dialog and the screen factory
# ---------------------------------------------------------------------------

def test_an_older_dialogs_destruction_does_not_forget_the_live_one(qtbot):
    """Six lookups leave one dictionary open, so the cache must be exact."""
    dialog = fd.open_feature_dictionary(None, "cell_area")
    stale = fd.FeatureDictionaryDialog()
    qtbot.addWidget(stale)

    fd._forget_dialog(stale)
    assert fd.open_feature_dictionary() is dialog

    fd._forget_dialog(dialog)
    assert fd.open_feature_dictionary() is not dialog


def test_the_screen_factory_builds_a_panel(qtbot):
    screen = fd.make_screen()
    qtbot.addWidget(screen)

    assert isinstance(screen, fd.FeatureDictionaryPanel)


def test_a_theme_that_will_not_take_the_stylesheet_still_registers_the_app(
        monkeypatch):
    """A stylesheet is decoration; refusing to start over it is not."""
    from spacr.qt import theme

    def refuse(*_args, **_kwargs):
        raise RuntimeError("the theme registry is closed")

    monkeypatch.setattr(theme, "register_widget_qss", refuse)
    monkeypatch.setattr(theme, "widget_qss_names", lambda: ())

    assert fd.register() is True


# ---------------------------------------------------------------------------
# Finding the Help menu
# ---------------------------------------------------------------------------

def test_a_window_with_no_menu_bar_gets_no_help_action(qtbot):
    class _Bare(QMainWindow):
        def menuBar(self):
            return None

    window = _Bare()
    qtbot.addWidget(window)

    assert fd.install_help_action(window) is None


def test_a_menu_whose_c_object_is_gone_is_stepped_over(qtbot, monkeypatch):
    """``findChildren`` can hand back a menu Qt is in the middle of deleting."""
    window = QMainWindow()
    qtbot.addWidget(window)
    help_menu = window.menuBar().addMenu("&Help")

    class _Dead:
        def title(self):
            raise RuntimeError("Internal C++ object already deleted.")

    real = type(window.menuBar()).findChildren
    monkeypatch.setattr(
        type(window.menuBar()), "findChildren",
        lambda self, *a, **k: [_Dead()] + list(real(self, *a, **k)))

    assert fd._find_menu(window, "Help") is help_menu


# ---------------------------------------------------------------------------
# Reading the column under the pointer
# ---------------------------------------------------------------------------

def test_a_row_header_names_no_column(qtbot):
    table = _table(qtbot, ["cell_area", "cell_perimeter"])

    assert fd.column_name_at(table.verticalHeader(), QPoint(2, 2)) is None


def test_a_point_past_the_last_column_names_nothing(qtbot):
    table = _table(qtbot, ["cell_area"])

    header = table.horizontalHeader()
    assert fd.column_name_at(header, QPoint(header.width() + 500, 2)) is None


def test_a_widget_that_is_not_part_of_a_table_names_nothing(qtbot):
    stray = QWidget()
    qtbot.addWidget(stray)

    assert fd.column_name_at(stray, QPoint(2, 2)) is None
    assert fd._model_of(stray) is None


def test_a_view_with_no_model_names_nothing(qtbot):
    from PySide6.QtWidgets import QTableView

    view = QTableView()
    qtbot.addWidget(view)

    assert fd.column_name_at(view, QPoint(2, 2)) is None


def test_the_model_behind_a_viewport_is_the_views_own(qtbot):
    table = _table(qtbot, ["cell_area"])

    assert fd._model_of(table.viewport()) is table.model()
    assert fd._model_of(table.horizontalHeader()) is table.model()


def test_a_column_with_no_header_text_is_not_counted_as_a_measurement(qtbot):
    """A model that names no columns cannot be recognised as measurements."""
    from PySide6.QtCore import QAbstractTableModel, QModelIndex

    class _Nameless(QAbstractTableModel):
        def rowCount(self, _parent=QModelIndex()):
            return 1

        def columnCount(self, _parent=QModelIndex()):
            return 4

        def data(self, _index, _role=Qt.DisplayRole):
            return None

        def headerData(self, _section, _orientation, _role=Qt.DisplayRole):
            return None

    assert fd._table_looks_measured(_Nameless()) is False
    assert fd._table_looks_measured(None) is False
    assert fd._table_looks_measured(QTableWidget(1, 4).model()) is False


def test_a_view_whose_parts_raise_still_yields_a_family(qtbot):
    """The family walk is best-effort; a dying part must not stop it."""
    from PySide6.QtWidgets import QTableView

    class _Dying(QTableView):
        def horizontalHeader(self):
            raise RuntimeError("Internal C++ object already deleted.")

        def viewport(self):
            raise RuntimeError("Internal C++ object already deleted.")

    view = _Dying()
    qtbot.addWidget(view)

    family = fd._menu_family(view)

    assert family == [view, view], "the view itself must survive the walk"
    assert fd._menu_family(_table(qtbot, ["cell_area"]).horizontalHeader())


# ---------------------------------------------------------------------------
# The event filter
# ---------------------------------------------------------------------------

def _context_event(widget, point):
    return QContextMenuEvent(QContextMenuEvent.Mouse, point,
                             widget.mapToGlobal(point))


def test_a_right_click_on_a_measurement_column_offers_an_explanation(qtbot,
                                                                     menus):
    table = _table(qtbot, ["cell_area", "cell_perimeter", "cell_eccentricity"])
    filt = fd.FeatureHelpFilter()
    header = table.horizontalHeader()

    handled = filt.eventFilter(header, _context_event(header, QPoint(5, 5)))

    assert handled is True
    assert menus and menus[0].actions()[0].text() == fd.CONTEXT_ACTION_TEXT


def test_a_right_click_on_the_row_header_is_not_a_column_gesture(qtbot, menus):
    """A second delivery of an ignored event must not become column zero."""
    table = _table(qtbot, ["cell_area", "cell_perimeter", "cell_eccentricity"])
    filt = fd.FeatureHelpFilter()
    vertical = table.verticalHeader()
    point = vertical.mapTo(table, QPoint(2, 20))

    handled = filt.eventFilter(table, _context_event(table, point))

    assert handled is False
    assert menus == []


def test_a_plain_object_is_not_a_table(qtbot, menus):
    filt = fd.FeatureHelpFilter()
    stray = QObject()

    handled = filt.eventFilter(stray, _context_event(QWidget(), QPoint(1, 1)))

    assert handled is False


def test_a_filter_that_raises_lets_the_right_click_through(qtbot, menus,
                                                           monkeypatch):
    """Help must never take a right-click, or the window, down with it."""
    table = _table(qtbot, ["cell_area", "cell_perimeter", "cell_eccentricity"])
    filt = fd.FeatureHelpFilter()

    def refuse(*_args, **_kwargs):
        raise RuntimeError("the column parser blew up")

    monkeypatch.setattr(fd, "column_name_at", refuse)
    header = table.horizontalHeader()

    assert filt.eventFilter(header, _context_event(header, QPoint(5, 5))) is False
    assert menus == []


# ---------------------------------------------------------------------------
# Installing it
# ---------------------------------------------------------------------------

def test_install_swallows_both_of_its_failures(qtbot, monkeypatch):
    """A missing help entry must not cost anyone a window."""
    window = QMainWindow()
    qtbot.addWidget(window)

    def refuse_action(_window):
        raise RuntimeError("no menu bar yet")

    def refuse_filter(*_args, **_kwargs):
        raise RuntimeError("no application object")

    monkeypatch.setattr(fd, "install_help_action", refuse_action)
    monkeypatch.setattr(fd, "install_context_menu_filter", refuse_filter)

    fd.install_window_hooks(window)
