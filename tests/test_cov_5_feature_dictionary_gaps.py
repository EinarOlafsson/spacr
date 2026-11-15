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

from PySide6.QtCore import Qt          # noqa: E402
from PySide6.QtWidgets import (QMainWindow, QTableWidget,       # noqa: E402
                               QTableWidgetItem)

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


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# The shared dialog and the screen factory
# ---------------------------------------------------------------------------

def test_the_screen_factory_builds_a_panel(qtbot):
    screen = fd.make_screen()
    qtbot.addWidget(screen)

    assert isinstance(screen, fd.FeatureDictionaryPanel)


# ---------------------------------------------------------------------------
# Finding the Help menu
# ---------------------------------------------------------------------------

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
