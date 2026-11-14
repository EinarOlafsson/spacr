"""Clicking, dropping, restyling and exporting a real volcano.

Every case drives the widget with a real results frame and the real
renderer: the point of the module is that one function draws both the
headless PDF and the widget, so a stubbed renderer would test nothing that
ships. Only the file dialogs -- which cannot be answered offscreen -- are
replaced, and only by making them return a path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QMimeData, QPoint, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFileDialog, QMessageBox

from spacr.qt.widgets.volcano_explorer import VolcanoExplorer
from spacr.volcano_style import VolcanoStyle


def _results(n: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "guide": [f"G{i:03d}" for i in range(n)],
        "gene": [f"TGGT1_{i // 3:06d}" for i in range(n)],
        "standardized_marginal_effect": rng.normal(size=n),
        "adjusted_p_value": rng.random(n) * 0.5 + 1e-6,
        "p_value": rng.random(n),
    })


@pytest.fixture
def explorer(qapp, qtbot):
    widget = VolcanoExplorer(_results())
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def annotation_csv(tmp_path):
    path = tmp_path / "annotations.csv"
    pd.DataFrame({
        "gene": [f"TGGT1_{i:06d}" for i in range(8)],
        "localisation": ["rhoptry", "micronemes"] * 4,
        "essentiality": np.linspace(0.0, 1.0, 8),
    }).to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------
# results in, plot out
# --------------------------------------------------------------------------


def test_a_new_style_redraws_and_moves_the_side_panel_with_it(explorer):
    """Two ways to change one setting that disagree about what it now is
    would be worse than having only one of them."""
    style = VolcanoStyle()
    style.title = "A different title"
    style.point_size = 44
    explorer.set_style(style)
    assert explorer.style() is style
    assert explorer._controls["title"].text() == "A different title"


# --------------------------------------------------------------------------
# clicking is a lookup, not a hit test on pixels
# --------------------------------------------------------------------------


def test_a_click_where_no_point_has_a_finite_position_selects_nothing(
        qapp, qtbot):
    frame = _results(4)
    frame["standardized_marginal_effect"] = ["a", "b", "c", "d"]
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    assert widget.nearest_point(0.0, 0.0) is None


def test_the_nearest_point_is_normalised_by_each_axis_range(explorer):
    """Raw Euclidean distance picks the wrong point whenever the axes have
    different scales, which on a volcano they always do."""
    explorer.refresh()
    axes = explorer._panels[0]
    frame = explorer.results()
    row = 2
    x = float(frame["standardized_marginal_effect"][row])
    y = float(-np.log10(frame["adjusted_p_value"][row]))
    assert explorer.nearest_point(x, y, axes) == row
    # Without the axis ranges the same click is measured in raw data units.
    assert explorer.nearest_point(x, y, None) in (row, None)


# --------------------------------------------------------------------------
# the split axis
# --------------------------------------------------------------------------


def test_no_y_values_means_no_split_to_suggest(qapp, qtbot):
    widget = VolcanoExplorer(_results(0))
    qtbot.addWidget(widget)
    assert widget._suggest_split() is None


def test_a_flat_y_axis_has_no_split_worth_making(qapp, qtbot):
    frame = _results(8)
    frame["adjusted_p_value"] = 0.5
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    assert widget._suggest_split() is None


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def test_a_column_survives_a_missing_value_checker_that_refuses_it(
        explorer, monkeypatch):
    original = pd.isna

    def sometimes_unavailable(value):
        if isinstance(value, str) and value == "guide":
            raise TypeError("cannot check this column label")
        return original(value)

    monkeypatch.setattr(pd, "isna", sometimes_unavailable)
    explorer._repopulate_column_menus()

    assert explorer._controls["label_column"].findData("guide") >= 0


# --------------------------------------------------------------------------
# the style is a value: saved, loaded, dropped
# --------------------------------------------------------------------------


def test_picking_an_annotation_file_reports_what_it_added(explorer,
                                                          annotation_csv,
                                                          monkeypatch):
    told = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(annotation_csv), "")))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *args: told.append(args)))
    explorer._pick_annotation_file()
    assert told and "Added 2 columns" in told[0][2]


def test_an_annotation_file_that_cannot_be_merged_says_why(explorer,
                                                           tmp_path,
                                                           monkeypatch):
    path = tmp_path / "unrelated.csv"
    pd.DataFrame({"orf": ["a"]}).to_csv(path, index=False)
    warned = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *args: warned.append(args)))
    explorer._pick_annotation_file()
    assert warned and "Could not merge annotations" in warned[0][1]


# --------------------------------------------------------------------------
# drag and drop
# --------------------------------------------------------------------------

def _urls(paths):
    """A live QMimeData carrying local file URLs.

    The caller HOLDS it: a Qt drag event keeps a bare pointer to the mime
    data, so letting the temporary die takes the interpreter with it.
    """
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return mime


def test_a_drag_carrying_files_is_accepted(explorer, annotation_csv):
    mime = _urls([annotation_csv])
    event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    explorer.dragEnterEvent(event)
    assert event.isAccepted()


def test_a_drag_carrying_no_file_is_refused(explorer):
    mime = QMimeData()
    mime.setText("just some text")
    event = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                            Qt.NoModifier)
    explorer.dragEnterEvent(event)
    assert not event.isAccepted()


def test_dropping_a_style_file_restyles_the_plot(explorer, tmp_path):
    style = VolcanoStyle()
    style.title = "Dropped style"
    path = tmp_path / "style.json"
    style.save(str(path))
    mime = _urls([path])
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime,
                       Qt.LeftButton, Qt.NoModifier)
    explorer.dropEvent(event)
    assert explorer.style().title == "Dropped style"


def test_a_drop_with_nothing_local_in_it_is_ignored(explorer):
    mime = QMimeData()
    mime.setUrls([QUrl("https://example.org/volcano.csv")])
    event = QDropEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton,
                       Qt.NoModifier)
    explorer.dropEvent(event)
    assert not event.isAccepted()


# --------------------------------------------------------------------------
# the right-click menu
# --------------------------------------------------------------------------

def test_the_style_menu_is_built_from_the_style_itself(explorer):
    """A style that gains a field gains a menu entry without anyone
    remembering to add one."""
    menu = explorer.build_style_menu()
    assert menu.actions()
    assert menu.toolTipsVisible()


def test_right_clicking_the_canvas_shows_that_menu_at_the_pointer(explorer,
                                                                  monkeypatch):
    shown = []

    class _Menu:
        def exec(self, position):
            shown.append(position)

    monkeypatch.setattr(explorer, "build_style_menu", _Menu)
    explorer._style_menu(QPoint(4, 4))
    assert shown == [explorer._canvas.mapToGlobal(QPoint(4, 4))]


def test_a_mapping_menu_that_is_not_on_this_panel_is_skipped(explorer):
    """The column menus are refilled by name, and a panel that does not
    build one of them must not cost the others their columns."""
    del explorer._controls["shape_by"]
    explorer.set_results(_results(6))
    offered = [explorer._controls["color_by"].itemData(i)
               for i in range(explorer._controls["color_by"].count())]
    assert "gene" in offered


# --------------------------------------------------------------------------
# the menu and the panel are two routes to one figure
# --------------------------------------------------------------------------

def _menu_words(menu) -> list:
    """Every word the menu shows a reader: entries and the groups holding
    them."""
    from spacr.qt.widgets.fast_plots import menu_entries, menu_groups

    return ([action.text() for action in menu_entries(menu)]
            + list(menu_groups(menu)))


def test_the_menu_offers_a_setting_in_the_panels_own_words(explorer):
    """The panel says "Circle" and the style stores "o". A menu offering the
    stored value is the same setting in a second vocabulary, and only one of
    the two is the one a reader recognises."""
    from spacr.qt.widgets.fast_plots import menu_entries, menu_groups

    menu = explorer.build_style_menu()
    assert any(title.startswith("Marker: Circle")
               for title in menu_groups(menu)), menu_groups(menu)
    offered = {action.text() for action in menu_entries(menu)}
    assert {"Circle", "Square", "Solid", "Dashed"} <= offered
    assert "o" not in offered and "--" not in offered


# --------------------------------------------------------------------------
# colour by localization, several at once
# --------------------------------------------------------------------------

@pytest.fixture
def localised(qapp, qtbot):
    """A screen whose gene ids the bundled LOPIT table actually knows.

    Two compartments, ten genes each, so "dense granules and rhoptries 1" is
    a question this frame can answer.
    """
    from spacr.localisation import table

    lookup = table()
    if not lookup:
        pytest.skip("no bundled localisation table to colour by")
    genes = []
    for compartment in ("dense granules", "rhoptries 1"):
        genes += [f"TGGT1_{key}" for key, place in lookup.items()
                  if place == compartment][:10]
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({
        "guide": [f"{gene}_1" for gene in genes],
        "gene": genes,
        "standardized_marginal_effect": rng.normal(size=len(genes)),
        "adjusted_p_value": rng.random(len(genes)) * 0.5 + 1e-6,
    })
    widget = VolcanoExplorer(frame)
    qtbot.addWidget(widget)
    return widget


def test_two_compartments_can_be_ticked_at_once_on_the_menu(localised):
    """"all or any combination of localizations, e.g. dense granuals and
    rhoptries 1 should be possible" -- so this is a submenu of tick boxes,
    not a pick-one."""
    menu = localised.build_style_menu()
    ticks = [a for a in _menu_words(menu)]
    assert "dense granules" in ticks and "rhoptries 1" in ticks

    from spacr.qt.widgets.fast_plots import menu_entries

    by_text = {a.text(): a for a in menu_entries(menu)}
    by_text["dense granules"].setChecked(True)
    assert localised.style().localizations == ("dense granules",)

    # Applying a style rebuilds the context menu, just as it rebuilds the
    # plotted artists. Reopen it before making the second user choice rather
    # than retaining a QAction whose owning QMenu has been deleted.
    menu = localised.build_style_menu()
    by_text = {a.text(): a for a in menu_entries(menu)}
    assert by_text["dense granules"].isChecked()
    by_text["rhoptries 1"].setChecked(True)
    assert localised.style().localizations == ("dense granules",
                                               "rhoptries 1")


def test_the_menus_ticks_reach_the_side_panel(localised):
    """Two ways to change one setting that disagree about what it now is
    would be worse than having only one of them."""
    from spacr.qt.widgets.fast_plots import menu_entries

    menu = localised.build_style_menu()
    next(a for a in menu_entries(menu)
         if a.text() == "rhoptries 1").setChecked(True)

    assert localised._controls["localizations"].values() == ("rhoptries 1",)


def test_a_compartment_this_screen_lacks_survives_a_refill(localised):
    """A style file naming a compartment the screen has none of must not
    lose it the moment the panel is refilled."""
    localised._style.localizations = ("apicoplast",)
    localised.set_style(localised._style)
    localised._repopulate_column_menus()

    assert "apicoplast" in localised._controls["localizations"].values()


# --------------------------------------------------------------------------
# nothing says "aspect ratio"
# --------------------------------------------------------------------------

def test_nothing_the_volcano_says_to_a_user_mentions_an_aspect_ratio(
        explorer):
    """A ratio is a number and the choice is a shape."""
    from PySide6.QtWidgets import QAbstractButton, QGroupBox, QLabel, QWidget

    words = _menu_words(explorer.build_style_menu())
    for widget in explorer.findChildren(QWidget):
        words.append(widget.toolTip())
        caption = widget.property("caption")
        if caption:
            words.append(str(caption))
        if isinstance(widget, QGroupBox):
            words.append(widget.title())
        elif isinstance(widget, (QAbstractButton, QLabel)):
            words.append(widget.text())
    offending = [word for word in words
                 if word and "aspect ratio" in word.lower()]
    assert offending == [], offending
