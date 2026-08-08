"""A selected row must be readable, on every theme and every view kind.

The SQL column picker is where it was reported -- multi-selecting columns
painted them with Qt's own selection colours, which assume a light
background and use BLACK text. On the dark theme the chosen rows were the
only unreadable thing on screen.

It was never one dialog's bug. There was a `:hover` rule for item views and
no `:selected` one, so every multi-select list, table and tree in the
application had it.

Why it matters more than it sounds: in a column picker the selection IS the
state of the dialog. Invisible selection means no way to tell what you are
about to query, and the only recovery is to start over.
"""

from __future__ import annotations

import re

import pytest

from PySide6.QtGui import QColor

#: Every view kind that can be multi-selected. `QListWidget` is listed
#: separately because it is not a `QTableView` and the table rules do not
#: reach it -- which is exactly how it was missed.
VIEWS = ("QListView", "QListWidget", "QTableView", "QTableWidget",
         "QTreeView", "QTreeWidget")


@pytest.fixture()
def sheets(qapp):
    from spacr.qt import theme

    return {name: theme.stylesheet(name) for name in ("dark", "light")}


@pytest.mark.parametrize("view", VIEWS)
def test_every_item_view_styles_its_selection(sheets, view):
    for name, sheet in sheets.items():
        assert f"{view}::item:selected" in sheet, (
            f"{name}: {view} has no selection rule, so it falls through to "
            f"Qt's default, which assumes a light background")


@pytest.mark.parametrize("view", VIEWS)
def test_the_selection_survives_losing_focus(sheets, view):
    """Qt dims an inactive selection to a grey close to the surface.

    That is how a picked column vanishes the moment the user clicks OK.
    """
    for name, sheet in sheets.items():
        assert f"{view}::item:selected:!active" in sheet, (
            f"{name}: {view} loses its selection colour when unfocused")


def _rule_after(sheet: str, needle: str) -> str:
    start = sheet.index(needle)
    return sheet[start:sheet.index("}", start)]


@pytest.mark.parametrize("theme_name", ["dark", "light"])
def test_selected_text_is_readable_against_the_selected_background(
        qapp, theme_name):
    """Measured as a contrast ratio, not eyeballed.

    The failure being guarded is precisely a colour pair that a person
    picked without checking: black on the accent looked fine to whoever
    chose it on a light screen.
    """
    from spacr.qt import theme

    sheet = theme.stylesheet(theme_name)
    rule = _rule_after(sheet, "QListWidget::item:selected")
    colours = re.findall(r"(?:background-color|color)\s*:\s*(#[0-9a-fA-F]{6})",
                         rule)
    assert len(colours) >= 2, f"could not read the pair out of: {rule!r}"

    background, foreground = QColor(colours[0]), QColor(colours[1])
    ratio = theme.contrast_ratio(foreground.name(), background.name())
    assert ratio >= 4.5, (
        f"{theme_name}: selected text is {ratio:.2f}:1 against its own "
        f"background ({foreground.name()} on {background.name()})")


def test_the_selection_is_not_the_same_colour_as_the_surface(qapp):
    """Otherwise "selected" and "not selected" look identical."""
    from spacr.qt import theme

    for name in ("dark", "light"):
        palette = theme.palette_for(name)
        sheet = theme.stylesheet(name)
        rule = _rule_after(sheet, "QListWidget::item:selected")
        background = re.search(
            r"background-color\s*:\s*(#[0-9a-fA-F]{6})", rule).group(1)
        assert background.lower() != palette["surface"].lower()
        assert background.lower() != palette["surface_alt"].lower()
