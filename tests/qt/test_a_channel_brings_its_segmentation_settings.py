"""Typing an object's channel puts that object's segmentation settings up.

GitHub issue #120, jak18015, spaCR 1.5.0.8 on macOS: "when defining a
channel number for pathogen, no pathogen segmentation list appears in the
settings." Steps: open Mask generation, add a pathogen channel number, the
pathogen segmentation settings section does not appear.

Measured on nightly 5b9207b21 and on v1.5.0.8 alike, driving the channel
field the way a user does (type, press Enter): under "All settings" the
section appeared; under "Essentials", which is where every module opens
until the user changes it, it never did. Essentials was a fixed list --
the inputs and the workflow switches -- so no object's segmentation was in
it, and the object rule's own pass then undid the level filter for rows it
put back. Two more faces of the same thing were found on the way:

* with the per-object table switched on, Essentials hid the table itself,
  and the table is the only place the channels are then shown -- no channel
  could be set at all;
* clearing a channel left its heading on the form over no rows.

Everything below presses the control the user presses and asks the built
form what it shows.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _fresh_disclosure():
    """Every test starts where a first visit starts: Essentials."""
    from spacr.qt.settings_search import forget_disclosure

    forget_disclosure()
    yield
    forget_disclosure()


@pytest.fixture
def grid_preference():
    """Restore the per-object table preference whatever a test does."""
    from spacr.qt import preferences as prefs

    was = prefs.get_object_grid_enabled()
    yield prefs
    prefs.set_object_grid_enabled(was)


def _open_mask(qtbot, level=None):
    """Open Mask generation in a window, the way the navigation does."""
    from PySide6.QtWidgets import QApplication

    import spacr.qt.app as app_module
    from spacr.qt.settings_search import remember_disclosure

    if level is not None:
        remember_disclosure("mask", level)
    window = app_module.MainWindow()
    qtbot.addWidget(window)
    window.resize(1400, 900)
    window.show()
    window._on_nav_selected("mask")
    for _ in range(10):
        QApplication.processEvents()
    return window


def _screen(window):
    """The Mask screen on show now; a bulk load may have replaced it."""
    return window._screens["mask"]


def _commit(qtbot, screen, key, text):
    """Type ``text`` into ``key``'s field and press Enter."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    field = screen._settings_model._widgets[key]
    field.setFocus()
    field.clear()
    if text:
        qtbot.keyClicks(field, text)
    qtbot.keyClick(field, Qt.Key_Return)
    for _ in range(5):
        QApplication.processEvents()


def _heading(screen, title):
    """The top-level settings heading called ``title``."""
    for section in screen.rendered_settings_sections():
        if section.property("settingsCategorySource") == title:
            return section
    return None


def _heading_shown(screen, title) -> bool:
    """Whether the heading is on the form, not held back by any rule."""
    section = _heading(screen, title)
    return section is not None and not section.isHidden()


def test_a_pathogen_channel_brings_pathogen_segmentation_into_essentials(
        qtbot):
    """The reported steps, at the level a new user is on."""
    from spacr.qt.settings_search import ESSENTIALS

    window = _open_mask(qtbot)
    screen = _screen(window)
    assert screen._settings_search.level() == ESSENTIALS
    assert not _heading_shown(screen, "Pathogen Segmentation")

    _commit(qtbot, screen, "pathogen_channel", "2")

    assert _heading_shown(screen, "Pathogen Segmentation")
    for key in ("pathogen_model_name", "pathogen_diameter",
                "pathogen_cellprob_threshold", "pathogen_flow_threshold"):
        assert key in screen._settings_search.visible_keys(), key
    assert screen.setting_row_is_visible("pathogen_diameter")


def test_essentials_still_hides_what_it_hid(qtbot):
    """A channel commit re-applies the level; it does not lift it."""
    window = _open_mask(qtbot)
    screen = _screen(window)
    before = set(screen._settings_search.visible_keys())

    _commit(qtbot, screen, "pathogen_channel", "2")

    after = set(screen._settings_search.visible_keys())
    assert not screen.setting_row_is_visible("dry_run")
    assert not screen.setting_row_is_visible("resume")
    assert after - before == {
        "pathogen_model_name", "pathogen_diameter",
        "pathogen_cellprob_threshold", "pathogen_flow_threshold"}


def test_a_cell_channel_brings_cell_segmentation_into_essentials(qtbot):
    """Cell rows are never hidden by the object rule; Essentials follows."""
    window = _open_mask(qtbot)
    screen = _screen(window)
    assert not _heading_shown(screen, "Cell Segmentation")

    _commit(qtbot, screen, "cell_channel", "0")

    assert _heading_shown(screen, "Cell Segmentation")
    assert screen.setting_row_is_visible("cell_diameter")


@pytest.mark.parametrize("level", ["essentials", "all"])
def test_clearing_the_channel_takes_the_heading_away(qtbot, level):
    """On, off and on again, at either level."""
    window = _open_mask(qtbot, level)
    screen = _screen(window)

    _commit(qtbot, screen, "pathogen_channel", "2")
    assert _heading_shown(screen, "Pathogen Segmentation")

    _commit(qtbot, screen, "pathogen_channel", "")
    assert screen._settings_model.collect()["pathogen_channel"] is None
    assert not _heading_shown(screen, "Pathogen Segmentation"), (
        "a heading over no rows is left on the form")
    assert not screen.setting_row_is_visible("pathogen_diameter")

    _commit(qtbot, screen, "pathogen_channel", "2")
    assert _heading_shown(screen, "Pathogen Segmentation")
    assert screen.setting_row_is_visible("pathogen_diameter")


def test_a_settings_file_naming_a_pathogen_channel_opens_its_section(qtbot):
    """Loading settings is the other way a channel arrives."""
    window = _open_mask(qtbot)
    _screen(window).apply_settings_dict({"pathogen_channel": 2})
    screen = _screen(window)

    assert screen._settings_model.collect()["pathogen_channel"] == 2
    assert _heading_shown(screen, "Pathogen Segmentation")
    assert screen.setting_row_is_visible("pathogen_diameter")


def _grid_section(screen):
    """The section the per-object table sits in."""
    node = screen._object_grid.parentWidget()
    while node is not None and not hasattr(node, "add_prose_row"):
        node = node.parentWidget()
    return node


def test_the_per_object_table_stays_on_screen_under_essentials(
        qtbot, grid_preference):
    """With the table on, it is the only place a channel can be set."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    grid_preference.set_object_grid_enabled(True)
    window = _open_mask(qtbot)
    screen = _screen(window)
    assert getattr(screen, "_object_grid", None) is not None
    assert not screen.setting_row_is_visible("pathogen_channel"), (
        "the table answers for the channel, so its flat row is hidden")
    section = _grid_section(screen)
    assert not section.isHidden(), (
        "Essentials hid the table, and with it every object channel")

    table = screen._object_grid._model
    row = list(table.table()).index("channel")
    column = list(table.objects()).index("pathogen")
    assert table.setData(table.index(row, column), "2", Qt.EditRole)
    QApplication.processEvents()

    assert screen._settings_model.collect()["pathogen_channel"] == 2
    assert not _grid_section(screen).isHidden()


def test_switching_the_table_on_later_keeps_it_on_screen(
        qtbot, grid_preference):
    """Mounted by Preferences after the search strip was built."""
    from PySide6.QtWidgets import QApplication

    grid_preference.set_object_grid_enabled(False)
    window = _open_mask(qtbot)
    screen = _screen(window)
    assert getattr(screen, "_object_grid", None) is None

    grid_preference.set_object_grid_enabled(True)
    assert screen.apply_object_grid_preference()
    QApplication.processEvents()
    section = _grid_section(screen)
    assert not section.isHidden()

    bar = screen._settings_search
    bar.set_query("illumination")
    assert section.isHidden(), "the strip is not deciding the table's section"
    bar.set_query("")
    assert not section.isHidden()

    grid_preference.set_object_grid_enabled(False)
    assert screen.apply_object_grid_preference()
    QApplication.processEvents()
    bar.apply()
    assert screen.setting_row_is_visible("pathogen_channel")
