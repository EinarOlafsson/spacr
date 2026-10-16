"""Make Masks' titled button groups, collapsible bars and the shared arrow (707).

The maintainer's 2026-10-16 layout: TOOLS top left and ANNOTATION top right
of the image, IO bottom left and NAVIGATION bottom right; Keep and Save mask
carry menus; the title, the top bar and the bottom bar fold behind the same
collapse arrow as the settings and stay folded; vvvv export is one
Preferences -> Plugins switch; the brush starts at 1 px; the field's index
and name are painted on the image; and the collapse arrow is 75 percent of
its former size, centred on its handle.
"""
from __future__ import annotations

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtWidgets import QPushButton

from spacr.qt.screens.make_masks import (
    DEFAULT_BRUSH_RADIUS,
    MAKE_MASKS_BOTTOM_BAR_FOLD_KEY,
    MAKE_MASKS_TITLE_FOLD_KEY,
    MAKE_MASKS_TOP_BAR_FOLD_KEY,
    MakeMasksScreen,
)


def _screen(qtbot, folder=None):
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    screen.resize(1920, 1080)
    screen.show()
    if folder is not None:
        assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: screen._image_container.isVisible())
    return screen


@pytest.fixture
def folder(tmp_path):
    fields = tmp_path / "fields"
    fields.mkdir()
    for index in range(3):
        imageio.imwrite(fields / f"field_{index}.tif",
                        np.full((64, 64), index + 1, dtype=np.uint16))
    return fields


@pytest.fixture
def screen(qtbot, qt_theme_applied, folder):
    built = _screen(qtbot, folder)
    yield built
    built._magnifier.close()
    built.close_folded()


def _text(button):
    return button.text().replace("&", "")


def _buttons(group):
    return [w for w in group.findChildren(QPushButton)
            if w.parentWidget() is group]


def _where(screen, widget):
    return widget.mapTo(screen, QPoint(0, 0))


def test_groups_are_titled_and_placed_around_the_image(screen, qtbot):
    groups = {
        "TOOLS": screen._tools_group,
        "ANNOTATION": screen._annotation_group,
        "IO": screen._io_group,
        "NAVIGATION": screen._navigation_group,
    }
    for title, group in groups.items():
        label = group.title_label
        assert label.text() == title
        assert label.geometry().top() <= 2 * label.height()
        assert label.geometry().left() < 20
    image = screen._body_stack
    qtbot.waitUntil(lambda: image.isVisible())
    top = _where(screen, image).y()
    bottom = top + image.height()
    left_edge = _where(screen, image).x()
    middle = left_edge + image.width() / 2
    tools, annotation = (_where(screen, screen._tools_group),
                         _where(screen, screen._annotation_group))
    io, navigation = (_where(screen, screen._io_group),
                      _where(screen, screen._navigation_group))
    assert tools.y() < top and annotation.y() < top
    assert io.y() >= bottom and navigation.y() >= bottom
    assert tools.x() < middle < annotation.x()
    assert io.x() < middle < navigation.x()
    assert tools.x() - left_edge < 20 and io.x() - left_edge < 20


def test_every_button_is_in_its_group_once(screen):
    expected = {
        screen._tools_group: ["Brush", "Erase", "Erase object", "Wand", "Draw",
                              "Box", "Divide / Merge", "Zoom", "Recrop",
                              "Magnifier"],
        screen._annotation_group: ["Keep", "Skip", "Discard"],
        screen._io_group: ["Open File / Folder…", "Measure…",
                           "Load test data…", "ROIs…"],
        screen._navigation_group: ["Save mask", "Previous", "Next"],
    }
    seen = []
    for group, labels in expected.items():
        texts = [_text(b) for b in _buttons(group)]
        assert texts == labels, (group.objectName(), texts)
        seen += texts
    assert len(seen) == len(set(seen))
    everything = [_text(b) for bar in (screen._tool_row, screen._bottom_bar)
                  for b in bar.findChildren(QPushButton)]
    for label in seen:
        assert everything.count(label) == 1, label


def test_removed_and_renamed_controls(screen):
    texts = {_text(b) for b in screen.findChildren(QPushButton)}
    for gone in ("Undo", "Redo", "Settings", "Open folder…", "Open 3D image…",
                 "Organize for Measure…", "Prev image", "Next image",
                 "Export for vvvv…", "Export on save", "Save on navigation",
                 "Hide judged fields", "First unreviewed", "Blind",
                 "Upload data…"):
        assert gone not in texts, gone
    assert not hasattr(screen, "_btn_settings")
    assert not hasattr(screen, "_btn_volume")
    assert screen._act_undo.text() == "Undo" and screen._act_redo.text() == "Redo"


def test_reset_zoom_and_ruler_sit_right_of_the_image_counter(screen):
    tally = screen._tally_layout
    order = [tally.itemAt(i).widget() for i in range(tally.count())
             if tally.itemAt(i).widget() is not None]
    assert order[-2:] == [screen._btn_reset_zoom, screen._btn_ruler]
    assert order.index(screen._tally_index) < order.index(screen._btn_reset_zoom)
    assert _where(screen, screen._btn_reset_zoom).x() > _where(
        screen, screen._tally_total).x()


def test_object_operations_hold_the_moved_actions(screen):
    holder = screen._object_ops_layout
    widgets = [holder.itemAt(i).widget() for i in range(holder.count())]
    assert widgets[0] is screen._btn_cellpose
    assert screen._btn_clear in widgets
    assert screen._virtual_stain in widgets
    assert screen._tools_group.isAncestorOf(screen._btn_cellpose) is False


def test_keep_menu(screen):
    menu = screen._btn_keep.split_menu()
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == ["Hide judged fields",
                                           "First unreviewed", "Blind"]
    assert actions[0] is screen._btn_hide_judged and actions[0].isCheckable()
    assert actions[1] is screen._btn_first_unreviewed
    assert actions[2] is screen._btn_blind and actions[2].isCheckable()
    assert actions[2].objectName() == "MakeMasksBlindToggle"


def test_save_menu_has_the_boolean_and_upload(screen, monkeypatch):
    menu = screen._btn_save.split_menu()
    actions = [a for a in menu.actions() if not a.isSeparator()]
    assert [a.text() for a in actions] == ["Save on navigation",
                                           "Upload data…"]
    toggle, upload = actions
    assert toggle is screen._btn_save_on_navigation
    assert toggle.isCheckable() and not toggle.isChecked()
    assert upload.objectName() == "MakeMasksUploadDataButton"
    called = []
    monkeypatch.setattr(screen, "contribute_images_and_masks",
                        lambda: called.append(True))
    upload.trigger()
    assert called == [True]


def test_measure_button_is_an_extension_point(screen):
    menu = screen._measure_menu
    assert [a.objectName() for a in menu.actions()] == ["MakeMasksMeasure2D", "MakeMasksMeasure3D"]
    hits = []
    action = screen._add_measure_section("Another section…", lambda: hits.append(1),
                                        object_name="MakeMasksMeasureOther")
    assert menu.actions()[-1] is action
    action.trigger()
    assert hits == [1]


def test_open_file_opens_a_2d_field_with_its_folder(screen, folder):
    screen._go_to_index(0)
    assert screen._on_pick_file(str(folder / "field_2.tif")) is True
    assert screen._image_files[screen._current_index] == "field_2.tif"


def test_brush_defaults_to_one_pixel(screen):
    assert DEFAULT_BRUSH_RADIUS == 1
    assert screen._brush_slider.value() == 1
    assert screen._canvas.brush_radius == 1
    assert screen._brush_size_label.text() == "1 px"


def test_canvas_overlay_puts_the_index_first(screen):
    assert screen._canvas.field_label == "1 / 3 · field_0.tif"
    screen._on_next()
    assert screen._canvas.field_label == "2 / 3 · field_1.tif"
    assert screen._canvas.field_label_rect() is not None


@pytest.mark.parametrize("name,key", [
    ("Title", MAKE_MASKS_TITLE_FOLD_KEY),
    ("Top bar", MAKE_MASKS_TOP_BAR_FOLD_KEY),
    ("Bottom bar", MAKE_MASKS_BOTTOM_BAR_FOLD_KEY),
])
def test_collapse_states_persist(qtbot, qt_theme_applied, folder, name, key):
    from spacr.qt.preferences import get_folded_panels

    first = _screen(qtbot, folder)
    splitter = (first._chrome_splitter if name == "Title"
                else first._image_container)
    assert not splitter.is_collapsed(name)
    splitter.toggle_pane(name, by_user=True)
    assert splitter.is_collapsed(name)
    assert get_folded_panels().get(key) is True
    second = _screen(qtbot, folder)
    again = (second._chrome_splitter if name == "Title"
             else second._image_container)
    assert again.is_collapsed(name)
    again.toggle_pane(name, by_user=True)
    assert not get_folded_panels().get(key)
    for built in (first, second):
        built._magnifier.close()
        built.close_folded()


def test_settings_hide_behind_their_arrow(screen):
    assert screen.settings_shown()
    screen._set_settings_shown(False)
    assert not screen.settings_shown()
    screen._set_settings_shown(True)
    assert screen.settings_shown()


@pytest.mark.parametrize("orientation,size", [
    (Qt.Horizontal, (12, 300)), (Qt.Vertical, (900, 12)),
    (Qt.Horizontal, (13, 301)), (Qt.Vertical, (901, 13)),
])
def test_collapse_arrow_is_three_quarters_and_centred(orientation, size):
    from spacr.qt.widgets import collapse_arrow

    rect = QRect(0, 0, *size)
    tab, points, _thickness = collapse_arrow._collapse_arrow_geometry(
        rect, orientation, True)
    centre = QRectF(rect).center()
    assert abs(tab.center().x() - centre.x()) < 1e-9
    assert abs(tab.center().y() - centre.y()) < 1e-9
    across = size[0] if orientation == Qt.Horizontal else size[1]
    along = max(24, across * 3)
    if orientation == Qt.Horizontal:
        assert tab.width() == pytest.approx(across * 0.75)
        assert tab.height() == pytest.approx(along * 0.75)
    else:
        assert tab.height() == pytest.approx(across * 0.75)
        assert tab.width() == pytest.approx(along * 0.75)
    xs = [p.x() for p in points]
    ys = [p.y() for p in points]
    if orientation == Qt.Horizontal:
        assert (min(xs) + max(xs)) / 2 == pytest.approx(centre.x())
        assert (min(ys) + max(ys)) / 2 == pytest.approx(centre.y())
    else:
        assert (min(xs) + max(xs)) / 2 == pytest.approx(centre.x())
        assert (min(ys) + max(ys)) / 2 == pytest.approx(centre.y())
    assert collapse_arrow.HANDLE_SCALE == 0.75


def test_vvvv_controls_left_the_modules(screen):
    names = {w.objectName() for w in screen.findChildren(QPushButton)}
    assert not names & {"MakeMasksVvvvExport", "MakeMasksVvvvExportOnSave"}
    from spacr.qt.screens.settings_model import (_APP_CATEGORY_SPECS,
                                                 _APP_HIDDEN_KEYS)
    for app in ("mask", "measure", "timelapse"):
        assert "export_vvvv" in _APP_HIDDEN_KEYS[app]
    for app, spec in _APP_CATEGORY_SPECS.items():
        for _title, keys in spec:
            assert "export_vvvv" not in keys, app


def test_the_preference_drives_the_export(screen, monkeypatch, tmp_path):
    from spacr.qt import preferences

    written = []
    monkeypatch.setattr(screen, "_on_export_vvvv",
                        lambda root, **kw: written.append(root) or root)
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    preferences._set_export_vvvv(False)
    assert screen._export_vvvv_on_save() is None and written == []
    preferences._set_export_vvvv(True)
    assert preferences._vvvv_export_enabled()
    root = screen._export_vvvv_on_save()
    assert root and root.endswith("make_masks") and written == [root]
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: False)
    assert not preferences._vvvv_export_enabled()
    preferences._set_export_vvvv(False)


def test_runs_take_export_vvvv_from_the_preference():
    import inspect

    from spacr.qt.screens.app_screen import AppScreen

    source = inspect.getsource(AppScreen._on_run)
    assert 'settings["export_vvvv"] = _vvvv_export_enabled()' in source


def test_preferences_plugins_tab_has_the_switch(qtbot, qt_theme_applied,
                                                 monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QCheckBox, QDialogButtonBox
    from spacr.qt import preferences
    from spacr.qt.widgets import ambient

    store = QSettings(str(tmp_path / "prefs.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: store)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer",
                        lambda self: None)
    monkeypatch.setattr(preferences, "_get_show_alpha_features", lambda: True)
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    check = dialog.findChild(QCheckBox, "PreferencesExportVvvv")
    assert check is not None and not check.isChecked()
    assert check.text() == "Export to vvvv"
    check.setChecked(True)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences._get_export_vvvv() is True
    assert preferences._vvvv_export_enabled() is True


def test_side_column_reads_shortcuts_console_chat_system(screen):
    """Like every module: console, its AI chat box with the usage row, System."""
    column = screen._shortcut_panel
    order = [column.widget(i) for i in range(column.count())]
    assert order == [screen._shortcut_card, screen._console_section,
                     screen._system_card]
    assert [p.name for p in column.panes()] == ["Shortcut list", "Console",
                                                "System"]
    console = screen._masks_console.console
    assert console._chat_row.isVisibleTo(console)
    assert console._usage_row.objectName() == "AiUsageRow"
    assert console._chat_row.isAncestorOf(console._usage_row)
    chat_top = console._chat_row.mapTo(screen, QPoint(0, 0)).y()
    scroll_top = console._console_box.mapTo(screen, QPoint(0, 0)).y()
    assert chat_top > scroll_top, "the chat box sits under the console"
    system = screen._system_card
    assert system.title_label.text().endswith("System")
    assert [b for b in (system.ram, system.gpu, system.vram, system.cpu)]
    system.apply({"ram": 40.0, "cpu": 12.0, "gpu": 0.0, "vram": 0.0})


@pytest.mark.parametrize("name", ["Shortcut list", "Console", "System"])
def test_each_side_panel_folds_and_the_fold_is_remembered(
        qtbot, qt_theme_applied, folder, name):
    first = _screen(qtbot, folder)
    column = first._shortcut_panel
    assert not column.is_collapsed(name)
    column.pane(name).folder.set_shut(True, by_user=True)
    assert column.is_collapsed(name)
    second = _screen(qtbot, folder)
    assert second._shortcut_panel.is_collapsed(name)
    second._shortcut_panel.pane(name).folder.set_shut(False, by_user=True)
    for built in (first, second):
        built._magnifier.close()
        built.close_folded()
