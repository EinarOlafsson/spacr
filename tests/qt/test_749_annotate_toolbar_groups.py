"""The Annotate toolbar is built from labelled groups that wrap as wholes.

Before, about twenty commands shared one wrapping row, so a button's place
depended on the window width and the Zoom, Clear page sat right after the
navigation buttons, and the one-time setup commands sat among the commands
used on every page. These tests pin the grouped layout:

* every command belongs to the group it was put in, in a fixed order;
* the Danger zone comes last, right-aligned, never beside Navigate;
* Load test data and Generate annotation database live on the Setup menu;
* each command keeps its group and its place in it at two window widths
  and at 100 % and 200 % Zoom, and a group is never split across lines.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import QSettings, QSize
from PySide6.QtWidgets import QApplication, QMenu, QPushButton

GROUP_ORDER = ["navigate", "source", "data", "view", "similar", "model",
               "bulk", "danger"]


@pytest.fixture(autouse=True)
def _private_preferences(tmp_path, monkeypatch):
    """Keep the Zoom changes of these tests out of the real preferences."""
    from spacr.qt import preferences

    path = tmp_path / "annotate-toolbar.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    yield
    preferences.set_font_scale(1.0)
    preferences.apply_preferences_to_app(QApplication.instance())


def _screen_at(qtbot, scale: float, width: int):
    """An Annotate screen drawn at ``scale`` Zoom and ``width`` pixels wide."""
    from spacr.qt import gui_scale, preferences
    from spacr.qt.screens.annotate import AnnotateScreen

    preferences.set_font_scale(scale)
    preferences.apply_preferences_to_app(QApplication.instance())
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    gui_scale._resize_actual(screen, QSize(width, 1400))
    screen.show()
    qtbot.waitExposed(screen)
    toolbar = screen._toolbar_groups["navigate"].parentWidget()
    qtbot.waitUntil(lambda: toolbar.height() >= toolbar.heightForWidth(
        toolbar.width()), timeout=5000)
    return screen


def _expected_members(screen) -> dict:
    """The commands each group must hold, left to right."""
    return {
        "navigate": [screen._btn_prev, screen._btn_next, screen._btn_skip,
                     screen._sort_order.parentWidget(), screen._page_label],
        "source": [screen._btn_open, screen._btn_setup, screen._btn_settings],
        "data": [screen._btn_count, screen._btn_coverage,
                 screen._btn_browse_db],
        "model": [screen._btn_retrain, screen._btn_suggest,
                  screen._btn_suggest_cancel, screen._suggest_progress,
                  screen._suggest_stage, screen._btn_suggest_model,
                  screen._btn_curve],
        "bulk": [screen._btn_train, screen._btn_auto,
                 screen._btn_annotate_page],
        "danger": [screen._btn_clear_page, screen._btn_clear],
    }


def _layout_record(screen) -> dict:
    """Each visible command's group and its place among the group's shown commands."""
    record = {}
    for key, group in screen._toolbar_groups.items():
        if group.isHidden():
            continue
        shown = [w for w in group.members if not w.isHidden()]
        for index, widget in enumerate(shown):
            name = (widget.objectName() or getattr(widget, "text", str)()
                    or type(widget).__name__)
            record[name] = (key, index)
    return record


def test_the_groups_hold_their_commands_in_order(qtbot):
    screen = _screen_at(qtbot, 1.0, 1366)
    groups = screen._toolbar_groups
    assert list(groups) == GROUP_ORDER
    for key, members in _expected_members(screen).items():
        assert groups[key].members == members, key
        assert groups[key].property("toolbarGroup") == key
    assert groups["navigate"].caption.text() == "Navigate"
    assert groups["model"].caption.text() == "Label model"
    assert groups["danger"].caption.text() == "Danger zone"
    flow = groups["navigate"].parentWidget().layout()
    order = [flow.itemAt(i).widget() for i in range(flow.count())]
    assert order == [groups[key] for key in GROUP_ORDER]


def test_the_danger_zone_is_never_beside_navigation(qtbot):
    for scale in (1.0, 2.0):
        for width in (1366, 900):
            screen = _screen_at(qtbot, scale, width)
            groups = screen._toolbar_groups
            shown = [g for g in groups.values() if not g.isHidden()]
            assert shown[0] is groups["navigate"]
            assert shown[-1] is groups["danger"]
            assert shown.index(groups["danger"]) - 1 > 0
            nav, danger = groups["navigate"].geometry(), groups["danger"].geometry()
            toolbar = groups["navigate"].parentWidget()
            assert danger.right() == toolbar.contentsRect().right(), (
                scale, width)
            same_line = nav.top() == danger.top()
            if same_line:
                assert danger.left() - nav.right() > 2 * 16, (scale, width)
            else:
                beneath = danger.top() > nav.bottom()
                overlaps = (danger.left() <= nav.right()
                            and nav.left() <= danger.right())
                next_line = danger.top() - nav.bottom() < nav.height()
                assert beneath and not (overlaps and next_line), (
                    scale, width, nav, danger)
            screen.close()


def test_the_setup_commands_are_on_the_setup_menu(qtbot, monkeypatch):
    from spacr.qt.screens.annotate import AnnotateScreen

    calls = []
    monkeypatch.setattr(AnnotateScreen, "_choose_the_test_data",
                        lambda self, **kw: calls.append("test data"))
    monkeypatch.setattr(AnnotateScreen, "_on_generate_annotation_db",
                        lambda self: calls.append("generate"))
    screen = _screen_at(qtbot, 1.0, 1366)
    button = screen._btn_setup
    assert isinstance(button, QPushButton)
    assert button.accessibleName() == "Setup"
    menu = button.menu()
    assert isinstance(menu, QMenu) and menu is screen._setup_menu
    assert menu.actions() == [screen._btn_test_data, screen._btn_generate,
                              screen._btn_pair_review]
    assert screen._btn_test_data.text() == "Load test data"
    assert screen._btn_generate.text() == "Generate annotation database…"
    toolbar_widgets = [w for g in screen._toolbar_groups.values()
                       for w in g.members]
    assert screen._btn_test_data not in toolbar_widgets
    assert screen._btn_generate not in toolbar_widgets
    assert screen._btn_pair_review.text() == "Review touching pairs…"
    screen._btn_test_data.trigger()
    screen._btn_generate.trigger()
    assert calls == ["test data", "generate"]


def test_a_command_keeps_its_group_and_place_at_every_width_and_zoom(qtbot):
    records = {}
    for scale in (1.0, 2.0):
        for width in (1366, 900):
            screen = _screen_at(qtbot, scale, width)
            records[(scale, width)] = _layout_record(screen)
            for group in screen._toolbar_groups.values():
                if group.isHidden():
                    continue
                shown = [w for w in group.members if not w.isHidden()]
                tops = {w.geometry().top() for w in shown}
                assert len(tops) == 1 or all(
                    abs(w.geometry().center().y()
                        - shown[0].geometry().center().y()) <= 2
                    for w in shown), (scale, width, group.key)
                lefts = [w.geometry().left() for w in shown]
                assert lefts == sorted(lefts), (scale, width, group.key)
            placed = [g.geometry() for g in screen._toolbar_groups.values()
                      if not g.isHidden()]
            for i, a in enumerate(placed):
                for b in placed[i + 1:]:
                    assert not a.intersects(b), (scale, width, a, b)
            screen.close()
    first = records[(1.0, 1366)]
    assert len(first) >= 15
    for key, record in records.items():
        assert record == first, key


def _toolbar_flow(qtbot, widths):
    """A bare ``_ToolbarFlow`` holding one plain 20 px high box per width.

    The boxes are plain widgets, not buttons: an earlier test in the same
    process can leave the theme's stylesheet on the application, and its
    ``QPushButton`` rule (8 px padding, 1 px border, 22 px minimum height)
    raised a fixed 20 px button to 40 px, so the layout arithmetic these
    tests pin was measured against boxes twice as tall as they asked for.
    """
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens.annotate import _ToolbarFlow

    host = QWidget()
    qtbot.addWidget(host)
    flow = _ToolbarFlow(host, spacing=10, line_spacing=6)
    boxes = []
    for width in widths:
        box = QWidget(host)
        box.setFixedSize(width, 20)
        flow.addWidget(box)
        boxes.append(box)
    return host, flow, boxes


def test_an_empty_toolbar_flow_needs_no_height_and_places_nothing(qtbot):
    from PySide6.QtCore import QRect

    _host, flow, _ = _toolbar_flow(qtbot, [])
    assert flow._do_layout(QRect(0, 0, 300, 100), False) == 0


def test_a_group_that_already_fills_the_line_is_not_moved(qtbot):
    from PySide6.QtCore import QRect

    _host, flow, (only,) = _toolbar_flow(qtbot, [200])
    assert flow._do_layout(QRect(0, 0, 200, 50), False) == 20
    assert only.geometry().right() == 199 and only.geometry().x() == 0
    # a group wider than the line stays at the left edge and does not slide right
    assert flow._do_layout(QRect(0, 0, 100, 50), False) == 20
    assert only.geometry().x() == 0


def test_the_last_group_is_pushed_to_the_right_edge_of_its_line(qtbot):
    from PySide6.QtCore import QRect

    _host, flow, (first, last) = _toolbar_flow(qtbot, [120, 100])
    assert flow._do_layout(QRect(5, 7, 300, 100), False) == 20
    assert first.geometry().x() == 5
    assert last.geometry().right() == 5 + 300 - 1


def test_a_toolbar_flow_whose_c_half_is_gone_lays_nothing_out(qtbot, monkeypatch):
    from PySide6.QtCore import QRect

    from spacr.qt.screens import annotate

    _host, flow, (only,) = _toolbar_flow(qtbot, [80])
    only.setGeometry(1, 2, 80, 20)
    monkeypatch.setattr(annotate, "_alive", lambda layout: False)
    assert flow._do_layout(QRect(0, 0, 300, 100), False) == 0
    assert only.geometry().getRect() == (1, 2, 80, 20)
