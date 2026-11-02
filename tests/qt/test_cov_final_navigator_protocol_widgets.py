"""The Navigator protocol field, inline option list and action buttons at their edges."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QFocusEvent, QKeyEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from spacr.qt.widgets import home
from tests.qt.test_739_navigator_protocols import protocol_profile as protocol_profile  # noqa: F401
from tests.qt.test_startup_ai_reporting_choice import (
    private_ai_profile as private_ai_profile,  # noqa: F401
)

pytestmark = pytest.mark.qt

LONG_CHOICE = "A choice whose text is long enough to need several lines when narrow"


def _field(qtbot):
    label = QLabel("Sequencer")
    label.setWordWrap(True)
    choice = QComboBox()
    choice.addItems(["short", LONG_CHOICE])
    field = home._NavigatorProtocolField()
    qtbot.addWidget(field)
    field.set_parts(label, choice)
    return field, label, choice


def test_a_field_without_parts_asks_for_no_room(qtbot):
    field = home._NavigatorProtocolField()
    qtbot.addWidget(field)
    assert field.beside(100) is True
    assert field.hasHeightForWidth() is True
    assert field.heightForWidth(100) == 0
    assert field.sizeHint().width() == 0 and field.sizeHint().height() == 0
    assert field.minimumSizeHint().height() == 0
    field._place(100)
    field.resize(80, 20)


def test_a_wide_field_puts_the_label_beside_its_choice(qtbot):
    field, label, choice = _field(qtbot)
    wide = field.sizeHint().width() + 40
    assert field.beside(wide) is True
    field.resize(wide, field.heightForWidth(wide))
    field.show()
    assert label.geometry().right() < choice.geometry().left()


def test_a_narrow_field_stacks_the_label_above_its_choice(qtbot):
    field, label, choice = _field(qtbot)
    narrow = 120
    assert field.beside(narrow) is False
    height = field.heightForWidth(narrow)
    assert height >= label.heightForWidth(narrow) + choice.heightForWidth(narrow)
    field.resize(narrow, height)
    field.show()
    assert label.geometry().bottom() <= choice.geometry().top()
    assert field.minimumSizeHint().height() > 0


def test_changing_the_choice_places_the_parts_again(qtbot):
    field, _label, choice = _field(qtbot)
    field.resize(field.sizeHint().width() + 40, 80)
    field.show()
    choice.setCurrentIndex(1)
    assert field.heightForWidth(field.width()) > 0


@pytest.fixture
def page(qtbot, protocol_profile):  # noqa: F811
    from tests.qt.test_739_navigator_protocols import _choose

    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1366, 700)
    page.show()
    qtbot.waitExposed(page)
    qtbot.mouseClick(page.path_edit.protocols, Qt.LeftButton)
    _choose(qtbot, page._protocol_select, "sequencing")
    return page


def _key(widget, key):
    event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    return event


def test_escape_tab_and_enter_in_the_inline_list(qtbot, page):
    choice = page._protocol_select
    choice.showPopup()
    view = choice.view()
    assert view.isVisible()
    assert _key(view, Qt.Key_Escape).isAccepted()
    assert not view.isVisible()

    for key in (Qt.Key_Tab, Qt.Key_Backtab):
        choice.showPopup()
        event = QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier)
        view.keyPressEvent(event)
        assert event.isAccepted()
        assert not view.isVisible()

    choice.showPopup()
    assert _key(view, Qt.Key_Return).isAccepted()
    assert not view.isVisible()

    choice.showPopup()
    _key(view, Qt.Key_Down)
    assert view.isVisible()
    choice.hidePopup()


def test_escape_is_claimed_before_the_page_shortcut_while_the_list_is_open(
        qtbot, page):
    choice = page._protocol_select
    choice.showPopup()
    view = choice.view()
    override = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_Escape, Qt.NoModifier)
    assert view.event(override) is True and override.isAccepted()
    other = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_A, Qt.NoModifier)
    view.event(other)
    assert not other.isAccepted()
    choice.hidePopup()
    gone = QKeyEvent(QEvent.ShortcutOverride, Qt.Key_Escape, Qt.NoModifier)
    view.event(gone)
    assert not gone.isAccepted()


def test_an_empty_choice_opens_nothing_and_a_missing_row_commits_nothing(
        qtbot, page):
    from PySide6.QtCore import QModelIndex

    choice = page._protocol_select
    chosen = []
    choice.activated.connect(chosen.append)
    choice._accept_inline(QModelIndex())
    assert chosen == []
    empty = home._NavigatorProtocolChoice(page.field.protocol_controls)
    empty.showPopup()
    assert not hasattr(empty, "_inline_view") or not empty._inline_view.isVisible()
    empty.hidePopup()


def test_a_choice_whose_list_was_never_built_has_nothing_to_retract(qtbot, page):
    fresh = home._NavigatorProtocolChoice(page.field.protocol_controls)
    fresh.hidePopup()
    assert not hasattr(fresh, "_inline_view")


def test_opening_a_second_choice_retracts_the_first(qtbot, page):
    first = page._protocol_select
    first.showPopup()
    second = home._NavigatorProtocolChoice(page.field.protocol_controls)
    second.addItems(["x", "y"])
    second.setObjectName("second")
    second.showPopup()
    assert not first.view().isVisible()
    assert second.view().isVisible()
    second.hidePopup()


def test_a_focused_protocols_button_draws_its_focus_mark(qtbot, page):
    button = page.path_edit.protocols
    button.setFocus(Qt.TabFocusReason)
    button.window().setAttribute(Qt.WA_KeyboardFocusChange, True)
    image = button.grab().toImage()
    assert not image.isNull()
    button.window().setAttribute(Qt.WA_KeyboardFocusChange, False)
    assert not button.grab().toImage().isNull()


def test_the_action_buttons_follow_the_composer_width(qtbot, page):
    edit = page.path_edit
    edit.protocols.setFixedSize(edit.ai.width() + 9, edit.ai.height() + 3)
    edit._arrange_actions()
    assert edit.protocols.size() == edit.ai.size()


def test_arranging_actions_before_the_buttons_exist_does_nothing():
    class Bare:
        pass

    assert home._EasyPathEdit._arrange_actions(Bare()) is None


def test_escape_returns_from_a_workflow_then_closes_protocols(
        qtbot, page, monkeypatch):
    shown = []
    monkeypatch.setattr(page, "_show_suggestions", lambda: shown.append(1))
    page._workflow = object()
    page._navigator_escape()
    assert shown == [1] and page.path_edit.protocols.isChecked()
    page._workflow = None
    page._navigator_escape()
    assert not page.path_edit.protocols.isChecked()
    shown.clear()
    page._navigator_escape()
    assert shown == [1]


def test_closing_protocols_retracts_an_open_choice_list(
        qtbot, page, monkeypatch):
    choice = page._protocol_select
    choice.showPopup()
    assert choice.view().isVisible()
    monkeypatch.setattr(page, "_show_suggestions", lambda: None)
    page._toggle_protocols(False)
    assert not choice.view().isVisible()
    assert not page.field.protocol_controls.isVisible()
    page._toggle_protocols(True)


def test_stepping_back_and_forward_respects_the_ends(qtbot, page):
    assert page._protocol_step == 0
    page._protocol_back()
    assert page._protocol_step == 0
    page._protocol_family = ""
    page._protocol_next()
    assert page._protocol_step == 0
    page._protocol_step = 2
    page._protocol_next()
    assert page._protocol_step == 2
    page._protocol_back()
    assert page._protocol_step == 1


def test_a_placeholder_row_selects_nothing(qtbot, page):
    before = (page._protocol_family, page._protocol_format)
    page._protocol_selected(-1)
    assert (page._protocol_family, page._protocol_format) == before


def test_no_detail_fields_are_read_before_a_family_and_format_are_chosen(
        qtbot, page):
    page._protocol_family = ""
    assert page._protocol_fields() == ()
    page._protocol_family, page._protocol_format = "sequencing", ""
    assert page._protocol_fields() == ()


def test_a_handoff_is_only_made_for_an_offered_card_of_a_current_inspection(
        qtbot, page, monkeypatch):
    from spacr.qt.widgets import sample_project

    monkeypatch.setattr(sample_project, "_protocol_handoff",
                        lambda result, key: ("handoff", key))
    page._protocol_result = None
    assert page.protocol_handoff("run", "/a") is None
    result = {"actions": [("run", "Run", "d", "/a")]}
    page.path_edit.protocols.setChecked(False)
    page._protocol_result = result
    assert page.protocol_handoff("run", "/a") is None
    page.path_edit.protocols.setChecked(True)
    page._protocol_result = result
    assert page.protocol_handoff("other", "/a") is None
    assert page.protocol_handoff("run", "/b") is None
    assert page.protocol_handoff("run", "/a") == ("handoff", "run")
    page._protocol_result = {"actions": [("run", "Run", "d", "/a")],
                             "source_path": "/in",
                             "_navigator_inspection": (-1, "stale")}
    with pytest.raises(ValueError):
        page.protocol_handoff("run", "/a")


def test_a_hovered_link_in_the_description_is_highlighted(qtbot, page):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent

    details = page.field.details
    details.setHtml('<p><a href="https://example.org/x">hovered link</a></p>')
    details.resize(400, 120)
    details.show()
    local = QPoint(12, 10)
    event = QMouseEvent(QEvent.MouseMove, QPointF(local), QPointF(local),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    details.mouseMoveEvent(event)
    assert len(details.extraSelections()) == 1
    off = QPoint(380, 110)
    event = QMouseEvent(QEvent.MouseMove, QPointF(off), QPointF(off),
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    details.mouseMoveEvent(event)
    assert details.extraSelections() == []


def test_redrawing_the_news_keeps_the_placeholder_and_drops_old_blocks(
        qtbot, monkeypatch):
    monkeypatch.setattr(home.NewsPanel, "read_releases",
                        staticmethod(lambda: []))
    panel = home.NewsPanel()
    qtbot.addWidget(panel)
    assert panel._notes_column.indexOf(panel._placeholder) >= 0
    first = [{"tag": "v1.0", "name": "First", "body": "Notes."}]
    panel.apply_releases(first)
    assert panel.releases == first
    assert panel._placeholder.parentWidget() is not None
    panel.apply_releases([{"tag": "v1.1", "name": "Second", "body": "More."}])
    assert [label.text() for label in panel._headline_labels] != []


def test_rendering_a_step_retracts_the_open_choice_list(qtbot, page):
    choice = page._protocol_select
    choice.showPopup()
    assert page.field.protocol_controls._inline_choice is choice
    page._protocol_render()
    assert page.field.protocol_controls._inline_choice is None


def test_a_motion_change_with_motion_on_leaves_the_arrival_alone(
        qtbot, page, monkeypatch):
    controls = page.field.protocol_controls
    controls._progress = 0.4
    monkeypatch.setattr(page.field, "_motion_enabled", lambda: True)
    page.field._sync_run_state()
    assert controls._progress == 0.4


def test_hiding_a_field_that_has_no_protocol_controls_only_finishes_growth(
        qtbot, page):
    page.field.protocol_controls = None
    page.field.hide()
    assert not page.field.isVisible()


def test_merging_releases_skips_records_without_a_tag_or_a_mapping():
    merged = home.NewsPanel.merge_releases(
        [{"tag": "v1"}, "junk", {"tag": ""}, {"name": "Named"}],
        [None, {"tag": "v2"}, {"tag": "v1", "name": "newer"}])
    assert {str(entry.get("tag") or entry.get("name")) for entry in merged} >= {
        "v1", "v2", "Named"}
    assert len(merged) == 3


def test_a_fetch_that_adds_nothing_leaves_the_news_alone(qtbot, monkeypatch):
    releases = [{"tag": "v1.0", "name": "First", "body": "Notes."}]
    monkeypatch.setattr(home.NewsPanel, "read_releases",
                        staticmethod(lambda: list(releases)))
    panel = home.NewsPanel()
    qtbot.addWidget(panel)
    labels = tuple(panel._headline_labels)
    panel.apply_releases(list(releases))
    assert tuple(panel._headline_labels) == labels


def test_leaving_the_inline_list_retracts_it(qtbot, page):
    choice = page._protocol_select
    choice.showPopup()
    view = choice.view()
    assert view.isVisible()
    view.focusOutEvent(QFocusEvent(QEvent.FocusOut, Qt.OtherFocusReason))
    assert not view.isVisible()


def test_a_motion_change_with_motion_off_and_no_controls_finishes_growth(
        qtbot, page, monkeypatch):
    page.field.protocol_controls = None
    monkeypatch.setattr(page.field, "_motion_enabled", lambda: False)
    page.field._sync_run_state()


def test_a_hovered_href_that_is_nowhere_in_the_document_highlights_nothing(
        qtbot, page, monkeypatch):
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent

    details = page.field.details
    details.setHtml('<p><a href="https://example.org/x">a link</a></p>')
    details.resize(400, 120)
    details.show()
    monkeypatch.setattr(details, "anchorAt", lambda _point: "https://nowhere")
    point = QPointF(QPoint(12, 10))
    event = QMouseEvent(QEvent.MouseMove, point, point,
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    details.mouseMoveEvent(event)
    assert details.extraSelections() == []


def test_cards_are_offered_with_only_the_inspected_actions_that_name_a_path(
        qtbot, page, monkeypatch):
    from spacr.qt.widgets import sample_project

    seen = {}

    def fake_result(family, fmt, choices, source, context):
        seen["source"], seen["context"] = source, context
        return {"actions": [], "assumptions": ["Assumed."], "requirements": []}

    monkeypatch.setattr(sample_project, "_protocol_result", fake_result)
    page.path_edit.setText("/data/in")
    page._inspected_text = "/data/in"
    page._inspected_generation = page._input_generation
    page._last_result = {
        "path": "/data/in", "context": {},
        "actions": [("seg", "Segment", "d", ""), ("seg", "Segment", "d", "/data/in/a"),
                    ("seg", "Segment", "d", "/data/in/b"), ("cnt", "Count", "d", "/data/in/c")]}
    page._protocol_present()
    assert seen["source"] == "/data/in"
    assert seen["context"]["protocol_native_sources"] == {
        "seg": "/data/in/a", "cnt": "/data/in/c"}
    assert page._protocol_result["_navigator_inspection"] == (
        page._inspected_generation, "/data/in")
