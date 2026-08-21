"""Instruction 158 wired into the real backend picker.

The unit tests beside this one prove the panel behaves and the offers are
right. This file is about the wiring, because a shared control that nothing
calls is exactly what instruction 141 shipped once already:
``describe_backends`` was written, tested and CALLED BY NOTHING IN spacr/qt.
So every assertion here goes through ``_RegressionBackendField`` as the
settings panel builds it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt          # noqa: E402
from PySide6.QtGui import QKeyEvent, QMouseEvent        # noqa: E402

from spacr.qt.screens import settings_model             # noqa: E402
from spacr.qt.widgets.availability_panel import (       # noqa: E402
    AvailabilityPanel)


@pytest.fixture
def field(qtbot):
    widget = settings_model._RegressionBackendField(
        default="statsmodels (CPU)", regression_type="mixed")
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    AvailabilityPanel.instance().dismiss()
    AvailabilityPanel.instance().set_install_handler(None)


def _greyed_rows(field):
    model = field.combo.model()
    return [i for i in range(field.combo.count())
            if not model.item(i).isEnabled()]


def test_every_greyed_row_is_unselectable_not_merely_disabled(field):
    """`setEnabled(False)` alone leaves `ItemIsSelectable` set, and a
    model-level selection could still land on it."""
    rows = _greyed_rows(field)
    assert rows, "nothing is greyed; this environment cannot check the rule"
    model = field.combo.model()
    for row in rows:
        assert not (model.item(row).flags() & Qt.ItemIsSelectable), row


def test_a_greyed_row_keeps_the_tooltip_the_panel_is_hung_off(field):
    model = field.combo.model()
    for row in _greyed_rows(field):
        assert model.item(row).toolTip().strip(), row


def test_an_available_row_stays_selectable_when_the_family_changes(
        field, monkeypatch):
    """Re-judging must RE-ENABLE as well as disable. `pyfixest` is refused
    under 'mixed' and offered under 'ols', and a one-way flag clear would
    leave it dead forever."""
    from spacr import regression_backends

    installed = regression_backends.package_installed
    monkeypatch.setattr(
        regression_backends, "package_installed",
        lambda package: package == "pyfixest" or installed(package))
    labels = [field.combo.itemData(i) for i in range(field.combo.count())]
    row = labels.index("pyfixest (CPU)")
    assert not (field.combo.model().item(row).flags() & Qt.ItemIsSelectable)
    field.set_regression_type("ols")
    item = field.combo.model().item(row)
    assert item.isEnabled()
    assert bool(item.flags() & Qt.ItemIsSelectable)


def test_the_unavailable_entries_are_exactly_the_greyed_rows(field):
    greyed = {field.combo.itemData(row) for row in _greyed_rows(field)}
    assert {entry['title'] for entry in field.unavailable_entries()} == greyed


def test_hovering_a_greyed_row_opens_the_panel_on_that_row(field):
    field.combo.showPopup()
    view = field.combo.view()
    row = _greyed_rows(field)[0]
    index = field.combo.model().index(row, 0)
    rect = view.visualRect(index)
    centre = QPointF(rect.center())
    event = QMouseEvent(QEvent.MouseMove, centre, centre, centre,
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    field._hover_popup_row(view, event)
    panel = AvailabilityPanel.instance()
    assert panel.isVisible()
    assert panel.current_entry()['title'] == field.combo.itemData(row)
    field.combo.hidePopup()


def test_hovering_an_available_row_does_not_open_the_panel(field):
    field.combo.showPopup()
    view = field.combo.view()
    available = [i for i in range(field.combo.count())
                 if i not in _greyed_rows(field)]
    index = field.combo.model().index(available[0], 0)
    centre = QPointF(view.visualRect(index).center())
    event = QMouseEvent(QEvent.MouseMove, centre, centre, centre,
                        Qt.NoButton, Qt.NoButton, Qt.NoModifier)
    field._hover_popup_row(view, event)
    assert AvailabilityPanel.instance().isVisible() is False
    field.combo.hidePopup()


def test_shift_f1_on_the_combo_is_the_keyboard_route(field):
    """The rows are disabled, so nothing about them can be tabbed to and no
    help can be inherited from them. The route has to be explicit."""
    handled = field.eventFilter(
        field.combo,
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_F1, Qt.ShiftModifier))
    panel = AvailabilityPanel.instance()
    assert handled is True
    assert panel.isVisible() and panel.is_pinned()
    assert panel.current_entry() in field.unavailable_entries()


def test_an_ordinary_key_is_left_alone(field):
    handled = field.eventFilter(
        field.combo,
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_F1, Qt.NoModifier))
    assert handled is not True
    assert AvailabilityPanel.instance().isVisible() is False


def test_pressing_install_reaches_the_three_answer_flow(field, monkeypatch):
    seen = {}

    def _fake(parent, offer, **kwargs):
        seen['offer'] = offer
        return "explained"

    monkeypatch.setattr(settings_model, "run_install_offer", _fake)
    field.show_availability_panel("cuml")
    panel = AvailabilityPanel.instance()
    assert panel.current_entry()['key'] == "cuml"
    panel._on_link("install")
    assert seen['offer'] is panel.current_entry()['offer']


def test_a_successful_install_re_judges_the_entries(field, monkeypatch):
    calls = []
    monkeypatch.setattr(settings_model, "run_install_offer",
                        lambda parent, offer, **kw: "installed")
    monkeypatch.setattr(type(field), "refresh",
                        lambda self: calls.append("refreshed"))
    field._run_install_offer(field.unavailable_entries()[0]['offer'])
    assert calls == ["refreshed"]


def test_the_panel_answers_this_field_and_not_a_previous_one(qtbot,
                                                             monkeypatch):
    """The panel is a process-wide singleton. Two fields on screen at once
    must not both answer one press."""
    seen = []
    monkeypatch.setattr(
        settings_model, "run_install_offer",
        lambda parent, offer, **kw: seen.append(parent) or "explained")
    first = settings_model._RegressionBackendField(regression_type="mixed")
    second = settings_model._RegressionBackendField(regression_type="mixed")
    qtbot.addWidget(first)
    qtbot.addWidget(second)
    first.show_availability_panel("cuml")
    second.show_availability_panel("cuml")
    panel = AvailabilityPanel.instance()
    panel._on_link("install")
    assert seen == [second]
    panel.dismiss()
    panel.set_install_handler(None)


def test_the_stale_selection_still_explains_itself_from_the_closed_combo(
        qtbot):
    """141 C keeps a selection that has gone unavailable rather than silently
    re-pointing it, so hovering the closed combo is a real state to be in."""
    widget = settings_model._RegressionBackendField(
        default="pyfixest (CPU)", regression_type="ols")
    qtbot.addWidget(widget)
    widget.set_regression_type("mixed")
    assert widget.get_value() == "pyfixest (CPU)"
    widget._hover_closed_combo()
    panel = AvailabilityPanel.instance()
    assert panel.isVisible()
    assert panel.current_entry()['title'] == "pyfixest (CPU)"
    panel.dismiss()
    panel.set_install_handler(None)


def test_an_available_selection_does_not_open_a_panel_on_hover(field):
    assert field.get_value() == "statsmodels (CPU)"
    field._hover_closed_combo()
    assert AvailabilityPanel.instance().isVisible() is False


def test_leaving_the_popup_closes_it_so_the_panel_can_be_clicked(field):
    """A QComboBox popup is a `Qt.Popup` with an active mouse grab. With it
    still open the first press on the panel is eaten by the grab, so the
    Install link would need two clicks and the first would look inert."""
    field.combo.showPopup()
    view = field.combo.view()
    assert view.isVisible()
    field.show_availability_panel("cuml")
    field.eventFilter(view.viewport(), QEvent(QEvent.Leave))
    assert view.isVisible() is False
    assert AvailabilityPanel.instance().isVisible()


def test_leaving_the_popup_with_no_panel_open_leaves_it_alone(field):
    field.combo.showPopup()
    view = field.combo.view()
    field.eventFilter(view.viewport(), QEvent(QEvent.Leave))
    assert view.isVisible()
    field.combo.hidePopup()
