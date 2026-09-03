"""Propagating from a preview nobody touched must not rewrite the run.

"Propagate settings" answers "should this dialog overwrite my Mask form?".
If the user opens the preview, looks, and presses it without moving a
control, the only honest result is that nothing about the run changes.

Two ways that was false, both of them a control sending a value it was never
given:

* ``adjust_cells`` was propagated and never SEEDED, so it always sent the
  unchecked box it was built with. Mask ships it True, so an untouched
  preview switched off the adjustment of cell masks by the nucleus and
  pathogen masks.
* ``cell_flow_threshold`` was seeded, but the spin box runs -1 to 3 and Mask
  ships 100 -- "accept everything Cellpose proposes". The box clamped to 3
  and propagation handed the 3 back as if it had been chosen.

The channels and the diameter are deliberately NOT in this list: the preview
has to pick planes to run at all, and telling the run which ones were
previewed is what propagation is for.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def opened(qtbot, qt_theme_applied):
    """A Mask screen with the preview switched on, as a user opens it."""
    from PySide6.QtWidgets import QApplication
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    screen._on_preview_switch(True)
    QApplication.processEvents()
    return screen


#: What the preview is entitled to decide for itself, and why.
ALLOWED_TO_DIFFER = {
    "cell_channel", "nucleus_channel", "pathogen_channel",
    "organelle_channel", "organelleb_channel", "organellec_channel",
    "organelled_channel", "cell_diameter",
}


def test_an_untouched_preview_rewrites_nothing_it_was_not_asked_to(opened):
    """The ratchet: anything new in this set is a control inventing an answer."""
    panel = opened._settings_model.collect()
    propagated = opened._live_preview.settings_for_propagation()

    differs = sorted(
        key for key, value in propagated.items()
        if key in panel and panel[key] != value
        and key not in ALLOWED_TO_DIFFER)

    assert not differs, (
        "an untouched preview would change these settings on an untouched "
        f"form: {differs}")


def test_the_cell_adjustment_toggle_arrives_the_way_the_form_holds_it(opened):
    """Seeded, not left at the box's own default."""
    panel = opened._settings_model.collect()

    shown = opened._live_preview._adjust_cells.isChecked()

    assert shown == bool(panel["adjust_cells"])
    assert opened._live_preview.settings_for_propagation()["adjust_cells"] \
        == panel["adjust_cells"]


def test_a_value_the_spin_box_cannot_hold_is_given_back_unchanged(opened):
    """The clamp is the editor's limit, not the user's answer."""
    preview = opened._live_preview
    panel = opened._settings_model.collect()
    shipped = panel["cell_flow_threshold"]

    assert preview._flow.maximum() < shipped, (
        "the spin box can hold the shipped value now, so this test no longer "
        "exercises the clamp -- check whether the range was widened instead")
    assert preview.settings_for_propagation()["cell_flow_threshold"] == shipped


def test_but_a_value_the_user_moves_is_theirs(opened):
    """The remembered original must not outlive the user touching the box."""
    preview = opened._live_preview

    preview._flow.setValue(0.4)

    assert preview.settings_for_propagation()["cell_flow_threshold"] == \
        pytest.approx(0.4)
