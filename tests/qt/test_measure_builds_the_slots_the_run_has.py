"""Measure builds a control per object the RUN has, not per object spaCR names.

Item 284. Opening Measure took 13.2 seconds and froze the event loop for 12.9
of them, against a 10 second budget and a ledger entry of 1.45 s. Counting
rather than timing found it: `MeasurePreviewPanel` built one mask-slice spin
box, one crop-mode toggle and one minimum-area spin box PER OBJECT ROLE, and
`ALL_ROLES` carries 702 organelle slots because `MAX_ORGANELLES` is 702. That
is ~2,117 controls -- 4,113 QWidgets, each spin box dragging a QLineEdit and a
validator and each toggle a QPropertyAnimation -- against 63 settings rows,
where Annotate builds 84 widgets in a quarter of a second. Ten application-wide
event filters then run for every Qt event delivered to every one of them.

`DEFAULT_NUMBER_OF_ORGANELLES` IS ZERO, so in the common case all 702 slots
were controls for something the run does not have.

WHAT THESE TESTS PIN IS THE RELATIONSHIP, not a number. "Fewer than N widgets"
passes on a panel that builds 700 slots when the run declares one; these
assert that the controls are a function of the declared count, so a change
that reconnects the panel to `ALL_ROLES` fails whatever the constant becomes.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.object_roles import ALL_ROLES
from spacr.organelle_types import MAX_ORGANELLES
from spacr.qt.widgets.measure_preview import (
    _AFTER_THE_SLOTS, _BEFORE_THE_SLOTS, _objects_for, CropSettingsDialog,
    MeasurePreviewPanel,
)


@pytest.fixture
def panel(qapp):
    return MeasurePreviewPanel(threaded=False)


def _slot_controls(panel):
    """The three per-object dicts, by the name the panel keeps them under."""
    return {
        "mask slice": panel._mask_dims,
        "minimum area": panel._min_sizes,
        "crop mode": panel._crop_mode_checks,
    }


def _form_captions(form):
    """The captions down one of the crop dialog's forms, in row order."""
    from PySide6.QtWidgets import QFormLayout

    captions = []
    for row in range(form.rowCount()):
        item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
        widget = item.widget() if item is not None else None
        captions.append(widget.text() if hasattr(widget, "text") else "")
    return captions


def _between(captions, first, last):
    """What sits between two captions: a row's NEIGHBOURS are the subject."""
    return captions[captions.index(first) + 1:captions.index(last)]


def _visible_slot_rows(dialog):
    """Which organelle slots the crop-settings dialog is showing rows for."""
    return sorted({role for role, form, widget in dialog._organelle_rows
                   if form.isRowVisible(form.getWidgetPosition(widget)[0])})


def test_the_schema_still_declares_far_more_slots_than_a_run_has():
    """The premise, established rather than assumed.

    Every test below is a report on nothing if `MAX_ORGANELLES` shrinks to
    four: 702 slots is what made the difference between the two readings
    ~2,100 controls rather than a handful.
    """
    assert MAX_ORGANELLES > 100
    assert len(ALL_ROLES) == (len(_BEFORE_THE_SLOTS) + MAX_ORGANELLES
                              + len(_AFTER_THE_SLOTS))


def test_the_default_count_builds_no_slot_control(panel):
    """The regression itself: zero organelles, zero organelle controls."""
    for kind, controls in _slot_controls(panel).items():
        slots = [role for role in controls if role.startswith("organelle")]
        assert slots == [], f"{len(slots)} {kind} controls for no organelle"


@pytest.mark.parametrize("count", [0, 1, 3, 7])
def test_the_controls_are_one_per_object_the_run_has(panel, count):
    """One control per object, and the objects are the run's, not spaCR's."""
    panel.apply_settings({"number_of_organelles": count})

    wanted = _objects_for(count)
    assert tuple(panel._min_sizes) == wanted
    assert tuple(panel._crop_mode_checks) == wanted
    assert tuple(panel._mask_dims) == tuple(
        role for role in wanted if role != "cytoplasm")


@pytest.mark.parametrize("count", [0, 1, 3, 7])
def test_the_combo_lists_what_the_run_has(panel, count):
    """Trap 4: the preview-object box listed all 706 objects."""
    panel.apply_settings({"number_of_organelles": count})

    listed = [panel._object_box.itemText(i)
              for i in range(panel._object_box.count())]
    assert tuple(listed) == _objects_for(count)


def test_one_more_organelle_is_three_more_controls(panel):
    """The slope, so the relationship cannot be satisfied by a constant."""
    panel.apply_settings({"number_of_organelles": 2})
    before = sum(len(c) for c in _slot_controls(panel).values())

    panel.apply_settings({"number_of_organelles": 3})

    assert sum(len(c) for c in _slot_controls(panel).values()) == before + 3


def test_raising_the_count_builds_a_control_that_is_wired_for_refresh(panel):
    """TRAP 1. A lazily created widget nobody connected is a silent dead
    control: it looks right and changes nothing.

    `_connect_controls` wires the mask slices and the size floors to
    `_on_setting_changed`, which re-crops; a slot built later must be wired
    the same way or tuning it leaves the preview showing the old crops.
    """
    panel._data = np.zeros((8, 8, 8), dtype=np.uint16)
    panel.set_organelle_count(2)
    refreshed = []
    panel.refresh = lambda *a, **k: refreshed.append(1)

    panel._mask_dims["organelleb"].setValue(3)
    assert refreshed, "a slot's mask slice re-previews nothing"

    refreshed.clear()
    panel._min_sizes["organelleb"].setValue(250)
    assert refreshed, "a slot's minimum area re-previews nothing"


def test_raising_the_count_builds_a_control_that_is_wired_to_propagate(panel):
    """The other half of trap 1: the settings the run reads.

    The crop-mode toggles propagate rather than re-preview, which is the
    wiring `_connect_controls` gives them, so this is the same assertion
    against the other slot.
    """
    pushed = []
    panel.set_propagate_callback(pushed.append)
    panel._propagate_btn.setChecked(True)
    panel.set_organelle_count(2)
    pushed.clear()

    panel._crop_mode_checks["organelleb"].setChecked(True)

    assert pushed, "a slot's crop mode reaches the run's settings not at all"
    assert "organelleb" in pushed[-1]["crop_mode"]


def test_a_slot_built_later_reaches_a_dialog_that_is_already_open(panel):
    """The count is a live setting, so it can rise while the dialog is up.

    A control with no row is as invisible as one that was never built.
    """
    dialog = CropSettingsDialog(panel)
    panel._crop_settings_dialog = dialog
    assert _visible_slot_rows(dialog) == []

    panel.set_organelle_count(2)

    assert _visible_slot_rows(dialog) == ["organelle", "organelleb"]
    shown = {role: widget for role, _form, widget in dialog._organelle_rows}
    assert shown["organelleb"] in (panel._mask_dims["organelleb"],
                                   panel._min_sizes["organelleb"])


def test_a_slot_the_file_carries_beyond_the_count_round_trips(panel):
    """TRAP 3, and the settings grid's rule: CUT FROM THE VIEW, NOT FROM THE
    SETTINGS.

    A file written at seven and opened at two still carries slots three to
    seven. This panel writes its controls back over the run's settings, so a
    slot with no control propagates nothing and lowering the count would
    silently rewrite the user's file.
    """
    written = {"number_of_organelles": 2}
    for index, role in enumerate(
            ("organelle", "organelleb", "organellec", "organelled",
             "organellee", "organellef", "organelleg"), start=1):
        written[f"{role}_mask_dim"] = index
        written[f"{role}_min_area"] = 100 + index

    panel.apply_settings(written)
    read_back = panel.settings_for_propagation()

    lost = {key: value for key, value in written.items()
            if key != "number_of_organelles" and read_back.get(key) != value}
    assert lost == {}, f"a lowered count rewrote the file: {lost}"
    # And the ones past the count are hidden rather than gone.
    assert _visible_slot_rows(CropSettingsDialog(panel)) == [
        "organelle", "organelleb"]


def test_the_count_is_applied_before_the_values_it_creates_controls_for(panel):
    """TRAP 3's ordering. `apply_settings` seeds the slot controls from the
    same dict that declares how many there are; a value written into a slot
    whose control does not exist yet is a value dropped on the floor."""
    panel.apply_settings({"number_of_organelles": 3,
                          "organellec_mask_dim": 11,
                          "organellec_min_area": 4321})

    assert panel._mask_dims["organellec"].value() == 11
    assert panel._min_sizes["organellec"].value() == 4321


def test_opening_measure_does_not_build_a_control_per_declared_slot(qtbot):
    """The module, not the widget: what the benchmark opens.

    Measured in this worktree before the fix: measure 13.23 s, worst
    event-loop stall 12,926 ms, against annotate's 0.25 s / 251 ms.
    """
    from spacr.organelle_types import declared_organelle_roles
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("measure")
    qtbot.addWidget(screen)
    preview = screen._measure_preview
    assert preview is not None, "the Measure screen has no crop preview"

    # WHAT THE SCREEN'S OWN SETTINGS SPEAK FOR. The count says what is
    # SHOWN; a slot whose keys the dict carries keeps a control so that its
    # value round-trips, which is one more than the count at Measure's
    # defaults -- `organelle_min_area` and `organelle_type` are shipped keys.
    def slots_built():
        return [role for role in preview._crop_mode_checks
                if role.startswith("organelle")]

    shipped = screen._settings_model.collect()
    speaks_for = len(declared_organelle_roles(shipped))
    assert slots_built() == [], (
        f"{len(slots_built())} slot controls on a screen that has not even "
        "seeded its preview yet")

    # AND SEEDED, which is what the Live toggle does. Still a function of the
    # settings rather than of the schema: Measure ships `organelle_min_area`
    # and `organelle_type`, so its defaults speak for one slot and get one.
    screen._prime_preview()

    assert len(slots_built()) == speaks_for, (
        f"{len(slots_built())} slot controls for settings that speak for "
        f"{speaks_for}")
    assert speaks_for < MAX_ORGANELLES, (
        f"{speaks_for} of {MAX_ORGANELLES} declared slots built")


def test_a_slot_built_later_lands_inside_the_organelle_block(panel):
    """WHERE the row goes, not merely that it exists.

    A dialog that is already open gets its new rows INSERTED; appending them
    would put "Organelle 1 mask slice" under Timelapse and "Organelle
    minimum area" under "Cytoplasm minimum area", which is a form that reads
    as though the cytoplasm had organelles. The rows either side are the
    assertion, so the test survives a row being added elsewhere in the tab.
    """
    dialog = CropSettingsDialog(panel)
    panel._crop_settings_dialog = dialog

    panel.set_organelle_count(2)

    assert _between(_form_captions(dialog._general_form),
                    "Pathogen mask slice", "Measure cytoplasm") == [
        "Organelle 1 mask slice", "Organelle 2 mask slice"]
    assert _between(_form_captions(dialog._filter_form),
                    "Pathogen minimum area", "Cytoplasm minimum area") == [
        "Organelle minimum area", "Organelleb minimum area"]


def test_a_crop_mode_built_later_lands_inside_the_organelle_block(panel):
    """The same question for the Crop modes group, which is a box of toggles
    rather than a form: appending files Organelle under Cytoplasm."""
    dialog = CropSettingsDialog(panel)
    panel._crop_settings_dialog = dialog

    panel.set_organelle_count(2)

    toggles = [dialog._mode_layout.itemAt(i).widget().text()
               for i in range(dialog._mode_layout.count())]
    assert _between(toggles, "Pathogen", "Cytoplasm") == [
        "Organelle", "Organelleb"]


def test_a_slot_control_is_not_left_floating_over_the_panel(panel):
    """A control is built parented to the PANEL and laid out by the DIALOG,
    so between the two it must be hidden.

    `_build_controls` ends by hiding every control it made for exactly this
    reason; a slot built later and left visible is an unlaid-out spin box
    drawn at the panel's top-left corner, over the crop grid. The fixed
    controls are the comparison, so this cannot pass by hiding everything.
    """
    panel.set_organelle_count(2)

    assert panel._mask_dims["cell"].isHidden(), "the premise has moved"
    showing = [f"{kind} {role}"
               for kind, controls in _slot_controls(panel).items()
               for role in ("organelle", "organelleb")
               if not controls[role].isHidden()]
    assert showing == [], f"{showing} float over the panel"
