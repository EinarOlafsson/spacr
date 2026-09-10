"""The organelle-preset machinery, and two guards that outlive their widgets.

Round 4 closed the settings panel's layout and tooltip arms. What was left is
almost all one feature: the ORGANELLE TYPE PRESET -- the picker that fills a
slot's morphology, method and thresholds from a named cell-biology type and
then gets out of the user's way. Every arm below is a state a user reaches:

* A PRESET THAT RECOMMENDS NOTHING. A type name the table does not know (a
  hand-edited CSV) must leave the slot alone rather than raise into the panel.
* A DIAMETER CONTROL THAT IS NOT A LINE EDIT. Diameter waits for
  ``editingFinished`` so the panel does not rearrange itself between two
  keystrokes; a control without that signal has to fall back to its ordinary
  change signal, and so does one whose ``editingFinished`` will not connect.
* AN EDIT THE PRESET DOES NOT OWN. ``overwrite=False`` -- what a diameter
  change does -- may only rewrite values the preset itself last wrote.
* A SLOT WITH NOWHERE TO PUT THE VALUE. An imported preset for a slot that
  has neither a widget nor a declared home is dropped, not remembered as
  owned.
* A FILE BEING POURED IN. Both preset handlers stand down while
  ``apply_settings_dict`` is mid-flight.
* A WIDGET WHOSE C++ HALF HAS GONE, which is what a screen teardown leaves
  behind, and the two ``hasattr``/``callable`` guards that let a panel
  without the dependency-rule pass still apply the analysis-unit lock.

Offscreen Qt, no network, no modal dialogs.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

import shiboken6                                                  # noqa: E402
from PySide6.QtWidgets import QComboBox, QLineEdit                # noqa: E402

from spacr.qt.screens import settings_model as SM                 # noqa: E402


def _organelle_panel(*, diameter="20", type_name="vesicular",
                     morphology="spots"):
    """A panel holding only slot one's three preset-relevant controls.

    ``build_sections`` on a real module builds hundreds of widgets and none
    of them changes what the preset code reads, which is exactly these three.
    """
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets = {
        "organelle_type": QLineEdit(type_name),
        "organelle_diameter": QLineEdit(diameter),
        "organelle_morphology": QLineEdit(morphology),
    }
    return widgets


class _Unequal:
    """A value that cannot be compared, as a NumPy array cannot."""

    def __eq__(self, other):
        raise ValueError("truth value of an array is ambiguous")

    __hash__ = None


# ---------------------------------------------------------------------------
# reading one value out of the form
# ---------------------------------------------------------------------------

def test_a_widget_whose_cpp_half_is_gone_reads_back_as_the_default():
    """A screen teardown deletes the C++ objects while Python still holds the
    map; the panel answers with the module default rather than raising."""
    widgets = _organelle_panel()
    widgets._defaults["organelle_diameter"] = 30

    assert widgets._setting_value("organelle_diameter") == 20

    shiboken6.delete(widgets._widgets["organelle_diameter"])

    assert widgets._setting_value("organelle_diameter") == 30


def test_a_comparison_that_cannot_be_made_is_not_an_equal_value(monkeypatch):
    """``_values_equal`` is imported through a seam; when it cannot answer,
    the plain ``==`` is tried, and a value that refuses that too is unequal."""
    from spacr.qt import settings_diff

    widgets = _organelle_panel()
    assert widgets._setting_value_equals("organelle_morphology", "spots")

    def _explode(_a, _b):
        raise ValueError("the truth value of an array is ambiguous")

    monkeypatch.setattr(settings_diff, "_values_equal", _explode)

    # First fallback: Python's own equality still answers.
    assert widgets._setting_value_equals("organelle_morphology", "spots")
    assert not widgets._setting_value_equals("organelle_morphology", "ring")
    # Second fallback: nothing can answer, so nothing is owned.
    assert widgets._setting_value_equals(
        "organelle_morphology", _Unequal()) is False


# ---------------------------------------------------------------------------
# a type the preset table does not know
# ---------------------------------------------------------------------------

def test_a_type_name_the_table_does_not_know_recommends_nothing():
    """Settings CSVs are hand-edited. A typo must cost the recommendation,
    not the panel."""
    known = _organelle_panel(type_name="vesicular")
    unknown = _organelle_panel(type_name="vesicular_")

    assert known._organelle_recommendations("organelle")[
        "organelle_morphology"] == "ring"
    assert unknown._organelle_recommendations("organelle") == {}


# ---------------------------------------------------------------------------
# which signal the diameter control is followed by
# ---------------------------------------------------------------------------

def test_a_line_edit_diameter_is_followed_only_once_it_is_committed():
    """Typing must not rearrange the panel; leaving the field must."""
    widgets = SM.SettingsWidgets("measure")
    edit = QLineEdit("30")
    widgets._widgets = {"organelle_diameter": edit}
    seen = []
    # Bound by `partial(self._on_organelle_diameter_changed, role)` at
    # connect time, so the stand-in has to be in place before connecting.
    widgets._on_organelle_diameter_changed = lambda *args: seen.append(args)

    widgets._connect_object_visibility_signals()
    edit.setText("40")
    assert seen == []

    edit.editingFinished.emit()
    assert seen == [("organelle",)]


def test_a_diameter_control_with_no_commit_signal_follows_its_value():
    """A combo has no ``editingFinished``; choosing from it is the commit.

    The same panel has no ``organelle_type`` control at all, which is the
    other half of this pass: a slot may offer a diameter and no picker.
    """
    widgets = SM.SettingsWidgets("measure")
    combo = QComboBox()
    combo.addItem("10", 10)
    combo.addItem("20", 20)
    widgets._widgets = {"organelle_diameter": combo}
    seen = []
    widgets._on_organelle_diameter_changed = lambda *args: seen.append(args)

    assert not hasattr(combo, "editingFinished")
    widgets._connect_object_visibility_signals()
    combo.setCurrentIndex(1)

    assert [args[0] for args in seen] == ["organelle"]


def test_a_commit_signal_that_refuses_the_connection_falls_back_to_the_value():
    """A dead C++ half raises from ``connect``. Losing the commit signal must
    not lose the diameter rule -- the ordinary change signal takes it."""

    class _Refuses:
        def connect(self, _handler):
            raise RuntimeError("Signal source has been deleted")

    class _Records:
        def __init__(self):
            self.handlers = []

        def connect(self, handler):
            self.handlers.append(handler)

    class _Diameter:
        """A diameter control offering both signals, one of them broken."""

        def __init__(self):
            self.editingFinished = _Refuses()
            self.textChanged = _Records()

    widgets = SM.SettingsWidgets("measure")
    control = _Diameter()
    widgets._widgets = {"organelle_diameter": control}
    seen = []
    widgets._on_organelle_diameter_changed = lambda *args: seen.append(args)

    widgets._connect_object_visibility_signals()

    assert len(control.textChanged.handlers) == 1
    control.textChanged.handlers[0]("41")
    assert seen == [("organelle", "41")]


# ---------------------------------------------------------------------------
# what a preset is allowed to overwrite
# ---------------------------------------------------------------------------

def test_a_diameter_change_leaves_a_value_the_preset_never_wrote():
    """``overwrite=False`` is what a diameter edit does: it may only move
    values the preset still owns, so a hand-set morphology survives."""
    unowned = _organelle_panel()
    unowned._apply_organelle_recommendations("organelle", overwrite=False)

    assert unowned._widgets["organelle_morphology"].text() == "spots"
    assert unowned._organelle_preset_owned["organelle"] == {}

    # The same slot, once the preset owns the value it wrote there.
    owned = _organelle_panel()
    owned._organelle_preset_owned["organelle"] = {
        "organelle_morphology": "spots"}
    owned._apply_organelle_recommendations("organelle", overwrite=False)

    assert owned._widgets["organelle_morphology"].text() == "ring"
    assert owned._organelle_preset_owned["organelle"][
        "organelle_morphology"] == "ring"


def test_a_deliberate_type_choice_overwrites_and_then_owns_the_values():
    widgets = _organelle_panel()
    widgets._apply_organelle_recommendations("organelle", overwrite=True)

    assert widgets._widgets["organelle_morphology"].text() == "ring"
    assert "organelle_morphology" in widgets._organelle_preset_owned[
        "organelle"]


# ---------------------------------------------------------------------------
# both handlers stand down while a settings file is being poured in
# ---------------------------------------------------------------------------

def test_a_type_change_mid_import_does_not_populate_the_slot():
    """``apply_settings_dict`` sets one widget at a time; acting on the type
    before the diameter beside it has landed writes the wrong morphology."""
    widgets = _organelle_panel()
    widgets._applying_settings = True

    widgets._on_organelle_type_changed("organelle")
    assert widgets._widgets["organelle_morphology"].text() == "spots"

    widgets._applying_settings = False
    widgets._on_organelle_type_changed("organelle")
    assert widgets._widgets["organelle_morphology"].text() == "ring"


def test_a_diameter_change_mid_import_does_not_repopulate_the_slot():
    widgets = _organelle_panel()
    widgets._organelle_preset_owned["organelle"] = {
        "organelle_morphology": "spots"}
    widgets._applying_settings = True

    widgets._on_organelle_diameter_changed("organelle")
    assert widgets._widgets["organelle_morphology"].text() == "spots"

    widgets._applying_settings = False
    widgets._on_organelle_diameter_changed("organelle")
    assert widgets._widgets["organelle_morphology"].text() == "ring"


# ---------------------------------------------------------------------------
# a sparse imported mapping
# ---------------------------------------------------------------------------

def test_an_imported_diameter_alone_re_reads_the_size_dependent_values():
    """A file that names a diameter and no type is asking the preset it
    already has to be re-read at the new size."""
    resized = _organelle_panel()
    resized._organelle_preset_owned["organelle"] = {
        "organelle_morphology": "spots"}
    resized.apply_organelle_presets_from_mapping({"organelle_diameter": 20})

    assert resized._widgets["organelle_morphology"].text() == "ring"

    # A file naming neither the type nor the diameter of this slot asks for
    # nothing, and the slot is left exactly as it was.
    untouched = _organelle_panel()
    untouched._organelle_preset_owned["organelle"] = {
        "organelle_morphology": "spots"}
    untouched.apply_organelle_presets_from_mapping({"organelle_min_area": 5})

    assert untouched._widgets["organelle_morphology"].text() == "spots"


def test_an_imported_preset_with_nowhere_to_land_is_not_recorded_as_owned():
    """A recommendation the panel can neither show nor store must not be
    remembered as a value the preset wrote -- the next diameter change would
    then believe it had put something there that is not there."""
    homeless = SM.SettingsWidgets("measure")
    homeless._widgets = {}
    # No `number_of_organelles`, so `set_hidden_value` refuses a slot key
    # that has no widget: the panel does not own the organelle count.
    homeless._defaults = {"organelle_type": "vesicular",
                          "organelle_diameter": 20}

    homeless.apply_organelle_presets_from_mapping(
        {"organelle_type": "vesicular"})

    assert homeless._organelle_preset_owned["organelle"] == {}

    # The same import into a panel that DOES own the count keeps every value.
    housed = SM.SettingsWidgets("measure")
    housed._widgets = {}
    housed._defaults = {"organelle_type": "vesicular",
                        "organelle_diameter": 20,
                        "number_of_organelles": 1}

    housed.apply_organelle_presets_from_mapping(
        {"organelle_type": "vesicular"})

    assert housed._organelle_preset_owned["organelle"][
        "organelle_morphology"] == "ring"
    assert housed._defaults["organelle_morphology"] == "ring"


# ---------------------------------------------------------------------------
# the two guards on the panel's own methods
# ---------------------------------------------------------------------------

class _NoDependencyRules(SM.SettingsWidgets):
    """A panel that does not carry the dependency-rule pass.

    ``_refresh_analysis_unit_lock`` re-runs the other refreshers over the
    controls it just released, and guards that call with ``hasattr``. The
    property below is how a panel can genuinely lack the attribute -- the
    method is defined on ``SettingsWidgets`` itself, so nothing else makes
    that guard false.
    """

    @property
    def _refresh_setting_dependencies(self):
        raise AttributeError("_refresh_setting_dependencies")


def _unit_locked_panel(cls=SM.SettingsWidgets):
    widgets = cls("measure")
    mode = QComboBox()
    mode.addItem("regression", "regression")
    mode.addItem("guide_permutation", "guide_permutation")
    mode.setCurrentIndex(1)
    inference = QComboBox()
    inference.addItem("auto", "auto")
    inference.addItem("parametric", "parametric")
    widgets._widgets = {"analysis_unit": QLineEdit("cell"),
                        "analysis_mode": mode, "inference": inference}
    return widgets, mode, inference


def test_the_analysis_unit_lock_applies_without_the_dependency_rules():
    """The lock is this method's own job; re-asserting the OTHER rules
    afterwards is a courtesy, and a panel without them still gets locked."""
    ordinary, mode, inference = _unit_locked_panel()
    reasserted = []
    ordinary._refresh_setting_dependencies = lambda: reasserted.append(1)

    ordinary._refresh_analysis_unit_lock()

    assert mode.currentText() == "regression" and not mode.isEnabled()
    assert inference.currentText() == "parametric"
    assert reasserted == [1]

    bare, bare_mode, bare_inference = _unit_locked_panel(_NoDependencyRules)
    assert not hasattr(bare, "_refresh_setting_dependencies")

    bare._refresh_analysis_unit_lock()

    assert bare_mode.currentText() == "regression"
    assert not bare_mode.isEnabled()
    assert bare_inference.currentText() == "parametric"
    assert bare._unit_locked == {"analysis_mode", "inference"}


def test_a_panel_with_no_writer_shows_no_resolved_value():
    """``_show_the_value_it_will_have`` writes through ``set_value_for_key``
    and checks it is callable first, because the greyed control it is fixing
    is only worth fixing on a panel that can be written to at all."""
    current = {"inference": "nonparametric", "analysis_mode": "regression"}

    writable, mode, _inference = _unit_locked_panel()
    mode.setCurrentIndex(0)
    writable._show_the_value_it_will_have("analysis_mode", current)

    # `inference='nonparametric'` decides the mode at run time, so the panel
    # shows what the fit will actually do rather than what the box said.
    assert mode.currentText() == "guide_permutation"

    unwritable, blocked, _blocked_inference = _unit_locked_panel()
    blocked.setCurrentIndex(0)
    # A panel whose writer has been taken away -- what a stripped-down or
    # read-only host leaves behind.
    unwritable.set_value_for_key = None
    unwritable._show_the_value_it_will_have("analysis_mode", current)

    assert blocked.currentText() == "regression"
