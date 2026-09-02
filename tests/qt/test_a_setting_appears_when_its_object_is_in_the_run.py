"""Instruction 246, section 2: a setting is visible when its object is here.

"object settings should become visable as needed. same with cell, nucleus,
pathogen. so if cell, nucleus, pathogen channel has a number and is not None
the repective settings visability are toggledd on."

Three rules, and one promise that runs underneath all three:

  * the CHANNEL is the switch for nucleus and pathogen, and the mask plane is
    the switch on a module that reads masks somebody else made;
  * cell is the deliberate exception: its settings always remain available;
  * an organelle SLOT is switched the same way, one slot at a time;
  * the slot's TYPE narrows it further -- a punctate organelle shows the
    punctate controls and not the network or ring ones;
  * and HIDDEN IS NOT DELETED. A hidden setting keeps the value the user
    typed into it, is still written to the settings file, and comes back with
    its old answer when the channel comes back.

Everything below is measured off the built panel -- the row on the form, and
in one case the screen's own read-back -- rather than off the rule that
decided it.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _screen(qtbot, app_key: str):
    """Build a visibility-rule harness containing the rows it will toggle.

    Production screens now omit optional-object families and organelle slots
    until a committed form-shaping value asks the window to rebuild them.
    These tests exercise the row-visibility rule itself, so seed that rebuild
    shape explicitly, then put the switches back to their unset state before
    making assertions. The separate count/rebuild tests cover the optimized
    production path.
    """
    from spacr.qt.screens.app_screen import AppScreen

    current = {"number_of_organelles": 1}
    if app_key == "mask":
        current.update({
            "number_of_organelles": 4,
            "nucleus_channel": 1,
            "pathogen_channel": 2,
        })
    before = AppScreen.values_the_next_screen_is_built_for
    AppScreen.values_the_next_screen_is_built_for = current
    try:
        screen = AppScreen(app_key)
    finally:
        AppScreen.values_the_next_screen_is_built_for = before
    qtbot.addWidget(screen)
    model = screen._settings_model
    if app_key == "mask":
        model.set_value_for_key("nucleus_channel", None)
        model.set_value_for_key("pathogen_channel", None)
        model.refresh_object_visibility()
    # THE FIRST PASS IS SCHEDULED, not immediate: the rows do not exist until
    # the screen has laid them out, which happens after the settings model
    # has handed them over. See `SettingsWidgets.build_sections`.
    qtbot.wait(1)
    return screen, model


def _row_shown(screen, key: str) -> bool:
    """Whether ``key`` is a row on the form, label and all.

    The ROW, not the widget: the widget of a hidden row is still there
    holding its value, which is the whole point of hiding rather than
    dropping it.
    """
    from PySide6.QtWidgets import QFormLayout

    # THE WIDGET IN THE ROW IS NOT ALWAYS THE FIELD. A setting that takes a
    # Cellpose checkpoint sits in a little holder next to its "Model zoo…"
    # button, and it is the HOLDER the form knows -- so an identity match
    # against the field alone reported `organelle_model_name` as being "on no
    # form at all" once that button was added. The panel's own
    # `_set_row_visible` walks up for the same reason; this walks with it, so
    # the check still measures the row rather than quietly passing.
    field = screen._settings_model._widgets[key]
    candidates = []
    node = field
    for _ in range(3):
        if node is None:
            break
        candidates.append(node)
        node = node.parentWidget()
    for section in screen._settings_sections:
        form = getattr(section, "_form", None)
        if not isinstance(form, QFormLayout):
            continue
        for index in range(form.rowCount()):
            item = form.itemAt(index, QFormLayout.FieldRole)
            if item is not None and any(item.widget() is c for c in candidates):
                return bool(form.isRowVisible(index))
    raise AssertionError(f"{key} is on no form at all")


def _shown_suffixes(screen, role: str) -> set:
    """The suffixes of ``role``'s settings that are on the form."""
    return {key[len(role) + 1:]
            for key in screen._settings_model._widgets
            if key.startswith(f"{role}_") and _row_shown(screen, key)}


def _set(model, key, value) -> None:
    """Put a value in a control and let the panel answer for it."""
    assert model.set_value_for_key(key, value), f"{key} has no control"
    from spacr.organelle_types import organelle_role_of

    role = organelle_role_of(key)
    suffix = key[len(role) + 1:] if role else ""
    if suffix in {"type", "morphology"}:
        # The combo's own choice signal is the production user path.
        return
    if suffix == "diameter":
        # A typed size is committed on Enter/focus-out, not per keystroke.
        committed = getattr(model._widgets[key], "editingFinished", None)
        if committed is not None:
            committed.emit()
            return
    model.refresh_object_visibility()


# ---------------------------------------------------------------------------
# The channel is the switch
# ---------------------------------------------------------------------------

def test_cell_is_never_gated_by_its_channel(qtbot):
    screen, model = _screen(qtbot, "mask")

    # Instruction 300 superseded the earlier all-object rule for cell: it is
    # what every other object is measured against, so a fresh form must keep
    # its family available even while its channel is empty.
    assert model.collect()["cell_channel"] is None
    assert _row_shown(screen, "cell_channel") is True
    assert _row_shown(screen, "cell_diameter") is True
    assert _row_shown(screen, "cell_cellprob_threshold") is True

    _set(model, "cell_channel", 1)
    assert _row_shown(screen, "cell_diameter") is True
    assert _row_shown(screen, "cell_cellprob_threshold") is True
    # ONE OBJECT AT A TIME. Turning the cell on says nothing about a nucleus.
    assert _row_shown(screen, "nucleus_diameter") is False


def test_each_optional_object_is_switched_by_its_own_channel(qtbot):
    screen, model = _screen(qtbot, "mask")

    for role in ("nucleus", "pathogen"):
        _set(model, f"{role}_channel", 2)
        assert _row_shown(screen, f"{role}_diameter") is True
        others = [r for r in ("nucleus", "pathogen") if r != role]
        for other in others:
            assert _row_shown(screen, f"{other}_diameter") is False
        assert _row_shown(screen, "cell_diameter") is True
        _set(model, f"{role}_channel", None)
        assert _row_shown(screen, f"{role}_diameter") is False


def test_the_switch_itself_is_never_hidden(qtbot):
    """Hiding the channel would leave nothing to turn the object back on."""
    screen, model = _screen(qtbot, "mask")

    for role in ("cell", "nucleus", "pathogen", "organelle", "organelleb"):
        assert _row_shown(screen, f"{role}_channel") is True


def test_the_screen_agrees_that_cell_remains_on_the_form(qtbot):
    """Read back through the screen's own answer, not through the rule."""
    screen, model = _screen(qtbot, "mask")
    screen.show()
    qtbot.wait(1)

    assert screen.setting_row_is_visible("cell_channel") is True
    assert screen.setting_row_is_visible("cell_diameter") is True
    _set(model, "cell_channel", 0)
    assert screen.setting_row_is_visible("cell_diameter") is True


# ---------------------------------------------------------------------------
# Hidden, not deleted
# ---------------------------------------------------------------------------

def test_a_hidden_setting_keeps_its_value_and_is_still_saved(qtbot):
    screen, model = _screen(qtbot, "mask")
    everything = set(model.collect())

    _set(model, "nucleus_channel", 1)
    _set(model, "nucleus_diameter", 77)
    _set(model, "nucleus_channel", None)

    assert _row_shown(screen, "nucleus_diameter") is False
    saved = model.collect()
    # A settings CSV must not lose a key because the panel was not showing it
    # when Save was pressed.
    assert set(saved) == everything
    assert saved["nucleus_diameter"] == 77


def test_changing_the_channel_back_brings_the_old_answers_with_it(qtbot):
    screen, model = _screen(qtbot, "mask")

    _set(model, "nucleus_channel", 1)
    _set(model, "nucleus_diameter", 77)
    _set(model, "nucleus_channel", None)
    _set(model, "nucleus_channel", 3)

    assert _row_shown(screen, "nucleus_diameter") is True
    assert model.collect()["nucleus_diameter"] == 77


def test_importing_a_settings_file_brings_its_objects_back(qtbot):
    """The bulk apply is the path an imported CSV takes."""
    screen, model = _screen(qtbot, "mask")
    assert _row_shown(screen, "pathogen_diameter") is False

    screen.apply_settings_dict({"pathogen_channel": 2})

    assert _row_shown(screen, "pathogen_diameter") is True


def test_bulk_import_rebuilds_once_and_applies_every_slot_value(qapp, qtbot):
    """A shape signal must not continue the import on the retired screen."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._on_nav_selected("mask")
    qapp.processEvents()
    original = window._screens["mask"]
    loaded = {
        "number_of_organelles": 1,
        "organelle_channel": 2,
        "organelle_type": "filamentous",
        # An explicit advanced override wins over the type recommendation.
        "organelle_morphology": "spots",
        "organelle_min_area": 77,
    }

    assert original.apply_settings_dict(loaded) == len(loaded)
    qapp.processEvents()

    rebuilt = window._screens["mask"]
    assert rebuilt is not original
    values = rebuilt._settings_model.collect()
    assert {key: values[key] for key in loaded} == loaded


def test_a_legacy_slot_mapping_infers_the_count_before_it_builds(qapp, qtbot):
    """A pre-count CSV keeps its meaning on today's count-zero panel."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._on_nav_selected("mask")
    qapp.processEvents()
    original = window._screens["mask"]
    loaded = {
        "organelleb_channel": 4,
        "organelleb_type": "tubular",
        "organelleb_min_size": 91,
    }

    original.apply_settings_dict(loaded)
    qapp.processEvents()

    rebuilt = window._screens["mask"]
    values = rebuilt._settings_model.collect()
    assert rebuilt is not original
    assert values["number_of_organelles"] == 2
    assert {key: values[key] for key in loaded} == loaded


def test_a_bulk_type_import_deferred_by_a_run_keeps_preset_provenance(
        qapp, qtbot):
    """The post-run form still knows which preset values were omitted."""
    from spacr.qt.app import MainWindow

    class RunningThread:
        running = True

        def isRunning(self):
            return self.running

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._on_nav_selected("mask")
    qapp.processEvents()
    original = window._screens["mask"]
    thread = RunningThread()
    original._thread = thread
    original._worker = object()

    original.apply_settings_dict({
        "number_of_organelles": 1,
        "organelle_channel": 2,
        "organelle_type": "filamentous",
    })
    # A later, non-shaping next-run edit has no rebuild signal of its own.
    # It still has to beat the older deferred snapshot.
    original._settings_model._widgets["cell_channel"].setText("3")
    qapp.processEvents()
    assert window._screens["mask"] is original

    thread.running = False
    original._clear_thread_refs()
    qtbot.waitUntil(lambda: window._screens["mask"] is not original)
    values = window._screens["mask"]._settings_model.collect()
    assert values["organelle_type"] == "filamentous"
    assert values["organelle_morphology"] == "network"
    assert values["organelle_method"] == "ridge"
    assert values["cell_channel"] == 3


def test_an_imported_type_waiting_for_a_channel_keeps_its_preset(qapp, qtbot):
    """Off-form recommendations survive until the slot is activated."""
    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    window._on_nav_selected("mask")
    qapp.processEvents()
    screen = window._screens["mask"]

    screen.apply_settings_dict({
        "number_of_organelles": 1,
        "organelle_type": "filamentous",
    })
    qapp.processEvents()
    screen = window._screens["mask"]
    waiting = screen._settings_model.collect()
    assert waiting["organelle_morphology"] == "network"
    assert waiting["organelle_method"] == "ridge"

    channel = screen._settings_model._widgets["organelle_channel"]
    channel.setText("2")
    channel.editingFinished.emit()
    qapp.processEvents()
    active = window._screens["mask"]._settings_model.collect()
    assert active["organelle_type"] == "filamentous"
    assert active["organelle_morphology"] == "network"
    assert active["organelle_method"] == "ridge"


# ---------------------------------------------------------------------------
# One slot at a time, and the type narrows it
# ---------------------------------------------------------------------------

def test_an_organelle_slot_is_switched_by_its_own_channel(qtbot):
    screen, model = _screen(qtbot, "mask")

    _set(model, "organelleb_channel", 4)

    assert _row_shown(screen, "organelleb_type") is True
    assert _row_shown(screen, "organelleb_min_size") is True
    # Slot 1 was not asked about and is still off.
    assert _row_shown(screen, "organelle_type") is False
    assert _row_shown(screen, "organelle_min_area") is False


def test_the_type_decides_which_of_a_slots_controls_are_shown(qtbot):
    """The maintainer's own example, asserted as a set.

    "organelle 1 type is puncta and organelle type 2 is vesicular, then only
    puncta and vesicular settings are visable."

    Punctate is spots. Vesicular SPLITS ON SIZE -- a transport vesicle is a
    dot and a vacuole is a ring, and both are Vesicular -- so at the default
    30 px diameter it is a ring, which is what `organelle_types` resolves and
    what the panel therefore shows.
    """
    screen, model = _screen(qtbot, "mask")
    _set(model, "organelle_channel", 3)
    _set(model, "organelleb_channel", 4)
    _set(model, "organelle_type", "punctate")
    _set(model, "organelleb_type", "vesicular")

    punctate = _shown_suffixes(screen, "organelle")
    vesicular = _shown_suffixes(screen, "organelleb")

    # What each shows that the other does not, exactly.
    assert punctate - vesicular == {
        "tophat_radius", "watershed_spots",
        "dog_sigma_low", "dog_sigma_high",
    }
    assert vesicular - punctate == {
        "ring_sigma_inner", "ring_sigma_outer",
        "ring_min_prominence", "ring_fill_method",
    }
    # NEITHER IS A NETWORK. No ridge filter, no hysteresis, no skeleton.
    network_only = {"ridge_filter", "ridge_sigmas", "network_threshold",
                    "hysteresis_low", "hysteresis_high", "skeletonize",
                    "unet_model_path", "unet_threshold"}
    assert not punctate & network_only
    assert not vesicular & network_only


@pytest.mark.parametrize("type_name,expected", [
    ("filamentous", {"ridge_filter", "ridge_sigmas", "network_threshold",
                     "hysteresis_low", "hysteresis_high", "skeletonize",
                     "morph_radius", "unet_model_path", "unet_threshold"}),
    ("cisternal", {"morph_radius", "fill_holes"}),
    ("toroidal", {"ring_sigma_inner", "ring_sigma_outer",
                  "ring_min_prominence", "ring_fill_method",
                  "log_min_sigma", "log_max_sigma", "log_num_sigma",
                  "log_threshold"}),
])
def test_each_type_shows_its_own_morphologys_controls(qtbot, type_name,
                                                      expected):
    """The detection settings a slot shows are the ones its morphology reads.

    Compared against the whole claimed alphabet rather than against a couple
    of names, so a control leaking in from another morphology fails here.
    """
    from spacr.qt.screens.settings_model import _MORPHOLOGY_OWNED

    screen, model = _screen(qtbot, "mask")
    _set(model, "organelle_channel", 3)
    _set(model, "organelle_type", type_name)

    shown = _shown_suffixes(screen, "organelle")
    assert shown & _MORPHOLOGY_OWNED == expected
    from spacr.organelle_types import preset_for
    recommended = preset_for(type_name, model.collect()["organelle_diameter"])
    assert model.collect()["organelle_morphology"] == recommended[
        "organelle_morphology"]
    assert model.collect()["organelle_method"] == recommended[
        "organelle_method"]


def test_an_explicit_morphology_wins_in_the_live_populated_form(qtbot):
    """The rows and execution values agree after an advanced override."""
    from spacr.qt.screens.settings_model import _MORPHOLOGY_OWNED

    screen, model = _screen(qtbot, "mask")
    _set(model, "organelle_channel", 3)
    _set(model, "organelle_type", "punctate")
    _set(model, "organelle_morphology", "network")
    _set(model, "organelle_method", "ridge")

    values = model.collect()
    assert values["organelle_morphology"] == "network"
    assert values["organelle_method"] == "ridge"
    shown = _shown_suffixes(screen, "organelle") & _MORPHOLOGY_OWNED
    assert "ridge_filter" in shown
    assert "tophat_radius" not in shown


def test_a_slot_left_on_custom_follows_its_own_morphology(qtbot):
    """'custom' recommends nothing, so the morphology control is the answer."""
    from spacr.qt.screens.settings_model import _MORPHOLOGY_OWNED

    screen, model = _screen(qtbot, "mask")
    _set(model, "organelle_channel", 3)
    assert model.collect()["organelle_type"] == "custom"

    _set(model, "organelle_morphology", "irregular")
    assert _shown_suffixes(screen, "organelle") & _MORPHOLOGY_OWNED == {
        "morph_radius", "fill_holes"}

    _set(model, "organelle_morphology", "spots")
    assert _shown_suffixes(screen, "organelle") & _MORPHOLOGY_OWNED == {
        "tophat_radius", "watershed_spots",
        "log_min_sigma", "log_max_sigma", "log_num_sigma", "log_threshold",
        "dog_sigma_low", "dog_sigma_high"}


def test_a_setting_no_morphology_claims_is_shown_for_all_of_them(qtbot):
    """The block size is legal under every morphology, so it never goes."""
    screen, model = _screen(qtbot, "mask")
    _set(model, "organelle_channel", 3)

    for type_name in ("punctate", "filamentous", "cisternal", "toroidal"):
        _set(model, "organelle_type", type_name)
        assert _row_shown(screen, "organelle_adaptive_block_size") is True
        assert _row_shown(screen, "organelle_model_name") is True


# ---------------------------------------------------------------------------
# The count
# ---------------------------------------------------------------------------

def test_lowering_the_count_hides_whole_slots_and_keeps_their_values(qtbot):
    screen, model = _screen(qtbot, "mask")
    _set(model, "organellec_channel", 5)
    _set(model, "organellec_diameter", 41)
    assert _row_shown(screen, "organellec_channel") is True

    _set(model, "number_of_organelles", 2)

    # A slot the run does not have is not a slot with its channel showing.
    assert _row_shown(screen, "organellec_channel") is False
    assert _row_shown(screen, "organellec_diameter") is False
    assert _row_shown(screen, "organelleb_channel") is True
    # Its answers ride along and come back with it.
    assert model.collect()["organellec_diameter"] == 41
    _set(model, "number_of_organelles", 4)
    assert _row_shown(screen, "organellec_diameter") is True
    assert model.collect()["organellec_diameter"] == 41


# ---------------------------------------------------------------------------
# The module that has no channel
# ---------------------------------------------------------------------------

def test_measure_is_switched_by_the_mask_plane(qtbot):
    """Measure reads masks somebody else made, so it offers no channel."""
    screen, model = _screen(qtbot, "measure")
    assert "cell_channel" not in model._widgets

    # It ships with the cell, nucleus and pathogen planes filled in and the
    # organelle planes empty.
    assert _row_shown(screen, "cell_min_size") is True
    assert _row_shown(screen, "organelle_min_area") is False
    assert _row_shown(screen, "organelle_mask_dim") is True

    _set(model, "organelle_mask_dim", 7)
    assert _row_shown(screen, "organelle_min_area") is True

    _set(model, "organelle_mask_dim", None)
    assert _row_shown(screen, "organelle_min_area") is False
    # And the plane it was on is still there to put back.
    assert _row_shown(screen, "organelle_mask_dim") is True


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def test_a_setting_whose_switch_is_elsewhere_is_left_alone():
    """Hiding one would leave a control the user cannot bring back."""
    from spacr.qt.screens.settings_model import keys_hidden_by_their_object

    assert keys_hidden_by_their_object({"nucleus_diameter"}, {}) == set()
    assert keys_hidden_by_their_object(
        {"nucleus_diameter", "nucleus_channel"}, {}) == {
            "nucleus_diameter"}


def test_a_boolean_in_a_channel_does_not_name_a_plane():
    """`int(False)` is plane zero, which would switch an object on."""
    from spacr.qt.screens.settings_model import keys_hidden_by_their_object

    panel = {"nucleus_channel", "nucleus_diameter"}
    assert keys_hidden_by_their_object(
        panel, {"nucleus_channel": False}) == {"nucleus_diameter"}
    assert keys_hidden_by_their_object(
        panel, {"nucleus_channel": 0}) == set()
    # And the string a text box hands back is read as the number it holds.
    assert keys_hidden_by_their_object(
        panel, {"nucleus_channel": "2"}) == set()
    assert keys_hidden_by_their_object(
        panel, {"nucleus_channel": ""}) == {"nucleus_diameter"}


def test_the_count_is_only_obeyed_where_it_can_be_changed():
    """A panel with no count control must not hide a slot by one."""
    from spacr.qt.screens.settings_model import keys_hidden_by_their_object

    panel = {"organellec_channel", "organellec_diameter"}
    assert keys_hidden_by_their_object(panel, {"number_of_organelles": 2}) == {
        "organellec_diameter"}
    assert keys_hidden_by_their_object(
        panel | {"number_of_organelles"},
        {"number_of_organelles": 2}) == panel


def test_every_control_the_morphology_table_claims_is_a_real_setting():
    """A table naming a key that does not exist hides nothing and says so."""
    from spacr.qt.screens.settings_model import (
        _MORPHOLOGY_OWNED,
        resolve_default_settings,
    )

    defaults = resolve_default_settings("mask")
    missing = sorted(suffix for suffix in _MORPHOLOGY_OWNED
                     if f"organelle_{suffix}" not in defaults)
    assert missing == []


def test_the_morphology_table_names_only_settings_segmentation_reads():
    """Anti-drift against `spacr.object`, which is the authority.

    The classical worker's own key list is what each morphology's segmenter
    is handed. A suffix this table claims that is not in it is a control
    being hidden for a reason the pipeline does not have -- with the two
    U-Net keys excepted, because U-Net is a GPU backend and never reaches
    the classical worker at all.
    """
    from spacr.object import _extract_classical_settings
    from spacr.qt.screens.settings_model import _MORPHOLOGY_OWNED

    # The list is compiled to a tuple constant, so the names are one level
    # down rather than beside the docstring.
    read = {key[len("organelle_"):]
            for const in _extract_classical_settings.__code__.co_consts
            for key in (const if isinstance(const, tuple) else ())
            if isinstance(key, str) and key.startswith("organelle_")}
    assert read, "the classical key list moved out of the function body"
    assert _MORPHOLOGY_OWNED - read == {"unet_model_path", "unet_threshold"}


# ---------------------------------------------------------------------------
# The heading over the rows
# ---------------------------------------------------------------------------

def _headings(screen) -> dict:
    """``title -> whether the heading still has anything under it``."""
    from spacr.qt.screens.settings_model import section_shows_anything

    return {section.title(): section_shows_anything(section)
            for section in screen._settings_sections}


def test_a_heading_with_every_row_hidden_reports_itself_empty(qtbot):
    """A heading that opens onto nothing is a smaller wall, but a wall."""
    screen, model = _screen(qtbot, "mask")

    assert _headings(screen)["ORGANELLE 2"] is False
    _set(model, "organelleb_channel", 4)
    assert _headings(screen)["ORGANELLE 2"] is True


def test_a_heading_that_holds_a_switch_is_never_empty(qtbot):
    """The channel is on the form, so its heading has to be too.

    Mask files every object's channel under "Input & Metadata" -- the
    switches sit together, above the settings they switch -- and a switch is
    never hidden, because hiding it would leave nothing to turn its object
    back on with. So that heading holds something whatever the run has.
    """
    screen, model = _screen(qtbot, "mask")

    assert model.collect()["organelle_channel"] is None
    assert _headings(screen)["INPUT & METADATA"] is True


def test_an_objects_own_heading_empties_with_the_object(qtbot):
    """And the heading that holds only its SETTINGS goes when it does.

    The same answer for all four objects: with no channel naming a plane,
    "Organelle Segmentation" opens onto nothing exactly as "Nucleus
    Segmentation" and "Pathogen Segmentation" do, and naming a channel
    fills it. The count is not what props it open -- `number_of_organelles`
    belongs to no slot, so a card kept alive by it would be a card about an
    object the run does not have.
    """
    screen, model = _screen(qtbot, "mask")

    empty = _headings(screen)
    assert empty["NUCLEUS SEGMENTATION"] is False
    assert empty["PATHOGEN SEGMENTATION"] is False
    assert empty["ORGANELLE SEGMENTATION"] is False

    _set(model, "organelle_channel", 3)
    assert _headings(screen)["ORGANELLE SEGMENTATION"] is True


def test_a_heading_with_no_setting_rows_is_not_judged_empty(qtbot):
    """An umbrella or a prose panel was never carrying rows to lose."""
    from spacr.qt.screens.settings_model import section_shows_anything
    from spacr.qt.widgets.section import Section

    bare = Section("Nothing here")
    qtbot.addWidget(bare)
    assert section_shows_anything(bare) is True
