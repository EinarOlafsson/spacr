"""The settings panel's remaining defensive arms, driven rather than assumed.

``spacr/qt/screens/settings_model.py`` is the bridge every module screen is
built out of, and round 3 left 115 of its lines and branches unreached. They
are not exotic: each one is a state the panel is already written to survive,
and an arm nobody has run is a guess about what it does.

Five families are pinned here, because each is a real thing a user can do:

* A LAYOUT THAT CLAIMS EVERY KEY. ``categories_for_app`` grows an
  "Additional Settings" bucket for whatever no heading asked for, and drops a
  heading that hiding emptied. Both are decided by the *absence* of a key, so
  neither had ever been driven from the side where nothing is left over.
* A HELP STRING WITH NOTHING TO SAY. A blank category title, a setting with
  no prose, an unhashable language, a widget that is an API dot rather than a
  setting -- every one of them has a written-down answer that no test had
  read back.
* A CONTROL THAT IS NOT THE ONE THE PANEL EXPECTS. A combo whose model has no
  items, a family selector with none of the three signals, a widget the
  factory declines to build: the panel is written to carry on, and carrying on
  is only a claim until it is run.
* A C++ OBJECT THAT HAS GONE. The heading guard holds a weak reference to the
  model and the slot-heading pass hides sections a screen teardown may already
  have destroyed; ``RuntimeError`` and a dead weakref are the ordinary cases.
* A SLOT ABOVE THE COUNT. ``collect`` writes back a hidden organelle slot only
  when it holds something other than what the panel invented for it -- the
  whole of "a file written at seven opens at two and still carries seven".

Offscreen Qt, no network, no modal dialogs.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (QEvent, QPointF, Qt,                  # noqa: E402
                            QStringListModel)
from PySide6.QtGui import QStandardItemModel                      # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox,           # noqa: E402
                               QFormLayout, QLabel, QLineEdit,
                               QVBoxLayout, QWidget)

from spacr.qt.screens import settings_model as SM                 # noqa: E402


# ---------------------------------------------------------------------------
# module defaults and the plane-clearing rule
# ---------------------------------------------------------------------------

def test_the_external_mask_module_brings_its_own_defaults():
    """``external_masks`` is the one app whose defaults live in its module."""
    from spacr.external_masks import default_settings

    resolved = SM.resolve_default_settings("external_masks")

    assert resolved == default_settings({})
    assert "inputs" in resolved


def test_the_built_in_dispatch_answers_the_same_as_the_registration_seam(
        monkeypatch):
    """Two routes to one module's defaults must not be able to disagree.

    ``spacr.external_masks`` registers its defaults through the
    ``register_defaults`` seam, which ``resolve_default_settings`` consults
    before its own built-in dispatch -- so the explicit ``external_masks``
    arm further down only answers for a process where the seam has not been
    filled. It is the same answer or it is a bug, and nothing had ever read
    the second one back.
    """
    from spacr import settings as spacr_settings
    from spacr.external_masks import default_settings

    through_the_seam = SM.resolve_default_settings("external_masks")
    assert spacr_settings.has_registered_defaults("external_masks")

    monkeypatch.setattr(spacr_settings, "has_registered_defaults",
                        lambda _key: False)
    built_in = SM.resolve_default_settings("external_masks")

    assert built_in == through_the_seam == default_settings({})


def test_a_plane_setting_the_type_table_never_declared_keeps_its_spin_box():
    """No declaration is not a declaration that the object may be absent."""
    # `nucleus_channel` is declared to accept None, so it is clearable;
    # a made-up plane key is in `expected_types` at all, so it is not.
    assert SM._is_clearable_plane_setting("nucleus_channel") is True
    assert SM._is_clearable_plane_setting("_probe_channel") is False


def test_a_type_table_that_cannot_be_read_leaves_every_plane_alone(monkeypatch):
    """A broken declaration table must not make a clearable plane of every key."""
    from spacr import settings as spacr_settings

    assert SM._is_clearable_plane_setting("nucleus_channel") is True

    # `expected_types` is read through `.get`; a table that is not a mapping
    # is exactly what an import-time failure in spacr.settings leaves behind.
    monkeypatch.setattr(spacr_settings, "expected_types", object())

    assert SM._is_clearable_plane_setting("nucleus_channel") is False


# ---------------------------------------------------------------------------
# a layout that claims every key
# ---------------------------------------------------------------------------

def test_a_layout_that_places_every_key_grows_no_additional_bucket():
    """The trailing bucket exists for leftovers, and only for leftovers."""
    placed = SM.categories_for_app("power", {})
    leftover = SM.categories_for_app("power", {"X": ["_probe_unplaced"]})

    assert "Library Design" in placed          # the spec really did render
    assert "Additional Settings" not in placed
    assert leftover["Additional Settings"] == ["_probe_unplaced"]


def test_a_merged_classifier_layout_claims_every_key_it_is_given():
    """Classify's rebuild names every group, so nothing falls out of it."""
    claimed = SM.categories_for_app("classify", {})
    leftover = SM.categories_for_app("classify", {"X": ["_probe_unplaced"]})

    assert "Labels & Classes" in claimed
    assert "Additional Settings" not in claimed
    assert leftover["Additional Settings"] == ["_probe_unplaced"]


def test_a_category_emptied_by_hiding_disappears_with_its_keys(monkeypatch):
    """A heading with nothing under it is a heading about nothing."""
    monkeypatch.setitem(SM._APP_HIDDEN_KEYS, "_probe_app", {"hidden_one"})

    out = SM.categories_for_app("_probe_app", {
        "Only the hidden one": ["hidden_one"],
        "One of each": ["hidden_one", "src"],
    })

    assert "Only the hidden one" not in out
    assert out["One of each"] == ["src"]


# ---------------------------------------------------------------------------
# help with nothing to say
# ---------------------------------------------------------------------------

def test_a_blank_category_title_gets_no_invented_blurb():
    """"Settings that control ." is worse than silence."""
    assert SM.category_tooltip("measure", "   ") == ""
    assert SM.category_tooltip("measure", "Paths").strip() != ""


def test_an_unhashable_language_is_resolved_rather_than_refused():
    """A language that cannot be a memo key is still a language."""
    with SM.language_resolved_once():
        cached = SM._language_code(None)
        # A list cannot key the scope dict; resolving it must not raise.
        assert SM._language_code([]) == cached


def test_a_setting_with_no_prose_still_names_itself_in_the_footer():
    """The hover footer says what the control is even with no description.

    The invented sentence used to read "Controls src." -- the raw key,
    because the humaniser only capitalised it. Commit 53a40ebce gave
    ``src`` the exact label "Path" in ``object_roles.EXACT_LABELS`` so
    that every surface names it the same way, which is a deliberate
    change and not a regression: the footer still names the control it is
    describing, it just spells the name the way the user sees it on the
    form rather than the way the settings file spells it. The key is
    still what selects the label, so the substitution is asserted through
    ``src`` rather than by handing "path" straight to the function.
    """
    described = SM.plain_tooltip("How many plates to read.", "measure", "src")
    bare = SM.plain_tooltip("", "measure", "src")
    nameless = SM.plain_tooltip("", "measure", "")

    assert "How many plates to read." in described
    assert "Controls path." in bare
    assert bare.startswith("Path ")
    assert nameless.startswith("Controls this setting.")


# ---------------------------------------------------------------------------
# the regression design scan
# ---------------------------------------------------------------------------

def test_a_pairing_that_is_not_a_list_names_no_count_file():
    """`paired_data` is a list of pairs; one pair on its own is not one."""
    assert SM._count_files_of({"paired_data": [{"count": "a.csv"}]}) == ["a.csv"]
    assert SM._count_files_of({"paired_data": {"count": "a.csv"}}) == []


def test_a_pair_without_a_count_file_is_stepped_over():
    """One incomplete row must not stop the rest of the list being read."""
    assert SM._count_files_of(
        {"paired_data": [{"count": None}, {"score": "s.csv"},
                         {"count": "  "}, {"count": " b.csv "}]}) == ["b.csv"]


def test_a_legacy_count_list_holding_blanks_is_stepped_over():
    """The pre-migration `count_data` list is hand-edited, so it has holes."""
    assert SM._count_files_of(
        {"count_data": [None, "", "   ", "c.csv"]}) == ["c.csv"]


def test_a_count_table_with_no_guide_column_still_counts_its_wells(tmp_path):
    """The scan reports what it could not work out instead of inventing it."""
    path = tmp_path / "counts.csv"
    path.write_text("plate,row_name,column_name,count\n"
                    "1,A,1,5\n1,A,2,7\n", encoding="utf-8")

    scan = SM.regression_design_scan({"count_data": [str(path)]})

    assert scan["guides"] is None and scan["genes"] is None
    assert "grna" in scan["note"]
    assert scan["wells"] == 2 and scan["rows"] == 2


# ---------------------------------------------------------------------------
# the greyed-out note, and putting it back
# ---------------------------------------------------------------------------

def test_a_label_greyed_twice_still_restores_the_help_it_started_with():
    """The backup is the ORIGINAL help, not the previous note-laden copy."""
    label = QLabel("Nucleus channel")
    label.setToolTip("Which plane the nucleus is imaged in.")

    SM._note_on_label(label, "Not used by 'ml'.")
    SM._note_on_label(label, "Not used by 'cv'.")

    assert label.property(SM._NOTE_BACKUP_PROPERTY) == \
        "Which plane the nucleus is imaged in."
    assert label.toolTip().endswith("<i>Not used by 'cv'.</i>")


def test_a_control_with_no_help_keeps_the_note_text_when_the_note_lifts():
    """There is no help to restore, so the tooltip is left where it is."""
    with_help = QLineEdit()
    with_help.setProperty("apiTooltipHtml", "Where the plate is.")
    without_help = QLineEdit()

    SM._apply_greyed_note(with_help, "Not used here.")
    SM._apply_greyed_note(without_help, "Not used here.")
    SM._clear_greyed_note(with_help)
    SM._clear_greyed_note(without_help)

    assert with_help.toolTip() == "Where the plate is."
    assert without_help.toolTip() == "Not used here."
    assert without_help.property(SM._BASIS_NOTE_PROPERTY) is False


def test_a_label_hung_after_the_greying_is_re_enabled_with_its_own_help():
    """No backup and no reason to strip: the label's own help is untouched."""
    quiet = QLineEdit()
    named = QLineEdit()
    for control in (quiet, named):
        control.setProperty("apiTooltipHtml", "Where the plate is.")

    # Greyed BEFORE the label existed, which is the build-time order: the
    # reason is remembered on the control and there is no label to bake it
    # into. `quiet` is greyed with no reason at all, `named` with one.
    SM._apply_greyed_note(quiet, "")
    SM._apply_greyed_note(named, "Not used here.")

    quiet_label, named_label = QLabel("Source"), QLabel("Source")
    quiet_label.setToolTip("Where the plate is.")
    named_label.setToolTip("Where the plate is.<br><i>Not used here.</i>")
    for label in (quiet_label, named_label):
        label.setEnabled(False)
    quiet._spacr_setting_label = quiet_label
    named._spacr_setting_label = named_label

    SM._clear_greyed_note(quiet)
    SM._clear_greyed_note(named)

    assert quiet_label.isEnabled() and named_label.isEnabled()
    assert quiet_label.toolTip() == "Where the plate is."
    assert named_label.toolTip() == "Where the plate is."


# ---------------------------------------------------------------------------
# API dots are help, not settings
# ---------------------------------------------------------------------------

class _UrlSink(QWidget):
    """A dot that owns its destination, as the real API dot does."""

    def __init__(self):
        super().__init__()
        self.url = None

    def set_url(self, url):
        self.url = url


def test_an_api_dot_is_re_pointed_and_captioned_rather_than_re_described():
    """The dot's help names the setting it opens; the URL follows the language."""
    owner = QWidget()
    dot = _UrlSink()
    dot.setParent(owner)
    dot.setProperty("settingsAppKey", "measure")
    dot.setProperty("settingKey", "src")
    dot.setProperty("apiTooltipDisplayRole", "api-link")
    field = QLineEdit(owner)
    field.setProperty("settingsAppKey", "measure")
    field.setProperty("settingKey", "src")

    SM.refresh_api_tooltips(owner, "en")

    caption = SM._api_reference_tooltip("src", "en", "measure")
    # This asked for the raw key in the caption. Commit 53a40ebce made
    # ``src`` render as its exact label "Path" everywhere, from one place
    # in ``object_roles.EXACT_LABELS``, so a caption still carrying "src"
    # would now be the bug: the dot sits beside a field the form calls
    # Path, and its accessible name is read out on its own. What the
    # caption has to do is name the setting it opens, which it does.
    assert caption == "Open API reference for Path"
    assert dot.toolTip() == caption
    assert dot.accessibleName() == caption
    assert dot.url == SM.api_docs_url("measure", "src", "en")
    # The ordinary field is still described, so the dot's silence is a
    # decision about dots rather than about this owner.
    assert dot.toolTip() != field.toolTip()


def test_an_api_dot_with_no_destination_still_gets_its_caption():
    """A dot that is a plain widget carries the caption and nothing else."""
    owner = QWidget()
    dot = QWidget(owner)
    dot.setProperty("settingsAppKey", "measure")
    dot.setProperty("settingKey", "src")
    dot.setProperty("apiTooltipDisplayRole", "api-link")

    SM.refresh_api_tooltips(owner, "en")

    assert dot.toolTip() == SM._api_reference_tooltip("src", "en", "measure")
    assert dot.accessibleDescription()


def test_the_decoration_sweep_walks_past_the_dots_it_made():
    """A dot carries `settingKey`, so a second pass would decorate it again."""
    owner = QWidget()
    QVBoxLayout(owner)
    field = QLineEdit(owner)
    field.setProperty("settingKey", "src")
    dot = QLabel("i", owner)
    dot.setProperty("settingKey", "src")
    dot.setProperty("apiTooltipDisplayRole", "api-link")

    SM.install_api_tooltips(owner, "measure")

    assert field.property("settingsAppKey") == "measure"
    assert dot.property("settingsAppKey") is None
    assert dot.property("apiTooltipDisplayRole") == "api-link"


def test_a_tooltip_that_cannot_be_priced_still_reaches_the_regression_box(
        monkeypatch):
    """The mixed-model cost sentence is an addition, not a precondition."""
    def _explode():
        raise RuntimeError("no cost model")

    prose = {"regression_type": "Which model to fit."}
    priced = QLineEdit()
    SM.attach_api_tooltip(priced, "regression", "regression_type",
                          _descriptions=prose)
    monkeypatch.setattr(SM, "mixed_cost_note", _explode)
    unpriced = QLineEdit()
    SM.attach_api_tooltip(unpriced, "regression", "regression_type",
                          _descriptions=prose)

    assert unpriced.property("apiTooltipDescriptionSource") == \
        "Which model to fit."
    assert priced.property("apiTooltipDescriptionSource").startswith(
        "Which model to fit. ")
    assert len(priced.property("apiTooltipDescriptionSource")) > \
        len(unpriced.property("apiTooltipDescriptionSource"))


# ---------------------------------------------------------------------------
# finding the label a field belongs to
# ---------------------------------------------------------------------------

def test_a_label_host_hands_back_the_label_it_wraps():
    """Decorating the host instead gives the panel two tooltips per setting."""
    marked_host = QWidget()
    marked_host.setObjectName("SettingLabelWithInfo")
    QVBoxLayout(marked_host)
    marked = QLabel("Source", marked_host)
    marked.setProperty("settingHelpLabel", True)

    plain = QLabel("Source")

    assert SM._unwrap_setting_label(marked_host) is marked
    assert SM._unwrap_setting_label(plain) is plain
    assert SM._unwrap_setting_label(None) is None


def test_an_undecorated_host_hands_back_its_first_captioned_label():
    """Before the first pass there is no marked child, only a caption."""
    host = QWidget()
    host.setObjectName("SettingLabelWithInfo")
    QVBoxLayout(host)
    spacer = QLabel("   ", host)          # the info slot, still empty
    caption = QLabel("Source", host)

    found = SM._unwrap_setting_label(host)

    assert found is caption and found is not spacer


def test_a_field_in_no_form_has_no_form_label():
    """The walk up the parent chain ends at the top, not in a guess."""
    owner = QWidget()
    form = QFormLayout(owner)
    label = QLabel("Source")
    field = QLineEdit()
    form.addRow(label, field)
    orphan = QLineEdit()                  # never parented, so the walk ends

    assert SM._setting_label_for_field(owner, field) is label
    assert SM._setting_label_for_field(owner, orphan) is None


# ---------------------------------------------------------------------------
# the guard that watches for a hidden row coming back
# ---------------------------------------------------------------------------

def _show_event():
    return QEvent(QEvent.ShowToParent)


def test_a_row_guard_outliving_its_model_answers_without_it():
    """The guard is parented to the panel, so it can outlive the model."""
    model = SM.SettingsWidgets("measure")
    seen = []
    model._shown_against_the_rule = seen.append
    parent = QWidget()
    guard = SM._HiddenRowWatcher(model, parent)
    row = QWidget()

    assert guard.eventFilter(row, _show_event()) is False
    assert seen == [row]

    del model
    gc.collect()
    assert guard._model() is None
    assert guard.eventFilter(row, _show_event()) is False
    assert seen == [row]


def test_a_row_guard_survives_a_model_that_cannot_answer():
    """One failed re-assertion must not break Qt's event delivery."""
    calls = []

    def _explode(widget):
        calls.append(widget)
        raise RuntimeError("the panel went away mid-pass")

    model = SM.SettingsWidgets("measure")
    model._shown_against_the_rule = _explode
    guard = SM._HiddenRowWatcher(model, QWidget())
    row = QWidget()

    assert guard.eventFilter(row, _show_event()) is False
    assert calls == [row]


# ---------------------------------------------------------------------------
# the regression backend field
# ---------------------------------------------------------------------------

def test_a_backend_the_menu_does_not_offer_leaves_the_choice_alone(monkeypatch):
    """A settings CSV naming an unknown backend is answered at run time."""
    from spacr import regression_backends

    field = SM._RegressionBackendField()
    known = field.combo.itemData(field.combo.count() - 1)
    field.combo.setCurrentIndex(0)
    first = field.get_value()

    field.set_value(known)
    assert field.get_value() == known

    monkeypatch.setattr(regression_backends, "backend_label",
                        lambda value: "not a backend anybody registered")
    field.set_value(first)

    assert field.get_value() == known


def test_a_backend_menu_whose_model_has_no_items_is_still_labelled():
    """Greying an entry is optional; saying what it is, is not."""
    field = SM._RegressionBackendField()
    standard = field.combo.model()
    assert isinstance(standard, QStandardItemModel)
    assert standard.item(0) is not None

    labels = [field.combo.itemText(i) for i in range(field.combo.count())]
    # A plain string model has no `item()` at all -- the entry cannot be
    # greyed, only named.
    field.combo.setModel(QStringListModel(labels))
    field.refresh()

    assert field.combo.count() == len(labels)
    assert all(field.combo.itemText(i) for i in range(field.combo.count()))


class _ViewlessCombo(QComboBox):
    """A combo whose popup Qt has not built yet."""

    def view(self):                                      # noqa: D102
        return None


def test_a_combo_with_no_popup_is_still_watched_itself():
    """The closed control is the route that always exists."""
    field = SM._RegressionBackendField()
    hovered, popped = [], []
    field._hover_closed_combo = lambda *a: hovered.append(True)
    field._hover_popup_row = lambda *a: popped.append(True)

    viewport = field.combo.view().viewport()
    field._install_availability_hooks()
    QApplication.sendEvent(viewport, _mouse_move(viewport))
    assert popped

    field.combo = _ViewlessCombo(field)
    field._install_availability_hooks()
    QApplication.sendEvent(field.combo, QEvent(QEvent.Enter))

    assert hovered


def _mouse_move(widget):
    from PySide6.QtGui import QMouseEvent

    where = QPointF(1.0, 1.0)
    return QMouseEvent(QEvent.MouseMove, where, where,
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)


def test_an_event_from_somewhere_else_reaches_nobodys_hover_rule():
    """The filter is installed on two objects and speaks for those two."""
    field = SM._RegressionBackendField()
    hovered = []
    field._hover_closed_combo = lambda *a: hovered.append(True)
    stranger = QWidget()

    assert field.eventFilter(stranger, QEvent(QEvent.Enter)) is False
    assert hovered == []

    assert field.eventFilter(field.combo, QEvent(QEvent.Enter)) is False
    assert hovered == [True]


# ---------------------------------------------------------------------------
# the chip strip
# ---------------------------------------------------------------------------

def test_a_comma_with_nothing_before_it_makes_no_chip():
    """A pasted ',c1' is a leading separator, not an empty value."""
    strip = SM._ChipStrip()
    # `_on_typed` is wired to `textEdited`, which only a real keystroke
    # emits; the handler is what a paste runs.
    strip._on_typed(",c1")
    assert [chip.text() for chip in strip._chips] == []
    assert strip._entry.text() == "c1"      # kept, not thrown away

    strip._on_typed("c1,c2")
    assert [chip.text() for chip in strip._chips] == ["c1"]


def test_a_chip_the_strip_no_longer_holds_is_removed_without_complaint():
    """Qt can deliver the chip's own `removed` signal after the strip has it out."""
    strip = SM._ChipStrip()
    strip.set_values(["c1", "c2"])
    chip = strip._chips[0]

    strip._remove_chip(chip)
    assert [held.text() for held in strip._chips] == ["c2"]

    stranger = SM._Chip("c3", strip._colours, strip._host)
    strip._remove_chip(stranger)

    assert [held.text() for held in strip._chips] == ["c2"]


# ---------------------------------------------------------------------------
# building the panel
# ---------------------------------------------------------------------------

def test_a_setting_the_factory_declines_to_build_is_left_out_of_the_panel():
    """A widget that was never made cannot be tooltipped or laid out."""
    widgets = SM.SettingsWidgets("measure")
    real = widgets._widget_for
    declined = []

    def _decline(kind, options, default, key):
        if key == "src":
            declined.append(key)
            return None
        return real(kind, options, default, key)

    widgets._widget_for = _decline
    widgets.build_sections()

    assert declined == ["src"]
    assert "src" not in widgets._widgets
    assert widgets._widgets            # everything else was still built


class _Voiceless(QWidget):
    """A control with none of the three value-changed signals."""


def test_a_selector_with_no_signal_is_left_unconnected_rather_than_fatal():
    """Three spellings are tried; a control with none is simply not followed."""
    widgets = SM.SettingsWidgets("classify_merged")
    # Only the three selectors the connection block looks for, so the panel
    # is four widgets rather than four hundred.
    widgets._defaults = {"classifier_family": "torch",
                         "dataset_mode": "metadata",
                         "regression_type": "ols",
                         "regression_backend": "statsmodels"}

    def _voiceless(kind, options, default, key):
        if key == "regression_backend":
            return SM._RegressionBackendField()
        return _Voiceless()

    widgets._widget_for = _voiceless
    widgets.build_sections()

    assert set(widgets._widgets) == {
        "classifier_family", "dataset_mode",
        "regression_type", "regression_backend"}
    assert isinstance(widgets._widgets["regression_backend"],
                      SM._RegressionBackendField)
    assert all(isinstance(widgets._widgets[key], _Voiceless)
               for key in ("classifier_family", "dataset_mode",
                           "regression_type"))


def test_a_combo_whose_default_is_absent_is_not_offered_an_empty_choice():
    """An empty default is not a value the module asked for."""
    widgets = SM.SettingsWidgets("measure")

    named = widgets._widget_for("combo", ["a", "b"], "c", "_probe_named")
    blank = widgets._widget_for("combo", ["a", "b"], None, "_probe_blank")

    assert [named.itemText(i) for i in range(named.count())] == ["c", "a", "b"]
    assert [blank.itemText(i) for i in range(blank.count())] == ["a", "b"]


def test_a_count_that_is_not_a_number_grows_no_slots():
    """The count comes from a hand-editable CSV, so it is not always a number."""
    widgets = SM.SettingsWidgets("measure")
    widgets._slots_built_for = 2

    assert widgets.grow_to_fit_the_organelle_count("not a number") == 2
    assert widgets._slots_built_for == 2


# ---------------------------------------------------------------------------
# which organelle slots are worth writing back
# ---------------------------------------------------------------------------

class _Unequal:
    """A value that cannot be compared, as a NumPy array cannot."""

    def __eq__(self, other):
        raise ValueError("truth value is ambiguous")

    __hash__ = None


def test_a_slot_above_the_count_survives_only_when_it_says_something_new():
    """A file written at three opens at one and still carries three."""
    widgets = SM.SettingsWidgets("measure")
    # What the panel invented for the slots above the count. Set here
    # because `grow_to_fit_the_organelle_count` is the only writer and it
    # rebuilds the whole form to do it.
    widgets._slots_the_panel_added = {
        "organelleb_channel": None,
        "organellec_channel": None,
    }

    kept = widgets._organelle_slots_worth_keeping({
        "number_of_organelles": 1,
        "organelle_channel": 0,
        "organelleb_channel": None,    # exactly what the panel put there
        "organellec_channel": 2,       # the file's own answer
    })

    assert "organellec_channel" in kept          # slot three, so slot two too
    assert "organelleb_channel" in kept
    assert kept["organellec_channel"] == 2


def test_a_slot_the_panel_never_invented_is_always_written_back():
    """A value that came from anywhere else is the run's, whatever the count."""
    widgets = SM.SettingsWidgets("measure")
    widgets._slots_the_panel_added = {"organelleb_channel": None}

    kept = widgets._organelle_slots_worth_keeping({
        "number_of_organelles": 1,
        "organelle_channel": 0,
        "organelleb_channel": None,
        "organellec_channel": 3,       # never invented here
    })

    assert kept["organellec_channel"] == 3


def test_a_slot_holding_a_value_that_cannot_be_compared_is_kept():
    """An unanswerable comparison is not evidence the panel put it there."""
    widgets = SM.SettingsWidgets("measure")
    widgets._slots_the_panel_added = {"organelleb_channel": _Unequal()}

    kept = widgets._organelle_slots_worth_keeping({
        "number_of_organelles": 1,
        "organelle_channel": 0,
        "organelleb_channel": _Unequal(),
    })

    assert "organelleb_channel" in kept


def test_a_slot_whose_comparison_blows_up_is_kept(monkeypatch):
    """An unanswerable comparison is not evidence the panel wrote the value.

    The real ``_values_equal`` cannot raise -- it delegates to
    ``spacr.run_journal.values_equal``, which answers ``False`` rather than
    propagate anything -- so this drives the guard through the seam the
    method imports it by.
    """
    from spacr.qt import settings_diff

    def _explode(_a, _b):
        raise ValueError("the truth value of an array is ambiguous")

    widgets = SM.SettingsWidgets("measure")
    widgets._slots_the_panel_added = {"organelleb_channel": 1}
    monkeypatch.setattr(settings_diff, "_values_equal", _explode)

    kept = widgets._organelle_slots_worth_keeping({
        "number_of_organelles": 1,
        "organelle_channel": 0,
        "organelleb_channel": 1,
    })

    assert "organelleb_channel" in kept


def test_slots_within_the_count_are_kept_without_asking_what_they_hold():
    """The count reaches them, so what they hold is not the question."""
    widgets = SM.SettingsWidgets("measure")
    widgets._slots_the_panel_added = {"organelleb_channel": None}

    kept = widgets._organelle_slots_worth_keeping({
        "number_of_organelles": 2,
        "organelle_channel": 0,
        "organelleb_channel": None,
    })

    assert kept["organelleb_channel"] is None


# ---------------------------------------------------------------------------
# the UMAP reducer rule
# ---------------------------------------------------------------------------

def test_the_shared_metric_is_left_editable_only_when_the_panel_has_one():
    """DBSCAN reads it whatever the projection does -- when it is on the form."""
    widgets = SM.SettingsWidgets("umap")
    combo = QComboBox()
    combo.addItem("umap")
    widgets._widgets["reduction_method"] = combo
    metric = QLineEdit()
    metric.setEnabled(False)
    widgets._widgets["metric"] = metric

    widgets._refresh_umap_reducer_enablement()
    assert metric.isEnabled()

    # The same pass on a panel that never built the shared metric.
    del widgets._widgets["metric"]
    metric.setEnabled(False)
    widgets._refresh_umap_reducer_enablement()

    assert metric.isEnabled() is False


# ---------------------------------------------------------------------------
# reading plate identity out of a count file
# ---------------------------------------------------------------------------

def test_a_blank_plate_cell_names_no_plate(tmp_path):
    """An empty cell is a missing answer, not a plate called ''."""
    one = tmp_path / "one.csv"
    one.write_text("plate,count\n,5\nP1,7\n", encoding="utf-8")
    two = tmp_path / "two.csv"
    two.write_text("plate,count\nP1,5\nP2,7\n", encoding="utf-8")

    blank = SM.SettingsWidgets._plate_context([str(one)])
    named = SM.SettingsWidgets._plate_context([str(two)])

    assert blank == {"plate_count": 1, "has_plate_id": True}
    assert named["plate_count"] == 2


# ---------------------------------------------------------------------------
# the object-visibility pass
# ---------------------------------------------------------------------------

def test_a_visibility_rule_that_cannot_be_evaluated_hides_nothing(monkeypatch):
    """A panel with every row on it beats a panel that failed to open."""
    widgets = SM.SettingsWidgets("measure")
    # The rule reads the panel's own rows, and a model built for its values
    # has none until `build_sections` runs. These two are the whole gate.
    widgets._widgets = {"organelleb_channel": QLineEdit(),
                        "organelleb_diameter": QLineEdit(),
                        "src": QLineEdit()}
    working = widgets.keys_whose_object_the_run_lacks()
    assert working == {"organelleb_diameter"}

    def _explode(*_args, **_kwargs):
        raise RuntimeError("the settings table went away")

    monkeypatch.setattr(SM, "keys_hidden_by_their_object", _explode)

    assert widgets.keys_whose_object_the_run_lacks() == set()


def test_a_screen_that_cannot_lay_out_the_returning_rows_still_hides_them():
    """The layout callback belongs to the screen; the rule belongs here."""
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets = {"organelleb_channel": QLineEdit(),
                        "organelleb_diameter": QLineEdit(),
                        "src": QLineEdit()}
    asked = []

    def _explode(hidden):
        asked.append(set(hidden))
        raise RuntimeError("the screen is mid-teardown")

    widgets.rows_are_laid_out_by = _explode
    widgets.refresh_object_visibility()

    assert asked and asked[0]
    assert widgets.keys_hidden_by_the_run()


def test_a_queued_visibility_pass_clears_its_own_flag():
    """One pass is queued however many rows came back, so it must reset."""
    widgets = SM.SettingsWidgets("measure")
    widgets._widgets = {"organelleb_channel": QLineEdit(),
                        "organelleb_diameter": QLineEdit(),
                        "src": QLineEdit()}
    widgets._object_rule_pass_queued = True

    widgets._reassert_object_visibility()

    assert widgets._object_rule_pass_queued is False
    assert widgets.keys_hidden_by_the_run()


def test_a_field_that_is_not_in_the_form_is_hidden_on_its_own():
    """The screen builds the label and the row after the widgets come back."""
    host = QWidget()
    form = QFormLayout(host)
    label = QLabel("In the form")
    in_form = QLineEdit()
    form.addRow(label, in_form)
    loose = QLineEdit(host)               # a child of the host, in no row

    widgets = SM.SettingsWidgets("measure", parent=host)
    widgets._widgets = {"_probe_row": in_form, "_probe_loose": loose}

    widgets._set_row_visible("_probe_row", False)
    widgets._set_row_visible("_probe_loose", False)

    assert in_form.isHidden() and label.isHidden()
    assert loose.isHidden()


# ---------------------------------------------------------------------------
# headings for slots the run does not have
# ---------------------------------------------------------------------------

class _Heading(QWidget):
    """The part of ``Section`` the slot-heading rule reads."""

    def __init__(self, parent, stage="stable"):
        super().__init__(parent)
        self._stage = stage

    def maturity(self):                                  # noqa: D102
        return self._stage


def _model_with_heading(keys, has_children=False, stage="stable"):
    """A model holding one leaf heading that is on screen, not yet hidden."""
    screen = QWidget()
    section = _Heading(screen, stage)
    widgets = SM.SettingsWidgets("measure", parent=screen)
    widgets._screen_under_test = screen          # keep the parent alive
    widgets.remember_section_rows(section, keys, has_children)
    return widgets, section


def test_a_panel_of_nothing_but_parent_headings_caches_no_leaf():
    """A heading answered by its sub-headings is not a leaf."""
    parent_only, _outer = _model_with_heading(("organelleb_channel",),
                                              has_children=True)
    with_leaf, leaf = _model_with_heading(("organelleb_channel",))

    assert parent_only._slot_headings() == {}
    assert list(with_leaf._slot_headings()) == [id(leaf)]


def test_a_heading_of_an_absent_slot_is_hidden_and_shown_again():
    """A heading with every row hidden is a smaller wall, but a wall."""
    widgets, section = _model_with_heading(("organelleb_channel",))
    assert not section.isHidden()

    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 1})
    assert section.isHidden()
    assert id(section) in widgets._headings_of_absent_slots

    # The same pass again: already hidden, so nothing is recorded twice.
    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 1})
    assert list(widgets._headings_of_absent_slots) == [id(section)]

    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 2})

    assert not section.isHidden()
    assert widgets._headings_of_absent_slots == {}


def test_a_slot_heading_hidden_as_alpha_is_not_shown_by_the_count(monkeypatch):
    """Putting a slot back must not overrule Preferences."""
    from spacr.qt import preferences

    widgets, section = _model_with_heading(("organelleb_channel",),
                                           stage="alpha")
    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 1})
    assert section.isHidden()

    monkeypatch.setattr(preferences, "get_show_alpha", lambda: False)
    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 2})

    assert section.isHidden()
    assert widgets._headings_of_absent_slots == {}


def test_a_heading_that_went_away_with_its_screen_is_forgotten():
    """The section is destroyed by the teardown that owns it, not by this."""
    import shiboken6

    widgets, section = _model_with_heading(("organelleb_channel",))
    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 1})
    ident = id(section)
    assert ident in widgets._headings_of_absent_slots

    shiboken6.delete(section)          # what a screen teardown leaves behind

    widgets._hide_the_headings_of_slots_the_run_lacks(
        {"number_of_organelles": 1})

    assert ident not in widgets._headings_of_absent_slots


def test_a_heading_whose_body_is_not_a_form_is_walked_past():
    """The walk reads rows out of a QFormLayout; a section without one has none."""
    from spacr.qt.widgets.section import Section

    parent = QWidget()
    QVBoxLayout(parent)
    formless = Section("No form", parent)
    formless._form = QVBoxLayout()
    widgets = SM.SettingsWidgets("measure", parent=parent)

    assert widgets._slot_headings() == {}


def test_a_panel_whose_headings_cannot_be_walked_reports_none(monkeypatch):
    """A half-torn-down screen answers 'no headings', not a traceback."""
    parent = QWidget()
    widgets = SM.SettingsWidgets("measure", parent=parent)

    class _Hostile:
        def findChildren(self, *_args, **_kwargs):
            raise RuntimeError("the screen is going away")

    widgets._parent = _Hostile()

    assert widgets._slot_headings() == {}
