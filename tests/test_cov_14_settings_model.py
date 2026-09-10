"""The settings panel builds from whatever the tables and widgets give it.

Every value on this screen comes from somewhere that can be absent or
malformed: a translation catalog that will not take a composed caption, a
settings CSV saved before a key was renamed, a backend that is registered but
not installed, a count file with a column layout from two versions ago. A
settings panel that raised on any of them would leave the user with no way to
start a run at all.

The rule the tests below pin is the same one throughout: an input the panel
cannot use produces a control that is quiet or a value that is left alone --
never a guess, and never an exception on the GUI thread.
"""
from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

pytest.importorskip("PySide6")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

from spacr.qt.screens import settings_model as sm  # noqa: E402


# -- captions and tables ------------------------------------------------------

def test_a_catalog_that_refuses_a_caption_leaves_the_heading_readable(
        monkeypatch):
    """A composed heading is still returned when it cannot be catalogued.

    The heading is a KEY as well as a caption -- the blurb tables and the
    layout are written against the English text -- so failing to register a
    translation must not stop the panel being built.
    """
    import spacr.qt.i18n as i18n

    def _refuse(*args, **kwargs):
        raise ValueError("the catalog is closed")

    monkeypatch.setattr(i18n, "add_translation", _refuse)

    heading = sm._family_heading("Computer Vision", "Images & Cropping")

    assert heading == f"Computer Vision {sm._FAMILY_HEADING_DASH} Images & Cropping"


def test_apps_registered_before_this_module_get_their_api_module(monkeypatch):
    """The pull half of the registration seam picks up earlier registrations.

    Without it a module that registered itself first sends its ⓘ link to the
    generated API index instead of to its own page.
    """
    app_module = types.ModuleType("spacr.qt.app")
    app_module.registered_metadata = lambda field: {
        "a_new_app": "spacr/a_new_app"}
    monkeypatch.setitem(sys.modules, "spacr.qt.app", app_module)
    monkeypatch.delitem(sm._APP_API_MODULE, "a_new_app", raising=False)

    sm._absorb_registered_api_modules()

    assert sm._APP_API_MODULE["a_new_app"] == "spacr/a_new_app"
    del sm._APP_API_MODULE["a_new_app"]


# -- number boxes -------------------------------------------------------------

def test_a_value_that_is_not_a_number_leaves_the_box_on_auto(qapp):
    """Text that will not parse puts the box back on "auto", not on zero.

    Zero is a real setting for most of these; silently substituting it is a
    run configured differently from what the file said.
    """
    from PySide6.QtWidgets import QDoubleSpinBox

    box = QDoubleSpinBox()
    box.setRange(0.0, 10.0)
    box.setSpecialValueText(sm.AUTO_TEXT)
    box.setValue(5.0)

    sm._set_auto_or_number(box, "not a number")

    assert box.value() == box.minimum()

    sm._set_auto_or_number(box, 3.0)

    assert box.value() == 3.0

    sm._set_auto_or_number(box, None)

    assert box.value() == box.minimum()


def test_an_unreadable_tooltip_table_leaves_the_domain_at_its_default(
        monkeypatch):
    """A settings table that cannot be read gives the magnitude-based range.

    The domain is a nicety; the box has to exist either way.
    """
    from spacr import settings as spacr_settings

    class _Hostile:
        def get(self, *args, **kwargs):
            raise RuntimeError("the settings table is not importable")

    monkeypatch.setattr(spacr_settings, "tooltips", _Hostile())

    low, high, step = sm._float_domain("nms_threshold", 0.5)

    assert step == 0.01
    assert low <= 0.5 <= high


# -- labelling ---------------------------------------------------------------

def test_a_button_whose_text_is_gone_is_not_self_labelling(qapp):
    """A control whose C++ side has been freed carries no label.

    Treating it as self-labelling would install the hover help on a widget
    that is on its way out.
    """
    from PySide6.QtWidgets import QPushButton

    class _Freed(QPushButton):
        def text(self):  # noqa: D102 - the point is that it raises
            raise RuntimeError("Internal C++ object already deleted.")

    assert sm._is_self_labelling(_Freed()) is False


def test_a_labelled_button_is_self_labelling(qapp):
    """A checkbox with text IS its own label."""
    from PySide6.QtWidgets import QCheckBox

    assert sm._is_self_labelling(QCheckBox("Use GPU")) is True
    assert sm._is_self_labelling(QCheckBox("")) is False


def test_a_field_with_no_label_is_left_quiet(qapp):
    """A composite field with nowhere to put its help says nothing.

    Qt delivers ``Enter`` to a parent whenever the pointer crosses any child,
    so hover help left on a container fires from anywhere inside it.
    """
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    owner = QWidget()
    layout = QVBoxLayout(owner)
    field = QWidget(owner)
    field.setProperty("settingKey", "nucleus_channel")
    field.setToolTip("something stale")
    layout.addWidget(field)

    sm.install_api_tooltips(owner, "mask")

    assert field.toolTip() == ""
    assert field.property("apiTooltipHtml") == ""
    assert field.property("apiTooltipDisplayRole") == "metadata"


# -- reading what a run was given ---------------------------------------------

def test_a_single_legacy_count_path_is_a_list_of_one():
    """The pre-migration spelling stored one path as a bare string."""
    assert sm._count_files_of({"count_data": " /screen/counts.csv "}) == \
        ["/screen/counts.csv"]


def test_the_current_paired_shape_wins_over_the_legacy_one():
    """``paired_data`` is read first; the legacy key is only the fallback."""
    settings = {"paired_data": [{"count": "/new.csv"}],
                "count_data": "/old.csv"}

    assert sm._count_files_of(settings) == ["/new.csv"]


def test_a_frame_that_already_carries_prc_is_keyed_by_it():
    """A composed well key is used as it is rather than rebuilt."""
    keys = sm._well_keys(pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"]}))

    assert list(keys) == ["p1_r1_c1", "p1_r1_c2"]


def test_a_frame_with_plate_row_and_column_is_keyed_by_the_pair():
    """``plate_row`` already holds the plate; only the column is appended."""
    keys = sm._well_keys(pd.DataFrame({"plate_row": ["p1_r1"],
                                       "columnID": ["c1"]}))

    assert list(keys) == [f"p1_r1{sm.KEY_SEPARATOR}c1"]


def test_a_frame_with_none_of_the_key_columns_is_refused():
    """No recognised layout gives ``None``, not a guessed key."""
    assert sm._well_keys(pd.DataFrame({"count": [1]})) is None


def test_two_part_guide_names_are_split_gene_then_guide():
    """``<gene>_<guide>`` names still yield a gene count."""
    genes, guides = sm._split_guide_names(["tgme49_1", "tgme49_2"])

    assert genes == {"tgme49"}
    assert guides == {"tgme49_1", "tgme49_2"}


def test_names_of_a_mixed_shape_get_no_gene_count():
    """A rule the run will not apply produces no number here."""
    genes, guides = sm._split_guide_names(["a_b_c", "d_e"])

    assert genes is None
    assert guides == {"a_b_c", "d_e"}


# -- documentation links -------------------------------------------------------

def test_a_backend_with_no_documentation_gets_no_link():
    """An unknown backend name yields no label and no URL."""
    assert sm.model_api_link("not_a_backend") == ("", "")


def test_a_backend_with_no_documentation_renders_no_html():
    """And therefore no API paragraph in the explainer."""
    assert sm._api_html("not_a_backend", {"accent": "#00b3a4"}) == ""


def test_a_known_backend_does_get_a_link():
    """The same call returns a label and a URL for a documented backend."""
    key = next(iter(sm.MODEL_API_LINKS))
    name, url = sm.model_api_link(key)

    assert name
    assert url.startswith("http")


# -- the explainer -------------------------------------------------------------

def test_a_section_with_no_explainer_renders_nothing():
    """A section the app has no guidance for produces empty rich text."""
    assert sm.section_explainer_html("mask", "Not A Real Section") == ""


def test_the_model_section_renders_the_regression_explainer():
    """The Model & Inference section is the regression explainer itself."""
    html = sm.section_explainer_html(
        "regression", "Model & Inference",
        settings={"regression_type": "ols", "level": "gene"})

    assert html.startswith("<div")
    assert "MODEL" in html or "Model" in html


def test_a_blank_line_survives_the_indenting():
    """Paragraph breaks are kept, so the explainer is not one wall of text."""
    assert sm._wrap_block("one\n\ntwo") == "    one\n\n    two"


def test_nonparametric_inference_describes_the_permutation_not_a_formula():
    """With permutation inference there is no fitted formula to describe.

    Printing one would tell the reader a model was fitted that was not.
    """
    text = sm.regression_model_explainer("ols", inference="nonparametric")

    assert text.startswith("INFERENCE: nonparametric")
    assert "~" not in text.splitlines()[0]


def test_every_explainer_line_survives_a_family_that_cannot_render(
        monkeypatch):
    """One unrenderable family does not empty the width measurement.

    ``explainer_width`` sizes the pane from these lines; an exception here
    would leave the pane sized from nothing.
    """
    real = sm.regression_model_explainer
    seen = []

    def _picky(family, level, **position):
        seen.append(family)
        if len(seen) == 1:
            raise RuntimeError("no words for this one")
        return real(family, level, **position)

    monkeypatch.setattr(sm, "regression_model_explainer", _picky)

    lines = sm._every_explainer_line()

    assert lines
    assert len(seen) > 1


# -- the CSV column picker -----------------------------------------------------

def test_the_column_field_honours_the_line_edit_contract(qapp):
    """``text``/``setText`` reach the inner editor, not a separate string.

    Callers written against a plain ``QLineEdit`` still drive this field, and
    a ``setText`` that did not reach the editor would leave the panel showing
    one column while collecting another.
    """
    field = sm._CsvColumnField(key="score_column", default="area")

    field.setText("intensity")

    assert field.text() == "intensity"
    assert field.get_value() == "intensity"


def test_the_column_field_asks_qt_when_no_chooser_was_injected(qapp,
                                                               monkeypatch):
    """Without an injected chooser the field opens Qt's own item dialog.

    Which name it starts on matters: a dialog opening on the first column
    when the field already holds the fifth invites the wrong pick.
    """
    from PySide6.QtWidgets import QInputDialog

    asked = {}

    def _get_item(parent, title, prompt, choices, index, editable):
        asked.update(choices=list(choices), index=index, title=title)
        return choices[index], True

    monkeypatch.setattr(QInputDialog, "getItem", staticmethod(_get_item))

    field = sm._CsvColumnField(key="score_column", default="b")

    chosen = field.choose(["a", "b", "c"], "b")

    assert chosen == "b"
    assert asked["index"] == 1
    assert asked["choices"] == ["a", "b", "c"]


def test_a_cancelled_qt_chooser_picks_nothing(qapp, monkeypatch):
    """Cancelling the dialog returns ``None`` rather than the highlighted row."""
    from PySide6.QtWidgets import QInputDialog

    monkeypatch.setattr(QInputDialog, "getItem",
                        staticmethod(lambda *a, **k: ("a", False)))

    field = sm._CsvColumnField(key="score_column")

    assert field.choose(["a", "b"], None) is None


def test_the_column_field_reports_through_qt_when_no_reporter_was_injected(
        qapp, monkeypatch):
    """Without an injected reporter the refusal goes to a message box.

    A picker that offers nothing and says nothing reads as a broken button.
    """
    from PySide6.QtWidgets import QMessageBox

    said = {}

    def _information(parent, title, message):
        said.update(title=title, message=message)

    monkeypatch.setattr(QMessageBox, "information", staticmethod(_information))

    field = sm._CsvColumnField(key="score_column")
    field.report("no CSVs have been chosen yet")

    assert said["message"] == "no CSVs have been chosen yet"
    assert said["title"] == "No columns to offer"


# -- the regression backend selector -------------------------------------------

def _backend_field(qtbot):
    field = sm._RegressionBackendField(regression_type="ols")
    qtbot.addWidget(field)
    return field


def test_a_backend_selector_with_nothing_selected_has_no_value(qtbot):
    """An empty selection is ``None``, not the first entry."""
    field = _backend_field(qtbot)
    field.combo.setCurrentIndex(-1)

    assert field.get_value() is None
    assert field.text() == ""


def test_a_backend_name_that_is_not_a_backend_leaves_the_choice_alone(qtbot):
    """An unrecognised stored value does not silently re-point the selector.

    Choosing something else for the user is how a settings file comes back
    describing a different run from the one it recorded.
    """
    field = _backend_field(qtbot)
    before = field.get_value()

    field.set_value("not_a_backend_at_all")

    assert field.get_value() == before


def test_the_backend_selector_honours_the_combobox_contract(qtbot):
    """``text``/``setText`` select by label, as a plain combo would."""
    field = _backend_field(qtbot)
    label = field.combo.itemData(0)

    field.setText(label)

    assert field.text() == label


def test_more_backends_than_rows_does_not_overrun_the_combo(qtbot,
                                                            monkeypatch):
    """A backend registered after the panel was built is not written past the end.

    The entries were created once from ``backend_choices``; writing item text
    at an index the combo does not have is a Qt no-op on some builds and a
    crash on others, and either way the extra backend is not choosable.
    """
    from spacr import regression_backends

    field = _backend_field(qtbot)
    real_menu = regression_backends.backend_menu
    rows = field.combo.count()

    def _one_more(regression_type):
        statuses = list(real_menu(regression_type))
        extra = dict(statuses[0])
        extra.update(label="Newcomer", enabled=True, short_reason="",
                     reason="", summary="registered after the panel was built")
        return statuses + [extra]

    monkeypatch.setattr(regression_backends, "backend_menu", _one_more)

    field.refresh()

    assert field.combo.count() == rows
    assert "Newcomer" not in [field.combo.itemText(i) for i in range(rows)]


def test_nothing_unavailable_means_no_panel_to_open(qtbot, monkeypatch):
    """With every backend installed there is nothing to explain."""
    field = _backend_field(qtbot)
    monkeypatch.setattr(field, "unavailable_entries", lambda: [])

    assert field.show_availability_panel("anything") is None
    assert field.open_availability_panel() is None


def test_a_pointer_over_nothing_in_the_popup_opens_no_panel(qtbot):
    """A mouse move that lands on no row is ignored."""
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    field = _backend_field(qtbot)
    view = field.combo.view()
    viewport = view.viewport()
    move = QMouseEvent(QEvent.MouseMove, QPointF(-50.0, -50.0),
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)

    assert field.eventFilter(viewport, move) is False


class _StubIndex:
    def __init__(self, row, valid=True):
        self._row = row
        self._valid = valid

    def isValid(self):  # noqa: N802 - Qt name
        return self._valid

    def row(self):
        return self._row


class _StubView:
    """A popup view that reports a fixed row under the pointer."""

    def __init__(self, index):
        self._index = index

    def indexAt(self, _position):  # noqa: N802 - Qt name
        return self._index


def _move_event():
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(QEvent.MouseMove, QPointF(4.0, 4.0),
                       Qt.NoButton, Qt.NoButton, Qt.NoModifier)


def test_a_row_beyond_the_known_backends_opens_no_panel(qtbot, monkeypatch):
    """A row the availability table does not cover is passed over.

    The view and the table are two lists that can disagree for one repaint;
    indexing the shorter one by the longer one's row is an IndexError inside
    a mouse-move handler.
    """
    field = _backend_field(qtbot)
    monkeypatch.setattr(field, "availability_entries", lambda: [])
    opened = []
    monkeypatch.setattr(field, "show_availability_panel",
                        lambda *a, **k: opened.append(a))

    field._hover_popup_row(_StubView(_StubIndex(3)), _move_event())

    assert opened == []


def test_hovering_an_available_backend_dismisses_the_panel(qtbot,
                                                           monkeypatch):
    """Moving onto a row that needs no explanation hides the open panel.

    Leaving it up would explain the previous row while the pointer is over
    a different one.
    """
    from spacr.qt.widgets.availability_panel import AvailabilityPanel

    field = _backend_field(qtbot)
    monkeypatch.setattr(
        field, "availability_entries",
        lambda: [{"key": "ols", "title": "OLS", "enabled": True}])
    panel = AvailabilityPanel.instance()
    hidden = []
    monkeypatch.setattr(panel, "isVisible", lambda: True)
    monkeypatch.setattr(panel, "start_hide", lambda: hidden.append(True))

    field._hover_popup_row(_StubView(_StubIndex(0)), _move_event())

    assert hidden == [True]


def test_the_pointer_leaving_the_combo_dismisses_the_panel(qtbot,
                                                           monkeypatch):
    """The panel goes when the pointer leaves the control it explains."""
    from PySide6.QtCore import QEvent
    from spacr.qt.widgets.availability_panel import AvailabilityPanel

    field = _backend_field(qtbot)
    panel = AvailabilityPanel.instance()
    hidden = []
    monkeypatch.setattr(panel, "isVisible", lambda: True)
    monkeypatch.setattr(panel, "start_hide", lambda: hidden.append(True))

    field.eventFilter(field.combo, QEvent(QEvent.Leave))

    assert hidden == [True]


# -- the settings container ----------------------------------------------------

def test_a_canonical_reader_that_fails_leaves_the_value_alone(monkeypatch):
    """A reader that raises returns the widget's own answer unchanged.

    Dropping the value would silently reset the setting on every collect, and
    letting the failure out takes ``collect()`` -- and therefore the whole
    Start button -- down with it. The handler contains the failure and notes
    it through the module's own logger.
    """
    from spacr import settings

    def _explode(value):
        raise RuntimeError("cannot canonicalise that")

    monkeypatch.setattr(
        settings, "canonical_feature_selection", _explode, raising=False)

    widgets = sm.SettingsWidgets("regression")

    assert widgets._canonical("channel_of_interest", [3]) == [3]


def test_a_key_with_no_canonical_reader_is_returned_as_it_is():
    """Only the listed settings are re-read; everything else passes through."""
    widgets = sm.SettingsWidgets("regression")

    assert widgets._canonical("nucleus_channel", 1) == 1


def test_an_auto_capable_box_takes_auto_from_a_pushed_value(qapp):
    """Propagating "auto" into a box that offers it selects auto, not zero."""
    from PySide6.QtWidgets import QDoubleSpinBox

    box = QDoubleSpinBox()
    box.setRange(0.0, 10.0)
    box.setSpecialValueText(sm.AUTO_TEXT)
    box.setValue(4.0)

    widgets = sm.SettingsWidgets("regression")
    widgets._widgets["nms_threshold"] = box

    assert widgets.set_value_for_key("nms_threshold", "auto") is True
    assert box.value() == box.minimum()


def test_the_backend_follows_the_declared_default_when_nothing_shows_it(
        qtbot):
    """With no regression-type control on screen the declared default is used.

    That is what the run would fit anyway, so the greying matches the run.
    """
    widgets = sm.SettingsWidgets("regression")
    backend = sm._RegressionBackendField(regression_type=None)
    qtbot.addWidget(backend)
    widgets._widgets["regression_backend"] = backend
    widgets._defaults["regression_type"] = "ols"

    widgets._refresh_regression_backend()

    assert backend.regression_type() == "ols"


def test_a_regression_type_control_that_cannot_be_read_falls_back(qtbot,
                                                                  monkeypatch):
    """A widget whose value cannot be read uses the declared default.

    Leaving the backend judged against a stale family is how an entry stays
    greyed after the family that greyed it was changed.
    """
    from PySide6.QtWidgets import QComboBox

    widgets = sm.SettingsWidgets("regression")
    backend = sm._RegressionBackendField(regression_type=None)
    qtbot.addWidget(backend)
    widgets._widgets["regression_backend"] = backend
    widgets._widgets["regression_type"] = QComboBox()
    widgets._defaults["regression_type"] = "ridge"

    def _explode(_widget):
        raise RuntimeError("the control is half-built")

    monkeypatch.setattr(widgets, "_read_widget", _explode)

    widgets._refresh_regression_backend()

    assert backend.regression_type() == "ridge"


def test_no_constraint_registry_leaves_every_control_editable(monkeypatch,
                                                              qapp):
    """Without the advisor module nothing is greyed, and nothing raises.

    Greying a control the panel cannot justify is worse than leaving it
    editable: the user cannot tell why it is dead.
    """
    from PySide6.QtWidgets import QComboBox

    widgets = sm.SettingsWidgets("regression")
    unit = QComboBox()
    unit.addItem("cell")
    widgets._widgets["analysis_unit"] = unit
    monkeypatch.setitem(sys.modules, "spacr.settings_advisor", None)

    widgets._refresh_analysis_unit_lock()

    assert unit.isEnabled() is True


def test_a_registry_without_the_whole_table_locks_only_this_unit(monkeypatch,
                                                                 qapp):
    """Missing ``UNIT_REQUIREMENTS`` falls back to the current unit's keys.

    Refreshing only those leaves an older lock in place, which is why the
    whole table is preferred -- but it must not stop the panel building.
    """
    from PySide6.QtWidgets import QComboBox

    import spacr.settings_advisor as advisor

    widgets = sm.SettingsWidgets("regression")
    unit = QComboBox()
    unit.addItem("cell")
    widgets._widgets["analysis_unit"] = unit
    monkeypatch.setattr(advisor, "requirements_for_unit", lambda _unit: {})
    monkeypatch.delattr(advisor, "UNIT_REQUIREMENTS")

    widgets._refresh_analysis_unit_lock()

    assert unit.isEnabled() is True
    assert widgets._unit_locked == set()


def test_a_dependency_pass_that_fails_does_not_stop_the_unit_lock(monkeypatch,
                                                                  qapp):
    """The other greying rules failing leaves this rule's work applied."""
    from PySide6.QtWidgets import QComboBox

    widgets = sm.SettingsWidgets("regression")
    unit = QComboBox()
    unit.addItem("cell")
    widgets._widgets["analysis_unit"] = unit

    def _explode():
        raise RuntimeError("a dependency rule is broken")

    widgets._refresh_setting_dependencies = _explode

    widgets._refresh_analysis_unit_lock()

    assert isinstance(widgets._unit_locked, set)
