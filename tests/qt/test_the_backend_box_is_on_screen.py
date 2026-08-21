"""Instruction 141: the backend box is ON SCREEN, not merely written.

WHY THIS FILE EXISTS. `spacr.regression_backends.describe_backends` was
written, tested and complete on 2026-08-18 -- every backend, every summary,
every API link -- and NOTHING IN ``spacr/qt`` CALLED IT. The panel showed a
plain combo with all eight labels enabled: no descriptions, no links, and no
greying, so a user could pick ``numpyro (GPU)`` (no package here, and spaCR
routes no fit through it) and hear nothing about it until the run refused.

That is handoff section 0b in one sentence -- green tests, unreachable
feature -- so every assertion below is made through the REAL
``AppScreen("regression")``, on a shown widget, and uses ``isVisible()``
rather than ``isVisibleTo()``: the second answers "would this be visible if
its parents were", which is True inside a collapsed section, and a collapsed
section is exactly the failure mode being guarded against.

What is pinned here:

* the description is a widget on the screen, and its text names the backends
  and carries their API URLs AS ANCHORS a click can follow;
* it FOLLOWS ``regression_type`` -- an entry greyed for one family is
  choosable for another;
* each of the three refusals instruction 141 C names -- an incompatible
  regression type, a missing package, a missing CUDA device -- greys the
  entry out with ITS OWN reason, and that reason is readable without
  hovering anything.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


@pytest.fixture
def screen(qtbot):
    """The real regression settings screen, shown."""
    from spacr.qt.screens.app_screen import AppScreen

    panel = AppScreen("regression")
    qtbot.addWidget(panel)
    panel.resize(1400, 900)
    panel.show()
    qtbot.waitExposed(panel)
    return panel


@pytest.fixture
def field(screen):
    """The `regression_backend` control, with its section expanded."""
    from spacr.qt.screens.settings_model import _RegressionBackendField

    widget = screen._settings_model._widgets["regression_backend"]
    assert isinstance(widget, _RegressionBackendField), (
        "the backend setting is a plain combo again; it cannot say why an "
        "option is greyed out or what any of them are")
    for section in screen._settings_sections:
        if any(row is widget for _label, row in section._row_widgets):
            section.show()
            section.set_expanded(True)
    return widget


def _items(field):
    """``{item text: enabled}`` for every entry of the dropdown."""
    model = field.combo.model()
    return {field.combo.itemText(index): bool(model.item(index).isEnabled())
            for index in range(field.combo.count())}


def _entry_for(field, label):
    """The index of the entry whose VALUE is ``label``."""
    index = field.combo.findData(label)
    assert index >= 0, f"{label} is not offered at all"
    return index


# ---------------------------------------------------------------------------
# 1. It is on screen
# ---------------------------------------------------------------------------

def test_the_description_is_visible_on_the_real_screen(field):
    """isVisible(), not isVisibleTo(): the second is True in a closed box."""
    assert field.isVisible()
    assert field.description.isVisible()
    assert field.combo.isVisible()
    assert field.description.height() > 0
    assert field.description.width() > 0


def test_the_visible_text_describes_the_backends_and_carries_their_links(field):
    """"discribe all of the packages ... and linkt the the API for each"."""
    from spacr.regression_spec import REGRESSION_BACKENDS

    text = field.description.toPlainText()
    named = [name for name, spec in REGRESSION_BACKENDS.items()
             if spec["label"] in text]
    assert len(named) >= 3, f"only {named} appear in the box"
    # ALL EIGHT, in fact -- the >= 3 above is the floor the instruction sets,
    # and this is what the box actually promises.
    assert len(named) == len(REGRESSION_BACKENDS), sorted(
        set(REGRESSION_BACKENDS) - set(named))

    links = field.api_links()
    assert links, "no API URL is a link in the rendered document"
    assert any(url.startswith("https://") for url in links)
    for name, spec in REGRESSION_BACKENDS.items():
        assert spec["url"] in links, f"{name}'s API is not linked"


def test_the_links_are_clickable_rather_than_printed(field):
    """A URL a QTextBrowser will not open is a URL a user has to retype."""
    assert field.description.openExternalLinks() is True
    # The anchors are read back off the LAID-OUT DOCUMENT, so this is Qt
    # saying it parsed them, not the test finding a substring in a string it
    # supplied itself.
    assert len(field.api_links()) >= 3


def test_the_selected_backend_is_the_one_written_out_in_full(field, qtbot):
    """Instruction 135 keeps the panel to one page; the cost still shows.

    The seven a user did NOT pick are one line each. The one they DID pick
    carries its measured cost, because that is the run about to start
    (instruction 140).
    """
    from spacr.regression_spec import REGRESSION_BACKENDS

    field.set_value("torch (GPU)")
    text = field.description.toPlainText()
    assert REGRESSION_BACKENDS["torch"]["cost"][:40] in text
    # ...and the cost of the one that was not chosen is not also dumped in.
    assert REGRESSION_BACKENDS["glum"]["cost"] not in text
    assert REGRESSION_BACKENDS["glum"]["label"] in text


# ---------------------------------------------------------------------------
# 2. It follows the model being fitted
# ---------------------------------------------------------------------------

def test_the_box_and_the_greying_follow_regression_type(screen, field,
                                                        monkeypatch):
    """The torch backend is offered under `mixed` and refused under `lasso`.

    A description judged once when the panel was built would be wrong for
    every family the user picks afterwards.
    """
    from spacr import regression_backends

    monkeypatch.setattr(
        regression_backends, "cuda_present_without_importing_torch",
        lambda: True)
    types = screen._settings_model._widgets["regression_type"]

    types.setCurrentText("mixed")
    field.refresh()
    assert field.regression_type() == "mixed"
    mixed_items = _items(field)
    assert mixed_items[field.combo.itemText(_entry_for(field, "torch (GPU)"))]
    mixed_text = field.description.toPlainText()

    types.setCurrentText("lasso")
    assert field.regression_type() == "lasso"
    lasso_text = field.description.toPlainText()
    assert lasso_text != mixed_text, "the box did not follow regression_type"
    torch_entry = field.combo.itemText(_entry_for(field, "torch (GPU)"))
    assert not _items(field)[torch_entry]
    assert "lasso" in torch_entry


def test_auto_is_read_as_no_family_chosen_yet(screen, field):
    """'auto' is the readable spelling of the historical None.

    Only the default backend can promise to fit a family that will not be
    known until the response has been read, and the panel has to say so in
    the same words the run would.

    WHAT IT SAYS CHANGED ON 2026-08-21, and the assertion changed with it
    rather than being loosened. Every optional backend used to read
    "unavailable: needs an explicit regression type" -- seven identical
    lines naming what was MISSING and nothing about what any of them does
    or whether it is on the machine. Reported, and the request was
    explicit: "write the explisit regression type and what needs to be done
    if it is not installed. if it is intalled write installed."

    So the check is now for the two facts that replaced it.
    """
    types = screen._settings_model._widgets["regression_type"]
    types.setCurrentText("auto")
    assert field.regression_type() is None
    text = field.description.toPlainText()
    # THE PANEL RENDERS THE SHORT FORM, which is the one that goes inside
    # the dropdown entry -- so this asserts the short contract, not the long
    # sentence behind it.
    #
    # It names a family the backend can actually fit...
    assert "fits" in text
    assert "ols" in text or "mixed" in text
    # ...and whether choosing that family would be enough: "installed" for
    # one that is here, a pip command for one that is not.
    assert "installed" in text
    assert "pip install" in text
    assert not _items(field)[
        field.combo.itemText(_entry_for(field, "torch (GPU)"))]
    assert _items(field)[
        field.combo.itemText(_entry_for(field, "statsmodels (CPU)"))]


# ---------------------------------------------------------------------------
# 3. Three refusals, three reasons, each readable
# ---------------------------------------------------------------------------

def _reason_is_readable(field, label, short, full):
    """The reason is in the entry's own text, its tooltip, and the box.

    THREE PLACES BECAUSE TWO OF THEM ARE MISSABLE. A greyed row says only
    "not this one"; Qt shows an item tooltip lazily, under the cursor, and
    only while the popup is open. The box is on screen either way -- so the
    short form goes in the entry text and in the box, and the full sentence
    on the tooltip for whoever hovers it.

    :param short: the combo-entry-length refusal.
    :param full: a fragment of the sentence the run itself would print.
    """
    index = _entry_for(field, label)
    from PySide6.QtCore import Qt

    assert short in field.combo.itemText(index), "not in the entry text"
    tooltip = field.combo.itemData(index, Qt.ToolTipRole) or ""
    assert full in tooltip, f"not on the entry tooltip: {tooltip!r}"
    assert short in field.description.toPlainText(), "not in the box"


def test_an_incompatible_regression_type_greys_the_entry_with_its_reason(
        screen, field):
    """"cuML has no mixed model" is instruction 141 C's own example."""
    screen._settings_model._widgets["regression_type"].setCurrentText("mixed")
    assert not _items(field)[
        field.combo.itemText(_entry_for(field, "cuML (GPU)"))]
    _reason_is_readable(field, "cuML (GPU)", "no mixed model",
                        "cannot fit regression_type='mixed'")


def test_a_missing_package_greys_the_entry_with_its_pip_command(
        screen, field, monkeypatch):
    """Driven by forcing it, because torch IS installed on this machine.

    The six optional backends are inventory-only here -- not installed AND
    not routed through -- so this is the only way to reach the
    package-missing branch through the panel.
    """
    import spacr.regression_backends as backends

    monkeypatch.setattr(backends, "package_installed",
                        lambda name: name != "torch")
    screen._settings_model._widgets["regression_type"].setCurrentText("mixed")
    field.refresh()
    assert not _items(field)[
        field.combo.itemText(_entry_for(field, "torch (GPU)"))]
    _reason_is_readable(field, "torch (GPU)", "pip install torch",
                        "needs a package that is not installed")


def test_a_missing_gpu_greys_the_entry_with_its_reason(screen, field,
                                                       monkeypatch):
    """No silent fall back to the CPU under a user who chose the GPU."""
    import spacr.regression_backends as backends

    monkeypatch.setattr(backends, "cuda_present_without_importing_torch",
                        lambda: False)
    screen._settings_model._widgets["regression_type"].setCurrentText("mixed")
    field.refresh()
    assert not _items(field)[
        field.combo.itemText(_entry_for(field, "torch (GPU)"))]
    _reason_is_readable(field, "torch (GPU)", "needs a CUDA device",
                        "will not quietly run it on the CPU instead")


def test_the_six_unwired_backends_say_so_and_say_what_would_install_them():
    """Instruction 141 C, and honesty about what is inventory.

    Only statsmodels and torch fit anything today. The other six are
    described so the plan is visible -- and on this machine none of their
    packages is installed either, which is a SECOND reason and is stated
    alongside the first rather than instead of it. Installing pymer4 would
    not make it choosable, and neither would wiring it up alone.
    """
    from spacr.regression_backends import backend_status
    from spacr.regression_spec import REGRESSION_BACKENDS

    for name, spec in REGRESSION_BACKENDS.items():
        if spec["implemented"]:
            continue
        status = backend_status(name, spec["types"][0])
        assert not status["enabled"], name
        assert "does not route any fit through it yet" in status["reason"]
        assert spec["pip"] in status["reason"] or spec["pip"] is None, (
            f"{name}: the pip command is not on the greyed-out entry")


# ---------------------------------------------------------------------------
# 4. The value survives all of it
# ---------------------------------------------------------------------------

def test_the_panel_still_collects_the_label_it_always_did(screen, field):
    """The entry text carries the refusal, so the VALUE is not the text."""
    assert screen._settings_model.collect()[
        "regression_backend"] == "statsmodels (CPU)"
    field.set_value("torch")
    assert field.get_value() == "torch (GPU)"
    assert screen._settings_model.collect()[
        "regression_backend"] == "torch (GPU)"


def test_a_selection_the_new_type_cannot_fit_is_kept_and_refused_out_loud(
        screen, field):
    """Re-pointing it at statsmodels would be the silent fallback 141 forbids.

    The setting keeps the value the user chose, the box states the refusal in
    the words `spacr.ml._require_backend` will use, and the run is the thing
    that refuses -- not the panel, quietly, one step earlier.
    """
    field.set_value("torch (GPU)")
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    assert field.get_value() == "torch (GPU)"
    assert screen._settings_model.collect()[
        "regression_backend"] == "torch (GPU)"
    text = field.description.toPlainText()
    assert "This run will be refused." in text
    assert "cannot fit regression_type='ols'" in text


# ---------------------------------------------------------------------------
# Instruction 141 G: the two backends wired on 2026-08-18 reach the panel
# ---------------------------------------------------------------------------


def test_the_wired_backends_are_choosable_for_their_own_families(screen,
                                                                 field):
    """`pyfixest` under ols, `glum` under poisson -- ON THE REAL COMBO.

    Before this, the panel offered exactly two enabled entries for every
    family -- statsmodels always and torch under `mixed` -- and every other
    row read "is described here but spaCR does not route any fit through it
    yet". Two of them now do, and the only place that can prove it is the
    widget: a spec table saying ``implemented: True`` proves nothing about
    what a dropdown lets a user press.
    """
    pytest.importorskip("pyfixest")
    pytest.importorskip("glum")
    types = screen._settings_model._widgets["regression_type"]
    items = field.combo.model()

    def offered(label):
        return bool(items.item(_entry_for(field, label)).isEnabled())

    types.setCurrentText("ols")
    assert offered("pyfixest (CPU)")
    assert not offered("glum (CPU)"), "glum has no least-squares family"

    types.setCurrentText("poisson")
    assert offered("glum (CPU)")
    assert not offered("pyfixest (CPU)"), "pyfixest fits ols and wls"


def test_the_box_states_the_measured_cost_of_a_wired_backend(screen, field):
    """Instruction 141 B: measured, never "may be faster".

    Read off the SHOWN description widget rather than off
    `describe_backends`, because the compact box writes the SELECTED backend
    out in full and one line for each of the others -- so the measured
    numbers are on screen only when the backend is the one chosen.
    """
    pytest.importorskip("pyfixest")
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    index = _entry_for(field, "pyfixest (CPU)")
    field.combo.setCurrentIndex(index)
    assert field.description.isVisible()
    text = field.description.toPlainText()
    assert "16.7x" in text, text
    assert "3.9e-9" in text, "the box does not say what it agrees to"
    assert "may be faster" not in text.lower()


def test_the_box_says_what_the_absorbed_fit_does_not_report(screen, field):
    """141 D: where it cannot agree by construction, the box says so."""
    pytest.importorskip("pyfixest")
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    field.combo.setCurrentIndex(_entry_for(field, "pyfixest (CPU)"))
    text = field.description.toPlainText()
    assert "DIFFERENT ANSWER" in text
    assert "Intercept" in text


def test_the_permutation_resolution_reaches_the_setting_on_screen(screen,
                                                                  qtbot):
    """Instruction 149 C, on the widget rather than in the dict.

    ``guide_permutations`` is a QSpinBox and its hover help is the only place
    the arithmetic can be read, so this asserts the sentence is on the
    control the user actually points at.
    """
    widget = screen._settings_model._widgets["guide_permutations"]
    # INSTRUCTION 113 MOVED THE HOVER HELP OFF THE EDITABLE FIELD, so the
    # QSpinBox itself answers with an empty string and the sentence lives on
    # the setting's LABEL. Reading the spin box would have passed before this
    # change and failed after it, for a reason that has nothing to do with
    # what the sentence says.
    for section in screen._settings_sections:
        if any(row is widget for _label, row in section._row_widgets):
            section.show()
            section.set_expanded(True)
    # THE HOVER HELP IS INSTALLED BY THE APPLICABILITY REFRESH, not by the
    # first layout: instruction 106 re-greys every setting whenever one of
    # them changes, and that pass is what writes each control's hint. Reading
    # a freshly built panel finds an empty string, which is why this touches
    # a setting first rather than only expanding the section.
    screen._settings_model._widgets["regression_type"].setCurrentText("ols")
    qtbot.wait(100)
    # READ THE WAY THE HELP IS ACTUALLY DELIVERED. `install_api_tooltips`
    # renders the rich HTML into the `apiTooltipHtml` property and shows it
    # from an EVENT FILTER on the label; it then sets the editor's role to
    # "metadata" and clears both `setToolTip` strings on purpose. So
    # `toolTip()` is empty by design on this path, and reading it asserted
    # that a mechanism the panel does not use was not being used.
    #
    # The property is the contract every other reader uses -- the API dot,
    # the docs link and the localisation pass all read it.
    hint = ""
    for section in screen._settings_sections:
        for label, row in section._row_widgets:
            if row is widget:
                hint = (str(row.property("apiTooltipHtml") or "")
                        or str(label.property("apiTooltipHtml") or "")
                        or widget.toolTip() or label.toolTip())
    assert hint, "guide_permutations carries no hover help at all"
    assert "(exceedances + 1) / (permutations + 1)" in hint, (
        "the hover help gives the P-value FLOOR but not the estimator, so a "
        "reader cannot tell where their number came from")
    assert "1e-3" in hint
