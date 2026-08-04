"""The Dose–Response screen: does the grid say what the engine said?

The screen has no statistics of its own, so every assertion here is about
whether the engine's answer survives the trip to the widget. The frame it is
fed is built from
:func:`spacr.qt.widgets.dose_response.four_parameter_logistic` with four
deliberately different genes::

    geneA   a clean 4PL, EC50 1 µM        -> fitted, an EC50 and an interval
    geneB   a clean 4PL, EC50 0.05 µM     -> fitted, a different EC50
    geneC   bell-shaped (toxic top dose)  -> REFUSED, with the reason
    geneD   dilution series cut off short -> UNBOUNDED, "EC50 > 0.333 µM"

Two of those four are the reason the screen exists, and the assertions on them
are the sharp ones: a refused group and an unbounded group must both be
**visible rows in the table**, labelled as what they are, with an empty EC50
cell and the engine's sentence in the note. A screen that silently dropped
them would turn "we checked and the answer is no" into "no data".

Everything runs ``threaded=False``, which is the same code path the shipped
screen takes with a worker thread in between, and offscreen — no display
needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import (
    APP_KEY, APP_NAME, NOTE_WIDTH, TABLE_COLUMNS, DoseResponseScreen,
    make_dose_response_screen, register,
)
from spacr.qt.widgets.dose_response import (
    CI_WALD, STATUS_FITTED, STATUS_REFUSED, STATUS_UNBOUNDED,
    four_parameter_logistic,
)

pytestmark = pytest.mark.qt

DOSES = 27.0 / 3.0 ** np.arange(10)
TRUNCATED = DOSES[DOSES < 1.0][:6]


def _series(ec50, doses=DOSES, seed=0, hill=-1.0):
    rng = np.random.default_rng(seed)
    dose = np.repeat(np.asarray(doses, dtype=float), 3)
    clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(ec50), hill)
    return dose, clean + rng.normal(0.0, 1.0, dose.size)


@pytest.fixture()
def frame() -> pd.DataFrame:
    """Four genes: two that fit, one bell, one truncated."""
    parts = []
    for gene, ec50, seed in (("geneA", 1.0, 1), ("geneB", 0.05, 2)):
        dose, response = _series(ec50, seed=seed)
        parts.append(pd.DataFrame({"gene": gene, "conc_uM": dose,
                                   "signal": response,
                                   "well": [f"A{i:02d}" for i in
                                            range(dose.size)]}))
    dose = np.repeat(DOSES, 3)
    rise = four_parameter_logistic(dose, 0.0, 100.0, np.log10(0.3), 1.5)
    survival = four_parameter_logistic(dose, 0.0, 1.0, np.log10(3.0), -3.0)
    rng = np.random.default_rng(1)
    parts.append(pd.DataFrame({
        "gene": "geneC", "conc_uM": dose,
        "signal": rise * survival + rng.normal(0.0, 2.0, dose.size),
        "well": [f"C{i:02d}" for i in range(dose.size)]}))
    dose, response = _series(1.0, doses=TRUNCATED, seed=20260804)
    parts.append(pd.DataFrame({"gene": "geneD", "conc_uM": dose,
                               "signal": response,
                               "well": [f"D{i:02d}" for i in
                                        range(dose.size)]}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture()
def screen(qtbot, frame):
    """A screen with the frame loaded and the columns chosen, not yet fitted."""
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(frame, label="synthetic")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    widget.unit_edit.setText("µM")
    return widget


def _cell(widget, row, key):
    column = [k for k, _h in TABLE_COLUMNS].index(key)
    return widget.table.item(row, column).text()


def test_the_screen_offers_the_dose_column_and_not_the_well_id(screen):
    """The concentration picker is not the continuous-column picker.

    ``conc_uM`` has ten levels, which the shared column classifier calls
    categorical; ``well`` is a key. The first must be offered and the second
    must not.
    """
    options = [screen.concentration_picker.itemText(i)
               for i in range(screen.concentration_picker.count())]
    assert "conc_uM" in options
    assert "well" not in options
    assert "gene" not in options
    responses = [screen.response_picker.itemText(i)
                 for i in range(screen.response_picker.count())]
    assert "signal" in responses
    # The dose column is also pre-selected, because it is called conc_uM.
    assert screen.concentration_picker.currentText() == "conc_uM"


def test_the_screen_fits_every_group_and_shows_the_engine_numbers(screen):
    screen.fit()
    result = screen.result_set()
    assert result is not None
    assert result.groups == ("geneA", "geneB", "geneC", "geneD")
    assert screen.table.rowCount() == 4

    assert _cell(screen, 0, "group") == "geneA"
    assert _cell(screen, 0, "status") == "fitted"
    # Every number on screen is the engine's, formatted and nothing else.
    engine = result.fits[0].result
    assert float(_cell(screen, 0, "ec50")) == pytest.approx(engine.ec50,
                                                            rel=1e-3)
    assert float(_cell(screen, 0, "ec50")) == pytest.approx(1.0, rel=0.10)
    assert float(_cell(screen, 1, "ec50")) == pytest.approx(0.05, rel=0.10)
    assert float(_cell(screen, 0, "ec50_low")) < 1.0
    assert float(_cell(screen, 0, "ec50_high")) > 1.0


def test_a_refused_group_is_a_visible_row_with_its_reason(screen):
    screen.fit()
    assert _cell(screen, 2, "group") == "geneC"
    assert _cell(screen, 2, "status") == "refused"
    # No number is offered in place of the refusal.
    assert _cell(screen, 2, "ec50") == "—"
    assert _cell(screen, 2, "ec50_low") == "—"
    assert _cell(screen, 2, "hill") == "—"
    note = _cell(screen, 2, "note")
    assert "not monotone" in note
    # The cell elides; the tooltip has all of it.
    assert len(note) <= NOTE_WIDTH + 1
    column = [k for k, _h in TABLE_COLUMNS].index("note")
    assert "cytotoxicity" in screen.table.item(2, column).toolTip()
    assert screen.result_set().fits[2].status == STATUS_REFUSED


def test_an_unbounded_group_shows_the_one_sided_bound_and_no_ec50(screen):
    screen.fit()
    assert _cell(screen, 3, "group") == "geneD"
    assert _cell(screen, 3, "status") == "unbounded"
    assert _cell(screen, 3, "ec50") == "—"
    assert _cell(screen, 3, "ec50_low") == "—"
    assert _cell(screen, 3, "ec50_high") == "—"
    note = _cell(screen, 3, "note")
    assert note.startswith("EC50 > ")
    assert "0.333" in note
    # The Hill slope and R² are still there — the fit happened, the EC50 is
    # what is withheld.
    assert float(_cell(screen, 3, "hill")) < 0
    assert screen.result_set().fits[3].status == STATUS_UNBOUNDED


def test_selecting_a_row_shows_that_group_report(screen):
    screen.fit()
    # Row 0 is selected automatically after a fit.
    assert screen.report.toPlainText().startswith("4PL dose–response · geneA")
    screen.show_group(3)
    text = screen.report.toPlainText()
    assert "geneD" in text
    assert "ec50_bounded = False" in text
    assert "highest concentration tested" in text
    screen.show_group(2)
    refusal = screen.report.toPlainText()
    assert refusal.startswith("geneC: REFUSED")
    assert "cytotoxicity" in refusal
    # Out-of-range indices are a no-op, not a crash.
    screen.show_group(99)
    assert screen.report.toPlainText() == refusal


def test_the_figure_draws_points_a_curve_and_the_ec50_marker(screen):
    screen.fit()
    screen.show_group(0)
    axes = screen._figure.axes[0]
    assert axes.get_xscale() == "log"
    # Three groups produced a curve, so three point series and three lines,
    # plus one dashed EC50 line for the selected group.
    assert len(axes.lines) == 3 * 2 + 1
    assert axes.lines[-1].get_linestyle() == "--"
    # The shaded interval is drawn as a patch.
    assert axes.patches
    assert "conc_uM" in axes.get_xlabel() and "µM" in axes.get_xlabel()
    assert axes.get_ylabel() == "signal"

    # The unbounded group gets a bound marker at the edge instead.
    screen.show_group(3)
    axes = screen._figure.axes[0]
    assert axes.lines[-1].get_linestyle() == ":"
    engine = screen.result_set().fits[3].result
    assert axes.lines[-1].get_xdata()[0] == pytest.approx(engine.dose_max)


def test_the_ci_method_and_the_force_switch_reach_the_spec(screen):
    screen.ci_picker.setCurrentIndex(1)
    screen.force_check.setChecked(True)
    spec = screen.spec()
    assert spec.ci_method == CI_WALD
    assert spec.allow_non_monotone is True
    assert spec.concentration == "conc_uM"
    assert spec.response == "signal"
    assert spec.group == "gene"
    assert spec.unit == "µM"

    screen.fit()
    # Forced, the bell-shaped gene now fits — and is still not silent about it.
    assert screen.result_set().fits[2].result is not None
    screen.show_group(2)
    assert "monotonicity check, which failed" in screen.report.toPlainText()


def test_no_grouping_column_fits_the_whole_table_as_one_curve(qtbot, frame):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    only_a = frame[frame["gene"] == "geneA"].reset_index(drop=True)
    widget.set_frame(only_a)
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.fit()
    assert widget.table.rowCount() == 1
    assert _cell(widget, 0, "group") == "all rows"
    assert _cell(widget, 0, "status") == "fitted"
    assert widget.result_set().fits[0].result.ec50 == pytest.approx(1.0,
                                                                    rel=0.10)


def test_a_table_with_no_dose_column_says_so_instead_of_drawing(qtbot):
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({"gene": list("abcd"),
                                   "flag": [0, 1, 0, 1]}))
    assert widget.concentration_picker.count() == 0
    assert widget.fit_button.isEnabled() is False
    assert "dilution series" in widget.report.toPlainText()
    # Fit on an empty selection must not raise.
    widget.fit()


def test_a_bad_column_choice_lands_in_the_report_not_a_dialog(screen):
    screen.concentration_picker.addItem("missing")
    screen.concentration_picker.setCurrentText("missing")
    screen.fit()
    assert "not a column of this table" in screen.report.toPlainText()
    assert screen.result_set() is None


def test_the_worker_retires_and_the_screen_closes_clean(screen):
    screen.fit()
    assert screen.is_busy() is False
    assert screen.active_jobs() == 0
    screen.close()
    assert screen.active_jobs() == 0


@pytest.fixture()
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    keyboard binding for every test that runs afterwards, so the list object
    is restored in place rather than trusting an unregister call. The same
    fixture :mod:`tests.qt.test_control_chart_screen` uses, for the same
    reason.
    """
    from spacr.qt import app as app_mod
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    app_mod._refresh_sections()


def test_importing_the_module_does_not_touch_the_registry():
    """The row that switches this app on lives in ``spacr/qt/__init__.py``.

    Importing the screen — which a test, a notebook or another screen may do
    just to reach the class — must not mutate process-wide state. The app
    reaches ``APPS`` when something *runs*
    :func:`spacr.qt.register_self_registering_modules`, never as a side effect
    of an import.

    The obvious form of this test — "in ``APPS`` exactly when it is in
    ``SELF_REGISTERING_MODULES``" — was wrong in a way worth recording,
    because it passed for as long as the screen was unwired and then failed
    the moment the row was added. Being listed is not being registered: the
    list is read by :func:`spacr.qt.run`, and a test session that imports
    ``spacr.qt.app`` directly has never run it. What the docstring actually
    claims is that the *import* changes nothing, so that is what is measured
    here, by importing and comparing.
    """
    import importlib
    import sys

    from spacr.qt import SELF_REGISTERING_MODULES
    from spacr.qt.app import APPS

    assert "spacr.qt.screens.dose_response" in SELF_REGISTERING_MODULES, (
        "the screen is finished and tested but nothing runs its register(); "
        "add the row to spacr.qt.SELF_REGISTERING_MODULES")

    before = list(APPS)
    sys.modules.pop("spacr.qt.screens.dose_response", None)
    importlib.import_module("spacr.qt.screens.dose_response")
    assert list(APPS) == before


def test_registering_the_screen_reaches_every_reader_of_the_registry(
        registry_sandbox):
    """Driving `register()` is the same thing that one line will do."""
    from spacr import cli
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    from spacr.qt.screens import dose_response as screen_module

    app_mod = registry_sandbox
    already = any(row[0] == APP_KEY for row in app_mod.APPS)
    assert register() is not already
    assert register() is False          # idempotent, not a raise

    row = next(r for r in app_mod.APPS if r[0] == APP_KEY)
    assert row[0] == "dose_response"
    assert row[1] == APP_NAME
    # Design, not Explore: an EC50 is fitted to choose the concentration the
    # next experiment will use, the same job the Power screen beside it does
    # for n. Explore is for a question asked of a finished table; a dilution
    # series is the most planned experiment there is.
    assert row[3] == app_mod.SECTION_DESIGN
    assert app_mod.APP_FACTORIES[APP_KEY] is make_dose_response_screen
    assert app_mod.APP_STAGE[APP_KEY] == app_mod.STAGE_ALPHA
    assert APP_TITLES[APP_KEY] == APP_NAME
    assert APP_INTROS[APP_KEY] == screen_module.APP_INTRO
    assert cli.INTERACTIVE_ONLY[APP_KEY] == screen_module.APP_CLI_NOTE
    assert len(screen_module.APP_NAME_TRANSLATIONS) == 9
    assert all(name.strip() for name in screen_module.APP_NAME_TRANSLATIONS)


def test_the_factory_builds_a_screen(qtbot):
    widget = make_dose_response_screen()
    qtbot.addWidget(widget)
    assert isinstance(widget, DoseResponseScreen)
    assert widget.objectName() == "DoseResponseScreen"
