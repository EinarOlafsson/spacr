"""The Control Charts screen: the form, the picture, and the refusals.

Everything statistical is asserted against the engine in
``tests/test_control_chart.py``; this file only asserts that the screen shows
*that* result and no other. So every number checked here is compared with the
engine's own — the screen is allowed to be wrong about layout and never about
arithmetic.

The screen is built with ``threaded=False`` so the table read and the chart run
inline. :class:`spacr.qt.job_runner.JobRunner` emits the same signals in the
same order either way, which is the point of the flag: the test drives the real
code path rather than a synchronous stand-in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.control_chart import (
    ESTIMATOR_MOVING_RANGE, ESTIMATOR_ROBUST, ESTIMATOR_SUBGROUP_S,
    RULES_ALL, RULES_DEFAULT, RULES_LIMITS_ONLY, ControlChartSpec,
    control_chart,
)
from spacr.qt.screens.control_chart import (
    APP_KEY, RULE_SETS, ControlChartCanvas, ControlChartScreen,
    make_control_chart_screen,
)

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# The campaign
# ---------------------------------------------------------------------------

def campaign(plates: int = 30, wells: int = 3, seed: int = 11
             ) -> pd.DataFrame:
    """Thirty plates whose negative control drifts upward after plate twenty.

    Well-level rows, like a real ``measurements.db``: three control wells and
    three sample wells per plate, so the screen has to filter and group rather
    than being handed one row per plate.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(plates):
        drift = 0.0 if index < 20 else 2.4 * (index - 19)
        for _ in range(wells):
            rows.append({"plateID": f"P{index + 1:02d}",
                         "run_date": f"2026-01-{index + 1:02d}",
                         "well_type": "neg",
                         "signal": 100.0 + drift + rng.normal(0.0, 2.0)})
            rows.append({"plateID": f"P{index + 1:02d}",
                         "run_date": f"2026-01-{index + 1:02d}",
                         "well_type": "pos",
                         "signal": 20.0 + rng.normal(0.0, 2.0)})
            rows.append({"plateID": f"P{index + 1:02d}",
                         "run_date": f"2026-01-{index + 1:02d}",
                         "well_type": "sample",
                         "signal": 60.0 + rng.normal(0.0, 9.0)})
    return pd.DataFrame(rows)


@pytest.fixture
def screen(qtbot) -> ControlChartScreen:
    """A screen that runs its jobs inline, with the campaign loaded."""
    made = ControlChartScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def pick_control(screen: ControlChartScreen, level: str) -> None:
    """Tick exactly ``level`` in the control list, as a click would."""
    for index in range(screen._levels.count()):
        item = screen._levels.item(index)
        item.setSelected(item.text() == level)


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

def test_loading_a_table_guesses_the_columns_and_draws_a_chart(screen):
    screen.set_frame(campaign())

    assert screen._plate.currentText() == "plateID"
    assert screen._order.currentText() == "run_date"
    assert screen._value.currentText() == "signal"
    assert screen._control_column.currentText() == "well_type"
    assert [screen._levels.item(i).text()
            for i in range(screen._levels.count())] == ["neg", "pos", "sample"]

    result = screen.result
    assert result is not None
    assert len(result) == 30
    assert screen.report.toPlainText().startswith("Control chart of signal")
    # Something was actually painted: the centre line, the two limits, the
    # series, and a filled band per sigma zone.
    ax = screen.canvas.figure.axes[0]
    assert len(ax.lines) >= 4
    assert len(ax.collections) >= 3


def test_picking_the_control_narrows_the_chart_to_that_control(screen):
    frame = campaign()
    screen.set_frame(frame)
    pick_control(screen, "neg")

    result = screen.result
    assert result is not None
    assert result.control_column == "well_type"
    assert result.control_levels == ("neg",)
    assert result.n_control_rows == int((frame["well_type"] == "neg").sum())
    # Three control wells per plate, so this is an X-bar/S chart.
    assert result.estimator == ESTIMATOR_SUBGROUP_S
    assert result.subgroup_sizes.max() == 3


def test_the_screen_shows_exactly_what_the_engine_computes(screen):
    """No statistic is computed in the screen, and this is how that is kept true."""
    frame = campaign()
    screen.set_frame(frame)
    pick_control(screen, "neg")
    shown = screen.result

    expected = control_chart(frame, screen.spec())
    assert shown.plates == expected.plates
    assert shown.values == pytest.approx(expected.values)
    assert shown.centre == pytest.approx(expected.centre)
    assert shown.sigma == pytest.approx(expected.sigma)
    assert shown.violations == expected.violations
    assert screen.report.toPlainText() == expected.report()


def test_the_drift_is_caught_and_listed_in_the_violations_table(screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")

    result = screen.result
    assert result.violations, "a control drifting after plate 20 must be caught"
    assert screen.violations.rowCount() == len(result.violations)
    first = result.violations[0]
    assert screen.violations.item(0, 0).text().startswith(f"{first.rule} — ")
    assert screen.violations.item(0, 1).text() == ", ".join(first.plates)
    assert screen.violations.item(0, 3).text() == first.describe()
    # The flagged plates are the drifted end of the campaign, not a scatter
    # through it: the baseline is the first twenty, and rule 6 reaches back at
    # most one plate into it because its window is five wide.
    flagged = [int(p[1:]) for p, f in zip(result.plates, result.flagged) if f]
    assert flagged and min(flagged) >= 20
    assert 30 in flagged


def test_a_violating_plate_is_marked_on_the_chart_once_per_rule(screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")
    result = screen.result
    ax = screen.canvas.figure.axes[0]

    rules = sorted({v.rule for v in result.violations})
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    for rule in rules:
        assert any(label.startswith(f"rule {rule} — ") for label in labels)


def test_changing_the_rule_set_changes_what_is_reported(screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")

    screen._rules.setCurrentIndex(
        [rules for _label, rules in RULE_SETS].index(RULES_ALL))
    everything = screen.result
    screen._rules.setCurrentIndex(
        [rules for _label, rules in RULE_SETS].index(RULES_LIMITS_ONLY))
    limits_only = screen.result

    assert everything.rules == RULES_ALL
    assert limits_only.rules == RULES_LIMITS_ONLY
    assert len(limits_only.violations) < len(everything.violations)
    assert limits_only.false_alarm_rate() < everything.false_alarm_rate()


def test_changing_the_estimator_changes_where_sigma_comes_from(screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")
    classical = screen.result

    index = screen._estimator.findData(ESTIMATOR_ROBUST)
    assert index >= 0
    screen._estimator.setCurrentIndex(index)
    robust = screen.result

    assert classical.estimator == ESTIMATOR_SUBGROUP_S
    assert robust.estimator == ESTIMATOR_ROBUST
    assert robust.sigma != pytest.approx(classical.sigma)
    assert "robust" in screen.report.toPlainText()


def test_shortening_the_baseline_moves_the_limits(screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")
    long_baseline = screen.result
    screen._baseline.setValue(10)
    short_baseline = screen.result

    assert len(long_baseline.baseline_plates) == 20
    assert len(short_baseline.baseline_plates) == 10
    assert short_baseline.baseline_plates == long_baseline.baseline_plates[:10]


def test_clearing_the_order_column_makes_the_chart_say_the_order_was_guessed(
        screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")
    assert not screen.result.order_inferred

    screen._order.setCurrentText("")
    assert screen.result.order_inferred
    assert "INFERRED" in screen.report.toPlainText()
    assert "INFERRED" in screen.canvas.figure.axes[0].get_xlabel()


def test_the_zprime_toggle_charts_the_assay_window(screen):
    screen.set_frame(campaign())
    screen._positive.setCurrentText("pos")
    screen._negative.setCurrentText("neg")
    screen._zprime.setChecked(True)

    result = screen.result
    assert result is not None
    assert result.value_column == "zprime"
    assert result.estimator == ESTIMATOR_MOVING_RANGE
    assert not result.order_inferred
    assert screen.canvas.figure.axes[0].get_ylabel() == "zprime"

    screen._zprime.setChecked(False)
    assert screen.result.value_column == "signal"


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_refused_chart_becomes_a_message_not_a_traceback(screen):
    failures = []
    screen.failed.connect(failures.append)
    screen.set_frame(campaign(plates=4))

    assert screen.result is None
    assert failures and "smallest defensible baseline is 8" in failures[0]
    assert screen.report.toPlainText() == failures[0]
    assert screen.violations.rowCount() == 0
    # The message is on the canvas too, not just in the panel below it.
    ax = screen.canvas.figure.axes[0]
    assert any("baseline" in text.get_text() for text in ax.texts)


def test_a_constant_control_says_so_rather_than_flagging_every_plate(screen):
    frame = campaign()
    frame["signal"] = 42.0
    screen.set_frame(frame)
    # A control that never moved is not a *continuous* column, so the picker's
    # normal offer is empty and it falls back to the numeric ones — which is
    # what keeps the degenerate case reachable at all from the screen.
    assert screen._value.currentText() == "signal"
    pick_control(screen, "neg")

    result = screen.result
    assert result is not None and result.degenerate
    assert screen.violations.rowCount() == 0
    assert "Sigma came out zero" in " ".join(result.caveats())


def test_recomputing_before_a_table_is_loaded_does_nothing(screen):
    screen.recompute()
    assert screen.result is None
    assert not screen.is_busy()


# ---------------------------------------------------------------------------
# Export and lifecycle
# ---------------------------------------------------------------------------

def test_exporting_writes_one_row_per_plate(screen, tmp_path):
    screen.set_frame(campaign())
    pick_control(screen, "neg")
    path = tmp_path / "control_chart.csv"

    assert screen.export_points(str(path)) == str(path)
    written = pd.read_csv(path)
    assert len(written) == len(screen.result)
    assert list(written["plate"]) == list(screen.result.plates)
    assert "rules" in written.columns


def test_exporting_before_anything_is_charted_refuses_politely(screen,
                                                               tmp_path):
    assert screen.export_points(str(tmp_path / "nothing.csv")) is None
    assert "Nothing charted yet" in screen._source.text()


def test_the_canvas_draws_a_message_when_it_has_no_result(qtbot):
    canvas = ControlChartCanvas()
    qtbot.addWidget(canvas)
    canvas.set_result(None, message="pick a measurement")
    ax = canvas.figure.axes[0]
    assert [text.get_text() for text in ax.texts] == ["pick a measurement"]
    canvas.close()


def test_loading_a_csv_from_disk_goes_through_the_same_path(screen, tmp_path):
    path = tmp_path / "campaign.csv"
    campaign().to_csv(path, index=False)
    screen.load_path(str(path))

    assert screen.result is not None
    assert len(screen.result) == 30
    assert "campaign.csv" in screen._source.text()


def test_an_unreadable_file_is_reported_inline_and_never_as_a_dialog(screen,
                                                                     tmp_path):
    path = tmp_path / "not_a_database.db"
    path.write_bytes(b"this is not sqlite")
    screen.load_path(str(path))
    assert "could not read not_a_database.db" in screen._source.text()


def test_the_screen_closes_without_leaving_work_running(screen):
    screen.set_frame(campaign())
    screen.close()
    assert screen.active_jobs() == 0


# ---------------------------------------------------------------------------
# The registry seam
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_sandbox():
    """Restore the whole app registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    keyboard binding for every test that runs afterwards, so the list object is
    restored in place rather than trusting an unregister call.

    **The side tables have to come back too.** ``register_app`` fans the app's
    strings out through ``app._META_TARGETS`` into ``cli.INTERACTIVE_ONLY``,
    ``app_screen.APP_TITLES`` / ``APP_INTROS`` and
    ``settings_model._APP_API_MODULE``. Rolling back ``APPS`` alone left this
    app's "GUI only" sentence in ``cli.INTERACTIVE_ONLY`` with no row in
    ``APPS`` behind it — a module that does not exist still answering
    ``spacr-run`` with a helpful lie, which is precisely what
    ``tests/test_app_registry_parity.py::
    test_the_gui_only_list_holds_no_apps_that_no_longer_exist`` is for. It only
    failed when this file happened to run first, which is the worst way for it
    to fail. Driven off ``_META_TARGETS`` so a fifth side table is undone
    without this fixture being edited.

    **It also rolls this one key FORWARD to "not registered" before the test.**
    ``spacr.qt.SELF_REGISTERING_MODULES`` now carries
    ``spacr.qt.screens.control_chart``, so any earlier test in the same process
    that calls ``spacr.qt.register_self_registering_modules()`` —
    ``tests/qt/test_settings_search.py`` does, to render the real stylesheet —
    leaves the row already in place. ``register()`` then answers ``False`` on
    its first call, and the test below is asserting the collection order of the
    suite rather than the seam. Clearing the key first makes the drive genuine
    in either order; the wholesale restore afterwards still puts back whatever
    was there.
    """
    import sys

    from spacr.qt import app as app_mod
    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    side = []
    for module_name, attribute, _field in app_mod._META_TARGETS:
        module = sys.modules.get(module_name)
        table = getattr(module, attribute, None) if module else None
        if isinstance(table, dict):
            side.append((table, dict(table)))

    for row in [r for r in app_mod.APPS if r[0] == APP_KEY]:
        app_mod.APPS.remove(row)
    app_mod.APP_FACTORIES.pop(APP_KEY, None)
    app_mod.APP_STAGE.pop(APP_KEY, None)
    app_mod.APP_META.pop(APP_KEY, None)
    for table, _saved in side:
        table.pop(APP_KEY, None)
    app_mod._refresh_sections()

    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    for table, saved in side:
        table.clear()
        table.update(saved)
    app_mod._refresh_sections()


def test_the_registry_list_says_so_and_a_launch_registers_the_screen(
        qtbot, registry_sandbox):
    """Inverted 2026-08-04. This used to read "the screen is NOT registered
    until the registry list says so" and assert the row was absent, because
    the seam shipped switched off — the row belonged to a file this one does
    not own.

    The row now exists: ``spacr.qt.SELF_REGISTERING_MODULES`` carries
    ``spacr.qt.screens.control_chart``, so every launched app has Control
    Charts in the registry. Left as it was, this test pinned the feature as
    unreachable and would have kept a finished screen invisible — and it only
    went red when something earlier in the process happened to run the
    registration pass, so it read as a flake rather than as stale. It asserts
    the switched-ON state instead: the module is listed, and the launch-time
    call is what puts the row into ``APPS``.
    """
    import spacr.qt

    app_mod = registry_sandbox
    assert ("spacr.qt.screens.control_chart"
            in spacr.qt.SELF_REGISTERING_MODULES)
    # The sandbox cleared the key, so the row below is this call's doing.
    assert not any(row[0] == APP_KEY for row in app_mod.APPS)

    registered = spacr.qt.register_self_registering_modules()

    assert "spacr.qt.screens.control_chart" in registered
    assert any(row[0] == APP_KEY for row in app_mod.APPS)
    qtbot.addWidget(make_control_chart_screen())


def test_registering_the_screen_reaches_every_reader_of_the_registry(
        registry_sandbox):
    """Driving `register()` is the same thing the one line will do."""
    from spacr.qt.screens import control_chart as screen_module
    app_mod = registry_sandbox

    assert screen_module.register() is True
    assert screen_module.register() is False        # idempotent, not a raise

    row = next(r for r in app_mod.APPS if r[0] == APP_KEY)
    assert row[0] == "control_chart"
    assert row[1] == screen_module.APP_NAME
    assert row[3] == app_mod.SECTION_RESULTS
    assert app_mod.APP_FACTORIES[APP_KEY] is make_control_chart_screen
    assert app_mod.APP_STAGE[APP_KEY] == app_mod.STAGE_ALPHA

    from spacr import cli
    from spacr.qt.screens.app_screen import APP_INTROS, APP_TITLES
    assert APP_TITLES[APP_KEY] == screen_module.APP_NAME
    assert APP_INTROS[APP_KEY] == screen_module.APP_INTRO
    assert cli.INTERACTIVE_ONLY[APP_KEY] == screen_module.APP_CLI_NOTE
    assert len(screen_module.APP_NAME_TRANSLATIONS) == 9


def test_the_default_rule_set_is_the_first_one_offered():
    assert RULE_SETS[0][1] == RULES_DEFAULT
    assert {rules for _label, rules in RULE_SETS} == {
        RULES_DEFAULT, RULES_ALL, RULES_LIMITS_ONLY}


def test_the_spec_the_form_builds_round_trips_through_json(screen):
    screen.set_frame(campaign())
    pick_control(screen, "neg")
    spec = screen.spec()
    assert ControlChartSpec.from_json(spec.to_json()) == spec
    assert spec.value == "signal" and spec.plate == "plateID"
    assert spec.control_levels == ("neg",)
