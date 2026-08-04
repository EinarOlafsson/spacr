"""The Power / Design screen: does it answer the question, and honestly?

These are not smoke tests. The screen's whole value is that the number it
prints is the number the library computes, so the suite runs real sweeps —
small ones, but real fits of the real model — and checks:

* **the GUI path is the library path.** A sweep started from the screen and
  the identical :func:`spacr.power_model.scan_parameters` call typed at a
  prompt produce the same rows, run keys and metrics for the same seed. That
  is the assertion that stops the screen from ever growing maths of its own.
* **a null design reports near chance**, not a flattering number, and its
  average precision sits on the prevalence baseline.
* **the power curve rises with cells per well** where the model requires it —
  a starved design cannot see what a fed one can.
* **the caveats are on screen**, in the visible half of the panel, including
  the one that says the R package overstates power.
* **the worker thread retires** — ``active_jobs()`` returns to 0 after a run
  completes AND after a cancel. This codebase has a live thread-teardown
  defect (``thread.finished`` connected to a closure makes the QThread the
  receiver, and ``make_thread``'s own ``deleteLater`` then eats the call), so
  this is a regression test for a bug that has bitten four screens already.

Sweeps are shared through module-scoped fixtures: a fit is a second even at
this size, and re-running the same one per test would put minutes on the
suite for no extra coverage.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr import power_model as pm
from spacr.qt.screens.power import (
    APP_KEY,
    APP_NAME,
    CaveatPanel,
    PowerCurveView,
    PowerScreen,
    make_power_screen,
    power_default_settings,
    register,
    register_settings,
    run_power_sweep,
    spec_from_settings,
)
from spacr.qt.widgets.power_design import (
    CAVEATS,
    DesignSpec,
    cells_grid,
    changes_the_number,
    simulator_kwargs,
    wells_grid,
)


#: Small enough that nine fits take about ten seconds, real enough that the
#: model actually converges at the fed end of the curve and actually fails to
#: at the starved end — which is the behaviour the power curve is made of.
def _tiny(**changes) -> DesignSpec:
    base = dict(
        n_genes=24, n_grnas_per_gene=1, cells_per_well=64.0,
        wells_per_plate=96, n_plates=1, constructs_per_well=4.0,
        background_positive_rate=0.10, effect_fold=6.0, hit_rate=0.25,
        reads_per_well=8000.0, gene_abundance_alpha=5.0,
        cells_per_well_var=200.0, class_pos_var=0.005, class_neg_var=0.005,
        sequencing_cells_per_well=300.0, pcr_factor_mu=1.0,
        pcr_factor_var=0.3, read_depth_cv=0.0,
        n_replicates=3, detection_auroc=0.80, seed=11, backend="torch",
    )
    base.update(changes)
    return DesignSpec(**base)


#: ADVI settings that keep a fit under a second without stopping it
#: converging where the design supports it.
FAST_FIT = {"n_steps": 400, "n_draws": 128}

#: Even smaller, for the threading tests, where the point is the QThread and
#: not the statistics.
def _threading_spec() -> DesignSpec:
    return _tiny(n_genes=16, cells_per_well=32.0, n_replicates=1)


THREAD_FIT = {"n_steps": 80, "n_draws": 32}


def _run_inline(qapp, spec, fit_kwargs=FAST_FIT):
    """Build a screen, run it on the calling thread, hand it back."""
    screen = PowerScreen(threaded=False)
    screen.set_spec(spec)
    screen.fit_kwargs = dict(fit_kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert screen.run() is True, screen.status_text()
    return screen


@pytest.fixture(scope="module")
def signal_screen(qapp):
    """A screen that has run a design with a real 6-fold effect."""
    screen = _run_inline(qapp, _tiny())
    yield screen
    screen.close()
    screen.deleteLater()


@pytest.fixture(scope="module")
def null_screen(qapp):
    """A screen that has run a design with NO effect at all."""
    screen = _run_inline(qapp, _tiny(effect_fold=1.0))
    yield screen
    screen.close()
    screen.deleteLater()


# ---------------------------------------------------------------------------
# The screen is the library
# ---------------------------------------------------------------------------

def test_the_screen_reproduces_the_librarys_own_numbers_for_a_fixed_seed(
        signal_screen):
    """The GUI is a caller of ``scan_parameters``, never a reimplementation.

    The same design, the same seed and the same backend must give the same
    rows whether the sweep was started by pressing Run or typed at a Python
    prompt. Anything less means the screen has an opinion of its own about
    the statistics, and a screen with an opinion is a screen that can be
    wrong in a way the library's 112 tests cannot catch.
    """
    spec = signal_screen.spec()
    from_screen = signal_screen.result()["cells_scan"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = pm.scan_parameters(
            **{**simulator_kwargs(spec),
               "imaging_n_cells_per_well_mu": cells_grid(spec)},
            n_replicates=spec.n_replicates,
            backend=spec.backend,
            seed=spec.seed,
            fit_kwargs=FAST_FIT,
        )

    assert len(from_screen) == len(direct) == len(cells_grid(spec)) * 3
    assert list(from_screen["run_key"]) == list(direct["run_key"])
    assert list(from_screen["status"]) == list(direct["status"])
    assert list(from_screen["seed_used"]) == list(direct["seed_used"])
    assert np.allclose(from_screen["model_auroc"].astype(float),
                       direct["model_auroc"].astype(float), equal_nan=True)
    assert np.allclose(from_screen["model_ap"].astype(float),
                       direct["model_ap"].astype(float), equal_nan=True)


def test_the_wells_sweep_is_the_library_call_too(signal_screen):
    """The second axis gets the same treatment as the first."""
    spec = signal_screen.spec()
    from_screen = signal_screen.result()["wells_scan"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        direct = pm.scan_parameters(
            **{**simulator_kwargs(spec),
               "n_wells_per_screen": [int(w) for w in wells_grid(spec)]},
            n_replicates=spec.n_replicates,
            backend=spec.backend,
            seed=spec.seed,
            fit_kwargs=FAST_FIT,
        )
    assert list(from_screen["run_key"]) == list(direct["run_key"])
    assert np.allclose(from_screen["model_auroc"].astype(float),
                       direct["model_auroc"].astype(float), equal_nan=True)


def test_a_spec_survives_a_round_trip_through_the_form(qapp):
    """Fields the form does not show are carried, not silently defaulted.

    ``gene_abundance_alpha`` and the classifier variances are not on the
    form. If ``set_spec`` dropped them, a design set programmatically would
    be simulated with the real screen's skew instead of the one asked for and
    nothing on screen would say so.
    """
    screen = PowerScreen(threaded=False)
    try:
        spec = _tiny(gene_abundance_alpha=2.5, class_pos_var=0.02,
                     read_depth_cv=0.4, imaging_split="uniform")
        screen.set_spec(spec)
        assert screen.spec() == spec
        assert "gene_abundance_alpha=2.5" in screen._held_note.text()
        assert "imaging_split=uniform" in screen._held_note.text()
    finally:
        screen.close()
        screen.deleteLater()


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

def test_a_design_with_a_real_effect_is_detected_and_said_so_in_words(
        signal_screen):
    sentence = signal_screen.answer_text()
    assert "64 cells per well" in sentence
    assert "of simulations" in sentence
    assert "AUROC" in sentence
    curve = signal_screen.result()["cells_curve"]
    fed = curve.loc[curve["value"] == 64.0].iloc[0]
    assert fed["power"] == 1.0
    assert fed["mean_auroc"] > 0.8


def test_a_null_design_reports_near_chance_and_not_a_flattering_number(
        null_screen, signal_screen):
    """No effect must read as no effect.

    Three separate ways of being wrong are checked, because a power screen
    that flatters a null design is worse than no power screen at all:

    * the detection probability at the user's own design is 0;
    * the mean AUROC of the fits that converged sits near 0.5 — not near 1,
      and not the ``1 - AUROC`` a flipped sign convention would give;
    * average precision does not beat the prevalence baseline by much, which
      is the metric that would otherwise look impressive purely because a
      quarter of this small library is a hit.

    And the comparison that ties it down: the same design with a 6-fold
    effect scores far higher on the same seed, so this is the design being
    null rather than the pipeline being broken.
    """
    curve = null_screen.result()["cells_curve"]
    design_point = curve.loc[curve["value"] == 64.0].iloc[0]
    assert design_point["power"] == 0.0
    assert "0% of simulations" in null_screen.answer_text()

    converged = curve.loc[curve["n_ok"] > 0]
    assert len(converged) >= 1, "nothing converged, so 'near chance' is untested"
    assert converged["mean_auroc"].between(0.3, 0.7).all(), \
        converged[["value", "mean_auroc"]]
    assert (converged["mean_ap"] - converged["ap_baseline"]).max() < 0.25

    signal = signal_screen.result()["cells_curve"]
    signal_point = signal.loc[signal["value"] == 64.0].iloc[0]
    assert signal_point["mean_auroc"] > design_point["mean_auroc"] + 0.2


def test_the_power_curve_rises_with_cells_per_well(signal_screen):
    """More cells cannot buy less information, and the curve shows it.

    Starving a well of imaged cells starves the Poisson offset: at the bottom
    of this grid the ADVI fit does not converge at all, and at the top every
    replicate clears the bar. The monotonicity is asserted on the pinned seed,
    which makes it deterministic; it was separately confirmed to hold at
    seeds 11, 12, 13 and 101, so it is a property of the model and not of one
    lucky draw.
    """
    curve = signal_screen.result()["cells_curve"]
    assert len(curve) == len(cells_grid(signal_screen.spec()))
    power = curve["power"].tolist()
    assert power == sorted(power), curve[["value", "power", "n_ok"]]
    assert power[0] < power[-1]
    assert power[-1] == 1.0
    # The starved end is starved because the fit dies, not because the model
    # ranks badly — which is exactly why those replicates must count against
    # the design rather than being dropped.
    assert curve.iloc[0]["n_not_converged"] + curve.iloc[0]["n_failed"] > 0


def test_a_withheld_metric_is_a_dash_on_screen_and_never_one_half(null_screen):
    """A non-converged point prints "—". 0.50 would read as chance."""
    rows = null_screen.table_rows()
    assert rows
    withheld = [row for row in rows if row[7] != "0" or row[8] != "0"]
    assert withheld, "the null design should have produced some unusable fits"
    for row in withheld:
        # power is still a real fraction; the metrics are the withheld part
        assert row[2].endswith("%")
    assert any(row[4] == "—" for row in rows), rows


def test_the_table_reports_the_denominator_beside_every_power(signal_screen):
    """"1 of 3" beside "33%" is what stops a percentage over three replicates
    being read as a percentage over three hundred."""
    for row in signal_screen.table_rows():
        assert "/" in row[3]
        detected, total = row[3].split("/")
        assert 0 <= int(detected) <= int(total)


def test_both_curves_are_drawn_with_the_values_that_were_simulated(
        signal_screen):
    cells = signal_screen._cells_view.describe()
    wells = signal_screen._wells_view.describe()
    for value in cells_grid(signal_screen.spec()):
        assert f"{value:g}=" in cells
    for value in wells_grid(signal_screen.spec()):
        assert f"{value:g}=" in wells
    assert not signal_screen._cells_view.is_empty()


# ---------------------------------------------------------------------------
# The caveats
# ---------------------------------------------------------------------------

def test_the_caveats_that_change_the_answer_are_visible_without_a_click(qapp):
    """They belong next to the number, not one import away.

    The port's departures from spaCRPower are recorded in its docstrings,
    which is the right place for a reader who already opened the file and no
    place at all for the person about to quote the number in a grant.
    """
    panel = CaveatPanel()
    try:
        visible = panel.visible_text()
        for caveat in changes_the_number():
            assert caveat.headline in visible, caveat.key
        assert "OVERSTATES" in visible
    finally:
        panel.deleteLater()


def test_the_screen_shows_the_caveats_beside_the_answer(signal_screen):
    visible = signal_screen.visible_caveat_text()
    assert "OVERSTATES" in visible
    assert "non-detection" in visible
    assert "ADVI" in visible
    # The rest are held, not lost.
    everything = signal_screen.caveat_text()
    for caveat in CAVEATS:
        assert caveat.headline in everything, caveat.key


def test_the_secondary_caveats_are_one_click_away(qapp):
    panel = CaveatPanel()
    try:
        assert "COM-Poisson" not in panel.visible_text()
        panel._more.setChecked(True)
        assert "COM-Poisson" in panel.visible_text()
        panel._more.setChecked(False)
        assert "COM-Poisson" not in panel.visible_text()
    finally:
        panel.deleteLater()


def test_every_caveat_carries_its_explanation_as_a_tooltip(qapp):
    panel = CaveatPanel()
    try:
        by_key = {label.property("caveatKey"): label
                  for label in panel._labels}
        assert set(by_key) == {caveat.key for caveat in CAVEATS}
        for caveat in CAVEATS:
            assert by_key[caveat.key].toolTip() == caveat.detail
    finally:
        panel.deleteLater()


def test_a_run_that_withholds_fits_says_so_in_the_status_line(null_screen):
    status = null_screen.status_text()
    assert "no usable fit" in status
    assert "non-detections" in status
    assert not null_screen.status_is_error()


# ---------------------------------------------------------------------------
# Threading — the part with a live defect behind it
# ---------------------------------------------------------------------------

def test_the_worker_thread_retires_after_a_run_completes(qtbot):
    """``active_jobs()`` returns to 0, which is not automatic here.

    ``make_thread`` connects ``thread.finished -> thread.deleteLater`` FIRST.
    Slots run in connection order, so a retirement slot connected as a
    *closure* — which PySide6 delivers with the QThread itself as receiver —
    has its metacall discarded, because Qt drops queued events for a
    destroyed receiver. The job then never leaves ``_jobs`` and every
    ``waitUntil(active_jobs() == 0)`` in the suite times out with the
    QThread's C++ half already gone. The fix, and what this test pins, is
    that the slot is a bound method of the screen.
    """
    screen = PowerScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_spec(_threading_spec())
    screen.fit_kwargs = dict(THREAD_FIT)

    with qtbot.waitSignal(screen.job_finished, timeout=180000) as blocker:
        assert screen.run() is True
        assert screen.active_jobs() >= 1
        assert screen.is_busy()

    assert blocker.args == [True]
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)
    assert not screen.is_busy()
    assert screen.result() is not None
    assert len(screen.table_rows()) > 0


def test_the_worker_thread_retires_after_a_cancel(qtbot):
    """Stop must also leave the registry empty, not merely stop the maths."""
    screen = PowerScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_spec(_threading_spec())
    screen.fit_kwargs = dict(THREAD_FIT)

    with qtbot.waitSignal(screen.job_finished, timeout=180000):
        assert screen.run() is True
        screen.cancel()

    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)
    assert not screen.is_busy()


def test_a_cancelled_sweep_keeps_the_points_that_finished(qtbot):
    """Stop is not delete.

    The user pressed it because they had seen enough. The finished grid
    points are real fits; throwing them away would make Stop destructive, and
    the partial curve is labelled partial so it cannot be misread as a design
    that ran out of power.
    """
    screen = PowerScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_spec(_threading_spec())
    screen.fit_kwargs = dict(THREAD_FIT)

    with qtbot.waitSignal(screen.job_finished, timeout=180000):
        assert screen.run() is True
        screen.cancel()
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)

    result = screen.result()
    if result is None:
        # The cancel landed before the worker reached its first checkpoint —
        # legitimate, and it has to say so rather than look like a failure.
        assert "Stopped before the first fit" in screen.status_text()
        return
    assert result["cancelled"] is True
    assert "Stopped early" in screen.status_text()
    assert len(result["cells_scan"]) < (
        len(cells_grid(screen.spec())) * screen.spec().n_replicates)


def test_progress_reaches_the_gui_thread_as_a_signal(qtbot):
    """Progress travels as printed text, which is the one safe mechanism.

    A signal emitted from the worker thread and connected with a
    DirectConnection would touch widgets off the GUI thread — the failure
    ``bridge.PipelineWorker`` documents having aborted the process once. So
    the job prints ``Progress: n/total``, ``PipelineWorker`` re-emits it as
    ``line_ready``, and the screen turns it back into a signal here.
    """
    screen = PowerScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_spec(_threading_spec())
    screen.fit_kwargs = dict(THREAD_FIT)

    seen = []
    screen.progressed.connect(lambda done, total: seen.append((done, total)))
    with qtbot.waitSignal(screen.job_finished, timeout=180000):
        assert screen.run() is True
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)

    assert seen, "no progress reached the GUI thread"
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert all(1 <= done <= total for done, total in seen)


def test_a_second_run_is_refused_while_one_is_in_flight(qtbot):
    screen = PowerScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_spec(_threading_spec())
    screen.fit_kwargs = dict(THREAD_FIT)
    with qtbot.waitSignal(screen.job_finished, timeout=180000):
        assert screen.run() is True
        assert screen.run() is False
    qtbot.waitUntil(lambda: screen.active_jobs() == 0, timeout=30000)


def test_closing_a_screen_mid_sweep_does_not_leave_a_running_thread(qtbot):
    """A QThread garbage-collected while running takes the process down."""
    screen = PowerScreen(threaded=True)
    qtbot.addWidget(screen)
    screen.set_spec(_threading_spec())
    screen.fit_kwargs = dict(THREAD_FIT)
    assert screen.run() is True
    screen.close()
    for thread, _worker in list(screen._jobs):
        assert not thread.isRunning()


# ---------------------------------------------------------------------------
# Refusing to run a design that cannot be simulated
# ---------------------------------------------------------------------------

def test_an_impossible_design_is_refused_with_one_sentence_not_a_grid_of_failures(
        qapp):
    screen = PowerScreen(threaded=False)
    try:
        screen.set_spec(_tiny(hit_rate=1e-6))
        assert screen.run() is False
        assert screen.status_is_error()
        assert "expects" in screen.status_text()
        assert screen.result() is None
        assert not screen._btn_run.isEnabled()
    finally:
        screen.close()
        screen.deleteLater()


def test_a_protective_effect_is_refused_rather_than_reported_as_no_power(qapp):
    """The model scores one direction; the other is a different question.

    Left to run, a fold below 1 gives the true hits negative coefficients and
    the screen would print "you have no power" for a design with plenty of it.
    """
    screen = PowerScreen(threaded=False)
    try:
        screen.set_spec(_tiny(effect_fold=0.5))
        assert screen.run() is False
        assert "LESS likely" in screen.status_text()
    finally:
        screen.close()
        screen.deleteLater()


# ---------------------------------------------------------------------------
# The registration seams
# ---------------------------------------------------------------------------

@pytest.fixture
def registry_sandbox():
    """Restore the app registry and the defaults registry after the test.

    A leaked row is a leaked tile, a leaked sidebar button and a leaked
    keyboard binding for every test that runs afterwards.
    """
    from spacr import settings as settings_mod
    from spacr.qt import app as app_mod

    apps = list(app_mod.APPS)
    factories = dict(app_mod.APP_FACTORIES)
    stages = dict(app_mod.APP_STAGE)
    meta = dict(app_mod.APP_META)
    had_defaults = settings_mod.has_registered_defaults(APP_KEY)
    yield app_mod
    app_mod.APPS[:] = apps
    app_mod.APP_FACTORIES.clear()
    app_mod.APP_FACTORIES.update(factories)
    app_mod.APP_STAGE.clear()
    app_mod.APP_STAGE.update(stages)
    app_mod.APP_META.clear()
    app_mod.APP_META.update(meta)
    app_mod._refresh_sections()
    if not had_defaults:
        settings_mod.unregister_defaults(APP_KEY)


def test_register_puts_the_app_in_the_design_section_and_is_idempotent(
        registry_sandbox):
    """The row is on, and registering it is what makes Design appear.

    ``SECTION_DESIGN`` was declared and empty from the day the sections
    were named — its note already read "Plan the experiment before it
    runs: power, sample size, plate layout…" — and this is the app it was
    written for. ``app.py`` now calls :func:`register` from
    ``_SELF_REGISTERING_APPS``, so the row exists on ``import
    spacr.qt.app`` and the tab is drawn.

    The registration is unwound and redone inside the sandbox, so what is
    asserted is what a fresh call does rather than what some earlier
    import left behind — including the idempotence, which is load-bearing
    because three different seams call this function.
    """
    app_mod = registry_sandbox
    assert APP_KEY in {row[0] for row in app_mod.APPS}, \
        "the row is registered from app.py._SELF_REGISTERING_APPS"

    # A section with no rows under it is not drawn, which is what makes
    # removing the last app close the tab. Power was Design's only app when
    # this was written; `experiment_design` has since joined it, so the
    # claim is stated as the rule rather than as the count — otherwise the
    # test reads as a regression in Power the day anything else is filed
    # under Design.
    neighbours = [row[0] for row in app_mod.APPS
                  if row[3] == app_mod.SECTION_DESIGN and row[0] != APP_KEY]
    app_mod.unregister_app(APP_KEY)
    assert (app_mod.SECTION_DESIGN in app_mod.SECTIONS) == bool(neighbours), (
        "a section is drawn exactly when something is filed under it; "
        f"Design still holds {neighbours}")

    assert register() is True
    assert register() is False, "a second import must not raise or duplicate"

    row = next(row for row in app_mod.APPS if row[0] == APP_KEY)
    assert row[3] == app_mod.SECTION_DESIGN
    assert app_mod.SECTION_DESIGN in app_mod.SECTIONS
    assert app_mod.APP_FACTORIES[APP_KEY] is make_power_screen
    # `spacr.qt.maturity` reassessed every alpha module against the
    # evidence in the repository and this one no longer qualifies; the
    # reason is recorded beside the decision. Applied here because the
    # promotions land in `register_self_registering_modules`, which every
    # launch calls but a bare test process may not have. `apply` alone,
    # not the whole registration pass: it touches only APP_STAGE, so it
    # cannot re-register a module a test has deliberately removed.
    from spacr.qt import maturity
    maturity.apply()
    assert app_mod.APP_STAGE[APP_KEY] == app_mod.STAGE_BETA
    assert app_mod.APP_META[APP_KEY]["api_module"] == "qt/screens/power"
    assert "power" in app_mod.APP_META[APP_KEY]["cli_note"].lower()
    # GUI-only, and the seam is what delivers the sentence that says so.
    from spacr import cli
    assert cli.INTERACTIVE_ONLY.get(APP_KEY, "").strip()
    assert APP_KEY not in cli.MODULES
    # Nine translations, so no window draws this row in English by default.
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES
    for code in VALID_LANGUAGE_CODES[1:]:
        assert CATALOGS[code].get(APP_NAME, "").strip(), code


def test_the_factory_builds_a_working_screen(registry_sandbox):
    screen = make_power_screen(APP_KEY)
    try:
        assert isinstance(screen, PowerScreen)
        assert screen.spec() == DesignSpec()
    finally:
        screen.close()
        screen.deleteLater()


def test_the_settings_seam_registers_typed_and_documented_keys(
        registry_sandbox):
    """`register_defaults` gets types, tooltips and a category, not bare keys.

    An untyped key cannot be validated by ``check_settings`` and an
    untooltipped one fails the GUI suite's help coverage, so registering
    without them would be registering a problem for somebody else.

    The defaults are unregistered first: ``register()`` calls
    ``register_settings()`` and ``register()`` now runs at ``import
    spacr.qt.app``, so by the time any test runs the keys are already
    there and a bare first call would answer False. Undoing it and doing
    it again is what keeps this a test of the call rather than of what
    some earlier import happened to leave behind.
    """
    from spacr import settings as settings_mod

    settings_mod.unregister_defaults(APP_KEY)
    assert not settings_mod.has_registered_defaults(APP_KEY)

    assert register_settings() is True
    assert register_settings() is False
    assert settings_mod.has_registered_defaults(APP_KEY)

    defaults = power_default_settings()
    assert defaults["power_n_genes"] == 452
    assert defaults["power_cells_per_well"] == pytest.approx(123.0)
    for key in defaults:
        assert key.startswith("power_")
        assert key in settings_mod.expected_types
        assert settings_mod.tooltips[key].startswith("(")
    assert set(defaults) <= set(settings_mod.categories["Power analysis"])


def test_settings_round_trip_into_the_same_design(registry_sandbox):
    """A design saved as settings and typed into the form are one object."""
    assert spec_from_settings(power_default_settings()) == DesignSpec()
    changed = power_default_settings({"power_n_genes": 96,
                                      "power_score_per": "guide",
                                      "power_seed": 7})
    spec = spec_from_settings(changed)
    assert (spec.n_genes, spec.score_per, spec.seed) == (96, "guide", 7)


def test_the_screen_contributes_its_own_qss_through_the_theme_seam():
    """Styling goes through ``register_widget_qss``, not a per-label sheet.

    An inline stylesheet is baked at construction and survives a theme
    switch, so a warning-orange caveat set under the dark theme would stay
    dark-theme orange on a light background.
    """
    from spacr.qt import theme as theme_mod

    assert "PowerDesign" in theme_mod.widget_qss_names()
    block = theme_mod._WIDGET_QSS["PowerDesign"](
        theme_mod.palette_for("dark"), None)
    assert "spacrPowerAnswer" in block
    assert "spacrCaveatSeverity" in block
    light = theme_mod._WIDGET_QSS["PowerDesign"](
        theme_mod.palette_for("light"), None)
    assert light != block, "the block must follow the theme"


# ---------------------------------------------------------------------------
# The curve widget on its own
# ---------------------------------------------------------------------------

def test_an_empty_curve_view_says_so_rather_than_drawing_nothing(qapp):
    view = PowerCurveView("t")
    try:
        assert view.is_empty()
        assert "no data" in view.describe()
        view.resize(240, 160)
        view.render(view.grab())  # paintEvent must survive an empty state
    finally:
        view.deleteLater()


def test_the_curve_view_reports_exactly_what_it_plots(qapp):
    import pandas as pd

    from spacr.qt.widgets.power_design import power_curve

    scan = pd.DataFrame({
        "imaging_n_cells_per_well_mu": [10.0, 10.0, 90.0, 90.0],
        "status": ["ok", "not_converged", "ok", "ok"],
        "model_auroc": [0.95, np.nan, 0.99, 0.97],
        "model_ap": [0.9, np.nan, 0.99, 0.95],
        "ap_baseline": [0.2, 0.2, 0.2, 0.2],
    })
    curve = power_curve(scan, "imaging_n_cells_per_well_mu", 0.8)
    view = PowerCurveView("Power")
    try:
        view.set_curve(curve, "cells", marker=90.0, threshold=0.8)
        described = view.describe()
        assert "10=0.50 (1/2)" in described
        assert "90=1.00 (2/2)" in described
        view.resize(320, 200)
        view.render(view.grab())
    finally:
        view.deleteLater()


def test_the_result_carries_the_design_it_was_run_with(signal_screen):
    """A sweep is minutes long and the form stays editable throughout.

    Labelling the result with whatever is on the form when it lands would
    put a design that was never simulated above numbers that were.
    """
    result = signal_screen.result()
    assert result["spec"] == signal_screen.spec()


def test_abundance_clipping_is_reported_rather_than_absorbed():
    """A clipped run is usable, but not for the number written on it.

    ``gene_abundance x well_abundance`` is used as a Bernoulli probability and
    nothing constrains it to [0, 1]; a skewed library saturates wells with its
    most abundant genes. The port clips and warns instead of returning R's
    NA-poisoned counts, and the realised constructs per well is then BELOW the
    figure the user typed — which is exactly the kind of silent substitution
    the whole screen exists not to make.
    """
    spec = _threading_spec().with_values(gene_abundance_alpha=0.6)
    payload = {"spec": spec, "fit_kwargs": dict(THREAD_FIT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_power_sweep(payload)
    assert result["n_clipped_screens"] > 0
    assert "clipped" in result["clip_message"]
    assert "below the requested" in result["clip_message"]


def test_a_clipped_run_says_so_on_screen(qapp):
    screen = _run_inline(
        qapp, _threading_spec().with_values(gene_abundance_alpha=0.6),
        fit_kwargs=THREAD_FIT)
    try:
        assert screen.result()["n_clipped_screens"] > 0
        status = screen.status_text()
        assert "clipped" in status
        assert "below the" in status
    finally:
        screen.close()
        screen.deleteLater()


def test_run_power_sweep_is_callable_without_any_widget():
    """The job is a plain function, so it works headless and under test."""
    spec = _threading_spec()
    payload = {"spec": spec, "fit_kwargs": dict(THREAD_FIT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_power_sweep(payload)
    assert result is payload["result"]
    assert set(result) >= {"cells_scan", "wells_scan", "cells_curve",
                           "wells_curve", "cancelled", "n_clipped_screens"}
    assert len(result["cells_curve"]) == len(cells_grid(spec))
    assert len(result["wells_curve"]) == len(wells_grid(spec))
    assert result["cancelled"] is False
