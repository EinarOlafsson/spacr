"""Right-click the regression plot, choose another model, and it re-fits.

Asked for on 2026-08-16: "when all the analasees are done id like to be able
to right click on the regression plot and choose a different regression and
the other related settings as well as FDR etc."

A RE-FIT IS NOT A RESTYLE. Every other entry on that menu changes how the
figure looks; this one changes the numbers. So it lands in its own folder
beside the run it came from -- the user asked for it in order to COMPARE, and
a re-fit that overwrote the run on screen would destroy the comparison it was
started for.

The settings a finished run leaves behind cannot simply be handed back. Each
test below is one way that silently produces the wrong second run.
"""
from __future__ import annotations

import os

import pytest

from spacr import refit
from spacr.ml import (REGRESSION_SETTINGS_USED, REGRESSION_TYPES,
                      _reject_unused_settings)


def _base(**over):
    settings = {
        "count_data": ["/data/screen/counts.csv"],
        "score_data": ["/data/screen/scores.csv"],
        "regression_type": "ols",
        "multiple_testing_method": "fdr_bh",
        "fdr_alpha": 0.05,
        "plot": False,
        "test_mode": False,
        "src": "/data/screen/results/ols",
    }
    settings.update(over)
    return settings


# --------------------------------------------------------------------------- #
#  The knobs the new backend cannot read
# --------------------------------------------------------------------------- #

def test_switching_off_a_penalty_drops_its_weight():
    """lasso -> ols carrying alpha=0.3 does not quietly do nothing: the fit
    REFUSES it, by design, so the re-fit would die at the entry point."""
    settings, notes = refit.refit_settings(
        _base(regression_type="lasso", alpha=0.3), regression_type="ols")

    assert settings["alpha"] == 1.0
    assert any("alpha" in note for note in notes), notes


def test_it_says_what_it_reset():
    """A re-fit that silently dropped a penalty weight is a re-fit whose
    numbers the user cannot account for."""
    _settings, notes = refit.refit_settings(
        _base(regression_type="elasticnet", alpha=0.3, l1_ratio=0.9),
        regression_type="rlm")

    joined = " ".join(notes)
    assert "0.3" in joined and "0.9" in joined, notes


def test_an_unset_knob_is_not_reported_as_dropped():
    """The GUI posts every widget on the panel. A value still at its default
    was never a request, so reporting it as lost would be noise -- and noise
    is how the notes that DO matter get skipped."""
    _settings, notes = refit.refit_settings(
        _base(alpha=1.0, l1_ratio=0.5), regression_type="rlm")

    assert not any("alpha" in n or "l1_ratio" in n for n in notes), notes


def test_auto_alpha_is_not_a_request():
    """'auto' means "cross-validate it", which is not a number an unpenalised
    model is being asked to honour -- the same reading regression_model
    takes."""
    settings, notes = refit.refit_settings(
        _base(regression_type="lasso", alpha="auto"), regression_type="ols")

    assert settings["alpha"] == 1.0
    assert not any("alpha" in n for n in notes), notes


@pytest.mark.parametrize("regression_type", sorted(REGRESSION_TYPES))
def test_every_backend_can_be_refit_into(regression_type):
    """THE GUARD AGAINST DRIFT. For all 17 backends, a settings dict with
    every policed knob set away from its default must come out of the prune
    acceptable to the check that would otherwise refuse it.

    Written as the fit's own check rather than a restatement of it: a second
    copy of the defaults table is exactly what goes stale the first time a
    backend gains a knob.
    """
    loaded = _base(alpha=0.3, l1_ratio=0.9, cov_type="HC3", quantile=0.25,
                   hinge_threshold=0.5, huber_t=2.0, lasso_n_boot=50,
                   lasso_selection_threshold=0.9, hinge_n_boot=50)

    settings, _notes = refit.refit_settings(loaded,
                                            regression_type=regression_type)

    used = REGRESSION_SETTINGS_USED[regression_type]
    for name, default in refit.policed_settings().items():
        if name not in used:
            assert settings[name] == default, (
                f"{regression_type!r} cannot read {name}, so a re-fit into it "
                f"left {name}={settings[name]!r} for the fit to refuse")

    # And the fit's own check agrees, which is the half that cannot go stale.
    _reject_unused_settings(regression_type, {
        name: (settings[name], default)
        for name, default in refit.policed_settings().items()})


def test_the_prune_covers_every_policed_knob():
    """If a backend gains a knob and the defaults table does not, the prune
    silently stops resetting it and the re-fit raises on that backend only."""
    policed = set(refit.policed_settings())
    for _type, used in REGRESSION_SETTINGS_USED.items():
        assert set(used) <= policed, set(used) - policed


def test_choosing_from_the_data_prunes_as_strictly():
    """regression_type=None reads none of the knobs either -- "it might pick
    lasso" is not a reason to let a penalty weight through."""
    settings, _notes = refit.prune_for_type(_base(alpha=0.3), None)

    assert settings["alpha"] == 1.0


# --------------------------------------------------------------------------- #
#  Where it lands
# --------------------------------------------------------------------------- #

def test_it_does_not_inherit_the_last_runs_output_folder():
    """`src` was rewritten by the finished run to point at its own output
    root. Carrying it over nests the re-fit inside the run it is meant to sit
    beside, so the comparison is a folder deep in the thing it compares to."""
    settings, _notes = refit.refit_settings(_base(), regression_type="rlm")

    assert "src" not in settings


def test_a_refit_writes_somewhere_new(tmp_path):
    """The run on screen survives. This is the whole reason the re-fit was
    asked for -- to compare two models -- and an overwrite destroys it."""
    counts = tmp_path / "counts.csv"
    counts.write_text("x\n")
    taken = tmp_path / "results" / "rlm"
    taken.mkdir(parents=True)
    (taken / "results.csv").write_text("a\n")

    settings, _notes = refit.refit_settings(
        _base(count_data=[str(counts)]), regression_type="rlm")

    where = refit.destination(settings)
    assert where is not None
    assert os.path.abspath(where) != os.path.abspath(str(taken))
    assert not os.path.exists(where) or not os.listdir(where)


def test_the_figures_come_back_on():
    """save_settings writes plot=False so a reload reproduces the run
    headlessly. A re-fit asked for FROM a figure, that then drew no figures,
    reads as a re-fit that failed."""
    settings, _notes = refit.refit_settings(_base(plot=False),
                                            regression_type="rlm")

    assert settings["plot"] is True


# --------------------------------------------------------------------------- #
#  The contradictions the run refuses rather than guesses
# --------------------------------------------------------------------------- #

def test_random_effects_give_way_to_a_named_model():
    """The flag fits a MixedLM whatever the type says, and the run REFUSES
    the combination. Asking for 'rlm' from the plot is the user saying which
    of the two they meant."""
    settings, notes = refit.refit_settings(
        _base(random_row_column_effects=True, regression_type="mixed"),
        regression_type="rlm")

    assert settings["random_row_column_effects"] is False
    assert any("random" in n for n in notes), notes


def test_asking_for_mixed_keeps_the_flag():
    settings, _notes = refit.refit_settings(
        _base(random_row_column_effects=True, regression_type="mixed"),
        regression_type="mixed")

    assert settings["random_row_column_effects"] is True


# --------------------------------------------------------------------------- #
#  What it changes, and what it must not
# --------------------------------------------------------------------------- #

def test_the_correction_is_refittable():
    settings, notes = refit.refit_settings(_base(),
                                           correction_method="bonferroni")

    assert settings["multiple_testing_method"] == "bonferroni"
    assert any("correction" in n for n in notes), notes


def test_the_correction_is_written_where_the_run_reads_it():
    """`correction_method` is read by NOTHING. perform_regression looks up
    `multiple_testing_method`, so writing the other spelling gives a run that
    corrects the old way and labels its output the new way -- the two
    disagreeing in a file nobody re-reads."""
    settings, _notes = refit.refit_settings(_base(),
                                            correction_method="holm")

    assert settings[refit.CORRECTION_KEY] == "holm"
    assert "correction_method" not in settings


def test_an_unknown_correction_is_refused_while_the_dialog_is_open():
    """The run raises on a spelling it does not know. Better here than
    twenty minutes into a fit."""
    with pytest.raises(ValueError, match="Unsupported"):
        refit.refit_settings(_base(), correction_method="benjamini")


def test_a_spelling_the_run_accepts_is_accepted_here():
    """The dialog and the run must agree on the inventory, so the
    canonicaliser is shared rather than re-implemented."""
    settings, _notes = refit.refit_settings(_base(),
                                            correction_method="Benjamini-Hochberg")

    assert settings[refit.CORRECTION_KEY] == "fdr_bh"


def test_the_significance_level_is_refittable():
    settings, notes = refit.refit_settings(_base(), fdr_alpha=0.1)

    assert settings["fdr_alpha"] == 0.1
    assert any("significance" in n for n in notes), notes


def test_the_penalty_weight_is_not_the_significance_level():
    """Two different numbers one letter apart. `alpha` is the penalty weight
    of a penalised fit; `fdr_alpha` is where the correction cuts. Swapping
    them silently changes either the hit list or the model."""
    settings, _notes = refit.refit_settings(
        _base(regression_type="lasso"), regression_type="lasso", alpha=0.3,
        fdr_alpha=0.01)

    assert settings["alpha"] == 0.3
    assert settings["fdr_alpha"] == 0.01


def test_the_data_is_not_refittable():
    """"The same screen through a different model". Changing the data would
    make the two runs incomparable, which is the one thing this is for."""
    settings, _notes = refit.refit_settings(_base(), regression_type="rlm")

    assert settings["count_data"] == ["/data/screen/counts.csv"]
    assert settings["score_data"] == ["/data/screen/scores.csv"]


def test_keeping_the_model_is_allowed():
    """Re-fitting OLS with a different correction is a legitimate ask, and
    None means "leave it"."""
    settings, notes = refit.refit_settings(_base(), regression_type=None,
                                           correction_method="holm")

    assert settings["regression_type"] == "ols"
    assert not any("model" in n for n in notes), notes


# --------------------------------------------------------------------------- #
#  Refusing rather than failing later
# --------------------------------------------------------------------------- #

def test_no_settings_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="no settings"):
        refit.refit_settings({}, regression_type="rlm")


def test_no_count_data_is_refused_with_a_reason():
    """Started from a table opened off disk with no settings beside it: a run
    from these would fail much later with a much worse message."""
    base = _base()
    base.pop("count_data")
    with pytest.raises(ValueError, match="count data"):
        refit.refit_settings(base, regression_type="rlm")


# --------------------------------------------------------------------------- #
#  Reading the settings back off disk
# --------------------------------------------------------------------------- #

def test_a_runs_settings_are_found_beside_its_results(tmp_path):
    from spacr.utils import save_settings

    run = tmp_path / "results" / "ols"
    run.mkdir(parents=True)
    save_settings({"src": str(run), "regression_type": "ols",
                   "correction_method": "fdr_bh"}, name="regression")

    found = refit.settings_of_run(str(run / "results.csv"))

    assert found is not None
    assert found["regression_type"] == "ols"


def test_the_nearest_settings_win(tmp_path):
    """The shared settings/ copy is overwritten by every LATER run of the
    same screen, so seeding from it offers a model the table on screen was
    never fitted with."""
    from spacr.utils import save_settings

    run = tmp_path / "results" / "ols"
    run.mkdir(parents=True)
    save_settings({"src": str(tmp_path), "regression_type": "quantile"},
                  name="regression")
    save_settings({"src": str(run), "regression_type": "ols"},
                  name="regression")

    found = refit.settings_of_run(str(run))

    assert found["regression_type"] == "ols"


def test_a_run_with_no_settings_says_none_rather_than_guessing(tmp_path):
    run = tmp_path / "results" / "ols"
    run.mkdir(parents=True)

    assert refit.settings_of_run(str(run)) is None


def test_save_settings_and_load_settings_are_inverses(tmp_path):
    """They are documented as inverses and the docstring example used the
    defaults, but save writes Key/Value and load asked for
    setting_key/setting_value -- so the pair raised. Every caller had worked
    around it separately, which is how it survived."""
    from spacr.utils import load_settings, save_settings

    save_settings({"src": str(tmp_path), "regression_type": "ols",
                   "alpha": 0.3, "plot": True, "count_data": ["/a/b.csv"]})

    back = load_settings(str(tmp_path / "settings" / "settings.csv"))

    assert back["regression_type"] == "ols"
    assert back["alpha"] == 0.3
    assert back["count_data"] == ["/a/b.csv"]
