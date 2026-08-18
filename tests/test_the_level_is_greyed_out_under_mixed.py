"""Instruction 132 A: mixed is the default, and `level` says why it is off.

Three claims are under test here, and each is driven through the real code
rather than described:

* a fresh regression dict fits ``'mixed'`` -- "mixed answers the most central
  question best" (maintainer, 2026-08-17);
* ``level`` exists, defaults to ``'both'``, and a settings CSV written before
  2026-08-17 -- which every regression run before then produced, and none of
  which carries the key -- still loads and means ``'both'``;
* ``level`` is DISABLED AND SAYING WHY under a mixed model (instruction 106),
  never absent and never present-but-inert.

The last test in the file is the load-bearing one: it confirms NUMERICALLY,
on a design built by spaCR's own :func:`spacr.ml.prepare_formula` and spaCR's
own ``gene_fraction`` rule, that ``fraction:grna + gene_fraction:gene`` is
rank deficient BY CONSTRUCTION. Everything above it exists because of that.
"""

import numpy as np
import pandas as pd
import pytest

from spacr.settings import (
    REGRESSION_LEVELS,
    get_perform_regression_default_settings,
    get_setting_dependencies,
    tooltips,
)
from spacr.settings_spec import convert_settings_dict_for_gui


def _regression_defaults(**overrides):
    return get_perform_regression_default_settings(dict(overrides))


# ---------------------------------------------------------------------------
# mixed is the default
# ---------------------------------------------------------------------------

def test_a_fresh_regression_dict_fits_mixed():
    """The default moved from 'ols' to 'mixed' on 2026-08-17."""
    assert _regression_defaults()["regression_type"] == "mixed"


def test_a_settings_csv_that_named_a_model_still_gets_that_model():
    """setdefault, not an override: an existing file does not silently move."""
    assert _regression_defaults(regression_type="ols")["regression_type"] == "ols"
    assert _regression_defaults(regression_type="lasso")["regression_type"] == "lasso"


def test_the_tooltip_names_the_new_default_and_what_changes():
    """The tooltip is the only place a user meets this decision."""
    text = tooltips["regression_type"]
    assert "Default 'mixed'" in text
    # what it replaced, so a returning user knows their numbers will move
    assert "'ols' until 2026-08-17" in text
    # and what the choice actually changes
    assert "'level' greys out" in text
    assert "nested" in text


# ---------------------------------------------------------------------------
# the new setting
# ---------------------------------------------------------------------------

def test_level_defaults_to_both():
    assert _regression_defaults()["level"] == "both"
    assert REGRESSION_LEVELS == ("both", "grna", "gene")


def test_a_settings_file_from_before_today_has_no_level_and_means_both():
    """Every regression run before 2026-08-17 wrote a settings CSV with no
    `level` in it. Those files must load, not raise, and must behave as the
    two-level runs they were."""
    before_today = {
        "regression_type": "ols", "dependent_variable": "pred",
        "agg_type": "mean", "alpha": 1, "transform": "log",
    }
    loaded = get_perform_regression_default_settings(dict(before_today))
    assert "level" in loaded
    assert loaded["level"] == "both"


@pytest.mark.parametrize("given,expected",
                         [("grna", "grna"), ("GENE", "gene"), (" Both ", "both")])
def test_a_level_is_normalised_rather_than_compared_raw(given, expected):
    assert _regression_defaults(level=given)["level"] == expected


def test_an_unrecognised_level_is_refused_and_names_the_three():
    """Falling back to 'both' would fit two models when one was asked for."""
    with pytest.raises(ValueError) as excinfo:
        _regression_defaults(level="guide")
    message = str(excinfo.value)
    assert "'guide'" in message
    for name in REGRESSION_LEVELS:
        assert repr(name) in message or f"'{name}'" in message


def test_the_level_tooltip_covers_both_modules_that_own_the_key():
    """One key, two modules: the proportion plots have meant
    'object'/'well'/'plate' by `level` for years, and this table is keyed by
    NAME with no module scope. A hover that describes only one of them is
    wrong in the other panel."""
    text = tooltips["level"]
    for word in ("both", "grna", "gene", "results_gene.csv", "results_grna.csv"):
        assert word in text
    for word in ("object", "well", "plate"):
        assert word in text
    assert len(text) <= 600


# ---------------------------------------------------------------------------
# the greying rule (instruction 106: disabled, and saying why)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def level_rule():
    return get_setting_dependencies()["level"]


def test_level_is_greyed_out_under_mixed_and_the_reason_says_why(level_rule):
    settings = {"regression_type": "mixed"}
    assert level_rule["predicate"](settings, {}) is False
    reason = level_rule["reason"](settings, {})
    assert "mixed" in reason
    assert "nest" in reason.lower()          # the mechanism, not just "off"
    assert "kept and saved" in reason        # the value is not discarded


def test_a_fresh_dict_greys_the_level_out_because_mixed_is_now_the_default(
        level_rule):
    """The two halves of this instruction have to agree with each other."""
    assert level_rule["predicate"](_regression_defaults(), {}) is False


@pytest.mark.parametrize("family", ["ols", "wls", "lasso", "quantile", "glm"])
def test_level_is_live_for_every_fixed_effects_family(level_rule, family):
    assert level_rule["predicate"]({"regression_type": family}, {}) is True


def test_random_row_column_effects_greys_the_level_out_too(level_rule):
    """ml._reconcile_random_row_column_effects rewrites regression_type to
    'mixed' before anything is fitted, so ticking that box with
    regression_type='ols' fits a mixed model and ignores the level. A control
    left enabled for a run that cannot use it is the failure the rule exists
    to prevent."""
    settings = {"regression_type": "ols", "random_row_column_effects": True}
    assert level_rule["predicate"](settings, {}) is False
    reason = level_rule["reason"](settings, {})
    assert "random_row_column_effects" in reason


def test_the_rule_watches_both_settings_it_depends_on(level_rule):
    """`sources` is what the panel reconnects; a missing source means the
    control does not re-grey when that widget changes."""
    assert set(level_rule["sources"]) == {
        "regression_type", "random_row_column_effects"}


def test_the_other_modules_level_is_never_greyed_by_this_rule(level_rule):
    """The proportion and endodyogeny panels use the same key for a different
    vocabulary and carry no regression_type at all."""
    from spacr.settings import (set_analyze_replication_defaults,
                                set_analyze_endodyogeny_defaults,
                                set_analyze_class_proportion_defaults)
    for factory in (set_analyze_replication_defaults,
                    set_analyze_endodyogeny_defaults,
                    set_analyze_class_proportion_defaults):
        settings = factory({"src": "/tmp/x"})
        assert "level" in settings
        assert level_rule["predicate"](settings, {}) is True


# ---------------------------------------------------------------------------
# the widget the panel builds
# ---------------------------------------------------------------------------

def test_the_regression_panel_offers_the_three_levels_as_a_dropdown():
    built = convert_settings_dict_for_gui(_regression_defaults())
    assert built["level"] == ("combo", ["both", "grna", "gene"], "both")


def test_the_proportion_panels_keep_their_own_level_vocabulary():
    """Keying the widget table by name alone would have offered
    'both'/'grna'/'gene' on a panel that cannot use any of them."""
    from spacr.settings import set_analyze_class_proportion_defaults
    settings = set_analyze_class_proportion_defaults({"src": "/tmp/x"})
    kind, options, default = convert_settings_dict_for_gui(settings)["level"]
    assert kind == "combo"
    assert options == ["object", "well", "plate"]
    assert default == settings["level"]


def test_a_level_in_neither_vocabulary_falls_through_untouched():
    """The dispatch is not a fallback: an unknown value keeps the old widget
    rather than being silently rewritten into one of the two lists."""
    assert convert_settings_dict_for_gui({"level": "elephant"})["level"] == (
        "entry", None, "elephant")


def test_the_model_dropdown_opens_on_the_model_that_will_be_fitted():
    """A combo default that disagrees with the settings default posts a
    different model than the panel was built for."""
    from spacr.settings_spec import convert_settings_dict_for_gui as build
    kind, options, default = build({"regression_type": "ols"})["regression_type"]
    assert kind == "combo"
    assert default == _regression_defaults()["regression_type"] == "mixed"
    assert "mixed" in options


# ---------------------------------------------------------------------------
# the new default has to be coherent with the flag that also means "mixed"
# ---------------------------------------------------------------------------

def test_a_fresh_dict_survives_the_random_effects_reconciliation():
    """_reconcile_random_row_column_effects REFUSES the flag beside a named
    non-mixed type. With 'mixed' the default, a fresh dict is compatible in
    both positions of the flag rather than newly refused."""
    from spacr.ml import _reconcile_random_row_column_effects

    off = _reconcile_random_row_column_effects(_regression_defaults())
    assert off["regression_type"] == "mixed"

    on = _reconcile_random_row_column_effects(
        _regression_defaults(random_row_column_effects=True))
    assert on["regression_type"] == "mixed"


def test_the_flag_still_refuses_a_named_non_mixed_model():
    """The new default must not have loosened the check."""
    from spacr.ml import _reconcile_random_row_column_effects

    with pytest.raises(ValueError, match="random_row_column_effects"):
        _reconcile_random_row_column_effects(
            _regression_defaults(regression_type="lasso",
                                 random_row_column_effects=True))


# ---------------------------------------------------------------------------
# THE FINDING THIS WHOLE INSTRUCTION RESTS ON
# ---------------------------------------------------------------------------

#: The formula spaCR fitted until 2026-08-17, written out here because it no
#: longer exists in the source: :func:`spacr.ml.prepare_formula` now builds one
#: level at a time. It is kept as a literal so this file can still show WHY it
#: had to go, and so that "the design was fine, really" cannot be argued back
#: in without re-running these two tests.
RETIRED_FORMULA = "pred ~ fraction:grna + gene_fraction:gene + rowID + columnID"


def _one_screen(guides_per_gene):
    """A well x guide frame with spaCR's own gene_fraction: the SUM of the
    gene's gRNA fractions in that well (ml.check_and_clean_data)."""
    rng = np.random.default_rng(0)
    rows = []
    for well in range(24):
        for gene, n_guides in guides_per_gene.items():
            for guide in range(1, n_guides + 1):
                rows.append({
                    "prc": f"plate1_r{well // 6 + 1}_c{well % 6 + 1}",
                    "rowID": f"r{well // 6 + 1}",
                    "columnID": f"c{well % 6 + 1}",
                    "gene": gene, "grna": f"{gene}_{guide}",
                    "fraction": float(rng.random()),
                })
    df = pd.DataFrame(rows)
    per_grna = df[["prc", "gene", "grna", "fraction"]].drop_duplicates()
    totals = per_grna.groupby(["prc", "gene"])["fraction"].sum()
    df["gene_fraction"] = pd.MultiIndex.from_arrays(
        [df["prc"], df["gene"]]).map(totals)
    df["pred"] = rng.normal(size=len(df))
    return df


def test_the_two_blocks_are_collinear_by_construction_not_by_bad_luck():
    """`gene_fraction` is the SUM of the gene's gRNA fractions, so a gene's
    column is a linear combination of its own guides' columns and the design
    statsmodels is handed is rank deficient. No number of wells fixes it.

    Measured the same way on the maintainer's real tsg101 screen
    (results/ols_13/regression_data.csv, 1945 rows, 610 wells): 1248 design
    columns, rank 862 -- 386 redundant -- and 386 of the 389
    gene_fraction:gene columns exactly reproducible from their own gene's
    fraction:grna columns.
    """
    from patsy import dmatrices

    # A single-guide gene beside multi-guide ones: 24 wells against 21
    # design columns, so there is no shortage of data anywhere.
    df = _one_screen({"g1": 1, "g2": 3, "g3": 3})
    _, X = dmatrices(RETIRED_FORMULA, df, return_type="dataframe")

    rank = np.linalg.matrix_rank(X.values)
    assert rank < X.shape[1], (
        f"expected a rank-deficient design, got rank {rank} of {X.shape[1]}")

    # And the exact identity behind it: the single-guide gene's column IS its
    # guide's column, which is why 244480 and 244480_3 came back identical.
    same = np.abs(X["gene_fraction:gene[g1]"].values
                  - X["fraction:grna[g1_1]"].values).max()
    assert same == 0.0, f"columns differ by {same}"


def test_statsmodels_answers_the_deficient_design_instead_of_refusing():
    """That is why the bug was silent: one arbitrary solution out of
    infinitely many, no warning, a results table that looks fine."""
    import statsmodels.api as sm
    from patsy import dmatrices

    df = _one_screen({"g1": 1, "g2": 3, "g3": 3})
    y, X = dmatrices(RETIRED_FORMULA, df, return_type="dataframe")
    fit = sm.OLS(y, X).fit()

    gene = float(fit.params["gene_fraction:gene[g1]"])
    guide = float(fit.params["fraction:grna[g1_1]"])
    assert gene == pytest.approx(guide), (
        "a single-guide gene and its guide are the same column, so the fit "
        "cannot tell them apart")
