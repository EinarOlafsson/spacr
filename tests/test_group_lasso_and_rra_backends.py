"""``group_lasso`` and ``rra`` as selectable backends, driven end to end.

Instruction 133, the last item: "group lasso and RRA are selectable and fit."

:mod:`spacr.group_lasso` and :mod:`spacr.rra` were already built and tested on
their own. What is tested HERE is the wiring: that a user can pick either name,
that the fit produces the same coefficient table every other backend produces
-- ``feature``, ``coefficient``, ``p_value``, ``-log10(p_value)``, ``grna``,
``condition`` -- and that the numbers in it come from the model whose name is
on the results folder. "It ran" is what every regression-dispatch bug in this
project has also done, so every case below plants an effect and asks for it
back.

The two backends make opposite claims about inference and the tests hold them
to opposite standards:

* ``group_lasso`` has NO p-value, so it is in ``NO_P_VALUE_TYPES`` and the run
  ranks it by bootstrap selection frequency. The bootstrap has to fit the
  GROUP lasso; an ordinary lasso there would report the stability of a
  different model under this one's name.
* ``rra`` HAS one -- a permutation p-value with a permutation null -- so it is
  NOT in ``NO_P_VALUE_TYPES`` and is corrected and ranked like every
  likelihood fit.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# A small screen with one gene planted in it
# ---------------------------------------------------------------------------

GENES = ("000000", "233460", "239740", "111111", "222222")
N_GRNA_PER_GENE = 3
ROWS = tuple(f"r{i}" for i in range(1, 9))
COLS = tuple(f"c{i}" for i in range(1, 13))
HIT_GENE = "239740"
EFFECT = 0.45
GRNAS_PER_WELL = 4

#: The columns everything downstream of the fit reads -- the volcano's axes and
#: labels, the hit table, the metadata merge. A backend that fits and produces
#: a table missing any of these is not wired in.
PIPELINE_COLUMNS = ("feature", "coefficient", "p_value", "-log10(p_value)",
                    "grna", "condition")

#: Well below `spacr.group_lasso`'s own default of 0.05, and deliberately so.
#: The penalty is compared against `group_lasso.max_lambda`, which is a
#: property of the design and of the response scale; on this fixture it is
#: 0.384, every gene block is empty by 0.002, and the planted gene alone
#: survives at 0.001. `test_group_lasso_refuses_a_penalty_that_empties_every_
#: gene` pins the other side of that.
WORKING_LAMBDA = 0.001


def write_screen(tmp_path, seed=0, n_cells=12):
    """Write the score and count CSVs for a screen with one planted gene.

    :param tmp_path: directory to write into.
    :param seed: RNG seed; the whole screen is deterministic from it.
    :param n_cells: objects scored per well.
    :returns: ``(score_csv, count_csv)``.
    """
    rng = np.random.default_rng(seed)
    grnas = [f"TGGT1_{g}_{i}"
             for g in GENES for i in range(1, N_GRNA_PER_GENE + 1)]
    counts, scores = [], []
    for row in ROWS:
        for column in COLS:
            chosen = rng.choice(len(grnas), size=GRNAS_PER_WELL, replace=False)
            fractions = rng.dirichlet(np.ones(GRNAS_PER_WELL) * 3)
            total = int(rng.integers(500, 2000))
            effect = 0.0
            for index, fraction in zip(chosen, fractions):
                grna = grnas[index]
                counts.append({"plateID": "plate1", "rowID": row,
                               "columnID": column, "grna": grna,
                               "count": int(round(fraction * total))})
                if grna.split("_")[1] == HIT_GENE:
                    effect += EFFECT * fraction
            mean = 0.30 + effect
            for _ in range(n_cells):
                scores.append({"plateID": "plate1", "rowID": row,
                               "columnID": column, "fieldID": "f1",
                               "pred": float(np.clip(
                                   mean + rng.normal(0, 0.05), 0.02, 0.98))})
    sdir = tmp_path / "scores"
    cdir = tmp_path / "counts"
    sdir.mkdir(exist_ok=True)
    cdir.mkdir(exist_ok=True)
    score = str(sdir / "screen_scores.csv")
    count = str(cdir / "counts.csv")
    pd.DataFrame(scores).to_csv(score, index=False)
    pd.DataFrame(counts).to_csv(count, index=False)
    return score, count


def settings_for(score, count, **over):
    """Settings as a dispatcher builds them, finished by the same defaults."""
    from spacr.settings import get_perform_regression_default_settings

    settings = {
        "score_data": [score],
        "count_data": [count],
        "dependent_variable": "pred",
        "min_cell_count": 3,
        "fraction_threshold": 0.01,
        "metadata_files": [],
        # `toxo` rather than `Toxoplasma`: instruction 133 renames the setting
        # and migrates the old spelling, so the old name is the one that works
        # on both sides of that rename and this file is not a hostage to it.
        "toxo": False,
        "controls": None,
        "outlier_detection": False,
        "filter_value": [],
        "control_wells": [],
        "level": "grna",
    }
    settings.update(over)
    return get_perform_regression_default_settings(settings)


def design_for(tmp_path, level="grna"):
    """The design matrix and response the pipeline itself would build.

    Built through ``prepare_formula`` / ``check_and_clean_data`` rather than by
    hand, because the column NAMES are what both backends group by: a fixture
    that spelled its own terms could pass while the real ones did not parse.
    """
    from patsy import dmatrices

    from spacr.ml import check_and_clean_data, prepare_formula

    score, count = write_screen(tmp_path)
    scores = pd.read_csv(score)
    counts = pd.read_csv(count)
    wells = (scores.groupby(["plateID", "rowID", "columnID"])["pred"]
             .mean().reset_index())
    frame = counts.merge(wells, on=["plateID", "rowID", "columnID"])
    frame["gene"] = frame["grna"].str.split("_").str[1]
    frame["grna"] = (frame["grna"].str.split("_").str[1] + "_"
                     + frame["grna"].str.split("_").str[2])
    totals = frame.groupby(["plateID", "rowID", "columnID"])["count"].transform("sum")
    frame["fraction"] = frame["count"] / totals
    # `check_and_clean_data` keys the per-well gene share on `prc`, so the well
    # id has to exist before it is called; `perform_regression` composes the
    # same three parts.
    frame["prc"] = (frame["plateID"].astype(str) + "_"
                    + frame["rowID"].astype(str) + "_"
                    + frame["columnID"].astype(str))
    frame = check_and_clean_data(frame, "pred")
    # dmatrices answers (y, X); these tests want them the way regression_model
    # takes them.
    response, design = dmatrices(prepare_formula("pred", level=level),
                                 data=frame, return_type="dataframe")
    return design, response


def gene_of(feature):
    """The gene a coefficient row names, or None."""
    from spacr.ml import _gene_of_design_column

    return _gene_of_design_column(feature)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    plt.close("all")


@pytest.fixture
def stubs(monkeypatch):
    """Stub only the visual helpers and the Monte-Carlo cell-count sweep."""
    import spacr.plot as P
    import spacr.ml as ML
    import spacr.toxo   # noqa: F401 - warm the lazy imports

    monkeypatch.setattr(P, "plot_plates", lambda df, **kwargs: None)
    monkeypatch.setattr(P, "plot_histogram",
                        lambda df, column, dst=None: None)
    monkeypatch.setattr(P, "plot_data_from_csv", lambda settings: (None, None))
    monkeypatch.setattr(ML, "minimum_cell_simulation",
                        lambda settings, **kwargs: 3)


# ---------------------------------------------------------------------------
# The spec: both names are offered, and each declares what it reads
# ---------------------------------------------------------------------------

def test_both_backends_are_offered_and_declare_the_settings_they_read():
    from spacr.regression_spec import (REGRESSION_SETTINGS_USED,
                                       REGRESSION_TYPES)

    assert "group_lasso" in REGRESSION_TYPES
    assert "rra" in REGRESSION_TYPES
    assert REGRESSION_SETTINGS_USED["group_lasso"] == (
        "group_lasso_lambda", "lasso_n_boot", "lasso_selection_threshold")
    assert REGRESSION_SETTINGS_USED["rra"] == ("rra_alpha", "rra_permutations")


def test_group_lasso_has_no_p_value_and_rra_does():
    """The membership that decides how each one's hit list is RANKED.

    ``NO_P_VALUE_TYPES`` is not a label, it is a branch: a type in it is ranked
    by bootstrap selection frequency and ``_call_level_hits`` refuses the run
    outright without a bootstrap to do it with. ``rra`` reports a permutation
    P value with a permutation null, so putting it there would replace a real
    test with a stability score it never computed.
    """
    from spacr.regression_spec import NO_P_VALUE_TYPES

    assert "group_lasso" in NO_P_VALUE_TYPES
    assert "rra" not in NO_P_VALUE_TYPES


def test_every_advertised_type_still_has_a_coefficient_branch():
    """The round trip, re-asserted for the two new names.

    ``process_model_coefficients`` raises "Unsupported regression type" for
    anything with no branch, and it raises AFTER the fit -- which on a real
    screen is the expensive half.
    """
    from spacr.ml import (REGRESSION_TYPES, _SKLEARN_COEF_TYPES,
                          _STATSMODELS_COEF_TYPES)

    covered = set(_STATSMODELS_COEF_TYPES) | set(_SKLEARN_COEF_TYPES) | {
        "beta", "hinge"}
    assert set(REGRESSION_TYPES) == covered
    assert "group_lasso" in _SKLEARN_COEF_TYPES
    assert "rra" in _STATSMODELS_COEF_TYPES


@pytest.mark.parametrize(("regression_type", "setting", "value"), [
    ("ols", "group_lasso_lambda", 0.2),
    ("ols", "rra_alpha", 0.5),
    ("lasso", "rra_permutations", 500),
    ("rra", "group_lasso_lambda", 0.2),
    ("group_lasso", "rra_alpha", 0.5),
])
def test_a_new_setting_is_refused_by_a_model_that_cannot_read_it(
        regression_type, setting, value):
    """A silently ignored setting is this pipeline's most expensive failure.

    ``group_lasso_lambda`` on an OLS run is the archetype for the new pair:
    nothing would have applied it and the run would have completed with a
    number on the panel that changed nothing.
    """
    from spacr.ml import regression_model

    X = pd.DataFrame({"Intercept": 1.0,
                      "fraction:grna[111111_1]": np.linspace(0, 1, 30)})
    y = pd.Series(np.linspace(0, 1, 30) * 0.4 + 0.1)
    with pytest.raises(ValueError, match=setting):
        regression_model(X, y, regression_type=regression_type,
                         **{setting: value})


# ---------------------------------------------------------------------------
# The grouping both backends run on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("column", "expected"), [
    ("fraction:grna[239740_1]", "239740"),
    ("fraction:grna[T.239740_12]", "239740"),
    ("fraction:grna[TGGT1_239740_3]", "TGGT1_239740"),
    ("gene_fraction:gene[239740]", "239740"),
    ("gene_fraction:gene[T.239740]", "239740"),
    # A single-guide gene named with no guide number is its own gene, not the
    # empty string -- which would otherwise be ONE group holding every such
    # guide in the screen.
    ("fraction:grna[239740]", "239740"),
    # Nuisance terms name no gene. An unanchored search would read `r2` out of
    # the row dummy and hand the plate layout to the grouping as genes.
    ("Intercept", None),
    ("rowID[T.r2]", None),
    ("columnID[T.c7]", None),
    ("screenID[T.second]", None),
])
def test_the_gene_of_a_design_column_is_read_from_its_name(column, expected):
    from spacr.ml import _gene_of_design_column

    assert _gene_of_design_column(column) == expected


def test_a_nuisance_column_is_its_own_group_and_never_joins_a_gene_block():
    """A row dummy pulled into a gene's block would be zeroed with the gene."""
    from spacr.ml import _design_column_groups

    columns = ["Intercept", "rowID[T.r2]", "fraction:grna[239740_1]",
               "fraction:grna[239740_2]", "columnID[T.c3]"]
    groups = _design_column_groups(columns)
    assert groups == ["Intercept", "rowID[T.r2]", "239740", "239740",
                      "columnID[T.c3]"]
    assert len(set(groups)) == 4


@pytest.mark.parametrize("name", ["group_lasso", "rra"])
def test_both_backends_refuse_a_design_with_no_gene_in_it(name):
    """A design of bare covariates has no gene to group or aggregate by.

    Fitted anyway, the group lasso would make every column its own block --
    ordinary lasso reported as a group lasso -- and RRA would rank nothing and
    report a P value of NaN for every row.
    """
    from spacr.ml import regression_model

    X = pd.DataFrame({"Intercept": 1.0, "a": np.linspace(0, 1, 40),
                      "b": np.linspace(1, 0, 40) ** 2})
    y = pd.Series(np.linspace(0, 1, 40) * 0.4 + 0.1)
    with pytest.raises(ValueError, match="gRNA or gene term"):
        regression_model(X, y, regression_type=name)


def test_both_backends_refuse_a_design_with_no_column_names(tmp_path):
    """The grouping lives in the column names; an array has none.

    Fitted anyway, the group lasso would put every column in a group of its
    own -- i.e. an ordinary lasso reported as a group lasso, which is the one
    result this backend exists to not produce.
    """
    from spacr.ml import regression_model

    X, y = design_for(tmp_path)
    for name in ("group_lasso", "rra"):
        with pytest.raises(ValueError, match="COLUMN NAMES"):
            regression_model(np.asarray(X, dtype=float),
                             np.asarray(y, dtype=float).ravel(),
                             regression_type=name)


# ---------------------------------------------------------------------------
# group_lasso
# ---------------------------------------------------------------------------

def test_group_lasso_tables_the_columns_the_pipeline_reads(tmp_path):
    from spacr.ml import process_model_coefficients, regression_model

    X, y = design_for(tmp_path)
    model = regression_model(X, y, regression_type="group_lasso",
                             group_lasso_lambda=WORKING_LAMBDA)
    coef_df = process_model_coefficients(model, "group_lasso", X, y,
                                         None, None, None)

    assert list(PIPELINE_COLUMNS) == [c for c in PIPELINE_COLUMNS
                                      if c in coef_df.columns]
    assert coef_df["coefficient"].notna().all()
    assert coef_df["p_value"].between(0.0, 1.0).all()
    # One row per surviving design column, exactly as lasso produces: the
    # row/column nuisance terms are dropped and everything else stays.
    assert set(coef_df["feature"]) == {
        str(c) for c in X.columns if "row" not in str(c) and "column" not in str(c)}


def test_group_lasso_selects_a_genes_guides_as_a_block(tmp_path):
    """The claim the backend exists to make, checked against ordinary lasso.

    Lasso keeps whichever one of a gene's correlated guides fits best and drops
    the rest, which reads as "one guide works and two do not". Group lasso
    takes the block or leaves it. Measured on this fixture: the group fit keeps
    all three guides of the planted gene and no guide of any other gene.
    """
    from spacr.ml import process_model_coefficients, regression_model

    X, y = design_for(tmp_path)
    model = regression_model(X, y, regression_type="group_lasso",
                             group_lasso_lambda=WORKING_LAMBDA)
    coef_df = process_model_coefficients(model, "group_lasso", X, y,
                                         None, None, None)
    coef_df["gene"] = coef_df["feature"].map(gene_of)

    non_zero = coef_df[(coef_df["coefficient"] != 0)
                       & coef_df["gene"].notna()]
    assert set(non_zero["gene"]) == {HIT_GENE}, (
        "group lasso selected a gene other than the planted one: "
        f"{sorted(set(non_zero['gene']))}")
    assert len(non_zero) == N_GRNA_PER_GENE, (
        "the planted gene's guides were not kept as a block; kept "
        f"{sorted(non_zero['feature'])}")
    assert (non_zero["coefficient"] > 0).all(), (
        "the planted effect came back with the wrong sign")

    # Every group is entirely in or entirely out -- which is what makes the
    # per-COLUMN selection frequency the run reports a per-GENE statement.
    for gene, block in coef_df.dropna(subset=["gene"]).groupby("gene"):
        assert len(set(block["coefficient"] != 0)) == 1, gene


def test_group_lasso_refuses_a_penalty_that_empties_every_gene(tmp_path):
    """All-zero gene coefficients reach the user as "0 significant gRNAs".

    Indistinguishable from a screen with no hits, which is exactly why the
    lasso branch refuses the same outcome. The row and column dummies are
    singleton groups with far larger correlations than any guide block, so
    they survive a penalty that has already emptied every gene -- `np.any` over
    the whole coefficient vector would therefore NOT have caught this.
    """
    from spacr.ml import regression_model

    X, y = design_for(tmp_path)
    with pytest.raises(ValueError, match="max_lambda"):
        regression_model(X, y, regression_type="group_lasso",
                         group_lasso_lambda=0.02)


def test_group_lasso_lambda_actually_reaches_the_solver(tmp_path):
    """A setting a backend declares and quietly ignores is the same failure
    as one it reads and should not, wearing the opposite label."""
    from spacr.ml import regression_model

    X, y = design_for(tmp_path)
    # Halved rather than doubled: 2 * WORKING_LAMBDA already empties every
    # gene block on this fixture, so the comparison would be against the
    # refusal instead of against a fit.
    loose = regression_model(X, y, regression_type="group_lasso",
                             group_lasso_lambda=WORKING_LAMBDA / 2)
    tight = regression_model(X, y, regression_type="group_lasso",
                             group_lasso_lambda=WORKING_LAMBDA)
    assert not np.allclose(loose.coef_, tight.coef_)
    assert np.abs(tight.coef_).sum() < np.abs(loose.coef_).sum(), (
        "a larger group_lasso_lambda did not shrink the fit")


# ---------------------------------------------------------------------------
# rra
# ---------------------------------------------------------------------------

def test_rra_tables_the_columns_the_pipeline_reads(tmp_path):
    from spacr.ml import process_model_coefficients, regression_model

    X, y = design_for(tmp_path)
    model = regression_model(X, y, regression_type="rra",
                             rra_permutations=2000)
    coef_df = process_model_coefficients(model, "rra", X, y, None, None, None)

    assert list(PIPELINE_COLUMNS) == [c for c in PIPELINE_COLUMNS
                                      if c in coef_df.columns]
    assert coef_df["coefficient"].notna().all()
    guides = coef_df[coef_df["feature"].str.contains(r"grna\[")]
    assert len(guides) == len(GENES) * N_GRNA_PER_GENE
    assert guides["p_value"].between(0.0, 1.0).all()
    # A row that names no gene was never ranked, so it has no P value. NaN
    # rather than 1.0: `adjust_p_values` leaves a non-finite entry out of the
    # family, and a 1.0 would read as "tested and found null".
    assert coef_df.loc[coef_df["feature"] == "Intercept",
                       "p_value"].isna().all()


def test_rra_recovers_the_planted_gene_as_the_top_call(tmp_path):
    from spacr.ml import process_model_coefficients, regression_model

    X, y = design_for(tmp_path)
    model = regression_model(X, y, regression_type="rra",
                             rra_permutations=2000)
    coef_df = process_model_coefficients(model, "rra", X, y, None, None, None)
    coef_df["gene"] = coef_df["feature"].map(gene_of)
    per_gene = (coef_df.dropna(subset=["gene"])
                .groupby("gene")[["p_value", "coefficient"]].first())

    assert per_gene["p_value"].idxmin() == HIT_GENE, per_gene
    assert per_gene.loc[HIT_GENE, "p_value"] < 0.05
    assert per_gene.loc[HIT_GENE, "coefficient"] > 0


def test_rra_reports_the_gene_call_on_every_one_of_its_guides(tmp_path):
    """The mapping, asserted rather than described.

    RRA tests GENES, so the P value on a guide row is its gene's aggregated
    permutation P value and every guide of one gene carries the same one; the
    coefficient stays the guide's OWN marginal slope, so the volcano still has
    one point per guide at the guide's own effect size.
    """
    from spacr.ml import process_model_coefficients, regression_model

    X, y = design_for(tmp_path)
    model = regression_model(X, y, regression_type="rra",
                             rra_permutations=2000)
    coef_df = process_model_coefficients(model, "rra", X, y, None, None, None)
    coef_df["gene"] = coef_df["feature"].map(gene_of)
    guides = coef_df.dropna(subset=["gene"])

    assert guides.groupby("gene")["p_value"].nunique().eq(1).all()
    assert guides.groupby("gene")["coefficient"].nunique().eq(
        N_GRNA_PER_GENE).all(), (
        "guides of one gene were given one shared coefficient; the score is "
        "per guide and only the P value is aggregated")

    # The scores ARE the marginal slopes, one parameter estimated at a time --
    # which is what lets RRA answer at all where the joint fit is undefined.
    design = np.asarray(X, dtype=float)
    centred = design - design.mean(axis=0)
    response = np.asarray(y, dtype=float).ravel()
    response = response - response.mean()
    spread = (centred ** 2).sum(axis=0)
    expected = pd.Series(
        np.where(spread > 0, centred.T @ response / np.where(spread > 0, spread, 1.0), 0.0),
        index=[str(c) for c in X.columns])
    for feature, coefficient in zip(guides["feature"], guides["coefficient"]):
        assert coefficient == pytest.approx(expected[feature])


def test_rra_permutations_and_alpha_reach_the_statistic(tmp_path):
    """Both declared settings must move the answer.

    ``rra_permutations`` sets the resolution of the P value -- the smallest
    one reportable is ``1 / (n + 1)`` -- and ``rra_alpha`` decides how much of
    the ranking is aggregated over, which is the difference between finding a
    gene with one strong guide and missing it.
    """
    from spacr.ml import regression_model

    X, y = design_for(tmp_path)
    few = regression_model(X, y, regression_type="rra", rra_permutations=200)
    many = regression_model(X, y, regression_type="rra", rra_permutations=5000)
    # 200 draws cannot report anything below 2/(200 + 1); the planted gene is
    # strong enough to sit on that floor, and 5000 draws move it off it.
    assert few.pvalues.min() == pytest.approx(2.0 / 201.0)
    assert many.pvalues.min() < few.pvalues.min()

    narrow = regression_model(X, y, regression_type="rra", rra_alpha=0.05,
                              rra_permutations=1000)
    wide = regression_model(X, y, regression_type="rra", rra_alpha=1.0,
                            rra_permutations=1000)
    assert not np.allclose(narrow.genes["rho_neg"].to_numpy(),
                           wide.genes["rho_neg"].to_numpy())


def test_rra_p_value_is_two_sided_and_not_the_smaller_tail(tmp_path):
    """Taking the better tail without doubling halves every P value for free."""
    from spacr.ml import regression_model

    X, y = design_for(tmp_path)
    model = regression_model(X, y, regression_type="rra",
                             rra_permutations=1000)
    table = model.genes
    expected = np.minimum(1.0, 2.0 * np.minimum(
        table["p_neg"].to_numpy(dtype=float),
        table["p_pos"].to_numpy(dtype=float)))
    by_gene = dict(zip(table["gene"].astype(str), expected))

    for feature, p_value in model.pvalues.items():
        gene = gene_of(feature)
        if gene is None:
            assert np.isnan(p_value)
        else:
            assert p_value == pytest.approx(by_gene[gene])


# ---------------------------------------------------------------------------
# End to end -- the only way a user reaches a regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_group_lasso_runs_through_perform_regression(tmp_path, stubs):
    """A backend that fits in isolation and dies in the pipeline is the bug."""
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path)
    settings = settings_for(
        score, count, regression_type="group_lasso",
        group_lasso_lambda=WORKING_LAMBDA,
        # Five resamples, not two hundred: this asserts that the stability
        # branch runs and fits the right model, and the frequency's precision
        # is not what is under test.
        lasso_n_boot=5, lasso_selection_threshold=0.5)

    out = perform_regression(settings)
    results = out["results"]

    for column in PIPELINE_COLUMNS:
        assert column in results.columns, column
    assert results["coefficient"].notna().all()
    # The selection frequency is what a NO_P_VALUE_TYPES run is RANKED by, so
    # its absence is not cosmetic -- the hit list would be empty.
    assert {"selection_frequency", "mean_coefficient"} <= set(results.columns)
    guides = results[results["feature"].str.contains(r"grna\[")]
    assert len(guides) > 0
    top = guides.reindex(
        guides["coefficient"].abs().sort_values(ascending=False).index)
    assert gene_of(top.iloc[0]["feature"]) == HIT_GENE, (
        f"the planted gene is not the largest coefficient; "
        f"{top.head(4)[['feature', 'coefficient']]}")

    folder = out["res_folder"]
    assert os.path.isfile(os.path.join(folder, "results.csv"))
    assert os.path.basename(folder).startswith("group_lasso")


@pytest.mark.slow
def test_group_lassos_stability_bootstrap_fits_the_group_lasso(tmp_path,
                                                               stubs):
    """The selection frequency must come from the model that was reported.

    Falling through to sklearn's ``Lasso`` -- which is what an untouched
    ``bootstrap_selection_frequencies`` did for any name that is not
    ``elasticnet`` -- would report the stability of an ORDINARY lasso under
    the group lasso's name, selecting one guide out of a gene's correlated set:
    the exact behaviour the group penalty exists to remove. A group lasso takes
    a block or leaves it, so its per-column frequencies are constant within a
    gene, and a per-guide lasso's are not.
    """
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path)
    settings = settings_for(
        score, count, regression_type="group_lasso",
        group_lasso_lambda=WORKING_LAMBDA,
        lasso_n_boot=8, lasso_selection_threshold=0.5)

    results = perform_regression(settings)["results"]
    results = results[results["feature"].str.contains(r"grna\[")].copy()
    results["gene"] = results["feature"].map(gene_of)

    spread = results.groupby("gene")["selection_frequency"].nunique()
    assert spread.eq(1).all(), (
        "a gene's guides were selected at different rates, so the bootstrap "
        "did not fit a GROUP lasso:\n"
        f"{results.groupby('gene')['selection_frequency'].unique()}")
    per_gene = results.groupby("gene")["selection_frequency"].first()
    assert per_gene.idxmax() == HIT_GENE, per_gene
    # Measured: 7 of the 8 resamples keep the planted gene's block. Asserted as
    # a floor rather than as 1.0, because a bootstrap that lands on the same
    # answer every time out of eight draws is a property of eight draws, not of
    # the method.
    assert per_gene[HIT_GENE] >= 0.5


@pytest.mark.slow
def test_rra_runs_through_perform_regression(tmp_path, stubs):
    from spacr.ml import perform_regression

    score, count = write_screen(tmp_path)
    settings = settings_for(score, count, regression_type="rra",
                            rra_permutations=2000)

    out = perform_regression(settings)
    results = out["results"]

    for column in PIPELINE_COLUMNS:
        assert column in results.columns, column
    assert results["coefficient"].notna().all()
    guides = results[results["feature"].str.contains(r"grna\[")]
    assert len(guides) == len(GENES) * N_GRNA_PER_GENE
    assert guides["p_value"].between(0.0, 1.0).all()

    # RRA is NOT in NO_P_VALUE_TYPES, so the run corrects it like any test.
    assert "q_value" in results.columns
    assert guides["q_value"].notna().all()
    ranked = guides.sort_values("p_value")
    assert gene_of(ranked.iloc[0]["feature"]) == HIT_GENE, ranked.head(4)

    folder = out["res_folder"]
    assert os.path.isfile(os.path.join(folder, "results.csv"))
    assert os.path.basename(folder).startswith("rra")
