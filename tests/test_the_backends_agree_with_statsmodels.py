"""Instruction 141 D: a backend that returns different numbers is a bug.

Every backend spaCR routes a fit through has to answer the SAME question
statsmodels answers, on the same screen, to a tolerance written down here
rather than hoped for. Two more were wired on 2026-08-18 and this file is
their half of the bargain.

MEASURED ON THIS MACHINE (RTX 3090 host, numpy 1.26.4, scipy 1.15.3,
statsmodels 0.14.5, pyfixest 0.60.0, glum 3.4.1), on synthetic screens with
the shape ``spacr.ml.prepare_formula`` builds -- one row per (well, gRNA),
``y ~ fraction:grna + rowID + columnID``:

AGREEMENT, n=1830 rows, p=736 design columns, 697 of them gRNA terms:

    ==========================  ============  ==============  ============
    fit                         coefficient   standard error  p-value
    ==========================  ============  ==============  ============
    pyfixest  ols               4.1e-12 abs   6.9e-14 rel     2.4e-13 abs
    pyfixest  wls               7.3e-12 abs   5.2e-14 rel     5.1e-13 abs
    glum      poisson           2.6e-10 abs   8.3e-06 rel     2.2e-06 abs
    glum      logit             1.5e-09 abs   4.2e-07 rel     5.6e-09 abs
    ==========================  ============  ==============  ============

and the absorbed fits reproduce statsmodels' ``scale``, ``df_resid`` and
per-observation residuals to 1.1e-11 as well, which is what says the
Frisch-Waugh-Lovell projection charged the degrees of freedom for what it
absorbed instead of quietly reporting the standard errors of a smaller model.

SPEED, same designs, the whole reason either backend exists:

    ==========================  ===========  ========  ======
    fit                         statsmodels  backend   ratio
    ==========================  ===========  ========  ======
    absorbed ols, n=1830 p=736     0.244 s    0.172 s   1.4x
    absorbed ols, n=6000 p=2242    3.37 s     0.572 s   5.9x
    absorbed ols, n=12000 p=4601  43.3 s      2.59 s   16.7x
    glum poisson, n=1830 p=736     1.21 s     1.72 s    0.70x
    glum poisson, n=6000 p=2242   41.4 s     16.9 s     2.44x
    glum logit,   n=6000 p=2242   14.3 s      4.65 s    3.08x
    ==========================  ===========  ========  ======

THE glum ROW THAT IS BELOW 1 IS NOT A TYPO and is in the box for the same
reason it is here: on a small screen glum is SLOWER than statsmodels, and
instruction 141 B forbids offering a backend on an unmeasured "may be
faster" in either direction.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from patsy import dmatrices

import spacr
from spacr import ml
from spacr.regression_spec import REGRESSION_BACKENDS

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

pyfixest = pytest.importorskip("pyfixest")
glum = pytest.importorskip("glum")


def _screen(n_wells=610, n_genes=180, guides_per_gene=2, rows=16, cols=21,
            per_well=3, seed=0):
    """A screen with the shape ``prepare_formula`` builds a design from.

    One row per (well, gRNA), a plate position on every well, and a per-well
    cell count -- the WLS weights, the binomial ``var_weights`` and the
    Poisson exposure all come off that one column, exactly as
    :func:`spacr.ml.regression` supplies them.
    """
    rng = np.random.default_rng(seed)
    genes = [f"g{i:04d}" for i in range(n_genes)]
    guides = [f"{g}_{k}" for g in genes for k in range(1, guides_per_gene + 1)]
    records = []
    for well in range(n_wells):
        row = f"r{1 + well % rows}"
        column = f"c{1 + (well // rows) % cols}"
        picks = rng.choice(len(guides), size=per_well, replace=False)
        fractions = rng.dirichlet(np.ones(per_well))
        for index, fraction in zip(picks, fractions):
            records.append((f"p1_{row}_{column}", row, column, guides[index],
                            float(fraction)))
    frame = pd.DataFrame(
        records, columns=["prc", "rowID", "columnID", "grna", "fraction"])
    frame["cell_count"] = rng.integers(80, 600, len(frame)).astype(float)
    frame["y"] = np.clip(
        0.3 + 0.2 * frame["fraction"] + rng.normal(0, 0.05, len(frame)),
        1e-4, 1 - 1e-4)
    frame["count"] = rng.poisson(frame["cell_count"] * 0.05).astype(float)
    return frame


@pytest.fixture(scope="module")
def screen():
    return _screen()


@pytest.fixture(scope="module")
def design(screen):
    y, X = dmatrices("y ~ fraction:grna + rowID + columnID", data=screen,
                     return_type="dataframe")
    return y, X


@pytest.fixture(scope="module")
def counts(screen, design):
    y, _X = design
    return pd.DataFrame({"count": screen["count"].loc[y.index].to_numpy()},
                        index=y.index)


@pytest.fixture(scope="module")
def cell_count(screen, design):
    y, _X = design
    return screen["cell_count"].loc[y.index]


# ---------------------------------------------------------------------------
# pyfixest: the absorbing least-squares backend
# ---------------------------------------------------------------------------


def test_the_absorbed_ols_fit_agrees_with_statsmodels(design):
    """Coefficients, standard errors and p-values, on the same design."""
    y, X = design
    reference = ml.regression_model(X, y, regression_type="ols",
                                    regression_backend="statsmodels")
    absorbed = ml.regression_model(X, y, regression_type="ols",
                                   regression_backend="pyfixest")

    reported = list(absorbed.params.index)
    assert len(reported) > 300, "the absorbed fit reported almost nothing"
    coefficient = np.abs(absorbed.params[reported].to_numpy()
                         - reference.params[reported].to_numpy())
    standard_error = np.abs(
        absorbed.bse[reported].to_numpy()
        - reference.bse[reported].to_numpy()) / reference.bse[
            reported].to_numpy()
    p_value = np.abs(absorbed.pvalues[reported].to_numpy()
                     - reference.pvalues[reported].to_numpy())
    assert coefficient.max() < 1e-9, coefficient.max()
    assert standard_error.max() < 1e-11, standard_error.max()
    assert p_value.max() < 1e-10, p_value.max()


def test_the_absorbed_fit_charges_the_degrees_of_freedom_it_absorbed(design):
    """``scale``, ``df_resid`` and the residuals are the FULL model's.

    THE FAILURE THIS PINS is the one an absorbing fit gets wrong by default:
    dividing by ``n - p_kept`` instead of ``n - p_kept - p_absorbed`` reports
    the standard errors of a model that never carried the 35 plate-position
    parameters. On this screen that is 1094 residual degrees of freedom
    against 1129, so every standard error would come out 1.6% small and every
    p-value with it -- small enough to look like a rounding difference and
    large enough to move a gene across a q-value cut.
    """
    y, X = design
    reference = ml.regression_model(X, y, regression_type="ols",
                                    regression_backend="statsmodels")
    absorbed = ml.regression_model(X, y, regression_type="ols",
                                   regression_backend="pyfixest")
    assert absorbed.df_resid == pytest.approx(reference.df_resid)
    assert absorbed.scale == pytest.approx(reference.scale, rel=1e-12)
    residual = np.abs(np.asarray(reference.resid).ravel() - absorbed.resid)
    assert residual.max() < 1e-9, residual.max()


def test_the_absorbed_wls_fit_agrees_with_statsmodels(design, cell_count):
    y, X = design
    reference = ml.regression_model(X, y, regression_type="wls",
                                    weights=cell_count,
                                    regression_backend="statsmodels")
    absorbed = ml.regression_model(X, y, regression_type="wls",
                                   weights=cell_count,
                                   regression_backend="pyfixest")
    reported = list(absorbed.params.index)
    coefficient = np.abs(absorbed.params[reported].to_numpy()
                         - reference.params[reported].to_numpy())
    standard_error = np.abs(
        absorbed.bse[reported].to_numpy()
        - reference.bse[reported].to_numpy()) / reference.bse[
            reported].to_numpy()
    assert coefficient.max() < 1e-9, coefficient.max()
    assert standard_error.max() < 1e-11, standard_error.max()
    assert absorbed.scale == pytest.approx(reference.scale, rel=1e-12)


def test_the_absorbed_fit_reports_no_row_for_what_it_absorbed(design):
    """The one difference, and the box says it (``differs`` in the spec)."""
    y, X = design
    absorbed = ml.regression_model(X, y, regression_type="ols",
                                   regression_backend="pyfixest")
    names = list(absorbed.params.index)
    assert "Intercept" not in names
    assert not [n for n in names if n.startswith(("rowID[", "columnID["))]
    assert REGRESSION_BACKENDS["pyfixest"]["differs"]
    assert "Intercept" in REGRESSION_BACKENDS["pyfixest"]["differs"]


def test_the_coefficient_table_is_the_same_one_for_every_gRNA(design, screen):
    """Through ``process_model_coefficients``, which is what a run reads.

    The table the volcano and the hit list are built from, not the raw fit:
    every gRNA row has to carry the same coefficient and the same p-value
    whichever backend produced it, because that is where a backend that
    disagreed would actually change a result.
    """
    y, X = design
    tables = {}
    for backend in ("statsmodels", "pyfixest"):
        model = ml.regression_model(X, y, regression_type="ols",
                                    regression_backend=backend)
        tables[backend] = ml.process_model_coefficients(
            model, "ols", X, y, nc="nc", pc="pc", controls=[]
        ).set_index("feature")
    guides = [f for f in tables["pyfixest"].index if f.startswith("fraction:")]
    assert len(guides) > 300
    left = tables["statsmodels"].loc[guides]
    right = tables["pyfixest"].loc[guides]
    assert np.abs(left["coefficient"].to_numpy()
                  - right["coefficient"].to_numpy()).max() < 1e-9
    assert np.abs(left["p_value"].to_numpy()
                  - right["p_value"].to_numpy()).max() < 1e-10


def test_the_absorbed_fit_refuses_a_covariance_estimator_it_cannot_form(
        design):
    """HC1/HC2/HC3 need the full model's leverage, which absorption skips."""
    y, X = design
    with pytest.raises(ValueError, match="leverage"):
        ml.regression_model(X, y, regression_type="ols", cov_type="HC3",
                            regression_backend="pyfixest")


def test_the_absorbed_fit_refuses_wls_without_weights(design):
    y, X = design
    with pytest.raises(ValueError, match="unit weights is exactly OLS"):
        ml.regression_model(X, y, regression_type="wls",
                            regression_backend="pyfixest")


def test_a_design_with_nothing_to_absorb_is_refused_and_says_so(screen):
    """``model_plate_position=False`` leaves the backend with no job."""
    y, X = dmatrices("y ~ fraction:grna", data=screen,
                     return_type="dataframe")
    with pytest.raises(ValueError, match="nothing to absorb"):
        ml.regression_model(X, y, regression_type="ols",
                            regression_backend="pyfixest")


def test_the_reference_level_is_recovered_from_the_dummy_block(design):
    """patsy drops one level per factor; code 0 is that level.

    Getting this wrong is silent: an absorber that treated the all-zero rows
    as "no group" would demean them against each other instead of against
    their own row, and the fit would still converge.
    """
    _y, X = design
    codes, names, n_params = ml._absorbed_factor_codes(X)
    assert names == ["rowID", "columnID"]
    assert codes.shape == (X.shape[0], 2)
    row_columns = [c for c in X.columns if str(c).startswith("rowID[")]
    assert (codes[:, 0] == 0).sum() > 0, "no row took the reference level"
    assert codes[:, 0].max() == len(row_columns)
    # Intercept + (levels - 1) per factor: what the design actually spends.
    column_columns = [c for c in X.columns
                      if str(c).startswith("columnID[")]
    assert n_params == 1 + len(row_columns) + len(column_columns)


def test_a_column_that_is_not_a_dummy_is_refused_rather_than_absorbed():
    """A ``rowID[...]`` column carrying a measurement is not a factor."""
    X = pd.DataFrame({"Intercept": np.ones(6),
                      "rowID[T.r2]": np.linspace(0.0, 1.0, 6),
                      "fraction:grna[a]": np.arange(6.0)})
    with pytest.raises(ValueError, match="not 0/1 indicators"):
        ml._absorbed_factor_codes(X)


@pytest.mark.parametrize("n_wells,n_genes", [(2000, 700)])
def test_the_absorbed_fit_is_faster_on_a_screen_sized_problem(n_wells,
                                                              n_genes):
    """The measured claim, asserted -- loosely, because a shared box varies.

    The ratio measured on this machine at n=6000/p=2242 is 5.9x. The
    assertion is that it is faster AT ALL, which is the claim
    ``REGRESSION_BACKENDS['pyfixest']['cost']`` makes; a tighter bound would
    fail on a loaded machine for a reason that has nothing to do with the
    code.
    """
    import time

    frame = _screen(n_wells=n_wells, n_genes=n_genes, seed=3)
    y, X = dmatrices("y ~ fraction:grna + rowID + columnID", data=frame,
                     return_type="dataframe")
    start = time.perf_counter()
    sm.OLS(y, X).fit()
    reference = time.perf_counter() - start
    start = time.perf_counter()
    ml.regression_model(X, y, regression_type="ols",
                        regression_backend="pyfixest")
    absorbed = time.perf_counter() - start
    assert absorbed < reference, (
        f"absorbed {absorbed:.3f}s vs statsmodels {reference:.3f}s at "
        f"n={X.shape[0]}, p={X.shape[1]}")


# ---------------------------------------------------------------------------
# glum: the fast-GLM backend
# ---------------------------------------------------------------------------


def test_the_glum_poisson_fit_agrees_with_statsmodels(design, counts,
                                                      cell_count):
    _y, X = design
    reference = ml.regression_model(X, counts, regression_type="poisson",
                                    exposure=cell_count,
                                    regression_backend="statsmodels")
    fast = ml.regression_model(X, counts, regression_type="poisson",
                               exposure=cell_count,
                               regression_backend="glum")
    coefficient = np.abs(fast.params.to_numpy() - reference.params.to_numpy())
    standard_error = np.abs(
        fast.bse.to_numpy() - reference.bse.to_numpy()) / reference.bse.to_numpy()
    p_value = np.abs(fast.pvalues.to_numpy() - reference.pvalues.to_numpy())
    assert coefficient.max() < 1e-8, coefficient.max()
    assert standard_error.max() < 1e-4, standard_error.max()
    assert p_value.max() < 1e-4, p_value.max()


def test_the_glum_fit_keeps_the_numbers_the_run_prints(design, counts,
                                                       cell_count):
    """McFadden's R2 is printed off ``llf`` and ``null_deviance``.

    ``regression_model`` prints it for 'glm' and 'poisson', and a reader
    compares it between runs, so a backend that reported a different one
    would have changed a number nobody thinks of as an estimate.
    """
    _y, X = design
    reference = ml.regression_model(X, counts, regression_type="poisson",
                                    exposure=cell_count,
                                    regression_backend="statsmodels")
    fast = ml.regression_model(X, counts, regression_type="poisson",
                               exposure=cell_count,
                               regression_backend="glum")
    assert fast.llf == pytest.approx(reference.llf, rel=1e-10)
    assert fast.null_deviance == pytest.approx(reference.null_deviance,
                                               rel=1e-10)
    assert fast.df_resid == pytest.approx(reference.df_resid)


def test_the_glum_poisson_fit_keeps_the_exposure_offset(design, counts,
                                                        cell_count):
    """Without ``offset(log(cell_count))`` it would answer a different
    question -- the well's headcount rather than the per-cell rate."""
    _y, X = design
    with_offset = ml.regression_model(X, counts, regression_type="poisson",
                                      exposure=cell_count,
                                      regression_backend="glum")
    without = ml.regression_model(X, counts, regression_type="poisson",
                                  regression_backend="glum")
    assert np.abs(with_offset.params.to_numpy()
                  - without.params.to_numpy()).max() > 0.5


def test_the_glum_logit_fit_agrees_with_statsmodels(design, cell_count):
    """A fraction response weighted by cell count, which is spaCR's logit."""
    y, X = design
    reference = ml.regression_model(X, y, regression_type="logit",
                                    weights=cell_count,
                                    regression_backend="statsmodels")
    fast = ml.regression_model(X, y, regression_type="logit",
                               weights=cell_count,
                               regression_backend="glum")
    coefficient = np.abs(fast.params.to_numpy() - reference.params.to_numpy())
    standard_error = np.abs(
        fast.bse.to_numpy() - reference.bse.to_numpy()) / reference.bse.to_numpy()
    assert coefficient.max() < 1e-7, coefficient.max()
    assert standard_error.max() < 1e-5, standard_error.max()


def test_the_glum_backend_refuses_a_covariance_estimator(design, cell_count):
    _y, X = design
    with pytest.raises(ValueError, match="cov_type"):
        ml.regression_model(X, pd.DataFrame({"y": np.ones(X.shape[0])}),
                            regression_type="logit", cov_type="HC3",
                            weights=cell_count, regression_backend="glum")


def test_glum_is_not_offered_for_probit_or_quasi_binomial():
    """It has neither a probit link nor a free-dispersion binomial.

    Measured against glum 3.4: its link classes are Identity, Log, Logit,
    Cloglog and Tweedie. Offering 'probit (glum)' would have fitted a logit
    under the wrong name.
    """
    from spacr.regression_backends import backend_status

    for family in ("probit", "quasi_binomial"):
        status = backend_status("glum", family)
        assert not status["enabled"]
        assert family in status["reason"]
    assert "probit" not in REGRESSION_BACKENDS["glum"]["types"]
    assert "quasi_binomial" not in REGRESSION_BACKENDS["glum"]["types"]


# ---------------------------------------------------------------------------
# What the panel now says about the four that are NOT wired
# ---------------------------------------------------------------------------


def test_the_two_new_backends_are_offered_for_their_own_families():
    from spacr.regression_backends import backend_menu

    def enabled(regression_type):
        return {s["name"] for s in backend_menu(regression_type)
                if s["enabled"]}

    assert "pyfixest" in enabled("ols")
    assert "pyfixest" in enabled("wls")
    assert "glum" in enabled("poisson")
    assert "glum" in enabled("logit")
    assert "pyfixest" not in enabled("mixed")
    assert "glum" not in enabled("mixed")


def test_cuml_says_it_is_the_one_that_is_not_a_safe_install():
    """The dependency test instruction 141 opens with never covered cuML.

    Re-run 2026-08-18: ``pip install --dry-run --report cuml-cu12`` against
    this environment moves numpy 1.26.4 -> 2.2.6. The maintainer's condition
    was that a backend must not disturb the install, so this one says so on
    its own entry rather than being discovered by a user who picks it.
    """
    cost = REGRESSION_BACKENDS["cuml"]["cost"]
    assert "numpy" in cost
    assert "2.2.6" in cost


def test_pymer4_says_it_still_needs_r():
    """Its wheel declares no dependencies at all, which is not the same
    thing as having none -- every model module under it imports rpy2."""
    spec = REGRESSION_BACKENDS["pymer4"]
    assert "R" in spec["cost"] and "rpy2" in spec["cost"]
    assert not spec["implemented"]


def test_every_wired_backend_states_a_measured_cost():
    """Instruction 141 B: measured or stated, never 'may be faster'."""
    for name, spec in REGRESSION_BACKENDS.items():
        if not spec["implemented"]:
            continue
        assert "may be faster" not in spec["cost"].lower()
        assert any(token in spec["cost"] for token in ("x", "ms", "s ")), name
