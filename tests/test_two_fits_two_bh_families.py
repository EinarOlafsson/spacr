"""Instruction 132 B/C: two separate fits, two BH families, one nested mixed model.

THE FINDING THIS FILE EXISTS FOR, measured before anything was built.

``spacr.ml.check_and_clean_data`` defines ``gene_fraction`` as the SUM of the
gene's gRNA fractions within a well. The formula spaCR fitted until this
instruction was

    y ~ fraction:grna + gene_fraction:gene + rowID + columnID

so every ``gene_fraction:gene[G]`` column is the sum of gene G's
``fraction:grna`` columns whenever G's guides do not share a well. On the
maintainer's TSG101 screen (``regression_data.csv``, 1945 rows, 610 wells, 823
guides, 389 genes) that is true of 386 of the 389 genes -- the other three are
the only genes whose guides ever co-occur in one well:

    design            1945 x 1248 parameters, RANK 862, DEFICIENCY 386
    condition number  2.3e18
    null space        386 vectors built analytically; max |X v| = 0.0 EXACTLY
    single-guide genes whose gene column duplicates their guide column: 102
    SSE at statsmodels' answer          4.501363280047
    SSE at that answer + 7 * nullvector 4.501363280047   (difference 0.0)

That last pair is the whole defect: statsmodels does not refuse a singular
design, it pseudo-inverts, and the numbers it reports are one arbitrary
solution out of infinitely many. Split in two, each level is FULL RANK:
859 parameters at rank 859 for the guide fit, 425 at 425 for the gene fit.

These tests reconstruct that measurement on a frame small enough to read, and
then assert the fix: one level per design, two fits, two BH families.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg", force=True)


SPACR = pathlib.Path(__file__).resolve().parents[1] / "spacr"


# --------------------------------------------------------------------------- #
#  A frame with the real shape: some single-guide genes, some multi-guide
# --------------------------------------------------------------------------- #

def _screen(seed=0, n_plates=3, multi_guide=("g1", "g2"), single_guide=("s1",)):
    """One row per (well, guide), which is the shape the pipeline builds."""
    rng = np.random.default_rng(seed)
    guides = {}
    for gene in multi_guide:
        guides[gene] = [f"{gene}_1", f"{gene}_2", f"{gene}_3"]
    for gene in single_guide:
        guides[gene] = [f"{gene}_1"]
    recs = []
    for plate in range(n_plates):
        for row in ("r1", "r2", "r3"):
            for col in ("c1", "c2", "c3", "c4"):
                well = f"plate{plate + 1}_{row}_{col}"
                for gene, gene_guides in guides.items():
                    # ONE guide of each gene per well, which is the ordinary
                    # pooled-screen shape and the case the collinearity is
                    # exact in.
                    guide = gene_guides[rng.integers(len(gene_guides))]
                    recs.append({
                        "plateID": f"plate{plate + 1}",
                        "rowID": row,
                        "columnID": col,
                        "prc": well,
                        "gene": gene,
                        "grna": guide,
                        "fraction": float(rng.uniform(0.05, 0.6)),
                        "cell_count": int(rng.integers(40, 120)),
                    })
    frame = pd.DataFrame(recs)
    hot = multi_guide[0] if multi_guide else sorted(guides)[0]
    frame["predictions"] = (
        0.9 * frame["fraction"] * (frame["gene"] == hot)
        + 0.2 * frame["fraction"]
        + rng.normal(0, 0.05, len(frame)))
    return frame


# --------------------------------------------------------------------------- #
#  THE PREMISE. If this stops holding, the rest of the instruction is moot.
# --------------------------------------------------------------------------- #

def test_gene_fraction_is_the_sum_of_the_genes_guide_fractions():
    """The definition everything rests on, taken from the real function."""
    from spacr.ml import check_and_clean_data

    clean = check_and_clean_data(_screen(), "predictions")

    summed = clean.groupby(["prc", "gene"], observed=True)["fraction"].sum()
    stored = clean.groupby(["prc", "gene"], observed=True)["gene_fraction"].first()

    assert np.allclose(summed.to_numpy(), stored.to_numpy(), atol=0)


def test_the_old_design_is_rank_deficient_by_construction():
    """Not by accident of data: an exact null space, built analytically.

    Reproduces the TSG101 measurement in miniature. The guide columns and
    their own per-gene sums are in one design, so for every gene whose guides
    never share a well there is a vector v with X v = 0 EXACTLY -- and the
    fitted coefficients can be moved along v without changing a single fitted
    value.
    """
    from patsy import dmatrices

    from spacr.ml import check_and_clean_data

    clean = check_and_clean_data(_screen(), "predictions")
    collinear = ("predictions ~ fraction:grna + gene_fraction:gene "
                 "+ rowID + columnID")
    y, X = dmatrices(collinear, data=clean, return_type="dataframe")
    design = X.to_numpy(dtype=float)

    rank = np.linalg.matrix_rank(design)
    assert rank < design.shape[1], (
        "the premise of instruction 132 no longer holds: the combined design "
        "is full rank on this frame")

    # The null vectors, CONSTRUCTED rather than found numerically: one per
    # gene, +1 on its gene column and -1 on each of its guide columns.
    columns = {name: index for index, name in enumerate(X.columns)}
    null_vectors = []
    for gene, gene_guides in clean.groupby("gene", observed=True)["grna"]:
        gene_column = f"gene_fraction:gene[{gene}]"
        guide_columns = [f"fraction:grna[{g}]" for g in sorted(set(gene_guides))]
        if gene_column not in columns:
            continue
        if any(name not in columns for name in guide_columns):
            continue
        vector = np.zeros(design.shape[1])
        vector[columns[gene_column]] = 1.0
        for name in guide_columns:
            vector[columns[name]] -= 1.0
        null_vectors.append(vector)

    basis = np.array(null_vectors).T
    residual = np.abs(design @ basis).max()
    assert residual == 0.0, (
        f"the collinearity must be EXACT, not merely severe; max |X v| = "
        f"{residual!r}")

    # ...and therefore the fit is one answer out of infinitely many.
    import statsmodels.api as sm

    fitted = sm.OLS(y, X).fit()
    beta = fitted.params.to_numpy()
    moved = beta + 7.0 * null_vectors[0]
    response = y.to_numpy().ravel()

    # Coefficients that differ by SEVEN produce the same fitted values, so the
    # data cannot tell the two answers apart. (On the maintainer's real screen
    # the residual sums of squares are bit-identical: 4.501363280047 both
    # ways. On a frame this small the x7 shift rides a little float noise, so
    # the fitted values are what the assertion measures.)
    assert np.abs(moved - beta).max() == pytest.approx(7.0)
    assert np.allclose(design @ beta, design @ moved, rtol=0, atol=1e-9)
    sse_beta = float(np.sum((response - design @ beta) ** 2))
    sse_moved = float(np.sum((response - design @ moved) ** 2))
    assert abs(sse_moved - sse_beta) <= 1e-12 * sse_beta, (
        "moving the coefficients along a null vector must not change the fit "
        "-- that is what makes the reported numbers arbitrary")


def test_each_single_level_design_is_full_rank_where_the_combined_one_is_not():
    """The fix, measured the same way the defect was."""
    from patsy import dmatrices

    from spacr.ml import check_and_clean_data, prepare_formula

    clean = check_and_clean_data(_screen(), "predictions")

    for level in ("grna", "gene"):
        _y, X = dmatrices(prepare_formula("predictions", level=level),
                          data=clean, return_type="dataframe")
        design = X.to_numpy(dtype=float)
        assert np.linalg.matrix_rank(design) == design.shape[1], (
            f"the {level} design is rank deficient with "
            f"{design.shape[1]} parameters")


# --------------------------------------------------------------------------- #
#  1. NEVER FIT THE COLLINEAR FORMULA AGAIN
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("level", ["grna", "gene"])
@pytest.mark.parametrize("random_row_column_effects", [False, True])
@pytest.mark.parametrize("block_screen", [False, True])
def test_prepare_formula_emits_exactly_one_level(level, random_row_column_effects,
                                                 block_screen):
    from spacr.ml import prepare_formula

    formula = prepare_formula("predictions", level=level,
                              random_row_column_effects=random_row_column_effects,
                              block_screen=block_screen)

    has_guide = "fraction:grna" in formula
    has_gene = "gene_fraction:gene" in formula
    assert has_guide != has_gene, formula
    assert has_guide is (level == "grna"), formula


def test_prepare_formula_refuses_both_because_both_is_two_fits():
    from spacr.ml import prepare_formula

    with pytest.raises(ValueError, match="TWO fits"):
        prepare_formula("predictions", level="both")


def test_prepare_formula_refuses_a_level_that_is_not_one():
    from spacr.ml import prepare_formula

    with pytest.raises(ValueError, match="not a model level"):
        prepare_formula("predictions", level="object")


def test_no_formula_anywhere_in_spacr_names_both_levels():
    """THE OLD FORMULA IS GONE FROM THE CODEBASE.

    Every string literal in every shipped module that looks like a patsy
    formula (it contains ``~``) is checked. Docstrings are excluded on
    purpose: :func:`spacr.ml.prepare_formula` and its neighbours DESCRIBE the
    removed design at length, and having to describe a bug in order to say it
    is fixed must not be what trips this test. A literal that a run could
    actually hand to patsy is what is forbidden.
    """
    from spacr.ml import COLLINEAR_FORMULA_FRAGMENT

    assert COLLINEAR_FORMULA_FRAGMENT == "fraction:grna + gene_fraction:gene"

    offenders = []
    for path in sorted(SPACR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        # Docstrings describe the removed design at length -- saying what was
        # wrong is how the fix is explained -- so they are not literals a run
        # could fit and are excluded.
        exempt = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    exempt.add(id(body[0].value))

        # A constant whose NAME says it is the retired formula is allowed to
        # hold it: the GUI has to be able to show the user what changed, and
        # spacr.ml keeps it so this test has something exact to match. What is
        # forbidden is the fragment anywhere a design gets built.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            names = [t.id for t in targets if isinstance(t, ast.Name)]
            if not any("COLLINEAR" in name.upper() for name in names):
                continue
            for inner in ast.walk(node):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    exempt.add(id(inner))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, str) or id(node) in exempt:
                continue
            if COLLINEAR_FORMULA_FRAGMENT in node.value:
                offenders.append(f"{path.relative_to(SPACR.parent)}:"
                                 f"{node.lineno}: {node.value!r}")
    assert offenders == [], (
        "these string literals build a design containing both levels, which "
        "is the collinear model instruction 132 removed. gene_fraction is the "
        "SUM of the gene's guide fractions, so the two blocks cannot be "
        "estimated together:\n" + "\n".join(offenders))


def test_a_real_run_never_builds_a_design_with_both_levels():
    """Behavioural, not textual: look at the columns actually fitted."""
    from patsy import dmatrices

    from spacr.ml import check_and_clean_data, prepare_formula, resolve_levels

    clean = check_and_clean_data(_screen(), "predictions")

    for level in resolve_levels("ols", "both"):
        _y, X = dmatrices(prepare_formula("predictions", level=level),
                          data=clean, return_type="dataframe")
        names = [str(name) for name in X.columns]
        guide_columns = [n for n in names if n.startswith("fraction:grna")]
        gene_columns = [n for n in names if n.startswith("gene_fraction:gene")]
        assert not (guide_columns and gene_columns), (level, names[:6])


# --------------------------------------------------------------------------- #
#  level='both' is TWO FITS
# --------------------------------------------------------------------------- #

def test_resolve_levels_says_both_is_two_and_mixed_is_one():
    from spacr.ml import resolve_levels

    assert resolve_levels("ols", "both") == ("grna", "gene")
    assert resolve_levels("ols", "grna") == ("grna",)
    assert resolve_levels("ols", "gene") == ("gene",)
    # mixed contains both levels in one model, so it never fits twice.
    assert resolve_levels("mixed", "both") == ("gene",)
    assert resolve_levels("mixed", "grna") == ("gene",)
    with pytest.raises(ValueError, match="not a model level"):
        resolve_levels("ols", "well")


def test_regression_levels_fits_twice_and_the_two_models_differ(tmp_path):
    """Two models, two tables -- NOT one design containing both."""
    from spacr.ml import regression_levels

    frame = _screen(seed=3)
    fits = regression_levels(
        frame, str(tmp_path / "counts.csv"), dependent_variable="predictions",
        regression_type="ols", level="both", dst=str(tmp_path),
        controls=[], qc=False, plot=False)

    assert list(fits) == ["grna", "gene"]
    guide_table = fits["grna"][1]
    gene_table = fits["gene"][1]

    # Two DIFFERENT fitted objects, not one model read twice.
    assert fits["grna"][0] is not fits["gene"][0]

    assert set(guide_table["level"]) == {"grna"}
    assert set(gene_table["level"]) == {"gene"}
    assert guide_table["feature"].str.startswith("fraction:grna").any()
    assert not guide_table["feature"].str.startswith("gene_fraction").any()
    assert gene_table["feature"].str.startswith("gene_fraction:gene").any()
    assert not gene_table["feature"].str.startswith("fraction:grna").any()

    # The single-guide gene is the case that used to come back IDENTICAL to
    # its one guide (244480 / 244480_3, both 3.389291 at 2.873149e-13). Two
    # separate fits give it two different numbers.
    guide = float(guide_table.loc[
        guide_table["feature"] == "fraction:grna[s1_1]", "coefficient"].iloc[0])
    gene = float(gene_table.loc[
        gene_table["feature"] == "gene_fraction:gene[s1]", "coefficient"].iloc[0])
    assert guide != gene, (
        "a single-guide gene coming back identical to its guide is the "
        "signature of the collinear design")


def test_regression_refuses_to_fit_both_in_one_call(tmp_path):
    from spacr.ml import regression

    with pytest.raises(ValueError, match="fits ONE level"):
        regression(_screen(), str(tmp_path / "c.csv"), "predictions",
                   regression_type="ols", level="both", dst=None, qc=False)


def test_two_fits_write_their_figures_to_separate_folders(tmp_path):
    """One regression_figure.pdf per fit, so the second cannot overwrite it."""
    from spacr.ml import regression_levels

    regression_levels(
        _screen(seed=4), str(tmp_path / "counts.csv"),
        dependent_variable="predictions", regression_type="ols",
        level="both", dst=str(tmp_path), controls=[], qc=False, plot=False)

    assert (tmp_path / "grna" / "regression_figure.pdf").is_file()
    assert (tmp_path / "gene" / "regression_figure.pdf").is_file()


# --------------------------------------------------------------------------- #
#  3. TWO BH FAMILIES
# --------------------------------------------------------------------------- #

def _settings(**over):
    base = {
        "controls": [],
        "threshold_method": "std",
        "threshold_multiplier": 3.0,
        "multiple_testing_method": "fdr_bh",
        "fdr_alpha": 0.05,
        "l1_ratio": 0.5,
    }
    base.update(over)
    return base


def test_each_fit_is_corrected_within_itself_and_never_pooled(tmp_path):
    """BH over the guide family alone, and BH over the gene family alone.

    Pooling would be wrong twice: the two sets come from the same wells and
    the gene regressor IS the sum of the guide regressors, so they are not
    independent; and doubling the family costs power for no protection.
    """
    from spacr.ml import (_annotate_level_coefficients, _call_level_hits,
                          regression_levels)
    from spacr.multiple_testing import adjust_p_values

    frame = _screen(seed=5)
    fits = regression_levels(
        frame, str(tmp_path / "counts.csv"), dependent_variable="predictions",
        regression_type="ols", level="both", dst=None, controls=[],
        qc=False, plot=False)

    counts_grna = frame["grna"].value_counts().reset_index()
    counts_grna.columns = ["grna", "n_grna"]
    counts_gene = frame["gene"].value_counts().reset_index()
    counts_gene.columns = ["gene", "n_gene"]

    corrected = {}
    for level, (_model, table, _type) in fits.items():
        table = _annotate_level_coefficients(table, counts_grna, counts_gene)
        table, _hits, _cut, _rule = _call_level_hits(
            table, level, _settings(), "ols", frame, "predictions")
        corrected[level] = table

    guide_q = corrected["grna"].dropna(subset=["q_value"])
    gene_q = corrected["gene"].dropna(subset=["q_value"])
    assert len(guide_q) > 0 and len(gene_q) > 0

    # THE FAMILY SIZES ARE DIFFERENT, which is the whole point.
    assert len(guide_q) != len(gene_q)

    # Each level's q values are exactly BH over THAT level's p values.
    for level, table in (("grna", guide_q), ("gene", gene_q)):
        expected, _rejected = adjust_p_values(
            table["p_value"].to_numpy(dtype=float), method="fdr_bh",
            alpha=0.05)
        assert np.allclose(table["q_value"].to_numpy(dtype=float), expected), \
            level

    # ...and NOT BH over the two stacked together. A pooled correction would
    # scale every q by the larger family, so it cannot coincide.
    pooled_p = np.concatenate([guide_q["p_value"].to_numpy(dtype=float),
                               gene_q["p_value"].to_numpy(dtype=float)])
    pooled_q, _ = adjust_p_values(pooled_p, method="fdr_bh", alpha=0.05)
    separate_q = np.concatenate([guide_q["q_value"].to_numpy(dtype=float),
                                 gene_q["q_value"].to_numpy(dtype=float)])
    assert not np.allclose(pooled_q, separate_q), (
        "the two families were pooled: the q values match a single BH over "
        "both tables")


def test_the_gene_fit_measures_its_effect_size_cut_on_gene_controls(tmp_path):
    """A control list names GUIDES; the gene fit has none of them by name.

    Matching the guide list against the gene table selects nothing, so the
    gene table silently got no effect-size cut at all. The control guide
    identifies its gene by spaCR's own rule (truncate at the first
    underscore), which is what the gene fit matches on.
    """
    from spacr.ml import _annotate_level_coefficients, _level_control_rows, \
        regression_levels

    frame = _screen(seed=6)
    fits = regression_levels(
        frame, str(tmp_path / "counts.csv"), dependent_variable="predictions",
        regression_type="ols", level="both", dst=None, controls=["s1_1"],
        qc=False, plot=False)

    counts_grna = frame["grna"].value_counts().reset_index()
    counts_grna.columns = ["grna", "n_grna"]
    counts_gene = frame["gene"].value_counts().reset_index()
    counts_gene.columns = ["gene", "n_gene"]

    guide_table = _annotate_level_coefficients(fits["grna"][1], counts_grna,
                                               counts_gene)
    gene_table = _annotate_level_coefficients(fits["gene"][1], counts_grna,
                                              counts_gene)

    guide_controls = _level_control_rows(guide_table, "grna", ["s1_1"])
    gene_controls = _level_control_rows(gene_table, "gene", ["s1_1"])

    assert list(guide_controls["grna"]) == ["s1_1"]
    assert list(gene_controls["gene"]) == ["s1"], (
        "the gene fit found no controls, so it would get no effect-size cut")


# --------------------------------------------------------------------------- #
#  2. MIXED GETS THE REAL NESTING
# --------------------------------------------------------------------------- #

def test_mixed_fits_the_nested_model_and_returns_blups_not_p_values(tmp_path):
    """y ~ gene_fraction:gene + (1 | gene/grna) + rowID + columnID."""
    from spacr.ml import TERM_BLUP, TERM_FIXED, regression_levels

    frame = _screen(seed=7, n_plates=4,
                    multi_guide=("g1", "g2", "g3"), single_guide=())
    fits = regression_levels(
        frame, str(tmp_path / "counts.csv"), dependent_variable="predictions",
        regression_type="mixed", level="both", dst=None, controls=[],
        qc=False, plot=False)

    # ONE fit, not two: the model already contains both levels.
    assert list(fits) == ["gene"]
    _model, table, kind = fits["gene"]
    assert kind == "mixed"

    fixed = table[table["term_type"] == TERM_FIXED]
    blups = table[table["term_type"] == TERM_BLUP]

    # The gene is a FIXED effect with a p-value...
    assert fixed["feature"].str.startswith("gene_fraction:gene").any()
    gene_rows = fixed[fixed["feature"].str.startswith("gene_fraction:gene")]
    assert np.isfinite(gene_rows["p_value"].to_numpy()).all()

    # ...and the guide is a RANDOM effect, so it comes back as a BLUP with NO
    # p-value. Instruction 132: "do not manufacture a p-value for a random
    # effect".
    assert len(blups) == frame["grna"].nunique()
    assert blups["p_value"].isna().all()
    assert blups["feature"].str.startswith("blup:grna[").all()

    # The guide's old FIXED term is gone entirely -- that is the collinearity.
    assert not table["feature"].str.startswith("fraction:grna").any()


def test_the_mixed_run_writes_a_guide_table_of_blups_with_no_q_value(tmp_path):
    """results_grna.csv from a mixed run carries BLUPs and no correction."""
    from spacr.ml import (TERM_BLUP, _annotate_level_coefficients,
                          _call_level_hits, regression_levels)

    frame = _screen(seed=8, n_plates=4,
                    multi_guide=("g1", "g2", "g3"), single_guide=())
    fits = regression_levels(
        frame, str(tmp_path / "counts.csv"), dependent_variable="predictions",
        regression_type="mixed", level="both", dst=None, controls=[],
        qc=False, plot=False)

    counts_grna = frame["grna"].value_counts().reset_index()
    counts_grna.columns = ["grna", "n_grna"]
    counts_gene = frame["gene"].value_counts().reset_index()
    counts_gene.columns = ["gene", "n_gene"]
    whole = _annotate_level_coefficients(fits["gene"][1], counts_grna,
                                         counts_gene)

    blups = whole[whole["term_type"] == TERM_BLUP]
    gene_only = whole[whole["term_type"] != TERM_BLUP]

    # The BLUP rows carry a guide id, so they can be written as a guide table.
    assert set(blups["grna"].dropna()) == set(frame["grna"].unique())

    # Correcting the gene fit must not touch, or be enlarged by, the BLUPs.
    corrected, _hits, _cut, _rule = _call_level_hits(
        gene_only, "gene", _settings(), "mixed", frame, "predictions")
    tested = corrected["q_value"].notna().sum()
    assert tested == corrected["feature"].str.startswith(
        "gene_fraction:gene").sum()


def test_a_blup_row_never_enters_a_multiple_testing_family(tmp_path):
    """A NaN p-value is not a test, and must not enlarge the family.

    Left in, the BLUPs would both weaken every real q value (a larger family)
    and come back with a q value of their own -- a p-value manufactured for a
    quantity that has none.
    """
    from spacr.ml import (TERM_BLUP, _annotate_level_coefficients,
                          _call_level_hits, regression_levels)

    frame = _screen(seed=9, n_plates=4,
                    multi_guide=("g1", "g2", "g3"), single_guide=())
    fits = regression_levels(
        frame, str(tmp_path / "counts.csv"), dependent_variable="predictions",
        regression_type="mixed", level="both", dst=None, controls=[],
        qc=False, plot=False)
    counts_grna = frame["grna"].value_counts().reset_index()
    counts_grna.columns = ["grna", "n_grna"]
    counts_gene = frame["gene"].value_counts().reset_index()
    counts_gene.columns = ["gene", "n_gene"]

    whole = _annotate_level_coefficients(fits["gene"][1], counts_grna,
                                         counts_gene)
    assert (whole["term_type"] == TERM_BLUP).any()

    corrected, _hits, _cut, _rule = _call_level_hits(
        whole, "gene", _settings(), "mixed", frame, "predictions")

    blups = corrected[corrected["term_type"] == TERM_BLUP]
    assert len(blups) > 0
    assert blups["q_value"].isna().all(), (
        "a BLUP was given a q value, which means a p-value was manufactured "
        "for a random effect")


def test_mixed_says_so_rather_than_faking_a_nesting_it_cannot_fit():
    """One guide per gene confounds the guide variance with the residual."""
    from spacr.ml import fit_mixed_model, prepare_formula

    frame = _screen(seed=10, multi_guide=(), single_guide=("s1", "s2", "s3"))
    formula = prepare_formula("predictions", level="gene")

    with pytest.raises(ValueError, match="no gene in this frame has more"):
        fit_mixed_model(frame, formula, dst=None)


def test_mixed_refuses_a_single_gene_screen():
    from spacr.ml import fit_mixed_model, prepare_formula

    frame = _screen(seed=11, multi_guide=("g1",), single_guide=())
    formula = prepare_formula("predictions", level="gene")

    with pytest.raises(ValueError, match="at least two clusters"):
        fit_mixed_model(frame, formula, dst=None)


# --------------------------------------------------------------------------- #
#  The identifiability warning is counted at the level being fitted
# --------------------------------------------------------------------------- #

def test_the_identifiability_warning_counts_the_level_it_is_about():
    """823 guides against 610 wells is unidentifiable; 389 genes is not.

    Counting guides for the gene fit would fire a warning the gene fit does
    not deserve, and a warning that cries wolf is one people scroll past.
    """
    from spacr.ml import _identifiability_warning

    # 12 wells, 9 guides across 3 genes: wide at the guide level once the
    # plate blocks are counted, narrow at the gene level.
    frame = _screen(seed=12, n_plates=1,
                    multi_guide=("g1", "g2", "g3"), single_guide=())
    settings = {"guide_permutation_block": "plateID"}

    assert _identifiability_warning(frame, settings, level="gene") is None

    narrow = frame[frame["prc"].isin(sorted(frame["prc"].unique())[:3])]
    guide_warning = _identifiability_warning(narrow, settings, level="grna")
    gene_warning = _identifiability_warning(narrow, settings, level="gene")
    assert guide_warning is not None and "grnas" in guide_warning
    assert gene_warning is not None and "genes" in gene_warning
