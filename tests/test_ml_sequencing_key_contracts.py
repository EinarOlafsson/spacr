"""Key parsing and merge cardinality contracts in ml.py and sequencing.py.

Two families of bug live here, and both are silent:

**Positional underscore splitting.** ``spacr.schema`` exists so that a key is
parsed in exactly one place, right to left, because the plate id is the only
component of ``prc`` / ``prcf`` / ``prcfo`` that may itself contain the
separator. Seven sites in these two modules still spelled the split by hand as
``df['prc'].str.split('_', expand=True)`` assigned to a fixed number of column
names. On a plate called ``exp1_plate1`` that either raises ``ValueError:
Columns must be same length as key`` inside a ``try`` that swallows it, or --
worse -- shifts every metadata column one place along for the plates that do
*not* carry the extra underscore.

**Merges with no key contract.** A ``pd.merge`` with no ``validate=`` that
should have been many-to-one but is many-to-many returns MORE rows than it was
given, and every count computed downstream inflates with nothing anywhere
saying so. The regression path's own join (``independent_df`` x
``dependent_df`` on ``prc``) is the one that matters most: it is the step that
decides the hits.

Every test here states the OLD behaviour explicitly -- by replaying the code
that was replaced -- so it fails against the old implementation rather than
merely describing the new one.
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
# Synthetic screen builders (a pooled CRISPR screen small enough to fit)
# ---------------------------------------------------------------------------

GENES = ("000000", "233460", "239740", "111111")
ROWS = ("r1", "r2", "r3")
COLS = ("c1", "c2", "c3", "c4", "c5", "c6")
CONTROLS = [f"000000_{i}" for i in (1, 2, 3)]


def _grnas():
    """gRNA names in the ``org_gene_guide`` form process_reads splits on."""
    return [f"TGGT1_{g}_{i}" for g in GENES for i in (1, 2, 3)]


def write_scores(path, plate="plate1", seed=0, n_cells=6):
    """Per-object score CSV: plate x row x column x cell."""
    rng = np.random.default_rng(seed)
    recs = []
    for r in ROWS:
        for c in COLS:
            base = float(rng.uniform(0.2, 0.8))
            for _ in range(n_cells):
                recs.append({
                    "plateID": plate, "rowID": r, "columnID": c,
                    "fieldID": "f1",
                    "pred": float(np.clip(base + rng.normal(0, 0.1),
                                          0.02, 0.98)),
                })
    pd.DataFrame(recs).to_csv(path, index=False)
    return str(path)


def write_counts(path, plate="plate1", seed=1, row_id=lambda plate, r: r):
    """Per-well sgRNA count CSV.

    ``row_id`` decides what goes in the ``rowID`` column, so a test can write
    the composite ``'<plate>_<row>'`` form for some wells and the bare form for
    others -- which is the shape that used to erase two thirds of a screen.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for r in ROWS:
        for c in COLS:
            for g in _grnas():
                recs.append({"plateID": plate, "rowID": row_id(plate, r),
                             "columnID": c, "grna": g,
                             "count": int(rng.integers(20, 400))})
    pd.DataFrame(recs).to_csv(path, index=False)
    return str(path)


def base_settings(score, count, **over):
    """Settings finished by the same defaults builder every entry point uses."""
    from spacr.settings import get_perform_regression_default_settings

    settings = {
        "score_data": [score],
        "count_data": [count],
        "dependent_variable": "pred",
        "regression_type": "ols",
        "min_cell_count": 3,
        "fraction_threshold": 0.005,
        "toxo": False,
        "controls": list(CONTROLS),
        "outlier_detection": False,
        "alpha": 1.0,
    }
    settings.update(over)
    return get_perform_regression_default_settings(settings)


@pytest.fixture(autouse=True)
def _no_figure_leak():
    yield
    plt.close("all")


@pytest.fixture
def stubs(monkeypatch):
    """Stub only the visual helpers and the Monte-Carlo cell-count simulation.

    Everything this file is actually testing -- the key splitting, the merges,
    the QC tables -- runs for real.
    """
    import spacr.plot as P
    import spacr.ml as ML
    import spacr.toxo  # noqa: F401  (warm the lazy imports)
    import spacr.sequencing  # noqa: F401

    rec = {"plates": [], "csv_plots": []}

    monkeypatch.setattr(P, "plot_plates",
                        lambda df, **kw: rec["plates"].append(len(df)))
    monkeypatch.setattr(P, "plot_histogram", lambda df, column, dst=None: None)
    monkeypatch.setattr(P, "plot_data_from_csv",
                        lambda settings: (rec["csv_plots"].append(dict(settings))
                                          or (None, None)))
    monkeypatch.setattr(ML, "minimum_cell_simulation", lambda settings, **kw: 3)
    return rec


def _results_dir(count_csv, score_stem, regression_type="ols"):
    return os.path.join(os.path.dirname(count_csv), "results", score_stem,
                        regression_type, "list")


# ===========================================================================
# Job A -- prc / prcf splits routed through schema
# ===========================================================================

def test_split_prc_parses_right_to_left_like_schema_does():
    """The plate keeps its own underscores; row and column are read off the end."""
    from spacr.ml import _split_prc

    assert _split_prc("plate1_r1_c1") == ("plate1", "r1", "c1")
    assert _split_prc("exp1_plate1_r2_c12") == ("exp1_plate1", "r2", "c12")
    # Tokens are returned exactly as written -- nothing is canonicalised,
    # because the caller rebuilds prc from these columns and a rewritten token
    # would change the identity every downstream frame is joined on.
    assert _split_prc("plate1_A_1") == ("plate1", "A", "1")


def test_split_prc_names_a_string_that_is_not_a_well_key():
    """Too few components is a named schema error, not an IndexError later."""
    from spacr import schema
    from spacr.ml import _split_prc

    with pytest.raises(schema.KeyParseError, match="is not a prc"):
        _split_prc("plate1_r1")
    with pytest.raises(schema.KeyParseError, match="is not a prc"):
        _split_prc(float("nan"))


def test_assign_prc_parts_handles_an_empty_frame():
    """A count table filtered down to nothing must not raise on its way out."""
    from spacr.ml import _assign_prc_parts

    empty = pd.DataFrame({"prc": pd.Series([], dtype=object)})
    out = _assign_prc_parts(empty)

    assert list(out.columns) == ["prc", "plateID", "rowID", "columnID"]
    assert len(out) == 0


def test_an_underscored_plate_id_keeps_the_regression_qc_tables(tmp_path, stubs):
    """A plate called 'exp1_plate1' no longer loses grna_well.csv / well_grna.csv.

    ``merged_df[['plateID','rowID','columnID']] = merged_df['prc'].str.split(...)``
    raised ``ValueError: Columns must be same length as key`` on a four-token
    prc. In :func:`perform_regression` that assignment and the two QC tables
    that follow it sit inside one ``try``/``except Exception: print(e)``, so
    the whole QC block was skipped and the only evidence was one line of
    stdout. The regression itself then ran on a frame with no plateID column.
    """
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    plate = "exp1_plate1"
    score = write_scores(sdir / "scores.csv", plate=plate)
    count = write_counts(cdir / "counts.csv", plate=plate)

    # settings['plateID'] is stamped over a single-plate score frame by
    # process_scores, so it has to name the plate the CSVs actually carry.
    out = perform_regression(base_settings(score, count, plateID=plate))

    res = _results_dir(count, "scores")
    data = pd.read_csv(os.path.join(res, "regression_data.csv"))
    assert set(data["plateID"].unique()) == {plate}
    assert set(data["rowID"].unique()) <= set(ROWS)
    assert set(data["columnID"].unique()) <= set(COLS)

    # The QC tables the swallowed ValueError used to cost us, and the plate id
    # inside them -- grna_metricks splits prc twice more.
    grna_well = pd.read_csv(os.path.join(res, "grna_well.csv"))
    well_grna = pd.read_csv(os.path.join(res, "well_grna.csv"))
    assert set(grna_well["plateID"].unique()) == {plate}
    assert set(well_grna["plateID"].unique()) == {plate}
    assert set(well_grna["rowID"].unique()) <= set(ROWS)
    assert len(out["results"]) > 0

    # ... and the old spelling, on the very frame that just worked.
    with pytest.raises(ValueError, match="Columns must be same length as key"):
        legacy = data[["prc"]].copy()
        legacy[["plateID", "rowID", "columnID"]] = \
            legacy["prc"].str.split("_", expand=True)


def test_a_count_table_where_only_some_rows_carry_the_plate_prefix(tmp_path,
                                                                  stubs):
    """Every well survives; the old iloc[0] rule silently deleted two thirds.

    ``rowID`` in a count CSV is sometimes the composite ``'<plate>_<row>'``.
    The old code looked at ``rowID.iloc[0]``, counted its parts, and then
    applied ``split[1]`` to the whole column -- so on a table where only the
    r1 wells carry the prefix, every r2 and r3 row got ``NaN`` for rowID,
    ``NaN`` for prc, and was dropped by the groupby and the merge without a
    word.
    """
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv")
    count = write_counts(
        cdir / "counts.csv",
        row_id=lambda plate, r: f"{plate}_{r}" if r == "r1" else r)

    perform_regression(base_settings(score, count))

    data = pd.read_csv(os.path.join(_results_dir(count, "scores"),
                                    "regression_data.csv"))
    assert set(data["rowID"].unique()) == set(ROWS)

    # The old rule, on the same column: r2 and r3 become NaN.
    raw = pd.read_csv(count)
    legacy = raw.copy()
    assert len(legacy["rowID"].iloc[0].split("_")) == 2
    legacy["rowID"] = legacy["rowID"].str.split("_", expand=True)[1]
    assert legacy["rowID"].isna().sum() == (raw["rowID"] == "r2").sum() \
        + (raw["rowID"] == "r3").sum()


def test_a_row_id_carrying_a_plate_whose_name_has_an_underscore(tmp_path,
                                                               stubs):
    """'exp1_plate1_r2' has three parts, so the old num_parts==2 rule skipped it."""
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    plate = "exp1_plate1"
    score = write_scores(sdir / "scores.csv", plate=plate)
    count = write_counts(cdir / "counts.csv", plate=plate,
                         row_id=lambda p, r: f"{p}_{r}")

    perform_regression(base_settings(score, count, plateID=plate))

    data = pd.read_csv(os.path.join(_results_dir(count, "scores"),
                                    "regression_data.csv"))
    assert set(data["rowID"].unique()) == set(ROWS)
    assert set(data["plateID"].unique()) == {plate}

    # The old rule left it untouched: three parts is not two.
    raw = pd.read_csv(count)
    assert len(raw["rowID"].iloc[0].split("_")) == 3


def test_process_reads_plate_row_splits_on_the_last_separator():
    """'plate_row' is <plate>_<row>; only the row is guaranteed separator-free."""
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plate_row": ["exp1_plate2_rA"] * 4 + ["exp1_plate2_rB"] * 4,
        "columnID": ["c3"] * 8,
        "grna": _grnas()[:4] * 2,
        "count": [10, 20, 30, 40] * 2,
    })

    out = process_reads(df.copy(), fraction_threshold=None, plate=None)

    assert set(out["prc"]) == {"exp1_plate2_rA_c3", "exp1_plate2_rB_c3"}
    assert np.allclose(
        sorted(out.loc[out["prc"] == "exp1_plate2_rA_c3", "fraction"]),
        [0.1, 0.2, 0.3, 0.4])

    # The old two-column positional split on the same input.
    with pytest.raises(ValueError, match="Columns must be same length as key"):
        legacy = df.copy()
        legacy[["plateID", "rowID"]] = \
            legacy["plate_row"].str.split("_", expand=True)


def test_process_reads_names_a_plate_row_with_no_separator():
    """A 'plate_row' that is only a plate says so, instead of a pandas message."""
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plate_row": ["plate2"] * 4,
        "columnID": ["c3"] * 4,
        "grna": _grnas()[:4],
        "count": [10, 20, 30, 40],
    })

    with pytest.raises(ValueError, match=r"'plate_row' must be"):
        process_reads(df, fraction_threshold=None, plate=None)


def test_process_reads_refuses_a_half_applied_grna_split(capsys):
    """Mixed-width gRNA names are refused whole rather than half-applied.

    ``str.split('_', expand=True)`` pads with ``None`` instead of raising when
    the widths differ, so a two-token name next to three-token names got
    ``gene`` = its GUIDE token and then ``grna`` = ``NaN`` from the
    concatenation -- that gRNA's reads were deleted from the screen while
    every other row went through untouched.
    """
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plateID": ["plate1"] * 4,
        "rowID": ["r1"] * 4,
        "columnID": ["c1"] * 4,
        "grna": ["TGGT1_GENEA_g1", "GENEA_g2",
                 "TGGT1_GENEB_g1", "TGGT1_GENEB_g2"],
        "count": [10, 20, 30, 40],
    })

    out = process_reads(df.copy(), fraction_threshold=None, plate=None)

    assert "gene" not in out.columns          # refused, not half-applied
    assert out["grna"].isna().sum() == 0      # and nothing was deleted
    assert set(out["grna"]) == set(df["grna"])
    message = capsys.readouterr().out
    assert "Not splitting 'grna'" in message
    assert "[2, 3] component(s)" in message

    # The old spelling, on the same names.
    legacy = df[["grna"]].copy()
    legacy[["org", "gene", "grna"]] = legacy["grna"].str.split("_", expand=True)
    assert legacy["grna"].isna().sum() == 1   # the two-token name lost its guide
    assert legacy.loc[1, "gene"] == "g2"      # and its guide became its gene


def test_process_reads_still_splits_a_well_formed_grna_library():
    """The org_gene_guide convention is still honoured -- it is only now checked.

    Deliberately passes both before and after the change: it is the guard that
    the new width check did not turn a working library into a refused one.
    """
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plateID": ["plate1"] * 4,
        "rowID": ["r1"] * 4,
        "columnID": ["c1"] * 4,
        "grna": ["TGGT1_GENEA_g1", "TGGT1_GENEA_g2",
                 "TGGT1_GENEB_g1", "TGGT1_GENEB_g2"],
        "count": [10, 20, 30, 40],
    })

    out = process_reads(df, fraction_threshold=None, plate=None)

    assert list(out["gene"]) == ["GENEA", "GENEA", "GENEB", "GENEB"]
    assert list(out["grna"]) == ["GENEA_g1", "GENEA_g2", "GENEB_g1", "GENEB_g2"]


def test_process_reads_leaves_a_library_with_no_org_gene_structure_alone(capsys):
    """Single-token gRNA names are a real library, and are passed through."""
    from spacr.ml import process_reads

    df = pd.DataFrame({
        "plateID": ["plate1"] * 3,
        "rowID": ["r1"] * 3,
        "columnID": ["c1"] * 3,
        "grna": ["g0", "g1", "g2"],
        "count": [10, 20, 30],
    })

    out = process_reads(df, fraction_threshold=None, plate=None)

    assert list(out.columns) == ["prc", "grna", "fraction"]
    assert list(out["grna"]) == ["g0", "g1", "g2"]
    assert "[1] component(s)" in capsys.readouterr().out


def test_graph_sequencing_stats_survives_a_mixed_row_id_column(tmp_path):
    """One composite rowID no longer makes every plain rowID raise IndexError.

    ``has_underscore = df['rowID'].str.contains('_').any()`` guarded the whole
    column, and then ``x.split('_')[1]`` ran on EVERY row. One
    ``'plate1_r1'`` anywhere in the table therefore indexed ``['r2']`` at
    position 1 and the caller lost the threshold it had already computed.
    """
    from spacr.sequencing import graph_sequencing_stats

    rng = np.random.default_rng(0)
    recs = []
    for w in range(9):
        r = f"r{(w % 3) + 1}"
        recs.extend({
            "plateID": "plate1",
            # only the r1 wells carry the plate prefix
            "rowID": f"plate1_{r}" if r == "r1" else r,
            "columnID": f"c{(w // 3) + 1}",
            "grna": f"g{g}",
            "count": int(rng.integers(5, 500)),
        } for g in range(15) if rng.random() < 0.7)
    csv = tmp_path / "counts.csv"
    pd.DataFrame(recs).to_csv(csv, index=False)

    threshold = graph_sequencing_stats({
        "count_data": str(csv),
        "target_unique_count": 5,
        "filter_column": "columnID",
        "control_wells": ["c1"],
        "log_x": False, "log_y": False,
    })

    assert 0.0 <= float(threshold) <= 1.0

    # The old expression, on the same column.
    with pytest.raises(IndexError):
        pd.Series(["plate1_r1", "r2", "r3"]).apply(lambda x: x.split("_")[1])


def test_graph_sequencing_stats_keeps_the_row_of_an_underscored_plate(
        tmp_path, monkeypatch):
    """'exp1_plate1_r2' reduces to 'r2', not to 'plate1'."""
    from spacr.sequencing import graph_sequencing_stats

    rng = np.random.default_rng(1)
    recs = []
    for w in range(9):
        r = f"r{(w % 3) + 1}"
        recs.extend({
            "plateID": "exp1_plate1", "rowID": f"exp1_plate1_{r}",
            "columnID": f"c{(w // 3) + 1}", "grna": f"g{g}",
            "count": int(rng.integers(5, 500)),
        } for g in range(15) if rng.random() < 0.7)
    csv = tmp_path / "counts.csv"
    pd.DataFrame(recs).to_csv(csv, index=False)

    seen = {}
    # Patch the callable's actual globals.  Package-lazy-loader tests can
    # replace ``spacr.sequencing`` in sys.modules, leaving a separately
    # imported module object whose ``plot_plates`` is not the global this
    # already-imported function resolves at call time.
    monkeypatch.setitem(
        graph_sequencing_stats.__globals__, "plot_plates",
        lambda df, **kw: seen.update(rows=sorted(df["rowID"].unique())),
    )
    graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 5,
        "filter_column": "columnID", "control_wells": ["c1"],
        "log_x": False, "log_y": False,
    })

    assert seen["rows"] == ["r1", "r2", "r3"]
    # The old [1] index would have handed plot_plates the plate's second token.
    assert "exp1_plate1_r2".split("_")[1] == "plate1"


# ===========================================================================
# Job B -- merge key contracts
# ===========================================================================

def test_the_regression_merge_refuses_a_duplicated_well_score(tmp_path, stubs,
                                                              monkeypatch):
    """pd.merge(independent_df, dependent_df, on='prc') is many-to-one.

    ``dependent_df`` is one row per well -- ``process_scores`` groups on
    ``prc``. If it ever holds a well twice (two score CSVs for one plate
    concatenated, an aggregation that did not aggregate), the join multiplies
    every gRNA row of that well: cell_count, the per-well gRNA counts and the
    regression's effective n all inflate, with no error and nothing in the
    output that looks wrong.
    """
    import spacr.ml as ML

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv")
    count = write_counts(cdir / "counts.csv")

    captured = {}
    real_process_scores = ML.process_scores

    def duplicating_process_scores(*args, **kwargs):
        dependent_df, variable = real_process_scores(*args, **kwargs)
        captured["dependent"] = dependent_df
        # One well aggregated twice -- exactly what a doubled score CSV gives.
        return (pd.concat([dependent_df, dependent_df.iloc[[0]]],
                          ignore_index=True), variable)

    monkeypatch.setattr(ML, "process_scores", duplicating_process_scores)

    with pytest.raises(pd.errors.MergeError, match="many-to-one"):
        ML.perform_regression(base_settings(score, count))

    # And what the un-validated merge did with the same two frames: silently
    # more rows out than the independent variable had in.
    independent = ML.process_reads(pd.read_csv(count), 0.005, "plate1")
    dependent = captured["dependent"]
    doubled = pd.concat([dependent, dependent.iloc[[0]]], ignore_index=True)
    inflated = pd.merge(independent, doubled, on="prc")
    honest = pd.merge(independent, dependent, on="prc")
    assert len(inflated) > len(honest)


def test_unaggregated_scores_are_still_allowed_to_cross_join(tmp_path, stubs):
    """agg_type=None is a deliberate many-to-many and must not be validated away.

    ``settings.py`` forces ``agg_type=None`` for quantile regression, and with
    no aggregation ``process_scores`` returns one row per OBJECT -- so the join
    pairs each of a well's gRNAs with each of its cells on purpose. A blanket
    ``validate='many_to_one'`` here would abort every quantile run on
    perfectly good data.

    The proof used to be that the run got past the merge and died further
    down, at "Unsupported regression type quantile" -- quantile had no backend
    at all. It has one now, so the proof is the stronger one: the run
    completes and returns a coefficient for every term of the cross join.
    """
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores(sdir / "scores.csv")
    count = write_counts(cdir / "counts.csv")

    settings = base_settings(score, count, regression_type="quantile")
    assert settings["agg_type"] is None       # set by the defaults builder

    out = perform_regression(settings)

    assert len(out["results"]) > 0
    assert out["results"]["coefficient"].notna().all()


def test_process_reads_well_total_merge_is_many_to_one():
    """The per-well total is broadcast onto the well's gRNAs, never the reverse."""
    import inspect

    from spacr import ml

    source = inspect.getsource(ml.process_reads)
    assert "validate='many_to_one'" in source

    # The contract holds on real input: one row in, one row out, per gRNA.
    df = pd.DataFrame({
        "plateID": ["plate1"] * 4, "rowID": ["r1"] * 4,
        "columnID": ["c1"] * 4, "grna": _grnas()[:4],
        "count": [10, 20, 30, 40],
    })
    out = ml.process_reads(df.copy(), fraction_threshold=None, plate=None)
    assert len(out) == len(df)
    assert out["fraction"].sum() == pytest.approx(1.0)


def test_graph_sequencing_stats_unique_count_merge_is_many_to_one(
        tmp_path, monkeypatch):
    """The per-well unique-gRNA count must not fan the read table out.

    ``unique_counts`` comes straight off a groupby on the join key, so it is
    one row per well; the read table is one row per (well, gRNA). If the right
    side ever duplicated, the plate heatmap would average a well's rows more
    than once and simply print the wrong number.
    """
    import inspect

    from spacr import sequencing as SEQ

    assert "validate='many_to_one'" in inspect.getsource(
        SEQ.graph_sequencing_stats)

    rng = np.random.default_rng(2)
    recs = []
    for w in range(9):
        recs.extend({
            "plateID": "plate1", "rowID": f"r{(w % 3) + 1}",
            "columnID": f"c{(w // 3) + 1}", "grna": f"g{g}",
            "count": int(rng.integers(5, 500)),
        } for g in range(12) if rng.random() < 0.7)
    csv = tmp_path / "counts.csv"
    pd.DataFrame(recs).to_csv(csv, index=False)

    seen = {}
    monkeypatch.setitem(
        SEQ.graph_sequencing_stats.__globals__, "plot_plates",
        lambda df, **kw: seen.update(n=len(df)),
    )
    SEQ.graph_sequencing_stats({
        "count_data": str(csv), "target_unique_count": 4,
        "filter_column": "columnID", "control_wells": ["c1"],
        "log_x": False, "log_y": False,
    })

    # The merge is the last thing to touch the row count before plot_plates,
    # and it is a LEFT many-to-one: it can never add a row.
    kept = pd.read_csv(csv)
    kept = kept[kept["columnID"] != "c1"]
    assert seen["n"] <= len(kept)


@pytest.mark.parametrize("merge_line", [
    "pd.merge(unique_triplets, grna_well_counts",
    "pd.merge(merged_df, gene_well_counts",
    "coef_df.merge(n_grna",
    "coef_df.merge(n_gene",
    "coef_df.merge(sel_df",
])
def test_every_merge_in_perform_regression_states_its_cardinality(merge_line):
    """No merge on the regression path is left without a key contract.

    69 of the 70 ``.merge()`` calls in this package used to pass no
    ``validate=``. These five are the ones inside ``perform_regression`` that
    are not otherwise reachable from a test without fitting a model per case;
    the contract itself is asserted here so a future edit cannot quietly drop
    it.
    """
    import inspect

    from spacr import ml

    source = inspect.getsource(ml.perform_regression)
    index = source.index(merge_line)
    window = source[index:index + 400]
    assert "validate=" in window, f"{merge_line} has no key contract"


# ===========================================================================
# Job C -- the repairs the adversarial review forced
# ===========================================================================

def test_split_prc_refuses_a_prcf_instead_of_absorbing_it_into_the_plate():
    """A four-component key is only an underscored plate if its tail is a well.

    This is the regression the review proved. Routing ``prc`` through
    ``schema.parse_prcf`` (by appending a field token) made the parse
    unconditionally right-to-left, so ``'plate1_r1_c1_f1'`` -- a ``prcf``
    handed to the function that takes a ``prc`` -- came back as
    ``('plate1_r1', 'c1', 'f1')``: half the well swallowed into the plate id,
    the row in the column slot and a FIELD id in the ``columnID`` slot. The
    ``str.split`` it replaced raised ``ValueError: Columns must be same length
    as key`` on exactly that input. Trading a loud failure for a silent
    mis-attribution is worse than the bug being fixed, so the ambiguous shape
    is refused and the error names which mistake was made.
    """
    from spacr import schema
    from spacr.ml import _split_prc

    # What the old positional split did with the same string: raise.
    legacy = pd.DataFrame({"prc": ["plate1_r1_c1_f1"]})
    with pytest.raises(ValueError, match="Columns must be same length as key"):
        legacy[["plateID", "rowID", "columnID"]] = \
            legacy["prc"].str.split("_", expand=True)

    with pytest.raises(schema.KeyParseError, match="prcf"):
        _split_prc("plate1_r1_c1_f1")
    with pytest.raises(schema.KeyParseError, match="prcfo"):
        _split_prc("plate1_r1_c1_f1_o2")
    with pytest.raises(schema.KeyParseError, match="timepoint"):
        _split_prc("plate1_r1_c1_t3")

    # ...and the case the right-to-left rule exists for still works, because
    # its trailing pair IS a row and a column.
    assert _split_prc("exp1_plate1_r2_c12") == ("exp1_plate1", "r2", "c12")
    assert _split_prc("a_b_exp1_plate1_r2_c12") == ("a_b_exp1_plate1",
                                                    "r2", "c12")
    # Row letters / bare column, and the equal-valued positional passthrough
    # parse_well produces for a well like '12', are rows and columns too.
    assert _split_prc("exp1_plate1_A_14") == ("exp1_plate1", "A", "14")
    assert _split_prc("exp1_plate1_12_12") == ("exp1_plate1", "12", "12")


def test_split_prc_refuses_an_unrecognisable_four_component_key():
    """When the tail proves nothing, the old loud failure is kept.

    ``'a_b_c_d'`` may be a plate ``'a_b'`` in row ``'c'``, or it may be a key
    from a pipeline spaCR has never seen. There is no evidence either way, so
    it raises -- which is what the ``str.split`` did -- rather than picking
    the reading that happens to be convenient.
    """
    from spacr import schema
    from spacr.ml import _split_prc

    with pytest.raises(schema.KeyParseError, match="are not a row and a column"):
        _split_prc("a_b_c_d")
    # A three-component key is still accepted whatever it holds: that is what
    # the split did, and nothing that used to parse may stop parsing.
    assert _split_prc("a_b_c") == ("a", "b", "c")
    # An empty plate is not a plate.
    with pytest.raises(schema.KeyParseError, match="no plate"):
        _split_prc("_r1_c1")


def test_assign_prc_parts_names_the_row_of_a_frame_holding_a_prcf():
    """The frame-level split reports the bad key rather than shifting a column."""
    from spacr import schema
    from spacr.ml import _assign_prc_parts

    df = pd.DataFrame({"prc": ["plate1_r1_c1", "plate1_r1_c1_f1"]})
    with pytest.raises(schema.KeyParseError, match="plate1_r1_c1_f1"):
        _assign_prc_parts(df)


# ---------------------------------------------------------------------------
# interpret_vision_model: the scores join is many-to-one, and _report_fan_out
# is not what enforces it
# ---------------------------------------------------------------------------

def _vision_measurements(n=6):
    """A frame shaped like ``io._read_and_merge_data``'s first return value."""
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": [f"o{i}" for i in range(1, n + 1)],
        "cell_channel_0_mean_intensity": np.linspace(100.0, 900.0, n),
        "cell_area": np.linspace(200.0, 1200.0, n),
    })


def test_report_fan_out_cannot_see_a_fan_out_across_an_inner_join():
    """The premise the un-validated merge was justified with is false.

    ``_report_fan_out`` checks ``len(merged) <= len(left)``. That is the
    cardinality contract only for a LEFT join. ``interpret_vision_model``
    joins INNER, so objects that fan out and objects that drop out cancel in
    the row count: four objects, a scores file holding ``o1`` twice and
    ``o2`` once, three rows out, three <= four, no error -- and ``o1``'s
    measurements are in the training set twice.
    """
    from spacr.io import _report_fan_out

    left = pd.DataFrame({"object_label": ["1", "2", "3", "4"],
                         "feature": [1.0, 2.0, 3.0, 4.0]})
    right = pd.DataFrame({"object_label": ["1", "1", "2"],
                          "score": [0.1, 0.9, 0.5]})
    merged = pd.merge(left, right, on=["object_label"], how="inner")

    assert len(merged) == 3 < len(left)
    assert (merged["object_label"] == "1").sum() == 2   # duplicated
    _report_fan_out(left, merged, ["object_label"],
                    left_name="object", right_name="scores")  # says nothing


def test_interpret_vision_model_refuses_a_scores_file_that_holds_an_object_twice(
        tmp_path, monkeypatch):
    """The duplicate is caught even though the inner join SHRANK the frame.

    Half the objects have no score, so ``len(merged) < len(df)`` and the
    row-count check is silent -- this is the exact shape
    ``_report_fan_out`` cannot see. The ``validate='many_to_one'`` on the
    merge is what stops it, and the message names the file, the repeated key
    and the fix instead of pandas' "Merge keys are not unique in right
    dataset".
    """
    import spacr.io
    from spacr.io import JoinFanOut
    from spacr.ml import interpret_vision_model

    df = _vision_measurements(6)

    def _fake_merge(locs, tables, verbose=False, nuclei_limit=None,
                    pathogen_limit=None, **kwargs):
        return df.copy(), []

    monkeypatch.setattr(spacr.io, "_read_and_merge_data", _fake_merge)

    src = tmp_path / "plateA"
    src.mkdir()
    scores_csv = tmp_path / "scores.csv"
    # o1 scored twice (the scoring step ran twice and appended), o4..o6 never
    # scored at all -- so the join returns 4 rows against 6 objects.
    pd.DataFrame({
        "plateID": ["plate1"] * 4,
        "rowID": ["r1"] * 4,
        "columnID": ["c1"] * 4,
        "fieldID": ["f1"] * 4,
        "object": [1, 1, 2, 3],
        "cv_predictions": [0.1, 0.9, 0.4, 0.6],
    }).to_csv(scores_csv, index=False)

    settings = {
        "src": str(src), "scores": str(scores_csv), "tables": ["cell"],
        "score_column": "cv_predictions", "feature_importance": False,
        "permutation_importance": False, "shap": False, "top_features": 3,
        "n_jobs": 1, "save": False,
    }

    with pytest.raises(JoinFanOut) as excinfo:
        interpret_vision_model(dict(settings))

    message = str(excinfo.value)
    assert str(scores_csv) in message
    assert "de-duplicate" in message
    # The old code got here without a word: the same two frames, merged the
    # way it merged them, shrink AND duplicate at the same time.
    scores = pd.read_csv(scores_csv)
    scores["object_label"] = scores["object"].astype(str)
    left = df.copy()
    left["object_label"] = left["object_label"].str.replace("o", "")
    keys = ["plateID", "rowID", "columnID", "fieldID", "object_label"]
    unvalidated = pd.merge(left, scores[keys + ["cv_predictions"]], on=keys,
                           how="inner")
    assert len(unvalidated) == 4 < len(left)
    assert unvalidated["object_label"].duplicated().any()


def test_interpret_vision_model_still_joins_a_partially_scored_plate(
        tmp_path, monkeypatch):
    """The guard must not turn "not every object was scored" into a crash.

    An inner join legitimately drops objects with no score -- ``save_png``
    off for some fields, a crop that failed to write, a run that was
    interrupted. ``validate='many_to_one'`` says nothing about the LEFT side,
    so this has to keep working, and this is the test that proves the fix was
    not bought with a new refusal.
    """
    import spacr.io
    from spacr.ml import interpret_vision_model

    df = _vision_measurements(6)

    monkeypatch.setattr(
        spacr.io, "_read_and_merge_data",
        lambda locs, tables, verbose=False, nuclei_limit=None,
        pathogen_limit=None, **kw: (df.copy(), []))

    src = tmp_path / "plateB"
    src.mkdir()
    scores_csv = tmp_path / "partial_scores.csv"
    pd.DataFrame({
        "plateID": ["plate1"] * 3,
        "rowID": ["r1"] * 3,
        "columnID": ["c1"] * 3,
        "fieldID": ["f1"] * 3,
        "object": [1, 2, 3],
        "cv_predictions": [0.1, 0.4, 0.6],
    }).to_csv(scores_csv, index=False)

    merged = interpret_vision_model({
        "src": str(src), "scores": str(scores_csv), "tables": ["cell"],
        "score_column": "cv_predictions", "feature_importance": False,
        "permutation_importance": False, "shap": False, "top_features": 3,
        "n_jobs": 1, "save": False,
    })

    assert len(merged) == 3
    assert not merged["object_label"].duplicated().any()


# ---------------------------------------------------------------------------
# generate_ml_scores: the annotation join is many-to-one on the MEASUREMENTS
# ---------------------------------------------------------------------------

def _ml_score_frame(prcfos):
    """A measurement frame indexed on prcfo, as _read_and_merge_data returns."""
    n = len(prcfos)
    return pd.DataFrame({
        "plateID": ["plate1"] * n,
        "rowID": ["r1"] * n,
        "columnID": [f"c{(i % 2) + 1}" for i in range(n)],
        "fieldID": ["f1"] * n,
        "object_label": [str(i + 1) for i in range(n)],
        "cell_area": np.linspace(200.0, 1200.0, n),
        "cell_channel_0_mean_intensity": np.linspace(100.0, 900.0, n),
    }, index=pd.Index(list(prcfos), name="prcfo"))


def _install_ml_score_fakes(monkeypatch, frame, png_list_df):
    """Fake the two DB reads generate_ml_scores makes, nothing else."""
    import spacr.io
    import spacr.predictions

    monkeypatch.setattr(
        spacr.io, "_read_and_merge_data",
        lambda locs, tables, verbose=False, nuclei_limit=None,
        pathogen_limit=None, **kw: (frame.copy(), []))
    monkeypatch.setattr(spacr.io, "_read_db",
                        lambda db, tables=None: [png_list_df.copy()])
    monkeypatch.setattr(spacr.predictions, "migrate_prediction_columns",
                        lambda db: None)


def test_generate_ml_scores_refuses_two_sources_sharing_an_object_identity(
        tmp_path, monkeypatch):
    """Two src folders under one plate id put the same prcfo in twice.

    ``generate_ml_scores`` concatenates one frame per ``src``. Each frame is
    keyed on ``prcfo``, so if two folders were measured under the same plate
    id the SAME object identity now describes two different objects. The
    annotation join is ``annotated_df.merge(df, ...)``: png_list is
    legitimately 'many' (a database measured for cells and then for pathogens
    appends to the same table), the measurements must be 'one'. Without the
    contract the training set silently gains a copy of every duplicated
    object -- and it is the training set, so the model is fitted on it.
    """
    from spacr.ml import generate_ml_scores

    prcfos = [f"plate1_r1_c1_f1_o{i}" for i in range(1, 5)]
    frame = _ml_score_frame(prcfos)
    png_list = pd.DataFrame({"prcfo": prcfos,
                             "test": [1, 2, 1, 2]})
    _install_ml_score_fakes(monkeypatch, frame, png_list)

    src_a = tmp_path / "a"
    src_b = tmp_path / "b"
    for d in (src_a, src_b):
        (d / "measurements").mkdir(parents=True)

    with pytest.raises(pd.errors.MergeError, match="many.to.one"):
        generate_ml_scores({
            "src": [str(src_a), str(src_b)],
            "annotation_column": "test",
            "channel_of_interest": None,
            "verbose": False,
        })

    # What the un-validated merge did with the same two frames: every object
    # counted twice, and no error anywhere.
    doubled = pd.concat([frame, frame])
    unvalidated = png_list.set_index("prcfo").merge(
        doubled, left_index=True, right_index=True)
    assert len(unvalidated) == 2 * len(frame)


def test_generate_ml_scores_allows_a_second_crop_of_the_same_object(
        tmp_path, monkeypatch):
    """png_list holding two crops of one object is legal and must still join.

    A database measured twice -- cell crops, then pathogen crops -- appends to
    ``png_list``, so the annotation side genuinely repeats a ``prcfo``. That is
    the 'many' in many-to-one and a blanket one-to-one contract here would
    abort a perfectly good run. The proof the merge is reached and passes is
    that the failure that comes back is from further down, once the frame is
    handed to the model.
    """
    from spacr.ml import generate_ml_scores

    prcfos = [f"plate1_r1_c1_f1_o{i}" for i in range(1, 5)]
    frame = _ml_score_frame(prcfos)
    # o1 crops twice.
    png_list = pd.DataFrame({"prcfo": prcfos + [prcfos[0]],
                             "test": [1, 2, 1, 2, 1]})
    _install_ml_score_fakes(monkeypatch, frame, png_list)

    src_a = tmp_path / "one"
    (src_a / "measurements").mkdir(parents=True)

    captured = {}
    import spacr.ml as ML

    def _stop_after_merge(df, *args, **kwargs):
        captured["n"] = len(df)
        raise RuntimeError("stop after the merge")

    monkeypatch.setattr(ML, "ml_analysis", _stop_after_merge)

    with pytest.raises(RuntimeError, match="stop after the merge"):
        generate_ml_scores({
            "src": [str(src_a)],
            "annotation_column": "test",
            "channel_of_interest": None,
            "verbose": False,
        })

    # Five annotation rows against four objects: the 'many' side repeated and
    # the join was allowed to.
    assert captured["n"] == len(png_list)


def test_generate_ml_scores_annotation_merge_states_its_cardinality():
    """The contract is on the merge itself, not only on the cases above."""
    import inspect

    from spacr import ml

    source = inspect.getsource(ml.generate_ml_scores)
    index = source.index("annotated_df.merge(df")
    assert "validate='many_to_one'" in source[index:index + 200]


# ---------------------------------------------------------------------------
# plate_from_order: a crop FILE NAME is read left to right on purpose
# ---------------------------------------------------------------------------

def write_scores_with_paths(path, plate="PLATE1", seed=0, n_cells=4,
                            stem=lambda plate, well, i: f"{plate}_{well}_1_1_{i}",
                            columns_hold_the_well=True):
    """Per-object score CSV carrying the crop file name in 'path'.

    ``plate_from_order=True`` reads the well out of that name instead of
    trusting the rowID / columnID columns. ``columns_hold_the_well=False``
    fills those columns with junk, so the only way the run can land on the
    real wells is by parsing the file name; ``True`` leaves them correct, so
    the only way it can land on the real wells with an unparseable name is by
    leaving them alone.
    """
    from spacr import schema

    rng = np.random.default_rng(seed)
    recs = []
    for r in ROWS:
        for c in COLS:
            well = schema.well_id(r, c)          # ('r2','c12') -> 'B12'
            base = float(rng.uniform(0.2, 0.8))
            for i in range(n_cells):
                recs.append({
                    "plateID": "plate1",
                    "rowID": r if columns_hold_the_well else "rWRONG",
                    "columnID": c if columns_hold_the_well else "cWRONG",
                    "fieldID": "f1",
                    "path": stem(plate, well, i) + ".png",
                    "pred": float(np.clip(base + rng.normal(0, 0.1),
                                          0.02, 0.98)),
                })
    pd.DataFrame(recs).to_csv(path, index=False)
    return str(path)


def test_plate_from_order_reads_the_well_from_the_crop_name(tmp_path, stubs,
                                                            capsys):
    """The well is parts[1] of the crop FILE NAME, and schema says what it means.

    This positional read is not the one ``_split_prc`` replaced, and the
    difference is the reason it stays. A ``prc`` is a KEY: fixed number of
    components, so it can be read right to left past a plate id containing the
    separator. A crop name is ``<plate>_<well>_<field>[_<time>]_<object>``: a
    variable-length tail with no right anchor, so the well can only be found by
    counting from the left, which is exactly what the package's own file-name
    parsers (``schema.parse_field_stem``, ``schema.parse_object_stem``) do.

    What the token MEANS is still schema's decision, and that is what the
    inline ``([A-Pa-p])(\\d+)`` regex got wrong: a 1536-plate row and a
    lowercase well both failed to match and silently kept whatever rowID the
    CSV already carried.
    """
    from spacr import schema
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    # rowID / columnID in the CSV are junk, so the only route to a real well
    # is the crop name.
    score = write_scores_with_paths(sdir / "scores.csv",
                                    columns_hold_the_well=False)
    count = write_counts(cdir / "counts.csv")

    perform_regression(base_settings(score, count, plate_from_order=True))

    out = capsys.readouterr().out
    assert "did not match" not in out

    data = pd.read_csv(os.path.join(_results_dir(count, "scores"),
                                    "regression_data.csv"))
    assert len(data) > 0
    assert set(data["rowID"].unique()) == set(ROWS)
    assert set(data["columnID"].unique()) <= set(COLS)
    assert "rWRONG" not in set(data["rowID"])
    assert "cWRONG" not in set(data["columnID"])

    # The wells schema handles and the replaced regex did not.
    assert schema.parse_well("AA14", strict=True) == ("r27", "c14")
    assert schema.parse_well("a14", strict=True) == ("r1", "c14")


def test_plate_from_order_refuses_a_shifted_well_instead_of_inventing_one(
        tmp_path, stubs, capsys):
    """An underscored plate id shifts every token, and that is not silent.

    The crop-name grammar carries nothing that can undo the shift -- the same
    hole ``schema.parse_object_stem`` has -- so what matters here is that the
    shifted token is REFUSED rather than passed through into both slots. It is
    counted in a printed warning, the rows keep the rowID / columnID they came
    in with, and the run stays on the real wells.
    """
    from spacr import schema
    from spacr.ml import perform_regression

    sdir = tmp_path / "s"
    cdir = tmp_path / "c"
    sdir.mkdir()
    cdir.mkdir()
    score = write_scores_with_paths(
        sdir / "scores.csv",
        stem=lambda plate, well, i: f"exp1_{plate}_{well}_1_1_{i}")
    count = write_counts(cdir / "counts.csv")

    # parts[1] is the tail of the plate id, not a well, and parse_well refuses
    # it -- parse_well without strict= would pass 'PLATE1' through into BOTH
    # slots and key every row on it.
    assert "exp1_PLATE1_B12_1_1_0".split("_")[1] == "PLATE1"
    with pytest.raises(schema.WellParseError):
        schema.parse_well("PLATE1", strict=True)

    perform_regression(base_settings(score, count, plate_from_order=True))

    out = capsys.readouterr().out
    assert "did not match" in out
    assert "plate id contains '_'" in out

    # The incoming rowID / columnID survived, so the wells are still the real
    # ones rather than a well invented from the plate name.
    data = pd.read_csv(os.path.join(_results_dir(count, "scores"),
                                    "regression_data.csv"))
    assert len(data) > 0
    assert set(data["rowID"].unique()) == set(ROWS)
    assert set(data["columnID"].unique()) <= set(COLS)
