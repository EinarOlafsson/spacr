"""Branch coverage for :func:`spacr.submodules.compare_reads_to_scores`.

The function stitches together four nested helpers:

``calculate_well_read_fraction``
    per-well read fraction for every gRNA (and the ``ValueError`` raised when
    the plate/row/column columns are missing),
``calculate_well_score_fractions``
    per-well class_0/class_1 fractions from per-object classifier calls (and
    its own missing-column ``ValueError``),
``calculate_grna_fraction_ratio``
    pc/nc read-fraction ratio, including the ``inf`` -> 0 and ``NaN`` -> 0
    sanitisation,
``plot_line``
    the seaborn line plot, both the list-of-y-columns branch and the single
    y-vector branch, with and without a ``save_path``.

Everything below runs on tiny synthetic CSVs written to ``tmp_path``: CPU-only,
offline and sub-second. Assertions are made on the *plotted data* (the line
artists carry the numbers the helpers computed), which is the only externally
visible form the intermediate frame takes.

Layout of the synthetic screen (4 rows x 2 columns, one plate):

* column ``c3`` — the pc gRNA climbs 10 -> 40 reads while the nc gRNA falls
  40 -> 10, so ``fraction_ratio`` is ``(i+1)/(4-i)`` and the pc read fraction
  is ``(i+1)/5``; the classifier calls ``i+1`` of 6 objects class 1.
* column ``c4`` — a flat 10/10 control split and a flat 1-of-2 class call, so
  the ``column``/``value`` selector is observable in the output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

PC = "TGGT1_220950_1"
NC = "TGGT1_233460_4"
FILLER = "TGGT1_111111_1"
ROWS = ("r1", "r2", "r3", "r4")

# rowID -> (pc units, nc units); pc_fraction = .9/.8/.7/.6, nc_fraction = .1/.2/.3/.4
EMPIRICAL = {"r1": (90, 10), "r2": (80, 20), "r3": (70, 30), "r4": (60, 40)}

# x axis after the natsort on 'pc_fraction' (ascending) -> r4, r3, r2, r1
X_PC = [0.6, 0.7, 0.8, 0.9]
X_NC = [0.1, 0.2, 0.3, 0.4]


@pytest.fixture(autouse=True)
def _close_figs():
    """Never let a figure leak out of a test in this module."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


# ---------------------------------------------------------------------------
# synthetic input builders
# ---------------------------------------------------------------------------

def _reads_frame(plate="plate1", with_plate=True):
    """Per-gRNA read counts, 4 rows x 2 columns."""
    recs = []
    for i, row in enumerate(ROWS):
        per_col = {
            "c3": {PC: 10 * (i + 1), NC: 10 * (4 - i), FILLER: 5},
            "c4": {PC: 10, NC: 10, FILLER: 5},
        }
        for col, counts in per_col.items():
            for grna, cnt in counts.items():
                rec = {"rowID": row, "columnID": col, "grna_name": grna, "count": cnt}
                if with_plate:
                    rec["plateID"] = plate
                recs.append(rec)
    return pd.DataFrame(recs)


def _scores_frame(plate="plate1", with_plate=True, n_class1=None):
    """Per-object classifier calls, 4 rows x 2 columns.

    ``n_class1`` overrides the c3 class-1 count per row index (used to give a
    second plate a different mixture).
    """
    recs = []
    for i, row in enumerate(ROWS):
        n1_c3 = (i + 1) if n_class1 is None else n_class1[i]
        n0_c3 = 6 - n1_c3
        per_col = {"c3": (n1_c3, n0_c3), "c4": (1, 1)}
        for col, (n1, n0) in per_col.items():
            for label, n in ((1, n1), (0, n0)):
                for _ in range(n):
                    rec = {"rowID": row, "columnID": col, "cv_predictions": label}
                    if with_plate:
                        rec["plateID"] = plate
                    recs.append(rec)
    return pd.DataFrame(recs)


def _write_pair(tmp_path, reads=None, scores=None, stem=""):
    """Write a reads/scores CSV pair and return the two paths."""
    reads = _reads_frame() if reads is None else reads
    scores = _scores_frame() if scores is None else scores
    rp = tmp_path / f"reads{stem}.csv"
    sp = tmp_path / f"scores{stem}.csv"
    reads.to_csv(rp, index=False)
    scores.to_csv(sp, index=False)
    return str(rp), str(sp)


def _lines(fig):
    """{label: (xdata, ydata)} for every line artist on the figure."""
    ax = fig.axes[0]
    return {ln.get_label(): (ln.get_xdata(), ln.get_ydata()) for ln in ax.lines}


# ---------------------------------------------------------------------------
# the happy path: one reads CSV + one scores CSV
# ---------------------------------------------------------------------------

def test_single_pair_computes_fractions_and_saves_both_pdfs(tmp_path, capsys):
    """Full pipeline: read fractions, score fractions, pc/nc ratio, 2 PDFs."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)
    pc_pdf = tmp_path / "pc.pdf"
    nc_pdf = tmp_path / "nc.pdf"

    figs = compare_reads_to_scores(
        rp, sp, empirical_dict=EMPIRICAL, save_paths=[str(pc_pdf), str(nc_pdf)]
    )

    assert isinstance(figs, list) and len(figs) == 2
    fig_pc, fig_nc = figs

    # --- first figure is plotted against the empirical pc fraction ---------
    ax = fig_pc.axes[0]
    assert ax.get_xlabel() == "pc_fraction"
    assert ax.get_ylabel() == "Fraction"
    assert ax.get_title() == "Line Plot"
    assert ax.get_legend().get_title().get_text() == "Legend"
    # top/right spines removed by sns.despine
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()

    lines = _lines(fig_pc)
    assert set(lines) == {"class_1_fraction", f"{PC}_fraction", "nc_fraction"}

    x, y = lines["class_1_fraction"]
    assert np.allclose(x, X_PC)                       # natsorted ascending
    assert np.allclose(y, [4 / 6, 3 / 6, 2 / 6, 1 / 6])

    x, y = lines[f"{PC}_fraction"]
    assert np.allclose(y, [0.8, 0.6, 0.4, 0.2])       # 10(i+1) / 50 reads

    x, y = lines["nc_fraction"]
    assert np.allclose(y, [0.4, 0.3, 0.2, 0.1])       # straight from EMPIRICAL

    # --- second figure: same y data, x is the empirical nc fraction -------
    ax2 = fig_nc.axes[0]
    assert ax2.get_xlabel() == "nc_fraction"
    x2, y2 = _lines(fig_nc)["class_1_fraction"]
    assert np.allclose(x2, X_NC)
    assert np.allclose(y2, [1 / 6, 2 / 6, 3 / 6, 4 / 6])

    # --- both PDFs really hit the disk ------------------------------------
    assert pc_pdf.is_file() and pc_pdf.stat().st_size > 0
    assert nc_pdf.is_file() and nc_pdf.stat().st_size > 0
    with open(pc_pdf, "rb") as fh:
        assert fh.read(4) == b"%PDF"

    out = capsys.readouterr().out
    assert f"Plot saved to {pc_pdf}" in out
    assert f"Plot saved to {nc_pdf}" in out


def test_column_value_selects_the_other_well_column(tmp_path):
    """``column``/``value`` pick the wells; c4 carries a flat 10/10 split."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)

    figs = compare_reads_to_scores(
        rp, sp, empirical_dict=EMPIRICAL, column="columnID", value="c4",
        save_paths=[None, None],
    )

    lines = _lines(figs[0])
    x, y = lines[f"{PC}_fraction"]
    assert np.allclose(x, X_PC)
    assert np.allclose(y, 0.5)                 # 10 pc reads of 20 control reads
    assert np.allclose(lines["class_1_fraction"][1], 0.5)   # 1 of 2 objects
    # ... and that is genuinely different from the c3 wells
    c3 = _lines(compare_reads_to_scores(
        rp, sp, empirical_dict=EMPIRICAL, value="c3", save_paths=[None, None])[0])
    assert not np.allclose(c3["class_1_fraction"][1], 0.5)


def test_plate_argument_stamps_plate_id_when_csvs_lack_one(tmp_path):
    """``plate=`` supplies the plateID the CSVs are missing; no PDF written."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(
        tmp_path,
        reads=_reads_frame(with_plate=False),
        scores=_scores_frame(with_plate=False),
        stem="_noplate",
    )

    # Without `plate` the frames have no plateID at all -> the helper refuses.
    with pytest.raises(ValueError, match="Cannot find plate, row or column"):
        compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                save_paths=[None, None])

    figs = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL, plate="p9",
                                   save_paths=[None, None])

    assert len(figs) == 2
    assert np.allclose(_lines(figs[0])["class_1_fraction"][1],
                       [4 / 6, 3 / 6, 2 / 6, 1 / 6])
    # save_path was None for both -> nothing saved next to the inputs
    assert list(tmp_path.glob("*.pdf")) == []


def test_default_empirical_dict_covers_sixteen_rows(tmp_path):
    """``empirical_dict=None`` falls back to the built-in 16-row mixture."""
    from spacr.submodules import compare_reads_to_scores

    # r13 -> (30, 70) in the built-in dict -> pc_fraction 0.3, nc_fraction 0.7
    reads = _reads_frame()
    reads["rowID"] = reads["rowID"].replace({"r1": "r13"})
    scores = _scores_frame()
    scores["rowID"] = scores["rowID"].replace({"r1": "r13"})
    rp, sp = _write_pair(tmp_path, reads=reads, scores=scores, stem="_default")

    figs = compare_reads_to_scores(rp, sp, save_paths=[None, None])

    x, y = _lines(figs[0])["nc_fraction"]
    # rows r2/r3/r4 keep (90,10)/(80,20)/(80,20); r13 contributes 0.3 / 0.7,
    # so r3 and r4 share x=0.8 and seaborn collapses them to one point
    assert np.allclose(sorted(np.round(x, 3)), [0.3, 0.8, 0.9])
    assert np.isclose(y[0], 0.7)            # r13 -> nc fraction 0.7
    assert np.isclose(y[-1], 0.1)           # r2  -> nc fraction 0.1


# ---------------------------------------------------------------------------
# list-of-files branch
# ---------------------------------------------------------------------------

def test_list_inputs_concat_plates_and_rename_legacy_columns(tmp_path, capsys):
    """Two plates, legacy ``column`` / ``column_name`` / ``row_name`` headers."""
    from spacr.submodules import compare_reads_to_scores

    # plate 1 uses the old 'column' header, plate 2 the old 'column_name' one
    r1 = _reads_frame(with_plate=False).rename(columns={"columnID": "column"})
    r2 = _reads_frame(with_plate=False).rename(columns={"columnID": "column_name"})
    # both score tables use the old 'row_name' header
    s1 = _scores_frame(with_plate=False).rename(columns={"rowID": "row_name"})
    s2 = _scores_frame(with_plate=False, n_class1=[5, 5, 5, 5]).rename(
        columns={"rowID": "row_name"})

    paths = []
    for name, frame in (("r1", r1), ("r2", r2), ("s1", s1), ("s2", s2)):
        p = tmp_path / f"{name}.csv"
        frame.to_csv(p, index=False)
        paths.append(str(p))
    rp1, rp2, sp1, sp2 = paths

    figs = compare_reads_to_scores([rp1, rp2], [sp1, sp2],
                                   empirical_dict=EMPIRICAL,
                                   save_paths=[None, None])

    out = capsys.readouterr().out
    assert "Reads: 48 Scores: 64" in out          # 2 x 24 reads, 2 x 32 objects

    x, y = _lines(figs[0])["class_1_fraction"]
    # seaborn averages the two plates that share an x value
    assert np.allclose(sorted(set(np.round(x, 3))), X_PC)
    # plate2 is a flat 5-of-6, plate1 is (i+1)-of-6 -> mean at r4 (x=0.6) is 4.5/6
    assert np.isclose(y[0], (4 / 6 + 5 / 6) / 2)
    assert np.isclose(y[-1], (1 / 6 + 5 / 6) / 2)


def test_mismatched_list_lengths_should_raise_valueerror(tmp_path):
    """A 2-vs-1 list mismatch should be a clean error, not UnboundLocalError."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)
    with pytest.raises(ValueError):
        compare_reads_to_scores([rp, rp], [sp], empirical_dict=EMPIRICAL,
                                save_paths=[None, None])


def test_legacy_row_header_should_be_renamed_to_rowid(tmp_path):
    """``if 'row' in columns: rename({'row_name': 'rowID'})`` renames nothing."""
    from spacr.submodules import compare_reads_to_scores

    reads = _reads_frame(with_plate=False).rename(columns={"rowID": "row"})
    scores = _scores_frame(with_plate=False)
    rp, sp = _write_pair(tmp_path, reads=reads, scores=scores, stem="_legacyrow")

    figs = compare_reads_to_scores([rp], [sp], empirical_dict=EMPIRICAL,
                                   save_paths=[None, None])
    assert len(figs) == 2


def test_row_and_row_name_headers_together_are_renamed(tmp_path):
    """With both legacy headers present the 'row' check does fire usefully."""
    from spacr.submodules import compare_reads_to_scores

    reads = _reads_frame(with_plate=False).rename(columns={"rowID": "row_name"})
    reads["row"] = reads["row_name"]
    scores = _scores_frame(with_plate=False)
    rp, sp = _write_pair(tmp_path, reads=reads, scores=scores, stem="_bothrow")

    figs = compare_reads_to_scores([rp], [sp], empirical_dict=EMPIRICAL,
                                   save_paths=[None, None])

    assert np.allclose(_lines(figs[0])[f"{PC}_fraction"][1], [0.8, 0.6, 0.4, 0.2])


# ---------------------------------------------------------------------------
# missing-column guards inside the nested helpers
# ---------------------------------------------------------------------------

def test_reads_without_row_column_raise_value_error(tmp_path):
    """calculate_well_read_fraction needs plateID/rowID/columnID."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path, reads=_reads_frame().drop(columns=["rowID"]),
                         stem="_norow")
    with pytest.raises(ValueError, match="Cannot find plate, row or column"):
        compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                save_paths=[None, None])


def test_scores_without_row_column_raise_value_error(tmp_path):
    """calculate_well_score_fractions has its own (differently worded) guard."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path, scores=_scores_frame().drop(columns=["columnID"]),
                         stem="_nocol")
    with pytest.raises(ValueError,
                       match="Cannot find 'plateID', 'rowID', or 'columnID'"):
        compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                save_paths=[None, None])


# ---------------------------------------------------------------------------
# fraction-ratio sanitisation
# ---------------------------------------------------------------------------

def test_missing_nc_reads_turn_infinite_ratio_into_zero(tmp_path):
    """A well with no nc reads divides by zero -> inf -> replaced with 0."""
    from spacr.submodules import compare_reads_to_scores

    reads = _reads_frame()
    reads = reads[~((reads["rowID"] == "r4") & (reads["grna_name"] == NC))]
    rp, sp = _write_pair(tmp_path, reads=reads, stem="_nonc")

    figs = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                   y_columns=["fraction_ratio"],
                                   save_paths=[None, None])

    x, y = _lines(figs[0])["fraction_ratio"]
    assert np.allclose(x, X_PC)
    # r4 (x=0.6) would be 40/0 = inf; the others are (i+1)/(4-i)
    assert y[0] == 0.0
    assert np.allclose(y[1:], [3 / 2, 2 / 3, 1 / 4])
    # the pc read fraction for that well is 40 of 40 control reads
    figs2 = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                    save_paths=[None, None])
    assert np.isclose(_lines(figs2[0])[f"{PC}_fraction"][1][0], 1.0)


def test_zero_control_reads_fill_nan_ratio_with_zero(tmp_path):
    """0/0 in a well gives NaN, which is filled with 0 by ``fillna``."""
    from spacr.submodules import compare_reads_to_scores

    reads = _reads_frame()
    zeroed = (reads["rowID"] == "r4") & (reads["grna_name"].isin([PC, NC]))
    reads.loc[zeroed, "count"] = 0
    rp, sp = _write_pair(tmp_path, reads=reads, stem="_zero")

    figs = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                   y_columns=["fraction_ratio"],
                                   save_paths=[None, None])

    x, y = _lines(figs[0])["fraction_ratio"]
    assert np.allclose(x, X_PC)
    assert y[0] == 0.0
    assert not np.isnan(y).any()

    # the pc fraction of that well is a genuine 0/0 -> NaN, dropped by seaborn
    figs2 = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                    save_paths=[None, None])
    x2, y2 = _lines(figs2[0])[f"{PC}_fraction"]
    assert np.allclose(x2, X_PC[1:])
    assert np.allclose(y2, [0.6, 0.4, 0.2])


# ---------------------------------------------------------------------------
# plot_line: single y-vector branch (y_columns is not a list)
# ---------------------------------------------------------------------------

def test_non_list_y_columns_plot_one_unlabelled_line(tmp_path):
    """A y *vector* takes the ``else`` branch: one line, no legend."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)
    vector = pd.Series([0.1, 0.2, 0.3, 0.4], name="my_vector")

    figs = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                   y_columns=vector, save_paths=[None, None])

    ax = figs[0].axes[0]
    assert len(ax.lines) == 1
    x, y = ax.lines[0].get_xdata(), ax.lines[0].get_ydata()
    assert np.allclose(x, X_PC)
    # the frame is re-ordered by pc_fraction; the vector follows by index
    assert np.allclose(y, [0.4, 0.3, 0.2, 0.1])
    assert ax.get_legend() is None          # group_column is None -> no legend
    assert ax.get_ylabel() == "Fraction"


# ---------------------------------------------------------------------------
# the missing-y-column guard
# ---------------------------------------------------------------------------

def test_guard_prints_columns_and_returns_none(tmp_path, capsys):
    """The missing-y-column guard bails out and lists what is available.

    ``y_columns=(any,)`` names a "column" no frame can hold, so the guard
    prints the real column names and returns ``None``. This used to be the
    *only* input that reached the bail-out, back when the guard was the
    chained comparison ``any in y_columns not in df.columns``.
    """
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)

    result = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                     y_columns=(any,), save_paths=[None, None])

    assert result is None
    out = capsys.readouterr().out
    assert "columns in dataframe:" in out
    for col in ("pc_fraction", "nc_fraction", "class_1_fraction",
                f"{PC}_fraction", "fraction_ratio"):
        assert f"\n{col}\n" in out


def test_unknown_y_column_should_bail_out(tmp_path):
    """An unknown y column should print the columns and return None."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)
    result = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                     y_columns=["not_a_real_column"],
                                     save_paths=[None, None])
    assert result is None


# ---------------------------------------------------------------------------
# defaults that do not survive contact with the code
# ---------------------------------------------------------------------------

def test_default_save_paths_should_not_crash(tmp_path):
    """Calling with the declared defaults should just skip saving."""
    from spacr.submodules import compare_reads_to_scores

    rp, sp = _write_pair(tmp_path)
    figs = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL)
    assert len(figs) == 2


def test_single_class_scores_should_not_crash(tmp_path):
    """An all-positive classifier output should give class_1_fraction == 1."""
    from spacr.submodules import compare_reads_to_scores

    scores = _scores_frame()
    scores["cv_predictions"] = 1
    rp, sp = _write_pair(tmp_path, scores=scores, stem="_oneclass")

    figs = compare_reads_to_scores(rp, sp, empirical_dict=EMPIRICAL,
                                   save_paths=[None, None])
    assert np.allclose(_lines(figs[0])["class_1_fraction"][1], 1.0)
