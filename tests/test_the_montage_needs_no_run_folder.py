"""The montage builds its guide fractions from the counts, not from a folder.

REPORTED, twice, with rising exasperation: "i just loaded all necessary
documents into my current spacr run and still nothing ... still getting the
LOADED coefficients table was not read from a run folder, whatever the fuck
that means."

The requirement was never real. A guide's fraction in a well is

    fraction = count / (sum of count over that well)

which the input table's COUNT CSVs carry outright. `regression_data.csv` is
that join PERSISTED, not the source of it -- so a user who has loaded scores
and counts, run a regression, and is looking at its coefficients has
everything the montage needs and was being turned away.
"""

import os
import tempfile

import pandas as pd
import pytest

import spacr

assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__


def _counts(path):
    pd.DataFrame({
        "plateID": ["plate1"] * 4,
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c1", "c1", "c1"],
        "grna": ["g1", "g2", "g1", "g3"],
        "gene": ["GENEA", "GENEB", "GENEA", "GENEC"],
        "count": [30, 70, 50, 50],
    }).to_csv(path, index=False)
    return str(path)


def test_fractions_come_out_of_the_counts(tmp_path):
    from spacr.cell_montage import fractions_from_counts

    frame = fractions_from_counts([_counts(tmp_path / "unique_combinations.csv")])
    got = dict(zip(zip(frame["prc"], frame["grna"]), frame["fraction"]))
    assert got[("plate1_r1_c1", "g1")] == pytest.approx(0.3)
    assert got[("plate1_r1_c1", "g2")] == pytest.approx(0.7)
    # Every well's guides account for the whole well, or the fraction is not
    # a fraction and every montage count derived from it is wrong.
    for _, share in frame.groupby("prc")["fraction"].sum().items():
        assert share == pytest.approx(1.0)


def test_the_gene_column_survives(tmp_path):
    """A gene-level montage needs it, and the counts carry it."""
    from spacr.cell_montage import fractions_from_counts

    frame = fractions_from_counts([_counts(tmp_path / "c.csv")])
    assert "gene" in frame.columns
    assert set(frame["gene"]) == {"GENEA", "GENEB", "GENEC"}


def test_no_counts_and_no_folder_says_so(tmp_path):
    """The ONE case where there is genuinely nothing to compute from."""
    from spacr.cell_montage import MontageError, fractions_from_counts

    with pytest.raises(MontageError) as caught:
        fractions_from_counts([])
    message = str(caught.value)
    assert "count CSV" in message, message
    # It must not send the user looking for a run folder, which is what the
    # old message did and what this whole change exists to stop.
    assert "run folder" not in message.lower(), message


def test_a_counts_file_short_of_a_column_is_named(tmp_path):
    from spacr.cell_montage import MontageError, fractions_from_counts

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"grna": ["g1"], "plateID": ["plate1"]}).to_csv(bad, index=False)
    with pytest.raises(MontageError) as caught:
        fractions_from_counts([str(bad)])
    assert "count" in str(caught.value)


def test_the_panel_reads_the_counts_off_the_input_table(qtbot):
    """The counts were always one field away from the databases.

    The input table's rows are {"plate", "score", "count", "database"} and the
    montage already asked that provider for `database`. It now asks for
    `count` too, which is the whole of the plumbing this needed.
    """
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    rows = [{"plate": "plate1", "score": "/s/scores.csv",
             "count": "/s/counts.csv", "database": "/s/measurements.db"},
            {"plate": "plate2", "score": "/s/scores2.csv",
             "count": "/s/counts2.csv", "database": ""}]
    view = CellMontageView(database_provider=lambda: rows, threaded=False)
    qtbot.addWidget(view)
    assert view.count_csvs() == ("/s/counts.csv", "/s/counts2.csv")


def test_no_run_folder_is_not_a_reason_when_counts_are_attached(qtbot, tmp_path):
    """THE REPORTED BUG. A run with counts attached must not be turned away.

    `reason()` used to return RESULTS_WITHOUT_A_FOLDER the moment the results
    provider yielded nothing, whatever else was loaded.
    """
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    path = _counts(tmp_path / "counts.csv")
    rows = [{"plate": "plate1", "score": "", "count": path,
             "database": "/s/measurements.db"}]
    view = CellMontageView(database_provider=lambda: rows,
                           results_provider=lambda: "", threaded=False)
    qtbot.addWidget(view)
    assert view.count_csvs() == (path,)
    # Whatever else stops it, it is no longer the missing folder.
    assert view.RESULTS_WITHOUT_A_FOLDER not in view.reason()
