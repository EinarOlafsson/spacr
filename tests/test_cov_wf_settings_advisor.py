"""Settings advisor -- the four quiet paths where a measurement is absent.

Every test here drives a table (or a settings mapping) whose *interesting*
property is that one measurement legitimately cannot be made, and checks that
the advisor leaves it out instead of inventing it. That is the failure mode
this module is dangerous for: a fraction median computed from a screen with no
reads, or a gene count computed from a guide column that names no gene, would
be printed beside the honest numbers with no mark on it, and the user would
tune `fraction_threshold` against a fiction.

Each absence is paired, inside the same test, with the input that DOES produce
the measurement, so the assertion "it is not there" is anchored to a run where
it is.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from spacr.regression_qc import QC_NUMBERS_FILE
from spacr.settings_advisor import advise_the_screen

# ---------------------------------------------------------------------------
# Count tables on disk. The readers read FILES, so a DataFrame fixture would
# skip the header handling the real inputs go through.
# ---------------------------------------------------------------------------

def _write_counts(path, rows):
    """Write count rows in the spelling the real count CSVs use."""
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _library_rows(count=100, guides=("TGGT1_000001_1", "TGGT1_000001_2",
                                     "TGGT1_000002_1")):
    """A one-plate, four-well screen over the given guide names."""
    out = []
    for row in ("r1", "r2"):
        for column in ("c1", "c2"):
            for guide in guides:
                out.append({"plate": "plate1", "row_name": row,
                            "column_name": column, "grna_name": guide,
                            "count": count})
    return out


# ---------------------------------------------------------------------------
# read_the_counts: the fraction distribution
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# advise_the_screen: a run folder that holds nothing
# ---------------------------------------------------------------------------

def _write_qc(folder, numbers, *, regression_type="ols"):
    """Write a run's QC numbers where `read_the_last_run` looks for them."""
    qc = os.path.join(str(folder), "regression_qc")
    os.makedirs(qc, exist_ok=True)
    path = os.path.join(qc, QC_NUMBERS_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"regression_type": regression_type, "numbers": numbers,
                   "panels": {}, "verdicts": {}}, handle)
    return path


def _scores(path, n=60):
    """A score table with one well column and a bounded response."""
    frame = pd.DataFrame({
        "prc": [f"plate1_r{i % 2 + 1}_c{i % 2 + 1}" for i in range(n)],
        "pred": [(i % 10) / 10.0 for i in range(n)]})
    frame.to_csv(path, index=False)
    return str(path)


def test_a_run_folder_with_no_diagnostics_does_not_disturb_the_reading(
        tmp_path):
    """Pointing at a folder that never finished a fit must change nothing.

    The button is handed whatever folder the panel currently names, and that
    folder is very often a fresh output directory with no `regression_qc` in
    it yet. `read_the_last_run` then returns nothing, and the advisor has to
    leave the input-derived reading exactly as it was -- in particular
    `run_folder` must stay empty, because `Reading.read_a_run` is what tells
    the user whether the advice in front of them came from a fit or only from
    the input tables. Stamping the folder name on a reading that read no
    diagnostics would make an unfitted screen claim it had been fitted.
    """
    counts = _write_counts(tmp_path / "counts.csv", _library_rows())
    scores = _scores(tmp_path / "scores.csv")

    barren = tmp_path / "run_never_finished"
    barren.mkdir()
    finished = tmp_path / "run_with_qc"
    finished.mkdir()
    _write_qc(finished, {"durbin_watson": 2.1, "max_vif": 3.4,
                         "shapiro_p": 0.31})

    without = advise_the_screen([counts], [scores], run_folder="")
    barren_advice = advise_the_screen([counts], [scores],
                                      run_folder=str(barren))
    fitted = advise_the_screen([counts], [scores], run_folder=str(finished))

    # A finished run IS merged: the folder is named and its numbers arrive.
    assert fitted.reading.run_folder.endswith("regression_qc")
    assert fitted.reading.durbin_watson == pytest.approx(2.1)
    assert fitted.reading.max_vif == pytest.approx(3.4)
    assert fitted.reading.read_a_run is True

    # The barren folder leaves the reading identical to no folder at all.
    assert barren_advice.reading == without.reading
    assert barren_advice.reading.run_folder == ""
    assert barren_advice.reading.durbin_watson is None
    assert barren_advice.reading.read_a_run is False
    # And it is the SAME screen underneath, so the input numbers survived.
    assert barren_advice.reading.wells == 4
    assert barren_advice.as_settings() == without.as_settings()
