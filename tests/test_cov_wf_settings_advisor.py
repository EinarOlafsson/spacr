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
from spacr.settings_advisor import (advise_the_screen, read_the_counts,
                                    read_the_last_run, refusals)


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

def test_a_screen_with_no_reads_reports_no_fraction_instead_of_a_fake_one(
        tmp_path):
    """A well with zero reads has no guide share, and none may be printed.

    `fractions_from_counts` divides each guide's count by the well total, so a
    plate that sequenced to nothing gives 0/0 -- NaN for every row. The
    fraction numbers this reading carries are the ones the user sets
    `fraction_threshold` from: `kept_at_two_percent` is shown as "this default
    keeps N% of your library". Computed from an all-NaN column it would come
    out as 0.0 or nan and read as a measured catastrophe, when the truth is
    that the counts say nothing at all. The reading must simply not offer the
    number, while still reporting the plates, wells and guides it really did
    count.
    """
    empty = _write_counts(tmp_path / "no_reads.csv", _library_rows(count=0))
    sequenced = _write_counts(tmp_path / "reads.csv", _library_rows(count=100))

    silent = read_the_counts([empty])
    spoken = read_the_counts([sequenced])

    # The same design was measured from both files.
    assert silent["plates"] == spoken["plates"] == 1
    assert silent["wells"] == spoken["wells"] == 4
    assert silent["guides"] == spoken["guides"] == 3
    assert silent["trouble"] == []

    # The file with reads produces every fraction number ...
    assert spoken["fraction_median"] == pytest.approx(1 / 3)
    assert spoken["fraction_q90"] == pytest.approx(1 / 3)
    assert spoken["guides_per_well"] == 3.0
    assert spoken["kept_at_two_percent"] == 1.0

    # ... and the file without reads produces none of them.
    for key in ("fraction_median", "fraction_q90", "guides_per_well",
                "kept_at_two_percent"):
        assert key not in silent, f"{key} was invented from zero reads"


def test_a_guide_column_that_is_only_the_organism_names_no_genes(tmp_path):
    """`TGGT1_` on every row is an organism, not a library of one gene.

    This is the 145 failure in miniature: truncating a guide name at the first
    underscore turns the whole screen into a single "gene". `gene_of_guide`
    strips the measured organism prefix and finds nothing left, so no guide
    names a gene. If the reading answered `genes=1` here, the advisor would go
    on to recommend a gene-level aggregation and a multiple-testing threshold
    for one test, on a screen where the guide identifiers are simply broken --
    and the user would never be told the names were unusable.
    """
    nameless = _write_counts(tmp_path / "prefix_only.csv",
                             _library_rows(guides=("TGGT1_",)))
    named = _write_counts(tmp_path / "named.csv", _library_rows())

    blank = read_the_counts([nameless])
    real = read_the_counts([named])

    # A real library names its genes and their guides.
    assert real["genes"] == 2
    assert real["guides"] == 3
    assert real["guides_per_gene"] == 1.5

    # The prefix-only column still measures the design it can see ...
    assert blank["wells"] == 4
    assert blank["guides"] == 1
    assert blank["wells_per_guide"] == 4.0
    assert blank["fraction_median"] == pytest.approx(1.0)
    # ... but claims no genes at all.
    assert "genes" not in blank
    assert "guides_per_gene" not in blank


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


def test_a_directory_where_the_numbers_file_belongs_is_read_as_no_run(
        tmp_path):
    """Only a real file counts as a finished run, at either of the two places.

    `read_the_last_run` accepts the numbers file directly inside the run
    folder or inside its `regression_qc` subfolder, and takes the first that
    `os.path.isfile` accepts. A DIRECTORY carrying that name -- which is what
    a wrongly-nested output tree leaves behind -- is not a file, so the run is
    read as absent instead of being opened and raising out of a button press.
    """
    counts = _write_counts(tmp_path / "counts.csv", _library_rows())

    # The numbers file written straight into the run folder IS found.
    plain = tmp_path / "run_flat"
    plain.mkdir()
    with open(os.path.join(str(plain), QC_NUMBERS_FILE), "w",
              encoding="utf-8") as handle:
        json.dump({"regression_type": "ols",
                   "numbers": {"max_cooks_distance": 0.8}}, handle)
    found = read_the_last_run(str(plain))
    assert found["run_folder"] == str(plain)
    assert found["max_cooks_distance"] == pytest.approx(0.8)

    # A directory of that name, in the other candidate place, is not.
    folder = tmp_path / "run_nested"
    os.makedirs(os.path.join(str(folder), "regression_qc", QC_NUMBERS_FILE))
    assert read_the_last_run(str(folder)) == {}

    advice = advise_the_screen([counts], run_folder=str(folder))
    assert advice.reading.run_folder == ""
    assert advice.reading.wells == 4
    assert advice.reading.guides == 3


# ---------------------------------------------------------------------------
# refusals: control_center
# ---------------------------------------------------------------------------

_CONTROL_CENTER_REFUSAL = (
    "batch_correction='control_center' requires batch_control_column and at "
    "least one batch_control_value.")


def test_control_center_is_refused_only_when_it_has_nothing_to_centre_on():
    """The preflight must not fire on a correctly configured centring.

    `refusals` exists so the panel can explain a refusal BEFORE the run spends
    thirty seconds getting there. That value is destroyed in both directions:
    a missing message lets the run fail late, and a message on a valid mapping
    trains the user to ignore the panel. `control_center` needs a column to
    identify the control wells; once that column is named the correction is
    runnable, and this preflight has to fall silent and let the remaining
    checks speak for themselves.
    """
    missing = refusals({"batch_correction": "control_center",
                        "regression_type": "ols"})
    named = refusals({"batch_correction": "control_center",
                      "batch_control_column": "condition",
                      "batch_control_value": "untreated",
                      "regression_type": "ols"})

    # Without the column the user is told exactly what to supply.
    assert _CONTROL_CENTER_REFUSAL in missing

    # With it, nothing about control_center is said -- and the later checks
    # still ran, which the next mapping proves by tripping one of them.
    assert named == ()
    still_checked = refusals({"batch_correction": "control_center",
                              "batch_control_column": "condition",
                              "analysis_unit": "cell",
                              "agg_type": "mean"})
    assert not any("control_center" in message for message in still_checked)
    assert any("agg_type='mean' is never read" in message
               for message in still_checked)

    # And the check is scoped to control_center: a mapping that names a
    # different correction never reaches it, however empty its control
    # column is, and is judged only on its own faults.
    other = refusals({"batch_correction": "combat",
                      "batch_control_column": ""})
    assert not any("control_center" in message for message in other)
    assert any("no batch_covariate_column is set" in message
               for message in other)


def test_control_center_refusal_is_case_insensitive_about_the_setting():
    """A panel that writes `Control_Center` must get the same answer.

    The setting reaches this function from a combo box, a settings CSV and a
    hand-edited dictionary, and only one of those guarantees lower case. If
    the check were case-sensitive the refusal would silently disappear for a
    capitalised spelling and the run would fail late instead.
    """
    shouted = refusals({"batch_correction": "CONTROL_CENTER"})
    quiet = refusals({"batch_correction": "control_center",
                      "batch_control_column": "well_type"})

    assert shouted == (_CONTROL_CENTER_REFUSAL,)
    assert quiet == ()
