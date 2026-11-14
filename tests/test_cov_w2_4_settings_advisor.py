"""Settings advisor — what it does when the tables are not what it hoped.

The advisor's danger is an authoritative guess, so the paths that matter
most are the ones where a table is short, misspelled, unreadable or simply
missing a column. Each of those has to end in a sentence the user can act
on, recorded in ``trouble`` — never in a silently smaller measurement, which
is exactly the failure that once measured a four-plate screen from one plate
while the reading still said four.

Everything here is driven against CSVs written to disk, because the reading
functions read files and a fixture that hands them a DataFrame would skip
the header/usecols logic that the bug lived in.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from spacr import settings_advisor as sa
from spacr.settings_advisor import (
    Advice,
    Choice,
    Reading,
    Undecided,
    advise_that_runs,
    advise_the_screen,
)

# ---------------------------------------------------------------------------
# Tables on disk
# ---------------------------------------------------------------------------

def _counts_csv(path, plates=2, genes=6, guides_per_gene=3, rows=2,
                columns=3):
    rng = np.random.default_rng(0)
    out = []
    for plate in range(1, plates + 1):
        for row in range(1, rows + 1):
            for column in range(1, columns + 1):
                for gene in range(genes):
                    for guide in range(guides_per_gene):
                        out.append({
                            "plate": f"plate{plate}",
                            "row_name": f"r{row}",
                            "column_name": f"c{column}",
                            "grna_name": f"TGGT1_{gene:06d}_{guide + 1}",
                            "count": int(rng.integers(1, 500)),
                        })
    pd.DataFrame(out).to_csv(path, index=False)
    return str(path)


def _scores_csv(path, n=40, column="pred", well=True):
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({column: rng.beta(2, 5, n)})
    if well:
        frame.insert(0, "prc", [f"plate1_r{i % 2 + 1}_c{i % 3 + 1}"
                                for i in range(n)])
    frame.to_csv(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# _columns_of and _well_key
# ---------------------------------------------------------------------------

def test_a_header_that_cannot_be_read_is_no_columns_not_a_crash(tmp_path):
    """A header read is not the moment to raise at the user."""
    assert sa._columns_of(str(tmp_path / "never_written.csv")) == ()


def test_the_header_is_read_in_the_files_own_spelling_and_canonically(
        tmp_path):
    """``usecols`` matches the FILE's names, so both answers are needed."""
    path = tmp_path / "one.csv"
    pd.DataFrame({"plate": ["p1"], "row": ["r1"], "col": ["c1"],
                  "pred": [0.4]}).to_csv(path, index=False)

    raw = sa._columns_of(str(path))
    canonical = sa._columns_of(str(path), canonical=True)

    assert "col" in raw and "plate" in raw
    assert "columnID" in canonical and "plateID" in canonical


@pytest.mark.parametrize("frame,expected", [
    (pd.DataFrame({"prc": ["p1_r1_c1"]}), "p1_r1_c1"),
    (pd.DataFrame({"prcf": ["p1_r1_c1_f3"]}), "p1_r1_c1"),
    (pd.DataFrame({"plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"]}),
     "p1_r1_c1"),
    (pd.DataFrame({"plate": ["p1"], "row": ["r1"], "col": ["c1"]}),
     "p1_r1_c1"),
    (pd.DataFrame({"plate": ["p1"], "row_name": ["r1"],
                   "column_name": ["c1"]}), "p1_r1_c1"),
])
def test_a_well_is_recognised_in_every_spelling_a_table_uses(frame, expected):
    assert sa._well_key(frame).iloc[0] == expected


def test_a_table_that_names_no_well_has_no_well_key():
    assert sa._well_key(pd.DataFrame({"pred": [0.1, 0.2]})) is None


# ---------------------------------------------------------------------------
# read_the_counts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# read_the_response
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The Advice container
# ---------------------------------------------------------------------------

def test_a_reason_is_available_for_every_key_and_empty_for_the_rest():
    advice = Advice(
        chosen=(Choice("regression_type", "beta", "the response is a share"),),
        undecided=(Undecided("min_observations_per_hit", "replication was not measured"),),
        reading=Reading())

    assert advice.why("regression_type") == "the response is a share"
    assert advice.why("min_observations_per_hit") == "replication was not measured"
    assert advice.why("a_setting_nobody_proposed") == ""


# ---------------------------------------------------------------------------
# The arguments the advisor makes
# ---------------------------------------------------------------------------

def _reading(**kwargs):
    base = dict(plates=4, wells=200, guides=300, genes=100, rows=8,
                columns=12, response="pred", n_response=5000, low=0.0,
                high=1.0, inside_unit=True, on_unit=True, normal_p=0.5,
                skew=1.2, wells_per_guide=6.0, guides_per_gene=3.0,
                objects_per_well=25.0, fraction_median=0.01,
                fraction_q90=0.05, guides_per_well=40.0,
                kept_at_two_percent=0.8)
    base.update(kwargs)
    return Reading(**base)


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------

def _qc_folder(tmp_path, payload):
    from spacr.regression_qc import QC_NUMBERS_FILE

    folder = tmp_path / "run_1" / "regression_qc"
    folder.mkdir(parents=True)
    (folder / QC_NUMBERS_FILE).write_text(json.dumps(payload), encoding="utf-8")
    return str(tmp_path / "run_1")


def test_the_screen_reading_is_refined_by_a_finished_run(tmp_path):
    """The run's residual shape reaches the Reading the advice argues from."""
    folder = _qc_folder(tmp_path, {"numbers": {"normality_p": 0.001,
                                               "max_vif": 12.0}})
    counts = _counts_csv(tmp_path / "counts.csv")
    scores = _scores_csv(tmp_path / "scores.csv", n=60)

    advice = advise_the_screen([counts], [scores], run_folder=folder)

    assert advice.reading.residual_normal_p == 0.001
    assert advice.reading.max_vif == 12.0
    assert advice.reading.run_folder.endswith("regression_qc")


# ---------------------------------------------------------------------------
# refusals / advise_that_runs
# ---------------------------------------------------------------------------

def test_defaults_that_cannot_be_filled_fall_back_to_the_proposal(
        monkeypatch):
    """The preflight still runs on what the advisor itself proposed."""
    from spacr import settings as settings_mod

    def refuse(_settings):
        raise RuntimeError("the defaults table could not be built")

    monkeypatch.setattr(settings_mod,
                        "get_perform_regression_default_settings", refuse)

    advice = advise_that_runs(_reading(normal_p=0.001, wells=400), {})
    assert advice.as_settings()["inference"] == "nonparametric"


def test_a_choice_the_run_would_refuse_is_withdrawn_with_the_reason(
        monkeypatch):
    """A per-object panel makes the permutation choice unrunnable.

    ``analysis_unit`` is not something the advisor proposes -- it arrives
    from the panel through the defaults fill, which is why that is the seam
    this drives.
    """
    from spacr import settings as settings_mod

    real = settings_mod.get_perform_regression_default_settings

    def per_object(proposed):
        whole = dict(real(dict(proposed)))
        whole["analysis_unit"] = "cell"
        return whole

    monkeypatch.setattr(settings_mod,
                        "get_perform_regression_default_settings", per_object)

    advice = advise_that_runs(_reading(normal_p=0.001, wells=400), {})
    withdrawn = {u.key: u.why for u in advice.undecided}

    # ``agg_type`` is the choice the refusal names, so it is the one that
    # goes -- an aggregation a per-object run would never read.
    assert "agg_type" not in advice.as_settings()
    assert "withdrawn: the run would refuse it" in withdrawn["agg_type"]
    assert "analysis_unit='cell'" in withdrawn["agg_type"]
    # The choices the refusal did not name survive.
    assert advice.as_settings()["regression_type"]
    assert advice.as_settings()["inference"] == "nonparametric"
