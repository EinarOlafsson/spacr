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
    Advice, Choice, Reading, Undecided, advise, advise_that_runs,
    advise_the_screen, read_the_counts, read_the_last_run, read_the_response,
    refusals,
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

def test_no_count_table_is_reported_rather_than_measured_as_zero():
    got = read_the_counts([])
    assert got["trouble"] == ["no count table is attached, so the design "
                              "could not be measured"]
    assert "wells" not in got


def test_a_count_table_that_will_not_parse_says_so(tmp_path):
    junk = tmp_path / "notes.csv"
    junk.write_text("this file is a note about the screen\n")

    got = read_the_counts([str(junk)])

    assert got["trouble"] and "could not be read" in got["trouble"][0]
    assert "wells" not in got


def test_an_empty_count_table_says_so(tmp_path, monkeypatch):
    """A header with no rows measures nothing, and must say which."""
    from spacr import cell_montage

    monkeypatch.setattr(cell_montage, "fractions_from_counts",
                        lambda paths: pd.DataFrame(
                            columns=["prc", "grna", "fraction"]))
    path = tmp_path / "counts.csv"
    path.write_text("plate,row_name,column_name,grna_name,count\n")

    got = read_the_counts([str(path)])
    assert got["trouble"] == ["the count tables are empty"]


def test_a_fraction_column_that_goes_missing_does_not_lose_the_rest(
        tmp_path, monkeypatch):
    """Plates, wells and guides are still measured without the fractions."""
    from spacr import cell_montage

    monkeypatch.setattr(cell_montage, "fractions_from_counts",
                        lambda paths: pd.DataFrame({
                            "prc": ["p1_r1_c1", "p1_r1_c2"],
                            "grna": ["TGGT1_000001_1", "TGGT1_000001_2"]}))
    path = tmp_path / "counts.csv"
    path.write_text("plate\np1\n")

    got = read_the_counts([str(path)])

    assert got["wells"] == 2
    assert got["guides"] == 2
    assert "fraction_median" not in got


def test_a_real_count_table_is_measured_end_to_end(tmp_path):
    got = read_the_counts([_counts_csv(tmp_path / "counts.csv")])

    assert got["plates"] == 2
    assert got["wells"] == 2 * 2 * 3
    assert got["guides"] == 6 * 3
    assert got["genes"] == 6
    assert got["rows"] == 2 and got["columns"] == 3
    assert got["fraction_median"] > 0
    assert 0 <= got["kept_at_two_percent"] <= 1


# ---------------------------------------------------------------------------
# read_the_response
# ---------------------------------------------------------------------------

def test_no_score_table_is_reported_rather_than_measured():
    got = read_the_response([])
    assert got["trouble"] == ["no score table is attached, so the response "
                              "could not be measured"]


def test_a_dependent_variable_that_is_not_a_column_names_the_file(tmp_path):
    path = _scores_csv(tmp_path / "scores.csv")
    got = read_the_response([path], "recruitment_ratio")
    assert "recruitment_ratio" in got["trouble"][0]
    assert "scores.csv" in got["trouble"][0]


def test_a_table_with_no_recognisable_response_lists_the_names_it_tried(
        tmp_path):
    path = tmp_path / "areas.csv"
    pd.DataFrame({"prc": ["p1_r1_c1"], "area": [12.0]}).to_csv(path,
                                                               index=False)

    got = read_the_response([str(path)])

    assert "no dependent variable is named" in got["trouble"][0]
    assert "pred" in got["trouble"][0]
    assert "response" not in got


def test_a_plate_missing_the_response_column_is_named_and_the_rest_read(
        tmp_path):
    """The bug this guards: three plates dropped silently, reading said four."""
    good = _scores_csv(tmp_path / "plate1.csv", n=20)
    other = tmp_path / "plate2.csv"
    pd.DataFrame({"prc": ["p2_r1_c1"], "area": [3.0]}).to_csv(other,
                                                              index=False)

    got = read_the_response([good, str(other)])

    assert got["score_files_read"] == 1
    assert any("plate2.csv" in note and "'pred'" in note
               for note in got["trouble"])
    assert got["n_response"] == 20


def test_a_row_cap_reached_on_the_first_file_stops_and_says_capped(tmp_path):
    first = _scores_csv(tmp_path / "plate1.csv", n=20)
    second = _scores_csv(tmp_path / "plate2.csv", n=20)

    got = read_the_response([first, second], row_cap=5)

    assert got["capped"] is True
    assert got["score_files_read"] == 1
    assert got["n_response"] == 5


def test_a_file_that_raises_while_being_read_is_named_and_skipped(tmp_path):
    """A good header over a body that will not tokenise -- an unterminated
    quote is what a half-written export looks like."""
    broken = tmp_path / "half_written.csv"
    broken.write_text('prc,pred\np1_r1_c1,0.5\n"p1_r1_c2,0.6\n')

    assert "pred" in sa._columns_of(str(broken))
    got = read_the_response([str(broken)])

    assert got["score_files_read"] == 0
    assert any("half_written.csv" in note for note in got["trouble"])
    assert "n_response" not in got


def test_a_response_column_holding_no_numbers_says_so(tmp_path):
    path = tmp_path / "words.csv"
    pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                  "pred": ["high", "low"]}).to_csv(path, index=False)

    got = read_the_response([str(path)])

    assert got["response"] == "pred"
    assert "holds no number" in got["trouble"][0]
    assert "n_response" not in got


def test_a_score_table_with_no_well_is_read_at_the_object_level_out_loud(
        tmp_path):
    """The fit sees wells; reading objects instead has to be declared."""
    path = _scores_csv(tmp_path / "objects.csv", n=40, well=False)

    got = read_the_response([path])

    assert got["n_response"] == 40
    assert "objects_per_well" not in got
    assert any("names no well" in note for note in got["trouble"])


def test_the_response_is_summarised_at_the_well_level(tmp_path):
    got = read_the_response([_scores_csv(tmp_path / "s.csv", n=60)])

    assert got["response"] == "pred"
    assert got["n_response"] == 60
    assert 0.0 <= got["low"] <= got["high"] <= 1.0
    assert got["on_unit"] is True
    assert got["binary"] is False
    assert got["objects_per_well"] == 10


def test_a_missing_shape_test_does_not_lose_the_range(tmp_path,
                                                      monkeypatch):
    """scipy is optional here; without it the range is still measured."""
    import scipy.stats

    def refuse(*args, **kwargs):
        raise RuntimeError("scipy.stats is unavailable in this build")

    monkeypatch.setattr(scipy.stats, "skew", refuse)

    got = read_the_response([_scores_csv(tmp_path / "s.csv", n=60)])

    assert "skew" not in got
    assert got["low"] is not None and got["high"] is not None


# ---------------------------------------------------------------------------
# The Advice container
# ---------------------------------------------------------------------------

def test_a_reason_is_available_for_every_key_and_empty_for_the_rest():
    advice = Advice(
        chosen=(Choice("regression_type", "beta", "the response is a share"),),
        undecided=(Undecided("min_n", "replication was not measured"),),
        reading=Reading())

    assert advice.why("regression_type") == "the response is a share"
    assert advice.why("min_n") == "replication was not measured"
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


def test_a_non_normal_response_on_too_few_wells_stays_parametric():
    """A permutation null needs wells to permute; 40 has no resolution."""
    advice = advise(_reading(normal_p=0.001, wells=40), {})

    assert advice.as_settings()["inference"] == "parametric"
    assert "too few to build a permutation null" in advice.why("inference")


def test_a_non_normal_response_on_many_wells_goes_nonparametric():
    advice = advise(_reading(normal_p=0.001, wells=400), {})
    assert advice.as_settings()["inference"] == "nonparametric"


def test_a_directional_hit_list_still_draws_the_line_on_the_adjusted_p():
    advice = advise(_reading(binary=False), {"direction": "up"})

    assert advice.as_settings()["p_threshold_kind"] == "adjusted"
    why = advice.why("p_threshold_kind")
    assert "an increase" in why
    assert "coefficient's sign" in why


def test_an_uncrowded_well_keeps_the_usual_fraction_threshold():
    advice = advise(_reading(kept_at_two_percent=0.95, fraction_median=0.2),
                    {})

    assert advice.as_settings()["fraction_threshold"] == 0.02
    assert "not crowded enough" in advice.why("fraction_threshold")


def test_a_crowded_well_lowers_the_threshold_and_says_what_it_would_cost():
    advice = advise(_reading(kept_at_two_percent=0.2, fraction_median=0.01),
                    {})

    assert advice.as_settings()["fraction_threshold"] < 0.02
    assert "would keep only 20.0%" in advice.why("fraction_threshold")


# ---------------------------------------------------------------------------
# Reading a finished run
# ---------------------------------------------------------------------------

def _qc_folder(tmp_path, payload):
    from spacr.regression_qc import QC_NUMBERS_FILE

    folder = tmp_path / "run_1" / "regression_qc"
    folder.mkdir(parents=True)
    (folder / QC_NUMBERS_FILE).write_text(json.dumps(payload))
    return str(tmp_path / "run_1")


def test_no_run_folder_reads_nothing():
    assert read_the_last_run("") == {}


def test_a_folder_with_no_diagnostics_reads_nothing(tmp_path):
    assert read_the_last_run(str(tmp_path)) == {}


def test_a_diagnostics_file_that_is_not_json_is_reported_not_used(tmp_path):
    from spacr.regression_qc import QC_NUMBERS_FILE

    folder = tmp_path / "run_1" / "regression_qc"
    folder.mkdir(parents=True)
    (folder / QC_NUMBERS_FILE).write_text("{not json")

    got = read_the_last_run(str(tmp_path / "run_1"))

    assert "could not be read" in got["run_note"]
    assert got["run_folder"].endswith("regression_qc")


def test_a_diagnostics_file_with_no_numbers_is_reported_not_used(tmp_path):
    """A stale or empty summary is worse than none: it looks like measurement."""
    folder = _qc_folder(tmp_path, {"settings": {}})

    got = read_the_last_run(folder)

    assert got["run_note"] == ("the run's diagnostics file holds no numbers, "
                               "so nothing was taken from it")
    assert "residual_normal_p" not in got


def test_the_numbers_are_read_under_any_spelling_the_qc_wrote(tmp_path):
    folder = _qc_folder(tmp_path, {"numbers": {"shapiro_p": 0.02,
                                               "kurtosis": 4.5,
                                               "durbin_watson": 1.9,
                                               "cooks_max": 0.8,
                                               "vif_max": 3.1}})

    got = read_the_last_run(folder)

    assert got["residual_normal_p"] == 0.02
    assert got["residual_kurtosis"] == 4.5
    assert got["max_vif"] == 3.1


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

def test_a_regression_type_the_spec_does_not_know_is_refused():
    said = refusals({"regression_type": "quantum"})
    assert any("quantum" in message for message in said)


def test_an_unreadable_regression_spec_does_not_take_the_preflight_down(
        monkeypatch):
    """The other five checks still run when the spec cannot be imported."""
    from spacr import regression_spec

    monkeypatch.delattr(regression_spec, "REGRESSION_SETTINGS_USED")

    said = refusals({"regression_type": "quantum",
                     "analysis_mode": "guide_permutation",
                     "analysis_unit": "cell"})

    assert not any("quantum" in message for message in said)
    assert any("guide_permutation" in message for message in said)


def test_advice_that_passes_preflight_is_returned_unchanged():
    advice = advise_that_runs(_reading(), {})
    assert advice.as_settings()["regression_type"]
    assert refusals(dict(advice.as_settings())) == ()


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
