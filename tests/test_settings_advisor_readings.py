"""Table-driven tests of what ``spacr.settings_advisor`` reads and argues.

The advisor's danger is an authoritative guess, so every table here pairs a
measurable input with inputs that are short, misspelled, unreadable or
missing a column, and asserts the exact numbers, or the exact sentence in
``trouble`` / ``run_note``, that comes back.  One general test per unit, each
walking a case table with pytest's ``subtests`` fixture: ``read_the_counts``,
``read_the_response``, ``read_the_last_run``, ``advise`` and ``refusals``
(with the preflight-clean path of ``advise_that_runs``).  Tables are written to
disk because the readers read files.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from spacr.regression_qc import QC_NUMBERS_FILE
from spacr.settings_advisor import (Reading, advise, advise_that_runs,
                                    read_the_counts, read_the_last_run,
                                    read_the_response, refusals)

# ---------------------------------------------------------------------------
# Tables on disk
# ---------------------------------------------------------------------------


def _counts_csv(path, plates=2, genes=6, guides_per_gene=3, rows=2, columns=3,
                count=lambda guide: 100 if guide == 0 else 1,
                name=lambda gene, guide: f"TGGT1_{gene:06d}_{guide + 1}"):
    """A count table in the spelling the real count CSVs use."""
    out = [{"plate": f"plate{plate}", "row_name": f"r{row}",
            "column_name": f"c{column}", "grna_name": name(gene, guide),
            "count": count(guide)}
           for plate in range(1, plates + 1)
           for row in range(1, rows + 1)
           for column in range(1, columns + 1)
           for gene in range(genes)
           for guide in range(guides_per_gene)]
    pd.DataFrame(out).to_csv(path, index=False)
    return str(path)


def _scores_csv(path, n=60, well=True, column="pred"):
    """``pred = (i % 10) / 10`` over six wells (or no well column)."""
    frame = pd.DataFrame({column: [(i % 10) / 10 for i in range(n)]})
    if well:
        frame.insert(0, "prc", [f"plate1_r{i % 2 + 1}_c{i % 3 + 1}"
                                for i in range(n)])
    frame.to_csv(path, index=False)
    return str(path)


def test_read_the_counts_measures_the_design_or_says_why_not(
        tmp_path, monkeypatch, subtests):
    """Plates, wells, guides, genes and fractions, or the sentence explaining their absence."""
    from spacr import cell_montage

    junk = tmp_path / "notes.csv"
    junk.write_text("this file is a note about the screen\n", encoding="utf-8")
    header_only = tmp_path / "header.csv"
    header_only.write_text("plate\np1\n", encoding="utf-8")

    # 2 plates x 2 rows x 3 columns, 18 guides per well: one guide per gene
    # at 100 reads, two at 1, so a well's total is 612.
    real = {"plates": 2, "wells": 12, "guides": 18, "rows": 2, "columns": 3,
            "fraction_median": 1 / 612, "fraction_q90": 100 / 612,
            "guides_per_well": 18.0, "kept_at_two_percent": 1 / 3,
            "wells_per_guide": 12.0, "genes": 6, "guides_per_gene": 3.0,
            "trouble": []}
    # One plate, four wells, three guides of two genes, 100 reads each.
    small = {"plates": 1, "wells": 4, "guides": 3, "rows": 2, "columns": 2,
             "wells_per_guide": 4.0}
    fractions = {"fraction_median": 1 / 3, "fraction_q90": 1 / 3,
                 "guides_per_well": 3.0, "kept_at_two_percent": 1.0}
    names = ["TGGT1_000001_1", "TGGT1_000001_2", "TGGT1_000002_1"]

    def library(path, count, guide_names):
        return _counts_csv(path, plates=1, genes=1, rows=2, columns=2,
                           guides_per_gene=len(guide_names),
                           count=lambda guide: count,
                           name=lambda gene, guide: guide_names[guide])

    cases = [
        ("no count table", lambda: [], None,
         {"trouble": ["no count table is attached, so the design could not "
                      "be measured"]}),
        ("a missing file is no table", lambda: [str(tmp_path / "gone.csv")],
         None, {"trouble": ["no count table is attached, so the design could "
                            "not be measured"]}),
        ("a real two-plate screen",
         lambda: [_counts_csv(tmp_path / "counts.csv")], None, real),
        ("a sequenced one-plate library",
         lambda: [library(tmp_path / "reads.csv", 100, names)], None,
         dict(small, **fractions, genes=2, guides_per_gene=1.5, trouble=[])),
        ("a screen with no reads offers no fraction",
         lambda: [library(tmp_path / "no_reads.csv", 0, names)], None,
         dict(small, genes=2, guides_per_gene=1.5,
              trouble=["the count tables contain no positive guide "
                       "fractions, so no well has usable reads for a "
                       "threshold"])),
        ("a guide column of only the organism names no genes",
         lambda: [library(tmp_path / "prefix.csv", 100, ["TGGT1_"])], None,
         dict(small, guides=1, fraction_median=1.0, fraction_q90=1.0,
              guides_per_well=1.0, kept_at_two_percent=1.0, trouble=[])),
        ("an empty count table",
         lambda: [str(header_only)],
         lambda paths: pd.DataFrame(columns=["prc", "grna", "fraction"]),
         {"trouble": ["the count tables are empty"]}),
        ("a missing fraction column keeps the rest",
         lambda: [str(header_only)],
         lambda paths: pd.DataFrame({
             "prc": ["p1_r1_c1", "p1_r1_c2"],
             "grna": ["TGGT1_000001_1", "TGGT1_000001_2"]}),
         {"plates": 1, "wells": 2, "guides": 2, "rows": 1, "columns": 2,
          "wells_per_guide": 1.0, "genes": 1, "guides_per_gene": 2.0,
          "trouble": []}),
    ]
    for case_id, paths, fractions_double, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                if fractions_double is not None:
                    mp.setattr(cell_montage, "fractions_from_counts",
                               fractions_double)
                got = read_the_counts(paths())
            assert got == pytest.approx(expected)

    with subtests.test(msg="a table that will not parse names the reason"):
        got = read_the_counts([str(junk)])
        assert list(got) == ["trouble"]
        assert len(got["trouble"]) == 1
        assert got["trouble"][0].startswith(
            "the count tables could not be read: The per-well guide fractions "
            "could not be built from the count CSVs: notes.csv has no grna or "
            "count column")


def test_read_the_response_measures_the_response_or_says_why_not(
        tmp_path, monkeypatch, subtests):
    """Range, shape and per-well support of the response, or the named trouble."""
    import scipy.stats

    six_wells = _scores_csv(tmp_path / "s.csv")
    areas = tmp_path / "areas.csv"
    pd.DataFrame({"prc": ["p1_r1_c1"], "area": [12.0]}).to_csv(areas,
                                                               index=False)
    plate2 = tmp_path / "plate2.csv"
    pd.DataFrame({"prc": ["p2_r1_c1"], "area": [3.0]}).to_csv(plate2,
                                                              index=False)
    broken = tmp_path / "half_written.csv"
    broken.write_text('prc,pred\np1_r1_c1,0.5\n"p1_r1_c2,0.6\n',
                      encoding="utf-8")
    words = tmp_path / "words.csv"
    pd.DataFrame({"prc": ["p1_r1_c1", "p1_r1_c2"],
                  "pred": ["high", "low"]}).to_csv(words, index=False)

    # Six wells of ten objects: each well's mean is 0.4 or 0.5.
    measured = {"trouble": [], "response": "pred", "score_files_read": 1,
                "objects_per_well": 10.0, "n_response": 60, "low": 0.4,
                "high": 0.5, "binary": False, "inside_unit": True,
                "on_unit": True, "integral": False,
                "skew": pytest.approx(0.0, abs=1e-12)}
    no_skew = dict(measured)
    del no_skew["skew"]

    cases = [
        ("no score table", lambda: read_the_response([]),
         {"trouble": ["no score table is attached, so the response could "
                      "not be measured"]}),
        ("summarised at the well level", lambda: read_the_response([six_wells]),
         measured),
        ("a named response is used as is",
         lambda: read_the_response([_scores_csv(tmp_path / "r.csv",
                                                column="recruitment")],
                                   "recruitment"),
         dict(measured, response="recruitment")),
        ("a missing shape test keeps the range", None, no_skew),
        ("a dependent variable that is not a column names the file",
         lambda: read_the_response([six_wells], "recruitment_ratio"),
         {"trouble": ["'recruitment_ratio' is not a column of s.csv; the "
                      "response was not measured"]}),
        ("no recognisable response lists the names it tried",
         lambda: read_the_response([str(areas)]),
         {"trouble": ["no dependent variable is named and none of pred, "
                      "prediction, predictions, score, recruitment is a "
                      "column"]}),
        ("a plate missing the response is named, the rest read",
         lambda: read_the_response([_scores_csv(tmp_path / "plate1.csv", n=20),
                                    str(plate2)]),
         {"trouble": ["plate2.csv has no 'pred' column"], "response": "pred",
          "score_files_read": 1, "objects_per_well": 3.0, "n_response": 20,
          "low": 1 / 3, "high": 17 / 30, "binary": False, "inside_unit": True,
          "on_unit": True, "integral": False,
          "skew": pytest.approx(0.0, abs=1e-12)}),
        ("a row cap on the first file stops and says capped",
         lambda: read_the_response([_scores_csv(tmp_path / "p1.csv", n=20),
                                    _scores_csv(tmp_path / "p2.csv", n=20)],
                                   row_cap=5),
         {"trouble": [], "response": "pred", "capped": True,
          "score_files_read": 1, "objects_per_well": 1.0, "n_response": 5,
          "low": 0.0, "high": 0.4, "binary": False, "inside_unit": False,
          "on_unit": True, "integral": False,
          "skew": pytest.approx(0.0, abs=1e-12)}),
        ("a file that raises while read is named and skipped",
         lambda: read_the_response([str(broken)]),
         {"trouble": ["half_written.csv: Error tokenizing data. C error: EOF "
                      "inside string starting at row 2"],
          "response": "pred", "score_files_read": 0}),
        ("a response holding no numbers says so",
         lambda: read_the_response([str(words)]),
         {"trouble": ["'pred' holds no number in the sample"],
          "response": "pred", "score_files_read": 1}),
        ("no well column is read at the object level out loud",
         lambda: read_the_response([_scores_csv(tmp_path / "objects.csv",
                                                n=40, well=False)]),
         {"trouble": ["the score table names no well, so the response was "
                      "read at the object level rather than the well level "
                      "the fit uses"],
          "response": "pred", "score_files_read": 1, "n_response": 40,
          "low": 0.0, "high": 0.9, "binary": False, "inside_unit": False,
          "on_unit": True, "integral": False,
          "normal_p": pytest.approx(0.010202794955170567, rel=1e-9),
          "skew": pytest.approx(0.0, abs=1e-12)}),
    ]
    for case_id, call, expected in cases:
        with subtests.test(msg=case_id):
            if call is None:
                def refuse(*args, **kwargs):
                    raise RuntimeError("scipy.stats is unavailable")

                with monkeypatch.context() as mp:
                    mp.setattr(scipy.stats, "skew", refuse)
                    got = read_the_response([six_wells])
            else:
                got = call()
            assert got == pytest.approx(expected)


def _qc_file(folder, payload, nested=True):
    """Write a run's QC numbers file; return the folder the reader is given."""
    where = folder / "regression_qc" if nested else folder
    where.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (where / QC_NUMBERS_FILE).write_text(text, encoding="utf-8")
    return str(folder)


def test_read_the_last_run_takes_numbers_only_from_a_usable_file(
        tmp_path, subtests):
    """A run's numbers under any spelling, or a note saying why none were taken."""
    nested_dir = tmp_path / "run_nested"
    os.makedirs(nested_dir / "regression_qc" / QC_NUMBERS_FILE)
    qc = lambda name: str(tmp_path / name / "regression_qc")   # noqa: E731

    cases = [
        ("no folder", "", None, {}),
        ("a folder with no numbers file", str(tmp_path / "empty"), None, {}),
        ("a directory where the numbers file belongs", str(nested_dir), None,
         {}),
        ("not json", _qc_file(tmp_path / "bad", "{not json"), None,
         {"run_folder": qc("bad"),
          "run_note": "the run's diagnostics could not be read "
                      "(JSONDecodeError), so nothing was taken from it"}),
        ("no numbers", _qc_file(tmp_path / "bare", {"settings": {}}), None,
         {"run_folder": qc("bare"),
          "run_note": "the run's diagnostics file holds no numbers, so "
                      "nothing was taken from it"}),
        ("every spelling the qc writes",
         _qc_file(tmp_path / "spelt", {"numbers": {
             "shapiro_p": 0.02, "kurtosis": 4.5, "durbin_watson": 1.9,
             "cooks_max": 0.8, "vif_max": 3, "max_vif_flag": True}}), None,
         {"run_folder": qc("spelt"), "residual_normal_p": 0.02,
          "residual_kurtosis": 4.5, "durbin_watson": 1.9,
          "max_cooks_distance": 0.8, "max_vif": 3.0}),
        ("the first spelling wins and booleans are not numbers",
         _qc_file(tmp_path / "first", {"numbers": {
             "normality_p": True, "dagostino_p": 0.3, "shapiro_p": 0.9}}),
         None, {"run_folder": qc("first"), "residual_normal_p": 0.3}),
        ("a flat numbers file in the run folder",
         _qc_file(tmp_path / "flat", {"regression_type": "ols", "numbers": {
             "max_cooks_distance": 0.8}}, nested=False), None,
         {"run_folder": str(tmp_path / "flat"), "max_cooks_distance": 0.8}),
        ("a run of another model family is stale",
         _qc_file(tmp_path / "stale", {"regression_type": "OLS",
                                       "numbers": {"max_vif": 2.0}}),
         {"regression_type": "beta"},
         {"run_folder": qc("stale"),
          "run_note": "the last run fitted 'ols' and the panel now says "
                      "'beta', so its residual diagnostics describe a "
                      "different model and were not used"}),
        ("the same family is read",
         _qc_file(tmp_path / "same", {"regression_type": "ols",
                                      "numbers": {"max_vif": 2.0}}),
         {"regression_type": " OLS "},
         {"run_folder": qc("same"), "max_vif": 2.0}),
    ]
    for case_id, folder, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert read_the_last_run(folder, settings) == expected


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


def test_advise_argues_inference_hit_line_and_fraction_threshold(subtests):
    """Each choice's value and its reason follow from the reading and answers."""
    cases = [
        ("non-normal on too few wells stays parametric",
         _reading(normal_p=0.001, wells=40), {}, "inference", "parametric",
         "the response is not normal (p = 0.001), but 40 wells is too few to "
         "build a permutation null with resolution — a robust parametric fit "
         "is the better trade"),
        ("non-normal on many wells goes nonparametric",
         _reading(normal_p=0.001, wells=400), {}, "inference", "nonparametric",
         "the well response fails a normality test (D'Agostino p = 0.001) and "
         "there are 400 wells to permute, so an empirical null costs time "
         "rather than assumptions"),
        ("96 wells is enough to permute",
         _reading(normal_p=0.01, wells=96), {}, "inference", "nonparametric",
         "the well response fails a normality test (D'Agostino p = 0.01) and "
         "there are 96 wells to permute, so an empirical null costs time "
         "rather than assumptions"),
        ("a normal response stays parametric", _reading(normal_p=0.5), {},
         "inference", "parametric",
         "the well response is consistent with normal (D'Agostino p = 0.5), "
         "which is the assumption a parametric p-value needs"),
        ("an increase-only hit list keeps the adjusted p",
         _reading(binary=False), {"direction": "up"}, "p_threshold_kind",
         "adjusted",
         "a hit is an increase only, and the line is still drawn on the "
         "corrected p-value; the direction is read off the coefficient's "
         "sign"),
        ("a decrease-only hit list keeps the adjusted p",
         _reading(binary=False), {"direction": "down"}, "p_threshold_kind",
         "adjusted",
         "a hit is a decrease only, and the line is still drawn on the "
         "corrected p-value; the direction is read off the coefficient's "
         "sign"),
        ("a binary response ignores the direction",
         _reading(binary=True), {"direction": "up"}, "p_threshold_kind",
         "adjusted",
         "the hit line is drawn on the corrected p-value, so the picture and "
         "the exported hit list mean the same thing by 'significant'"),
        ("an uncrowded well keeps 0.02",
         _reading(kept_at_two_percent=0.95, fraction_median=0.2), {},
         "fraction_threshold", 0.02,
         "the wells are not crowded enough for the cut to bite: 0.02 keeps "
         "95.0% of the guide-in-well pairs"),
        ("no kept share falls back to the usual default",
         _reading(kept_at_two_percent=None, fraction_median=0.2), {},
         "fraction_threshold", 0.02,
         "the usual default, with nothing here arguing against it"),
    ]
    for case_id, reading, answers, key, value, why in cases:
        with subtests.test(msg=case_id):
            advice = advise(reading, answers)
            assert advice.as_settings()[key] == value
            assert advice.why(key) == why
            assert advice.reading is reading

    no_fraction = [
        ("a screen with no reads leaves the threshold undecided",
         _reading(fraction_median=None, kept_at_two_percent=None,
                  trouble=("the count tables contain no positive guide "
                           "fractions, so no well has usable reads for a "
                           "threshold",)),
         "the count tables contain no positive guide fractions; without "
         "usable reads in any well, nothing here can say what a threshold "
         "would cost"),
        ("a table with no fraction column leaves it undecided",
         _reading(fraction_median=None, kept_at_two_percent=None),
         "the count tables did not yield a fraction column, so nothing here "
         "can say what a threshold would cost"),
    ]
    for case_id, reading, why in no_fraction:
        with subtests.test(msg=case_id):
            advice = advise(reading, {})
            assert "fraction_threshold" not in advice.as_settings()
            assert advice.why("fraction_threshold") == why

    with subtests.test(msg="a crowded well lowers the threshold and says the cost"):
        advice = advise(_reading(kept_at_two_percent=0.2,
                                 fraction_median=0.01), {})
        assert advice.as_settings()["fraction_threshold"] == 0.001
        assert advice.why("fraction_threshold").startswith(
            "the usual 0.02 would keep only 20.0% of the guide-in-well pairs "
            "here, because the typical guide is 1.000% of its well. THE "
            "THRESHOLD IS A NOISE FLOOR")


_CONTROL_CENTER = ("batch_correction='control_center' requires "
                   "batch_control_column and at least one batch_control_value.")
_GUIDE_PERMUTATION = (
    "analysis_mode='guide_permutation' tests each guide across WELLS, so it "
    "needs one row per well -- but analysis_unit='cell' gives one row per "
    "object, and a well's phenotype then has many values. Set "
    "analysis_unit='well' (with an agg_type such as 'mean'), or choose "
    "analysis_mode='regression', which can model objects.")


def test_refusals_name_every_unrunnable_combination(monkeypatch, subtests):
    """Exactly the preflight messages a mapping earns; a clean one earns none."""
    from spacr import regression_spec

    known = ", ".join(sorted(regression_spec.REGRESSION_SETTINGS_USED))
    agg_mean = ("analysis_unit='cell' keeps one row per object, so "
                "agg_type='mean' is never read. Clear it, or set "
                "analysis_unit='well' if the aggregation was the intent.")
    combat = ("batch_correction='combat' needs to know which biology to keep, "
              "and no batch_covariate_column is set. Name the "
              "condition/treatment column, or set it to 'no_covariate' to "
              "state that there is nothing to preserve.")
    cases = [
        ("an unknown regression type",
         {"regression_type": "quantum"},
         (f"regression_type='quantum' is not one of {known}.",)),
        ("a known regression type", {"regression_type": "OLS"}, ()),
        ("control_center with nothing to centre on",
         {"batch_correction": "control_center", "regression_type": "ols"},
         (_CONTROL_CENTER,)),
        ("control_center is case-insensitive",
         {"batch_correction": "CONTROL_CENTER"}, (_CONTROL_CENTER,)),
        ("control_center with its column",
         {"batch_correction": "control_center",
          "batch_control_column": "condition",
          "batch_control_value": "untreated", "regression_type": "ols"}, ()),
        ("later checks still run after control_center",
         {"batch_correction": "control_center",
          "batch_control_column": "condition", "analysis_unit": "cell",
          "agg_type": "mean"}, (agg_mean,)),
        ("combat with an empty covariate never reaches control_center",
         {"batch_correction": "combat", "batch_control_column": ""},
         (combat,)),
        ("guide permutation over cells",
         {"analysis_mode": "guide_permutation", "analysis_unit": "cell"},
         (_GUIDE_PERMUTATION,)),
    ]
    for case_id, settings, expected in cases:
        with subtests.test(msg=case_id):
            assert refusals(settings) == expected

    with subtests.test(msg="an unreadable regression spec skips only its check"):
        with monkeypatch.context() as mp:
            mp.delattr(regression_spec, "REGRESSION_SETTINGS_USED")
            said = refusals({"regression_type": "quantum",
                             "analysis_mode": "guide_permutation",
                             "analysis_unit": "cell"})
        assert said == (_GUIDE_PERMUTATION,)

    with subtests.test(msg="advice that passes preflight is returned unchanged"):
        reading = _reading()
        advice = advise_that_runs(reading, {})
        assert advice == advise(reading, {})
        assert advice.as_settings()["regression_type"] == "beta"
        assert refusals(advice.as_settings()) == ()


def test_advise_decides_nothing_the_unread_response_would_have_decided():
    """No score table: the family/transform and the aggregation stay undecided, saying why."""
    advice = advise(_reading(n_response=0, low=None, high=None, skew=None,
                             normal_p=None))
    chosen = advice.as_settings()
    for key in ("regression_type", "transform", "agg_type"):
        assert key not in chosen, key
    assert advice.why("regression_type") == (
        "the response was not read, so its distribution decided nothing. "
        "Attach the score table and press this again.")
    assert advice.why("agg_type") == (
        "the response was not read, so nothing here knows whether its "
        "tail would drag a mean")
