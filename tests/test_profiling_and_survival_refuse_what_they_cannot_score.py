"""Image-based profiling and time-to-event statistics on the inputs a user
gets wrong or a screen too small to score: each is refused with a sentence
naming the problem, or finished with a note saying what was left out.

The lifelines engine is optional and is not installed here; a stand-in
``lifelines`` module answers with the built-in engine's numbers, so what is
checked is that spaCR reads lifelines' tables into its own the same way.
"""
from __future__ import annotations

import importlib.machinery
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s
from spacr.tabular import database_tables, write_database


def _wells(seed=0, plates=("p1", "p2"), features=8, treatments=("t1", "t2"),
           controls=4, replicates=2, effect=4.0):
    rng = np.random.default_rng(seed)
    signature = {t: rng.normal(0, 1, features) for t in treatments}
    rows = []
    for plate in plates:
        names = ["neg"] * controls + [t for t in treatments
                                      for _ in range(replicates)]
        for index, name in enumerate(names):
            values = rng.normal(0, 1, features) + 10.0
            if name in signature:
                values = values + effect * signature[name]
            rows.append({"plateID": plate, "rowID": "r1",
                         "columnID": f"c{index + 1}", "treatment": name,
                         **{f"f{k}": v for k, v in enumerate(values)}})
    frame = pd.DataFrame(rows)
    return frame, [f"f{k}" for k in range(features)]


def _db(path, table="cell", rows=None):
    rows = rows if rows is not None else [
        {"plateID": "p1", "rowID": "r1", "columnID": f"c{c}",
         "object_label": o, f"{table}_area": 10.0 + c + o}
        for c in (1, 2) for o in (1, 2)]
    write_database(pd.DataFrame(rows), str(path), table, if_exists="replace")
    return str(path)


def test_boolean_columns_are_not_features_and_strata_must_exist():
    frame = pd.DataFrame({"plateID": ["p"], "cell_area": [1.0],
                          "cell_is_edge": [True]})
    assert s._profile_features(frame) == ["cell_area"]
    with pytest.raises(s._ProfilingError, match="no rowID, columnID column"):
        s._aggregate_profiles(frame, ["cell_area"])


def test_reading_databases_names_a_missing_table_and_skips_empty_ones(
        tmp_path):
    one = _db(tmp_path / "a.db")
    with pytest.raises(s._ProfilingError, match="has no nucleus table"):
        s._read_well_profiles([one], tables=["cell", "nucleus"], report=None)
    empty = tmp_path / "empty.db"
    write_database(pd.DataFrame({"plateID": pd.Series([], dtype=str),
                                 "rowID": pd.Series([], dtype=str),
                                 "columnID": pd.Series([], dtype=str),
                                 "cell_area": pd.Series([], dtype=float)}),
                   str(empty), "cell", if_exists="replace")
    other = tmp_path / "other.db"
    write_database(pd.DataFrame({"x": [1]}), str(other), "notes",
                   if_exists="replace")
    with pytest.raises(s._ProfilingError, match="none of the databases"):
        s._read_well_profiles([str(empty), str(other)], report=None)
    wells, features = s._read_well_profiles([one, str(empty)], report=None)
    assert features == ["cell_area"] and len(wells) == 2


def test_a_plate_map_must_name_its_wells_and_a_gap_is_reported():
    wells, _features = _wells()
    with pytest.raises(s._ProfilingError, match="names no well"):
        s._annotate_profiles(wells, pd.DataFrame({"dose": [1.0]}),
                             report=None)
    said = []
    plate_map = pd.DataFrame({"rowID": ["r1"], "columnID": ["c1"],
                              "dose": [1.0]})
    annotated, columns = s._annotate_profiles(
        wells.drop(columns="treatment"), plate_map, report=said.append)
    assert columns == ["dose"]
    assert said and "does not cover" in said[0]
    assert annotated["dose"].notna().sum() == 2


def test_normalisation_none_and_one_group_for_the_whole_table():
    wells, features = _wells()
    same = s._normalize_profiles(wells, features, method="none")
    assert same[features].equals(wells[features])
    pooled = s._normalize_profiles(wells, features, method="standardize",
                                   by=None)
    assert np.allclose(pooled[features].mean(), 0.0, atol=1e-9)
    missing_by = s._normalize_profiles(wells, features,
                                       method="standardize", by="batch")
    assert np.allclose(missing_by[features].to_numpy(),
                       pooled[features].to_numpy())


def test_small_inputs_to_the_selection_consensus_and_precision_helpers():
    assert s._redundant(pd.DataFrame({"a": [1.0, 2.0]}), 0.9) == []
    wells, features = _wells()
    with pytest.raises(s._ProfilingError, match="unknown consensus 'mode'"):
        s._consensus_profiles(wells, features, ["treatment"], operation="mode")
    assert s._expected_ap(1, 0) == 1.0
    assert s._expected_ap(0, 1) == 0.0
    assert s._expected_ap(0, 3) == 0.0
    assert s._expected_ap(3, 0) == 1.0
    meta = pd.DataFrame({"t": ["a", "a", "b", "b"]})
    feats = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]])
    ap = s._profile_average_precision(meta, feats, ["t"], (), (), ["t"],
                                      similarity="euclidean")
    assert (ap["average_precision"] == 1.0).all(), (
        "under euclidean similarity the nearest well is the replicate")
    with pytest.raises(s._ProfilingError, match="unknown similarity"):
        s._profile_average_precision(meta, feats, ["t"], similarity="dot")
    alone = pd.DataFrame({"t": ["a", "b", "c"]})
    with pytest.raises(s._ProfilingError, match="no profile has a replicate"):
        s._profile_average_precision(alone, feats[:3], ["t"], (), (), ["t"])
    same = pd.DataFrame({"t": ["a", "a", "a"]})
    with pytest.raises(s._ProfilingError, match="no profile has a negative"):
        s._profile_average_precision(same, feats[:3], ["t"], (), (), ["t"])


def test_empty_and_unreplicated_scores_give_empty_tables():
    empty = s._profile_map(pd.DataFrame(columns=[
        "t", "average_precision", "n_pos_pairs", "n_total_pairs"]), ["t"])
    assert empty.empty and "mean_average_precision" in empty.columns
    wells, features = _wells()
    table, percent = s._percent_replicating(wells.iloc[:0], features,
                                            ["treatment"])
    assert table.empty and np.isnan(percent)
    single = wells.drop_duplicates("treatment")
    table, percent = s._percent_replicating(single, features, ["treatment"])
    assert table.empty and np.isnan(percent)
    one_group = wells[wells["treatment"] == "t1"]
    table, percent = s._percent_replicating(one_group, features,
                                            ["treatment"], n_null=20)
    assert table["n_replicates"].tolist() == [4]


def test_metadata_already_prefixed_or_absent_is_left_alone():
    frame = pd.DataFrame({"Metadata_Batch": ["b"], "plateID": ["p"],
                          "f0": [1.0]})
    out = s._external_profiles(frame, ["Metadata_Batch", "plateID", "dose"])
    assert list(out.columns) == ["Metadata_Batch", "Metadata_Plate", "f0"]


def test_a_large_similarity_map_drops_its_names():
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    consensus = pd.DataFrame(rng.normal(0, 1, (61, 3)), columns=list("abc"))
    figure = s._similarity_figure(consensus, list("abc"),
                                  [f"t{i}" for i in range(61)])
    try:
        assert list(figure.axes[0].get_xticks()) == []
    finally:
        plt.close(figure)


@pytest.mark.parametrize("value,expected", (
    (None, []), ("", []), ("a, b", ["a", "b"]), ("['a', 'b']", ["a", "b"]),
    ("[a, b", ["[a", "b"]), ("[not python]", ["[not python]"]),
    (("x", " "), ["x"]), (3, ["3"]),
))
def test_a_list_setting_is_read_however_it_was_typed(value, expected):
    assert s._as_list(value) == expected


def test_too_few_kept_features_are_refused():
    wells, features = _wells(features=3)
    wells[features] = 1.0
    with pytest.raises(s._ProfilingError, match="feature selection kept"):
        s._profile_wells(wells, features, group_columns=["treatment"],
                         report=None)


def test_a_well_with_a_missing_feature_is_left_out_with_a_note(monkeypatch):
    wells, features = _wells(treatments=("t1", "t2", "t3"))
    wells.loc[len(wells) - 1, "f0"] = np.nan
    wells["moa"] = wells["treatment"].map({"t1": "m1", "t2": "m1",
                                           "t3": "m2", "neg": None})

    def refuse(*_a, **_k):
        raise s._ProfilingError("nothing to rank")

    monkeypatch.setattr(s, "_phenotypic_activity", refuse)
    monkeypatch.setattr(s, "_phenotypic_consistency", refuse)
    result = s._profile_wells(wells, features, group_columns=["treatment"],
                              negative_controls=["neg"], feature_selection=(),
                              phenotype_column="moa", null_size=50,
                              report=None)
    notes = " ".join(result.notes)
    assert "1 well(s) with a missing kept feature" in notes
    assert "Phenotypic activity was not scored: nothing to rank." in notes
    assert "Phenotypic consistency was not scored: nothing to rank." in notes


def test_consistency_is_charted_and_a_parquet_writer_can_be_missing(
        tmp_path, monkeypatch):
    import spacr.tabular as tabular

    wells, features = _wells(treatments=("t1", "t2", "t3", "t4"), effect=6.0)
    wells["moa"] = wells["treatment"].map({"t1": "m1", "t2": "m1",
                                           "t3": "m2", "t4": "m2",
                                           "neg": None})
    result = s._profile_wells(wells, features, group_columns=["treatment"],
                              negative_controls=["neg"], feature_selection=(),
                              phenotype_column="moa", null_size=50,
                              report=None)
    assert len(result.consistency_map)
    assert result.summary()["phenotypically_consistent"] is not None
    real = tabular.write_table

    def no_parquet(frame, path, *a, **k):
        if str(path).endswith(".parquet"):
            raise ImportError("pyarrow is not installed")
        return real(frame, path, *a, **k)

    monkeypatch.setattr(tabular, "write_table", no_parquet)
    db = tmp_path / "measurements.db"
    extra = pd.DataFrame(0.0, index=result.consensus.index,
                         columns=[f"w{k}" for k in
                                  range(s._SQLITE_COLUMN_LIMIT)])
    result.consensus = pd.concat([result.consensus, extra], axis=1)
    written = s._write_profiles(result, str(tmp_path / "profiles"),
                                db_path=str(db))
    assert "map_consistency_figure" in written
    assert "well_profiles.csv" in written
    assert not any(name.endswith(".parquet") for name in written)
    assert "profile_consensus" not in database_tables(str(db)), (
        "a table wider than SQLite allows is left out of the database")
    assert "profile_wells" in database_tables(str(db))
    summary = json.loads(open(written["summary"]).read())
    assert summary["phenotypically_consistent"] == result.summary()[
        "phenotypically_consistent"]


def test_a_measure_run_without_a_plate_map_profiles_by_column(tmp_path):
    rows = []
    rng = np.random.default_rng(3)
    for c in range(1, 7):
        for o in range(1, 6):
            rows.append({"plateID": "p1", "rowID": "r1", "columnID": f"c{c}",
                         "object_label": o,
                         **{f"cell_f{k}": rng.normal(10, 1) + (c % 3) * 3
                            for k in range(4)}})
    db = _db(tmp_path / "measurements.db", rows=rows)
    result, written = s._profile_measurements(
        {"profiling_feature_selection": []}, db, report=None)
    assert result.group_columns == ["columnID"]
    assert "summary" in written


class _Fitter:
    """A stand-in for lifelines' fitters, answering from spaCR's builtin."""

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, *args):
        if len(args) == 2:
            times, observed = args
            curve = s._kaplan_meier(times, observed, alpha=self.alpha,
                                    engine="builtin")
            self.event_table = pd.DataFrame({
                "at_risk": curve["at_risk"].to_numpy(),
                "observed": curve["events"].to_numpy(),
                "censored": curve["censored"].to_numpy()},
                index=curve["time"].to_numpy())
            self.survival_function_ = curve[["survival"]]
            self.confidence_interval_survival_function_ = curve[
                ["ci_lower", "ci_upper"]]
            return self
        data, duration, event = args
        covariates = [c for c in data.columns if c not in (duration, event)]
        table, fit = s._cox_regression(data, duration, event, covariates,
                                       alpha=self.alpha, engine="builtin")
        self.summary = pd.DataFrame(
            {"coef": table["coef"].to_numpy(), "se(coef)": table["se"]
             .to_numpy()}, index=covariates)
        self.log_likelihood_ = fit["log_likelihood"]
        self._lr = fit["lr_statistic"]
        return self

    def log_likelihood_ratio_test(self):
        return types.SimpleNamespace(test_statistic=self._lr)


@pytest.fixture
def lifelines(monkeypatch):
    module = types.ModuleType("lifelines")
    module.__spec__ = importlib.machinery.ModuleSpec("lifelines", None)
    module.KaplanMeierFitter = _Fitter
    module.CoxPHFitter = _Fitter
    statistics = types.ModuleType("lifelines.statistics")

    def multivariate_logrank_test(times, labels, observed):
        out = s._logrank(times, observed, labels, engine="builtin")
        return types.SimpleNamespace(test_statistic=out["statistic"],
                                     p_value=out["p_value"])

    statistics.multivariate_logrank_test = multivariate_logrank_test
    module.statistics = statistics
    monkeypatch.setitem(sys.modules, "lifelines", module)
    monkeypatch.setitem(sys.modules, "lifelines.statistics", statistics)
    return module


def _survival_table():
    rng = np.random.default_rng(4)
    dose = np.repeat([0.0, 1.0], 20)
    times = rng.exponential(10.0 / (1.0 + dose))
    events = rng.random(40) < 0.8
    return pd.DataFrame({"t": times.round(2), "e": events, "dose": dose,
                         "g": np.where(dose > 0, "drug", "vehicle")})


def test_the_lifelines_engine_is_read_the_way_the_builtin_one_is(lifelines):
    frame = _survival_table()
    assert s._survival_engine() == "lifelines"
    ours = s._kaplan_meier(frame["t"], frame["e"], engine="builtin")
    theirs = s._kaplan_meier(frame["t"], frame["e"])
    assert np.allclose(theirs["survival"], ours["survival"])
    assert theirs["at_risk"].tolist() == ours["at_risk"].tolist()
    test = s._logrank(frame["t"], frame["e"], frame["g"])
    assert test["p_value"] == pytest.approx(s._logrank(
        frame["t"], frame["e"], frame["g"], engine="builtin")["p_value"])
    table, fit = s._cox_regression(frame, "t", "e", ["dose"])
    assert fit["engine"] == "lifelines"
    assert table.loc[0, "hazard_ratio"] > 1.0


def test_without_lifelines_the_builtin_engine_answers_or_the_ask_is_refused(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "lifelines", None)
    assert s._survival_engine() == "builtin"
    with pytest.raises(ImportError, match="pip install lifelines"):
        s._survival_engine("lifelines")


def test_survival_inputs_that_do_not_line_up_are_refused():
    with pytest.raises(ValueError, match="3 durations but 2 event flags"):
        s._kaplan_meier([1, 2, 3], [1, 0])
    with pytest.raises(ValueError, match="no observations"):
        s._kaplan_meier([], [])
    with pytest.raises(ValueError, match="3 durations but 2 groups"):
        s._logrank([1, 2, 3], [1, 1, 0], ["a", "b"])
    frame = _survival_table()
    with pytest.raises(ValueError, match="at least one covariate"):
        s._cox_regression(frame, "t", "e", [])
    frame.loc[0, "dose"] = np.nan
    with pytest.raises(ValueError, match="must be finite on every row"):
        s._cox_regression(frame, "t", "e", ["dose"])


def test_a_logrank_risk_set_of_one_adds_no_variance():
    out = s._logrank([1.0, 2.0, 3.0], [1, 1, 1], ["a", "b", "b"],
                     engine="builtin")
    assert out["observed"] == {"a": 1.0, "b": 2.0}
    assert np.isfinite(out["statistic"])


def test_a_run_without_controls_or_figures_writes_its_tables_only(tmp_path):
    wells, features = _wells(treatments=("t1", "t2", "t3", "t4"))
    wells["moa"] = wells["treatment"].map({"t1": "m1", "t2": "m1",
                                           "t3": "m2", "t4": "m2",
                                           "neg": "m0"})
    result = s._profile_wells(wells, features, group_columns=["treatment"],
                              feature_selection=(), phenotype_column="moa",
                              null_size=50, report=None)
    assert result.activity_map.empty
    assert set(result.consistency_map["moa"]) == {"m1", "m2"}
    written = s._write_profiles(result, str(tmp_path / "out"), figures=False)
    assert not any("figure" in name for name in written)
    assert "consensus_profiles.gct" in written


def test_two_profiles_are_charted_in_the_order_given():
    import matplotlib.pyplot as plt

    consensus = pd.DataFrame({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    figure = s._similarity_figure(consensus, ["a", "b"], ["x", "y"])
    try:
        labels = [t.get_text() for t in figure.axes[0].get_xticklabels()]
        assert labels == ["x", "y"]
    finally:
        plt.close(figure)
