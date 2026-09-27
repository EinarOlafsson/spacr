"""Image-based profiling: aggregate, normalise, select, consensus, mAP.

The helpers live in spacr/sp_stats.py and follow pycytominer (aggregate,
normalize, feature_select, consensus) and copairs (average precision, mAP and
its permutation p value). Every step is checked here against values worked by
hand on small tables; the parity tests at the bottom compare with pycytominer
and copairs themselves and skip where those are not installed. The full
comparison on ten public LINCS Cell Painting plates and on CPJUMP1 single
cells is recorded in features/future/547_image_based_profiling.txt.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as s
from spacr.tabular import read_table, write_database, write_table


def _plate(plate, rng, signatures, *, inactive=("t5", "t6"), controls=8,
           replicates=2, features=12, effect=4.0):
    """One synthetic plate of well profiles with planted phenotypes."""
    rows = []
    well = 0
    names = ["neg"] * controls + [t for t in (*signatures, *inactive)
                                  for _ in range(replicates)]
    for name in names:
        row, col = divmod(well, 12)
        well += 1
        values = rng.normal(0, 1, features) + 10.0
        if name in signatures:
            values = values + effect * signatures[name]
        rows.append({"plateID": plate, "rowID": f"r{row + 1}",
                     "columnID": f"c{col + 1}", "treatment": name,
                     **{f"cell_f{i}": v for i, v in enumerate(values)}})
    return pd.DataFrame(rows)


def _screen(seed=0, plates=("p1", "p2")):
    """Two plates; t1-t4 share a signature across plates, t5 and t6 do not."""
    rng = np.random.default_rng(seed)
    signatures = {name: rng.normal(0, 1, 12) for name in ("t1", "t2", "t3",
                                                           "t4")}
    wells = pd.concat([_plate(p, rng, signatures) for p in plates],
                      ignore_index=True)
    features = [c for c in wells.columns if c.startswith("cell_f")]
    return wells, features


def test_features_are_numeric_measurements_not_identity_or_position():
    frame = pd.DataFrame({
        "plateID": ["p1"], "rowID": ["r1"], "columnID": ["c1"],
        "fieldID": ["f1"], "object_label": [3], "cell_id": [3],
        "cell_area": [10.0], "cell_channel_0_mean_intensity": [2.0],
        "cell_channel_0_centroid_weighted-0": [512.0],
        "cell_channel_0_centroid_weighted_local-0": [4.0],
        "cell_bbox-1": [7.0], "Metadata_dose": [1.0], "label": ["x"],
    })
    features = s._profile_features(frame)
    assert features == ["cell_area", "cell_channel_0_mean_intensity",
                        "cell_channel_0_centroid_weighted_local-0"]


def test_aggregation_is_a_per_well_median_with_object_counts():
    frame = pd.DataFrame({
        "plateID": ["p1"] * 5, "rowID": ["r1"] * 3 + ["r2"] * 2,
        "columnID": ["c1"] * 5, "cell_a": [1.0, 5.0, 3.0, 2.0, np.nan],
        "cell_b": [1, 2, 3, 4, 6],
    })
    out = s._aggregate_profiles(frame, ["cell_a", "cell_b"],
                                count_column="n_cell")
    assert out["n_cell"].tolist() == [3, 2]
    assert out["cell_a"].tolist() == [3.0, 2.0]
    assert out["cell_b"].tolist() == [2.0, 5.0]
    mean = s._aggregate_profiles(frame, ["cell_a"], operation="mean")
    assert mean["cell_a"].tolist() == [3.0, 2.0]
    with pytest.raises(s._ProfilingError):
        s._aggregate_profiles(frame, ["cell_a"], operation="mode")


def test_mad_robustize_uses_the_plate_controls_only():
    frame = pd.DataFrame({
        "plateID": ["p1"] * 4 + ["p2"] * 4,
        "ctrl": [True, True, True, False] * 2,
        "x": [1.0, 2.0, 4.0, 10.0, 11.0, 12.0, 14.0, 20.0],
    })
    out = s._normalize_profiles(frame, ["x"], reference=frame["ctrl"])
    scale = 1.0 * s.MAD_SCALE
    assert out["x"].tolist() == pytest.approx(
        [-1 / scale, 0.0, 2 / scale, 8 / scale] * 2)
    assert frame["x"].iloc[0] == 1.0


def test_standardize_and_robustize_match_scikit_learn():
    from sklearn.preprocessing import RobustScaler, StandardScaler

    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(5, 2, (30, 3)), columns=list("abc"))
    frame["c"] = 1.0
    frame["plateID"] = "p1"
    for method, scaler in (("standardize", StandardScaler()),
                           ("robustize", RobustScaler())):
        out = s._normalize_profiles(frame, list("abc"), method=method)
        expected = scaler.fit_transform(frame[list("abc")])
        assert np.allclose(out[list("abc")].to_numpy(), expected)


def test_a_plate_without_reference_wells_is_refused():
    frame = pd.DataFrame({"plateID": ["p1", "p2"], "x": [1.0, 2.0]})
    with pytest.raises(s._ProfilingError, match="p2"):
        s._normalize_profiles(frame, ["x"],
                              reference=pd.Series([True, False]))
    with pytest.raises(s._ProfilingError):
        s._normalize_profiles(frame, ["x"], method="zscore")


def test_each_selection_step_removes_what_it_names():
    rng = np.random.default_rng(1)
    n = 40
    base = rng.normal(0, 1, n)
    frame = pd.DataFrame({
        "keep": rng.normal(0, 1, n),
        "constant": np.ones(n),
        "near_constant": np.r_[np.zeros(n - 1), 1.0],
        "twin": base,
        "twin_copy": base * 2 + 0.01 * rng.normal(0, 1, n),
        "gappy": np.where(np.arange(n) < 5, np.nan, rng.normal(0, 1, n)),
        "wild": np.r_[rng.normal(0, 1, n - 1), 900.0],
    })
    features = list(frame.columns)
    kept, excluded = s._select_profile_features(frame, features)
    assert excluded["variance_threshold"] == ["constant"]
    assert excluded["frequency_threshold"] == ["near_constant"]
    assert len(excluded["correlation_threshold"]) == 1
    assert excluded["correlation_threshold"][0] in {"twin", "twin_copy"}
    assert excluded["drop_na_columns"] == ["gappy"]
    assert excluded["drop_outliers"] == ["wild"]
    assert "keep" in kept and len(kept) == 2
    kept_all, excluded_none = s._select_profile_features(frame, features, ())
    assert kept_all == features and excluded_none == {}
    with pytest.raises(s._ProfilingError):
        s._select_profile_features(frame, features, ["blocklist"])


def test_the_more_redundant_member_of_a_correlated_pair_goes():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 200)
    b = a + 0.1 * rng.normal(0, 1, 200)
    c = a + 0.2 * rng.normal(0, 1, 200)
    frame = pd.DataFrame({"a": a, "b": b, "c": c,
                          "z": rng.normal(0, 1, 200)})
    corr = frame.corr().abs().sum()
    dropped = s._redundant(frame, 0.9)
    assert set(dropped) == set(corr[["a", "b", "c"]].sort_values().index[1:])


def test_exact_duplicates_keep_the_first_in_column_order():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, 50)
    frame = pd.DataFrame({"first": x, "second": x.copy(),
                          "other": rng.normal(0, 1, 50)})
    assert s._redundant(frame, 0.9) == ["second"]


def test_consensus_median_and_modz():
    frame = pd.DataFrame({
        "t": ["a", "a", "a", "b"],
        "x": [1.0, 2.0, 9.0, 4.0], "y": [0.0, 1.0, 2.0, 5.0],
        "z": [3.0, 1.0, 2.0, 1.0],
    })
    median = s._consensus_profiles(frame, ["x", "y", "z"], ["t"])
    assert median["n_replicates"].tolist() == [3, 1]
    assert median.loc[0, ["x", "y", "z"]].tolist() == [2.0, 1.0, 2.0]
    modz = s._consensus_profiles(frame, ["x", "y", "z"], ["t"],
                                 operation="modz")
    assert modz.loc[1, ["x", "y", "z"]].tolist() == [4.0, 5.0, 1.0]
    replicates = frame.loc[frame.t == "a", ["x", "y", "z"]]
    corr = replicates.T.corr(method="spearman").to_numpy()
    np.fill_diagonal(corr, np.nan)
    weights = np.maximum(np.nanmean(np.clip(corr, 0, None), axis=1), 0.01)
    weights = np.round(weights / weights.sum(), 4)
    expected = (replicates.to_numpy() * weights[:, None]).sum(axis=0)
    assert modz.loc[0, ["x", "y", "z"]].to_numpy() == pytest.approx(expected)


def test_average_precision_of_a_hand_ranked_list():
    meta = pd.DataFrame({"t": ["q", "q", "n", "q"], "ref": [-1, -1, 2, -1]})
    feats = np.array([[1.0, 0.0], [0.9, 0.1], [0.7, 0.3], [0.2, 0.8]])
    ap = s._profile_average_precision(meta, feats, ["t", "ref"], (), (),
                                      ["t", "ref"])
    assert ap.loc[0, "average_precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert ap.loc[0, "n_pos_pairs"] == 2 and ap.loc[0, "n_total_pairs"] == 3
    assert math.isnan(ap.loc[2, "average_precision"])
    expected = s._expected_ap(2, 1)
    assert ap.loc[0, "normalized_average_precision"] == pytest.approx(
        ((1 + 2 / 3) / 2 - expected) / (1 - expected))


def test_missing_metadata_neither_matches_nor_differs():
    codes = s._metadata_codes(pd.DataFrame({"t": ["a", None, "a", "b"]}),
                              ["t"])
    assert s._pair_mask(codes, 0, ["t"], ()).tolist() == [False, False,
                                                          True, False]
    assert s._pair_mask(codes, 0, (), ["t"]).tolist() == [False, False,
                                                          False, True]
    assert not s._pair_mask(codes, 1, (), ["t"]).any()


def test_the_null_is_average_precision_of_random_rankings():
    null = s._null_average_precision(2000, 2, 2, seed=0)
    assert np.all(null == 1.0)
    null = s._null_average_precision(20000, 1, 4, seed=0)
    assert np.allclose(np.unique(null), [0.25, 1 / 3, 0.5, 1.0])
    assert null.mean() == pytest.approx(s._expected_ap(1, 3), abs=0.01)
    chunked = s._null_average_precision(3000, 3, 17, seed=5, chunk_cells=100)
    whole = s._null_average_precision(3000, 3, 17, seed=5)
    assert np.array_equal(chunked, whole)


def test_planted_phenotypes_are_active_and_the_inert_ones_are_not():
    wells, features = _screen()
    result = s._profile_wells(wells, features, group_columns=["treatment"],
                              negative_controls=["neg"], null_size=2000,
                              report=None)
    mapped = result.activity_map.set_index("treatment")
    assert set(mapped.index) == {"t1", "t2", "t3", "t4", "t5", "t6"}
    for name in ("t1", "t2", "t3", "t4"):
        assert mapped.loc[name, "below_corrected_p"]
        assert mapped.loc[name, "mean_average_precision"] == 1.0
        assert mapped.loc[name, "p_value"] == pytest.approx(1 / 2001)
    for name in ("t5", "t6"):
        assert not mapped.loc[name, "below_corrected_p"]
    assert "neg" not in set(result.activity["treatment"])
    assert len(result.activity) == 6 * 4
    assert set(result.activity["n_pos_pairs"]) == {3}
    assert set(result.activity["n_total_pairs"]) == {3 + 16}
    assert result.consensus["n_replicates"].tolist() == [16] + [4] * 6
    above = result.replicating.set_index("treatment")["above_null"]
    assert above[["t1", "t2", "t3", "t4"]].all()
    assert 50 <= result.percent_replicating <= 100
    assert result.wells["wellID"].iloc[0] == "A01"


def test_consistency_scores_treatments_that_share_a_label():
    rng = np.random.default_rng(7)
    rows = []
    shared = {"moa1": rng.normal(0, 1, 20), "moa2": rng.normal(0, 1, 20)}
    for plate in ("p1", "p2"):
        for i in range(8):
            rows.append({"plateID": plate, "rowID": "r1",
                         "columnID": f"c{i + 1}", "treatment": "neg",
                         "moa": None,
                         **{f"f{k}": v for k, v in
                            enumerate(rng.normal(0, 1, 20))}})
        for j, (name, moa) in enumerate([("a", "moa1"), ("b", "moa1"),
                                          ("c", "moa1"), ("d", "moa2"),
                                          ("e", "moa2"), ("f", "moa2")]):
            values = 3 * shared[moa] + rng.normal(0, 0.5, 20)
            rows.append({"plateID": plate, "rowID": "r2",
                         "columnID": f"c{j + 1}", "treatment": name,
                         "moa": moa,
                         **{f"f{k}": v for k, v in enumerate(values)}})
    wells = pd.DataFrame(rows)
    features = [f"f{k}" for k in range(20)]
    result = s._profile_wells(wells, features, group_columns=["treatment"],
                              negative_controls=["neg"],
                              phenotype_column="moa", feature_selection=(),
                              null_size=2000, report=None)
    labels = result.consistency_map.set_index("moa")
    assert set(labels.index) == {"moa1", "moa2"}
    assert (labels["mean_average_precision"] == 1.0).all()
    assert len(result.consistency) == 6


def test_without_a_control_every_well_is_the_reference_and_activity_is_skipped():
    wells, features = _screen()
    result = s._profile_wells(wells, features, group_columns=["treatment"],
                              report=None)
    assert result.activity_map.empty
    assert any("No negative control" in note for note in result.notes)
    with pytest.raises(s._ProfilingError, match="treatment"):
        s._profile_wells(wells, features, group_columns=["treatment"],
                         negative_controls=["DMSO"], report=None)
    with pytest.raises(s._ProfilingError, match="dose"):
        s._profile_wells(wells, features, group_columns=["dose"],
                         report=None)


def _measurement_db(path, plate, rng, treatments):
    """A measurements.db with cell and nucleus tables for one plate."""
    cells, nuclei = [], []
    for w, name in enumerate(treatments):
        row, col = divmod(w, 12)
        shift = 3.0 if name == "hit" else 0.0
        for label in range(1, 21):
            cells.append({"plateID": plate, "rowID": f"r{row + 1}",
                          "columnID": f"c{col + 1}", "fieldID": "f1",
                          "prcf": f"{plate}_r{row + 1}_c{col + 1}_f1",
                          "object_label": label,
                          "cell_area": 100 + rng.normal(0, 5) + 20 * shift,
                          "cell_channel_0_mean_intensity":
                              rng.normal(10, 1) + shift,
                          "cell_channel_1_mean_intensity":
                              rng.normal(5, 1) - shift,
                          "cell_channel_0_centroid_weighted-0":
                              rng.uniform(0, 1000)})
            nuclei.append({"plateID": plate, "rowID": f"r{row + 1}",
                           "columnID": f"c{col + 1}", "fieldID": "f1",
                           "prcf": f"{plate}_r{row + 1}_c{col + 1}_f1",
                           "object_label": label, "cell_id": label,
                           "nucleus_area": 40 + rng.normal(0, 3) + shift,
                           "nucleus_channel_2_mean_intensity":
                               rng.normal(8, 1) + 2 * shift})
    write_database(pd.DataFrame(cells), path, "cell", if_exists="replace")
    write_database(pd.DataFrame(nuclei), path, "nucleus", if_exists="replace")
    return path


def test_measurement_databases_become_annotated_well_profiles(tmp_path):
    rng = np.random.default_rng(11)
    layout = ["neg"] * 4 + ["hit"] * 3 + ["dud"] * 3
    one = _measurement_db(str(tmp_path / "a" / "measurements.db"), "p1",
                          rng, layout)
    two = _measurement_db(str(tmp_path / "b" / "measurements.db"), "p2",
                          rng, layout)
    wells, features = s._read_well_profiles([one, two], report=None)
    assert len(wells) == 20
    assert set(wells["n_cell"]) == {20} and set(wells["n_nucleus"]) == {20}
    assert "cell_channel_0_centroid_weighted-0" not in features
    assert set(features) == {"cell_area", "cell_channel_0_mean_intensity",
                             "cell_channel_1_mean_intensity", "nucleus_area",
                             "nucleus_channel_2_mean_intensity"}
    plate_map = pd.DataFrame({
        "well": [f"A{i + 1:02d}" for i in range(len(layout))],
        "treatment": layout})
    annotated, columns = s._annotate_profiles(wells, plate_map, report=None)
    assert columns == ["treatment"]
    assert annotated["treatment"].tolist() == layout * 2
    assert annotated["columnID"].tolist()[:10] == [f"c{i + 1}"
                                                   for i in range(10)]
    with pytest.raises(s._ProfilingError, match="p1"):
        s._read_well_profiles([one, _copy(one)], report=None)
    with pytest.raises(s._ProfilingError, match="more than once"):
        s._annotate_profiles(wells, pd.concat([plate_map, plate_map]),
                             report=None)


def _copy(path):
    """A second database holding the same plate."""
    import shutil

    target = path.replace("measurements.db", "copy.db")
    shutil.copy(path, target)
    return target


def test_a_measure_run_writes_profiles_other_tools_read(tmp_path, capsys):
    from spacr.measure import _emit_profiles
    from spacr.settings import get_measure_crop_settings

    rng = np.random.default_rng(12)
    layout = ["neg"] * 4 + ["hit"] * 3 + ["dud"] * 3
    db = _measurement_db(str(tmp_path / "p1" / "measurements" /
                             "measurements.db"), "p1", rng, layout)
    other = _measurement_db(str(tmp_path / "p2" / "measurements" /
                                "measurements.db"), "p2", rng, layout)
    plate_map = str(tmp_path / "plate_map.csv")
    write_table(pd.DataFrame({
        "rowID": ["r1"] * len(layout),
        "columnID": [f"c{i + 1}" for i in range(len(layout))],
        "treatment": layout}), plate_map)
    settings = get_measure_crop_settings({"src": str(tmp_path / "p1")})
    assert settings["profiling"] is False
    settings.update(profiling=True, profiling_metadata=plate_map,
                    profiling_treatment_column="treatment",
                    profiling_negative_control="neg",
                    profiling_databases=[other],
                    profiling_feature_selection=[])
    _emit_profiles(settings, db)
    printed = capsys.readouterr().out
    assert "Profiles: 20 wells, 3 treatments" in printed
    out = tmp_path / "p1" / "measurements" / "profiles"
    summary = json.loads((out / "profiling_summary.json").read_text())
    assert summary["plates"] == ["p1", "p2"]
    assert summary["treatments_scored"] == 2
    exported = read_table(str(out / "well_profiles_feature_select.csv"),
                          canonicalise=False, report=None)
    assert {"Metadata_Plate", "Metadata_Well",
            "Metadata_treatment"} <= set(exported.columns)
    assert "cell_area" in exported.columns
    gct = (out / "consensus_profiles.gct").read_text().splitlines()
    assert gct[0] == "#1.3" and gct[1].split("\t")[:2] == ["5", "3"]
    mapped = read_table(str(out / "map_activity.csv"), report=None)
    assert set(mapped["treatment"]) == {"hit", "dud"}
    assert any(name.startswith("map_activity.") for name in os.listdir(out))
    assert any(name.startswith("consensus_similarity.")
               for name in os.listdir(out))
    from spacr.tabular import database_tables

    assert {"profile_wells", "profile_consensus",
            "profile_map"} <= set(database_tables(db))


def test_profiling_never_fails_a_finished_run(tmp_path, capsys):
    from spacr.measure import _emit_profiles

    rng = np.random.default_rng(13)
    db = _measurement_db(str(tmp_path / "measurements.db"), "p1", rng,
                         ["neg", "hit"])
    _emit_profiles({"profiling": True,
                    "profiling_metadata": str(tmp_path / "missing.csv")}, db)
    assert "Profiles could not be built" in capsys.readouterr().out
    _emit_profiles({"profiling": True}, str(tmp_path / "absent.db"))


def test_parity_with_pycytominer():
    pycytominer = pytest.importorskip("pycytominer")
    wells, features = _screen(seed=5)
    wells["Metadata_treatment"] = wells["treatment"]
    negative = wells["treatment"] == "neg"
    ours = s._normalize_profiles(wells, features, reference=negative)
    theirs = pd.concat([
        pycytominer.normalize(g, features=features,
                              meta_features=["plateID", "Metadata_treatment"],
                              samples="Metadata_treatment == 'neg'",
                              method="mad_robustize")
        for _, g in wells.groupby("plateID", sort=True)], ignore_index=True)
    assert np.allclose(ours[features].to_numpy(), theirs[features].to_numpy(),
                       rtol=2e-6)
    kept, _ = s._select_profile_features(ours, features)
    selected = pycytominer.feature_select(
        ours, features=features,
        operation=list(s._PROFILE_DEFAULT_SELECTIONS))
    assert set(kept) == {c for c in selected.columns if c in features}


def test_parity_with_copairs():
    pytest.importorskip("copairs")
    from copairs import map as cmap
    from copairs.matching import assign_reference_index

    wells, features = _screen(seed=6)
    wells = s._add_well_names(wells)
    ours_ap, ours_map = s._phenotypic_activity(
        wells, features, ["treatment"], wells["treatment"] == "neg",
        null_size=5000, seed=0)
    ref = assign_reference_index(wells, "treatment == 'neg'")
    column = "Metadata_Reference_Index"
    theirs = cmap.average_precision(
        ref[["treatment", column, "plateID", "wellID"]],
        ref[features].to_numpy(), ["treatment", column], [], [],
        ["treatment", column], progress_bar=False)
    theirs = theirs.query("treatment != 'neg'")
    joined = ours_ap.merge(theirs, on=["plateID", "wellID"])
    assert np.allclose(joined["average_precision_x"],
                       joined["average_precision_y"])
    mapped = cmap.mean_average_precision(theirs, ["treatment"],
                                         null_size=5000, threshold=0.05,
                                         seed=0, progress_bar=False)
    both = ours_map.merge(mapped, on="treatment")
    assert np.allclose(both["mean_average_precision_x"],
                       both["mean_average_precision_y"])
    assert (both["p_value_x"] == both["p_value_y"]).all()
