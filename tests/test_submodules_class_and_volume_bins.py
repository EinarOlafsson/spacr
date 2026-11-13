"""Table-driven tests of the two contingency-table assays in ``spacr.submodules``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture:

* ``analyze_endodyogeny``: every parasite's log2 volume-doubling bin, the
  ordered bin labels (``max_bins`` tail, appended last edge, unit scaling),
  the condition filter and the chi-squared test on the bin counts;
* ``analyze_class_proportion``: the class x condition contingency and its
  chi-squared, the NaN -> class 0 fill and its warning, source handling,
  ``png_list`` handling and the saved statistics tables.

The endodyogeny cases run against small real ``measurements.db`` files; the
class-proportion cases against a real database or a recording fake of
``spacr.io._read_and_merge_data`` returning the same merged frame.  Every
chi-squared is checked against :func:`scipy.stats.chi2_contingency` on the
returned data.
"""
from __future__ import annotations

import os
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io  # noqa: E402
from spacr.submodules import analyze_class_proportion, analyze_endodyogeny  # noqa: E402


@pytest.fixture(autouse=True)
def _no_show_and_clean_figs(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    plt.close("all")
    yield
    plt.close("all")


def _case_dir(tmp_path, name):
    d = tmp_path / re.sub(r"[^A-Za-z0-9_]+", "_", name)
    d.mkdir()
    return d


def _chi2(data, group, column):
    counts = data.groupby([group, column], observed=True).size().unstack(fill_value=0)
    return chi2_contingency(counts)


# ---------------------------------------------------------------------------
# analyze_endodyogeny
# ---------------------------------------------------------------------------

# area ** 1.5 relative to 500 ** 1.5: ~600 -> bin 1, ~1000 -> 2, ~1600 -> 3, ~2600 -> 4
AREAS_C1 = [600.0, 620.0, 640.0, 1000.0, 1050.0, 1100.0, 1600.0, 1650.0, 1700.0,
            2600.0, 2700.0, 2800.0]
AREAS_C2 = [610.0, 1010.0, 1020.0, 1030.0, 1610.0, 1620.0, 1630.0, 1640.0, 2610.0,
            2620.0, 2630.0, 2640.0]
V0 = 500 ** 1.5


def _endo_db(root, areas_by_column):
    """cell + pathogen + png_list tables; one parasite per cell, class k % 2."""
    meas = os.path.join(str(root), "measurements")
    os.makedirs(meas, exist_ok=True)
    cells, pathogens, pngs = [], [], []
    obj = 0
    for column, areas in areas_by_column.items():
        for k, area in enumerate(areas):
            obj += 1
            row, field = f"r{k % 3 + 1}", f"f{k % 2 + 1}"
            base = {"plateID": "plate1", "rowID": row, "columnID": column, "fieldID": field,
                    "prcf": f"plate1_{row}_{column}_{field}"}
            cells.append({**base, "object_label": obj, "cell_area": 20000.0 + area,
                          "cell_channel_0_mean_intensity": 100.0 + k})
            pathogens.append({**base, "object_label": obj, "cell_id": obj,
                              "pathogen_area": float(area),
                              "pathogen_channel_0_mean_intensity": 50.0 + k})
            pngs.append({**base, "cell_id": f"o{obj}",
                         "png_path": f"/x/plate1_{row}_{column}_{field}_o{obj}.png",
                         "predictions": k % 2})
    with sqlite3.connect(os.path.join(meas, "measurements.db")) as con:
        for name, rows in (("cell", cells), ("pathogen", pathogens), ("png_list", pngs)):
            pd.DataFrame(rows).to_sql(name, con, index=False, if_exists="replace")
    return str(root)


def _endo_settings(src, **over):
    s = {"src": src, "tables": ["cell", "pathogen"], "cell_types": ["Hela"],
         "cell_plate_metadata": None, "pathogen_types": ["nc", "pc"],
         "pathogen_plate_metadata": [["c1"], ["c2"]], "treatments": None,
         "treatment_plate_metadata": None, "um_per_px": None, "min_area_bin": 500,
         "max_area": 10 ** 9, "level": "object", "nuclei_limit": 10, "pathogen_limit": 1,
         "verbose": False, "save": False}
    s.update(over)
    return s


def _bin(volume, v0=V0):
    return int(np.floor(np.log2(volume / v0))) + 1


def _labels(edges):
    return [f"{a:.2f}-{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]


STANDARD = {"c1": AREAS_C1, "c2": AREAS_C2}
FOUR = _labels([V0 * 2 ** i for i in range(5)])


def test_analyze_endodyogeny_bins_every_parasite_by_volume_doubling(subtests, tmp_path):
    """Each parasite's volume (area ** 1.5) lands in its exact doubling bin,
    the bin labels are ordered and exact, and the chi-squared on the
    condition x bin counts is scipy's."""
    cases = [
        # (id, areas by column, settings overrides, expected
        #  (areas kept, bin labels, conditions, group column))
        ("four bins, two conditions", STANDARD, {"verbose": True},
         (AREAS_C1 + AREAS_C2, FOUR, {"Hela_nc", "Hela_pc"}, "condition")),
        ("a list of sources", STANDARD, {"src_list": True, "save": True},
         (AREAS_C1 + AREAS_C2, FOUR, {"Hela_nc", "Hela_pc"}, "condition")),
        ("unmapped wells are dropped", {"c1": AREAS_C1, "c2": AREAS_C2, "c3": AREAS_C1},
         {"cell_types": None}, (AREAS_C1 + AREAS_C2, FOUR, {"nc", "pc"}, "condition")),
        ("a volume on the last edge stays in the top bin",
         {"c1": [600.0, 1000.0, 1600.0, 2000.0], "c2": [620.0, 1010.0, 1610.0, 2000.0]}, {},
         ([600.0, 1000.0, 1600.0, 2000.0, 620.0, 1010.0, 1610.0, 2000.0], FOUR,
          {"Hela_nc", "Hela_pc"}, "condition")),
        ("max_bins collapses the tail", STANDARD, {"max_bins": 2},
         (AREAS_C1 + AREAS_C2, [FOUR[0], f">{V0 * 2:.2f}"], {"Hela_nc", "Hela_pc"},
          "condition")),
        ("max_bins above the bin count changes nothing", STANDARD, {"max_bins": 99},
         (AREAS_C1 + AREAS_C2, FOUR, {"Hela_nc", "Hela_pc"}, "condition")),
        ("min_area_bin and max_area filter objects", STANDARD,
         {"min_area_bin": 1000, "max_area": 1700},
         ([a for a in AREAS_C1 + AREAS_C2 if 1000 <= a <= 1700],
          _labels([1000 ** 1.5 * 2 ** i for i in range(3)]), {"Hela_nc", "Hela_pc"},
          "condition")),
        ("group_by_class splits each condition by class", STANDARD,
         {"group_by_class": True, "class_column": "predictions"},
         (AREAS_C1 + AREAS_C2, FOUR, {"Hela_nc", "Hela_pc"}, "new_condition")),
        ("um_per_px rescales areas but not bins", STANDARD, {"um_per_px": 0.1},
         ([a * 0.01 for a in AREAS_C1 + AREAS_C2],
          _labels([(500 * 0.01) ** 1.5 * 2 ** i for i in range(5)]), {"Hela_nc", "Hela_pc"},
          "condition")),
        ("png_list already named is read once", STANDARD,
         {"tables": ["cell", "pathogen", "png_list"]},
         (AREAS_C1 + AREAS_C2, FOUR, {"Hela_nc", "Hela_pc"}, "condition")),
        ("an unknown group column lists the columns", STANDARD,
         {"group_column": "does_not_exist"}, (KeyError, "Available columns")),
        ("nothing above the first bin edge", {"c1": [500.0] * 4, "c2": [500.0] * 4}, {},
         (ValueError, "is not greater than")),
    ]
    for case_id, areas, over, expected in cases:
        with subtests.test(msg=case_id):
            src = _endo_db(_case_dir(tmp_path, case_id) / "plate1", areas)
            over = dict(over)
            as_list = over.pop("src_list", False)
            settings = _endo_settings([src] if as_list else src, **over)
            plt.close("all")
            if isinstance(expected[0], type):
                with pytest.raises(expected[0], match=expected[1]):
                    analyze_endodyogeny(settings)
                continue
            out = analyze_endodyogeny(settings)
            kept, labels, conditions, group = expected
            data = out["data"]
            assert set(out) == {"data", "chi_squared"}
            np.testing.assert_allclose(sorted(data["pathogen_area"]), sorted(kept))
            np.testing.assert_allclose(data["pathogen_volume"], data["pathogen_area"] ** 1.5)
            v0 = (settings["min_area_bin"] * (settings["um_per_px"] or 1) ** 2) ** 1.5
            n_cap = settings.get("max_bins") or len(labels)
            assert list(data["bin_index"]) == [min(_bin(v, v0), n_cap)
                                               for v in data["pathogen_volume"]]
            binned = data["pathogen_volume_bin"]
            assert binned.cat.ordered and not binned.isna().any()
            assert list(binned.cat.categories) == labels
            assert set(data["condition"]) == conditions
            if group == "new_condition":
                assert list(data["new_condition"]) == list(
                    data["condition"].astype(str) + data["predictions"].astype(str))
            chi2, p, dof, _ = _chi2(data, group, "pathogen_volume_bin")
            res = out["chi_squared"]
            assert {"chi_squared_stat", "p_value", "degrees_of_freedom", "test", "unit",
                    "n"} <= set(res.columns)
            assert res.loc[0, "unit"] == "object"
            assert res.loc[0, "chi_squared_stat"] == pytest.approx(chi2)
            assert res.loc[0, "p_value"] == pytest.approx(p)
            assert int(res.loc[0, "degrees_of_freedom"]) == dof == (
                (data[group].nunique() - 1) * (len(labels) - 1))
            legend = plt.gca().get_legend()
            unit = "µm³" if settings["um_per_px"] else "px³"
            assert legend.get_title().get_text() == f"Volume Range ({unit})"
            assert [t.get_text().split(":")[0] for t in legend.get_texts()] == [
                str(i) for i in range(1, len(labels) + 1)]
            assert plt.gca().get_ylim() == (0.0, 1.0)
            assert settings["tables"].count("png_list") == 1
            if case_id == "a list of sources":
                out_dir = os.path.join(src, "results", "analyze_endodyogeny")
                assert sorted(os.listdir(out_dir)) == [
                    "chi_squared_pairwise_results.csv", "chi_squared_results.csv",
                    "chi_squared_results.pdf", "data.csv"]
                with open(os.path.join(out_dir, "chi_squared_results.pdf"), "rb") as fh:
                    assert fh.read(4) == b"%PDF"
                saved = pd.read_csv(os.path.join(out_dir, "data.csv"))
                np.testing.assert_allclose(sorted(saved["pathogen_volume"]),
                                           sorted(data["pathogen_volume"]))
                saved_chi = pd.read_csv(os.path.join(out_dir, "chi_squared_results.csv"))
                assert saved_chi["chi_squared_stat"].iloc[0] == pytest.approx(chi2)
                assert len(pd.read_csv(os.path.join(out_dir,
                                                    "chi_squared_pairwise_results.csv"))) == 1

    with subtests.test(msg="plate level keeps the object chi-squared and adds plate rows"):
        src = _endo_db(_case_dir(tmp_path, "levels") / "plate1", STANDARD)
        obj = analyze_endodyogeny(_endo_settings(src, level="object"))
        plate = analyze_endodyogeny(_endo_settings(src, level="plate"))
        for column in ("chi_squared_stat", "p_value", "degrees_of_freedom"):
            assert obj["chi_squared"].loc[0, column] == pytest.approx(
                plate["chi_squared"].loc[0, column])
        assert plate["chi_squared"].loc[0, "unit"] == "object"
        assert set(plate["chi_squared"]["unit"].astype(str)) - {"object"} == {"plateID"}
        assert len(plate["data"]) == len(obj["data"])


# ---------------------------------------------------------------------------
# analyze_class_proportion
# ---------------------------------------------------------------------------

ROWS3, COLS3, FIELDS2, OBJECTS4 = ("r1", "r2", "r3"), ("c1", "c2", "c3"), ("f1", "f2"), (1, 2, 3, 4)
N_OBJECTS = 72
WEIGHTS = {"c1": (0.70, 0.20, 0.10), "c2": (0.20, 0.60, 0.20), "c3": (0.10, 0.25, 0.65)}
CONDITIONS = {"HeLa_nc", "HeLa_pc", "HeLa_tc"}


def _class_tables(seed=0):
    rng = np.random.default_rng(seed)
    cells, pngs = [], []
    for r in ROWS3:
        for c in COLS3:
            for f in FIELDS2:
                for obj in OBJECTS4:
                    prcf = f"plate1_{r}_{c}_{f}"
                    cells.append({"plateID": "plate1", "rowID": r, "columnID": c, "fieldID": f,
                                  "object_label": obj, "prcf": prcf, "prc": f"plate1_{r}_{c}",
                                  "cell_area": float(rng.integers(300, 4000)),
                                  "cell_channel_0_mean_intensity": float(rng.normal(1200, 150))})
                    pngs.append({"plateID": "plate1", "rowID": r, "columnID": c, "fieldID": f,
                                 "cell_id": f"o{obj}", "png_path": f"/pngs/{prcf}_o{obj}.png",
                                 "test": int(rng.choice([0, 1, 2], p=WEIGHTS[c]))})
    return pd.DataFrame(cells), pd.DataFrame(pngs)


def _class_db(src):
    os.makedirs(os.path.join(str(src), "measurements"), exist_ok=True)
    cells, pngs = _class_tables()
    with sqlite3.connect(os.path.join(str(src), "measurements", "measurements.db")) as con:
        cells.to_sql("cell", con, index=False)
        pngs.to_sql("png_list", con, index=False)


def _merged_frame(blank_class_2=False):
    """What ``_read_and_merge_data`` returns for the database above."""
    cells, pngs = _class_tables()
    merged = cells.copy()
    merged["object_label"] = [f"o{o}" for o in cells["object_label"]]
    merged["test"] = pngs["test"].astype(float).to_numpy()
    merged["png_path"] = pngs["png_path"].to_numpy()
    merged.index = pd.Index(merged["prcf"] + "_" + merged["object_label"], name="prcfo")
    if blank_class_2:
        merged.loc[merged["test"] == 2.0, "test"] = np.nan
    return merged


def _class_settings(src, **over):
    s = {"src": [str(p) for p in src] if isinstance(src, list) else str(src),
         "tables": ["cell"], "cell_types": ["HeLa"], "cell_plate_metadata": None,
         "pathogen_types": ["nc", "pc", "tc"], "pathogen_plate_metadata": [["c1"], ["c2"], ["c3"]],
         "treatments": None, "treatment_plate_metadata": None, "group_column": "condition",
         "class_column": "test", "nuclei_limit": 1000, "pathogen_limit": 1000, "level": "well",
         "save": False, "verbose": False}
    s.update(over)
    return s


def test_analyze_class_proportion_tests_classes_against_conditions(subtests, tmp_path, capsys):
    """The returned data is every object annotated with its condition, NaN
    classes count as class 0 (with a warning), and the chi-squared is scipy's
    on the class x condition counts; source lists, ``png_list`` and saving
    behave exactly."""
    frame = _merged_frame()
    n_blank = int((frame["test"] == 2.0).sum())
    n_zero = int((frame["test"] == 0.0).sum())
    cases = [
        # (id, real database?, merged frame, settings overrides, expectations)
        ("a real database", True, None, {}, {"tables": None, "dof": 4}),
        ("a real database, saved", True, None, {"save": True}, {"dof": 4, "saved": True}),
        ("object level gives the same chi-squared", False, frame, {"level": "object"},
         {"dof": 4, "bars": True}),
        ("a bare string source with forwarded limits", False, frame,
         {"verbose": True, "nuclei_limit": 7, "pathogen_limit": 3},
         {"dof": 4, "merge": {"verbose": True, "nuclei_limit": 7, "pathogen_limit": 3},
          "src": "one"}),
        ("a list of sources, results under the first", False, frame, {"save": True, "two": True},
         {"dof": 4, "src": "two"}),
        ("png_list is appended once", False, frame, {}, {"dof": 4, "tables": ["cell", "png_list"]}),
        ("png_list already named stays once", False, frame, {"tables": ["png_list", "cell"]},
         {"dof": 4, "tables": ["png_list", "cell"]}),
        ("NaN classes become class 0", False, _merged_frame(blank_class_2=True), {},
         {"dof": 2, "zeros": n_zero + n_blank,
          "printed": [f"{n_blank} of {N_OBJECTS} objects have no value", "counted as class 0",
                      "annotation rather than a classifier call"]}),
        ("an unknown group column lists the columns", False, frame,
         {"group_column": "treatment"}, {"raises": True}),
    ]
    for case_id, real, merged, over, expect in cases:
        with subtests.test(msg=case_id):
            folder = _case_dir(tmp_path, case_id)
            over = dict(over)
            src = [folder / "plateA", folder / "plateB"] if over.pop("two", False) else folder
            for p in (src if isinstance(src, list) else [src]):
                p.mkdir(exist_ok=True)
            if real:
                _class_db(folder)
            settings = _class_settings(src, **over)
            rec = {}
            plt.close("all")
            capsys.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                if not real:
                    def _fake(locs, tables, verbose=False, nuclei_limit=10, pathogen_limit=10,
                              **kw):
                        rec.update(locs=list(locs), tables=list(tables), verbose=verbose,
                                   nuclei_limit=nuclei_limit, pathogen_limit=pathogen_limit)
                        return merged.copy(), []
                    mp.setattr(spacr.io, "_read_and_merge_data", _fake)
                if expect.get("raises"):
                    with pytest.raises(KeyError):
                        analyze_class_proportion(settings)
                    printed = capsys.readouterr().out
                    assert "treatment not found in DataFrame, please choose from:" in printed
                    for col in ("condition", "columnID", "prc", "test"):
                        assert f"\n{col}\n" in printed
                    continue
                out = analyze_class_proportion(settings)
            printed = capsys.readouterr().out
            data, chi = out["data"], out["chi_squared"]
            assert set(out) == {"data", "chi_squared"}
            assert len(data) == N_OBJECTS
            assert set(data["condition"]) == CONDITIONS
            assert set(data["pathogen"]) == {"nc", "pc", "tc"}
            assert not data["test"].isna().any()
            chi2, p, dof, _ = _chi2(data, "condition", "test")
            assert (chi.loc[0, "test"], chi.loc[0, "unit"]) == ("chi-squared on object counts",
                                                                "object")
            assert chi.loc[0, "chi_squared_stat"] == pytest.approx(chi2)
            assert chi.loc[0, "p_value"] == pytest.approx(p) and p < 0.05
            assert int(chi.loc[0, "degrees_of_freedom"]) == dof == expect["dof"]
            if "zeros" in expect:
                assert int((data["test"] == 0.0).sum()) == expect["zeros"]
            for text in expect.get("printed", []):
                assert text in printed
            if "printed" not in expect:
                assert "have no value in" not in printed
            if expect.get("tables"):
                assert rec["tables"] == expect["tables"]
            if "merge" in expect:
                assert {k: rec[k] for k in expect["merge"]} == expect["merge"]
            if expect.get("src") == "one":
                assert settings["src"] == [str(folder)]
                assert rec["locs"] == [os.path.join(str(folder), "measurements", "measurements.db")]
            if expect.get("src") == "two":
                assert rec["locs"] == [os.path.join(str(p), "measurements", "measurements.db")
                                       for p in src]
                assert (src[0] / "results" / "analyze_class_proportion" /
                        "class_chi_squared_results.csv").is_file()
                assert not (src[1] / "results").exists()
            if not settings["save"]:
                assert not (folder / "results").exists()
            if expect.get("bars"):
                fignums = plt.get_fignums()
                assert len(fignums) == 2
                ax = plt.figure(fignums[0]).axes[0]
                assert ax.get_ylim() == (0.0, 1.0)
                assert [t.get_text() for t in ax.get_xticklabels()] == sorted(CONDITIONS)
                assert [len(c) for c in ax.containers] == [3, 3, 3]
            if expect.get("saved"):
                outdir = folder / "results" / "analyze_class_proportion"
                assert sorted(p.name for p in outdir.iterdir()) == sorted([
                    "class_chi_squared_results.csv", "class_frequency_test.csv",
                    "class_chi_squared_data.csv", "class_chi_squared.pdf", "class_heatmap.pdf",
                    "normality_results.csv", "variance_results.csv",
                    "statistical_test_results.csv", "posthoc_results.csv"])
                for pdf in ("class_chi_squared.pdf", "class_heatmap.pdf"):
                    assert (outdir / pdf).read_bytes()[:4] == b"%PDF"
                saved = pd.read_csv(outdir / "class_chi_squared_data.csv")
                assert len(saved) == N_OBJECTS and set(saved["condition"]) == CONDITIONS
                assert pd.read_csv(outdir / "class_chi_squared_results.csv")[
                    "chi_squared_stat"].iloc[0] == pytest.approx(chi2)
                pairwise = pd.read_csv(outdir / "class_frequency_test.csv")
                assert len(pairwise) == 3
                assert {"Group 1", "Group 2", "p-value", "p-value_adj"} <= set(pairwise.columns)
                assert pairwise["p-value"].between(0, 1).all()
                normality = pd.read_csv(outdir / "normality_results.csv")
                assert len(normality) == 3 and set(normality["Column"]) == {"test"}
                variance = pd.read_csv(outdir / "variance_results.csv")
                assert variance["Test Name"].tolist() == ["Levene's Test"]
                stats = pd.read_csv(outdir / "statistical_test_results.csv")
                assert stats["Groups"].tolist() == [3]
                assert stats["Test Name"].iloc[0] in ("One-way ANOVA", "Kruskal-Wallis test")
                posthoc = pd.read_csv(outdir / "posthoc_results.csv")
                assert set(posthoc["Comparison"]) == {"HeLa_nc vs HeLa_pc", "HeLa_nc vs HeLa_tc",
                                                      "HeLa_pc vs HeLa_tc"}
                assert posthoc["Adjusted p-value"].between(0, 1).all()
                assert set(posthoc["Test Name"]) == {"Dunn's Post-hoc"}
                assert set(posthoc["Adjusted Method"]) == {"holm"}
