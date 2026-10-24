"""Annotation-group comparisons and representative-image definitions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.sp_stats import (
    _compare_image_groups,
    _image_comparison_figure,
    _image_groups,
    _image_measurement_columns,
    _representative_images,
    _write_image_comparison,
)


def _design(seed=0, per_well=4):
    rng = np.random.default_rng(seed)
    rows = []
    shift = {("HeLa", "WT"): 0.0, ("HeLa", "Ku80"): 5.0, ("HFF", "WT"): 10.0, ("HFF", "Ku80"): 10.0}
    well = 0
    for replicate in ("1", "2", "3"):
        for (cell, pathogen), offset in shift.items():
            well += 1
            for field in range(per_well):
                rows.append(dict(plate="plate1", well=f"W{well:02d}", field=str(field + 1),
                                 cell_type=cell, pathogen_type=pathogen, replicate=replicate,
                                 status="ok",
                                 channel_1_intensity_mean=100 + offset + rng.normal(0, 1),
                                 channel_1_focus_laplacian_variance=50 + rng.normal(0, 1),
                                 channel_1_saturation_fraction=0.0,
                                 channel_1_saturation_level=65535.0,
                                 channel_2_objects_count=float(rng.integers(5, 9))))
    return pd.DataFrame(rows)


def test_measurement_columns_exclude_identity_and_settings():
    names = _image_measurement_columns(_design())
    assert names == ["channel_1_intensity_mean", "channel_1_focus_laplacian_variance",
                     "channel_1_saturation_fraction", "channel_2_objects_count"]


def test_every_combination_of_dimensions_is_a_group():
    frame = _design()
    [only] = _image_groups(frame, ["replicate"])
    assert list(only["groups"]) == ["1", "2", "3"]
    assert [len(index) for index in only["groups"].values()] == [16, 16, 16]
    [combined] = _image_groups(frame, ["cell_type", "pathogen_type"])
    assert list(combined["groups"]) == ["HFF + Ku80", "HFF + WT", "HeLa + Ku80", "HeLa + WT"]


def test_levels_explicit_groups_and_strata():
    frame = _design()
    strata = _image_groups(frame, ["cell_type"], within=["replicate"])
    assert [entry["stratum"] for entry in strata] == [{"replicate": "1"}, {"replicate": "2"},
                                                      {"replicate": "3"}]
    assert all(list(entry["groups"]) == ["HFF", "HeLa"] for entry in strata)
    [explicit] = _image_groups(frame, [], groups=[
        dict(cell_type="HeLa", pathogen_type="WT"), dict(cell_type="HeLa", pathogen_type="Ku80")])
    assert list(explicit["groups"]) == ["HeLa + WT", "HeLa + Ku80"]
    assert all((frame.loc[explicit["groups"]["HeLa + Ku80"], "pathogen_type"] == "Ku80"))
    [kept] = _image_groups(frame, ["replicate"], levels={"replicate": ["1", "3"]})
    assert list(kept["groups"]) == ["1", "3"]
    with pytest.raises(ValueError):
        _image_groups(frame, ["missing"])
    with pytest.raises(ValueError):
        _image_groups(frame, ["replicate"], within=["replicate"])
    with pytest.raises(ValueError):
        _image_groups(frame, [], groups=[dict(cell_type="HeLa"), dict(cell_type="HeLa")])


def test_unannotated_images_are_left_out():
    frame = _design()
    frame.loc[:3, "cell_type"] = None
    frame.loc[4:5, "cell_type"] = ""
    [entry] = _image_groups(frame, ["cell_type"])
    assert sum(len(index) for index in entry["groups"].values()) == len(frame) - 6


def test_comparison_detects_real_differences_and_corrects_p_values():
    frame = _design()
    result = _compare_image_groups(frame, ["channel_1_intensity_mean"],
                                   ["cell_type", "pathogen_type"], correction="holm")
    summary, tests = result["summary"], result["tests"]
    assert set(summary["group"]) == {"HFF + Ku80", "HFF + WT", "HeLa + Ku80", "HeLa + WT"}
    hela = summary.set_index("group").loc["HeLa + WT"]
    assert hela["n"] == 12 and hela["mean"] == pytest.approx(100, abs=1)
    assert list(tests["stage"]).count("omnibus") == 1
    assert list(tests["stage"]).count("pairwise") == 6
    pairs = tests.set_index("groups")
    assert pairs.loc["HFF + Ku80 vs HeLa + WT", "p_adjusted"] < 0.001
    assert pairs.loc["HFF + Ku80 vs HFF + WT", "p_adjusted"] > 0.05
    assert (tests["p_adjusted"] >= tests["p_value"] - 1e-15).all()
    assert set(tests["correction"]) == {"holm"}


def test_well_unit_averages_fields_before_testing():
    frame = _design()
    result = _compare_image_groups(frame, ["channel_1_intensity_mean"], ["cell_type"], unit="well")
    summary = result["summary"].set_index("group")
    assert summary.loc["HeLa", "n"] == 6 and summary.loc["HeLa", "images"] == 24
    assert set(result["tests"]["unit"]) == {"well"}


def test_strata_and_families():
    frame = _design()
    per_replicate = _compare_image_groups(
        frame, ["channel_1_intensity_mean", "channel_2_objects_count"], ["cell_type"],
        within=["replicate"], family="measurement")
    tests = per_replicate["tests"]
    assert set(tests["stratum"]) == {"replicate=1", "replicate=2", "replicate=3"}
    assert len(tests) == 6
    together = _compare_image_groups(
        frame, ["channel_1_intensity_mean", "channel_2_objects_count"], ["cell_type"],
        within=["replicate"], family="all")["tests"]
    keys = ["stratum", "measurement", "groups"]
    paired = tests.merge(together, on=keys, suffixes=("_each", "_all"))
    assert len(paired) == 6
    assert (paired["p_adjusted_all"] >= paired["p_adjusted_each"] - 1e-12).all()
    assert (paired["p_value_all"] == paired["p_value_each"]).all()
    single = _compare_image_groups(frame, ["channel_1_intensity_mean"], ["cell_type"])["tests"]
    assert single["correction"].iloc[0] == "none (single test)"
    assert single["p_adjusted"].iloc[0] == single["p_value"].iloc[0]


def test_too_small_groups_get_a_summary_but_no_test():
    frame = _design().iloc[:5]
    result = _compare_image_groups(frame, ["channel_1_intensity_mean"], ["cell_type", "pathogen_type"])
    assert len(result["summary"]) == 2
    assert result["tests"].empty


@pytest.mark.parametrize("kind", ["box", "violin", "strip"])
def test_figures_are_drawn_styled_and_saved(tmp_path, kind):
    frame = _design()
    figure, saved = _image_comparison_figure(frame, "channel_1_intensity_mean",
                                             ["cell_type", "pathogen_type"], kind=kind,
                                             within=["replicate"] if kind == "strip" else None,
                                             path=tmp_path / f"cmp_{kind}.pdf")
    try:
        assert len(figure.axes) == (3 if kind == "strip" else 1)
        assert getattr(figure, "_spacr_graph_kind") == kind
        if kind == "box":
            texts = [text.get_text() for text in figure.axes[0].texts]
            assert any(text.startswith("*") for text in texts)
            assert "Kruskal-Wallis" in figure.axes[0].get_title() or "ANOVA" in figure.axes[0].get_title()
        assert saved and __import__("os").path.getsize(saved) > 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
    with pytest.raises(ValueError):
        _image_comparison_figure(frame, "channel_1_intensity_mean", ["cell_type"], kind="pie")


def test_comparison_tables_are_written(tmp_path):
    result = _compare_image_groups(_design(), ["channel_1_intensity_mean"], ["replicate"])
    paths = _write_image_comparison(result, tmp_path / "results")
    assert pd.read_csv(paths["summary"]).shape[0] == 3
    assert "p_adjusted" in pd.read_csv(paths["tests"]).columns


def _cloud():
    rng = np.random.default_rng(4)
    rows = []
    for label, centre in (("A", 0.0), ("B", 20.0)):
        for index in range(31):
            rows.append(dict(plate="p", well=label, field=str(index), cell_type=label, status="ok",
                             channel_1_x=centre + rng.normal(0, 1),
                             channel_1_y=centre + rng.normal(0, 1),
                             channel_1_focus_laplacian_variance=100 + rng.normal(0, 3),
                             channel_1_saturation_fraction=0.0))
    frame = pd.DataFrame(rows)
    frame.loc[0, ["channel_1_x", "channel_1_y"]] = [0.0, 0.0]
    return frame


def test_median_definition_ranks_the_image_at_the_class_median_first():
    frame = _cloud()
    others = frame.loc[1:30, ["channel_1_x", "channel_1_y"]]
    lower = others.sort_values("channel_1_x")["channel_1_x"].iloc[14]
    upper = others.sort_values("channel_1_y")["channel_1_y"].iloc[15]
    frame.loc[0, ["channel_1_x", "channel_1_y"]] = [
        (lower + others["channel_1_x"].sort_values().iloc[15]) / 2,
        (others["channel_1_y"].sort_values().iloc[14] + upper) / 2]
    ranked = _representative_images(frame, ["cell_type"], method="median",
                                    measurements=["channel_1_x", "channel_1_y"], top=3)
    first = ranked[(ranked["cell_type"] == "A") & (ranked["rank"] == 1)].iloc[0]
    assert first["row_index"] == 0 and first["distance"] == pytest.approx(0.0, abs=1e-9)
    assert list(ranked.groupby("cell_type").size()) == [3, 3]
    assert ranked.groupby("cell_type")["distance"].apply(lambda d: d.is_monotonic_increasing).all()
    assert {"plate", "well", "field", "method", "class_size"} <= set(ranked.columns)


def test_medoid_is_the_member_with_least_mean_distance():
    frame = pd.DataFrame(dict(cell_type="A", status="ok", plate="p", well="w",
                              field=list("abcde"), channel_1_x=[0.0, 1.0, 2.0, 3.0, 10.0]))
    ranked = _representative_images(frame, ["cell_type"], method="medoid",
                                    measurements=["channel_1_x"], top=5)
    assert ranked.iloc[0]["field"] == "c"
    scaled = (frame["channel_1_x"] - 2.0) / (1.4826 * 1.0)
    expected = np.mean([abs(scaled[2] - value) for value in scaled.drop(2)])
    assert ranked.iloc[0]["distance"] == pytest.approx(expected)


def test_pca_centroid_and_single_measurement_definitions():
    frame = _cloud()
    pca = _representative_images(frame, ["cell_type"], method="pca_centroid",
                                 measurements=["channel_1_x", "channel_1_y"], top=1)
    assert len(pca) == 2 and (pca["distance"] >= 0).all()
    one = _representative_images(frame, ["cell_type"], method="measurement",
                                 measurement="channel_1_x", top=1)
    a = frame[frame["cell_type"] == "A"]["channel_1_x"]
    best = (a - a.median()).abs().min()
    assert one[one["cell_type"] == "A"].iloc[0]["distance"] == pytest.approx(best)
    with pytest.raises(ValueError):
        _representative_images(frame, ["cell_type"], method="measurement")
    with pytest.raises(ValueError):
        _representative_images(frame, ["cell_type"], method="nearest")


def test_typical_in_focus_excludes_blurred_saturated_and_outlying_images():
    frame = _cloud()
    centre = frame.loc[:30, ["channel_1_x", "channel_1_y"]].median()
    for row in (1, 2, 3):
        frame.loc[row, ["channel_1_x", "channel_1_y"]] = centre.to_numpy()
    frame.loc[1, "channel_1_focus_laplacian_variance"] = 5.0
    frame.loc[2, "channel_1_saturation_fraction"] = 0.2
    frame.loc[4, "channel_1_x"] = 400.0
    frame.loc[5, "status"] = "error"
    plain = _representative_images(frame, ["cell_type"], method="median",
                                   measurements=["channel_1_x", "channel_1_y"], top=3)
    assert {1, 2} & set(plain[plain["cell_type"] == "A"]["row_index"])
    typical = _representative_images(frame, ["cell_type"], method="typical_in_focus",
                                     measurements=["channel_1_x", "channel_1_y"], top=3)
    chosen = set(typical[typical["cell_type"] == "A"]["row_index"])
    assert 3 in chosen and not {1, 2, 4, 5} & chosen
    reasons = dict(typical.attrs["excluded"])
    assert reasons[1].startswith("blurred") and reasons[2].startswith("saturated")
    assert reasons[4] == "measurement outlier" and reasons[5] == "measurement failed"


def test_boolean_columns_are_not_measurements():
    frame = _design()
    frame["channel_1_flag"] = True
    assert "channel_1_flag" not in _image_measurement_columns(frame)


def test_invalid_comparison_requests_are_refused():
    frame = _design()
    with pytest.raises(ValueError):
        _image_groups(frame, [])
    with pytest.raises(ValueError):
        _image_groups(frame, ["replicate"], levels={"missing": ["1"]})
    with pytest.raises(ValueError):
        _image_groups(frame, [], groups=[dict(missing="x")])
    with pytest.raises(ValueError):
        _compare_image_groups(frame, ["channel_1_intensity_mean"], ["replicate"], unit="plate")
    with pytest.raises(ValueError):
        _compare_image_groups(frame, ["channel_1_intensity_mean"], ["replicate"], family="every")
    with pytest.raises(ValueError):
        _compare_image_groups(frame, ["missing"], ["replicate"])


def test_omnibus_only_when_pairwise_is_off():
    tests = _compare_image_groups(_design(), ["channel_1_intensity_mean"], ["replicate"],
                                  pairwise=False)["tests"]
    assert list(tests["stage"]) == ["omnibus"]


def test_figure_reuses_given_tests_skips_foreign_pairs_and_tiny_violins():
    frame = _design()
    tests = _compare_image_groups(frame, ["channel_1_intensity_mean"], ["cell_type"])["tests"].copy()
    foreign = tests.iloc[[0]].copy()
    foreign["groups"] = "X vs Y"
    foreign["p_adjusted"] = 0.0
    given = pd.concat([tests, foreign], ignore_index=True)
    figure, saved = _image_comparison_figure(frame, "channel_1_intensity_mean", ["cell_type"],
                                             tests=given)
    try:
        assert saved is None
        assert figure.axes[0].get_title() == ""
        assert [text.get_text() for text in figure.axes[0].texts] == ["****"]
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)
    tiny = frame.iloc[[0, 12]]
    figure, _ = _image_comparison_figure(tiny, "channel_1_intensity_mean", ["pathogen_type"],
                                         kind="violin")
    try:
        assert not figure.axes[0].collections[0].get_offsets().size == 0
    finally:
        import matplotlib.pyplot as plt
        plt.close(figure)


def test_representatives_refuse_unknown_columns_and_flat_measurements():
    frame = _cloud()
    with pytest.raises(ValueError):
        _representative_images(frame, ["missing"])
    with pytest.raises(ValueError):
        _representative_images(frame[["cell_type", "status"]], ["cell_type"])
    frame["channel_1_constant"] = 1.0
    with pytest.raises(ValueError, match="do not vary"):
        _representative_images(frame, ["cell_type"], measurements=["channel_1_constant"])


def test_typical_in_focus_without_status_focus_or_enough_members():
    frame = pd.DataFrame(dict(cell_type=["A", "A", "B"], plate="p", well="w", field=["1", "2", "3"],
                              channel_1_x=[1.0, 2.0, 30.0]))
    ranked = _representative_images(frame, ["cell_type"], method="typical_in_focus", top=2)
    assert list(ranked["cell_type"]) == ["A", "A", "B"]
    assert ranked.attrs["excluded"] == []
    frame["channel_1_focus_laplacian_variance"] = [10.0, 10.0, 10.0]
    frame["status"] = ["ok", "ok", "error"]
    ranked = _representative_images(frame, ["cell_type"], method="typical_in_focus")
    assert set(ranked["cell_type"]) == {"A"}
    assert ranked.attrs["excluded"] == [(2, "measurement failed")]
