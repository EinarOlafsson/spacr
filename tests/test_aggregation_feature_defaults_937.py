"""Item 937: a default aggregation for every measured feature.

``spacr.merge_tables.SUMMARY_AGGREGATION_RULES`` gives every column of a
per-cell table its own method when cells are summarised per well, guide or
gene, and ``PER_FEATURE_AGGREGATION_RULES`` corrects the child -> parent
roll-up. The tests check each family's method on planted values, that the
legacy paths are unchanged, that every feature spaCR's measure and the OPS
phenotype path write has a rule, that old settings files rerun as before,
and that missing columns are tolerated.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import merge_tables as mt
from spacr.gene_measurement_compare import aggregate_genes, aggregate_guides


def _cells(n=1000, seed=0):
    """A synthetic per-cell table of two wells with one column per family."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "plate": "p1",
        "well": np.repeat(["A1", "A2"], n // 2),
        "object_label": np.arange(n),
        "cell_area": rng.gamma(2.0, 100.0, n),
        "cell_channel_0_mean_intensity": rng.lognormal(5.0, 1.0, n),
        "cell_channel_0_haralick_contrast": rng.random(n),
        "cell_channel_0_channel_1_pearson_correlation": rng.uniform(-1, 1, n),
        "cell_eccentricity": rng.random(n),
        "cell_channel_0_puncta_area_fraction": rng.random(n),
        "is_bystander": np.tile([True, True, True] + [False] * 7, n // 10),
        "n_pathogens": np.tile([0, 1, 2], n // 3 + 1)[:n],
        "cell_neighbors_within_50": rng.integers(0, 6, n),
        "cell_orientation": rng.uniform(-np.pi / 2, np.pi / 2, n),
        "cell_centroid-0": rng.random(n) * 1000,
        "cell_bbox-0": rng.integers(0, 1000, n),
        "measurement_units": "px",
    })


def test_legacy_rules_are_kept_verbatim_and_stay_the_default():
    assert mt.AGGREGATION_RULES_V1 is mt.AGGREGATION_RULES
    assert mt.PER_FEATURE_AGGREGATION_RULES[-len(mt.AGGREGATION_RULES):] == mt.AGGREGATION_RULES
    assert mt.MergePolicy().defaults == "legacy"
    names = ["area", "puncta_area_fraction", "n_pathogens", "orientation",
             "area_bbox", "channel_0_mean_intensity", "object_label",
             "is_bystander", "haralick_contrast", "euler_number"]
    for name in names:
        expected = next((how for pattern, how in mt.AGGREGATION_RULES
                         if __import__("re").search(pattern, name)),
                        mt.DEFAULT_AGGREGATION)
        assert mt.aggregation_for(name) == expected


def test_per_feature_roll_up_corrects_the_five_wrong_rules():
    for_ = mt.aggregation_for
    assert for_("puncta_area_fraction") == mt.SUM
    assert for_("puncta_area_fraction", defaults="per_feature") == mt.MEAN
    assert for_("n_pathogens", defaults="per_feature") == mt.SUM
    assert for_("orientation", defaults="per_feature") == mt.CIRCMEAN_AXIAL
    assert for_("area_bbox", defaults="per_feature") == mt.MAX
    assert for_("is_bystander", defaults="per_feature") == mt.MEAN
    assert for_("area", defaults="per_feature") == mt.SUM
    assert for_("n_z", defaults="per_feature") == mt.FIRST
    with pytest.raises(mt.MergeError):
        for_("area", defaults="newest")
    with pytest.raises(mt.MergeError):
        mt.MergePolicy(defaults="newest")


def test_roll_up_legacy_output_is_unchanged_and_per_feature_fixes_it():
    child = pd.DataFrame({
        "plateID": "p", "rowID": "r1", "columnID": "c1", "fieldID": "f1",
        "cell_id": [1, 1, 2, 2],
        "object_label": [1, 2, 3, 4],
        "area": [10.0, 20.0, 30.0, 40.0],
        "puncta_area_fraction": [0.2, 0.4, 0.6, 0.8],
        "orientation": [np.pi / 2 - 0.01, -(np.pi / 2 - 0.01), 0.1, 0.3],
    })
    keys = ["plateID", "rowID", "columnID", "fieldID", "cell_id"]
    legacy = mt.roll_up(child, keys, name="pathogen", policy=mt.MergePolicy())
    assert legacy["pathogen_puncta_area_fraction"].tolist() == [0.6000000000000001, 1.4]
    assert legacy["pathogen_orientation"].iloc[0] == pytest.approx(0.0)
    fixed = mt.roll_up(child, keys, name="pathogen",
                       policy=mt.MergePolicy(defaults="per_feature"))
    assert fixed["pathogen_puncta_area_fraction"].tolist() == pytest.approx([0.3, 0.7])
    assert abs(fixed["pathogen_orientation"].iloc[0]) == pytest.approx(np.pi / 2)
    assert fixed["pathogen_orientation"].iloc[1] == pytest.approx(0.2, abs=1e-12)
    assert fixed["pathogen_area"].tolist() == legacy["pathogen_area"].tolist()


def test_flags_counts_angles_fractions_and_positions_on_planted_cells():
    table = mt.aggregate_by_feature(_cells(), ["plate", "well"])
    assert table["n_cells"].tolist() == [500, 500]
    assert table["is_bystander_fraction"].tolist() == pytest.approx([0.3, 0.3])
    assert "is_bystander" not in table
    counts = mt.aggregate_by_feature(pd.DataFrame({"well": "A1", "n_vacuoles": [0, 1, 2]}), "well")
    assert counts["n_vacuoles"].iloc[0] == 1.0
    assert counts["n_vacuoles_sum"].iloc[0] == 3
    assert "cell_neighbors_within_50_sum" not in table
    assert not any(c.startswith(("cell_centroid", "cell_bbox", "object_label")) for c in table)
    cells = _cells()
    for well, row in table.set_index("well").iterrows():
        sub = cells[cells["well"] == well]
        assert row["cell_eccentricity"] == pytest.approx(sub["cell_eccentricity"].mean())
        assert row["cell_channel_0_puncta_area_fraction"] == pytest.approx(
            sub["cell_channel_0_puncta_area_fraction"].mean())
        assert row["cell_area"] == pytest.approx(sub["cell_area"].median())
        assert row["cell_channel_0_channel_1_pearson_correlation"] == pytest.approx(
            sub["cell_channel_0_channel_1_pearson_correlation"].median())
        assert row["cell_orientation_coherence"] < 0.15
    assert table["measurement_units"].tolist() == ["px", "px"]


def test_axial_circular_mean_does_not_wrap():
    angle, coherence = mt.circmean_axial([np.pi / 2 - 0.01, -(np.pi / 2 - 0.01)])
    assert abs(angle) == pytest.approx(np.pi / 2)
    assert coherence == pytest.approx(np.cos(0.02))
    rng = np.random.default_rng(1)
    _angle, spread = mt.circmean_axial(rng.uniform(-np.pi / 2, np.pi / 2, 20000))
    assert spread < 0.03
    assert np.isnan(mt.circmean_axial([np.nan])[0])
    assert mt.circmean_axial([0.3, 0.3, np.nan]) == pytest.approx((0.3, 1.0))


def test_a_stamp_that_differs_within_a_group_stops_the_summary():
    frame = pd.DataFrame({"well": ["A1", "A1"], "measurement_units": ["px", "um"],
                          "area": [1.0, 2.0]})
    with pytest.raises(mt.MergeError, match="measurement_units"):
        mt.aggregate_by_feature(frame, "well")


def test_overrides_fisher_z_and_trimmed_mean():
    frame = pd.DataFrame({"well": "A1", "x_spearman": [0.5, 0.9, 1.0],
                          "cell_area": [1.0, 2.0, 100.0] })
    out = mt.aggregate_by_feature(frame, "well", overrides={
        "spearman": "fisher_z_mean", "cell_area": "trimmed_mean"})
    clipped = np.clip([0.5, 0.9, 1.0], -0.999, 0.999)
    assert out["x_spearman"].iloc[0] == pytest.approx(np.tanh(np.arctanh(clipped).mean()))
    assert out["cell_area"].iloc[0] == pytest.approx(np.mean([1.0, 2.0, 100.0]))
    plan = pd.DataFrame(out.attrs["aggregation_plan"]).set_index("column")
    assert plan.loc["x_spearman", "rule"] == "override:spearman"
    with pytest.raises(mt.MergeError, match="not a summary aggregation"):
        mt.aggregate_by_feature(frame, "well", overrides={"cell_area": "mode"})
    kept = mt.aggregate_by_feature(_cells(), "well", drop_positions=False)
    assert "cell_centroid-0" in kept


def test_missing_columns_and_empty_tables_are_tolerated():
    cells = _cells()
    out = mt.aggregate_by_feature(cells, "well", columns=["cell_area", "not_here"])
    assert list(out.columns) == ["well", "n_cells", "cell_area"]
    empty = mt.aggregate_by_feature(cells.iloc[:0], "well")
    assert len(empty) == 0 and "n_cells" in empty
    plan = mt.summary_aggregation_plan(cells.iloc[:0])
    assert set(plan["column"]) == set(cells.columns)
    guides = cells.assign(guide=np.tile(["g1", "g2", "g3", "g4"], 250),
                          gene=np.tile(["G1", "G1", "G2", "G2"], 250))
    table = aggregate_guides(guides, ["cell_area", "is_bystander"], method="per_feature")
    genes = aggregate_genes(table.drop(columns="is_bystander_fraction"),
                            ["cell_area", "is_bystander"], method="per_feature")
    assert "is_bystander_fraction" not in genes and "cell_area" in genes


def test_guide_and_gene_profiles_stay_the_median_by_default():
    cells = _cells().assign(guide=np.tile(["g1", "g2", "g3", "g4"], 250),
                            gene=np.tile(["G1", "G1", "G2", "G2"], 250))
    features = ["cell_area", "cell_eccentricity", "n_pathogens", "is_bystander"]
    guides = aggregate_guides(cells, features)
    expected = cells.groupby(["guide", "gene"])[features].median()
    assert np.array_equal(guides[features].to_numpy(), expected.to_numpy())
    genes = aggregate_genes(guides, features)
    assert np.array_equal(genes[features].to_numpy(),
                          guides.groupby("gene")[features].median().to_numpy())
    by_cell = aggregate_genes(guides, features, level="cells", cells=cells)
    assert np.array_equal(by_cell[features].to_numpy(),
                          cells.groupby("gene")[features].median().to_numpy())
    with pytest.raises(ValueError, match="method"):
        aggregate_guides(cells, features, method="mean")


def test_guide_and_gene_per_feature_profiles():
    cells = _cells().assign(guide=np.tile(["g1", "g2", "g3", "g4"], 250),
                            gene=np.tile(["G1", "G1", "G2", "G2"], 250))
    features = ["cell_area", "cell_eccentricity", "n_pathogens", "is_bystander",
                "cell_orientation"]
    guides = aggregate_guides(cells, features, method="per_feature")
    assert {"n_pathogens_sum", "is_bystander_fraction",
            "cell_orientation_coherence"} <= set(guides.columns)
    by_guide = cells.groupby(["guide", "gene"])
    assert guides["cell_eccentricity"].to_numpy() == pytest.approx(
        by_guide["cell_eccentricity"].mean().to_numpy())
    genes = aggregate_genes(guides, features, method="per_feature")
    assert genes["n_pathogens_sum"].tolist() == guides.groupby("gene")["n_pathogens_sum"].sum().tolist()
    assert genes["cell_area"].to_numpy() == pytest.approx(
        guides.groupby("gene")["cell_area"].median().to_numpy())
    assert genes["n_cells"].tolist() == [500, 500]
    pooled = aggregate_genes(guides, features, level="cells", cells=cells,
                             method="per_feature")
    assert pooled["is_bystander_fraction"].tolist() == pytest.approx([0.3, 0.3], abs=0.01)


def _every_measure_column():
    """Columns of the OPS phenotype path with every family on, and spaCR's measure names."""
    from spacr.ops_sample import measure_phenotype_field

    rng = np.random.default_rng(0)
    cells = np.zeros((96, 96), np.int32)
    nuclei = np.zeros_like(cells)
    rr, cc = np.indices(cells.shape)
    for label, (r, c) in enumerate([(24, 24), (24, 70), (70, 24), (70, 70)], 1):
        cells[(rr - r) ** 2 + (cc - c) ** 2 <= 16 ** 2] = label
        nuclei[(rr - r) ** 2 + (cc - c) ** 2 <= 7 ** 2] = label
    image = rng.gamma(4.0, 50.0, size=cells.shape + (2,))
    import importlib.util

    zernike = importlib.util.find_spec("mahotas") is not None
    table = measure_phenotype_field(nuclei, cells, image, ["DAPI", "GFP"],
                                    funk_features=True, zernike=zernike,
                                    profile_features="all",
                                    granularity_subsample=0.5)
    spacr_measure = [
        "cell_area", "cell_area_bbox", "cell_convex_area", "cell_perimeter",
        "cell_eccentricity", "cell_solidity", "cell_extent", "cell_euler_number",
        "cell_orientation", "cell_centroid-0", "cell_centroid-1", "cell_bbox-0",
        "cell_channel_0_centroid_weighted-0", "cell_channel_0_mean_intensity",
        "cell_channel_0_integrated_intensity", "cell_channel_0_percentile_95",
        "cell_channel_0_outside_mean", "cell_channel_0_periphery_percentile_50",
        "cell_channel_0_channel_1_pearson_correlation",
        "cell_channel_0_channel_1_M1_correlation_95",
        "cell_channel_0_channel_1_manders_m1", "cell_channel_0_homogeneity_distance_8",
        "cell_skeleton_length", "cell_skeleton_branch_points", "n_pathogens",
        "n_vacuoles", "is_bystander", "is_distal", "touches_field_edge",
        "distance_to_field_edge", "cell_nearest_neighbor_distance",
        "cell_neighbors_within_50", "cell_touching_neighbors", "cell_percent_touching",
        "measurement_ndim", "measurement_units", "n_z", "voxel_size_z_um",
        "prcfo", "plateID", "rowID", "columnID", "fieldID", "file_name",
        "cell_volume_voxels", "pathogen_count", "cell_channel_0_puncta_count",
        "cell_channel_0_mass_displacement", "cell_correlation_channel_0_channel_1",
        "cell_channel_0_channel_1_pearson", "emb_0", "mitotic", "sgRNA", "gene",
    ]
    return list(table.columns) + spacr_measure


def test_every_feature_spacr_measures_has_a_rule():
    columns = _every_measure_column()
    frame = pd.DataFrame({name: pd.Series(dtype=float) for name in dict.fromkeys(columns)})
    plan = mt.summary_aggregation_plan(frame)
    unruled = plan.loc[plan["rule"] == "unruled", "column"].tolist()
    assert not unruled, f"columns without a summary rule: {unruled}"
    measured = plan.loc[~plan["method"].isin(["drop", "constant"])
                        & ~plan["column"].str.startswith("emb_"), "column"]
    roll = mt.aggregation_plan(frame[list(measured)], defaults="per_feature")
    named = [c for c in roll if roll[c] == mt.DEFAULT_AGGREGATION
             and not any(__import__("re").search(p, c.lower())
                         for p, _h in mt.PER_FEATURE_AGGREGATION_RULES)]
    stripped = [c for c in named if not c.startswith(("cell_", "nucleus_", "cytoplasm_"))]
    assert not stripped, f"roll-up columns falling to the default: {stripped}"


def test_the_ratchet_names_a_new_unruled_column():
    frame = pd.DataFrame({"cell_brand_new_feature": [1.0], "cell_area": [2.0]})
    plan = mt.summary_aggregation_plan(frame)
    assert plan.loc[plan["rule"] == "unruled", "column"].tolist() == ["cell_brand_new_feature"]


def test_io_join_and_roll_up_agree_under_per_feature(tmp_path):
    from spacr.io import _read_and_join_tables

    db = tmp_path / "measurements.db"
    ident = {"plateID": "p1", "rowID": "r1", "columnID": "c1", "fieldID": "f1"}
    cell = pd.DataFrame({**ident, "object_label": [1, 2], "prcf": "p1_r1_c1_f1",
                         "prcfo": ["p1_r1_c1_f1_o1", "p1_r1_c1_f1_o2"],
                         "cell_area": [100.0, 120.0]})
    pathogen = pd.DataFrame({**ident, "object_label": [1, 2, 3, 4],
                             "cell_id": [1, 1, 2, 2], "prcf": "p1_r1_c1_f1",
                             "prcfo": [f"p1_r1_c1_f1_o{i}" for i in range(1, 5)],
                             "pathogen_area": [10.0, 20.0, 30.0, 40.0],
                             "pathogen_puncta_area_fraction": [0.2, 0.4, 0.6, 0.8],
                             "pathogen_orientation": [1.56, -1.56, 0.1, 0.3]})
    with sqlite3.connect(db) as con:
        cell.to_sql("cell", con, index=False)
        pathogen.to_sql("pathogen", con, index=False)
    legacy = _read_and_join_tables(str(db), table_names=["cell", "pathogen"],
                                   require_crops=False)
    joined = _read_and_join_tables(str(db), table_names=["cell", "pathogen"],
                                   require_crops=False, aggregation_defaults="per_feature")
    rolled = mt.roll_up(pathogen, ["cell_id", "prcf"], name="pathogen",
                        policy=mt.MergePolicy(defaults="per_feature"))
    rolled = rolled.sort_values("cell_id").reset_index(drop=True)
    joined = joined.sort_values("object_label").reset_index(drop=True)
    for column in ("pathogen_area", "pathogen_puncta_area_fraction", "pathogen_orientation"):
        assert joined[column].to_numpy() == pytest.approx(rolled[column].to_numpy())
    assert legacy.sort_values("object_label")["pathogen_area"].tolist() == [15.0, 35.0]
    assert joined["pathogen_area"].tolist() == [30.0, 70.0]


def _plate(tmp_path):
    """One exported plate file for the screen aggregate."""
    frame = _cells().rename(columns={"object_label": "cell_label"})
    frame = frame.drop(columns=["measurement_units"])
    path = tmp_path / "p1.parquet"
    frame.to_parquet(path)
    return path, frame


def test_screen_well_mean_and_median_are_unchanged_and_well_default_is_added(tmp_path):
    from spacr.screen_staging import FEATURE_PREFIXES, aggregate_wells

    path, frame = _plate(tmp_path)
    record = aggregate_wells([path], tmp_path / "legacy")
    features = [c for c in frame.columns if c.startswith(FEATURE_PREFIXES)]
    means = pd.read_parquet(tmp_path / "legacy" / "wells_mean.parquet")
    expected = frame.groupby(["plate", "well"])[features].mean()
    assert np.array_equal(means[features].to_numpy(), expected.to_numpy())
    assert not (tmp_path / "legacy" / "wells_default.parquet").exists()
    assert len(record["paths"]) == 2
    aggregate_wells([path], tmp_path / "new", levels="well_default")
    default = pd.read_parquet(tmp_path / "new" / "wells_default.parquet")
    assert {"is_bystander_fraction", "n_pathogens", "n_pathogens_sum",
            "cell_orientation_coherence"} <= set(default.columns)
    assert "cell_label" not in default and "cell_centroid-0" not in default
    plan = pd.read_csv(tmp_path / "new" / "aggregation_plan.csv")
    assert {"column", "method", "rule", "reason", "family"} <= set(plan.columns)
    assert not default.isna().any().any()


def _settings_file(path, rows):
    """A two-column settings csv."""
    pd.DataFrame(rows, columns=["Key", "Value"]).to_csv(path, index=False)
    return str(path)


def test_old_screen_settings_files_rerun_legacy_and_new_ones_per_feature(tmp_path, caplog):
    import logging

    from spacr.cli import resolve_module, resolve_settings
    from spacr.screen_staging import _aggregate_levels, screen_defaults
    from spacr.settings import (DEFAULT_CHANGES, SETTINGS_SCHEMA_VERSION,
                                SETTINGS_VERSION_KEY)

    assert screen_defaults()["aggregation_defaults"] == "per_feature"
    row = [v for v, change in DEFAULT_CHANGES.items() if "aggregation_defaults" in change]
    assert len(row) == 1 and DEFAULT_CHANGES[row[0]] == {"aggregation_defaults": "legacy"}
    assert SETTINGS_SCHEMA_VERSION >= row[0] > 933
    screen = resolve_module("screen")
    with caplog.at_level(logging.WARNING, logger="spacr.settings"):
        old = resolve_settings(screen, _settings_file(tmp_path / "old.csv", [("src", "x")]))
    assert old["aggregation_defaults"] == "legacy"
    assert "aggregation_defaults='legacy'" in caplog.text
    stamped_old = resolve_settings(screen, _settings_file(
        tmp_path / "s933.csv", [(SETTINGS_VERSION_KEY, "933"), ("src", "x")]))
    assert stamped_old["aggregation_defaults"] == "legacy"
    new = resolve_settings(screen, _settings_file(
        tmp_path / "new.csv", [(SETTINGS_VERSION_KEY, str(row[0])), ("src", "x")]))
    assert new["aggregation_defaults"] == "per_feature"
    explicit = resolve_settings(screen, _settings_file(
        tmp_path / "explicit.csv", [("src", "x"), ("aggregation_defaults", "per_feature")]))
    assert explicit["aggregation_defaults"] == "per_feature"
    assert _aggregate_levels(old) == ["well_mean", "well_median"]
    assert _aggregate_levels(new) == ["well_mean", "well_median", "well_default"]
    with pytest.raises(ValueError, match="aggregation_defaults"):
        _aggregate_levels({"aggregate_levels": "well_mean", "aggregation_defaults": "x"})


def test_the_new_settings_are_alpha():
    from spacr.settings import ALPHA_FEATURES

    entry = ALPHA_FEATURES[937]["module_settings"]["screen"]
    assert set(entry) == {"aggregation_defaults", "aggregation_count_sum",
                          "aggregation_report"}
