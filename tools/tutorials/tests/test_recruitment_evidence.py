"""Independent tiny DB/CSV examples distinguish identity and scientific maths."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recruitment_evidence import inspect_results


# Explicit expected columns, independent of the evidence module's constants.
RATIOS = [
    "pathogen_cell_mean_mean", "pathogen_cytoplasm_mean_mean", "pathogen_nucleus_mean_mean",
    "pathogen_cell_q75_mean", "pathogen_cytoplasm_q75_mean", "pathogen_nucleus_q75_mean",
    "pathogen_outside_cell_mean_mean", "pathogen_outside_cytoplasm_mean_mean",
    "pathogen_outside_nucleus_mean_mean", "pathogen_outside_cell_q75_mean",
    "pathogen_outside_cytoplasm_q75_mean", "pathogen_outside_nucleus_q75_mean",
    "pathogen_periphery_cell_mean_mean", "pathogen_periphery_cytoplasm_mean_mean",
    "pathogen_periphery_nucleus_mean_mean", "recruitment",
]


def _csv(path, rows, *, duplicate_well_keys=False):
    names = list(rows[0])
    if duplicate_well_keys:
        names = ["plateID", "rowID", "columnID"] + names
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        writer.writerows([row[name] for name in names] for row in rows)


def _save_outputs(project, cells):
    _csv(project / "results" / "cells.csv", cells)
    nonnumeric = {"prcfo", "plateID", "rowID", "columnID", "fieldID", "object_label",
                  "prc", "file_name", "host_cells", "pathogen", "treatment", "condition"}
    well = {column: value if column in nonnumeric else sum(row[column] for row in cells) / len(cells)
            for column, value in cells[0].items()}
    _csv(project / "results" / "wells.csv", [well], duplicate_well_keys=True)


def _expected(field, cell_mean, cyto_mean, nucleus_mean, numerators, area, pathogen_count):
    """Input values are hand-derived per-cell aggregates, not read by app code."""
    p, q, outside, outside_q, boundary = numerators
    row = {
        "prcfo": f"private_r1_c1_{field}_o1", "plateID": "private", "rowID": "r1",
        "columnID": "c1", "fieldID": field, "object_label": "o1", "prc": "private_r1_c1",
        "file_name": f"private_r1_c1_{field}", "host_cells": "HeLa", "pathogen": "screen",
        "treatment": "demo", "condition": "HeLa_screen_demo",
        "cell_area": 5000.0, "nucleus_area": 1000.0, "cytoplasm_area": 4000.0,
        "pathogen_area": float(area), "cell_channel_1_mean_intensity": float(cell_mean),
        "cell_channel_1_percentile_95": 10.0,
        "nucleus_channel_1_mean_intensity": float(nucleus_mean),
        "cytoplasm_channel_1_mean_intensity": float(cyto_mean), "nucleus_prcfo_count": 1.0,
        "pathogen_prcfo_count": float(pathogen_count), "cells_per_well": 5.0,
        "pathogen_channel_1_mean_intensity": float(p),
        "pathogen_channel_1_percentile_75": float(q),
        "pathogen_channel_1_outside_mean": float(outside),
        "pathogen_channel_1_outside_percentile_75": float(outside_q),
        "pathogen_channel_1_periphery_mean": float(boundary),
    }
    # Explicit order: five numerator families crossed with three compartments.
    values = []
    for numerator in (p, q, outside, outside_q, boundary):
        for denominator in (cell_mean, cyto_mean, nucleus_mean):
            if denominator == 0:
                values.append(math.nan if numerator == 0 else math.copysign(math.inf, numerator))
            else:
                values.append(numerator / denominator)
    values.append(values[1])
    row.update(zip(RATIOS, values))
    for role in ("pathogen", "nucleus"):
        for channel in range(4):
            row[f"{role}_slope_channel_{channel}"] = 1.0
    return row


@pytest.fixture
def example(tmp_path):
    project = tmp_path / "private_subset"
    (project / "measurements").mkdir(parents=True)
    (project / "results").mkdir()
    settings = {
        "src": str(project), "channel_dims": [1], "channel_of_interest": 1,
        "cell_chann_dim": 1, "nucleus_chann_dim": 0, "pathogen_chann_dim": 3,
        "nuclei_limit": 1, "pathogen_limit": 10, "target_intensity_min": 1,
        "cells_per_well": 3, "cell_types": ["HeLa"], "cell_plate_metadata": None,
        "pathogen_types": ["screen"], "pathogen_plate_metadata": [["c1"]],
        "treatments": ["demo"], "treatment_plate_metadata": None,
        "cell_size_range": [3500, 100000], "nucleus_size_range": [700, 4000],
        "pathogen_size_range": [100, 1000], "cell_intensity_range": None,
        "nucleus_intensity_range": None, "pathogen_intensity_range": None,
    }
    database = project / "measurements" / "measurements.db"
    identity = 'plateID TEXT, rowID TEXT, columnID TEXT, fieldID TEXT, object_label INTEGER, file_name TEXT, prcf TEXT'
    with sqlite3.connect(database) as connection:
        for role in ("cell", "cytoplasm", "nucleus", "pathogen"):
            columns = f'{identity}, {role}_area REAL, {role}_channel_1_mean_intensity REAL'
            if role == "cell":
                columns += ', cell_channel_1_percentile_95 REAL'
            if role in ("nucleus", "pathogen"):
                columns += ', cell_id INTEGER'
            if role == "pathogen":
                columns += ', pathogen_channel_1_percentile_75 REAL, pathogen_channel_1_outside_mean REAL, pathogen_channel_1_outside_percentile_75 REAL, pathogen_channel_1_periphery_mean REAL'
            connection.execute(f'CREATE TABLE {role} ({columns})')

        def add(role, field, label, area, mean, *extra):
            stem = f"private_r1_c1_{field}"
            row = ("private", "r1", "c1", field, label, stem, stem, area, mean, *extra)
            connection.execute(f'INSERT INTO {role} VALUES ({",".join("?" for _ in row)})', row)

        # Repeated label 1 in f1 and f2 must be two distinct parent cells.
        for field, label, c, y in (("f1", 1, 7, 2), ("f2", 1, 3, 6),
                                    ("f1", 2, 7, 2), ("f1", 3, 7, 2), ("f1", 4, 7, 2)):
            add("cell", field, label, 5000, c, 10)
            add("cytoplasm", field, label, 4000, y)
        add("nucleus", "f1", 1, 1000, 4, 1)
        add("nucleus", "f2", 1, 1000, 12, 1)
        # Parent 2 fails nucleus count; parent 3 fails pathogen count; parent 4 is uninfected.
        add("nucleus", "f1", 2, 1000, 4, 2)
        add("nucleus", "f1", 3, 1000, 4, 2)
        add("nucleus", "f1", 4, 1000, 4, 3)
        add("nucleus", "f1", 5, 1000, 4, 4)
        add("pathogen", "f1", 1, 120, 8, 1, 10, 2, 3, 4)
        add("pathogen", "f1", 2, 180, 20, 1, 30, 6, 9, 12)
        add("pathogen", "f2", 1, 200, 6, 1, 12, 1, 2, 3)
        add("pathogen", "f1", 3, 200, 6, 2, 12, 1, 2, 3)
        for label in range(4, 15):
            add("pathogen", "f1", label, 20, 6, 3, 12, 1, 2, 3)
        # Missing parent link must not contribute to anyone's child count.
        add("pathogen", "f1", 15, 500, 999, None, 999, 999, 999, 999)
    cells = [_expected("f1", 7, 2, 4, (14, 20, 4, 6, 8), 300, 2),
             _expected("f2", 3, 6, 12, (6, 12, 1, 2, 3), 200, 1)]
    _save_outputs(project, cells)
    return project, settings, cells


def test_positive_aggregates_parent_keys_and_mean_of_ratios(example):
    project, settings, _ = example
    database = project / "measurements" / "measurements.db"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = inspect_results(project, settings)
    assert result["accepted"] is True
    assert result["cell_rows"] == 2
    assert result["well_rows"] == 1
    assert result["exact_surviving_cell_keys"] == ["private_r1_c1_f1_o1", "private_r1_c1_f2_o1"]
    assert result["filter_counts"]["after_nucleus_count_join"] == 4
    assert result["filter_counts"]["after_pathogen_area"] == 2
    assert result["well_counts"][0]["source_cells"] == 5
    assert result["well_counts"][0]["retained_cells"] == 2
    assert result["well_ratio_means"][0]["recruitment"] == 4.0  # (7 + 1) / 2, not 2.5.
    assert len(result["ratios_verified"]) == 16
    assert result["nonrestrictive_intensity_filters"] == []
    assert result["max_absolute_numeric_difference"] < 1e-12
    json.dumps(result, allow_nan=False)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


def _enable_nonrestrictive_filters(project, settings):
    settings["nucleus_intensity_range"] = [-1, 65536]
    settings["pathogen_intensity_range"] = [-1, 65536]
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        connection.execute(
            'ALTER TABLE nucleus ADD COLUMN nucleus_channel_3_mean_intensity REAL DEFAULT 100')


@pytest.mark.parametrize("role", ["nucleus", "pathogen"])
@pytest.mark.parametrize("value", [0, 65535])
def test_nonrestrictive_bounds_include_both_uint16_edges(example, role, value):
    project, settings, cells = example
    _enable_nonrestrictive_filters(project, settings)
    column = f"{role}_channel_{3 if role == 'nucleus' else 1}_mean_intensity"
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        connection.execute(f'UPDATE {role} SET {column}=? WHERE fieldID="f2"', (value,))
    if role == "pathogen":
        cells[1] = _expected("f2", 3, 6, 12, (value, 12, 1, 2, 3), 200, 1)
        _save_outputs(project, cells)
    database = project / "measurements" / "measurements.db"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    result = inspect_results(project, settings)
    assert result["accepted"] is True
    assert result["cell_rows"] == 2
    assert len(result["ratios_verified"]) == 16
    assert len(result["nonrestrictive_intensity_filters"]) == 2
    proof = next(item for item in result["nonrestrictive_intensity_filters"]
                 if item["setting"] == f"{role}_intensity_range")
    assert proof["actual_source_column"] == column
    assert proof["strict_bounds"] == [-1, 65536]
    assert proof["finite_source_domain"] == [0, 65535]
    assert proof["source_rows_checked"] == result["source_table_rows"][role]
    assert proof["additional_rows_excluded"] == 0
    assert proof["source_minimum" if value == 0 else "source_maximum"] == value
    json.dumps(result, allow_nan=False)
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("role", ["nucleus", "pathogen"])
def test_nonrestrictive_role_can_be_combined_with_other_role_disabled(example, role):
    project, settings, _ = example
    _enable_nonrestrictive_filters(project, settings)
    settings[f"{'pathogen' if role == 'nucleus' else 'nucleus'}_intensity_range"] = None
    result = inspect_results(project, settings)
    assert [item["setting"] for item in result["nonrestrictive_intensity_filters"]] == [
        f"{role}_intensity_range"]


@pytest.mark.parametrize("role", ["nucleus", "pathogen"])
@pytest.mark.parametrize("value", [-2, 65537, math.nan, math.inf, -math.inf])
def test_rejects_source_outside_proven_domain_before_aggregation(example, role, value):
    project, settings, _ = example
    _enable_nonrestrictive_filters(project, settings)
    column = f"{role}_channel_{3 if role == 'nucleus' else 1}_mean_intensity"
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        # SQLite stores float NaN as NULL; it must not be skipped by aggregation.
        connection.execute(f'UPDATE {role} SET {column}=? WHERE fieldID="f2"', (value,))
    with pytest.raises(ValueError, match=f"Cannot prove nonrestrictive {role} intensity filter"):
        inspect_results(project, settings)


@pytest.mark.parametrize("label", [1, 4, 15])
def test_domain_proof_checks_children_before_averaging_or_count_filtering(example, label):
    project, settings, _ = example
    _enable_nonrestrictive_filters(project, settings)
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        # 1: mean with sibling 20 would be 9 (apparently valid); 4: count-rejected
        # group; 15: orphan. The proof deliberately checks all three source cases.
        connection.execute(
            'UPDATE pathogen SET pathogen_channel_1_mean_intensity=-2 '
            'WHERE fieldID="f1" AND object_label=?', (label,))
    with pytest.raises(ValueError, match="Cannot prove nonrestrictive pathogen intensity filter"):
        inspect_results(project, settings)


def test_enabled_nucleus_bound_requires_actual_channel_three_source_column(example):
    project, settings, _ = example
    settings["nucleus_intensity_range"] = [-1, 65536]
    with pytest.raises(ValueError, match="Missing nucleus source columns.*nucleus_channel_3"):
        inspect_results(project, settings)


@pytest.mark.parametrize("role", ["nucleus", "pathogen"])
@pytest.mark.parametrize("bounds", [[0, 65535], [-1, 65535], [0, 65536],
                                    [-2, 65537], [-1.0, 65536], [-1, 65536.0],
                                    [], [-1], [-1, 65536, 65537], "[-1, 65536]"])
def test_rejects_excluding_or_unsupported_intensity_bounds(example, role, bounds):
    project, settings, _ = example
    settings[f"{role}_intensity_range"] = bounds
    with pytest.raises(ValueError, match=f"Unsupported {role}_intensity_range"):
        inspect_results(project, settings)


@pytest.mark.parametrize("column", RATIOS)
def test_rejects_wrong_ratio_despite_correct_rows(example, column):
    project, settings, cells = example
    cells[0][column] += 0.25
    _save_outputs(project, cells)
    with pytest.raises(ValueError, match=f"Mismatch cell .*{column}"):
        inspect_results(project, settings)


def test_rejects_same_count_with_another_field_identity(example):
    project, settings, cells = example
    cells[0]["fieldID"] = "f9"
    cells[0]["prcfo"] = "private_r1_c1_f9_o1"
    _save_outputs(project, cells)
    with pytest.raises(ValueError, match="Surviving cell identity mismatch"):
        inspect_results(project, settings)


def test_rejects_inconsistent_export_index(example):
    project, settings, cells = example
    cells[0]["prcfo"] = "private_r1_c1_f1_o2"
    _save_outputs(project, cells)
    with pytest.raises(ValueError, match="index identity"):
        inspect_results(project, settings)


def test_rejects_ratio_of_well_means(example):
    project, settings, _ = example
    path = project / "results" / "wells.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["recruitment"] = 2.5
    _csv(path, rows)
    with pytest.raises(ValueError, match="Mismatch well .*recruitment"):
        inspect_results(project, settings)


def test_source_count_is_not_the_final_count(example):
    project, settings, cells = example
    for row in cells:
        row["cells_per_well"] = 2
    _save_outputs(project, cells)
    with pytest.raises(ValueError, match="cells_per_well"):
        inspect_results(project, settings)


@pytest.mark.parametrize("table,column,value", [
    ("cell", "cell_area", 3500), ("cell", "cell_area", 100000),
    ("cell", "cell_channel_1_percentile_95", 1),
    ("nucleus", "nucleus_area", 700), ("nucleus", "nucleus_area", 4000),
    ("pathogen", "pathogen_area", 100), ("pathogen", "pathogen_area", 1000),
])
def test_strict_area_and_target_boundaries(example, table, column, value):
    project, settings, cells = example
    # f2 has exactly one child, so its source area is also its parent aggregate.
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        connection.execute(f'UPDATE {table} SET {column}=? WHERE fieldID="f2"', (value,))
    _save_outputs(project, cells[:1])
    result = inspect_results(project, settings)
    assert result["cell_rows"] == 1
    assert result["well_counts"][0]["source_cells"] == 5


def test_pathogen_area_is_sum_before_filtering(example):
    project, settings, cells = example
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        connection.execute('UPDATE pathogen SET pathogen_area=60 WHERE fieldID="f1" AND cell_id=1')
    # Each individual pathogen is below 100, but their total 120 passes.
    cells[0]["pathogen_area"] = 120.0
    _save_outputs(project, cells)
    assert inspect_results(project, settings)["cell_rows"] == 2


@pytest.mark.parametrize("numerator,expected_class", [(6, "positive_infinity"), (-6, "negative_infinity"), (0, "nan")])
def test_nonfinite_results_must_match_and_evidence_is_json_safe(example, numerator, expected_class):
    project, settings, cells = example
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        connection.execute('UPDATE cell SET cell_area=3500 WHERE fieldID="f1" AND object_label=1')
        connection.execute('UPDATE cytoplasm SET cytoplasm_channel_1_mean_intensity=0 WHERE fieldID="f2"')
        connection.execute('UPDATE pathogen SET pathogen_channel_1_mean_intensity=? WHERE fieldID="f2"', (numerator,))
    expected = _expected("f2", 3, 0, 12, (numerator, 12, 1, 2, 3), 200, 1)
    _save_outputs(project, [expected])
    result = inspect_results(project, settings)
    assert result["ratio_value_classes"]["recruitment"][expected_class] == 1
    json.dumps(result, allow_nan=False)
    expected["recruitment"] = 0.0
    _save_outputs(project, [expected])
    with pytest.raises(ValueError, match="Mismatch cell .*recruitment"):
        inspect_results(project, settings)


def test_unmapped_pathogen_still_has_host_and_treatment_condition(example):
    project, settings, cells = example
    settings["pathogen_plate_metadata"] = [["c2"]]
    for row in cells:
        row["pathogen"] = ""
        row["condition"] = "HeLa_demo"
    _save_outputs(project, cells)
    assert inspect_results(project, settings)["cell_rows"] == 2


@pytest.mark.parametrize("name,value", [
    ("channel_dims", [0, 1, 2, 3]), ("channel_of_interest", 2),
    ("nuclei_limit", 2), ("pathogen_limit", 1),
    ("cell_intensity_range", [0, 100000]), ("cell_size_range", [3500.0, 100000]),
])
def test_rejects_unsupported_settings(example, name, value):
    project, settings, _ = example
    settings[name] = value
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_results(project, settings)


def test_private_project_must_match_settings_source(example):
    project, settings, _ = example
    settings["src"] = str(project.parent / "original")
    with pytest.raises(ValueError, match="settings src"):
        inspect_results(project, settings)


def test_rejects_aliased_database_instead_of_reading_original(example):
    project, settings, _ = example
    database = project / "measurements" / "measurements.db"
    original = project.parent / "original.db"
    database.rename(original)
    database.symlink_to(original)
    with pytest.raises(ValueError, match="real copy"):
        inspect_results(project, settings)


def test_rejects_changed_child_identity_even_if_number_of_rows_matches(example):
    project, settings, _ = example
    with sqlite3.connect(project / "measurements" / "measurements.db") as connection:
        connection.execute('UPDATE pathogen SET cell_id=4 WHERE fieldID="f2"')
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_results(project, settings)
