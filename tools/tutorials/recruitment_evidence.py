"""Independently check the bounded recruitment tutorial's private-copy exports.

This module deliberately imports no spaCR code. It reads only the project
explicitly supplied by the caller, through a read-only SQLite connection; the
caller must supply the private tutorial copy, never the original acquisition.
Only the recorded channel-one, non-timelapse configuration is supported.
"""
from __future__ import annotations

import csv
import math
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


LOCATION = ("plateID", "rowID", "columnID", "fieldID")
ROLES = ("cell", "cytoplasm", "nucleus", "pathogen")
MAX_TABLE_ROWS = 50_000
REL_TOL = 1e-9
ABS_TOL = 1e-9
NUMERATORS = (
    ("pathogen", "mean_mean", "pathogen_channel_1_mean_intensity"),
    ("pathogen", "q75_mean", "pathogen_channel_1_percentile_75"),
    ("pathogen_outside", "mean_mean", "pathogen_channel_1_outside_mean"),
    ("pathogen_outside", "q75_mean", "pathogen_channel_1_outside_percentile_75"),
    ("pathogen_periphery", "mean_mean", "pathogen_channel_1_periphery_mean"),
)
RATIO_COLUMNS = tuple(
    f"{prefix}_{role}_{suffix}"
    for prefix, suffix, _ in NUMERATORS
    for role in ("cell", "cytoplasm", "nucleus")
) + ("recruitment",)
# These are the current consumer's actual filter channels, not the mask-plane
# names: its mask_chans order is nucleus, pathogen, cell but its filter calls
# use indices cell=0, nucleus=1, pathogen=2. Cell intensity remains disabled.
NONRESTRICTIVE_INTENSITY_COLUMNS = {
    "nucleus": "nucleus_channel_3_mean_intensity",
    "pathogen": "pathogen_channel_1_mean_intensity",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, context):
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Non-numeric {context}: {value!r}") from error


def _label(value, context):
    text = str(value).strip()
    if text.startswith("o"):
        text = text[1:]
    number = _number(text, context)
    _require(math.isfinite(number) and number.is_integer() and number > 0,
             f"Unsupported object identity in {context}: {value!r}")
    return int(number)


def _key(row, child=False):
    coordinates = tuple(str(row[column]).strip() for column in LOCATION)
    _require(all(value and value != "None" for value in coordinates),
             "Missing plate/row/column/field identity")
    column = "cell_id" if child else "object_label"
    return coordinates + (_label(row[column], column),)


def _prc(key):
    return "_".join(key[:3])


def _prcfo(key):
    return "_".join(key[:4]) + f"_o{key[4]}"


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return math.nan
    if any(math.isinf(value) for value in values):
        return sum(values) / len(values)
    return math.fsum(values) / len(values)


def _sum(values):
    values = [value for value in values if not math.isnan(value)]
    return sum(values) if any(math.isinf(v) for v in values) else math.fsum(values)


def _divide(numerator, denominator):
    if math.isnan(numerator) or math.isnan(denominator):
        return math.nan
    if denominator == 0:
        if numerator == 0:
            return math.nan
        sign = math.copysign(1, numerator) * math.copysign(1, denominator)
        return math.copysign(math.inf, sign)
    return numerator / denominator


def _json_number(value):
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Infinity" if value > 0 else "-Infinity"
    return value


def _settings(settings, project, database):
    required = {
        "src", "channel_dims", "channel_of_interest", "cell_chann_dim",
        "nucleus_chann_dim", "pathogen_chann_dim", "nuclei_limit",
        "pathogen_limit", "target_intensity_min", "cells_per_well",
        "cell_types", "cell_plate_metadata", "pathogen_types",
        "pathogen_plate_metadata", "treatments", "treatment_plate_metadata",
        *(f"{role}_{suffix}_range" for role in ("cell", "nucleus", "pathogen")
          for suffix in ("size", "intensity")),
    }
    _require(required <= settings.keys(),
             f"Missing recorded settings: {sorted(required - settings.keys())}")
    _require(Path(settings["src"]).expanduser().resolve() in (project, database),
             "settings src must identify the supplied private project or its database")
    fixed = {"channel_dims": [1], "channel_of_interest": 1,
             "cell_chann_dim": 1, "nucleus_chann_dim": 0,
             "pathogen_chann_dim": 3, "nuclei_limit": 1, "pathogen_limit": 10}
    for name, expected in fixed.items():
        _require(settings[name] == expected and not isinstance(settings[name], bool),
                 f"Unsupported setting {name}: expected {expected!r}")
    for role in ("cell", "nucleus", "pathogen"):
        intensity_bounds = settings[f"{role}_intensity_range"]
        nonrestrictive = (
            role in NONRESTRICTIVE_INTENSITY_COLUMNS
            and isinstance(intensity_bounds, list)
            and len(intensity_bounds) == 2
            and all(type(value) is int for value in intensity_bounds)
            and intensity_bounds == [-1, 65536]
        )
        _require(intensity_bounds is None or nonrestrictive,
                 f"Unsupported {role}_intensity_range: expected None"
                 + (" or integer [-1, 65536]"
                    if role in NONRESTRICTIVE_INTENSITY_COLUMNS else ""))
        bounds = settings[f"{role}_size_range"]
        _require(isinstance(bounds, list) and len(bounds) == 2
                 and all(type(value) is int for value in bounds)
                 and 0 <= bounds[0] < bounds[1],
                 f"Unsupported {role}_size_range: supply two ordered integer bounds")
    threshold = settings["target_intensity_min"]
    _require(threshold is None or (type(threshold) in (int, float)
                                   and math.isfinite(threshold)),
             "Unsupported target_intensity_min")
    minimum = settings["cells_per_well"]
    _require(type(minimum) is int and minimum >= 0,
             "Unsupported cells_per_well: expected a nonnegative integer")
    for labels_key, locations_key in (
        ("cell_types", "cell_plate_metadata"),
        ("pathogen_types", "pathogen_plate_metadata"),
        ("treatments", "treatment_plate_metadata"),
    ):
        labels, locations = settings[labels_key], settings[locations_key]
        _require(isinstance(labels, list) and labels
                 and all(isinstance(label, str) and label for label in labels),
                 f"Unsupported {labels_key}: expected nonempty string labels")
        if locations is not None:
            _require(isinstance(locations, list) and len(locations) == len(labels)
                     and all(isinstance(group, list) and all(
                         isinstance(item, str) and re.fullmatch(r"[rc]\d+", item)
                         for item in group) for group in locations),
                     f"Unsupported {locations_key}: use explicit row/column IDs")


def _read_table(connection, role, settings):
    columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{role}")')}
    _require(columns, f"Missing private measurement table {role}")
    _require(not ({"timeID", "time_id", "timepoint"} & columns),
             f"Unsupported timelapse metadata in {role}")
    required = set(LOCATION) | {"object_label", f"{role}_area",
                               f"{role}_channel_1_mean_intensity"}
    if role == "cell":
        required.add("cell_channel_1_percentile_95")
    if role in ("nucleus", "pathogen"):
        required.add("cell_id")
    if role == "pathogen":
        required.update(column for _, _, column in NUMERATORS)
    if (role in NONRESTRICTIVE_INTENSITY_COLUMNS
            and settings[f"{role}_intensity_range"] is not None):
        required.add(NONRESTRICTIVE_INTENSITY_COLUMNS[role])
    aliases = {}
    canonical = "pathogen_channel_1_outside_percentile_75"
    legacy = "pathogen_channel_1_outside_75_percentile"
    if role == "pathogen" and canonical not in columns and legacy in columns:
        aliases[canonical] = legacy
    missing = {column for column in required if aliases.get(column, column) not in columns}
    _require(not missing, f"Missing {role} source columns: {sorted(missing)}")
    selected = sorted(required | ({"file_name", "prcf"} & columns))
    query_columns = ", ".join(f'"{aliases.get(column, column)}" AS "{column}"'
                              for column in selected)
    rows = [dict(row) for row in connection.execute(
        f'SELECT {query_columns} FROM "{role}" LIMIT ?', (MAX_TABLE_ROWS + 1,))]
    _require(len(rows) <= MAX_TABLE_ROWS,
             f"{role} exceeds the bounded private-subset limit of {MAX_TABLE_ROWS} rows")
    seen = set()
    for row in rows:
        key = _key(row)
        _require(key not in seen, f"Duplicate {role} source identity {key}")
        seen.add(key)
        if "prcf" in row:
            _require(row["prcf"] == "_".join(key[:4]),
                     f"Unsupported inconsistent prcf in {role}: {key}")
        for column in selected:
            if column.startswith(f"{role}_") and column != "cell_id":
                row[column] = _number(row[column], f"{role}.{column}")
    return rows


def _intensity_filter_evidence(tables, settings):
    """Prove enabled strict bounds cannot reject finite uint16-domain means.

    Check every source child, even orphans and count-filtered groups, before
    aggregation: a bad individual value must not disappear into a mean or
    pandas' missing-value handling. A nonempty arithmetic mean of these values
    remains in [0, 65535], strictly within the only supported [-1, 65536]
    bounds. Missing pathogen groups already fail the preceding area filter.
    """
    evidence = []
    for role, column in NONRESTRICTIVE_INTENSITY_COLUMNS.items():
        bounds = settings[f"{role}_intensity_range"]
        if bounds is None:
            continue
        values = []
        for row in tables[role]:
            value = row[column]
            _require(math.isfinite(value) and 0 <= value <= 65535,
                     f"Cannot prove nonrestrictive {role} intensity filter: "
                     f"{column} at {_key(row)} must be finite in [0, 65535], "
                     f"got {value!r}")
            values.append(value)
        _require(values, f"No source values to prove nonrestrictive {role} intensity filter")
        evidence.append({
            "setting": f"{role}_intensity_range", "strict_bounds": list(bounds),
            "actual_source_column": column, "source_rows_checked": len(values),
            "source_minimum": min(values), "source_maximum": max(values),
            "finite_source_domain": [0, 65535],
            "additional_rows_excluded": 0,
        })
    return evidence


def _annotations(key, settings):
    result = {}
    for output, labels_key, locations_key in (
        ("host_cells", "cell_types", "cell_plate_metadata"),
        ("pathogen", "pathogen_types", "pathogen_plate_metadata"),
        ("treatment", "treatments", "treatment_plate_metadata"),
    ):
        labels, locations = settings[labels_key], settings[locations_key]
        value = labels[0] if locations is None else None
        if locations is not None:
            # Match the source's mapping order, including its last-label wins rule.
            mapping = {location: label for label, group in zip(labels, locations)
                       for location in group}
            for location, label in mapping.items():
                if location == key[1 if location.startswith("r") else 2]:
                    value = label
        result[output] = value
    result["condition"] = "_".join(value for value in result.values() if value is not None)
    return result


def _reconstruct(tables, settings):
    cells = {_key(row): row for row in tables["cell"]}
    cytoplasms = {_key(row): row for row in tables["cytoplasm"]}
    children = {role: defaultdict(list) for role in ("nucleus", "pathogen")}
    for role, grouped in children.items():
        for row in tables[role]:
            if row["cell_id"] is not None:
                grouped[_key(row, child=True)].append(row)
    nucleus_groups = {key: rows for key, rows in children["nucleus"].items()
                      if len(rows) <= settings["nuclei_limit"]}
    pathogen_groups = {key: rows for key, rows in children["pathogen"].items()
                       if len(rows) <= settings["pathogen_limit"]}
    # The current reader short-circuits empty grouped tables instead of doing
    # its usual joins. Do not claim to validate that exceptional code path.
    _require(nucleus_groups and pathogen_groups,
             "Unsupported empty child table after count limits")
    source_counts = defaultdict(int)
    for key in cells:
        source_counts[key[:3]] += 1
    stages = {"source_cells": len(cells), "after_nucleus_count_join": 0,
              "after_condition": 0, "after_cell_area": 0,
              "after_target_intensity": 0, "after_nucleus_area": 0,
              "after_pathogen_area": 0}
    survivors = {}
    for key, source in cells.items():
        if key not in nucleus_groups:
            continue
        stages["after_nucleus_count_join"] += 1
        annotations = _annotations(key, settings)
        if not annotations["condition"]:
            continue
        stages["after_condition"] += 1
        nucleus = nucleus_groups[key][0]
        pathogens = pathogen_groups.get(key, [])
        cytoplasm = cytoplasms.get(key, {})
        numeric = {
            "cell_area": source["cell_area"],
            "nucleus_area": nucleus["nucleus_area"],
            "cytoplasm_area": cytoplasm.get("cytoplasm_area", math.nan),
            "pathogen_area": _sum(p["pathogen_area"] for p in pathogens)
            if pathogens else math.nan,
            "cell_channel_1_mean_intensity": source["cell_channel_1_mean_intensity"],
            "cell_channel_1_percentile_95": source["cell_channel_1_percentile_95"],
            "nucleus_channel_1_mean_intensity": nucleus["nucleus_channel_1_mean_intensity"],
            "cytoplasm_channel_1_mean_intensity": cytoplasm.get(
                "cytoplasm_channel_1_mean_intensity", math.nan),
            "nucleus_prcfo_count": 1.0,
            "pathogen_prcfo_count": float(len(pathogens)),
            "cells_per_well": float(source_counts[key[:3]]),
        }
        for _, _, column in NUMERATORS:
            numeric[column] = _mean(p[column] for p in pathogens)
        low, high = settings["cell_size_range"]
        if not low < numeric["cell_area"] < high:
            continue
        stages["after_cell_area"] += 1
        target = settings["target_intensity_min"]
        if target not in (None, 0) and not numeric["cell_channel_1_percentile_95"] > target:
            continue
        stages["after_target_intensity"] += 1
        low, high = settings["nucleus_size_range"]
        if not low < numeric["nucleus_area"] < high:
            continue
        stages["after_nucleus_area"] += 1
        low, high = settings["pathogen_size_range"]
        if not low < numeric["pathogen_area"] < high:
            continue
        stages["after_pathogen_area"] += 1
        for prefix, suffix, numerator in NUMERATORS:
            for role in ("cell", "cytoplasm", "nucleus"):
                numeric[f"{prefix}_{role}_{suffix}"] = _divide(
                    numeric[numerator], numeric[f"{role}_channel_1_mean_intensity"])
        numeric["recruitment"] = _divide(
            numeric["pathogen_channel_1_mean_intensity"],
            numeric["cytoplasm_channel_1_mean_intensity"])
        for role in ("pathogen", "nucleus"):
            for channel in range(4):
                numeric[f"{role}_slope_channel_{channel}"] = 1.0
        if source_counts[key[:3]] >= settings["cells_per_well"]:
            survivors[key] = {"numeric": numeric, "annotations": annotations,
                              "file_name": source.get("file_name")}
    stages["after_source_count_well_filter"] = len(survivors)
    _require(survivors, "No surviving cells: cannot certify empty recruitment plots/exports")
    wells = defaultdict(list)
    for key, row in survivors.items():
        wells[key[:3]].append(row)
    well_means = {key: {column: _mean(row["numeric"][column] for row in rows)
                        for column in rows[0]["numeric"]}
                  for key, rows in wells.items()}
    return survivors, well_means, source_counts, stages


def _read_csv(path, wells=False):
    _require(path.is_file() and not path.is_symlink(), f"Missing or symlinked export: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        _require(header, f"Empty CSV: {path.name}")
        duplicates = {column for column in header if header.count(column) > 1}
        _require(duplicates <= (set(LOCATION[:3]) if wells else set()),
                 f"Ambiguous duplicate CSV columns in {path.name}: {duplicates}")
        rows = []
        for values in reader:
            _require(len(values) == len(header), f"Malformed CSV row in {path.name}")
            for column in duplicates:
                _require(len({values[i] for i, name in enumerate(header) if name == column}) == 1,
                         f"Conflicting repeated well identity {column} in {path.name}")
            rows.append(dict(zip(header, values)))
            _require(len(rows) <= MAX_TABLE_ROWS, f"Export exceeds bounded limit: {path.name}")
    return rows


def _compare(expected, actual, context, differences):
    actual = _number(actual, context)
    if math.isnan(expected):
        matches = math.isnan(actual)
    elif math.isinf(expected):
        matches = actual == expected
    else:
        matches = math.isfinite(actual) and math.isclose(
            actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    _require(matches, f"Mismatch {context}: expected {expected!r}, exported {actual!r}")
    if math.isfinite(expected):
        differences.append(abs(actual - expected))


def inspect_results(project, settings):
    """Return JSON-safe independently checked evidence, or raise ``ValueError``.

    Read ``measurements/measurements.db`` and ``results/{cells,wells}.csv``
    only inside the explicitly supplied private project. No app helpers, DB
    migrations, image reads, SQL writes, or pipeline execution occur here.
    NaN and signed infinities must agree; finite values use 1e-9 tolerances.
    """
    requested = Path(project).expanduser()
    _require(requested.is_dir() and not requested.is_symlink(),
             "Supply an existing private project directory, not a symlink")
    project = requested.resolve()
    database = project / "measurements" / "measurements.db"
    _require(database.is_file() and database.resolve() == database
             and database.stat().st_nlink == 1,
             "Private measurement database must be a real copy, not a symlink/hard link")
    _settings(settings, project, database)
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        tables = {role: _read_table(connection, role, settings) for role in ROLES}
    intensity_evidence = _intensity_filter_evidence(tables, settings)
    expected, expected_wells, source_counts, stages = _reconstruct(tables, settings)
    actual_rows = _read_csv(project / "results" / "cells.csv")
    actual = {}
    for row in actual_rows:
        _require(set(LOCATION) | {"object_label", "prcfo"} <= row.keys(),
                 "cells.csv lacks complete parent-cell identity columns")
        key = _key(row)
        _require(key not in actual, f"Duplicate exported cell identity {key}")
        _require(row["prcfo"] == _prcfo(key), f"Mismatch exported cell index identity {key}")
        actual[key] = row
    _require(actual.keys() == expected.keys(),
             f"Surviving cell identity mismatch: missing {sorted(expected.keys() - actual.keys())[:3]}, "
             f"unexpected {sorted(actual.keys() - expected.keys())[:3]}")
    differences = []
    for key, row in expected.items():
        export = actual[key]
        _require(export.get("prc") == _prc(key), f"Mismatch exported well identity for {key}")
        for column, value in row["annotations"].items():
            _require(export.get(column) == (value or ""),
                     f"Mismatch cell annotation {key} {column}")
        if row["file_name"] is not None:
            _require(export.get("file_name") == row["file_name"],
                     f"Mismatch source file identity for {key}")
        for column, value in row["numeric"].items():
            _require(column in export, f"Missing cells.csv column {column}")
            _compare(value, export[column], f"cell {key} {column}", differences)
    actual_wells = {}
    for row in _read_csv(project / "results" / "wells.csv", wells=True):
        _require(set(LOCATION[:3]) <= row.keys(), "wells.csv lacks full well identity")
        key = tuple(row[column] for column in LOCATION[:3])
        _require(key not in actual_wells, f"Duplicate exported well identity {key}")
        _require(row.get("prc") == _prc(key), f"Mismatch exported well index identity {key}")
        actual_wells[key] = row
    _require(actual_wells.keys() == expected_wells.keys(), "Surviving well identity mismatch")
    for key, numeric in expected_wells.items():
        for column, value in numeric.items():
            _require(column in actual_wells[key], f"Missing wells.csv column {column}")
            _compare(value, actual_wells[key][column], f"well {key} {column}", differences)
    final_counts = defaultdict(int)
    for key in expected:
        final_counts[key[:3]] += 1
    return {
        "accepted": True,
        "reason": "Independent source keys, aggregates, all 16 ratios and well means match exports",
        "project": str(project), "database": str(database),
        "source_table_rows": {role: len(rows) for role, rows in tables.items()},
        "source_field_identities": [list(key) for key in sorted({key[:4] for key in
                                     (_key(row) for row in tables["cell"])})],
        "filter_counts": stages, "cell_rows": len(expected), "well_rows": len(expected_wells),
        "nonrestrictive_intensity_filters": intensity_evidence,
        "exact_surviving_cell_keys": [_prcfo(key) for key in sorted(expected)],
        "ratios_verified": list(RATIO_COLUMNS),
        "max_absolute_numeric_difference": max(differences, default=0.0),
        "finite_tolerance": {"relative": REL_TOL, "absolute": ABS_TOL},
        "well_counts": [{"plateID": key[0], "rowID": key[1], "columnID": key[2],
                         "source_cells": source_counts[key], "retained_cells": final_counts[key],
                         "retained": key in expected_wells}
                        for key in sorted(source_counts)],
        "well_ratio_means": [{"prc": _prc(key), **{column: _json_number(numeric[column])
                              for column in RATIO_COLUMNS}}
                             for key, numeric in sorted(expected_wells.items())],
        "ratio_value_classes": {column: {
            "finite": sum(math.isfinite(row["numeric"][column]) for row in expected.values()),
            "nan": sum(math.isnan(row["numeric"][column]) for row in expected.values()),
            "positive_infinity": sum(row["numeric"][column] == math.inf for row in expected.values()),
            "negative_infinity": sum(row["numeric"][column] == -math.inf for row in expected.values()),
        } for column in RATIO_COLUMNS},
    }
