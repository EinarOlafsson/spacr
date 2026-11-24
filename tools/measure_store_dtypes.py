"""Measure the smallest storage dtype of every measurement column, each on its own.

Reads one well's per-field phenotype tables (``<well>/phenotype_cells/
site-<n>.parquet``) one field at a time, read-only, and writes the dtype map
that the spaCR store applies, per column, when a table is written through
:func:`spacr.store._stored_types` (``spacr/resources/data/store_dtypes.json``;
the OPS ``ph_tables`` phase first), plus a per-column evidence table.

A dtype passes for a column when, over every cell of the well:

* every finite value stays finite (float16 holds at most 65,504);
* the largest absolute rounding error is below 1 % of the column's
  between-cell SD, or of its IQR when the SD is 0 or not finite, and is 0
  when both are 0;
* Spearman's rank correlation of the cast with the original is at least
  0.9999 (on every ``--stride``-th cell, NaN excluded, inf kept);
* NaN and inf stay NaN and inf (IEEE casts keep them; counted to be sure).

Floats climb float16 -> float32 -> float64, integers take the narrowest of
int8/uint8, int16/uint16, int32, int64 that holds the range. The map is
PER COLUMN (maintainer, 2026-11-23): each column gets the smallest dtype
it passes on its own, with its evidence, and the store writer looks the
column up by name. The same data give the same map.

``--from-csv`` rebuilds the map from an earlier run's evidence table.

Usage::

    python tools/measure_store_dtypes.py WELL_TABLE CELLS_DIR \
        --channels DAPI,GFP,A594,AF750 --out-map MAP.json --out-csv COLUMNS.csv
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TOLERANCE_OF_SPREAD = 0.01
MIN_SPEARMAN = 0.9999
FLOAT_LADDER = ("float16", "float32", "float64")
INT_LADDER = ("uint8", "int8", "uint16", "int16", "uint32", "int32", "int64")
WIDTH = {"float16": 2, "float32": 4, "float64": 8, "uint8": 1, "int8": 1,
         "uint16": 2, "int16": 2, "uint32": 4, "int32": 4, "int64": 8}


def _integer_dtype(low: float, high: float) -> str:
    """The narrowest integer dtype holding ``[low, high]``."""
    for name in INT_LADDER:
        info = np.iinfo(name)
        if low >= info.min and high <= info.max:
            return name
    return "int64"


def _rank_corr(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman's rho of two equal-length vectors with ties averaged."""
    from scipy.stats import rankdata

    a = rankdata(left)
    b = rankdata(right)
    a -= a.mean()
    b -= b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else float("nan")
    return float((a * b).sum() / denominator)


def measure(well_table: str, cells_dir: str, channels, stride: int = 4,
            max_fields: int = 0):
    """Stream the fields and return the per-column evidence and the map."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from spacr.store import _dtype_family

    schema = pq.ParquetFile(well_table).schema_arrow
    columns, kinds = [], []
    for field in schema:
        if pa.types.is_floating(field.type):
            columns.append(field.name)
            kinds.append({"float": "float32", "double": "float64",
                          "halffloat": "float16"}[str(field.type)])
        elif pa.types.is_integer(field.type):
            columns.append(field.name)
            kinds.append("int")
    sites = pq.read_table(well_table, columns=["site"]).column(
        "site").to_numpy()
    order = list(dict.fromkeys(int(s) for s in sites))
    if max_fields:
        order = order[:max_fields]
    width = len(columns)
    n_fin = np.zeros(width)
    n_nan = np.zeros(width)
    n_pinf = np.zeros(width)
    n_ninf = np.zeros(width)
    low = np.full(width, np.inf)
    high = np.full(width, -np.inf)
    mean = np.zeros(width)
    m2 = np.zeros(width)
    integral = np.ones(width, bool)
    error = {"float32": np.zeros(width), "float16": np.zeros(width)}
    overflow = {"float32": np.zeros(width, bool),
                "float16": np.zeros(width, bool)}
    kept_special = {"float32": np.ones(width, bool),
                    "float16": np.ones(width, bool)}
    sample, offset, cells = [], 0, 0
    started = time.perf_counter()
    for number, site in enumerate(order):
        table = pq.read_table(os.path.join(cells_dir, f"site-{site}.parquet"))
        rows = table.num_rows
        if not rows:
            continue
        present = set(table.column_names)
        matrix = np.full((rows, width), np.nan)
        for position, name in enumerate(columns):
            if name in present:
                matrix[:, position] = table.column(name).to_numpy(
                    zero_copy_only=False).astype(np.float64)
        del table
        finite = np.isfinite(matrix)
        counts = finite.sum(0)
        n_fin_new = n_fin + counts
        n_nan += np.isnan(matrix).sum(0)
        n_pinf += (matrix == np.inf).sum(0)
        n_ninf += (matrix == -np.inf).sum(0)
        masked = np.where(finite, matrix, np.nan)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low = np.fmin(low, np.nanmin(masked, 0))
                high = np.fmax(high, np.nanmax(masked, 0))
                chunk_mean = np.where(counts > 0, np.nansum(masked, 0)
                                      / np.maximum(counts, 1), 0.0)
                chunk_m2 = np.nansum((masked - chunk_mean) ** 2, 0)
                delta = chunk_mean - mean
                total = np.maximum(n_fin_new, 1)
                mean = mean + delta * counts / total
                m2 = m2 + chunk_m2 + delta ** 2 * n_fin * counts / total
                n_fin = n_fin_new
                integral &= ~np.any(finite & (np.round(matrix) != matrix), 0)
                for name in ("float32", "float16"):
                    cast = matrix.astype(name).astype(np.float64)
                    lost = finite & ~np.isfinite(cast)
                    overflow[name] |= lost.any(0)
                    special = ~finite
                    kept_special[name] &= ~np.any(
                        special & ~((np.isnan(matrix) & np.isnan(cast))
                                    | (matrix == cast)), 0)
                    gap = np.where(finite & np.isfinite(cast),
                                   np.abs(matrix - cast), 0.0)
                    error[name] = np.maximum(error[name], gap.max(0))
        pick = np.flatnonzero((offset + np.arange(rows)) % stride == 0)
        sample.append(matrix[pick].astype(np.float64))
        offset += rows
        cells += rows
        if number % 100 == 0:
            print(f"{number}/{len(order)} fields, {cells} cells, "
                  f"{time.perf_counter() - started:.0f} s", flush=True)
    sample = np.concatenate(sample) if sample else np.empty((0, width))
    sd = np.sqrt(np.where(n_fin > 1, m2 / np.maximum(n_fin - 1, 1), np.nan))
    records = []
    for position, name in enumerate(columns):
        values = sample[:, position]
        finite = values[np.isfinite(values)]
        iqr = (float(np.subtract(*np.percentile(finite, [75, 25])))
               if finite.size else float("nan"))
        spread = float(sd[position])
        scale_kind = "sd"
        if not np.isfinite(spread) or spread == 0:
            spread, scale_kind = iqr, "iqr"
        record = {"column": name, "family": _dtype_family(name, channels),
                  "source": kinds[position], "cells": int(cells),
                  "finite": int(n_fin[position]), "nan": int(n_nan[position]),
                  "posinf": int(n_pinf[position]),
                  "neginf": int(n_ninf[position]),
                  "min": float(low[position]), "max": float(high[position]),
                  "sd": float(sd[position]), "iqr": iqr, "scale": spread,
                  "scale_kind": scale_kind,
                  "integral": bool(integral[position])}
        if kinds[position] == "int":
            record["dtype"] = (_integer_dtype(low[position], high[position])
                               if n_fin[position] else "int8")
            records.append(record)
            continue
        valid = ~np.isnan(values)
        original = values[valid]
        chosen = kinds[position]
        for candidate in FLOAT_LADDER:
            if WIDTH[candidate] >= WIDTH[kinds[position]]:
                chosen = kinds[position]
                break
            worst = float(error[candidate][position])
            relative = (worst / spread if spread and np.isfinite(spread)
                        else (0.0 if worst == 0 else float("inf")))
            rho = (_rank_corr(original, original.astype(candidate).astype(
                np.float64)) if original.size > 1 else 1.0)
            record[f"{candidate}_max_abs_error"] = worst
            record[f"{candidate}_error_over_scale"] = relative
            record[f"{candidate}_spearman"] = rho
            record[f"{candidate}_overflow"] = bool(overflow[candidate][position])
            passes = (not overflow[candidate][position]
                      and kept_special[candidate][position]
                      and relative < TOLERANCE_OF_SPREAD
                      and (np.isnan(rho) and worst == 0 or rho >= MIN_SPEARMAN))
            if passes:
                chosen = candidate
                break
        record["dtype"] = chosen
        records.append(record)
    return pd.DataFrame(records), {"cells": int(cells), "fields": len(order),
                                   "stride": int(stride)}


def _number(value):
    """A CSV or NumPy value as a JSON number, None when it is missing."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(value) else value


def _significant(value, digits: int = 4):
    """``value`` to ``digits`` significant digits, None when missing."""
    value = _number(value)
    return None if value is None else float(f"{value:.{digits - 1}e}")


def _column_entry(record) -> dict:
    """The map entry of one column: its own dtype, and for a narrowed float
    the error over its SD and the Spearman that let it narrow (4 digits).
    The full evidence stays in the evidence table the map names."""
    dtype = str(record["dtype"])
    kind = "int" if record["source"] == "int" else "float"
    item = {"dtype": dtype, "kind": kind}
    if kind == "float" and dtype != record["source"]:
        item["err_sd"] = _significant(record.get(f"{dtype}_error_over_scale"))
        item["rho"] = _significant(record.get(f"{dtype}_spearman"), 6)
    return item


def columns_sha256(columns) -> str:
    """The sha256 of a map's ``columns`` block, canonical JSON."""
    import hashlib

    text = json.dumps(columns, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_map(recorded, path: str) -> None:
    """Write the map: the header, then one column per line, sorted."""
    head = {k: v for k, v in recorded.items() if k != "columns"}
    lines = [json.dumps(name) + ": " + json.dumps(entry, sort_keys=True)
             for name, entry in sorted(recorded["columns"].items())]
    text = json.dumps(head, indent=1, sort_keys=True)[:-2]
    text += ',\n "columns": {\n  ' + ",\n  ".join(lines) + "\n }\n}\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _passes(row, step: str) -> bool:
    """Whether one cast (``64to32`` or ``32to16``) of a loss-table row passes."""
    rho = row[f"{step}_spearman"]
    return bool(row[f"{step}_overflow_fraction"] == 0
                and str(row[f"{step}_nan_preserved"]) == "True"
                and str(row[f"{step}_inf_preserved"]) == "True"
                and row[f"{step}_max_abs_error_over_sd"] < TOLERANCE_OF_SPREAD
                and rho == rho and rho >= MIN_SPEARMAN)


def from_loss_table(table):
    """This tool's evidence from a per-column loss table (``per_column_loss``).

    The loss table (Home, A1) holds, per column, the 64 -> 32 and 32 -> 16
    casts' largest error over the between-cell SD, Spearman, overflow
    fraction and NaN/inf preservation. Nothing is recomputed: each stored
    column is judged on its own with the same four criteria as
    :func:`measure` (float16 when the 32 -> 16 cast passes, and for a
    float64 source the 64 -> 32 cast too; float32 when only the latter
    passes; float64 otherwise), integers take the narrowest type holding
    their range, and the rows the table marks as not stored (source keys
    and centroids) are left out. Its ``n_inf`` is carried as ``posinf``
    (the table does not split the signs).
    """
    import pandas as pd

    source_of = {"double": "float64", "float": "float32",
                 "halffloat": "float16", "int64": "int"}
    records = []
    for row in table.to_dict("records"):
        if str(row.get("stored")) != "True":
            continue
        source = source_of[str(row["source_dtype"])]
        record = {"column": row["column"], "source": source,
                  "min": row["min"], "max": row["max"],
                  "nan": int(row["n_nan"]), "posinf": int(row["n_inf"]),
                  "neginf": 0, "sd": row["sd"]}
        for step, name in (("64to32", "float32"), ("32to16", "float16")):
            record[f"{name}_error_over_scale"] = row[
                f"{step}_max_abs_error_over_sd"]
            record[f"{name}_spearman"] = row[f"{step}_spearman"]
            record[f"{name}_overflow"] = bool(
                row[f"{step}_overflow_fraction"] > 0)
        if source == "int":
            record["dtype"] = _integer_dtype(row["min"], row["max"])
        elif _passes(row, "32to16") and (source == "float32"
                                         or _passes(row, "64to32")):
            record["dtype"] = "float16"
        elif source == "float32" or _passes(row, "64to32"):
            record["dtype"] = "float32"
        else:
            record["dtype"] = "float64"
        records.append(record)
    return pd.DataFrame(records)


RULE = ("per column, each column on its own (maintainer, 2026-11-23): the "
        "smallest of float16 < float32 < float64 whose cast keeps every "
        "finite value finite (no overflow), keeps NaN and inf exactly, has "
        "a largest absolute rounding error below 1 % of the column's "
        "between-cell SD and a Spearman correlation with the original of "
        "at least 0.9999; integers take the narrowest type holding their "
        "range; readers upcast float16 to float32")


def build_map(evidence, info, channels, source: str, *,
              evidence_sha256: str = None, measured: str = None):
    """The recorded map: one entry per column, each column judged on its own.

    Deterministic: the same evidence (and ``measured`` date) gives the same
    map, columns sorted by name; the header names the source, its sha256,
    the rule and the sha256 of the columns block.
    """
    columns = {str(record["column"]): _column_entry(record)
               for record in evidence.to_dict("records")}
    counts = {}
    for entry in columns.values():
        counts[entry["dtype"]] = counts.get(entry["dtype"], 0) + 1
    columns = dict(sorted(columns.items()))
    return {"version": 2,
            "measured": measured or datetime.date.today().isoformat(),
            "source": source, "evidence_sha256": evidence_sha256,
            "rule": RULE, "columns_sha256": columns_sha256(columns),
            "cells": info["cells"], "fields": info["fields"],
            "spearman_stride": info["stride"],
            "channels": [c.lower() for c in channels],
            "criteria": {"tolerance_of_sd": TOLERANCE_OF_SPREAD,
                         "min_spearman": MIN_SPEARMAN,
                         "float_ladder": list(FLOAT_LADDER),
                         "int_ladder": list(INT_LADDER),
                         "nan_inf": "kept exactly"},
            "dtype_counts": dict(sorted(counts.items())),
            "columns": columns}


def main(argv=None) -> int:
    """Measure one well and write the map and the evidence."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("well_table")
    parser.add_argument("cells_dir")
    parser.add_argument("--from-csv", default="",
                        help="rebuild the map from this evidence table")
    parser.add_argument("--cells", type=int, default=0)
    parser.add_argument("--fields", type=int, default=0)
    parser.add_argument("--channels", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--out-map", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--source", default="")
    parser.add_argument("--max-fields", type=int, default=0)
    parser.add_argument("--measured", default="",
                        help="the date the evidence was measured")
    args = parser.parse_args(argv)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    evidence_sha256 = None
    if args.from_csv:
        import hashlib

        import pandas as pd

        with open(args.from_csv, "rb") as handle:
            evidence_sha256 = hashlib.sha256(handle.read()).hexdigest()
        evidence = pd.read_csv(args.from_csv)
        if "32to16_max_abs_error_over_sd" in evidence.columns:
            evidence = from_loss_table(evidence)
        info = {"cells": args.cells, "fields": args.fields,
                "stride": args.stride}
    else:
        evidence, info = measure(args.well_table, args.cells_dir, channels,
                                 args.stride, args.max_fields)
        evidence.to_csv(args.out_csv, index=False)
    recorded = build_map(evidence, info, channels,
                         args.source or os.path.basename(args.well_table),
                         evidence_sha256=evidence_sha256,
                         measured=args.measured or None)
    write_map(recorded, args.out_map)
    counts = evidence["dtype"].value_counts().to_dict()
    print(json.dumps({"columns": int(len(evidence)), "dtypes": counts},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
