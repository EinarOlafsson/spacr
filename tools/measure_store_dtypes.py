"""Measure the smallest storage dtype of every measurement column family.

Reads one well's per-field phenotype tables (``<well>/phenotype_cells/
site-<n>.parquet``) one field at a time, read-only, and writes the dtype map
that the spaCR store applies when a table is written through
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
int8/uint8, int16/uint16, int32, int64 that holds the range. A family's
dtype is the widest any of its columns needs (floats) or the narrowest
holding every column's range (integers), so the map is conservative and
deterministic: the same data give the same map. Where a family's float
columns need different dtypes, the finer families (whole numbers kept, as
``{compartment}_zernike_0``) are recorded too and win on lookup, so one
near-constant column does not widen its whole family.

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


def _float_entry(label):
    """The map entry of a group of float columns: the widest dtype they need."""
    dtype = max(label["dtype"], key=lambda d: WIDTH[d])
    item = {"dtype": dtype, "columns": int(len(label))}
    column = f"{dtype}_error_over_scale"
    if dtype != "float64" and column in label:
        item["max_error_over_scale"] = float(label[column].max())
        item["min_spearman"] = float(label[f"{dtype}_spearman"].min())
    narrower = [d for d in FLOAT_LADDER if WIDTH[d] < WIDTH[dtype]]
    if narrower:
        below = narrower[-1]
        failing = label[label["dtype"] == dtype]
        reasons = []
        if f"{below}_overflow" in failing and failing[
                f"{below}_overflow"].eq(True).any():
            reasons.append(f"{below} overflows")
        if f"{below}_error_over_scale" in failing:
            worst = float(failing[f"{below}_error_over_scale"].max())
            if worst >= TOLERANCE_OF_SPREAD:
                reasons.append(f"{below} error {worst:.3g} of spread")
        if f"{below}_spearman" in failing:
            rho = float(failing[f"{below}_spearman"].min())
            if rho < MIN_SPEARMAN:
                reasons.append(f"{below} Spearman {rho:.6f}")
        item["why_not_" + below] = "; ".join(reasons) or "source"
    item["max_abs"] = float(np.nanmax(np.abs(label[["min", "max"]]
                                             .to_numpy(float))))
    return item


def build_map(evidence, info, channels, source: str):
    """The recorded map: one entry per family and source kind."""
    from spacr.store import _dtype_family

    evidence = evidence.copy()
    evidence["family"] = [_dtype_family(c, channels) for c in evidence["column"]]
    evidence["fine"] = [_dtype_family(c, channels, literal_numbers=True)
                        for c in evidence["column"]]
    families = {}
    for family, rows in evidence.groupby("family", sort=True):
        floats = rows[rows["source"] != "int"]
        ints = rows[rows["source"] == "int"]
        entry = {}
        if not floats.empty:
            entry["float"] = _float_entry(floats)
            if floats["dtype"].nunique() > 1:
                for fine, members in floats.groupby("fine", sort=True):
                    if fine != family:
                        families.setdefault(fine, {})["float"] = _float_entry(
                            members)
        if not ints.empty:
            low, high = float(ints["min"].min()), float(ints["max"].max())
            entry["int"] = {"dtype": _integer_dtype(low, high),
                            "columns": int(len(ints)), "min": low, "max": high}
        families.setdefault(family, {}).update(entry)
    return {"version": 1,
            "measured": datetime.date.today().isoformat(),
            "source": source, "cells": info["cells"], "fields": info["fields"],
            "spearman_stride": info["stride"],
            "channels": [c.lower() for c in channels],
            "criteria": {"tolerance_of_spread": TOLERANCE_OF_SPREAD,
                         "min_spearman": MIN_SPEARMAN,
                         "float_ladder": list(FLOAT_LADDER),
                         "int_ladder": list(INT_LADDER)},
            "families": dict(sorted(families.items()))}


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
    args = parser.parse_args(argv)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if args.from_csv:
        import pandas as pd

        evidence = pd.read_csv(args.from_csv)
        info = {"cells": args.cells, "fields": args.fields,
                "stride": args.stride}
    else:
        evidence, info = measure(args.well_table, args.cells_dir, channels,
                                 args.stride, args.max_fields)
        evidence.to_csv(args.out_csv, index=False)
    recorded = build_map(evidence, info, channels,
                         args.source or os.path.basename(args.well_table))
    with open(args.out_map, "w", encoding="utf-8") as handle:
        json.dump(recorded, handle, indent=1, sort_keys=True)
        handle.write("\n")
    counts = evidence["dtype"].value_counts().to_dict()
    print(json.dumps({"columns": int(len(evidence)), "dtypes": counts,
                      "families": len(recorded["families"])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
