#!/usr/bin/env python
"""Scaling benchmark for spacr-store (spacr/store.py) against legacy SQLite.

The tier data is a simulated optical pooled screen from spaCR's power module
(spacr.power_ops.write_ops_screen, with spacr.power_ops.nominal_calibration or a
calibration file from a real run), streamed well by well through
spacr.store.writer; there is no other generator in this file.

Subcommands (one process per tier; run each through heavy_slot.sh):

  store  --tier T --root R --out O   write tier T with spacr.power_ops.write_ops_screen,
                                     register it, then measure every module
                                     operation; tier data is deleted afterwards
                                     unless --keep (10k is always kept)
  legacy --tier T --root R --out O   same data as legacy measurements.db (spaCR's own
                                     INSERT path), same operations through sqlite3
  report --out O                     metrics.csv, metrics.md, charts

Every operation is labelled with how it is answered:
  "Store API"                        an existing spacr.store call (Store.page, Store.count,
                                     Store.query, Store.batches, store.register, ...)
  "Store.sql"                        the raw-SQL entry point over the views
  "harness emulation of planned API" an API from the store design that does not exist
                                     yet (the edit log and journal, Store.lookup),
                                     implemented here as the design describes
  "legacy sqlite3"                   the SQLite query spaCR runs today
Timing: one cold run (page cache of the project's files dropped with
posix_fadvise DONTNEED; the open Store keeps its in-process caches, as a GUI
would), then N warm repeats (median, max). Peak RSS is sampled every 20 ms
over the process and its children.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import resource
import shutil
import sqlite3
import statistics
import sys
import threading
import time
import uuid

import numpy as np

TIERS = {
    "tiny": dict(plates=2, wells=3, fields=2, cells=50, width=300),
    # name: plates, wells per plate, fields per well, cells per field, feature width
    "10k": dict(plates=1, wells=8, fields=10, cells=125, width=2800),
    "1M": dict(plates=2, wells=50, fields=20, cells=500, width=2800),
    "1M_w300": dict(plates=2, wells=50, fields=20, cells=500, width=300),
    # same as 1M_w300, written with row_copy=True: the opt-in row-major Zarr copy
    "1M_w300_rc": dict(plates=2, wells=50, fields=20, cells=500, width=300, row_copy=True),
    "10M": dict(plates=4, wells=100, fields=50, cells=500, width=300),
    "10M_w2800": dict(plates=2, wells=100, fields=50, cells=1000, width=2800),
    "100M": dict(plates=10, wells=200, fields=100, cells=500, width=300),
}
TABLES = ("cell", "nucleus", "cytoplasm")
SHARE = {"cell": 0.4, "nucleus": 0.3, "cytoplasm": 0.3}
F16_SHARE = 0.6          # 60 % float16, 40 % float32 ("mostly float16/float32 mixed")
KEY_COLUMNS = ("object_key", "timeID", "fieldID", "object_label", "rowID",
               "columnID", "gene", "condition", "prcfo")
N_GENES = 1000
N_LABELLED = 10_000
SCORE_TABLE = "score__bench__run1"
SLOW_REPEAT_S = 120.0
GUARD_GB = 16.0          # RAM guard for the training pulls at 100M (scheduler cap 24 GB)    # a cold run above this gets 1 warm repeat, not 3

# ----------------------------------------------------------------------------- helpers


_FEATURES = {}


def _feature_names(table, width):
    """``(bounded and count names, intensity names)`` of one table: A1's column names as the
    power module's streamed writer gives them (spacr.power_ops._stream_features)."""
    from spacr import power_ops
    if width not in _FEATURES:
        _FEATURES[width] = power_ops._stream_features(width)
    specs = _FEATURES[width][table]
    return ([x["name"] for x in specs if x["dtype"] == "float16" or x["kind"] == "int"],
            [x["name"] for x in specs if not (x["dtype"] == "float16" or x["kind"] == "int")])


def _wells(plates, wells):
    out = []
    for p in range(plates):
        for i in range(wells):
            r, c = i // 24 + 1, i % 24 + 1
            out.append((f"plate{p + 1:02d}", f"{chr(64 + r)}{c:02d}", r, c))
    return out


def integer_cell_key(row, col, field, label):
    """Same formula as spacr.ops_run_infra.integer_cell_key (checked at 10k)."""
    return ((np.int64(row) * 100 + col) * 2 ** 24 + field.astype(np.int64)) * 2 ** 20 \
        + label.astype(np.int64)


def _seed(*parts):
    import hashlib
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


CALIBRATION = None   # a calibration JSON from a real run (--calibration); None is the nominal one
DESIGN = {"cells_per_gene": 100}
SEED = 20261123


def _calibration():
    from spacr import power_ops
    if CALIBRATION:
        return power_ops.load_calibration(CALIBRATION)
    return power_ops.nominal_calibration(n_genes=N_GENES)


def write_wells(project, cfg, only, run_id="bench"):
    """Write the named wells of a tier into ``project`` with the power module's streamed writer."""
    from spacr import power_ops
    return power_ops.write_ops_screen(
        _calibration(), project, DESIGN, SEED, plates=cfg["plates"], wells=cfg["wells"],
        fields=cfg["fields"], cells_per_field=cfg["cells"], n_features=cfg["width"], only=only,
        run_id=run_id, row_copy=bool(cfg.get("row_copy")))


def well_frames(tier, plate, well, r, c, cfg, as_float64=False):
    """The three object tables of one well as pyarrow Tables (for the legacy writer):
    the power module writes the well into a scratch store, which is read back."""
    import tempfile
    import pyarrow as pa
    from spacr import store
    scratch = tempfile.mkdtemp(prefix="bench_well_", dir=os.environ.get("BENCH_SCRATCH"))
    try:
        write_wells(scratch, cfg, [f"{plate}/{well}"])
        store.register(scratch)
        out = {}
        with store.open(scratch) as s:
            for t in TABLES:
                tab = s.query(table=t, as_="arrow").drop_columns(["plate", "well"])
                if as_float64:
                    tab = tab.cast(pa.schema([pa.field(f.name, pa.float64()) if pa.types.is_floating(f.type)
                                              else f for f in tab.schema]))
                out[t] = tab
        return out, out["cell"].column("object_key").to_numpy()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def labelled_set(tier, cfg):
    """N_LABELLED cells spread over every well: (plate, well, object_key, label)."""
    wells = _wells(cfg["plates"], cfg["wells"])
    n_well = cfg["fields"] * cfg["cells"]
    total = len(wells) * n_well
    rng = np.random.default_rng(_seed(tier, "labels"))
    pick = np.sort(rng.choice(total, size=min(N_LABELLED, total), replace=False))
    w_idx, pos = pick // n_well, pick % n_well
    field = (pos // cfg["cells"] + 1).astype(np.uint32)
    label = (pos % cfg["cells"] + 1).astype(np.uint32)
    rows = []
    for wi, f, lb in zip(w_idx, field, label):
        p, w, r, c = wells[wi]
        rows.append((p, w, int(integer_cell_key(r, c, np.array([f]), np.array([lb]))[0]),
                     f"{p}_{r}_{c}_{f}_{lb}", int(rng.integers(0, 2))))
    return rows


def du(path):
    total = 0
    for a, _d, fs in os.walk(path):
        for f in fs:
            try:
                total += os.lstat(os.path.join(a, f)).st_blocks * 512
            except OSError:
                pass
    return total


def drop_cache(path):
    """posix_fadvise DONTNEED on every file under path (no sudo needed)."""
    paths = [path] if os.path.isfile(path) else [os.path.join(a, f) for a, _d, fs in os.walk(path) for f in fs]
    for p in paths:
        try:
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


class Peak:
    """Peak RSS (GB) of this process plus its children, sampled every 20 ms."""

    def __init__(self):
        import psutil
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = threading.Event()

    def _rss(self):
        total = 0
        try:
            total = self.proc.memory_info().rss
            for ch in self.proc.children(recursive=True):
                try:
                    total += ch.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        return total

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            self._stop.wait(0.02)

    def __enter__(self):
        self.peak = self._rss()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self._rss())


class Recorder:
    def __init__(self, out, tier, backend, cfg, append=False):
        self.path = os.path.join(out, f"metrics_{backend}_{tier}.jsonl")
        os.makedirs(out, exist_ok=True)
        self.done = set()
        if append and os.path.exists(self.path):
            with open(self.path) as h:
                self.done = {(r["module"], r["op"]) for r in map(json.loads, h)
                             if r.get("median_s") is not None and not r.get("error")}
        else:
            open(self.path, "w").close()
        self.tier, self.backend, self.cfg = tier, backend, cfg
        self.n_cells = cfg["plates"] * cfg["wells"] * cfg["fields"] * cfg["cells"]

    def write(self, row):
        row = {"tier": self.tier, "backend": self.backend, "cells": self.n_cells,
               "width": self.cfg["width"], **row}
        with open(self.path, "a") as h:
            h.write(json.dumps(row, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)

    def once(self, module, op, api, fn, **extra):
        """A one-shot measurement (writes)."""
        with Peak() as pk:
            t0 = time.perf_counter()
            res = fn()
            dt = time.perf_counter() - t0
        self.write({"module": module, "op": op, "api": api, "cold_s": None,
                    "median_s": dt, "max_s": dt, "n": 1, "peak_rss_gb": pk.peak / 1e9,
                    **extra})
        return res

    def timed(self, module, op, api, fn, cold_path=None, repeats=3, note="", check=None):
        """Cold run (after dropping the page cache) plus warm repeats."""
        if (module, op) in self.done:
            print("skip (already measured):", module, op, flush=True)
            return None
        cold = None
        times, res = [], None
        with Peak() as pk:
            try:
                if cold_path:
                    drop_cache(cold_path)
                    t0 = time.perf_counter()
                    res = fn()
                    cold = time.perf_counter() - t0
                    if cold > SLOW_REPEAT_S:
                        repeats = 1
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    res = fn()
                    times.append(time.perf_counter() - t0)
                err = None
            except Exception as e:  # recorded, not fatal
                err = f"{type(e).__name__}: {str(e)[:300]}"
        info = check(res) if (check and err is None) else None
        self.write({"module": module, "op": op, "api": api, "cold_s": cold,
                    "median_s": statistics.median(times) if times else None,
                    "max_s": max(times) if times else None, "n": len(times),
                    "peak_rss_gb": pk.peak / 1e9, "error": err, "note": note,
                    "check": info})
        return res


def _training_pull(s, by_plate):
    """Store.query per plate x table (all features) -> numpy, h-stacked."""
    mats = []
    for t in TABLES:
        parts = []
        for p, keys in by_plate.items():
            _meta, x = s.query({"plate": p, "object_key": keys}, table=t, as_="numpy")
            parts.append(x)
        mats.append(np.concatenate(parts))
    return list(np.hstack(mats).shape)


def _training_lookup(s, by_plate):
    """Harness emulation of planned Store.lookup: locate in <table>__keys, then read only
    the touched row groups of every column (Store._read_located), whatever the share of
    row groups touched; arrow -> float32 matrix."""
    mats = []
    for t in TABLES:
        cols = [c for c in s._columns(t) if c not in KEY_COLUMNS + ("plate", "well", "parent_cell_key")]
        parts = []
        for p, keys in by_plate.items():
            ks = ", ".join(str(int(k)) for k in keys)
            located = s.sql(f'SELECT _part, _row_group, _group_row FROM "{t}__keys" '
                            f"WHERE plate = '{p}' AND object_key IN ({ks})", as_="arrow")
            tab = s._read_located(located, cols)
            parts.append(np.column_stack([tab.column(i).to_numpy().astype(np.float32, copy=False)
                                          for i in range(tab.num_columns)]))
        mats.append(np.concatenate(parts))
    return list(np.hstack(mats).shape)


def _guarded_child(kind, project, tier, q):
    """Spawned child: open the store, run one training pull, report time."""
    from spacr import store
    by_plate = {}
    for p, _w, k, _pr, _v in labelled_set(tier, TIERS[tier]):
        by_plate.setdefault(p, []).append(k)
    s = store.open(project)
    t0 = time.perf_counter()
    shape = (_training_pull if kind == "query" else _training_lookup)(s, by_plate)
    q.put((time.perf_counter() - t0, shape))


def guarded(rec, module, op, api, kind, project, guard_gb):
    """One cold run in a spawned child, killed when its RSS passes guard_gb."""
    if (module, op) in rec.done:
        return
    import multiprocessing as mp
    import psutil
    drop_cache(project)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_guarded_child, args=(kind, project, rec.tier, q))
    t0 = time.perf_counter()
    proc.start()
    peak, err, res = 0, None, None
    ps = psutil.Process(proc.pid)
    while proc.is_alive():
        try:
            peak = max(peak, ps.memory_info().rss)
        except Exception:
            pass
        if peak > guard_gb * 1e9:
            proc.kill()
            err = (f"exceeds RAM: child RSS passed the {guard_gb} GB guard after "
                   f"{time.perf_counter() - t0:.0f} s; killed")
            break
        try:
            res = q.get(timeout=0.05)
            break
        except Exception:
            pass
    proc.join(timeout=30)
    if res is None and err is None:
        try:
            res = q.get(timeout=1)
        except Exception:
            err = f"child exited with {proc.exitcode}"
    rec.write({"module": module, "op": op, "api": api, "cold_s": res[0] if res else None,
               "median_s": res[0] if res else None, "max_s": res[0] if res else None,
               "n": 1 if res else 0, "peak_rss_gb": peak / 1e9, "error": err,
               "note": "one cold run in a spawned child with a RAM guard (100M tier)",
               "check": res[1] if res else None})


# ----------------------------------------------------------------------------- store tier


def _write_well(args):
    tier, project, plate, well, r, c, cfg = args
    t0 = time.perf_counter()
    write_wells(project, cfg, [f"{plate}/{well}"])
    return 0.0, time.perf_counter() - t0


def write_edits(project, labels, kind="annotation"):
    """Harness emulation of planned store.edit: append-only edit log per plate, with the
    two-column stamp (seq_ns, seq_writer) of spacr.store._edit_schema."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spacr import store
    by_plate = {}
    for p, w, k, _prcfo, v in labels:
        by_plate.setdefault(p, []).append((k, v))
    seq0, writer = store._edit_stamp(1)
    for p, rows in by_plate.items():
        folder = os.path.join(project, "edits", kind, f"plate={p}")
        os.makedirs(folder, exist_ok=True)
        n = len(rows)
        tab = pa.table({
            "table": pa.array(["cell"] * n), "object_key": pa.array([k for k, _ in rows], pa.int64()),
            "timeID": pa.array(np.zeros(n, np.int64)), "column": pa.array(["annotation"] * n),
            "value_num": pa.array([float(v) for _, v in rows]), "value_text": pa.nulls(n, pa.string()),
            "op": pa.array(["set"] * n), "seq_ns": pa.array(np.arange(seq0, seq0 + n, dtype=np.int64)),
            "seq_writer": pa.array(np.full(n, writer, np.int32)),
            "author": pa.array(["bench"] * n), "source": pa.array(["import"] * n),
            "session": pa.nulls(n, pa.string()), "run_id": pa.array(["bench"] * n)},
            schema=store._edit_schema())
        pq.write_table(tab, os.path.join(folder, f"edit-{seq0:020d}-{uuid.uuid4().hex}.parquet"),
                       compression="zstd")


def _build_store(rec, tier, root, project, cfg, wells, workers, labels):
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spacr import store
    shutil.rmtree(os.path.dirname(project), ignore_errors=True)
    os.makedirs(project)
    # --- 0. a float16 column must be readable by DuckDB SQL (store hardening S0)
    probe = os.path.join(root, f"probe_{tier}")
    shutil.rmtree(probe, ignore_errors=True)
    tab = pa.table({"object_key": np.arange(100, dtype=np.int64),
                    "a": np.random.rand(100).astype(np.float16)})
    try:
        store.write_table(probe, "cell", tab, plate="p", well="A01", sort_by=("object_key",),
                          key_columns=("object_key",))
        store.register(probe)
        try:
            with store.open(probe) as ps:
                ps.sql("SELECT avg(a) FROM cell")
            s0 = "float16 written by the store is readable by DuckDB SQL"
        except Exception as e:
            s0 = f"float16 NOT readable by DuckDB SQL: {e}"
    finally:
        shutil.rmtree(probe, ignore_errors=True)
    rec.write({"module": "write", "op": "float16 readable check", "api": "Store API", "note": s0})

    # --- 1. write
    t_all = time.perf_counter()
    gen_s = write_s = 0.0
    with Peak() as pk:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, maxtasksperchild=50) as pool:
            for g, w in pool.imap_unordered(
                    _write_well, [(tier, project, p, w, r, c, cfg) for p, w, r, c in wells]):
                gen_s += g
                write_s += w
    wall = time.perf_counter() - t_all
    rec.write({"module": "write", "op": "generate (part of the combined write below)", "api": "harness",
               "median_s": gen_s / workers, "note": f"sum over workers {gen_s:.1f} s, {workers} workers"})
    rec.write({"module": "write", "op": "write_ops_screen (3 object tables, all wells)", "api": "Store API (power_ops.write_ops_screen -> store.writer)",
               "median_s": write_s / workers, "max_s": wall, "peak_rss_gb": pk.peak / 1e9,
               "note": f"{workers} worker processes, one well each; wall incl. generation {wall:.1f} s; "
                       f"worker-time sum {write_s:.1f} s"})
    rec.once("write", "register (one transaction, all manifests)", "Store API",
             lambda: store.register(project))
    rec.once("write", "write 10k annotations as edit log", "harness emulation of planned API (store.edit)",
             lambda: write_edits(project, labels))
    sizes = {"total": du(project), "catalog": du(os.path.join(project, store.CATALOG))}
    for t in TABLES:
        sizes[t] = du(os.path.join(project, "tables", t))
        sizes[t + "__keys"] = du(os.path.join(project, "tables", t + "__keys"))
    sizes["edits"] = du(os.path.join(project, "edits"))
    nfiles = sum(len(fs) for _a, _d, fs in os.walk(project))
    rec.write({"module": "disk", "op": "disk", "api": "Store API", "bytes": sizes, "files": nfiles,
               "note": f"{sizes['total'] / 1e9:.2f} GB, {nfiles} files"})



def run_store(tier, root, out, keep, workers, reuse=False):
    import multiprocessing as mp
    import pyarrow as pa
    import pyarrow.parquet as pq
    from spacr import store
    cfg = TIERS[tier]
    project = os.path.join(root, f"store_{tier}", "measurements")
    reuse = reuse and os.path.exists(os.path.join(project, store.CATALOG))
    rec = Recorder(out, tier, "store", cfg, append=reuse)
    wells = _wells(cfg["plates"], cfg["wells"])
    labels = labelled_set(tier, cfg)
    if not reuse:
        _build_store(rec, tier, root, project, cfg, wells, workers, labels)
    else:
        rec.write({"module": "write", "op": "reuse", "api": "harness",
                   "note": "resumed on the tier written by the killed run; write metrics are from that run"})
        # a killed run may have left a registered score table and its files; register leftovers
        store.register(project)

    # --- 2. open
    def _open():
        s = store.open(project)
        s.close()
    rec.timed("open", "store.open (catalog + registry + quick verify)", "Store API", _open, cold_path=project)
    s = store.open(project)
    rec.timed("open", "column list (spacr_columns)", "Store API (Store.columns)",
              lambda: len(s.columns("cell")), cold_path=project, check=lambda r: r)

    f16, f32 = _feature_names("cell", cfg["width"])
    feat_sort = f32[0]
    feats = f16[:13] + f32[:13]                      # 26 features + 4 keys = 30 visible columns
    visible = ["plate", "well", "object_key", "gene"] + feats
    vis_sql = ", ".join(f'"{c}"' for c in visible)
    files_cell = [f for f in s._files if f["table_name"] == "cell" and f["kind"] == "parquet"]

    # --- 3. Database Browser
    cols = feats
    first_page = lambda: s.page("cell", columns=cols, limit=100)
    api_page = "Store API (Store.page, keyset)"
    rec.timed("browser", "first page 100 x 30 cols", api_page, first_page, cold_path=project,
              check=lambda r: len(r[0]))
    # page 1,000: the cursor of the row 99,900 positions in (a client holds the previous page's
    # cursor); built from the registry's well order and one key read, then one page call
    deep = max(0, min(99_900, rec.n_cells - 100))
    holder = {}

    def deep_cursor():
        if "c" not in holder:
            left = deep
            for f in files_cell:
                if left < int(f["rows"]):
                    key = s.sql(f"SELECT object_key FROM \"cell__keys\" WHERE plate = '{f['plate']}' AND "
                                f"well = '{f['well']}' ORDER BY object_key LIMIT 1 OFFSET {max(left - 1, 0)}"
                                ).object_key[0]
                    holder["c"] = {"plate": f["plate"], "well": f["well"], "after": [int(key)]}
                    break
                left -= int(f["rows"])
        return holder["c"]
    deep_cursor()
    rec.timed("browser", "page 1,000 (rows 99,900-99,999)", api_page,
              lambda: s.page("cell", columns=cols, limit=100, after=deep_cursor()), cold_path=project,
              check=lambda r: len(r[0]), note="" if rec.n_cells >= 100_000 else "tier has < 1,000 pages")
    rec.timed("browser", "sorted by 1 feature, first page 100", "Store API (Store.page, order_by)",
              lambda: s.page("cell", columns=cols, order_by=(feat_sort, "desc"), limit=100),
              cold_path=project, check=lambda r: len(r[0]))
    rec.timed("browser", "page 1,000 via SQL OFFSET", "Store.sql",
              lambda: s.sql(f"SELECT {vis_sql} FROM cell LIMIT 100 OFFSET "
                            f"{min(99_900, rec.n_cells - 100)}"), cold_path=project, check=len)
    rec.timed("browser", "sort by 1 feature, top 100", "Store.sql",
              lambda: s.sql(f'SELECT {vis_sql} FROM cell ORDER BY "{feat_sort}" DESC LIMIT 100'),
              cold_path=project, check=len)
    lo, hi = math.exp(6.0), math.exp(6.5)  # ~16 % of cells
    rec.timed("browser", "filter 1 feature range, first page", "Store.sql",
              lambda: s.sql(f'SELECT {vis_sql} FROM cell WHERE "{feat_sort}" BETWEEN {lo} AND {hi} LIMIT 100'),
              cold_path=project, check=len)
    rec.timed("browser", "filter 1 feature range, count", "Store API (Store.count)",
              lambda: s.count("cell", f'"{feat_sort}" BETWEEN {lo} AND {hi}'), cold_path=project,
              check=lambda r: r)
    rec.timed("browser", "distinct values of gene", "Store.sql (cell__keys view)",
              lambda: s.sql('SELECT DISTINCT gene FROM "cell__keys"'), cold_path=project, check=len)
    rec.timed("browser", "total count (registry)", "Store API (Store.count)",
              lambda: s.count("cell"), cold_path=None, check=lambda r: r)
    rec.timed("browser", "total count (SQL count(*))", "Store.sql",
              lambda: s.sql("SELECT count(*) n FROM cell"), cold_path=project, check=lambda r: int(r.n[0]))

    # --- 4. Annotation app
    plate0, well0 = wells[len(wells) // 2][:2]
    journal_dir = os.path.join(project, "edits", "_journal")
    os.makedirs(journal_dir, exist_ok=True)
    journal = os.path.join(journal_dir, f"bench-{uuid.uuid4().hex}.jsonl")
    a3 = f16[:2] + f32[:1]

    def ann_view(plate, well):
        """Harness emulation of the planned cell_labeled view (DESIGN 2.4): edit log
        Parquet + open journals, latest (seq_ns, seq_writer) wins; one well, key order, 500 rows."""
        edits = os.path.join(project, "edits", "annotation", f"plate={plate}", "*.parquet")
        srcs = [f"SELECT object_key, value_num, op, seq_ns, seq_writer FROM read_parquet('{edits}') "
                f"WHERE \"table\" = 'cell' AND \"column\" = 'annotation'"]
        if os.path.exists(journal) and os.path.getsize(journal):
            srcs.append(f"SELECT object_key, value_num, op, seq_ns, seq_writer FROM read_json('{journal}', "
                        f"columns = {{plate: 'VARCHAR', object_key: 'BIGINT', value_num: 'DOUBLE', "
                        f"op: 'VARCHAR', seq_ns: 'BIGINT', seq_writer: 'INTEGER'}}) WHERE plate = '{plate}'")
        q = (f"WITH pg AS (SELECT object_key, {', '.join(chr(34) + c + chr(34) for c in a3)} "
             f"FROM cell WHERE plate = '{plate}' AND well = '{well}' ORDER BY object_key LIMIT 500), "
             f"lab AS (SELECT object_key, value_num, op FROM ({' UNION ALL '.join(srcs)}) "
             f"QUALIFY row_number() OVER (PARTITION BY object_key ORDER BY seq_ns DESC, seq_writer DESC) = 1) "
             f"SELECT pg.*, CASE WHEN lab.op = 'set' THEN lab.value_num END AS annotation "
             f"FROM pg LEFT JOIN lab USING (object_key) ORDER BY object_key")
        return s.sql(q)

    em_view = "Store.sql + harness emulation of planned API (cell_labeled view)"
    page0 = rec.timed("annotate", "list 500 objects of 1 well, key order, 3 features + labels", em_view,
                      lambda: ann_view(plate0, well0), cold_path=project, check=len)

    def journal_add(key, value):
        """Harness emulation of planned store.journal(...).add: one fsynced JSON line."""
        with open(journal, "a") as h:
            h.write(json.dumps({"plate": plate0, "table": "cell", "object_key": int(key), "timeID": 0,
                                "column": "annotation", "value_num": float(value), "op": "set",
                                "seq_ns": store._edit_stamp(1)[0], "seq_writer": 1, "author": "bench",
                                "source": "annotator"}) + "\n")
            h.flush()
            os.fsync(h.fileno())

    if page0 is None:  # skipped in --reuse mode
        page0 = ann_view(plate0, well0)
    keys0 = page0.object_key.to_numpy()
    counter = iter(range(10 ** 6))
    rec.timed("annotate", "record 1 annotation (journal append + fsync)",
              "harness emulation of planned API (store.journal)",
              lambda: journal_add(keys0[next(counter) % len(keys0)], 1), cold_path=None)

    def roundtrip():
        k = int(keys0[next(counter) % len(keys0)])
        v = float(next(counter) % 7 + 2)
        journal_add(k, v)
        view = ann_view(plate0, well0)
        shown = view.loc[view.object_key == k, "annotation"].iloc[0]
        assert shown == v, (shown, v)
        return True
    rec.timed("annotate", "record + latency until view shows it", em_view, roundtrip,
              cold_path=project, check=lambda r: r)

    # --- 5. Classify
    by_plate = {}
    for p, _w, k, _pr, v in labels:
        by_plate.setdefault(p, []).append(k)

    big = rec.n_cells >= 50_000_000
    if big:
        guarded(rec, "classify", "pull 10k labelled cells x all features", "Store API (Store.query, numpy)",
                "query", project, GUARD_GB)
    else:
        rec.timed("classify", "pull 10k labelled cells x all features", "Store API (Store.query, numpy)",
                  lambda: _training_pull(s, by_plate), cold_path=project, check=lambda r: r)

    rng = np.random.default_rng(1)
    w = {t: rng.standard_normal(len(sum(_feature_names(t, cfg["width"]), []))).astype(np.float32) * 0.01
         for t in TABLES}
    counts = [(f["plate"], f["well"], int(f["rows"])) for f in files_cell]

    def score_all():
        """Store.batches (numpy) per table, streamed; a linear model decomposes over the
        three tables, so each table streams independently into one logit vector in
        registry order. Then the derived score table is written per well
        (store.write_table) and registered."""
        n = sum(c for *_x, c in counts)
        logit = np.zeros(n, np.float64)
        okey = np.empty(n, np.int64)
        for t in TABLES:
            names = sum(_feature_names(t, cfg["width"]), [])
            pos = 0
            for meta, x in s.batches(table=t, features=names, batch_size=65536, as_="numpy"):
                m = len(x)
                logit[pos:pos + m] += x.astype(np.float32, copy=False) @ w[t]
                k = meta["object_key"].to_numpy()
                if t == "cell":
                    okey[pos:pos + m] = k
                elif not np.array_equal(okey[pos:pos + m], k):
                    raise RuntimeError(f"{t} rows out of order with cell at {pos}")
                pos += m
            assert pos == n, (t, pos, n)
        score = (1 / (1 + np.exp(-logit))).astype(np.float32)
        pos = 0
        for p, wl, c in counts:
            store.write_table(project, SCORE_TABLE, pa.table({
                "object_key": okey[pos:pos + c], "timeID": np.zeros(c, np.uint16),
                "score": score[pos:pos + c], "pred": (score[pos:pos + c] > 0.5).astype(np.uint8),
                "model_id": pa.array(["bench"] * c).dictionary_encode()}),
                plate=p, well=wl, sort_by=("object_key", "timeID"), key_columns=("object_key", "timeID"),
                run_id="bench")
            pos += c
        store.register(project)
        return n
    rec.timed("classify", "score all cells (stream) + write score table", "Store API (Store.batches, store.write_table)",
              score_all, cold_path=project, check=lambda r: r)
    s.refresh()

    # --- 6. Graph builder
    g1, g2 = f16[0], f32[1]
    agg = f'avg("{g1}") m1, stddev_samp("{g1}") s1, avg("{g2}") m2, stddev_samp("{g2}") s2, count(*) n'
    rec.timed("graph", "mean/SD of 2 features per well", "Store.sql",
              lambda: s.sql(f"SELECT plate, well, {agg} FROM cell GROUP BY plate, well"), cold_path=project,
              check=len)
    rec.timed("graph", "mean/SD of 2 features per gene", "Store.sql",
              lambda: s.sql(f"SELECT gene, {agg} FROM cell GROUP BY gene"), cold_path=project, check=len)
    rec.timed("graph", "histogram of 1 feature, all cells (100 bins)", "Store.sql",
              lambda: s.sql(f'WITH r AS (SELECT min("{g2}") lo, max("{g2}") hi FROM cell) '
                            f'SELECT least(99, floor(("{g2}" - lo) / (hi - lo) * 100))::INT b, count(*) n '
                            f'FROM cell, r GROUP BY b ORDER BY b'), cold_path=project, check=len)

    # --- 6b. before-S0 probes: the same files read through a view with union_by_name = true,
    # which is what the catalog views did before every part of a table had one schema
    paths_cell = [os.path.join(project, f["path"]) for f in files_cell]
    listed = ", ".join("'" + x + "'" for x in paths_cell)
    s._conn.execute(f"CREATE OR REPLACE VIEW cell_before AS SELECT * FROM read_parquet([{listed}], "
                    f"hive_partitioning = true, hive_types = {{'plate': VARCHAR, 'well': VARCHAR}}, "
                    f"union_by_name = true)")
    fx = "before S0: the same view with union_by_name = true"
    rec.timed("browser", "total count (SQL count(*)) [before S0]", fx,
              lambda: s.sql("SELECT count(*) n FROM cell_before"), cold_path=project, check=lambda r: int(r.n[0]))
    rec.timed("browser", "filter 1 feature range, count [before S0]", fx,
              lambda: s.sql(f'SELECT count(*) n FROM cell_before WHERE "{feat_sort}" BETWEEN {lo} AND {hi}'),
              cold_path=project, check=lambda r: int(r.n[0]))
    rec.timed("browser", "sort by 1 feature, top 100 [before S0]", fx,
              lambda: s.sql(f'SELECT {vis_sql} FROM cell_before ORDER BY "{feat_sort}" DESC LIMIT 100'),
              cold_path=project, check=len)
    rec.timed("graph", "mean/SD of 2 features per well [before S0]", fx,
              lambda: s.sql(f"SELECT plate, well, {agg} FROM cell_before GROUP BY plate, well"), cold_path=project,
              check=len)
    rec.timed("graph", "histogram of 1 feature, all cells (100 bins) [before S0]", fx,
              lambda: s.sql(f'WITH r AS (SELECT min("{g2}") lo, max("{g2}") hi FROM cell_before) '
                            f'SELECT least(99, floor(("{g2}" - lo) / (hi - lo) * 100))::INT b, count(*) n '
                            f'FROM cell_before, r GROUP BY b ORDER BY b'), cold_path=project, check=len)

    if big:
        guarded(rec, "classify", "pull 10k labelled cells x all features [fix probe]",
                "harness emulation of planned API (Store.lookup: keys locate + row-group read)",
                "lookup", project, GUARD_GB)
    else:
        rec.timed("classify", "pull 10k labelled cells x all features [fix probe]",
                  "harness emulation of planned API (Store.lookup: keys locate + row-group read)",
                  lambda: _training_lookup(s, by_plate), cold_path=project, check=lambda r: r)

    # --- 7. single-cell lookup
    pick = labels[len(labels) // 3]

    def lookup():
        rows = [s.query({"plate": pick[0], "object_key": pick[2]}, table=t, as_="arrow") for t in TABLES]
        return sum(r.num_columns for r in rows), [r.num_rows for r in rows]
    rec.timed("single cell", "1 cell, all features (3 tables)", "Store API (Store.query, routed)",
              lookup, cold_path=project, check=lambda r: list(r))
    s.close()

    total_gb = du(project) / 1e9
    rec.write({"module": "disk", "op": "disk after score table", "api": "Store API",
               "bytes": {"total": du(project)}, "note": f"{total_gb:.2f} GB"})
    if not keep and tier != "10k":
        shutil.rmtree(os.path.dirname(project), ignore_errors=True)
        rec.write({"module": "disk", "op": "tier data deleted", "api": "harness"})

# ----------------------------------------------------------------------------- legacy tier


def _plain_pandas(tab):
    """Arrow -> pandas with plain int64/float64/str columns, as Measure's frames hold."""
    import pyarrow as pa
    cols = {}
    for f in tab.schema:
        c = tab.column(f.name)
        if pa.types.is_dictionary(f.type):
            c = c.cast(pa.string())
        elif pa.types.is_integer(f.type):
            c = c.cast(pa.int64())
        cols[f.name] = c
    return pa.table(cols).to_pandas()


def run_legacy(tier, root, out):
    import pandas as pd
    from spacr.utils import _append_frame
    cfg = TIERS[tier]
    rec = Recorder(out, tier, "legacy", cfg)
    folder = os.path.join(root, f"legacy_{tier}", "measurements")
    shutil.rmtree(os.path.dirname(folder), ignore_errors=True)
    os.makedirs(folder)
    db = os.path.join(folder, "measurements.db")
    wells = _wells(cfg["plates"], cfg["wells"])
    labels = labelled_set(tier, cfg)

    # --- write: spaCR's own append path (_append_frame: parameterised executemany),
    # one call per field as measure_crop does, then the OPS legacy indexes.
    def write():
        conn = sqlite3.connect(db, timeout=60)
        try:
            for p, wl, r, c in wells:
                frames, _k = well_frames(tier, p, wl, r, c, cfg, as_float64=True)
                n_c = cfg["cells"]
                for t, tab in frames.items():
                    df = _plain_pandas(tab)
                    df.insert(0, "plate", p)
                    df.insert(1, "well", wl)
                    for f0 in range(0, len(df), n_c):
                        _append_frame(conn, t, df.iloc[f0:f0 + n_c])
                pl = _plain_pandas(frames["cell"].select(["prcfo", "rowID", "columnID", "fieldID",
                                                          "object_label", "gene"]))
                pl.insert(0, "png_path", [f"/data/{p}/data/{x}.png" for x in pl.prcfo])
                pl.insert(1, "plate", p)
                pl["annotation"] = np.nan
                _append_frame(conn, "png_list", pl)
            for t in TABLES:
                conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{t}_prcfo ON {t}(prcfo)")
                conn.execute(f"CREATE INDEX IF NOT EXISTS ix_{t}_field ON {t}(plate, rowID, columnID, fieldID)")
            conn.execute('CREATE INDEX IF NOT EXISTS "idx_png_path" ON png_list (png_path)')
            conn.executemany("UPDATE png_list SET annotation = ? WHERE png_path = ?",
                             [(v, f"/data/{p}/data/{pr}.png") for p, _w, _k, pr, v in labels])
            conn.commit()
        finally:
            conn.close()
    rec.once("write", "INSERT via spacr.utils._append_frame + indexes", "legacy sqlite3", write)
    size = os.path.getsize(db)
    rec.write({"module": "disk", "op": "disk", "api": "legacy sqlite3", "bytes": {"total": size},
               "note": f"{size / 1e9:.2f} GB"})

    def _open():
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        cols = conn.execute('PRAGMA table_info("cell")').fetchall()
        conn.execute("SELECT max(_rowid_) FROM cell").fetchone()
        conn.close()
        return len(cols)
    rec.timed("open", "connect + table list + column list + max(_rowid_)", "legacy sqlite3", _open, cold_path=db,
              check=lambda r: r)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
    q = lambda sql, *p: conn.execute(sql, p).fetchall()
    f16, f32 = _feature_names("cell", cfg["width"])
    feat_sort = f32[0]
    visible = ["plate", "well", "object_key", "gene"] + f16[:13] + f32[:13]
    vis = ", ".join(f'"{c}"' for c in visible)
    n = rec.n_cells
    rec.timed("browser", "first page 100 x 30 cols", "legacy sqlite3 (rowid keyset)",
              lambda: q(f"SELECT _rowid_, {vis} FROM cell WHERE _rowid_ > 0 ORDER BY _rowid_ LIMIT 100"),
              cold_path=db, check=len)
    rec.timed("browser", "page 1,000 (rows 99,900-99,999)", "legacy sqlite3 (rowid keyset)",
              lambda: q(f"SELECT _rowid_, {vis} FROM cell WHERE _rowid_ > ? ORDER BY _rowid_ LIMIT 100",
                        min(99_900, n - 100)), cold_path=db, check=len)
    rec.timed("browser", "page 1,000 via SQL OFFSET", "legacy sqlite3",
              lambda: q(f"SELECT _rowid_, {vis} FROM cell ORDER BY _rowid_ LIMIT 100 OFFSET ?", min(99_900, n - 100)),
              cold_path=db, check=len)
    rec.timed("browser", "sort by 1 feature, top 100", "legacy sqlite3",
              lambda: q(f'SELECT _rowid_, {vis} FROM cell ORDER BY "{feat_sort}" DESC LIMIT 100'), cold_path=db,
              check=len)
    lo, hi = math.exp(6.0), math.exp(6.5)
    rec.timed("browser", "filter 1 feature range, first page", "legacy sqlite3",
              lambda: q(f'SELECT _rowid_, {vis} FROM cell WHERE "{feat_sort}" BETWEEN ? AND ? '
                        f'ORDER BY _rowid_ LIMIT 100', lo, hi), cold_path=db, check=len)
    rec.timed("browser", "filter 1 feature range, count", "legacy sqlite3",
              lambda: q(f'SELECT count(*) FROM cell WHERE "{feat_sort}" BETWEEN ? AND ?', lo, hi), cold_path=db,
              check=lambda r: r[0][0])
    rec.timed("browser", "distinct values of gene", "legacy sqlite3",
              lambda: q("SELECT DISTINCT gene FROM cell"), cold_path=db, check=len)
    rec.timed("browser", "total count (SQL count(*))", "legacy sqlite3",
              lambda: q("SELECT count(*) FROM cell"), cold_path=db, check=lambda r: r[0][0])
    rec.timed("browser", "total count (registry)", "legacy sqlite3 (max(_rowid_), Browser's first paint)",
              lambda: q("SELECT max(_rowid_) FROM cell"), cold_path=db, check=lambda r: r[0][0])

    # annotation app
    p0, w0, r0, c0 = wells[len(wells) // 2]
    a3 = f16[:2] + f32[:1]
    a3s = ", ".join(f'c."{x}"' for x in a3)
    wconn = sqlite3.connect(db, timeout=30, check_same_thread=False)

    def ann_view():
        return q(f"SELECT p.png_path, p.annotation, {a3s} FROM png_list p JOIN cell c ON c.prcfo = p.prcfo "
                 f"WHERE p.plate = ? AND p.rowID = ? AND p.columnID = ? ORDER BY p._rowid_ LIMIT 500", p0, r0, c0)
    page0 = rec.timed("annotate", "list 500 objects of 1 well, key order, 3 features + labels",
                      "legacy sqlite3 (png_list JOIN cell)", ann_view, cold_path=db, check=len)
    paths = [r[0] for r in page0]
    counter = iter(range(10 ** 6))

    def record(v=1):
        pth = paths[next(counter) % len(paths)]
        wconn.execute('UPDATE png_list SET annotation = ? WHERE png_path = ?', (v, pth))
        wconn.commit()
        return pth
    rec.timed("annotate", "record 1 annotation (journal append + fsync)", "legacy sqlite3 (UPDATE + commit)",
              record, cold_path=None)

    def roundtrip():
        v = next(counter) % 7 + 2
        pth = record(v)
        got = dict((r[0], r[1]) for r in ann_view())[pth]
        assert got == v
        return True
    rec.timed("annotate", "record + latency until view shows it", "legacy sqlite3", roundtrip, cold_path=db,
              check=lambda r: r)

    lab_prcfo = [pr for _p, _w, _k, pr, _v in labels]

    def pull_indexed():
        out = []
        for t in TABLES:
            rows = []
            for i in range(0, len(lab_prcfo), 900):
                ch = lab_prcfo[i:i + 900]
                rows += conn.execute(f"SELECT * FROM {t} WHERE prcfo IN ({','.join('?' * len(ch))})",
                                     ch).fetchall()
            out.append(len(rows))
        return out
    rec.timed("classify", "pull 10k labelled cells x all features (indexed IN, best SQLite)",
              "legacy sqlite3 (prcfo index)", pull_indexed, cold_path=db, check=lambda r: r)

    rng = np.random.default_rng(1)

    def score_all():
        logit = np.zeros(n, np.float64)
        keys = None
        for t in TABLES:
            names = sum(_feature_names(t, cfg["width"]), [])
            wv = rng.standard_normal(len(names)) * 0.01
            cur = conn.execute(f"SELECT prcfo, {', '.join(chr(34) + c + chr(34) for c in names)} FROM {t} "
                               f"ORDER BY _rowid_")
            pos, kk = 0, []
            while True:
                rows = cur.fetchmany(65536)
                if not rows:
                    break
                x = np.array([r[1:] for r in rows], dtype=np.float32)
                logit[pos:pos + len(rows)] += x @ wv
                if t == "cell":
                    kk += [r[0] for r in rows]
                pos += len(rows)
            if t == "cell":
                keys = kk
        score = 1 / (1 + np.exp(-logit))
        wconn.execute("DROP TABLE IF EXISTS score__bench")
        wconn.execute("CREATE TABLE score__bench (prcfo TEXT, score REAL, pred INTEGER)")
        wconn.executemany("INSERT INTO score__bench VALUES (?, ?, ?)",
                          zip(keys, score.tolist(), (score > 0.5).astype(int).tolist()))
        wconn.commit()
        return n
    rec.timed("classify", "score all cells (stream) + write score table", "legacy sqlite3 (fetchmany, executemany)",
              score_all, cold_path=db, check=lambda r: r)

    g1, g2 = f16[0], f32[1]
    agg = (f'avg("{g1}"), avg("{g1}"*"{g1}") - avg("{g1}")*avg("{g1}"), avg("{g2}"), '
           f'avg("{g2}"*"{g2}") - avg("{g2}")*avg("{g2}"), count(*)')
    rec.timed("graph", "mean/SD of 2 features per well", "legacy sqlite3",
              lambda: q(f"SELECT plate, well, {agg} FROM cell GROUP BY plate, well"), cold_path=db, check=len)
    rec.timed("graph", "mean/SD of 2 features per gene", "legacy sqlite3",
              lambda: q(f"SELECT gene, {agg} FROM cell GROUP BY gene"), cold_path=db, check=len)
    rec.timed("graph", "histogram of 1 feature, all cells (100 bins)", "legacy sqlite3",
              lambda: q(f'WITH r AS (SELECT min("{g2}") lo, max("{g2}") hi FROM cell) '
                        f'SELECT min(99, CAST(("{g2}" - lo) / (hi - lo) * 100 AS INT)) b, count(*) '
                        f'FROM cell, r GROUP BY b ORDER BY b'), cold_path=db, check=len)
    pick = labels[len(labels) // 3]
    rec.timed("single cell", "1 cell, all features (3 tables)", "legacy sqlite3 (prcfo index)",
              lambda: [q(f"SELECT * FROM {t} WHERE prcfo = ?", pick[3]) for t in TABLES], cold_path=db,
              check=lambda r: [len(x[0]) for x in r])
    # classify: spaCR today reads whole tables with spacr.io._read_db, then merges in pandas
    from spacr.io import _read_db
    before = [r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(cell)')]
    if n <= 10_000:
        def pull_today():
            dfs = _read_db(db, list(TABLES) + ["png_list"])
            merged = dfs[3][dfs[3].annotation.notna()][["prcfo", "annotation"]]
            for d in dfs[:3]:
                merged = merged.merge(d, on="prcfo", how="left", suffixes=("", "_dup"))
            return merged.shape
        rec.timed("classify", "pull 10k labelled cells x all features",
                  "legacy sqlite3 (spacr.io._read_db full tables + merge)", pull_today, cold_path=db,
                  check=lambda r: list(r))
    else:
        rec.timed("classify", "pull 10k labelled cells: _read_db of the cell table alone",
                  "legacy sqlite3 (spacr.io._read_db)", lambda: _read_db(db, ["cell"])[0].shape,
                  cold_path=db, repeats=1, check=lambda r: list(r),
                  note="spaCR today reads all 3 tables (x2.5 the cell table's columns) and merges; the full "
                       "path needs ~2.5x this time and RAM; measured on cell only to stay under the 40 GB cap")
    after = [r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(cell)')]
    rec.write({'module': 'classify', 'op': '_read_db schema side effect', 'api': 'legacy sqlite3',
               'note': f'cell columns before {len(before)}, after {len(after)}; removed {sorted(set(before)-set(after))[:10]}, added {sorted(set(after)-set(before))[:10]}'})
    conn.close()
    wconn.close()
    if tier != "10k":
        shutil.rmtree(os.path.dirname(folder), ignore_errors=True)
        rec.write({"module": "disk", "op": "tier data deleted", "api": "harness"})

# ----------------------------------------------------------------------------- report

ORDER = ["10k", "1M", "1M_w300", "1M_w300_rc", "10M", "100M"]


def verdict(sec):
    if sec is None:
        return "n/a"
    if sec < 0.3:
        return "interactive"
    if sec < 3:
        return "fast"
    if sec < 60:
        return "slow"
    return "unusable"


def report(out):
    import pandas as pd
    rows = []
    for p in sorted(glob.glob(os.path.join(out, "metrics_*.jsonl"))):
        with open(p) as h:
            rows += [json.loads(x) for x in h if x.strip()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "metrics.csv"), index=False)
    ops = df[df.module.isin(["open", "browser", "annotate", "classify", "graph", "single cell", "write"])
             & df.median_s.notna()].copy()
    ops["verdict"] = ops.median_s.map(verdict)
    lines = ["# spacr-store scaling benchmark", "",
             "Times in seconds: warm median (max) [cold]. Verdict on the warm median: interactive < 0.3 s, "
             "fast < 3 s, slow < 60 s, unusable >= 60 s.", ""]
    key = ["module", "op"]
    tiers = [t for t in ORDER if t in set(ops.tier)]
    hdr = "| module | operation | " + " | ".join(f"{t} store | {t} legacy" for t in tiers) + " |"
    lines += [hdr, "|" + "---|" * (2 + 2 * len(tiers))]
    seen = []
    for _i, r in ops.iterrows():
        k = (r.module, r.op)
        if k not in seen:
            seen.append(k)

    def cell(sub):
        if sub.empty:
            return "–"
        r = sub.iloc[0]
        if isinstance(r.get("error"), str):
            return "ERROR"
        txt = f"{r.median_s:.3g}"
        if r.n and r.n > 1 and r.max_s is not None:
            txt += f" ({r.max_s:.3g})"
        if pd.notna(r.get("cold_s")):
            txt += f" [{r.cold_s:.3g}]"
        return txt + f" {r.verdict[:5]}"
    for m, o in seen:
        cells = []
        for t in tiers:
            for b in ("store", "legacy"):
                cells.append(cell(ops[(ops.module == m) & (ops.op == o) & (ops.tier == t) & (ops.backend == b)]))
        lines.append(f"| {m} | {o} | " + " | ".join(cells) + " |")
    disk = df[df.op == "disk"]
    lines += ["", "## Disk per tier", "", "| tier | cells | width | backend | GB | files |", "|---|---|---|---|---|---|"]
    for _i, r in disk.iterrows():
        lines.append(f"| {r.tier} | {r.cells:,} | {r.width} | {r.backend} | {r.bytes['total'] / 1e9:.2f} | "
                     f"{r.get('files', '') if pd.notna(r.get('files', None)) else ''} |")
    notes = df[df.note.notna() & (df.note != "")] if "note" in df else df.iloc[:0]
    lines += ["", "## Notes", ""] + [f"- {r.tier} {r.backend} {r.op}: {r.note}" for _i, r in notes.iterrows()]
    errs = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    lines += ["", "## Errors", ""] + [f"- {r.tier} {r.backend} {r.op}: {r.error}" for _i, r in errs.iterrows()]
    with open(os.path.join(out, "metrics.md"), "w") as h:
        h.write("\n".join(lines) + "\n")
    charts(df, ops, out)
    print("\n".join(lines))


def charts(df, ops, out):
    sys.path.insert(0, os.path.expanduser("~/.claude/skills/spacr-graphs"))
    import spacr_figstyle as sfs
    import matplotlib.pyplot as plt
    sfs.apply_style()
    # Panel A: time per key operation vs cells (log-log), one square panel per module.
    key_ops = {
        "open": [("store.open (catalog + registry + quick verify)", "open store"),
                 ("connect + table list + column list + max(_rowid_)", "open SQLite")],
        "browser": [("first page 100 x 30 cols", "first page"), ("sort by 1 feature, top 100", "sort, top 100"),
                    ("filter 1 feature range, count", "filtered count"), ("total count (SQL count(*))", "count(*)")],
        "annotate": [("list 500 objects of 1 well, key order, 3 features + labels", "list 500 of a well"),
                     ("record + latency until view shows it", "edit visible")],
        "classify": [("pull 10k labelled cells x all features", "10k training set"),
                     ("score all cells (stream) + write score table", "score all cells")],
        "graph": [("mean/SD of 2 features per well", "per-well mean/SD"),
                  ("histogram of 1 feature, all cells (100 bins)", "histogram")],
        "single cell": [("1 cell, all features (3 tables)", "1 cell, all features")],
    }
    mods = list(key_ops)
    main = ops[ops.tier != "1M_w300_rc"]
    fig = sfs.figure(7.2, 6.4)
    size, gapx = 1.6, 0.75
    from matplotlib.lines import Line2D
    for i, m in enumerate(mods):
        col, row = i % 3, i // 3
        ax = sfs.square_axes(fig, 0.6 + col * (size + gapx), 6.4 - 0.3 - size - row * (size + 1.45), size)
        handles = []
        for j, (o, short) in enumerate(key_ops[m]):
            colr = sfs.SERIES[j]
            for b, ls, mk in (("store", "-", "o"), ("legacy", (0, (2, 1.5)), "s")):
                for wdt in (2800, 300):
                    d = main[(main.module == m) & (main.op == o) & (main.backend == b)
                             & (main.width == wdt)].sort_values("cells")
                    if not d.empty:
                        ax.plot(d.cells, d.median_s, ls=ls, marker=mk, ms=2.5, lw=sfs.DATA_LW, color=colr,
                                alpha=sfs.ALPHA)
            handles.append(Line2D([], [], color=colr, alpha=sfs.ALPHA, lw=sfs.DATA_LW, label=short))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(5e3, 2e8)
        ax.set_ylim(1e-4, 1e4)
        for y in (0.3, 3):
            ax.axhline(y, **sfs.REFLINE)
        ax.set_xticks([1e4, 1e6, 1e8])
        ax.set_yticks([1e-4, 1e-2, 1, 1e2, 1e4])
        ax.set_title(m, fontsize=sfs.FONT_PT)
        if col == 0:
            ax.set_ylabel("time (s)")
        ax.set_xlabel("cells")
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, -0.28), frameon=False,
                  fontsize=sfs.SMALL_PT, handlelength=1.4, borderaxespad=0)
    fig.savefig(os.path.join(out, "time_vs_cells.png"), dpi=300)
    fig.savefig(os.path.join(out, "time_vs_cells.svg"))
    print("check time_vs_cells:", sfs.check_figure(fig))
    plt.close(fig)
    # legend file for the op numbers
    with open(os.path.join(out, "time_vs_cells_legend.md"), "w") as h:
        h.write("DEVELOPMENT PANEL – made by a script.\n\nTime per operation (warm median, log-log) against the "
                "number of cells, per module. Solid lines with circles: spacr-store (Parquet + DuckDB catalog); "
                "dashed lines with squares: legacy SQLite measurements.db (10k and 1M only). Dotted reference lines: "
                "0.3 s (interactive) and 3 s (fast). Two width series are drawn as separate lines: 2,800 feature "
                "columns (10k, 1M) and 300 columns (1M_w300, 10M, 100M); the two 1M points show the width effect. "
                "Only the key operations are drawn; every operation, the fix probes and the row-copy tier are in "
                "metrics.md. The 100M training-set pull has no point: both variants passed the 16 GB RAM guard and "
                "were killed. Colours name the operation within a panel (key under each panel):\n\n")
        for m in mods:
            for o, short in key_ops[m]:
                h.write(f"- {m}, {short}: {o}\n")
    # Panel B: disk per tier (bar per tier/backend, grey = legacy reference)
    disk = df[df.op == "disk"].copy()
    disk["gb"] = disk.bytes.map(lambda b: b["total"] / 1e9)
    disk = disk.set_index(["tier", "backend"]).gb
    labels_, vals, cols = [], [], []
    for t in ORDER:
        for b in ("legacy", "store"):
            if (t, b) in disk.index:
                labels_.append(f"{t}\n{b}")
                vals.append(disk[(t, b)])
                cols.append(sfs.CONTROL if b == "legacy" else sfs.BLUE)
    fig = sfs.figure(0.9 + len(vals) * sfs.SLOT_IN + 1.3, 2.6)
    ax = sfs.cond_axes(fig, 0.75, 0.75, len(vals), 1.6)
    x = np.arange(len(vals))
    ax.bar(x, vals, width=sfs.BAR_FRAC, color=cols, alpha=sfs.ALPHA)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1000)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split("\n")[0] for l in labels_], rotation=90)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_ylabel("disk (GB)")
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.3, f"{v:.3g}", ha="center", va="bottom", rotation=90, fontsize=sfs.SMALL_PT)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=sfs.CONTROL, alpha=sfs.ALPHA, label="legacy (ref.)"),
                       Patch(color=sfs.BLUE, alpha=sfs.ALPHA, label="store")], frameon=False,
              loc="upper left", fontsize=sfs.SMALL_PT, bbox_to_anchor=(1.0, 1.0))
    fig.savefig(os.path.join(out, "disk_per_tier.png"), dpi=300)
    fig.savefig(os.path.join(out, "disk_per_tier.svg"))
    print("check disk_per_tier:", sfs.check_figure(fig))
    plt.close(fig)
    with open(os.path.join(out, "disk_per_tier_legend.md"), "w") as h:
        h.write("DEVELOPMENT PANEL – made by a script.\n\nDisk per tier (GB, log scale). Grey: legacy SQLite "
                "measurements.db (reference; float64 REAL columns, indexes on prcfo and field). Blue: spacr-store "
                "(Parquet zstd, 60 % float16 / 40 % float32, keys tables, catalog, 10k edit rows). Random normal "
                "synthetic values, a worst case for compression. Widths as in time_vs_cells.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["store", "legacy", "report"])
    ap.add_argument("--tier", choices=list(TIERS))
    ap.add_argument("--root", default="/mnt/wd4tb/spacr_store_bench")
    ap.add_argument("--out", default="spacr_store_bench_results")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--calibration", default=None,
                    help="a calibration JSON from spacr.power_ops.calibrate_from_run; default: nominal")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--reuse", action="store_true", help="resume on an existing tier (skip writes and done ops)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SCHED_THREADS", "4")))
    a = ap.parse_args()
    global CALIBRATION, SEED
    CALIBRATION, SEED = a.calibration, a.seed
    if a.cmd == "store":
        run_store(a.tier, a.root, a.out, a.keep, a.workers, a.reuse)
    elif a.cmd == "legacy":
        run_legacy(a.tier, a.root, a.out)
    else:
        report(a.out)


if __name__ == "__main__":
    main()
