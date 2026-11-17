# Item spec 974: the spaCR store (Parquet tables, Zarr crops, one DuckDB catalog)

- **Number:** 974. Follow-ons: 975 (Mask Generation and Measure write to the store) and 976 (converter and readers for existing projects).
- **Lifecycle:** alpha, behind "Show alpha features". The new setting is `ops_storage` (`"current"` default, `"spacr_store"` opt-in), registered as `ALPHA_FEATURES[974]`.
- **Decision:** maintainer, 2026-11-17. The store becomes the default only after 975 and 976, as a separate decision.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes. The BMC OPS analysis reads 100 M+ cells.

## Why
spaCR must store and query OPS data at more than 100 M cells by more than 10,000 feature columns, plus one single-cell crop per cell. Today:

- OPS keeps `measurements.db` (SQLite) as the authority, with parquet caches of reads and barcodes (`spacr/ops_store.py`).
- The phenotype tables go into `measurements/measurements.db` (`cell`, `nucleus`, `cytoplasm`, optional `cell_wide`). Above `ops_tables_parquet_threshold` cells, the features go to `measurements/<compartment>/<plate>_<well>.parquet`.
- Crops are TIFF members of tar shards with a parquet index.

Stock SQLite holds at most 2,000 columns per table (`SQLITE_MAX_COLUMN`), so 10,000 features never fit one table. It also scans rows rather than columns, so every tall read decodes every column. PNG or TIFF crops need one file or one tar member per cell.

Every query shape must be fast:

- tall: all rows, a few columns;
- wide: a filtered subset of rows, all columns;
- point lookups;
- crops for any selection;
- ML batches.

## Layout (one project = one run folder, `dst_root`)

```
<project>/
  spacr.duckdb                                    catalog (views + registry)
  tables/<class>/plate=<P>/well=<W>/part-<N>.parquet
  crops/plate=<P>/well=<W>.zarr                   (n_crops, C, H, W), source dtype
  _pending/<uuid>.json                            write manifests not yet registered
```

**Tables**
- One table per object class. 974 writes `cells`, `reads` and `barcodes`; 975 adds `nuclei`, `pathogens`, `cytoplasm` and the other classes.
- Hive-partitioned. Plate and well values are percent-encoded in the path (`urllib.parse.quote(value, safe="")`); DuckDB decodes them. Partition columns are read as VARCHAR (`hive_types`), so a plate named `20200202` stays a string.
- Within each file, rows are sorted by the table's main filter column (`gene` for OPS cells, else the condition column), then by `cell_id`. The sort is stable, and nulls go last.
- Compression is zstd. Row groups hold about 128k rows, capped at about 256 MiB of uncompressed values per row group, so a 10,000-column table gets smaller groups (about 6.5k rows). Without the cap, a single row group of 128k × 10,000 float32 would be 5 GB in memory. Statistics are always written, so DuckDB skips row groups on min/max.
- Each row holds:
  - the metadata;
  - all measurements;
  - `cell_id` (BIGINT, `spacr.ops_run_infra.integer_cell_key` for OPS);
  - `crop_store`, the crop store's path relative to the project, null when the cell has no crop;
  - `crop_index`, the position in that store, null when the cell has no crop.

**Crops**
- One Zarr store per plate and well, holding one array `crops` of shape (n, C, H, W) in the source dtype. Chunks are (256, C, H, W), compressed with zstd level 3.
- Zarr format 2 is written through either zarr 2 or zarr 3 (`setup.py` already allows `zarr>=2.16,<4` and numcodecs on Python 3.10+). Both libraries read it.
- Crops are stored in the order of the table rows that have a crop. One gene's crops are therefore contiguous, and a crop batch for one gene reads whole chunks.
- Crops are raw, never stretched (rule "crops raw on export"). Normalisation belongs in the network transform.
- An optional second array `masks` (n, H, W) holds the masks when the source has them (`ops_crop_masks`).

**Catalog (`spacr.duckdb`)**

| table | columns |
|---|---|
| `spacr_schema` | `version` (store schema version, 1) |
| `spacr_tables` | `name`, `sort_by`, `key_columns` (JSON), `created_at` |
| `spacr_files` | `path` (relative), `table_name`, `kind` (`parquet`/`zarr`), `plate`, `well`, `rows`, `schema_version`, `sha256`, `bytes`, `mtime_ns`, `written_at`, `run_id` |
| `spacr_runs` | `run_id`, `started_at`, `spacr_version`, `settings` (JSON) |

- There is one view per table: `read_parquet([registered paths], hive_partitioning = true, hive_types = {...}, union_by_name = true)`.
- Each view lists its registered files explicitly instead of using a glob, so a reader never sees an unregistered or partial file.

## Dependencies
- duckdb ≥ 1.1, pyarrow, zarr and numcodecs are already core dependencies in `setup.py` for Python ≥ 3.10.
- No new dependency is added.
- On Python 3.9 (core install without these packages), `spacr.store` refuses with an install hint, and the default OPS path is untouched.

## Write protocol (parallel, no single-writer lock)
1. **Write each well.** `spacr.store.write_part(project, table, frame, plate=, well=, crops=None, ...)` writes one well's part. Each worker (well) calls it independently. Steps:
   1. Sort the rows.
   2. Write the Parquet to `part-<N>.parquet.tmp-<uuid>`, fsync it, then `os.replace` it to `part-<N>.parquet`.
   3. Write the Zarr store to `<W>.zarr.tmp-<uuid>` and rename it to `<W>.zarr`. An existing store is moved aside first and removed after registration.
   4. Hash every file it wrote: sha256 of the Parquet file; for the Zarr store, sha256 over each file's relative path and sha256, in sorted order.
   5. Write the manifest `_pending/<uuid>.json` the same way, last of all.
2. **Register.** `spacr.store.register(project)` registers every pending manifest in ONE DuckDB transaction:
   - inserts the registry rows;
   - marks the well's earlier files as superseded;
   - rebuilds the affected views.

   After the commit, it deletes the manifests and the superseded files.
3. **Crash safety.** A run killed at any point leaves only:
   - `*.tmp-*` files, which `register` and `cleanup` remove when they are older than the run;
   - final-named files with no manifest.

   Neither is ever registered, so neither is ever read.
4. **Readers.**
   - `open` reads the registry in read-only mode, closes the catalog, and serves queries from an in-memory DuckDB with the same views. A long-lived reader never blocks a registration.
   - `refresh()` re-reads the registry.

## Integrity (hash check, lazy)
- **On open (quick).** Each registered file's size and mtime are compared with the registry; this is one `stat` per file.
- **`verify(full=True)`.** Every hash is recomputed.
- **Mismatches.** Any mismatch raises `StoreError` naming the file, its expected and found hash, and the fix ("rewrite the well, or `register(..., accept_changes=True)` to adopt the new bytes").
- Queries do not rehash.

## Read API (`spacr.store`)
- `store = spacr.store.open(project)`.
  - A folder with `spacr.duckdb` opens the catalog.
  - A folder with only `measurements.db` (or a `.db` file) opens the old SQLite project through DuckDB's `sqlite` extension (`ATTACH ... (TYPE sqlite)`), with one view per SQLite table. If the extension cannot be loaded (offline), each table is read through `sqlite3` into Arrow when it is first used.
- `store.query(where=None, *, table="cells", features=None, crops=False, as_="pandas")`.
  - `where` is a SQL `WHERE` clause, a full `SELECT`/`WITH` statement, or a mapping `{column: value or list}`.
  - `features=None` returns every column. A list returns the key columns plus the crop pointer plus the named features.
  - `as_` is `"pandas"`, `"arrow"` or `"numpy"`. Numpy returns `(metadata DataFrame, float32 matrix of the features)`.
  - With `crops=True` it returns `(result, crops)`: crops are (n, C, H, W) in row order, and rows without a crop get an all-zero crop.
- `store.sql(statement, params=None, as_="pandas")` runs raw SQL against the views.
- `store.batches(where=None, *, table="cells", features=None, crops=False, batch_size=1024, shuffle=False, seed=0)` iterates over ML batches.
  - It streams Arrow record batches from DuckDB, so a 100 M-row table is never materialised.
  - `shuffle` shuffles the order of the files, then the rows within a buffer of batches. The order is reproducible for one seed.
- `store.verify(full=False)`, `store.refresh()`, `store.tables()`, `store.close()`.
- **Zero copy.** Arrow results are returned as DuckDB produces them. Numeric pandas columns come from Arrow without a copy where pandas allows it. A 2-D feature matrix needs one copy, because Parquet is columnar and the matrix is row-major. Crops come from Zarr straight into the output array.

## OPS write path (974)
- `ops_storage = "spacr_store"` adds a plate step after `ph_tables`/`ph_crops`, `_ops_store_write`. It does not replace them in 974; the current outputs are still written, so the two can be compared row by row. The step writes:
  - `cells`: one row per owned, whole cell, the same rows and values as `ph_tables` (keys + `cell_*`, `nucleus_*`, `cytoplasm_*` features), with `plate`, `well`, `cell_id` and the crop pointer added, sorted by `gene`, then `cell_id`;
  - crops: every crop of the `ph_crops` shards, copied pixel for pixel (and masks, if written) into `crops/plate=P/well=W.zarr`;
  - `reads` and `barcodes`: the well's rows of `ops_reads` and `ops_barcodes` from the run's `measurements.db`;
  - the catalog: one `register` per plate, plus a `spacr_runs` row with the settings.
- **Default (`"current"`).** Nothing changes, and every output is byte-identical to nightly.

## Proof (acceptance)
On the OPS test data (Funk 2022 plate A, well A1 subset, full tier, CPU):

1. **Values.** Every column and every row through `store.query` is identical to the SQLite tables (`cell`, `nucleus`, `cytoplasm` joined on `prcfo`; `ops_reads`; `ops_barcodes`). Floats are compared bit for bit after the SQLite REAL cast; a NULL in SQLite equals a NaN.
2. **Crops.** Every crop is pixel-identical to its tar-shard TIFF (`_ph_read_crop`).
3. **Off-switch.** Default OPS outputs are byte-identical to nightly 51eb488a60 on the same inputs. Run reports are compared after their timing keys are dropped.
4. **Atomic writes.** A writer killed mid-write (SIGKILL inside the Parquet write and inside the Zarr write) leaves no registered file, and the store still opens and returns the earlier data.
5. **Hash check.** A file changed with its size and mtime kept is caught by `verify(full=True)`. A changed size is caught on open.
6. **Synthetic scale.** 2 M cells × 10,000 float32 features + 2 M crops (4 × 64 × 64 uint16) on /mnt/wd4tb (under 150 GB). Timings:
   - tall: all rows × 5 columns;
   - wide: 5 % of rows × all columns;
   - point lookups: 1,000 random cells × all columns;
   - a crop batch: 10,000 cells of one gene;
   - the ML batch iterator.

   The same reads are timed against the current layout on a 200k-cell subset: SQLite rows split over tables of at most 2,000 columns, and one PNG per crop. The thread count is stated (DuckDB `threads=4`).

Report: `BMCspacr/reanalysis/item974_proof/REPORT.md`.

## Migration plan
- **974 (this item).** The store module, the catalog, the opt-in OPS output and the proof. The default stays the current layout.
- **975.** Mask Generation and Measure write their per-object tables (`cells`, `nuclei`, `pathogens`, `cytoplasm`, …) and their crops (today PNG per object, plus `png_list`) to the store. The tables are sorted by condition, then `cell_id`. Settings: `measurement_backend = "spacr_store"`. 975 also covers:
  - the readers of `png_list` (classifier datasets, annotation, UMAP) reading crops through `store.query(crops=True)`;
  - the Measure write queue writing one part per well (it has a single writer today).
- **976.**
  - **Converter.** `spacr-db-audit convert <project>` converts `measurements.db`, the parquet sidecars, the `measurements/<compartment>/*.parquet` features, the tar-shard crops and PNG crop folders into the store. It is verified row by row and pixel by pixel, and the originals are kept until the maintainer deletes them.
  - **Readers.** Every spaCR reader of `measurements.db` (`spacr.tabular.read_table` / `_read_query` callers: plots, the classifier, the ML, the OPS profile, TVN and aggregate) gains a store path.
  - **Old projects.** They stay readable through the SQLite attachment.
- **After 975 and 976.** The maintainer decides whether `spacr_store` becomes the default for OPS and Measure. Only then can the OPS path stop writing the SQLite copies of the phenotype tables.

## Effort
- 974: M.
- 975: L.
- 976: L.
