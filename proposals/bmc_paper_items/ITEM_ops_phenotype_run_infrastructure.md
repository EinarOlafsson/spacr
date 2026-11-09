# Item spec: OPS phenotype run infrastructure: two unmerged measurement/SQLite commits and the engineering items found by the large phenotype run

- **Lifecycle:** future (alpha) for every new setting, behind "Show alpha features" under `ALPHA_FEATURES[752]`. The two commits in Part A are speed fixes to existing code paths and need no alpha gate, provided their outputs stay bit-identical.
- **Number:** take the next free item numbers; check `features/feature_list/` and the claims files first. Part A can be one small item; each subsection of Part B can be its own item, or be folded into the item named under "see".
- **Rule:** a spaCR settings-file run must give output identical to the reference on the validation data, or within the tolerance stated in the subsection. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes for Part A and for B1–B9 (the phenotype run depends on them); no for the rest.
- **Companion spec:** `ITEM_ops_phenotype_run_v2.md` (merged/, standard tables, owned cells, crops, second-acquisition registration). Subsections marked **see v2** are already specified there; only what is new is kept here.
- **Paths:** P = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/funk_phenotype` (README.md, PROPOSED_FEATURE_ITEMS.md, `stress_test.csv`, `stress_rows.jsonl`, `test_work/`); S = `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/scripts` (driver scripts); L = `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/logs`.
- **Data:** the public Funk et al. 2022 OPS phenotype images (plates LaC024A–D), plate A pilot field A1/514.

## Part A: two unmerged commits to bring onto nightly

Both commits are on the local branch `ops-funk-phenotype-run` in `/mnt/firecuda2/Claude/repo/spacr` only. They are not on origin, and neither is an ancestor of origin/nightly. A session on this machine must cherry-pick them onto nightly in this order (8147e421e5 has 8474a5fde8 as its parent):

```
git cherry-pick 8474a5fde8f2f84b44c2fcf8214fde200ed475de 8147e421e5c12aca09f6adc41fd2b3401bb72c8d
```

Do this in its own worktree on wd4tb, not in the shared checkout. The branch was forked from `ops-phenotype-path` @ df9d9d9d8, so expect conflicts in `spacr/ops_engine.py` and `spacr/ops_sample.py` against current nightly.

### A1. Commit 8474a5fde8: per-object bounding-box loops, plus optional `measure_gpu` in the OPS phenotype path
- **What it does.** Three per-object loops in `spacr/measure.py` (`_measure_intensity_distance`, `get_components`, `_map_child_to_parent`) used to compare the whole field against one label per object. They now look only inside each object's `find_objects` bounding box (new helper `_label_bounding_boxes`). The pixels selected, and their order, are the same. On one core of a loaded machine, a 2,960 px OPS phenotype field with about 950 cells went from 511 s to 203 s (about 2.5×).
- **Also in the commit.** `measure_phenotype_field` (`spacr/ops_sample.py`) and `_ph_measure_field` (`spacr/ops_engine.py`) take an optional `measure_gpu` for the intensity and texture tables. The setting that exposes it, `ops_ph_measure_gpu`, is specified in v2.
- **Files:** `spacr/measure.py`, `spacr/ops_engine.py`, `spacr/ops_sample.py`, `tests/test_item60_measure_coverage.py` (the `coords` branch of `test_measure_intensity_distance_defensive_geometry` now monkeypatches `_label_bounding_boxes` to `{}` to reach the per-label `argwhere` fallback).
- **Review:**
  - the bounding-box slice must include the box edges and use the same label comparison as before (no off-by-one at box borders);
  - objects that are missing from `find_objects` (gaps in the label numbering) must take the fallback path, not be skipped;
  - `measure_gpu=False` must be the default everywhere, so existing callers are unchanged.
- **Tests to add (CPU, `CUDA_VISIBLE_DEVICES=''`, at most 4 test workers):**
  1. **Bit-identity on synthetic masks.** For random label images with touching objects, objects on the field edge, non-contiguous labels, one-pixel objects and an empty mask: the three functions return arrays equal byte for byte (`np.array_equal`, same dtype) before and after the change. Keep a copy of the old implementation in the test as the oracle.
  2. **Bit-identity on a real field.** `measure_phenotype_field` on pilot field A1/514 gives the reference table, all 406 of 406 columns equal (exact, CPU). Reference: `P/work/plateA/A1/phenotype_cells_A1.parquet` and `phenotype_masks/site-514.npz`, or the frozen-source run in `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/src_frozen` (commit in `src_frozen/COMMIT`).
  3. **Fallback reached.** The existing `coords` / `submask` branches in `tests/test_item60_measure_coverage.py` pass.
  4. **Speed (marked slow, not in the default run).** The time per field on A1/514 is at most half of the parent commit's time on the same machine. Record both times. Do not assert absolute seconds.

### A2. Commit 8147e421e5: one SQLite transaction per measurement frame
- **What it does.** spaCR opens its SQLite connections in autocommit mode, so the `executemany` in `_insert_frame` (`spacr/utils.py`) committed and synced every row separately. An explicit `BEGIN` around the frame makes it one commit and keeps the all-or-error contract. A connection that is already inside a queued write packet's transaction is left alone.
- **Measured effect (engineering).** One 900-cell frame of one table went from 18 s to 0.13 s. `_append_to_measurements_db` went from 47 rows/s to about 1,700–6,300 rows/s over the test runs: about 2,000–2,700 rows/s on the 2-field test, and 2,806–4,251 rows/s on full wells. The control row from before the fix is `P/test_work/stress_test_before_insert_fix.csv`.
- **Files:** `spacr/utils.py` (9 lines).
- **Review:**
  - `conn.in_transaction` is checked before `BEGIN`, so a nested call does not raise "cannot start a transaction within a transaction";
  - on an exception the frame is rolled back and the exception re-raised (nothing is half-written);
  - the write-queue path (`tests/test_database_write_queue.py`) still owns its own transaction.
- **Tests to add (CPU):**
  1. **Bit-identity of the stored data.** Writing the same frames with and without the change gives tables equal row for row and value for value (`SELECT * ORDER BY rowid`, compared exactly), with the same schema.
  2. **Atomicity.** A frame whose insert fails halfway (inject a bad row) leaves the table unchanged, and the error propagates.
  3. **Nested transaction.** `_insert_frame` called inside an open transaction neither commits it nor raises.
  4. **Throughput (marked slow).** On a temporary file DB, 10,000 rows insert at least 20× faster than the parent commit. Assert the ratio, not absolute rows/s.
- **Known failing test, not caused by this commit.** `tests/test_database_write_queue.py::test_simulation_shutdown_flushes_real_ram_packets_before_writer_sentinel` has a 25 s shutdown deadline. It fails identically with and without the commit on a loaded machine (6,974 database/measure tests pass; this is the one failure). Before merging:
  - run it on an idle machine, or under a lower load, with and without the commit, and record both results;
  - if it still fails without the commit, file it separately: make the deadline scale with measured load, or mark it `slow` with a generous deadline. Do not block A2 on it;
  - if it passes without the commit and fails with it, A2 has a shutdown regression and must not merge.

## Part B: proposed items from the large phenotype run (PROPOSED_FEATURE_ITEMS.md 1–19)

Reference implementation for all of Part B: the driver `S/funk_phenotype.py` and the helpers in `S/` named per subsection. Design notes: `P/README.md`, `P/PROPOSED_FEATURE_ITEMS.md`.

### B1. Pipelined phenotype phases (proposed item 1)
- **Now:** `run_ops` runs `ph_segment` and then `ph_measure`, so the GPU sits idle during measurement and the CPU sits idle during segmentation.
- **Reference:** `S/funk_phenotype.py` runs GPU segmentation and CPU measurement at the same time, with measurement in a process pool fed field by field (`stage_task`, `measure_task`).
- **Build:** setting `ops_ph_pipeline` (bool, default False) plus `ops_ph_measure_workers` (int). When on, segmentation of field n+1 overlaps measurement of field n.
- **Acceptance:** masks and measurement tables identical to the sequential run on a 4-field synthetic run folder (CPU segmentation stub) and on A1 fields 514–517; wall time reported in the run report.

### B2. Streaming raw fields from network storage (proposed item 2)
- **Now:** `_ph_fields` reads every field twice from the source (once to segment, once to measure).
- **Reference:** `stage_task`, `staged_files`: a bounded look-ahead stager (4 threads, 48 fields ahead) reads the raw TIFFs from the NAS and keeps only the max projection on local disk (about 70 MB per field instead of 280 MB). The local copy is deleted once the field is measured.
- **Build:** settings `ops_ph_stage_dir` (str, None = off), `ops_ph_stage_ahead` (int, 48), `ops_ph_stage_threads` (int, 4).
- **Acceptance:** outputs identical with staging on and off. Each raw file is read exactly once (count the opens in a test). Local staging space never exceeds `stage_ahead` × field size.

### B3. Phenotype acquisitions split across folders (proposed item 3); registration **see v2**
- **See v2:** phase-correlation registration of a second acquisition (`ops_ph_secondary_*`).
- **New (a bug, fix first):**
  - `_index_tiles` on the parent folder merges both acquisitions, and a repeated channel token silently overwrites another (A594 on plate B);
  - on plate D, `_plane_sources` prefers a single-file GFP re-image over the in-stack GFP.
- **New (setting):** `ops_ph_acquisitions` (dict: acquisition folder → channels, plus `register_on` channel), which makes the acquisition of each channel explicit. With it set, `_index_tiles` refuses duplicate channel tokens instead of overwriting, and ignores acquisitions not listed (the plate D re-image).
- **Reference:** `S/check_acq.py` (offset check), `S/check_registration.py`, `P/README.md` § "Acquisitions per plate".
- **Tests:** a synthetic two-folder plate with a duplicated channel token raises a clear error without the setting and indexes correctly with it. The plate B index lists A594 from the intended acquisition.

### B4. Sampled crop export (proposed item 4); store and per-guide sampling **see v2**
- **See v2:** `ph_crops` with `ops_crop_per_guide`, `ops_crop_size` (96), raw crops, sharded store.
- **New:** two extra sampling arms used by the reference:
  - `ops_crop_all_nt` (bool, default True): keep every non-targeting cell;
  - `ops_crop_control_fraction` (float, default 0.01): a hash-random control set, chosen by a stable hash of (plate, well, site, label) so it does not depend on run order.
  - Optional cell and nucleus mask channels in the crop (`ops_crop_masks`, bool).
- **Acceptance:** the sampled cell set equals the reference sampled shards (npz shards with a parquet index under `P/work/plateA/`), identical for the same plate and well.

### B5. Swappable genotype in aggregation (proposed item 5): **see v2**
- v2 sets `ops_genotype_table`. Keep only this check: `_ph_aggregate` must read the guide from any table keyed by (plate, well, object_id), not only `ops_barcodes`, so that a new decoding rule re-runs only the aggregation. Test: two genotype tables give two aggregates without re-running any earlier phase.

### B6. Interphase-only profiles, with non-targeting guides as separate rows (proposed item 6)
- **Now:** `_ph_aggregate` mixes interphase and mitotic cells.
- **Build:** `ops_profile_cells` ('all' | 'interphase', default 'all'), using the mitotic call (item 856), and `ops_nt_rows` (shared with `ITEM_ops_gene_level_aggregation.md`). This follows the convention of the source study (Funk et al. 2022).
- **Acceptance:** with 'interphase', the guide and gene tables equal the reference interphase tables from `S/funk_aggregate.py` (`genes_interphase`, `guides_*` under `/mnt/wd4tb/bmc_reanalysis/funk_phenotype/tables/`).

### B7. GPU measurement setting (proposed item 7): **see v2** (`ops_ph_measure_gpu`)
- New only: when the setting is on, a one-field self-check compares the GPU table with the CPU table before the run proceeds, as the reference launcher did. Report the columns outside rtol 1e-4. Setting `ops_ph_measure_gpu_check` (bool, default True).

### B8. Per-well reads of `ops_objects` (proposed item 8)
- **Now:** `_well_rows` reads the whole plate table (about 8 M rows) for each call: about 5 min and 6.6 GB each time.
- **Build:** a query filtered by well on the existing (plate, well, object_id) index takes about 13 s. Use it in `_well_rows`, and in every other per-well reader.
- **Acceptance:** identical rows (same order after sorting by object_id) to the full-table read; the peak RSS of a per-well call is bounded by the well's size. Test on a synthetic multi-well DB, with an assertion that the SQL contains the well filter.

### B9. Per-well export to archive storage with done markers (proposed item 9); NAS destination **see v2**
- **See v2:** `ops_merged_root`, atomic writes, resume by options marker.
- **New:** after each well, copy the outputs to the archive root, verify the copy (size + md5), then optionally delete the local files. A `WELL_DONE` / `PLATE_DONE` marker lets a restart resume at the next well. Settings: `ops_archive_root` (str, None = off), `ops_archive_verify` ('md5' | 'size'), `ops_archive_delete_local` (bool, False).
- **Acceptance:** a run interrupted after a well's copy resumes without re-measuring it. A corrupted copy (test: truncate one file) is detected and re-copied, and nothing local is deleted before verification.

### B10. Gene-cluster benchmark (proposed item 10)
- **Reference:** `S/funk_benchmark.py`, validated by `S/check_funk_benchmark.py`.
- **Algorithm:** PCA to 99 % variance → PHATE diffusion graph (Moon et al. 2019) → Leiden with RBConfiguration (Traag et al. 2019) → per-cluster CORUM / KEGG Fisher tests with BH correction within each cluster → STRING pair precision and recall → a shuffled-PC control.
- **spaCR home:** `spacr.ops_benchmarks.cluster_benchmark` (next to `mitocarta_benchmark`, `ITEM_ops_mito_benchmark.md`). Settings: `ops_bench_pca_variance` (0.99), `ops_bench_leiden_resolution`, `ops_bench_seed`, `ops_bench_references` (reference-set names from `ITEM_ops_reference_sets.md`), `ops_bench_shuffle_control` (bool).
- **Acceptance:** given a cluster table, the enrichment fractions are identical to the output of `S/check_funk_benchmark.py` on the same input (the published Funk et al. 2022 clusters), and the full pipeline reproduces `funk_benchmark.py` on the same seed.

### B11. GPU hand-over launcher (proposed item 11)
- **Reference:** `S/launch_gpu_v6.sh` (earlier versions `launch_gpu*.sh`), `S/vram_guard.sh`, `S/vram_watchdog*.sh`.
- **Behaviour:** wait for another job's release line, then for 0 % GPU utilisation 6 times in a row with no running compute process above 500 MiB. Write START and FINISH to `~/.spacr/gpu/log`. Honour a STOP file for a clean pause. A VRAM watchdog restarts with fewer measurement workers above a cap (22 GB in the reference).
- **Build:** a `spacr.gpu_admission` helper used by `run_ops` when `ops_gpu_admission` (bool, default False) is set, with `ops_gpu_idle_checks` (6), `ops_gpu_min_process_mib` (500) and `ops_gpu_vram_cap_gb` (None).
- **Acceptance (CPU, mocked `nvidia-smi`):** waits while a compute process above the threshold exists, starts after 6 idle samples, logs START/FINISH, and stops cleanly on the STOP file.

### B12. Remaining CPU measurement hot spots (proposed item 12)
About 200 s per 2,960 px field on the CPU path:
- `_extended_regionprops_table` calls scipy `skew`, `kurtosis` and `mode` once per object (about 50 s in the decorators alone);
- GLCM homogeneity (about 40 s);
- `_percentiles_of` (about 18 s).
- **Build:** vectorised exact versions (grouped two-pass moments by label; batched GLCM; grouped percentiles).
- **Reference profile:** `S/profile_measure.py`.
- **Acceptance:** the A1/514 table is bit-identical to the current CPU path, or within rtol 1e-12 for the moments, documented per column. Report the time per field before and after.

### B13. GPU vs CPU skew/kurtosis (proposed item 13): **see item 788** (fix) and v2
- **Engineering record:** on A1/514, 388 of 406 columns agree within rtol 1e-4. The 18 `*_skew_intensity` / `*_kurtosis_intensity` columns differ, by up to 1.1 % (`cytoplasm_af750_skew_intensity`). Logs: `L/gpu_measure_compare.log`, `L/gpu_measure_check.log`.
- **Cause and fix:** the torch path forms central moments in float32 or from raw power sums, which cancel badly. Compute them two-pass (subtract the per-object mean first) in float64.
- **Acceptance:** a test that the GPU and CPU skew/kurtosis agree within 1e-6 (relative) on A1/514 and on synthetic objects with a large mean and a small variance. If item 788 already landed, only add this test.

### B14. SQLite autocommit insert (proposed item 14): see Part A2.

### B15. cuCIM morphology kernel compile and touching-region fallback (proposed item 15)
- **Now:** the GPU morphology path (cuCIM) compiles its kernels on first use, about 25 s of NVRTC per new worker process. The same path falls back to the slow robust perimeter/euler for regions that touch, which is almost every cell in a confluent OPS field (944 of 954 on A1/514).
- **Build:** long-lived measurement workers (B1), or a warm kernel cache (`CUPY_CACHE_DIR` set under `~/.spacr/`), plus a vectorised perimeter/euler for touching regions.
- **Acceptance:** morphology columns identical to the CPU path on A1/514. Only the first field of a worker pays the compile cost (log it).

### B16. Crop store at OPS scale (proposed item 16); sharded store **see v2 / item 809**
- **Engineering record** (plate A, one PNG per cell in one directory, NFS):
  - about 1.6–1.8 M PNGs per well (≈ 29–32 GB);
  - listing the directory takes 151–604 s;
  - a random read of 1,000 crops (prcfo → png_path → cv2.imread) runs at 2.4–15 crops/s (0.04–0.27 MB/s);
  - rewriting names in a 650 k-entry directory costs about 270 ms per file;
  - copying to the NAS one file per crop runs at 49–88 files/s.
- **Engineering record** (tar shards of about 100 k crops):
  - NAS transfer at 88 MB/s (≈ 4,700 PNG/s);
  - random seek + imdecode at 17.2 crops/s;
  - a sequential stream of one 2 GB shard at 842 crops/s (14.7 MB/s, single thread).
- **New beyond 809:**
  - a streaming crop loader that reads whole shards and shuffles within and between shards (WebDataset-style; Aizman et al. 2019, arXiv:2001.01858). It should be about 60–350× faster than one file per cell;
  - hash-bucketed directories (e.g. `cell_png/<plate>/<well>/<site // 100>/`) when PNGs are kept;
  - a `png_shard_index` table in each well's DB (member, `offset_data`, size, prcfo, site, label);
  - shard offsets read back from the closed tar. `TarFile.addfile` works on a copy of the TarInfo, so offsets recorded at add time are wrong (the bug fixed by `S/fix_shard_index.py`).
- **Reference:** `S/png_packer.py`, `S/png_mover.py`, `S/png_stress.py`, `S/tar_stress.py`, `S/fix_shard_index.py`.
- **Acceptance:** crops read through the loader are pixel-identical to the per-file PNGs for a sample of 200 cells. Every index offset points at a PNG signature (test). The streaming throughput is reported by the stress test (B20).

### B17. SQLite measurement store at OPS scale (proposed item 17); parquet threshold, bulk load and indexes **see v2 / items 810, 811**
- **Engineering record** (one well, 4.76–5.50 M rows over cell, nucleus and cytoplasm; 5.6–6.5 GB DB):
  - write 2,806–4,251 rows/s with the A2 fix (19–28 min per well);
  - index build 79–164 s;
  - query one field 0.05–0.40 s;
  - one gene's cells 0.08–0.27 s;
  - the cell ⋈ nucleus ⋈ cytoplasm join 68–131 s even with prcfo indexes;
  - whole-well read 113–163 s;
  - driver peak RSS 26–32 GB during the store step.
- **New beyond v2:**
  - integer join keys in place of the text `prcfo`;
  - an optional single wide per-cell table (`measurement_layout` 'wide');
  - bulk `executemany` per table per well instead of per field.
- **Acceptance:** joins on integer keys return the same rows as joins on `prcfo`. The wide table equals the three-table join. The stress test reports the join time.

### B18. Raw phenotype files missing from the public download (proposed item 18)
- **Now:** on plate A, about 10 fields per well are absent from the public download (for example A1 sites 598, 757, 860, 864, 867, 871, 909, 988, 1133 and 1136); some are folders, and some lack a channel. `_ph_fields` retries them on every resume, so the well never completes.
- **Build:** record such fields once in `ops_missing_fields` (plate, well, site, reason) and carry on. A resume skips recorded fields unless `ops_retry_missing` (bool, False) is set.
- **Acceptance:** on a synthetic plate with a missing field, a missing channel and a field that is a folder, the run completes, the three are recorded once, and a second resume does not retry them. For A1, the recorded sites equal the reference list.

### B19. Launcher idle check ignores desktop GPU processes (proposed item 19)
- **Now:** a desktop browser's GPU process (`--type=gpu-process`, about 600 MiB) blocked the "no compute process > 500 MiB" check.
- **Build:** in B11, ignore graphics-only processes and display-server processes (an allowlist of process types, `ops_gpu_ignore_process_patterns`).
- **Acceptance:** with a mocked process list containing a browser GPU process at 600 MiB, admission proceeds. With a compute process at 600 MiB, it waits.

### B20. Stress-test report (README § "Stress test")
- **Reference:** `stress_test.csv` and `stress_rows.jsonl` in P; scripts `S/png_stress.py`, `S/tar_stress.py`, `S/io_test.sh`, `S/plate_size.sh`.
- **Build:** an optional `stress` phase in `run_ops` (`ops_stress_report`, bool, default False). It writes `results/ops_stress/stress_test.csv`, one row per well and one per plate:
  - **per well:** SQLite measurement rows/s; `png_list` rows/s; crops per field and seconds spent; DB size; index build time; query latency for one field, one gene's cells (genotype ⋈ cell), the cell ⋈ nucleus ⋈ cytoplasm join and the whole well; a training-style random read of 1,000 crops (prcfo → crop → decode) in images/s and MB/s, per crop layout; directory listing time and file count;
  - **per plate:** archive bytes and file count, time to walk the plate's files, aggregation time, and aggregation peak RSS.
- **Acceptance:** columns as in the reference `stress_test.csv`. The timing values are not compared (they are machine-dependent); the row, byte and file counts are identical on the same run folder.

## Effort
- Part A: S (cherry-pick, conflict resolution, the tests above): about 1 day.
- Part B: B1, B2, B9, B11, B12, B16, B17 are M each; the others are S. Several are folded into v2's L estimate (marked "see v2").
