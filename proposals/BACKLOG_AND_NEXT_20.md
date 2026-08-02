# spaCR backlog status & next 20
`nightly` @ `d882c0d7`, v1.4.9.9 — every verdict below re-checked against the code today. "PROVED" = I ran it.

---

## Part 1 — what is still open

**Headline: 76 backlog IDs. 60 are DONE, 5 are OBSOLETE, 11 are genuinely still open or partial.** The list has been pruned honestly — the 850 commits since `de15cac` closed almost everything. What's left is concentrated in three places: the settings UI, the demo/tutorial path, and the sim rewrite you never started.

### Still open, ordered by what would hurt most to leave

| ID | One line | Verdict | Evidence |
|---|---|---|---|
| **30c** | Measure CP toggle + live crop panel + the unnormalized-saturation bug | **PARTIAL — and worse than recorded** | UI shipped (`qt/widgets/measure_preview.py`, 855 lines; toggle at `qt/screens/app_screen.py:1268`, labelled "Live" not "CP"). **The bug is live and I PROVED it**: `measure.py:2958` runs `np.clip(crop,0,255).astype(np.uint8)` on raw uint16 when `normalize=False`. Ran it — `normalize=False` → **29.75% of pixels exactly 255**, which is *every* in-object pixel (576 of 1936); `normalize=True` → 0.31%. **Severity correction:** the same code exists a second time at `measure.py:2833` inside `generate_object_dataset` (`measure.py:2615`), which **writes PNGs to disk** at `measure.py:2845`. So this is not preview-cosmetic — it silently bakes saturated crops into training datasets. Missing too: hover white rim (`_CropThumb` at `measure_preview.py:84` has no `enterEvent`/`leaveEvent`; Annotate's is at `annotate.py:446`). |
| **34c** | Demos for each core module | **PARTIAL — 6 declared, 1 works** | `DEMO_TARGETS` (`qt/app.py:970`) has exactly 6: mask, measure, crop, classify, timelapse, map_barcodes. **Missing for Core:** motility, ml_analyze, regression, umap. Worse, of the 6 only `mask` runs — and it ends on a red `Segmentation QC (organelle): FAIL`. `generate_measure_demo` writes no `merged/*.npy` (PROVED: `grep -n merged spacr/qt/synthetic.py` → **zero hits**), the timelapse demo emits `channels=(0,1)` (`synthetic.py:300`) against a 4-channel `demo_settings` base (`synthetic.py:339`), the mask demo ships the misspelled `cell_signal_to_noise` (`synthetic.py:352`; real key is `cell_Signal_to_noise`), and the sequencing demo exits 0 having written nothing. |
| **3** | Diameter auto-estimator | **PARTIAL — built, unreachable** | `spacr/diameter.py` is 1065 lines with a full test file, and has **zero production importers**. PROVED: `grep -rn "from .diameter\\|import diameter" spacr/ tests/` returns only `tests/test_diameter_estimator.py`. `diameter_estimate_n_fields` is declared, typed, categorised and **renders as a spinbox** (`qt/screens/settings_model.py:301`) — read by nothing. Finished code, no caller. |
| **21c** | Read the Cellpose model list from the API | **OPEN — untouched, now locked by a test** | `settings.py:479` is still `CELLPOSE_MODEL_CHOICES = ('cpsam',)`. PROVED: whole-tree grep for a `models.MODEL_NAMES` / `get_user_models` **read** returns only comments (`settings.py:486`, `utils.py:5829`) — no live call. New blocker: `tests/test_cellpose4_model_story.py:453` now *asserts* `S.CELLPOSE_MODEL_CHOICES == ("cpsam",)`, so the fix must delete a test. |
| **12f** | Rethink settings *presentation*, not grouping | **OPEN** | None of the three named ideas exist. PROVED: `grep -n "search\\|advanced\\|differs"` across `settings_model.py` returns 7 hits, **all** of them hyperparameter-search keys (`umap_search_keys` at :762, `infection_pca_umap_search` at :421) — no settings search box, no advanced fold, no differs-from-default marker. Row-level polish did land (animation dots, chips, toggles); the heading-plus-rows *form* is untouched. |
| **11d** | Picker for fixed-alphabet combination settings | **PARTIAL — wrong shape** | `train_channels`/`channels`/`png_dims` route to the **free** add/remove chip editor via `CHANNEL_LIST_KEYS` (`settings_model.py:1410`, used at :2117) — not a closed-set multi-select. Nothing stops typing `x`. The in-file comment at :1405 admits it: *"use the same add/remove-chip editor as `manders_thresholds`"*. That is #9d's shape delivered for #11d's problem. |
| **27b** | Live view of CV training | **PARTIAL** | Aggregate loss+accuracy curves render live (`deep_spacr._plot_training_curves:1426`) and TensorBoard events are written. **The explicit ask is unmet:** per-class accuracy by epoch. `per_class_accuracy` *is* computed (`deep_spacr.py:370,416`) and then never plotted — the curve function reads only `d['loss']` and `d['accuracy']`. |
| **34b** | Narrated Mask tutorial in your own voice | **PARTIAL — far bigger, but not the ask** | 40 rendered lessons shipped (`docs/source/_extra/tutorials`, **723 MB**), 8 narration languages. But narration is **Kokoro stock voices** — PROVED: zero hits for `me_voice`/`OpenVoice`/`XTTS`/voice-cloning anywhere in the tree. Two things need your call: `docs/source/_extra/tutorials/TUTORIAL_MEDIA_NOTICE.md:4` says the media is **"for non-commercial use"** — exactly the licence trap the item said to settle first; and the in-app script says *"cyto for cells"* (`qt/tutorial/scripts.py:260`), which is factually wrong under Cellpose 4. |
| **20** | Fail-loud error policy (`on_error: stop/skip/retry`) | **PARTIAL — different spelling** | `errors.strict_errors` + `RunLedger`/`PartialRunError`/`run_status` all landed. But there is **no** pipeline-level `on_error` tri-state — `on_error` exists only job-level in `batch.py:1541` (`continue`/`stop`). Default is still non-strict, i.e. opt-in. This was your rank-1 item. |
| **30b** | Live Preview for every module that benefits | **PARTIAL** | Four preview cards exist (mask, timelapse, motility, measure) plus a hyperparam card in the same slot. The "audit every remaining module" sweep is not evidenced anywhere. |
| **29b** | Rewrite `sim.py` from spaCRPower + give it a GUI | **OPEN — 0%** | PROVED: `spacr/sim.py` is still 1559 lines, and greps for `spaCRPower`/`simulate_library`/`simulate_screen`/`brms`/`pymc`/`numpyro`/`betabinom` return **nothing**. No GUI: no `sim` key in `APPS`; the Tk entry was deleted in favour of `spacr-run simulation`. |

### Closed — the pruned list

**DONE (60):** 2 · 2b · 4 · 5 · 5b · 6 · 7 · 8 · 9 · 9c · 9d · 9e · 10 · 11 · 11c · 12 · 12b · 12c · 12d · 12e · 13 · 14 · 15 · 16 · 16b · 16d · 16g · 16h · 16j · 17 · 18 · 19 · 19b · 19c · 20b · 21 · 22 · 23 · 23b · 23c · 23d · 23e · 23f · 24 · 25 · 26 · 27 · 28 · 28b · 29 · 30 · 31 · 32 · 32b · 32c · 32d · 32e · 32f · 33 · 34

Spot-checked at HEAD rather than taken on trust: **#15** seg_qc is genuinely wired (`object.py:186 _run_seg_qc`, called at `:1072`, `:1412`, `:1667`); **#26** crop_source reaches all three consumers (`core.py:658`, `io.py:3979`, `io.py:4358`); **#23e** `predictions.py` is real (38 KB) and ML now merges (`ml.py:2223 merge_ml_predictions`); **#2b** `t_stack` has a consumer (`object.py:438 _t_stack_plan` → `zstack.plan_4d_from_settings`); **#16j** the three stage colours are exactly `#3B82F6`/`#FF00FF`/`#00CEC8` (`qt/theme.py:310-312`); **#9e** the stamp-column merge fix is at `io.py:3494` with `AcquisitionMetadataConflictError` at `io.py:2408`.

**OBSOLETE (5):** #1 and #1b (you dropped the coverage goal 2026-07-27; repo now measures 92.6% anyway, so it closed itself); #16e, #16f, #16i (all superseded by #16j, which shipped).

---

## Part 2 — the 20 next items

Ordered by value-to-effort. **★ = the three I would do first.**

---

### ★ 1. Fix the Tutorial button
Help → *Tutorial (web)* in the Qt GUI, and the Tk logo button, both open a URL that does not exist. The 723 MB tutorial library you just built is unreachable from inside the application that ships it.
**Evidence:** `spacr/qt/app.py:931` and `spacr/gui.py:195` both open `.../spacr/tutorial/` (singular). The library publishes at `/tutorials/` — `docs/source/conf.py:96` sets `html_extra_path = ['_extra']` and the content is at `docs/source/_extra/tutorials/` (PROVED: that is the only entry under `_extra/`). `grep -rn "tutorials/" --include=*.py spacr/` → **zero hits**; nothing in the Python package links to the library at all.
**Effort:** S (two string edits).
**Why now:** It is the cheapest possible unlock of the single largest thing that landed this cycle. Every hour of tutorial work is currently invisible to users.

---

### ★ 2. Stop saturating unnormalized crops
With `normalize` unticked, every crop comes out blown out instead of dark — the bug you flagged in #30c. It is not preview-only: the same code path writes PNGs to disk, so saturated crops are silently entering training datasets.
**Evidence:** PROVED by running `crop_objects_from_array`. `normalize=False` → **29.75% of pixels exactly 255** (576/1936 = every in-object pixel); `normalize=True` → 0.31%. Cause is `measure.py:2958`, `np.clip(crop, 0, 255).astype(np.uint8)` applied to raw uint16. Identical code at `measure.py:2833` inside `generate_object_dataset` (`measure.py:2615`), whose `save_png=True` path writes to disk at `measure.py:2845`. Not `crops.narrow_to_uint8` as originally suspected — that one is correct.
**Effort:** S.
**Why now:** It corrupts saved data, not just pixels on screen. A classifier trained from a `normalize=False` dataset has learned from clipped images with no error anywhere.

---

### ★ 3. `channel_axis=3` breaks three GUI buttons and the plaque assay
Every Cellpose call in `spacr_cellpose.py` passes `channel_axis=3`, which Cellpose 4 rejects for both array shapes spaCR produces. The fix landed in the Mask pipeline and never in this module.
**Evidence:** PROVED against the installed cellpose 4.0.7 — `transforms.convert_image(x, channel_axis=3)` raises `IndexError: tuple index out of range` on `(H,W,C)` and `ValueError: 2D image provided, but channel_axis is not None` on 2-D. Sites: `spacr/spacr_cellpose.py:185` and `:295`. `spacr/object.py` uses `channel_axis=-1` correctly at **7** sites (`:326,846,870,1250,1852,1917`). Blast radius: GUI **Cellpose Masks**, GUI **Cellpose All**, and `submodules.analyze_plaques`. No test caught it because the mock at `tests/test_coverage_fill_cellpose_gpu_funcs.py:27` is `def eval(self, x=None, **kwargs)` — it swallows `channel_axis` entirely, and 15 tests pass over it.
**Effort:** S (two lines, plus tightening the mock so it can't happen again).
**Why now:** Three buttons and one Toxo assay traceback on the first image. This is your own plaque assay.

---

### 4. Regression cannot be started from any entry point
`get_perform_regression_default_settings` omits six keys that `perform_regression` reads unconditionally. All three dispatch paths (Tk, Qt, CLI) feed that same builder, and the crash lands *after* both CSVs are read and `settings/regression.csv` is written — so it looks like a run that started fine.
**Evidence:** PROVED — `get_perform_regression_default_settings({})` returns 37 keys, and **all six of** `verbose`, `tolerance`, `score_column`, `invert_dependent_variable`, `control_wells`, `y_lims` are missing. They are read at `ml.py:1409`, `:1412`, `ml.py:408`, `:1424`, `sequencing.py:988`, `:1669`. No test caught it because the fixture at `tests/test_cov_ml_perform_regression.py:174` hand-supplies precisely those keys.
**Effort:** S.
**Why now:** Regression is the last step of the pooled-screen pipeline — the one that produces hits. It is unreachable from the GUI today.

---

### 5. Repair the demo set
Six demos are declared; one runs, and it ends on a red FAIL. This is the "try spaCR with no data of your own" path, and it is also what the shipped tutorials tell users to use.
**Evidence:** PROVED per demo. `generate_measure_demo` never writes `merged/*.npy` — `grep -n "merged" spacr/qt/synthetic.py` returns **zero hits**, and pre-flight says *"no merged folder for measure"*. Timelapse: `synthetic.py:300` emits `channels=(0,1)` while `demo_settings` base is `"channels": [0,1,2,3]` (`synthetic.py:339`) → two hard pre-flight errors. Mask: ships `"cell_signal_to_noise": 10` (`synthetic.py:352`) — not a spaCR setting, silently ignored; and the organelle QC verdict is FAIL on 4/4 fields. Sequencing: the file is named `synthetic_R1.fastq.gz` but `io.py:3868-3870` does `gz_file.split('_')` and takes `parts[1]`, so `parse_gz_files` yields `{'synthetic': {}}` → downstream `KeyError: 'R1'`; and `src` points at the parent of `fastq/` while `io.py:3864` is a flat `os.listdir`.
**Effort:** M.
**Why now:** You have asked for demos repeatedly (#34c), and the tutorial narration already promises *"Every pipeline in spaCR ships with a one-click synthetic demo."* Right now that sentence is false in the app it narrates.

---

### 6. Give the diameter estimator a caller
1065 lines of finished, tested code that no production module imports. The settings key even renders a spinbox.
**Evidence:** PROVED — `grep -rn "from .diameter\\|import diameter\\|spacr.diameter" spacr/ tests/` returns **only** `tests/test_diameter_estimator.py`. `spacr/diameter.py` is 1065 lines; `estimate_diameters` is at `:872`; `diameter_estimate_n_fields` renders via `qt/screens/settings_model.py:301`.
**Effort:** S (a button in the Mask live preview, or a `spacr-run` subcommand).
**Why now:** Highest value-per-hour left on the whole list — the work is already paid for. And diameter is the single setting users most often guess wrong.

---

### 7. Guard `Manager()` so the GUI doesn't show a bare `EOFError`
`measure_crop` starts a multiprocessing Manager with no error handling. When it fails the user gets a naked `EOFError` with no diagnostic — and a long-lived Qt process is exactly the thread-rich parent that triggers it.
**Evidence:** PROVED in the test suite — 33 deterministic failures across 7 files trace to `spacr/measure.py:2455`, `with ctx.Manager() as manager:` → `multiprocessing/managers.py:566 start` → `connection.py:383 EOFError`. Reproduces once ~4+ test modules are collected, from either half of the candidate set; independent of stdin and of system load. Default context is fork on Linux (`measure.py:129`). Mechanism (fork-with-threads) is INFERRED; the crash is PROVED.
**Effort:** S/M.
**Why now:** It is the only finding in the whole survey with real production reach into the GUI, and the fix (catch, name the start method, suggest `spawn`) is small even if the root cause takes longer.

---

### 8. Cellpose training discards all but `batch_size` images
`train_cellpose_model` selects at most `batch_size` base images from the whole training folder and trains on those. A user who annotates 300 fields trains on 8.
**Evidence:** PROVED by reading the whole path. `submodules.py:238` — `n_base = min(settings['batch_size'], max_base_images)`; `selected_indices = unique_base_indices[:n_base]` (`:242`); `images`/`labels` built only from those (`:245-250`); and `train_cp.train_seg(model.net, train_data=images, train_labels=labels, ...)` at `submodules.py:261` receives exactly that list. `batch_size` default is **8** (`settings.py:1195`). The test **encodes the bug**: `tests/test_cov_submodules_cellpose_train_test.py:202` asserts `len(call["train_data"]) == 2` with the comment *"batch_size caps the number of base images pulled from the dataset."* Related: `settings.py:1193` sets `learning_rate=0.2` for AdamW fine-tuning of CPSAM, where Cellpose's own default is 5e-5.
**Effort:** S (decouple the two, plus rewriting the test that locks it in).
**Why now:** Anyone who has trained a custom model in spaCR has trained a bad one and had no way to know.

---

### 9. Make the CLI reject dead settings keys
`spacr-run mask --set remove_border_pathogens=True` is accepted silently and does nothing — on a 40-plate cluster job. Its tooltip reads as working prose.
**Evidence:** PROVED — `remove_border_pathogens` has exactly three occurrences in `spacr/`: `settings.py:1914` (type), `:2547` (tooltip claiming it removes border pathogens), `:2619` (GUI category). **No fourth.** Nothing reads it. `cli.resolve_settings` accepts it and `validate_settings` raises no complaint. Its three siblings (`remove_border_cells` :1912, `_nuclei` :1913, `_organelles` :1915) are equally dead but their tooltips *say so*; this one doesn't — and `tests/test_settings_tooltip_quality.py:30` already lists it under *"verified dead."* The test file and the tooltip contradict each other. Same shape, lower severity: `pick_slice` (`settings.py:1411`, `:2538`). Working spellings are `pathogen_remove_border_objects` etc.
**Effort:** S.
**Why now:** Silent acceptance of a no-op setting is the failure mode that costs a GPU-week and produces a plausible-looking wrong answer.

---

### 10. Fix the environment and the editable install that is lying about it
27 test failures are one `pip install` away from green, and `pip check` cannot see the problem because the installed metadata is two patch versions stale.
**Evidence:** PROVED. `spacr` is installed **editable at 1.4.9.7** while the repo is 1.4.9.9; the installed `METADATA` says `umap-learn>=0.5.6` while `setup.py:312` says `>=0.5.11`. Installed is 0.5.9.post2, which satisfies the stale pin and violates the real one — so `pip check` is silent while `spacr/utils.py:301` raises `OptionalDependencyCompatibilityError` on every UMAP call. Also `tensorboard` 2.20.0 against a declared `>=2.21`. Side effect: `utils.py:301` raises *before* the TF blocker is installed at `:315`, so the umap-specific TensorFlow protection is not exercised at all right now.
**Effort:** S.
**Why now:** It is the largest single block of red in the suite, it restores TF-guard coverage as a side effect, and until it's fixed `pip check` will keep hiding the next one.

---

### 11. Retire the stale tests from the 2026-07-30 feature commits
Four commits landed that day changing a signature, a contract and two messages without updating pre-existing tests. 28 red tests, none of them product bugs — but they mask the ones that are.
**Evidence:** PROVED, four clusters. (a) `tests/qt/test_all_module_smoke.py:36` `_FactoryHost` lacks `_on_remote_submit_requested`, connected at `qt/app.py:1675` — wiring landed `1b02e8ec` 21:26, test last touched `f77c803f` 15:12, six hours earlier (17 tests). (b) Leakage-audit enforcement (`da7b5b00`) updated 4 test files and broke 4 others whose synthetic fixtures have too few wells (8 tests). (c) `tests/test_cov_object_preprocess_segment.py:86` monkeypatches `_check_masks` with a 3-arg lambda; `io.py:2175` now takes `resume=` (2 tests). (d) `tests/test_cov_object_organelle_sam.py:294` expects `"already processed"`, gets `"Segmentation QC (organelle): FAIL"` (1 test). Plus two more: `classifier_evaluation`, `distributed_jobs` and `run_history` are registered in `qt/app.py:229,245,246` but have **zero** occurrences in `spacr/resources/home/versions/_generators/common.py` — PROVED, `grep -c` returns 0 for all three — failing `test_home_variants.py`.
**Effort:** S/M.
**Why now:** A suite with 97 known-red tests is a suite nobody reads. The home-generator half is a real product gap, not just a test.

---

### 12. Collapse every positional `prc`/`prcf` split onto `schema.parse_*`
`schema.py` exists precisely to stop underscore-positional key parsing, and five live sites still do it by hand. Each one raises three frames from the cause.
**Evidence:** PROVED, all five are live code (AST-checked, not inside docstrings). `ml.py:1453` — `merged_df[['plateID','rowID','columnID']] = merged_df['prc'].str.split('_', expand=True)` → `ValueError: Columns must be same length as key` on a 4-token plate id. `timelapse.py:1530` and `:1573` — same shape on a 5-part `ID`; PROVED with `exp1_plate1_r1_c1_f1_o1`. `submodules.py:761`. And `sequencing.py:1009-1011` is the worst of them: `has_underscore = df['rowID'].str.contains('_').any()` guards, then `.apply(lambda x: x.split('_')[1])` runs on *every* row — PROVED, `['plate1_r1','r2','r3']` → `IndexError: list index out of range`. `schema.parse_prcf` is at `schema.py:978`, `parse_prcfo` at `:1010`; `timelapse.py:1657` already documents that the *caller* was fixed and these were not.
**Effort:** M.
**Why now:** A plate id containing an underscore is not exotic, and today it shifts every metadata column one place along — silently, when it doesn't crash.

---

### 13. Read the Cellpose model list from the API (#21c)
Your own ask, still fully untouched, and now the smallest genuinely-open item on the old list.
**Evidence:** `settings.py:479` `CELLPOSE_MODEL_CHOICES = ('cpsam',)`; `gui_utils.py:652,677` hard-code `['cpsam']`. PROVED: no live read of `models.MODEL_NAMES` or `get_user_models` anywhere — the only hits are prose in `settings.py:486` and `utils.py:5829`. Installed cellpose is 4.0.7 (`MODEL_NAMES == ['cpsam']`, `get_user_models` present) but PyPI is on 4.2.x, so the tuple bakes in a stale world permanently. **Blocker:** `tests/test_cellpose4_model_story.py:453` now asserts the tuple, so the fix must rewrite a test.
**Effort:** M.
**Why now:** Every month this sits, the hard-coded tuple diverges further from what Cellpose actually ships — and it silently prevents users from selecting their own trained checkpoints.

---

### 14. One `canonical_column_name`, not two that disagree
Two implementations, in scope simultaneously in the same module under different names, disagreeing on 11 aliases and on case sensitivity.
**Evidence:** PROVED at runtime. `spacr.utils.canonical_column_name is database_schema.canonical_column_name` → **True**; `... is schema.canonical_column_name` → **False** — in a file that also does `from . import schema`. Divergence: `plate_id` → `plateID` (schema) vs `plate_id` (db_schema); `time` → `timeID` vs `time`; `Row` → `rowID` vs `Row`; `col_name` → `columnID` vs `col_name`. `schema.py:245` is case-insensitive with 22 aliases; `database_schema.py:130` is case-sensitive with 11. `tests/test_schema.py:940` asserts schema ⊇ database_schema — i.e. the gap is asserted to *exist* rather than closed.
**Effort:** M.
**Why now:** A database with a `Row` column is canonicalized by one and not the other. That is exactly the class of bug `schema.py`'s own docstring was written about, and it will produce a wrong join before it produces an error.

---

### 15. i18n: kill the English string compares, translate the Demos menu, fix the home tagline
Three concrete localization defects, all on the first screen a non-English user sees.
**Evidence:** PROVED in a live offscreen `MainWindow` with `SPACR_LANGUAGE=sv`. (i) The menu title *does* translate (`&Demon` sv, `&演示` zh_CN, `&데모` ko, `&Démonstrations` fr), which breaks the two exact-English lookups: `first_run.py:138 _find_menu(w, "Demos")` returns `None`, losing the tour's coach-mark; and `app_screen.py:945 act.text().replace("&","") == "Demos"` makes the empty-state **"Open Demos menu" button silently do nothing** (swallowed by `except: pass` at :951). German is the only unaffected language. (ii) All seven demo action labels render in English under a Swedish menu title. (iii) The home headline machine-mangles: `sv → "End-to-end microscopy → single-Cell Mätningar → genotype-phenotype mapping."` — `_term_translation` (`i18n.py:1581`) word-substitutes inside an English sentence and splits the compound `single-cell`. This is what `docs/source/localization.rst:47` says is prevented and `README.rst:283` promises against.
**Effort:** M.
**Why now:** ~3,100 hand-written translations already shipped. These three bugs are what a reviewer would notice in the first ten seconds, and (i) is a dead button.

---

### 16. Settings presentation — search, progressive disclosure, differs-from-default (#12f)
Your open question: is the heading-plus-rows form right at all? Nothing has been tried yet.
**Evidence:** PROVED — no search box, no advanced fold, no differs-from-default marker anywhere in `settings_model.py` or `app_screen.py`; the 7 "search" hits are all hyperparameter-search keys. The scale that makes it necessary: `spacr-run --describe` reports **mask 205 settings, timelapse 205**, classify 78, motility 55, umap 53. And only **9 of 34 apps have a curated layout** — `_APP_CATEGORY_SPECS` (`settings_model.py:235`) covers activation, mask, measure, ml_analyze, motility, regression, replication, timelapse plus classify inline. The other 25 fall through to the generic table, so `map_barcodes` renders under **Cellpose / Cell / Nucleus / Pathogen** headings, none of which mean anything for barcode mapping.
**Effort:** M/L.
**Why now:** 205 settings in 14 sections cannot be navigated by scrolling, and #12e already did the regrouping half — this is the remaining half of the same problem.

---

### 17. Publishing hygiene: stop nightly republishing the public site, and get under the Pages ceiling
Three separate hazards in the docs pipeline, one of them already firing.
**Evidence:** PROVED by reading and measuring. `.github/workflows/docs.yml:5` triggers on `branches: [main, nightly]` with a shared `pages` concurrency group — **every nightly push republishes the public site from nightly**, overwriting main. The built site measures **876 MB** against GitHub Pages' documented **1 GB** cap (88% full) and every new lesson moves it closer; `docs/source/_extra/tutorials` alone is **723 MB**. And `MANIFEST.in:69-72` states *"The built docs are not a distribution artefact"* while pruning only `docs/.doctrees`, `docs/_build` and `docs/resources/tutorial` — `MANIFEST.in:19`'s `recursive-include docs *.html *.css *.js ...` then sweeps ~29 MB of built HTML into the sdist. Tracked `docs/` is **786 MB / 4,776 files**; `git count-objects` reports a **1.45 GiB** pack, which is what a `git clone` costs.
**Effort:** M.
**Why now:** The nightly-republish is live today. The Pages ceiling is a hard wall you will hit mid-lesson-batch with no warning.

---

### 18. Guard the missing-PySide6 launch, and fix the docs recipe that causes it
`pip install spacr` then `spacr` gives a bare `ImportError` — and the docs landing page prints exactly that recipe.
**Evidence:** PROVED by shimming `__import__`: `spacr.qt.run([])` → `ImportError: No module named 'PySide6'`, no hint. `spacr/qt/__init__.py:51` does an unguarded `from .app import launch`; PySide6 is extras-only (`setup.py:615`). `docs/source/index.rst:103-104` prints `pip install spacr` / `spacr   # launch the Qt GUI`. README:162 gets it right with `"spacr[qt]"`. While in there: `docs/source/index.rst:52` says *"spaCR is organised around five pipeline apps"* — PROVED `len(APPS) == 34`; and the landing page never links to `/tutorials/` at all.
**Effort:** S.
**Why now:** It is the literal first thing a new user does, and the documentation actively directs them into it.

---

### 19. One real Toxoplasma end-to-end lesson
The 40-lesson skeleton is complete and the coverage is genuinely impressive. What is missing is depth: nothing shows a real pooled screen going from plate to hit list.
**Evidence:** PROVED from `catalog/lessons_en.json`. Every lesson has exactly **5 scenes** (`overview → input → settings → run → output`); narration is 92–125 words for an entire lesson (≈45–60 s); one scene opening — *"Before the full run, review the visible settings and use a preview or test mode when one is available…"* — is **byte-identical in all 40**; and 38 of 40 carry the same objective, *"Configure the important inputs and settings."* Lesson titles are never translated (`07_mask` is "Mask" in es/fr/zh-CN/hi/ja) and NLLB artifacts survive in the descriptions (`zh-CN` *"…条件级 **fenotype** 相关"*, `ko` *"야간 지부"* — "night branch" for "nightly branch"). Also blocking: `TUTORIAL_MEDIA_NOTICE.md:4` limits the media to non-commercial use, and `qt/tutorial/scripts.py:260` narrates *"cyto for cells"*, wrong under Cellpose 4.
**Effort:** L.
**Why now:** This is the deliverable that makes spaCR adoptable by another lab — and it depends on item 5, because the demos are what a deep lesson would drive. Settle the licence question in the same pass.

---

### 20. Rewrite `sim.py` from spaCRPower and give it a GUI (#29b)
Untouched. Your R package is the reference implementation, and the Python side is still the 2024-era module.
**Evidence:** PROVED — `spacr/sim.py` is 1559 lines; greps for `spaCRPower`, `simulate_library`, `simulate_screen`, `brms`, `pymc`, `numpyro`, `betabinom` all return nothing. `sequencing_error` is **not** new (present at the `de15cac` baseline). No `sim` key in `APPS`; the Tk entry was deleted in favour of `spacr-run simulation` (`cli.py:455`).
**Effort:** L/XL. The real decision is the Bayesian fitting layer — `cmdstanr`+`brms` has no drop-in Python equivalent; choose between PyMC, NumPyro, cmdstanpy, or a direct implementation of the specific hierarchical model. `yardstick`→`sklearn.metrics` and `extraDistr`→`scipy.stats.betabinom` are free.
**Why now:** It has been deferred through three batches and it is the item that answers *"how many cells per well do I need?"* — the question that determines whether a screen is worth running at all. Also note `sim.py` coverage is still formally deferred behind this, so it keeps a hole open in the coverage picture too.

---

## What I would not do

**Do not restart the >95% coverage push.** You dropped it on 2026-07-27 and the repo has since drifted *up* to 92.6% on its own, with the core science modules at 93–98% (`utils` 98.2%, `io` 97.7%, `zstack` 98%, `deep_spacr` 93.1%). The remaining hole is almost entirely the legacy Tk GUI — `gui_core` 5.8%, `gui_utils` 34.3%, `gui` 53.9%, `gui_elements` 84.4%, together **2,114 of 6,225 total missed lines (34%)**. Writing tests for a GUI that `setup.py:495` no longer makes the default entry point is the worst hour-per-defect ratio available.

**Do not chase the `spacr-db-audit --probe` flake yet.** The mechanism is real and I can show the arithmetic — `database_concurrency.py:432` connects with `timeout=0.05` → 50 ms `busy_timeout`, then `:185` divides it by `attempts=40` for a **1 ms** per-attempt budget — but I could not reproduce the failure on demand across 5/5, 3/3, 8/8 under coverage, 10/10 under 40-way contention and 9/9 at 8/16/24 writers. One confirmed sighting. Fix the per-attempt arithmetic if you're in the file anyway; do not open an investigation.

**Do not fix `spacr-tutorial` by auto-fetching the Piper voice.** It is a maintainer video renderer, not a user-facing tutorial viewer, and it already prints a working `curl` command when the voice is missing (`qt/tutorial/engine.py:100-107`) — better than the "bare traceback" it was described as. Downloading a 63 MB model on first run to serve six English-only scripts against 40 shipped web lessons is effort pointed away from item 19. Unlist it from `console_scripts` or leave it; don't invest.

**Do not reverse BGR/RGB.** Still the right call for the reason recorded: it would change every crop on disk and silently invalidate any classifier trained on the current ordering, with no error to signal it. `image_colors.py` (58 lines, 97% covered) now gives it exactly one owner, which is the correct amount of attention.

**Do not chase `plugins.reload_plugins` or `cancellation.cancellation_requested`.** Both are referenced only from `tests/`, and both self-declare as dev conveniences (`plugins.py:392` says so in its docstring). There is no GUI affordance that silently fails — `grep reload_plugins spacr/qt/` is empty. Dead-ish API, not a broken feature.

**Do not open the conda-forge PR.** #32d is correctly parked at exactly the right line: the recipe is written and locally buildable (`conda-forge/recipe/recipe.yaml`), and the submission to `conda-forge/staged-recipes` is a third-party PR that needs your explicit go-ahead and your GitHub handle in `recipe-maintainers`. That is a decision, not a task."
  },
  "workflowProgress": [
    {
      "type": "workflow_phase",
      "index": 1,
      "title": "Survey"
    },
    {
      "type": "workflow_phase",
      "index": 2,
      "title": "Synthesise"
    },
    {
      "type": "workflow_agent",
      "index": 1,
      "label": "survey:backlog-audit",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "ad11c49fdac99f902",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785607549068,
      "queuedAt": 1785607528711,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /mnt/firecuda2/codex/repo/spacr && git show de15cac:spac…",
      "promptPreview": "**Audit the existing backlog against the current code.** This is the most important task
in this workflow — the user's first question is "what is still left?".

Read `/home/olafsson/.claude/projects/-home-olafsson-repo/memory/spacr_to_do.md` in full. It is a long markdown table of items, many added over several sessions,
each with an ID like #9c, #12e, #16j, #21c, #29b, #30c, #32e, #34b.

**For EV…",
      "lastProgressAt": 1785609097517,
      "tokens": 187752,
      "toolCalls": 132,
      "durationMs": 1548449,
      "resultPreview": "Audit complete. Every verdict below was reached by reading the code at HEAD `d882c0d7` (v1.4.9.9); "PROVED" means I executed it.

## Headline

**The eight items you flagged as open: 5 are done, 3 are not.** Done: #9e, #9c, #9d, #12e, #16j. Not done: **#21c** (untouched, and now locked in by a test), **#29b** (0% — nothing landed), **#12f** (untouched). Partial: **#30c** (UI done, the saturation bu…"
    },
    {
      "type": "workflow_agent",
      "index": 2,
      "label": "survey:new-surface",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "ae1daba1202c2ffbb",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785607540115,
      "queuedAt": 1785607528711,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /mnt/firecuda2/codex/repo/spacr && grep -n "classifier_e…",
      "promptPreview": "**Map the 36 new modules and what they are for.** For each, in one or two sentences:
what it does, whether it is reachable from the GUI and/or the CLI, and whether it is tested.

Then answer the question that matters: **is anything built but not wired up?** This codebase has
a documented history of exactly that — a backend landing without its GUI entry, or a setting
rendering a widget nothing read…",
      "lastProgressAt": 1785608444408,
      "tokens": 127734,
      "toolCalls": 84,
      "durationMs": 904292,
      "resultPreview": "# spaCR `nightly` @ `d882c0d7` — survey of the 36 new modules

Verified against `git diff --name-status de15cac..HEAD -- 'spacr/*.py' | grep '^A'` → exactly **36 files**, matching the brief.

---

## 1. The 36 modules

**Reachability legend:** GUI = reachable from `spacr-qt`; CLI = reachable from an installed console script or `spacr-run`. Coverage figures are from runs I executed (see §5).

### T…"
    },
    {
      "type": "workflow_agent",
      "index": 3,
      "label": "survey:health",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a8e02431a51b6143a",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785607545281,
      "queuedAt": 1785607528711,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /mnt/firecuda2/codex/repo/spacr && timeout 300 env MPLBA…",
      "promptPreview": "**Assess the health of the package as it stands.** Run the test suite (it is large; shard
it or select by directory) and report: how many tests, how many fail, and what the failures
actually are. Then measure coverage per module and report the ranking.

Also check, by running:
- Does `import spacr` pull in TensorFlow? It is banned; a guard exists at
  `tests/test_no_tensorflow_guard.py`. Verify th…",
      "lastProgressAt": 1785612125991,
      "tokens": 192278,
      "toolCalls": 164,
      "durationMs": 4580710,
      "resultPreview": "Confirmed — that one joins the leakage cluster. Final report:

---

# spaCR 1.4.9.9 health survey — `nightly` @ `d882c0d7`

Method: 14,083 tests sharded by file across 16 processes (no xdist installed), then every failing file re-run in **one** process twice (once under load, once quiet) to separate real defects from harness artifacts. Coverage measured as a second full sharded pass, `coverage com…"
    },
    {
      "type": "workflow_agent",
      "index": 4,
      "label": "survey:correctness-sweep",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a4fab97ec95200a9d",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785607536667,
      "queuedAt": 1785607528711,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "until ! pgrep -f "pytest -q -p no:cacheprovider -m" >/dev/n…",
      "promptPreview": "**Hunt for real defects in the code as it stands now.** Not a style review — look for
things that produce wrong results or crash.

Places this codebase has repeatedly gone wrong, as leads rather than a checklist:
- **Silent data loss in merges and joins**: a merge on the wrong key returns the same row count
  with null columns; a positional `.split('_')` on a key breaks the moment a timepoint appe…",
      "lastProgressAt": 1785609477889,
      "tokens": 217791,
      "toolCalls": 152,
      "durationMs": 1941221,
      "resultPreview": "The full suite is still running after ~80 min; I'll report the findings, all of which I verified independently.

## PROVED — crashes on the user's actual path

**1. `spacr_cellpose.py:185` and `:295` — `channel_axis=3` makes every Cellpose-4 `eval()` call raise.**
Cellpose 4.0.7 rejects `channel_axis=3` for *both* shapes spaCR produces: 2-D (`ValueError: 2D image provided, but channel_axis is not …"
    },
    {
      "type": "workflow_agent",
      "index": 5,
      "label": "survey:user-experience",
      "phaseIndex": 1,
      "phaseTitle": "Survey",
      "agentId": "a4c51095feb2cb5da",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785607553234,
      "queuedAt": 1785607528711,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /mnt/firecuda2/codex/repo/spacr && grep -in "notebook\\|j…",
      "promptPreview": "**Read the package as a user would meet it.** Install docs, README, the GUI entry
points, the CLI, the tutorials, the new i18n work, and the Sphinx docs that 54 commits went into.

Answer concretely:
- What happens on a clean install, and what would a new user hit first?
- Are the tutorials current, and do they match the code?
- The i18n work is large — how many languages, how complete, and is it …",
      "lastProgressAt": 1785609336384,
      "tokens": 198197,
      "toolCalls": 147,
      "durationMs": 1783150,
      "resultPreview": "I have enough evidence. Writing the report.

## spaCR 1.4.9.9 (`nightly`, `d882c0d7`) — user-facing survey

Verified by running the code in the repo's own conda env (editable install at `/mnt/firecuda2/codex/repo/spacr`), building the Sphinx site, rendering tutorial videos, generating and executing every synthetic demo, and probing the live GitHub Pages site.

---

## 1. The single most important …"
    },
    {
      "type": "workflow_agent",
      "index": 6,
      "label": "synthesise",
      "phaseIndex": 2,
      "phaseTitle": "Synthesise",
      "agentId": "a6632bfaf0b2e595c",
      "model": "claude-opus-5",
      "state": "done",
      "startedAt": 1785612130905,
      "queuedAt": 1785612127842,
      "attempt": 1,
      "lastToolName": "Bash",
      "lastToolSummary": "cd /mnt/firecuda2/codex/repo/spacr && grep -n "def generate…",
      "promptPreview": "Write the deliverable for the user, who has been away from this project and asked two questions:
**"what is still left on spacr_to_do?"** and **"what are the 20 next things I should put on it?"**


spaCR = spatial phenotype analysis of CRISPR screens in high-content microscopy. The user is a
Toxoplasma researcher running pooled screens on a Yokogawa high-content microscope.

REPO: /mnt/firecuda2/c…",
      "lastProgressAt": 1785612703851,
      "tokens": 110579,
      "toolCalls": 46,
      "durationMs": 572946,
      "resultPreview": "I have verified everything I need. Writing the deliverable.

---

# spaCR backlog status & next 20
`nightly` @ `d882c0d7`, v1.4.9.9 — every verdict below re-checked against the code today. "PROVED" = I ran it.

---

## Part 1 — what is still open

**Headline: 76 backlog IDs. 60 are DONE, 5 are OBSOLETE, 11 are genuinely still open or partial.** The list has been pruned honestly — the 850 commits s…"
    }
  ],
  "totalTokens": 1034331,
  "totalToolCalls": 725
}