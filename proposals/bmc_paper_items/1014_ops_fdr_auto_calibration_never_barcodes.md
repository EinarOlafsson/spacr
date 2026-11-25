# Item 1014: automatic genotyping error-rate calibration from never-should-appear barcodes, as the default for every library

- **Number:** 1014 (master, 2026-11-21). **Settings schema:** 957 (`DEFAULT_CHANGES[957]`, `SETTINGS_SCHEMA_VERSION = 957`). This lands after item 1013 (schema 956). If 1013 lands second, the two rows swap numbers by the rule in `settings._fold_older_defaults`.
- **Decided by:** the maintainer, 2026-11-21 ~15:00 (decision 1), then the coordinator's priority change: the Funk pilot must contain the BUILT-IN calibration, so this item is on the critical path (WS/Codex, P1 critical).
- **Code basis:** origin/nightly 19a95cd799. Spec only; nothing here is implemented.
- **Repo rule 43 (one operation, one code path):** this item EXTENDS the existing decoy and auto-threshold code, the decode's "Decoys & auto thresholds" section and the prefix "Spacer-prefix FDR & tuning" section. It adds a calibration *method* to the existing item-949 calibrator (`ops_decode.calibrate_fdr_scale`, `FDR_CALIBRATIONS`). There is no parallel calibrator, no second tuner and no second FDR formula: `ops_decode.fdr()` stays the only FDR. The never-barcode builder sits next to `decoy_library` and serves both decodes, and the full-barcode (Funk) and spacer-prefix (PERISCOPE) paths call the same estimator.
- **Design card and evidence:** `BMCspacr/reanalysis/experiment_designs/20261121_fdr_calibration_never_barcodes.md`. The script analysis that motivates the default is Supplementary Figure S2 J (paper repo, `supplementary/figure_S2/panel_j/`), with the development panel per well in `development/`. The numbers go into §6 when that panel is FINAL.

## 0. Why
- `ops_fdr_scale` = 1.268 (Funk) is an assumption: it is the ratio of library guides at Hamming distance 2 per guide (3.35) to those per decoy (2.62), measured once from the sequences. `ops_prefix_fdr_scale` = 1.0 (PERISCOPE) is also an assumption.
- Neither looks at the data. The item-949 'shuffle' calibration measures something else: a random library's call rate, which is 15 times the decoy FDR on Funk A1 (0.86 % vs 0.058 %). Being far from the wrong-call rate, it cannot calibrate it at the 0.1 % target.
- Never-should-appear barcodes are sequences next to library guides that no cell carries. They collect calls only by error, through the same decode, reads, quality, votes and thresholds as a real wrong call. Their call rate per opportunity, scaled to the library's own neighbour count, measures the wrong-call rate, and so the scale, for each run.

## 1. What exists today (origin/nightly 19a95cd799)
| Concern | Existing code | Now | Gap for 1014 |
|---|---|---|---|
| Decoys | `ops_sbs.decoy_library(library, k, seed)` (full barcode); `ops_prefix.prefix_decoy_library`, `prefix_nearmiss_decoys` (prefix) | Letter-shuffled guides, at least 2 from everything | No substitution-built never sets |
| Matcher | `ops_decode.matcher_for(library, decoys)` → `ops_sbs.LibraryMatcher`; prefix `_matched_rows` / `_shuffle_frame` | One whitelist of library + decoys | Reused as it is: a never set is passed where the decoys or the shuffled library go |
| Grid counts | `ops_decode._joint_field_candidates` → `_field_candidates` → `counts (…, 2)` real/decoy; `_field_shuffle_counts` (949) | Counts only, no guide identity | Needs, per field and grid point, the never calls and the opportunity sums (§2.3) |
| Calibrator | `ops_decode.calibrate_fdr_scale(counts, shuffled, …)`, `_needed_scale`, `shuffle_call_rate`; prefix `ops_prefix._calibrate_fixed`, `_well_report_949`; engine `ops_engine` ~3492–3586 | `method` 'shuffle' only raises the scale | Add `method` 'never' |
| Setting | `ops_decode_fdr_calibration`, `ops_prefix_fdr_calibration` ∈ `FDR_CALIBRATIONS = ("none", "shuffle")`, default 'none' | | Add 'never', make it the default |
| FDR | `ops_decode.fdr(real, decoy, k, scale)` = 100·max(scale,1)/k·decoy/real | | Unchanged; it gets the calibrated scale |
| Reports | `fdr_report`, `plate_fdr_summary`, `decode_thresholds_<well>.json`, prefix `site_summary.csv` + FDR report | `fdr_scale`, `fdr_scale_entered`, `calibration` block (949) | A calibration report per run (§2.5) |

## 2. Behaviour

### 2.1 Never-should-appear sets (one builder, both decodes)
New function `ops_sbs.never_barcodes(library, *, distance, k=4, seed=1, exclude=())`, beside `decoy_library`:
- `distance=1`, the **N1** set: a guide with one base substituted. The nearest guide is at distance 1, there is exactly one such parent, the barcode is not in the library, and it lies at least 2 from every barcode already taken. Up to k per guide, at k different positions, in sorted-guide order with `np.random.default_rng(seed)`. This is the decoy builder's ordering rule.
- `distance=2`, the **N2** set: two substitutions. The nearest guide is at distance exactly 2, and the barcode lies at least 2 from every barcode taken. Up to k per guide.
- It returns the barcodes plus, per guide, `m_d[g]`, the number of set barcodes at Hamming distance d from guide g for d = 1, 2, 3, as an int array aligned to the sorted library. It also returns `m_d` of the library itself (other guides at distance d). Everything is computed once per run, uses the masked index (`ops_sbs._library_index`), and is memoised like `matcher_for`.
- The prefix decode calls the same function on the prefix codes of its `PrefixLibrary`, over the sequenced cycles (cycle mask applied). Nothing is prefix-specific.

### 2.2 Extra vote passes (reuse, no new matching code)
- With 'never' on, each field is voted twice more, by the same call that votes it against the decoys, `_joint_field_candidates` (full barcode) or the 949 frame path (prefix):
  - against `matcher_for(library, N1)`;
  - against `matcher_for(library, N2)`.
- In each pass the never set takes the place of the decoys. The pixel votes are reused (`pixel_votes=`, item 952), as the shuffle passes already do.
- Cost: two vote passes per field, the same as two of the three 949 shuffle passes. In the full-barcode decode, 'never' runs no shuffle passes unless `ops_decode_fdr_calibration` is 'never+shuffle'. The spacer-prefix decode always runs and reports them under the PERISCOPE reporting rule (§3), so for it 'never' adds the two never passes to the existing three.

### 2.3 What each pass records (per field, per grid point)
- `never_calls` (cells called on the set) and `real_calls_pass`, the real calls in that pass, kept for the perturbation check.
- From the base pass (library + decoys), the opportunity sums. With `c_f(g)` = cells called guide g at that grid point in field f, they are `opp_d(X)_f = Σ_g c_f(g) · m_d(X, g)` for X ∈ {L, D, N1, N2} and d ∈ {1, 2, 3}.
- These are dot products of the per-guide call counts with the `m_d` vectors. `_field_candidates` already builds the per-object guide (`mine`), so it adds a `bincount` over guide index per grid point. Nothing else changes.
- The arrays are `(fields, *axis lengths)` like `shuffled`, so the tuner's split and bootstrap read them unchanged.

### 2.4 Estimator (one function, both decodes)
New `ops_decode.never_scale(real, decoy, never2_calls, opp2_N2, opp2_L, *, k, ci, reps, seed)`, beside `_needed_scale`:
- The per-opportunity rate is ρ2 = Σ never2_calls / Σ opp2_N2.
- The expected wrong library calls are E_L = ρ2 · Σ opp2_L, which gives FDR_cal = 100 · E_L / Σ real.
- The scale is s = k · E_L / Σ decoy. A sum with no decoy call is taken as `_AUTO_ZERO_DECOYS`, as the tuner already judges it.
- With `ci`, the result is the upper end of the 95 % field bootstrap (`_bootstrap_rows`, `_AUTO_REPS`, `_AUTO_BOOT_SEED`), the same convention as `_needed_scale`.
- The N1 rate ρ1 = Σ never1_calls / Σ opp1_N1 and the ratio ρ1/ρ2 are computed by the same function for the report. They do not enter s: neither library has guides at distance 1 from each other, so the library's distance-1 opportunity is 0. A library that does have distance-1 guide pairs adds the term ρ1 · Σ opp1_L to E_L. The formula is general, and the test in §5 covers it.

### 2.5 Calibration loop (EXTENDS `calibrate_fdr_scale`)
- `calibrate_fdr_scale(..., method="never", never=<arrays of §2.3>)` uses the same loop as 'shuffle':
  1. tune at the current scale;
  2. at the chosen setting, compute s on the tuning fields with `never_scale`;
  3. set the scale to max(s, 1) and tune again.
- Unlike 'shuffle', the scale can go down as well as up, because it is a measurement and not a floor. The loop stops when the chosen setting repeats. If it cycles, it takes the larger of the cycling scales (conservative), within the existing `max_rounds` = 20.
- The reported scale is the tuning-half value. The report half gets FDR_cal at that scale and its own point s with a CI. That is the held-out check.
- The prefix path keeps using it through `ops_prefix._calibrate_fixed` (one-setting grid) and the grid-search route (`ops_grid_search` ~524), with `never=` beside `shuffled=`.

### 2.6 Calibration report (per run, per well)
`<dst>/results/ops_decode/fdr_calibration_<well>.json` + `.csv`; prefix: `results/ops_prefix/fdr_calibration_<well>.json`. Keys:
- `method`, `entered_scale`, `scale`, `scale_ci95`, `scale_report_half`, `rounds`, `converged`;
- `rho1`, `rho2` (with CIs), `rho1_over_rho2`;
- `fdr_cal_pct` (with CI) and `fdr_at_entered_scale_pct` at the chosen setting;
- the shuffle call rate (`shuffle_call_rate`) when it was run, and always under the PERISCOPE reporting rule (prefix decode). See §3.
- `never_sets` (k, seed, sizes, distance profile mean m_d), `real_call_change_pct` per pass, and `fields` tune/report.
- The same keys are folded into `decode_thresholds_<well>.json`, `fdr_report` (`fdr_calibration`, `fdr_scale_entered` already exist) and `plate_fdr_summary`, which gets one row per well.
- One console line per well: `FDR scale calibrated from never-should-appear barcodes: 1.268 entered -> s [lo, hi]; FDR at the chosen setting x % (entered scale y %)`.

### 2.7 Guards
- **Perturbation:** for the N2 pass (the one that sets the scale), `real_call_change_pct` > 0.5 % logs a warning (`tr`). A never set takes some reads from the library, which can perturb the real calls. The N1 pass is report-only. Its distance-1 barcodes take about 1 % of real calls (measured −0.84 % Funk, −1.09 % PERISCOPE), so it is reported without a warning. Neither pass's calls ever replace the production calls.
- **No decoys** (`ops_decoy_library_k` 0 / `ops_prefix_decoy_k` 0): the scale has no decoy count to scale. 'never' then logs and falls back to 'none', the same validation 949 has (`ops_prefix.py:2089`). The message is reworded for 'never'.
- **Library smaller than 50 guides, or fewer than 30 never calls on the tuning half:** the scale stays as entered, and the report says `insufficient_never_calls`, the same pattern as `calibration_min_labels`.
- **Threshold mode 'manual':** the calibration still runs on one-setting grids (as `_calibrate_fixed` does for the prefix) and reports, but the thresholds are not re-tuned.

## 3. Settings
| Key | Change | Default (new files) | Old files (stamped < 957) | Status |
|---|---|---|---|---|
| `ops_decode_fdr_calibration` | new values 'never', 'never+shuffle' | **'never'** | 'none' | existing control; its new values are visible (not alpha), because it is the default |
| `ops_prefix_fdr_calibration` | same | **'never'** | 'none' | same |
| `ops_never_barcode_k` | new | 4 | n/a (new key) | **alpha** (`ALPHA_FEATURES[1014]`) |
| `ops_never_barcode_seed` | new | 1 | n/a | **alpha** |
| `ops_never_barcode_sets` | new: 'n1_n2' or 'n2' ('n2' skips the N1 report pass) | 'n1_n2' | n/a | **alpha** |

- `FDR_CALIBRATIONS` becomes `("none", "shuffle", "never", "never+shuffle")`. Its comment ("the first being the default") is rewritten: the default lives in the settings, not the tuple order.
- 'never+shuffle' runs both and takes the larger scale. It is kept for the PERISCOPE reporting rule: the prefix decode always REPORTS the shuffle call rate whatever the method (rule: matched decoy FDR + shuffle call rate in every PERISCOPE table), but uses it for the scale only under 'never+shuffle'.
- `ops_fdr_scale` (1.268) and `ops_prefix_fdr_scale` (1.0) stay. They become the *entered* scale, which is the starting point and the fallback, and their tooltips say so.
- Every new key goes into `_validate_ops_settings`/prefix validation, the type map, the "Decoys & auto thresholds" and "Spacer-prefix FDR & tuning" groups, `ops_grid_search` key lists (lines 78/90), the run journal (`spacr_resolved_parameters`), and the settings-flow docs (`docs/source/_generated/settings_flow.rst` regenerated).

**`DEFAULT_CHANGES` row (settings.py):**
```python
957: {'ops_decode_fdr_calibration': 'none', 'ops_prefix_fdr_calibration': 'none'},
```
`SETTINGS_SCHEMA_VERSION = 957`. Add the docstring paragraph: "957 is item 1014's automatic FDR-scale calibration from never-should-appear barcodes ('never'): files stamped before keep the entered fixed scale ('none')." Add the `docs/source/reproducibility_manifests.rst` line. An old settings file or run journal rerun gives the identical output (reproducible-via-settings rule).

## 4. Shared-path rules
- One builder (`never_barcodes`), one estimator (`never_scale`), one loop (`calibrate_fdr_scale`), one FDR (`fdr`).
- The prefix and full-barcode decodes differ only in how a field's reads are voted, which is the code they already have.
- `ops_live` (item 1000 parity) and `run_compare_calls` read the reported scale. They do not recompute it.

## 5. Tests (new file `tests/test_ops_fdr_never_1014.py`; CUDA_VISIBLE_DEVICES='' -n 4)
1. `never_barcodes`:
   - distances exact: N1 nearest = 1 with one parent, N2 nearest = 2, set-internal separation ≥ 2, none in the library;
   - deterministic for a seed, independent of library order;
   - `m_d` equals a brute-force count on a 200-guide library.
2. `never_scale` on synthetic counts:
   - with ρ2 known, s = k·ρ2·opp2_L/decoy exactly;
   - the CI upper end ≥ the point estimate;
   - zero decoys → `_AUTO_ZERO_DECOYS`;
   - a library with distance-1 guide pairs adds ρ1·opp1_L.
3. `calibrate_fdr_scale(method="never")`:
   - converges;
   - can lower the scale (unlike 'shuffle');
   - takes the larger scale on a cycle;
   - 'none' and 'shuffle' give results identical to today's (regression: the existing 939/942/949 tests pass unchanged).
4. Full barcode, end to end on the small A1 test subset (item 863 test data, `tests/data` OPS fixture):
   - the report file exists with every §2.6 key;
   - the FDR equals `fdr(real, decoy, k, scale)` at the calibrated scale;
   - `real_call_change_pct` < 0.5 %.
5. Prefix, end to end on the PERISCOPE fixture: the same, plus `shuffle_call_rate_pct` is present whatever the method.
6. `DEFAULT_CHANGES[957]`: an unstamped/956 file loads 'none' and gives byte-identical decode tables to today (`cli_repro` check); a 957 file gets 'never'.
7. Alpha: the three new keys are registered in `ALPHA_FEATURES[1014]`, hidden with alpha off, and the default still runs with alpha off (`test_alpha_registry`).
8. Ratchets before hand-over: raw-reader, nested-function, documented-callable and hash-comment. New documented callables (`never_barcodes`, `never_scale`) ship with arrival contract and 9-locale records in the same commit (catalogue rule).

## 6. Acceptance (paper lane, Home, CPU, through sched)
- **A. Script parity:** spaCR's report on Funk A1 (333 fields) and PERISCOPE CP186A W1 (85 sites) equals the S2 J script analysis (`reanalysis/fdr_calibration_1121/`) for ρ1, ρ2, s and FDR_cal, to 1e-9 relative on the same fields and the same never sets. The two are the same computation, ported.
- **B. Held-out:** the report-half s lies inside the tuning-half CI, or the difference is shown in the panel.
- **C. Reproducibility:** a rerun from the run journal gives identical tables, and a 956-stamped file gives today's tables.
- **D. Both error rates in every PERISCOPE table:** matched decoy FDR (at the calibrated scale) and the shuffle call rate.
- **Script results (S2 J, PRELIMINARY, 2026-11-21; `reanalysis/fdr_calibration_1121/tables/calibration_summary_{funk,periscope}.csv`):** these are the values acceptance A must reproduce. The figure stays the source of truth.

  | | Funk et al. A1 (333 fields, b = 1) | Ramezani et al. CP186A W1 (85 sites) |
  |---|---|---|
  | Base arm = production | 974,437 real / 1,772 decoy, exact | 85 of 85 sites exact |
  | Calls on never d2 / never d1 / decoys | 2,708 / 5,226 / 1,772 | 1,996 / 4,207 / 911 |
  | Calibrated scale s [95 % CI] | **1.356 [1.238, 1.475]**; 1.268 is inside: consistent | **1.965 [1.759, 2.160]**; 1.0 is outside: not consistent |
  | FDR at the working point, calibrated vs fixed | 0.062 % [0.057, 0.066] vs 0.058 % | **0.202 % [0.162, 0.246]** vs 0.103 % |
  | Decoy rate / never-d2 rate per opportunity | 0.95 [0.88, 1.04] | 0.65 [0.59, 0.72] |
  | ρ1/ρ2 (cost of one more substitution) | 7.3 [6.7, 7.8] | 8.9 [8.0, 10.0] |
  | Real-call change, N1 / N2 pass | −0.84 % / −0.08 % | −1.09 % / −0.21 % |
  | Joint-fit sensitivity s_joint | 1.30 | 1.70 |
  | b = 0.25 (item 1013) | s 1.356 [1.238, 1.475], FDR 0.062 % | n/a |

- **Consequences for the defaults:**
  - **Funk:** calibration confirms the fixed scale. With 'never' on by default, the Funk tuner should keep about the same working point. The full-grid replay (`funk_never_grid.py`, `post_grid.py`) records where it lands.
  - **PERISCOPE:** the prefix decoys under-count wrong calls about 2-fold. Under 'never' the prefix tuner will pick a stricter point to stay at 0.1 %. That is the intended effect, and it is reported per run.
- **N1 pass guard:** the real-call change of the N1 pass is above the 0.5 % guard (§2.7) on both datasets. The N1 pass is therefore a REPORT-only pass: its calls are never mixed with production calls, and the warning is expected for it. The guard applies to the N2 pass, which is the one that sets the scale.

## 7. Translations (tr task note)
New and changed tooltips:
- `ops_decode_fdr_calibration`, `ops_prefix_fdr_calibration`, `ops_fdr_scale`, `ops_prefix_fdr_scale` (now "entered scale");
- three new keys.

New and changed console and log strings: §2.6 line, §2.7 warnings and fallbacks. They go to the Codex task `/mnt/firecuda2/codex/scratch-docs/codex_inbox/tr_1014_task.md` (sources with SHA-256, 9 locales), written by L1/L2 at implementation once the strings freeze. This item writes no translations.

## 8. Lane split (file boundaries)
About 350–450 changed lines in `spacr/` plus about 350 test lines. 1–1.5 days of lane time.

| Lane | Owner | Files (exclusive) |
|---|---|---|
| **L1 core** | WS/Codex P1 | `spacr/ops_sbs.py` (`never_barcodes` only), `spacr/ops_decode.py` (`FDR_CALIBRATIONS`, `never_scale`, `calibrate_fdr_scale` method, `_field_candidates` per-guide bincount, `fdr_report`/`plate_fdr_summary` keys, calibration report writer), `tests/test_ops_fdr_never_1014.py` §5.1–5.3 |
| **L2 wiring** | WS/Codex P1, after L1 API freezes | `spacr/ops_engine.py` (~3492–3586: the two passes, arrays, report), `spacr/ops_prefix.py` (`_calibrate_fixed`, `_well_report_949`, validation 2087–2090, the two frame passes), `spacr/ops_grid_search.py` (key lists, `never=`), tests §5.4–5.5 |
| **L3 settings** | WS/Codex P1 | `spacr/ops_settings.py` (defaults, types, groups, tooltips), `spacr/settings.py` (`DEFAULT_CHANGES[957]`, schema, `ALPHA_FEATURES[1014]`), `docs/source/reproducibility_manifests.rst`, regenerated settings_flow, tests §5.6–5.7 |
| **L4 translations** | Codex tr | `tr_1014_task.md` → `spacr/qt/i18n_catalogs/*.py` after L3 freezes the strings |
| **L5 acceptance** | Home paper lane | §6, no `spacr/` files |

Conflicts:
- **With 1013:** L3 touches the same `DEFAULT_CHANGES` block (rows 956/957, one line each; the later lander rebases).
- **With 1011:** `settings.py` schema 955, the same kind of one-line rebase.
