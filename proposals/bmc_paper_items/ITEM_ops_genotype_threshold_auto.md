# Item spec: OPS genotyping thresholds, manual or auto (maximise genotyped cells at a maximum decoy FDR, chosen per well)

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** 939 (reserved 2026-11-13). Built after 932 (decode speed) lands. Related: 931 and 932 (settings-schema version table), S2-it1 arm (iii).
- **Decided by the maintainer, 2026-11-13:** "in spaCR the user should be able to set the thresholds for spot detection, or set spaCR to auto, where spaCR tries to maximise detection and minimise false discovery rate. One of the options should be FDR max, and its default should be 0.1."

## Problem and goal
spaCR's decode uses fixed thresholds. In the S2 proof, plate D's false-call rate was 0.121 % with the production thresholds, and 0.101 % with one global setting chosen from decoys on all plates. Choosing the two call thresholds per well from that well's own decoys (S2-it1 arm (iii)) kept every plate at or below 0.1 %: plate D reached 0.093 % (CI to 0.100 %) with 67.9 % of linked nuclei genotyped, against Funk's 57.7 % and Brieflow's 59.1 %. Today that selection exists only in scripts. The goal is to put it into spaCR as a deployable mode that uses only decoys and never a reference call set.

## Current state (origin/nightly)
- Decode thresholds are in `spacr/ops_settings.py`:
  - `ops_read_threshold` (315.0);
  - `ops_spot_min_peak` (20.0);
  - `ops_single_read_quality` (0.03);
  - `ops_pixel_quality` (0.1);
  - `ops_decode_pixel_fallback`;
  - `ops_decoy_library_k`;
  - `ops_fdr_scale` (1.268).
- The decoy FDR is computed per well in `spacr/ops_decode.py` (per-well summary rows).
- **Mapping to check while implementing:**
  - The S2-it1 scripts tune a "single-read floor" (production rule 0.06) and "pixel quality" (0.10).
  - Confirm which nightly keys these are (expected: `ops_single_read_quality` and `ops_pixel_quality`) and why the floor default differs (0.03 vs 0.06). Record the answer in this spec before coding.
- **Reference implementation:**
  - `BMCspacr/reanalysis/proof_stage1_stage2/iterations/scripts/s2it_analyse.py` (grid FLOORS × pixel quality; selection on a random half of each well's fields; confirmation on the other half);
  - `s2it_global.py` (one global setting);
  - results in `iterations/out/s2it/`.

## Proposed design
**Settings** (new; alpha):

| Setting | Type | Default | Meaning |
|---|---|---|---|
| `ops_decode_threshold_mode` | 'manual' \| 'auto' | 'manual' | manual = today's fixed thresholds; auto = chosen per well from decoys |
| `ops_decode_fdr_max` | float, percent | 0.1 | maximum decoy-estimated FDR that auto may accept |
| `ops_decode_auto_scope` | 'well' \| 'plate' \| 'global' | 'well' | where auto chooses one setting |
| `ops_decode_auto_grid` | dict or None | None (the S2-it1 grid) | candidate values per tuned threshold |
| `ops_decode_auto_ci` | bool | True | require the upper 95 % CI of the FDR (not the point estimate) to be ≤ `ops_decode_fdr_max` |
| `ops_decode_auto_tune_fraction` | float | 0.5 | share of fields used to choose; the rest only reports |
| `ops_decode_auto_seed` | int | 0 | field split seed |

**Manual mode** keeps every threshold editable exactly as today, including `ops_read_threshold` and `ops_spot_min_peak` (spot detection), plus the two call thresholds.

**Auto mode, per well:**
1. Split the fields into tuning and report halves (seeded).
2. On the tuning half, evaluate the grid.
3. Keep settings whose FDR (upper CI if `ops_decode_auto_ci`) is ≤ `ops_decode_fdr_max`.
4. Choose the one with the most nuclei genotyped. Ties go to the stricter setting.
5. Apply it to the whole well, and report FDR and genotyped share on the report half.

If no setting meets the target, the well gets the strictest grid setting and a warning, and the well's summary records `auto_met=False`.

**Recorded per well**, in the decode summary and the run journal: chosen thresholds, tuning and report FDR with CI, genotyped share, and `auto_met`.

**Spot-detection thresholds in auto mode** (`ops_read_threshold`, `ops_spot_min_peak`) stay fixed in this item: S2-it1 validated tuning of the two call thresholds only. Adding spot thresholds to the auto grid needs its own proof (open question 1).

**GUI:**
- the mode switch;
- FDR max next to it;
- a per-well table of the chosen thresholds after the run.

**Settings version:** none needed while 'manual' stays the default. If the maintainer later makes 'auto' the default, add a DEFAULT_CHANGES row (931 mechanism).

## Acceptance tests
1. **Identity with the script:** on plates A–D (23 wells), auto mode with the S2-it1 grid, seed and split reproduces arm (iii)'s chosen thresholds and calls per well, compared by md5 of sorted call tables and per-field stats.
2. **Manual mode:** byte-identical to nightly.
3. **FDR target:** with `ops_decode_fdr_max` = 0.1, every well's report-half FDR upper CI is ≤ 0.1 % on A–D, or `auto_met=False` is recorded.
4. **Deployability:**
   - auto never reads Funk or Brieflow calls (a test with those files absent);
   - the decoy library is required, with a clear error if `ops_decoy_library_k` and the library give no decoys.
5. **Time:** auto adds the grid evaluation on cached reads. Report CPU s per well against manual, in the same run; a one-off per well, so it should be minutes, not hours.
6. **Small-data guard:** a well with too few decoys to estimate the FDR at the target returns `auto_met=False` instead of a falsely confident pick (cf. item 935's FDR floor).

## Translations and catalog
New settings, tooltips and any new public callables ship with their arrival contract and nine-locale records in the same commit.

## Open questions for the maintainer
1. Should auto also tune spot detection (`ops_read_threshold`, `ops_spot_min_peak`)? That needs a separate proof against the same FDR rule.
2. Should the default scope be per well (validated, best on plate D) or per plate (simpler, fails plate D by 0.001)?
3. Should auto use the upper CI (stricter) or the point estimate of the FDR?
4. Should 'auto' become the default after it lands (with a settings-version row)?
