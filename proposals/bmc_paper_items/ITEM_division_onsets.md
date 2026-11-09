# Item spec: division analysis: onset of pre-division changes

- **Lifecycle:** future (alpha), behind "Show alpha features".
- **Number:** take the next free item number; check `features/feature_list/` and the claims files first. Related:
  - 771: division-aligned trajectories, done, alpha; `spacr.division_analysis`;
  - 864: units in minutes and µm, open;
  - 865: budding cycle model, open;
  - 866: division prediction, open;
  - 867: per-movie bootstrap, open.
  This spec covers the one step with no item.
- **Rule:** a spaCR settings-file run (spacr-run or GUI Import settings + Run) must reproduce the reference outputs identically on the validation data. Every seed is fixed as in the reference. Every new documented callable ships with its arrival contract and 9-locale records in the same commit.
- **Code basis:** origin/nightly.
- **Paper-critical:** yes.
- **Paths:** R = `/nas_mnt/data/sync/papers/BMCspacr/reanalysis`. Source inventory: `R/SPACR_PROVENANCE.md` (port P21).

## Why
The onset of each measure's pre-division change uses `onset()` from a custom stage-2 script. There is no counterpart in `spacr.division_analysis` on origin/nightly. This item only makes the step a spaCR function.

## Reference implementation
`R/division_biology/code/stage2_analyse.py::onset(te, measure)`:
- take the per-measure effect table at relative time `rel_t` ≤ 0;
- sign = sign of g at the last frame before division;
- significant where the CI excludes 0 in that direction;
- if the last frame is not significant, return NaN;
- otherwise walk back from division while it stays significant;
- the onset is the earliest `rel_t` of that unbroken run.

Inputs come from `da.unit_changes` / `effect_size` (stage2 `analyse`). Outputs are in `R/division_biology/tables/`, with provenance in `R/division_biology/provenance.json`.

## What to build
`spacr.division_analysis.onset_times(effects, settings)`, which returns one row per measure (onset, sign).
- Settings: `division_onset` (bool), `division_onset_ci` ('ci_low/ci_high', the columns of the effect table).
- It runs inside the division analysis as part of its outputs, `onsets.csv`.
- Once 864 lands, onsets are also reported in minutes.

## Validation data and acceptance
- **Data:** the stage-2 effect tables of the 13 datasets (`R/division_biology/tables/`).
- **Acceptance:** identical output to the reference at `R/division_biology/tables/`: the onset and sign per dataset × measure are identical to `stage2_analyse.onset`.
- Dependency: `stage1_measure.py` reads movies through an external movie reader outside spaCR. An end-to-end spaCR rerun also needs a spaCR movie reader for those 13 datasets; that is not part of this item.

## Effort
- **Effort:** S.
