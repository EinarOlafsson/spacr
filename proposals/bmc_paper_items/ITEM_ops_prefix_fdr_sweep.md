# Item 955: matched-FDR sweep report for prefix decodes (own and external reads)

- **Number:** 955 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features), off by default.
- **Rule (maintainer, 2026-11-15):** all analysis code ends up in spaCR. This item replaces the scripts `periscope_benchmark/scripts/score_s2_fdr_sweep.py`, `edge_ablation/scripts/reconcile.py` and `edge_ablation/scripts/shuffle_by_row.py` (register: `BMCspacr/reanalysis/SCRIPT_TO_SPACR_REGISTER.md`, gap D1).
- **Depends on:** 942 (decoy FDR, shared tuner), 943 (`prefix_external`, common rules), 949 (shuffle rate reporting). Build after 949.
- **Records:** `fdr_sweep` is a new documented callable, so its catalogue and arrival records and the 9-locale records ship in the same commit.

## Spec
1. **Callable** `spacr.ops_prefix.fdr_sweep(reads, cells, library, settings)` and **ops phase** `prefix_fdr_sweep`.
   - It scores one or more methods' reads: spaCR's own, or external per-spot reads through 943's `prefix_external`.
   - One matcher is used against the real library, the decoy library (`spacr.ops_sbs.decoy_library`) and the sequence-shuffled library (`shuffled_library`).
2. **Settings:**
   - `ops_fdr_sweep_targets` (default 0.5, 1, 2, 5 %)
   - `ops_fdr_sweep_max_distance` (0, 1)
   - `ops_fdr_sweep_min_reads` (1, 2, 3)
   - `ops_fdr_sweep_rules` (maxcount, purity50, purity100, from 943)
   - `ops_fdr_sweep_quality_quantiles`
   - `ops_fdr_sweep_bootstrap` (2000)
   - `ops_fdr_sweep_seed` (0)
   - `ops_fdr_sweep_shuffle_seeds` (0, 1, 2)
3. **For each FDR target:** pick the configuration with the most genotyped cells at pooled FDR ≤ target. Write `fdr_sweep_result.json` plus a table with:
   - genotyped %, decoy FDR and decoy counts;
   - the barcode-shuffle call rate per row (mean over the shuffle seeds; this absorbs shuffle_by_row.py);
   - site-bootstrap CIs, and the paired difference between methods at matched FDR.
4. **Off by default:** without the phase, outputs are unchanged. No schema bump.

## Proof check (pre-set)
- A settings-file rerun on the 85 PERISCOPE CP186A Well1 sites reproduces `periscope_benchmark/out/s2_fdr_sweep_85sites/fdr_sweep_result.json` exactly: chosen configurations, counts, CIs.
- From the same code path it also reproduces the `reconcile.py` sets A, B and C (`edge_ablation/reconcile/`) and the `shuffle_by_row.py` rates (`out/s2_fdr_sweep_85sites/shuffle_by_row.csv`).
- Synthetic tests: a known decoy and shuffle count gives the expected FDR and rate; ties keep the stricter configuration.
