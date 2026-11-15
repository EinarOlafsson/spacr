# Item 952: read mode for barcode reads (log | raw | auto)

- **Number:** 952 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features). New settings get 9-locale strings; the deck entry goes on the existing decode line in `tools/refresh_alpha_deck.py` GROUPS.
- **Maintainer decision:** 2026-11-14, "Held-out check, then build"; the held-out check passed on 2026-11-15.
- **Evidence:** /nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/edge_ablation/REPORT_round3.md (rounds 1–2: REPORT_round1.md, REPORT_round2.md in the same folder).
  - **PERISCOPE CP186A Well1, 75 held-out sites:** auto picks raw at every FDR level. With the f1 + f3 variant, genotyped cells rise from 42.31 to 47.05 % at FDR ≤ 1 % (+4.74 [+4.32, +5.21]). That is level with PERISCOPE on the same cells (+0.62 [+0.47, +0.79] at ≤ 1 %), at 99.7 % same gene.
  - **Funk plate A A1 (325 held-out fields) and plate B A1 (333 fields):** auto picks log at every level, Δ 0.0. Forced raw would cost 50–67 points.
  - **Cost:** the raw read takes 2 s CPU per Funk field and 40 s per CP186A site.
- **Order:** built by master in a master lane (2026-11-15).

## Spec
1. **Setting** `ops_read_mode` = `log` (default) | `raw` | `auto`.
   - **log:** current behaviour, unchanged: −LoG σ 1, 3×3 maximum, per-channel median normalisation, library calibration in the OPS decode.
   - **raw:** PERISCOPE's read. The maximum of the unfiltered pixels over a radius-3 disk at each spaCR peak (detection unchanged), and the brightest channel per cycle without per-channel median normalisation. In the OPS decode, raw also bypasses library calibration (as tested in round 3).
   - **auto:** chooses log or raw per well, on decoys only.
2. **One implementation** in `spacr/ops_sbs.py`: a `read_values(stack, filtered, peaks, mode)` helper plus `called_bases(..., normalise=False)`, used by both `prefix_decode` and the OPS decode. No second copy.
3. **auto in the shared tuner** `spacr.ops_decode.tune_thresholds`.
   - Add a leading axis `("read_mode", [1.0, 0.0])` (1 = raw, 0 = log; log last so an exact tie keeps the current default).
   - Use objective `recall_at_fdr`, `fdr_max` = the well's target, `ci=True`, `tune_fraction=0.5`, `seed=0`, `unreachable='lowest_fdr'`. Only the tuning share chooses, and only decoy calls are read.
   - Per field, the candidate reads are computed once and called in both modes. `chosen["read_mode"]` applies to the whole well.
   - The per-mode grid rows go into `threshold_tuning_<well>.csv`, and the chosen mode into the journal, so a settings rerun is byte-identical.
4. **Schema:** no bump while the default stays log and the output is byte-identical. A later item that makes auto or raw the default adds the bump and a DEFAULT_CHANGES row, after confirmation on CP186B Well1 and on the raw CP186A images.

## Acceptance tests (pre-set)
- **Off-switch:** with `log` (default) outputs are byte-identical to today on 2 Funk fields and 2 CP186A sites.
- **Harness identity:** `raw` and `auto` reproduce the round-3 harness on the round-3 sets:
  - CP186A report half at FDR ≤ 1 %: 47.05 % for raw + f3; mode chosen raw at every level;
  - Funk A and B: auto picks log, so the outputs equal log.
  - Exact equality of the per-unit counts where the harness used spaCR functions; otherwise state the tolerance.
- **Tuner:** a synthetic well where raw wins on decoys and one where log wins; ties keep log; no reference calls are read (a test enforces it).
- **Time:** the extra cost of auto is ≤ 3 s per Funk field and ≤ 60 s per CP186A site.
- **Ratchets:** raw-reader, nested-function, documented-callable, hash-comment; catalogue symbols and 9-locale records in the same commit.
