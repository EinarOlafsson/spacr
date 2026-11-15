# Item 956: split and merge counts, identity and displaced controls in mask agreement

- **Number:** 956 (assigned by master 2026-11-15). **Lifecycle:** alpha (Show alpha features), off by default.
- **Rule (maintainer, 2026-11-15):** all analysis code ends up in spaCR. This item replaces the scorer script `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/scripts/score_s3_nuclei.py` (register: `BMCspacr/reanalysis/SCRIPT_TO_SPACR_REGISTER.md`, gap D2).
- **Gap:** `spacr.ops_large_sites.mask_agreement` returns only n_spacr, n_reference, matched, precision and recall (IoU ≥ 0.5, border objects dropped). It has no split or merge counts and no built-in controls. Related: ITEM_ops_reference_mask_matching.md (matching and F1), READINESS row 922.

## Spec
1. **`ops_mask_agreement_split_merge`** (bool, default False). Each object is assigned to the object in the other image that holds more than 50 % of its pixels (majority overlap, both ways), on the overlap of both frames, border objects dropped.
   - **Split:** a reference object receives two or more objects.
   - **Merge:** an object of the compared image covers two or more reference objects.
   - Report per site: split and merge counts and rates (per reference object), and F1 at IoU 0.3, 0.5 and 0.7.
2. **`ops_mask_agreement_identity_control`** (bool, default False when off). Re-read the reference as the compared image. It must give F1 1, 0 splits and 0 merges.
3. **`ops_mask_agreement_displace_px`** (int, default 30; 0 = off). A displaced negative control: the compared labels are shifted by that many px in y and x, and F1 must collapse.
4. **Output:** one row per site plus a pooled row with site-bootstrap 95 % CIs (2,000 resamples, seed 0).
5. **Default off:** outputs are unchanged, so no schema bump.

## Proof check (pre-set)
A settings-file rerun on PERISCOPE CP186A Well1, 10 iterate sites (3, 15, 27, 36, 45, 55, 64, 75, 86, 98), reproduces the script's outputs exactly:
- per-site F1, count ratio, precision, recall, splits, merges, the identity control and the displaced control;
- pooled values with CIs.

| Input / output | Path |
|---|---|
| Script | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/scripts/score_s3_nuclei.py` |
| spaCR labels | `/mnt/wd4tb/periscope_p1_work/seg_s3_945/CP186A_Well1` (archive a copy to the NAS before the proof) |
| PERISCOPE masks and 04Align shifts | `/mnt/firecuda2/codex/scratch-docs/periscope_iter/workspace/analysis/20200805_A549_WG_Screen` |
| DNA images | `/mnt/firecuda2/codex/scratch-docs/periscope_iter/images/20200805_A549_WG_Screen/images_corrected_cropped/CP186A_Well1` |
| Reference outputs | `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/out/s3_nuclei_result.json`, `/nas_mnt/data/sync/papers/BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/out/s3_nuclei_per_site.csv` |

- **Frame:** the reference masks are in the 04Align-shifted SBS frame. The compared labels are shifted into it before matching, exactly as in the script's `overlap()`.
- **Synthetic tests:** a known split, a known merge, a shift, and the identity pair.
