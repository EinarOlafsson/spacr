# Item 965: PERISCOPE-style large-site nuclei default R2, with seeded cells and a small-piece nucleus merge

- **Number:** 965 (reserved by master on 2026-11-16). High priority: this completes the segmentation freeze.
- **Maintainer decision (2026-11-16):** "R2". It was chosen against the maintainer's own point annotations (1,261 points on 20 PERISCOPE CP186A crops, annotated blind in spaCR's Make Masks point mode, item 957). Earlier decision: R1 applies to the PERISCOPE / large-site path only, and Funk's phenotype segmentation stays unchanged.
- **Schema:** 943 (nightly is at 942; master reserves it). `DEFAULT_CHANGES[943] = {'seg_cells_from_nucleus_seeds': False, 'seg_merge_small_nuclei_frac': 0.0}`, so that older settings files reproduce the current large-site output byte for byte.

## Gap
On the large-site path (`spacr.ops_large_sites.segment_cell_painting_site`, the `cell_painting_phalloidin` preset), nuclei are paired one-to-one with cells. A cell outline that holds two nuclei keeps one of them, and a nucleus with no cell is dropped.
- On PERISCOPE this drops 11–13 % of nuclei. 84 % of those are a second, real nucleus inside a cell outline that the cell model merged (`s3_phenotype/split_tuning/REPORT_interim.md`).
- Seeding the cells from the nuclei (R1) removes the loss but adds small fragments. R2 merges those fragments.

## Spec
1. **New setting `seg_merge_small_nuclei_frac`** (float ≥ 0, default 0.0 = off globally; set to 0.25 on the large-site path by this item).
   - **When:** after the nucleus mask step and before cell seeding.
   - **What:** every nucleus label whose area is below frac × the site's median nucleus area is merged into the touching nucleus with which it shares the longest 4-connected border.
   - **Chains:** chains of small pieces are followed until they reach a kept nucleus.
   - **Isolated pieces:** a small piece touching no nucleus is kept. The rule never deletes.
   - **Labels:** relabelled consecutively, deterministically.
   - **Reference implementation:** `BMCspacr/reanalysis/periscope_benchmark/s3_phenotype/split_tuning/out_r23/scripts/r2_merge.py` (`merge_small_nuclei(nuclei, frac=0.25)`), which has a self-test. Port it, keeping one implementation in spaCR.
2. **Large-site path defaults** (`segment_cell_painting_site`, preset `cell_painting_phalloidin`): `seg_cells_from_nucleus_seeds=True` and `seg_merge_small_nuclei_frac=0.25`.
   - Funk's phenotype path (`ops_engine._ph_segment_field`) does not read either setting and stays byte-identical.
3. **Reporting:** per site, record the merged-piece count and the site median nucleus area in the site summary. The run report gets one line: "R2: merged n small nuclear pieces (x % of nuclei)".
4. **Docs and strings:**
   - 9-locale tooltip and help for the new setting;
   - the updated docstring of `segment_cell_painting_site`;
   - arrival contract and catalogue records in the same commit;
   - a release-note line in simple terms.

## Proof check (pre-set)
1. **Identity to the decided arm:** the landed code on the 20 hand-check crops (sites 3, 15, 27, 36, 45, 55, 64, 75, 86, 98 of CP186A Well1; crop list in `handcheck_crops/crop_manifest.csv`) gives masks byte-identical to `s3_phenotype/split_tuning/out_r23/crops/labels/<crop>_R2_nuclei.tif` (20/20).
2. **Off-switch:** with the pre-943 values (`seg_cells_from_nucleus_seeds=False`, `seg_merge_small_nuclei_frac=0`), the large-site output is byte-identical to nightly on the 4 timing sites (3, 45, 64, 86).
3. **Funk unchanged:** the Funk phenotype path on 2 plate-A fields is byte-identical before and after the change.
4. **Truth table, reproduced from the landed masks,** on the maintainer's points: pooled over 20 crops, crop bootstrap of 2,000 draws, with the depth/chord border rule. Scorer: `s3_phenotype/point_truth_1116/scripts/score_point_truth.py`. The control rows must show:

| Source | F1 | Precision | Recall | Count ratio | Fragments / 100 |
|---|---|---|---|---|---|
| R2 (this item) | 0.932 [0.921, 0.943] | 0.913 | 0.952 | 1.042 | 5.1 |
| A0, current default (control) | 0.932 [0.916, 0.945] | 0.957 | 0.908 | 0.949 | 1.2 |
| PERISCOPE, deposited masks (control; our scoring of its deposited masks) | 0.866 [0.814, 0.895] | 0.830 | 0.905 | 1.090 | 14.4 |
| PERISCOPE, published | n/a in paper (no point-truth nuclei metric published) | | | | |

5. **Speed:** the merge step costs under 2 s per site (measured 1.7 s, range 0.7–3.0 s). The per-site time is reported against A0 on the same machine.
6. **Controls:**
   - identity rerun, byte-identical;
   - shifted negative (outlines offset 30–60 px): F1 about 0.02.
