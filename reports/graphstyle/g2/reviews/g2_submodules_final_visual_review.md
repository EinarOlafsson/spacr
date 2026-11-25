> Curated publication copy. Original source: `.codex-audit/g2_submodules_final_visual_review.md`; original SHA256: `889542c6243ac98cd4c96c5d90beea1e70e3df22abf84120dc65e7091bcf5b6f`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# G2 submodules final visual review - job 1683

PASS for the bounded synthetic fixtures inspected here. The two presentation defects identified in [g2_submodules_visual_review.md](g2_submodules_visual_review.md) are resolved: used/reference cutoff roles are visibly distinct and the saved exports fit the 170 x 275 mm paper limit. No additional defect was found in the 16 newly emitted PNGs. The historical review and its failed intermediate measurements remain evidence; this document does not overwrite them.

The parent executed job 1683. This reviewer executed no Python, imports, tests, renders, model jobs or GPU work, and changed no production source. Review used individually opened actual PNGs, JSON receipts, direct `cmp`, hashes, `identify` metadata and `pdfinfo`/raw PDF MediaBox reads. PDF page dimensions and post-writer diagnostics were checked; no independent PDF raster-render inspection is claimed.

## Exact review coverage

Evidence root: [reports/graphstyle/g2/evidence/](../evidence). Every row below represents separate individual opening of the two listed light/dark files, not a composite thumbnail. Total: **16/16 PNGs**, including all four new original producer writes.

| Evidence directory and relative case/file | Light | Dark | Visible finding |
|---|---|---|---|
| `submodules-after-width-roles/{theme}/training_images/figure.png` ([light](../evidence/submodules-after-width-roles/light/training_images/figure.png), [dark](../evidence/submodules-after-width-roles/dark/training_images/figure.png)) | Viewed | Viewed | Both selected grayscale images and paired categorical labels remain intact. Complete caption below. Existing horizontal blank space remains; physical panels were preserved and the saved page now fits. |
| `submodules-after-width-roles/{theme}/test_images/figure.png` ([light](../evidence/submodules-after-width-roles/light/test_images/figure.png), [dark](../evidence/submodules-after-width-roles/dark/test_images/figure.png)) | Viewed | Viewed | All five original diagnostic planes and titles remain in order; complete below-panel caption, no clipping. |
| `submodules-after-width-roles/{theme}/apply_images/figure.png` ([light](../evidence/submodules-after-width-roles/light/apply_images/figure.png), [dark](../evidence/submodules-after-width-roles/dark/apply_images/figure.png)) | Viewed | Viewed | All four original diagnostic planes and titles remain in order; complete caption and intact categorical colors. |
| `submodules-after-width-roles/{theme}/thresholds/figure.png` ([light](../evidence/submodules-after-width-roles/light/thresholds/figure.png), [dark](../evidence/submodules-after-width-roles/dark/thresholds/figure.png)) | Viewed | Viewed | Both well histograms, original cutoff positions and complete n/coefficient notes remain. Used is dashed and reference dotted in actual plots and visible keys. Dark backing preserves the distinction. |
| `submodules-after-width-roles/{theme}/test_images/cellpose_result_000.png` ([light](../evidence/submodules-after-width-roles/light/test_images/cellpose_result_000.png), [dark](../evidence/submodules-after-width-roles/dark/test_images/cellpose_result_000.png)) | Viewed | Viewed | Actual nested producer export retains all planes/titles/caption with the correct exterior ground. |
| `submodules-after-width-roles/{theme}/apply_images/cellpose_result_000.png` ([light](../evidence/submodules-after-width-roles/light/apply_images/cellpose_result_000.png), [dark](../evidence/submodules-after-width-roles/dark/apply_images/cellpose_result_000.png)) | Viewed | Viewed | Actual nested producer export retains all planes/titles/caption with the correct exterior ground. |
| `submodules-default12-before/{theme}/thresholds_default12/figure.png` ([light](../evidence/submodules-default12-before/light/thresholds_default12/figure.png), [dark](../evidence/submodules-default12-before/dark/thresholds_default12/figure.png)) | Viewed | Viewed | All 12 baseline wells present in four columns, with their n=6 and BC=0.72 titles. Baseline width is oversized; dark references have poor contrast against the exterior ground. |
| `submodules-default12-after/{theme}/thresholds_default12/figure.png` ([light](../evidence/submodules-default12-after/light/thresholds_default12/figure.png), [dark](../evidence/submodules-default12-after/dark/thresholds_default12/figure.png)) | Viewed | Viewed | All 12 sorted wells appear in three columns and four rows. Every per-well n/BC entry is readable in the complete three-column key. BC definition and preservation sentence are visible once. No panel, label or key clipping/overlap observed. Used dashed/reference dotted patterns remain clear in every panel and both key glyphs. |

The default12 fixture retains all 72 input rows, including each original missing signal. Each histogram shows the five original finite signals. The common key explicitly defines BC as bimodality coefficient; it does not replace or recompute the original value. The measured helper test passed with unchanged 40 x 40 mm panels, true 8 pt text, all 12 names, exact numerical payloads and distinct, nonoverlapping panel/text envelopes.

## Actual exported page dimensions

All final PNG metadata reports 118.11 pixels/cm, equivalent to 299.9994 dpi after integer PNG density encoding; the normal writer was explicitly called with 300 dpi. PNG physical sizes below use the recorded nominal 300 dpi. Using encoded density instead changes the largest dimensions by less than 0.001 mm and does not change any result.

The PDF values come from each saved PDF MediaBox, independently of `Figure.get_size_inches`. Both themes have identical dimensions in every row. The PDF image-family width is stored as **481.8897637795 pt**, equal to 170 mm within PDF decimal precision; the rounded `pdfinfo` display of 481.89 pt must not be treated as a measurable excess.

| Export | PNG pixels | PNG mm at 300 dpi | PDF MediaBox mm | Bound |
|---|---:|---:|---:|---|
| training images | 2007 x 1170 | 169.926 x 99.060 | 170.000 x 99.132 | Pass |
| test images | 2007 x 920 | 169.926 x 77.893 | 170.000 x 77.905 | Pass |
| apply images | 2007 x 843 | 169.926 x 71.374 | 170.000 x 71.408 | Pass |
| two-well thresholds | 1307 x 1144 | 110.659 x 96.859 | 110.688 x 96.919 | Pass |
| default12 thresholds | 1931 x 3234 | 163.491 x 273.812 | 163.492 x 273.838 | Pass |
| test images, actual producer PNG | 1837 x 877 | 155.533 x 74.253 | Not emitted | Pass |
| apply images, actual producer PNG | 1837 x 830 | 155.533 x 70.273 | Not emitted | Pass |
| untouched default12 baseline | 4800 x 2700 | 406.400 x 228.600 | 406.400 x 228.600 | Baseline width fails |

The exact final default12 MediaBox is `[0 0 463.441074065 776.2339320866]` pt. Its measured final height is 273.838081597 mm. Tight-cropped producer PNG dimensions were checked separately from the complete figure export; no canvas-size substitution was used.

## Scientific and image identity evidence

The four changed original families compare to the complete immutable [submodules-before/report.json](../evidence/submodules-before/report.json); default12 compares to its separately rendered untouched baseline. The harness receipts record exact successful comparisons for eight and two theme/case pairs, respectively. Independent JSON comparison also found all **40 source/tables/rng_before/rng_after field comparisons exact** across these ten pairs.

Independent direct `cmp` found **270/270** CSV/NPY artifacts in the four-family subset byte-identical to the original baseline and **106/106** in default12 byte-identical to its new baseline: **376 exact files, zero changed, zero candidate-only and zero missing scoped baseline files**. These include registered inputs, masks, tables, histogram/cutoff coordinates and image/LUT arrays. The original default12 signal table hash remains `8b700f77266eab79c1b4f84246b96a714f01b2808bbec30c6d071213514ce5e4`; the wells table hash remains `f581a70cb09f09e9c74f8670c1d731b5a7eb131ac0a1756b77726a044bf5242d`.

For all three image families in both themes, producer image records plus post-PNG and post-PDF image appearance records compare exactly: **18/18 grouped comparisons**. This includes shape/dtype/data/mask, normalization class/vmin/vmax/clip/gamma, clim, alpha, origin, extent, interpolation and the complete categorical colormap LUT. Label identity, selected example order, preprocessing and raw image opacity remain unchanged. Recorded RNG states match the baseline before and after each creator.

The two-well cutoffs remain 6.5/5.5 for well_a and 6.0/5.0 for well_b. Default12 cutoffs remain used 3.5 and reference 2.0 for every well. The role distinction changes presentation only. Earlier coincident-cutoff suppression and selected-well cap/order passes remain retained; no new control or statistical result is inferred.

## Writer diagnostics and actual grounds

Every candidate post-writer PNG and PDF diagnostic is compliant with an empty problem list: **10/10 PNG and 10/10 PDF**. All two baseline default12 PNG and two PDF diagnostics fail early for unregistered semantics; these failures are preserved and are not a complete assessment of baseline geometry. Diagnostic success alone did not establish the resolved width or role distinction; those were checked independently above.

All **16/16 actual PNG corner receipts** match the requested theme at all four corners, including the four original producer writes: light `[255,255,255,255]`, dark `[24,24,24,255]`. Post-writer figure/axis ground and foreground observations agree with the visible theme. Actual source image values were not recolored to manufacture evidence. The dark reference contrast backing retains black reference cores with separate dashed/dotted patterns.

## Provenance and retained coverage

Job 1683 exited 0, with the default12 selector passing in its JUnit receipt. The serial resource receipt records scope `/user.slice/user-1000.slice/user@1000.service/app.slice/sched-home-1683-1274177.scope`, CPU 100000/100000, RAM 2147483648 bytes, swap 0, nice 19, one native thread and VRAM 0. Wall time was 2:27.84 and maximum RSS 1671848 KiB. No browser/graphics process was treated as a compute job.

The eight retained scoped passes from job 1679 remain valid. Failed default12 measurements from jobs 1679 and 1681 remain documented in `g2_submodules_followup.md` (local-only historical reference); neither failed run reached its render stage. Job 1683 supplies the passing measurement and new visual evidence.

Nine unaffected original families retain their job 1670 evidence: read_lines, score_heatmap, importance, permutation, legacy_radar, replication_sparse, invasion_sparse, guide_correlation and guide_effects. The earlier 72-PNG review and 360 common byte-identical artifacts remain intact. Four absent radar artist line/mask files in that earlier comparison are the documented polygon-to-bar representation change, with underlying source values exact. This review makes no claim of full training, inference, assay execution, real-data validation or unrestricted arbitrary-label layout coverage.

Frozen source and receipt SHA256:

| File | SHA256 |
|---|---|
| `spacr/submodules.py` | `197ed95d10bc44ee8b4766c93fdafca3415d071038f4b1f882abc8dc0764cc05` |
| `tests/test_graphstyle_g2_submodules.py` | `0189aa01ac2b522d658c8a7da22356128eedf70f629b4063da288926221a0de9` |
| [.codex-audit/g2_submodules_render.py](../scripts/g2_submodules_render.py) | `b9d5f63c3b5ad95dec6465ebc61dd748ec683476dc48e6ebe9c4478a3b9e93fc` |
| `spacr/figures/style.py` | `119defb238e524ee274fe7704ebefde0f13bb9a1cfb8c8635d77c6018dc2bc4a` |
| [submodules-default12-before/report.json](../evidence/submodules-default12-before/report.json) | `36a03f19147ab13074ece0a3bd03c33f2cefd188383d85fc2eea3c324752fca0` |
| [submodules-default12-after/report.json](../evidence/submodules-default12-after/report.json) | `af644523126f82c206a926d0e76320682f86807c4879b20c85bbbf33bb1e4a20` |
| [submodules-after-width-roles/report.json](../evidence/submodules-after-width-roles/report.json) | `c503df659c08fbe8d9b723294ae2f0dde43bc0c411badb2cf87695778315414a` |
| Original [submodules-before/report.json](../evidence/submodules-before/report.json) | `85c8e88636643d4a3f9a0d312652bf6fe977b0f344feaebb8c0ee65309652cae` |

The parent reports that shared style hash 119defb2 changes only the pickle method from the preceding 706cc90e dependency. The actual used hash is recorded above; no new shared-source review is implied here.
