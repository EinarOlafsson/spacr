> Curated publication copy. Original source: `.codex-audit/g2_submodules_visual_review.md`; original SHA256: `bc9e2250a25eca10e1f1b29b426574326712c5aa0d94d9a09fc14282722ba5b9`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# G2 submodules paired visual review — job 1670

Review status: scientific payload preservation passes for the frozen fixtures; presentation still requires the bounded corrections below. Empty finalizer diagnostics are not an all-clear.

Reviewed on 2026-11-23, read-only, from the parent-run paired evidence. This reviewer ran no Python, imports, tests, scientific jobs, renders or model work and made no application edits during this review. Actual PNGs were opened separately; ambiguous multi-image display strips were reopened individually. PDF files were not rasterized or visually inspected; their post-writer diagnostics, image appearance records and canvas dimensions were inspected.

## Frozen provenance and coverage

- Before source: untouched G1 checkpoint `9510143823dcc1b984ecc522058a1e3f15e985d8`, `/mnt/wd4tb/worktrees/codex-graphstyle-g2-before`.
- Candidate `spacr/submodules.py` SHA256: `4866a47eb4b76b73a030c7349bd38221087fdec50d3dc742cf002f5218cbcf6a`.
- Harness SHA256: `0e4c4bbb023d99f34e2e0c15bb730d6e75b126111f186cb9b3d45cc96264a2a8`.
- Before report SHA256: `85c8e88636643d4a3f9a0d312652bf6fe977b0f344feaebb8c0ee65309652cae`.
- After report SHA256: `606849c2bfaae2bb8a6fea41f036d6a4fe2d7683ffa25f905de32e4cf5f96919`.

Evidence roots, relative to the implementation tree:

- [reports/graphstyle/g2/evidence/submodules-before](../evidence/submodules-before)
- [reports/graphstyle/g2/evidence/submodules-after](../evidence/submodules-after)

Each has light/dark directories and 13 cases, each with `figure.png`, `figure.pdf` and source artifacts. All **52 figure PNGs** were individually viewed. All **20 additional producer PNGs** were individually viewed: the two phases × two themes × the five paths listed below. Total actual PNG coverage: **72/72**. No screenshot collage substitutes for this inspection.

The harness uses the public theme preference route outside `pipeline_figures`, actual production drawing owners or exact nested production source, and normal screen-mode `save_figure`. The nested drawing owners are exercised without executing their training/inference/assay parents. These are deterministic synthetic fixtures, not a claim of a full pipeline or GPU run.

## Scientific payload comparison

For every corresponding theme/case, the complete registered `source`, `tables`, `rng_before` and `rng_after` report objects compare exactly: **26/26 pairs** for each field. This includes source hashes, shape/dtype/mask hashes, column/index order, selected image arrays and the saved input/statistical tables. The claim concerns the frozen fixture outputs; it does not infer unexecuted production results.

A separate direct `cmp` of every baseline CSV/NPY artifact against the same relative candidate path found **360 byte-identical files, zero changed common files, and four absent files**. The absent files are the light/dark `legacy_radar/axis_0_line_0.npy` and `axis_0_line_0_mask.npy` artifacts: the polygon line was deliberately replaced by bars. The underlying ordered source values remain exact. These absent artist files are a representation change, not missing scientific values.

For `training_images`, `test_images` and `apply_images` in both themes, all recorded producer image-array records and post-PNG/post-PDF image-appearance records compare exactly before/after: normalization class, vmin/vmax, clip, gamma, clim, alpha, origin, extent, interpolation and full colormap LUT hash including categorical label colors. The original image opacity remains unchanged; the graph-mark opacity policy was not applied to raw image planes.

The threshold fixture explicitly contains different numerical roles:

| Well | Used cutoff | Reference cutoff | Existing source |
|---|---:|---:|---|
| well_a | 6.5 | 5.5 | control |
| well_b | 6.0 | 5.0 | fixed |

These are the exact saved `wells.csv` values; their input CSV SHA256 is `7ee3244cd8d13b589f11c2eb319380bdddf6919b793d7c70c05f59c5e441c773`. Histogram finite signals, bin results and reference coordinates are unchanged. The displayed wells retain the original selected order, despite a different order in the input well table.

## Actual saved themes and diagnostics

All **26 candidate figure PNGs** have the requested ground at all four actual saved pixel corners. All **10 candidate producer PNGs** do too: light `[255,255,255,255]`, dark `[24,24,24,255]`. The comparison uses the decoded PNG pixel receipt, not an inference from Figure facecolor.

All baseline normal `figure.png` corners also match their requested ground. However, all **five baseline dark producer PNGs** have white outer corners and pale text on the white exterior; this is plainly visible in each file. The candidate creation/export boundary correction fixes that concrete defect. Baseline light producer corners match.

Every candidate PNG and PDF post-writer diagnostics record says `compliant: true`, with an empty problem list: **26/26 PNG and 26/26 PDF**. Baseline PNG/PDF diagnostics stop at the missing registered-semantics check, **0/26 compliant** in each format. That early baseline failure cannot be interpreted as a complete evaluation of its geometry or visual semantics. The candidate checker does not detect the two remaining presentation defects described next.

## Remaining presentation corrections

### 1. Used/reference cutoffs are visually indistinguishable

Both light and dark candidate threshold PNGs draw different cutoffs using the same black dashed core and the same dashed external-key glyph. The dark contrast backing improves visibility but does not identify which cutoff is used. The baseline blue solid used cutoff and dashed reference were distinguishable. Labels naming the roles in the key do not resolve two identical glyphs.

Owner route: `spacr/submodules.py:6925` and `:6928` call shared `reference_line`; `spacr/figures/style.py:851` fixes every reference dash to `(0,(4,3))`. `external_legend` at `:4254` independently constructs all reference glyphs with the same dash. There is no supported role/dash argument in the frozen shared API. Changing only the artist locally would leave a misleading key.

Bounded correction: shared-owner additive reference-role/dash metadata, with the same style used by the actual reference artist, visible key and replay representation. Use two clear black reference patterns (for example used dashed, reference dotted), retaining the shared linewidth/opacity/contrast policy. The submodule supplies the existing used/reference roles, without new controls or label inference. Retain exact thresholds, histogram arrays/bins, lookup order, selected wells, source tables and statistical captions. Confirm both different-cutoff and coincident-cutoff cases; a coincident reference must continue to avoid a redundant line.

### 2. Final canvases exceed the paper width

The graph skill's later maintainer rule at `/home/olafsson/.claude/skills/spacr-graphs/SKILL.md:258` specifies **170 mm width and at most 275 mm height**, true 8 pt text. It supersedes the older 210 mm A4-width rule at line 197. Neither rule exempts image grids. The latest rule was independently confirmed by the parent. No figure should be shrunk after composition to hide excess width.

Both themes have identical measured final dimensions below. PNG and PDF receipts agree:

| Case | Final width × height, mm | Width issue |
|---|---:|---|
| training_images | 217.962 × 80.000 | Exceeds both 170 and 210 mm |
| test_images | 227.930 × 43.729 | Exceeds both 170 and 210 mm |
| apply_images | 223.062 × 40.000 | Exceeds both 170 and 210 mm |
| thresholds | 190.486 × 64.512 | Exceeds 170 mm |

The other nine case families fit 170 mm. There is no reported height excess.

The direct test/apply producer PNGs use the real producer's tight crop and therefore differ from the whole-canvas `figure.png`: test is 2473 × 516 pixels, apply 2417 × 419 pixels. At the normal 300 dpi these are approximately 209.381 and 204.639 mm wide, respectively: still over 170 mm even though under 210 mm. These actual crop widths must not be replaced by the uncropped canvas width in a receipt.

The image constructors begin with a 170 mm canvas, then `external_legend(..., pack=False)` appends a caption on the right. The training case also retains a large horizontal gap between its two image columns. The threshold panels first pack to 170 mm then append the key. In the shared implementation, `maximum_width` is a packing target, and the final right-key expansion is unbounded. A clean diagnostics result does not enforce final paper width.

Minimal shared correction: allow the visible key below the panels with a bounded final width, preserving each existing image plane's array, label identity, normalization, aspect and physical panel dimensions. Compact excess empty inter-column space; when the measured content itself will not fit, reflow the same ordered image/label pairs or diagnostic planes into more rows. Recomposition moves existing panels; it must not crop source images, change selected examples, resample their data or scale the composed figure/text. Keep training image/label pairing and row order explicit. A shared measured-layout extension is preferable to competing submodule geometry logic. The three image and threshold paths need fresh both-theme normal-write evidence after the shared fix.

### Caption cross-check — no change required

An initial review note confused the training grayscale image row with the label row. The frozen source at `spacr/submodules.py:1039` explicitly uses `interpolation='nearest'` for labels. The saved appearance records independently agree: axes0/1 grayscale use `auto`; axes2/3 labels use `nearest`. The existing “nearest-neighbour categorical display” label caption is accurate. The initial minor caption finding is withdrawn; no caption or interpolation change is required.

## Individual figure inspection ledger

Every row below covers separate inspection of **before-light, before-dark, after-light and after-dark** `figure.png` files. “No additional defect” is restricted to the shown fixture and does not waive the issues above.

| Case | Visible before/after assessment |
|---|---|
| training_images | Raw grayscale images and paired categorical labels retain content/aspect/colors. Candidate text and ground are readable in both themes. Final width and excessive inter-column whitespace require correction. Training label caption matches the actual nearest interpolation. |
| test_images | All five original diagnostic planes are present in order. Image/true-mask/predicted-mask/probability/flow titles and complete caption are visible. Final width requires correction. |
| apply_images | All four original diagnostic planes are present in order; labels retain colors and pixel structure. Complete key visible. Final width requires correction. |
| read_lines | Same two ordered curves and numerical coordinates. Candidate has square continuous axes, readable measured-series identity key and external caption. Both dark curves remain visible, though naturally darker at the mandated graph-mark opacity. No clipping observed. |
| score_heatmap | Same ordered 3×2 values and missing cell. Candidate uses square cells, a visible missing marker and labeled sequential range 0.1–0.9. Colorbar chrome remains readable in both themes. No clipping observed. |
| importance | Original scalar feature order and values retained in vertical bars; y label, category names and estimator caption readable. No additional defect observed. |
| permutation | Original scalar feature order and values retained in vertical bars; quantity and estimator caption readable. Both themes reopened individually after an ambiguous viewer strip. No additional defect observed. |
| legacy_radar | Original polygon becomes ordered scalar bars, without implying a distribution or inventing uncertainty. Registered original values are exact. Candidate labels and caption readable. |
| replication_sparse | Baseline right-hand key is clipped; baseline dark text is poorly visible. Candidate thin single-condition stacked bar retains fractions, has a fully visible category key and explanatory text. No additional clipping observed. |
| invasion_sparse | Baseline category key is clipped. Candidate retains exact stack and existing n=3 annotation content in its key; all categories and notes are visible. No fabricated statistics appear for the sparse condition. |
| thresholds | Original finite histograms and cutoff positions retained. Candidate well/sample/bimodality notes visible, but role glyphs are indistinguishable and final width exceeds170 mm. Both defects are concrete. |
| guide_correlation | Same 4×4 ordered Pearson matrix. Candidate square cells, fixed −1/0/+1 scale and readable quantity/units; complete explanatory caption. No clipping observed. |
| guide_effects | Same ordered anchor/propagated numerical values. Candidate visible key identifies the already supplied anchors separately from propagated scores, with no added CI. Axes/caption readable. No clipping observed. |

## Additional producer PNG inspection ledger

Each exact relative path below was separately viewed under **before/light, before/dark, after/light and after/dark** (20 files total):

| Relative path | Actual export assessment |
|---|---|
| `test_images/cellpose_result_000.png` | All five planes intact. Baseline dark titles nearly vanish against the erroneously white outer ground. Candidate dark exterior/titles correct; candidate width still exceeds170 mm after tight crop. |
| `apply_images/cellpose_result_000.png` | All four planes intact. Same baseline dark white-exterior defect and candidate correction; candidate tight-crop width still exceeds170 mm. |
| `read_lines/line_direct.png` | Baseline dark exterior leaves title/axis/key text pale on white. Candidate exterior and text are correct, both curves present, no clipping. |
| `guide_correlation/correlation_matrix_heatmap.png` | Baseline dark exterior leaves colorbar/axis text pale on white. Candidate full scale and text readable against the correct ground; same ordered matrix. |
| `guide_effects/effect_sizes_barplot.png` | Baseline dark exterior leaves axes/title text pale on white. Candidate source-role key and text readable with correct ground; same ordered scores. |

## Follow-up acceptance boundary

Keep this job1670 review and its frozen evidence; later rerenders must use new output directories and new hashes. Do not relabel the current files as corrected. The parent owns execution. Once shared APIs are available, submodule-only caller changes and focused tests may address the two findings together. Required evidence is unchanged scientific source/statistical/input bytes, unchanged image appearance/LUT/RNG, actual distinct threshold strokes and matching keys in both themes, final saved width ≤170 mm at unchanged true text size, and correct actual producer PNG corners. Existing 1670 diagnostics alone cannot satisfy that acceptance boundary.
