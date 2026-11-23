> Curated publication copy. Original source: `.codex-audit/g2_utils_final_review.md`; original SHA256: `8f5fecfc4f2e2a28810976d672e542c196dc78afc2b8efe7540e0195411a2f9f`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# Utility final candidate evidence review

Job 1678 completed successfully in the root-controlled validation worktree. Reviewed evidence is [reports/graphstyle/g2/evidence/utils-after-v2](../evidence/utils-after-v2), produced by harness `8fbc359d81cc63e4e2b4b74f17537e81418859d27178e795c223dc42cef0c0b8` and shared style `706cc90e1ef35b9e8ceba77f4d8aad1703fddfd4b01e7566dd3d5d681b2c371d`. Utility source remains `8d5b2653339bd6b88ab875d2cd47a5d5359275e43bf3af6c3e9f317f2f3337c1`. This reviewer ran no Python, tests, renders or jobs and made no source changes.

**The confirmed embedding label-clipping and dark navy visibility defects are resolved in the actual final PNGs.** Both coordinate labels are fully visible inside the light and dark canvases. The navy cluster 5 points and all three hull segments can be distinguished from black through thin neutral backing; the supplied navy cores remain identifiable. The saved figure has a visible, complete external key. No general claim of complete style certification is made for the remaining documented boundaries below.

## Exact scientific and image comparison

`diff -u` of baseline versus final `scientific.json` returned exit 0. Both hashes are `20747f1f11bf373a7e3f8c5744fd9754315625fb10a63648912e6981dd30d495`. `diff -qr` of their entire `scientific/` directories also returned exit 0: all 238 stored NPY files are byte-identical. This preserves all recorded inputs, statistical/plot coordinates, selected thumbnails and their locations, MLR schema/order/effects, model and optimized-image values, scalar/RGB layers, processed mask outputs, protected image transforms, instance-mask palette and RNG states. These checks cover the fixed evidence fixtures, not unexercised biological workflows.

Every one of the 14 input crop PNGs has the same SHA256 as its baseline counterpart. Ten of the twenty final exported PNGs have identical SHA256 values to the previously individually reviewed candidate; those reviews are retained. The other ten exported PNGs were each opened and reviewed again. Complete per-file hashes and disposition for all 34 PNGs are in [g2_utils_v2_png_comparison.json](g2_utils_v2_png_comparison.json).

The final manifest exists and is valid JSON: SHA256 `8da5b6426ea06ab36633ca558f318e142d0f7dd244ce3ba1c8f8e3f8f35a3909`. It records 20 PNGs and 20 PDFs, their file hashes, PNG raster hashes and diagnostics. Every PNG's actual four corner pixels match its requested opaque light/dark ground. Figure and axes backgrounds remain unchanged through both normal writer calls. No facecolor override or repaint was used to obtain this evidence. PDFs were generated and recorded; this task's visual inspection was of PNGs.

## Changed PNGs: individual final inspection

Paths below are relative to the final candidate `renders/` directory.

| PNG | Final inspection |
| --- | --- |
| `light/embedding/figure0.png` | Full bottom x label and left y label are visible with surrounding margin. Square frame, identity/count key, UMAP expansion and sampling statement remain readable. Point/hull positions and source thumbnail arrays are unchanged. Thin dark backing helps low-contrast teal/blue outlines against white without substituting colors. |
| `dark/embedding/figure0.png` | Full coordinate labels are inside the canvas. Navy cluster 5's three points and triangular hull are visibly separated from black by thin neutral backing; teal and blue marks are also readable. Core identity mapping still matches the external key and gallery. This fixes the specifically reported unreadable-mark defect. |
| `light/process_masks/figure0.png` | Complete PCA x/y labels now have margin; all three overlapping-location marks and external feature/method/count explanation remain visible. Neutral mark boundaries are clear. No observations were dropped: exact stored coordinates are unchanged. |
| `dark/process_masks/figure0.png` | Same complete labels, visible blue/teal locations and external explanation on black. Thin backing separates point boundaries. Missing PCA units remain explicitly unresolved. |
| `light/SaliencyMapGenerator1/figure0.png` | Both scalar composites and full predictions/processing/range/gamma explanation are inside the padded canvas. Shared scalar display mapping and retained eight-slot layout remain intact. |
| `dark/SaliencyMapGenerator1/figure0.png` | Both scalar composites remain distinguishable and the full white external explanation is inside the black canvas; no new clipping. |
| `light/SaliencyMapGenerator3/figure0.png` | Both RGB composites and the complete RGB-preservation, predictions and processing explanation remain visible. Padding affects placement only; source RGB layers compare exactly. |
| `dark/SaliencyMapGenerator3/figure0.png` | Same intact RGB composites and complete external white caption on black. No source-channel recoloring or new clipping is visible. |
| `light/GradCAMGenerator1/figure0.png` | Both scalar composites, expanded Grad-CAM method, predictions and full range/gamma/calibration note are inside the canvas. |
| `dark/GradCAMGenerator1/figure0.png` | Same complete scalar composites and external explanation on black, with comfortable outer padding. |

## Unchanged PNGs: review retained by hash

For each light and dark theme, `metrics/figure0.png`, `mlr/figure0.png`, `mlr/figure1.png`, `gallery/figure0.png` and `class_exemplar/figure0.png` are byte-identical to the earlier individually reviewed candidate. Their prior per-image assessments remain valid. In particular, MLR diagnostic captions are intact, gallery keys remain visible and the common range preserves visible intensity differences, and the class exemplar retains its original RGB raster with a truthful synthetic-image caption.

The two G1 volcano PNGs are also identical to the original baseline through the previously established baseline/candidate comparison. Their rectangular geometry, above-panel title, in-data gene labels and missing visible dark threshold remain G1-owned limitations; utility changes did not repair or regress them.

## Diagnostics and precise limits

Fourteen of twenty final figures have empty diagnostic problem lists. Six retain explicit findings:

* Two `process_masks` PCA figures report missing quantity units. The input combines object area/intensity/perimeter/eccentricity without a supplied shared coordinate unit. This review does not invent dimensionless units to remove those diagnostics.
* Two G1 volcano figures report absent chart/reference/quantity/legend registration. Their dark threshold visibility limitation remains documented above.
* Two gallery figures report clipped/overlapping `−5`/`10` tick text. The actual PNGs contain no such ticks and match the reviewed candidate bytes. Source inspection identifies a validator boundary: `plot_grid` calls `ax_img.axis('off')` for every image inset; G1 `figure_problems` collects suppressed axis text only from top-level `figure.axes`, then scans text recursively with `figure.findobj(Text)`. Hidden child-axis tick objects therefore enter the text-bounds/overlap check despite not being drawn. These diagnostics are preserved verbatim in the manifest and comparison JSON; they are not rewritten as a clean certification.

The new identity backing is recreated by the producer's additive embedding styling. This evidence establishes its appearance in the current creator/normal-writer PNGs; **it does not establish new serialization of the effect by G1 appearance capture/replay**. The G1 serialization contract was not extended. Root reports 31 relevant shared/embedding checks passed in job 1677; this reviewer did not execute them.

Large blank space in the attribution grids is the retained eight-column contract with only two fixture images, not missing observations. Tiny embedding thumbnails reflect the explicit fixture zoom and preserve their exact source/selection. Neither behavior is silently represented as an idealized publication layout.
