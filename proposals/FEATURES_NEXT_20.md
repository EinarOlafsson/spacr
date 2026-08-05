# spaCR — 20 more features, from Napari / JMP / professional image analysis

Proposed 2026-08-02. **Distinct from the 20 in `AUDIT_2026-08-01.md` §5** — none of these repeats
one of those. Each was checked against the code before being proposed; the absence claims below are
greps, not assumptions.

Verified absent at `nightly`: illumination correction, PCA, pivot/tabulate, linked brushing,
dose-response, lineage, robust outlier detection, orthogonal views, undo/redo.
Verified PRESENT and therefore *not* proposed: colocalisation measurement (per-object Pearson +
Manders already land in `measure.py:1262`).

---

## Interactive analysis — the JMP half

### ★1. Linked brushing across every open view — L
JMP's single best idea. Lasso a cluster in the UMAP and the same cells light up in the plate
heatmap, the measurement table and the crop grid at once; brush a plate region and see where those
wells sit in feature space. spaCR already ships all four views (`umap`, `plate_view`, `db_browser`,
`annotate`) and **none of them talk to each other**. This is what turns spaCR from a report
generator into an exploration tool, and it is the substrate for #2, #3 and #6.

### ★2. Local data filter — M
A dockable panel of live sliders and checkboxes — cell count, plate, gene, intensity range,
class — that subsets every open plot simultaneously. Answers "does this hit survive if I drop the
low-n wells?" in a second rather than a re-run. JMP's Local Data Filter is the feature its users
reach for most.

### 3. Graph Builder — L
Drag columns onto x / y / colour / facet and get a chart. Replaces "which of the ~40 `plot_*`
functions do I need, and what does it expect?" with a direct manipulation surface.

### 4. Prediction profiler — M/L
Drag a feature slider and watch the predicted phenotype move, with per-feature sensitivity and a
desirability optimiser. Makes a trained classifier interrogable rather than a black box — and it
reuses the attribution machinery already in `attribution.py`.

### 5. Multivariate platform — M
Correlation matrix, PCA with a loadings biplot, hierarchical clustering heatmap over features.
spaCR measures hundreds of features per object and has **no PCA anywhere**; this is the standard
first look at a feature table and its absence is conspicuous.

### 6. Tabulate / pivot builder — M
Interactive pivot of measurements by gene × condition × plate with a chosen aggregation, exportable
to CSV. The step everyone currently does in Excel, which is also where provenance goes to die.

### 7. Column formula editor — M
Define derived columns (`ratio = cell_channel_1_mean / cell_channel_2_mean`) that persist with the
project and re-apply to later runs. Today this means editing source. Pairs with `custom_features.py`,
which already has the registration hook but no user-facing surface.

### 8. Robust outlier detection — S/M
Mahalanobis distance or isolation forest over the well-level table to flag suspect wells *before*
they become hits. Complements `plate_qc`, which looks at plate geometry rather than at the feature
distribution.

### 9. Control charts across a campaign — M
SPC-style limits on focus score, cell count and staining intensity per plate over weeks. Catches
instrument drift that any per-plate QC is blind to by construction, because it only ever sees one
plate.

### 10. Dose–response / EC50 fitting — M
Four-parameter logistic per gene or compound with confidence bands and a hit table. Absent entirely,
and standard for any screen with a titration series.

---

## Viewer and layers — the Napari half

### ★11. Layer model viewer — L
Image / labels / points / shapes / tracks as independent layers, each with its own opacity,
colormap and blend mode. This is the substrate #12–#16 all need, so it is worth building once and
properly rather than bolting each on separately.

### 12. Interactive label brush — M/L
Paint, erase, merge and split masks by hand and write back to the mask stack. `make_masks` corrects
segmentation *by re-running Cellpose with different settings*; sometimes the right fix is to split
one touching pair and move on.

### 13. Points layer for manual counting — M
Click to place markers, export counts. The ground truth needed to prove segmentation is right, and
it feeds the existing `agreement` module directly.

### 14. ROI / shapes layer honoured by Measure — M
Draw polygons to exclude debris, bubbles, or out-of-focus regions per field, and have `measure`
respect them. Today a bad region silently contaminates every downstream number with no way to
exclude it short of dropping the whole field.

### 15. Orthogonal views + dimension sliders — L
Synchronised XY / XZ / YZ with a shared crosshair, plus z / t / channel / field scrubbing. Pairs
directly with the 3-D end-to-end work already on the other list, and makes `zstack.py`'s existing
4-D API visible for the first time.

### 16. Synchronised comparison grid — M
The same field across conditions, timepoints or two segmentation models, with locked zoom and pan.
`model_compare` does this for exactly two models, statically; Napari's grid mode generalises it.

---

## Image processing and workflow

### ★17. Illumination correction / flat-field — M
Estimate per-channel illumination from the plate itself and correct it before segmentation. A
CellProfiler staple, **completely absent**, and it is the root cause of the edge effects
`plate_view`'s edge-effect detector exists to find. Fixing the cause beats detecting the symptom,
and it is the only item on this list that changes the numbers rather than how they are viewed.

### 18. Interactive classifier training loop — L
CellProfiler Analyst's model: click a handful of cells, retrain in seconds, see updated predictions
across the plate, repeat. spaCR has `annotate` and `classify` as separate batch steps; nearly all
the value is in closing the loop between them.

### 19. Macro recorder to Python script — M
Record GUI actions and emit a runnable `spacr-run` script. Fiji's recorder is a large part of why
people trust Fiji, and it converts any exploratory GUI session into something reproducible — which
`run_journal` already half-supports from the other end.

### 20. Object relationship / lineage view — M
Cell → nucleus → pathogen as an interactive tree, and division lineages for timelapse. The
parent-child links and `cell_id` are already stored; nothing displays them, so the hierarchy the
whole measurement model rests on is invisible.

---

## If only four

**1** (linked brushing), **11** (layer model), **17** (illumination correction), **2** (data filter).

1 and 11 are platform bets that make many of the others cheap afterwards. 17 is the only one that
changes the science rather than the ergonomics. 2 is the highest ratio of daily usefulness to
effort on the list.
