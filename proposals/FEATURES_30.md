# 30 new analysis features for spaCR

44 proposals were generated across five lenses (single-cell database, image formation,
hit calling, sequencing, Toxoplasma biology) and scored by three independent judges on
usefulness, feasibility-in-this-codebase and novelty-against-what-ships. 14 were cut and
two pairs were merged. What follows is the 30 that survived, all verified against
`/mnt/firecuda2/Claude/repo/spacr` @ `spacr-nightly`. Every claim about a missing
capability below was checked by reading or grepping the code, not inferred.

Two merges worth naming up front: **`object_qc`** (Part 1, #3) absorbs both the
database-side per-object QC gate and the Cellpose-flow confidence proposal — one table,
two tiers. **`vacuole_shape`** (Part 1, #8) absorbs the separately-proposed materialised
`vacuole` table, which falls out of it for free. The cut list and a short list of bugs
found while checking are at the end.

---

# Part 1: Built on spaCR's own data (10)

All ten read `measurements.db`, `png_list`, the annotation columns, the classification
scores or the object-parent links. None needs a pixel re-read, a new acquisition or a
FASTQ file; most run retroactively on databases that already exist on disk.

### 1. `bystander` — infected vs. uninfected host cells, paired within field
A new reader that **left**-joins `pathogen` onto `cell`, keeping the host cells with zero
parasites, then computes for any measurement column the within-field (`prcf`) paired
difference between infected and uninfected hosts, graded by burden (0 / 1 / 2 / 4 / 8+).
Ships with `translocation_index` = `nucleus_channel_C_mean_intensity` /
`cytoplasm_channel_C_mean_intensity` as the headline readout, and yields the per-cell
infection rate as a by-product. This is *the* discriminator between cell-autonomous
effector action (GRA16, GRA24, TgIST, ROP16 rewiring the nucleus of the cell they are
inside) and a paracrine or IFN-γ response that also hits neighbours — and it is currently
impossible, because the uninfected hosts are deleted before any analysis sees them.
Within-field pairing is not optional on a Yokogawa plate: illumination and local
confluence both vary across a single field.
**Builds on:** `io._read_and_merge_data` — verified at `io.py:3041` the pathogen merge is
`merged_df.merge(pathogens_g_df, left_index=True, right_index=True)`, pandas default
`how='inner'`, so every uninfected host cell is dropped; `pathogen_prcfo_count`
(`io.py:3024`); `utils._group_by_well`; `utils.annotate_conditions`. Also verified
`utils._calculate_recruitment` (`utils.py:2592`) builds pathogen/cell, pathogen/cytoplasm
and pathogen/nucleus ratios but **no nucleus/cytoplasm ratio** — the one this needs.
**Effort:** M. **Risk:** it must be a *new* reader. Changing the join inside
`_read_and_merge_data` would silently alter row counts for `analyze_recruitment`,
`analyze_endodyogeny`, `analyze_percent_positive`, every ML feature table and every UMAP.

### 2. `burden_adjust` — burden-residualised phenotype scores
Fit, per plate and on control wells only, the relationship `phenotype ~ f(parasites per
cell)` (isotonic or a low-order spline), then write a per-well CSV carrying both the raw
well mean and the burden-residualised mean as two candidate `dependent_variable` columns,
plus the fraction of each mutant well's cells whose burden falls outside the control
burden range. In a pooled host-factor screen nearly every host readout scales with
parasites per cell, so a gene that only reduces invasion drops mean burden and then *any*
burden-correlated readout moves with it — the gene scores as a "recruitment hit". This is
the dominant false-positive class in this kind of screen and nothing in the pipeline
controls for it today.
**Builds on:** `pathogen_prcfo_count` (`io.py:3024`), `utils._group_by_well`
(`utils.py:2635`), and `ml.perform_regression`'s CSV contract (`ml.py:827` reads an
arbitrary CSV and takes `dependent_variable` by name), so the output plugs in with no
change to the regression code. `sklearn.isotonic`, already a dependency.
**Effort:** M. **Risk:** if a mutant shifts burden outside the control range the
correction extrapolates and can manufacture an effect. The extrapolation fraction has to
be reported per well and the well flagged, never silently corrected.

### 3. `object_qc` — one per-object verdict on what counts as a usable cell (two tiers)
A single `object_qc` table keyed on `prcfo` with a per-object verdict and a `qc_flags`
string, fed by two independent tiers. **Tier 1 (retroactive, no reprocessing)** is
computed from columns already in every database: `<obj>_channel_<c>_blur`, clipping from
`max_intensity` and `frac_high90`, edge-touching from the weighted centroid against field
bounds, area/solidity outliers against the plate distribution, and a missing `cell_id`
parent link. **Tier 2 (mask time)** keeps what Cellpose already computes and spaCR throws
away — mean and 10th-percentile `cellprob` inside each mask, the flow-consistency error,
and a boundary-gradient check — written to a small `seg_confidence` table keyed on
`(file_name, object_type, object_label)` and joined at read time. Then a confound audit:
recompute the well ranking with and without flagged objects and report which hits move.
This answers the referee's first question ("is your hit list real, or is it focus?") at
object level, and it stops a CNN learning "welded pair" as a phenotype.
**Builds on:** `<obj>_channel_<c>_blur` (`measure.py:774`), `frac_high90`
(`measure.py:951`, field-relative by construction) — grep confirms **both are read by
nothing but `feature_dict`**; `cell_id`→NaN for no parent (`measure.py:826`); the units
stamp `MEASUREMENT_STAMP_COLUMNS` (`measure.py:90`) so thresholds are not applied to
volumes as if they were areas; `spacr_cellpose.parse_cellpose4_output`, and
`object.py:1812` / `1873` where `masks, flows, _, _, _ = parse_cellpose4_output(output)`
binds flows and never uses them (at `818` / `1219` flows do feed the plotting path — but
cellprob is nowhere reduced to a per-object number). Distinct from `seg_qc`, which scores
whole *fields* and never marks an individual object.
**Effort:** M (tier 1) + M (tier 2). **Risk:** absolute blur and cellprob thresholds do
not transfer between magnifications or channels — the gate must be a per-plate,
per-channel percentile, and it must be advisory (a column) rather than deleting rows.

### 4. `label_audit` — find the mislabelled crops and rank them for review
Fit the classifier in k-fold over the annotated crops only, collect out-of-fold
probabilities, and apply confident-learning class-conditional thresholds (~80 lines of
numpy — no `cleanlab` dependency) to rank crops whose human label the model confidently
contradicts. Write `label_suspect` and `label_review_rank` into `png_list` by the same
`ALTER TABLE` route the Annotate app already uses, and hand Annotate an ordering it can
open directly. Two thousand hand-clicked labels always contain 3–8 % mistakes, and every
one is a permanent ceiling on every model trained afterwards; fixing the worst hundred is
the highest-return twenty minutes in a screen. It also produces the honest human error
rate a methods section needs.
**Builds on:** annotation columns in `png_list` (`gui_elements._ensure_annotation_column`
`gui_elements.py:4230`, `qt/annotate_engine.ensure_annotation_column:328`),
`agreement._METADATA_COLUMNS` to know which columns are annotations, the `deep_spacr`
training loop for the folds, `active_learning.as_probabilities:327` /
`rank_by_uncertainty:565` for the probability plumbing. Distinct from `agreement`
(annotator vs annotator, κ) and from `active_learning`, which ranks *unlabelled* crops —
this ranks *labelled* ones and disputes them.
**Effort:** M. **Risk:** a systematically biased model will "correct" labels toward its
own bias. Fold assignment must be grouped by well (never split a well across folds) and
the human's second click always wins.

### 5. `feature_rank` — which measurements actually separate my classes
Given any grouping (an annotation column, `cv_predictions`, a `columnID` condition, a UMAP
cluster), score every numeric column by AUROC, Cliff's delta and KS distance with
bootstrap CIs; cluster columns at |r| > 0.9 so forty redundant percentiles collapse to one
representative; roll up by compartment / channel / family; and add a per-plate
reproducibility column — is this feature's separation the same on every plate, or a
one-plate artefact? Answers "my classifier works, but what is it actually measuring?" in
words that go in a figure legend (size, brightness, texture, colocalisation), and the
per-plate column immediately kills features that only separate because one plate was
imaged differently.
**Builds on:** the object tables; `sp_stats.perform_statistical_tests:100`,
`choose_p_adjust_method:10`; `utils.remove_highly_correlated_columns:6362`;
`feature_dict.describe_columns:1969` and `parse_column` for the family roll-up. Distinct
from `ml.ml_analysis:2127`, which fits XGBoost on control wells and reports *model
importance* — importance is winner-take-all among correlated columns, has no CI, works
only against control `columnID`s and gives no per-plate stability.
**Effort:** M. **Risk:** with a million cells every p-value is zero. The report must lead
with effect size and CI and treat p as decoration, or it manufactures false confidence.

### 6. `penetrance` — fraction of cells affected, separated from how far they moved
For each well, compare the full single-cell score distribution to the pooled
negative-control distribution instead of comparing means: EMD and KS distance, the
fraction of cells above the NC 95th percentile, and quantile shifts, all with well-level
bootstrap CIs; flag wells whose mean is unchanged but whose distribution is not. Pooled
CRISPR with imperfect editing produces exactly this — a subpopulation of true knockouts
inside a well of unedited cells. A gene where 15 % of cells are strongly affected and one
where 100 % are weakly affected have the same well mean and completely different biology,
and the first is currently invisible because `ml.process_scores` reduces the well to a
mean or median before anything statistical happens.
**Builds on:** per-object `pred` / `cv_predictions` in `png_list` (written by
`deep_spacr.merge_predictions_into_db:1919`); `ml.process_scores:1774` for the well keying
and `min_cell_count`; `submodules._bimodality_coefficient:3138`; `scipy.stats` for
EMD/KS. **Effort:** M. **Risk:** two-component mixture fits on a few hundred cells are
unstable — ship the non-parametric shift as the product and leave the mixture layer to
`cell_level_screen` (#22), which is built for it.

### 7. `extracellular` — the parasites every module currently deletes
Read the `pathogen` table raw and keep exactly the rows everything else discards —
`cell_id` NaN or 0, the parasites overlapping no host cell. Report per field and per well:
extracellular count, extracellular:intracellular ratio, nearest-neighbour distance
against a Poisson expectation (clustering), and the size/eccentricity distribution. On
plates that also ran `analyze_invasion`, cross-tabulate the mask-based "no host cell" call
against that module's intensity-threshold "extracellular" call and report the disagreement
rate per well. An invasion screen needs a denominator — how many parasites were available
to invade — and there is none today, so a gene that reduces parasite yield looks like a
gene that reduces invasion. Clustering also catches parasites that egressed but failed to
disperse, which is a motility phenotype.
**Builds on:** three verified discard sites — `io.py:3020`
(`pathogens.dropna(subset=['cell_id'])`), `measure.py:826` (parent `0` → NaN), and
`analyze_replication`'s `require_host_cell` filter (`submodules.py:2748`); cross-check
against `submodules._invasion_classify:3439`; `pathogen_area` / `_eccentricity` /
`_solidity`. **Effort:** S–M. **Risk:** extracellular tachyzoites are small and dim and
frequently fall below the pathogen segmentation's `min_size`, so the count is a lower
bound. The module has to say so and report how many objects the debris filter rejected
rather than quoting a bare number.

### 8. `vacuole_shape` — rosette architecture and division synchrony
Promote the vacuole to a first-class table (one row per vacuole: `cell_id`, parasite
count, replication bucket, summed and mean parasite area, per-channel intensity rollups)
and then measure its *arrangement* rather than its count: within-vacuole CV of
`pathogen_area` (division synchrony), packing density (Σ parasite area / convex-hull area
of the parasite centroids), CV of nearest-neighbour distance, angular dispersion about the
vacuole centroid, and mean major/minor axis ratio. Emits a per-well table with the same
identity columns `_replication_well_distribution` already writes, so it drops straight
into `perform_regression`. A mutant can hit the right parasites-per-vacuole bucket and
still be badly broken: loss of division synchrony and loss of rosette order are the
readouts for daughter-budding, basal-complex and IMC mutants, and the shipped bucket
histogram scores a tidy 4 and a chaotic 4 identically. The materialised vacuole table also
makes "does recruitment differ between 2-parasite and 8-parasite vacuoles?" a join instead
of a scrape.
**Builds on:** `submodules._assign_vacuole_ids:2124`, `_derive_vacuole_link_distance:2092`,
`_find_centroid_columns:2061`, `qc_flag_non_power_of_two` (`submodules.py:2310`),
`measure._summarize_organelles_per_parent:588` as the rollup template,
`utils._merge_and_save_to_database:1889` as the writer, `scipy.spatial.ConvexHull`.
**Prerequisite (verified):** `pathogen_centroid-0/-1` **does not exist** —
`measure.MORPHOLOGICAL_PROPS` (`measure.py:60`) contains no `'centroid'`, so only
per-channel *intensity-weighted* centroids are written. Add `'centroid'` (valid in 2-D and
3-D, costs nothing) before building the geometry, or it runs on the wrong point.
**Effort:** M. **Risk:** if the pathogen mask fuses a rosette into one blob, every
geometry metric silently reports a single perfect circle. Gate hard on
`is_power_of_two` / `n_parasites >= 2` and fail loud when the non-power-of-two fraction is
above threshold. `_assign_vacuole_ids` is centroid-proximity-based and will merge adjacent
single-parasite vacuoles in a dense field, so the table needs a confidence column.

### 9. `cell_finder` — query-by-example retrieval over the whole screen
Right-click any crop (Annotate, db_browser, a montage) and get the *k* most similar
objects across the plate or across several plates as a ranked gallery. Similarity is
cosine or Mahalanobis distance in a standardised subspace of the object table, with the
feature set chosen by family (morphology only / intensity only / one channel /
everything), fitted with `sklearn.neighbors.NearestNeighbors` on the z-scored matrix and
cached as an `.npz` index next to `measurements.db`. "I just saw one weird vacuole — show
me every other cell in the screen that looks like it, and which wells they came from." The
only route to that answer today is eyeballing a UMAP; this turns an interesting single
cell into a candidate phenotype class, and into an annotation seed set, in a minute.
**Builds on:** the object tables and `prcfo`; `png_list.png_path`;
`io._read_and_join_tables:2158`; `utils.preprocess_data:6297` for scaling;
`feature_dict.parse_column:1778` for family filtering; `plot._plot_images_on_grid:1473`
for the gallery. Explicitly **not** `ml._calculate_similarity:2538`, which is a *well*'s
distance from a control centroid, nor `core.generate_image_umap`, which is a global
unsupervised layout with no query. **Effort:** M. **Risk:** distance over 150–350 raw
columns is dominated by whichever family has the most columns — intensity percentiles
swamp morphology. Needs per-family whitening or PCA-then-kNN plus a per-plate z-score, or
every neighbour comes back from the same plate.

### 10. `spatial` — local cell density as a first-class covariate
Per field, build a KD-tree over cell centroids and write per-cell covariates back to the
`cell` table: neighbour count within R, distance to the k-th nearest neighbour, a
field-border flag, and the infection rate among that cell's neighbours. Deliberately
descoped: the Ripley's-K / permutation-clustering layer is dropped — at one Yokogawa field
it is statistically thin and gets run once. The density columns are the point. Local
confluence is a massive confound in every pooled screen: crowded cells are smaller,
flatter, dimmer and less permissive, so a gRNA that happens to land in a denser corner of
a well reads as a phenotype. There is no density covariate anywhere in the database today,
which means `feature_rank`, `object_qc`, `concordance` and `burden_adjust` have nothing to
condition on.
**Builds on:** `submodules._find_centroid_columns:2061`, which already encodes the
centroid fallback chain; `pathogen_prcfo_count` (`io.py:3024`) for free infection labels;
`prcf` as the field key; `scipy.spatial.cKDTree`. Same one-line `'centroid'` prerequisite
as #8. **Effort:** M. **Risk:** the available centroids are intensity-weighted per channel
— a cell's DAPI-weighted centroid is really its nucleus. Fine for density, wrong for
anything contact-based. Field-edge truncation biases every neighbour count downward, so
the border flag has to be a real exclusion, not a column nobody reads.

---

# Part 2: Everything else (20)

## Image formation and measurement QC (4)

### 11. `illum` — retrospective flat-field / dark-field correction per plate per channel
Estimate one gain surface and one offset surface per channel from the plate's own
`stack/*.npy` fields, by taking a per-pixel-position robust quantile across all fields of
the plate and fitting a low-order smooth surface to it, then apply
`(raw - darkfield) / flatfield` inside `preprocess_img_data` *before* normalisation. Write
the surfaces to `<plate>/illumination/flatfield_ch<c>.npy` and a corner/centre vignette
ratio into provenance, so a run states how much correction it applied. Yokogawa
CV7000/CV8000 fields vignette for real — the same host cell measures 15–30 % dimmer in a
corner — and grep confirms spaCR has **no illumination correction anywhere**
(`flatfield`/`darkfield`/`vignett` appear only in `diameter.py`, `align.py` and the Qt
space theme). `remove_background` is a scalar clip and normalisation is a global
percentile stretch; both preserve the gradient, which then propagates into every
`*_mean_intensity`, into the recruitment ratios, and into `frac_high90`, which is
thresholded on the whole field's p90. `submodules.py` already concedes this by
thresholding invasion per field specifically so a plate-wide cut does not turn an
illumination gradient into an invasion gradient — fix the cause and those workarounds get
their statistical power back.
**Builds on:** `io.preprocess_img_data:1806`, `io._normalize_img_batch:1238`,
`diameter._illumination:540` (same decimate-and-blur trick, per image rather than per
plate), `io._save_settings_to_db`. numpy/scipy only.
**Effort:** M–L. **Risk:** the quantile-across-fields estimator assumes objects are
positionally random. A plate where cells reproducibly pile at the well edge bakes real
biology into the "flatfield". Pool all fields of all wells, refuse to fit when the
residual is not smooth, and always keep the uncorrected value alongside.

### 12. `chanreg` — inter-channel registration check and sub-pixel correction
Phase-correlate each channel against a reference channel per field, aggregate the
per-field shifts over the plate into a per-channel median with a CI, and write a
`channel_registration` table. Distinguish a *constant* offset (multi-camera or chromatic
misalignment, correctable by one sub-pixel `scipy.ndimage.shift`) from a *field-varying*
one (stage instability, reportable but not correctable), and refuse to report a shift for
channel pairs whose overlap NCC is below a floor. spaCR computes Pearson and six Manders
columns per object per channel pair, plus periphery and outside rings 1 and 5 px wide — a
2 px inter-camera offset, routine on a CV7000, is wider than the PVM ring itself and
silently collapses exactly the colocalisation and recruitment numbers the screen rests on.
Because it is identical in every well, no control catches it. `align.py`'s own header says
channels share one tile solution and are never registered independently — correct for
stitching, and it leaves within-field channel registration entirely unmeasured.
**Builds on:** `align.py:1152` already imports `skimage.registration.phase_cross_correlation`
with `ALIGN_UPSAMPLE` sub-pixel refinement (`align.py:170`), `align._ncc:1075` as the
confidence scorer, and the "unregistered pairs are reported, not guessed" convention worth
copying wholesale; reads through `crops.MergedField`. **Effort:** M. **Risk:** phase
correlation between structurally unrelated channels (DAPI vs a sparse punctate marker)
returns a confident-looking noise peak. Score the shift by NCC after applying it, prefer
pairs sharing the autofluorescent cell body, and pool hundreds of fields.

### 13. `artefact` — raw-pixel saturation and debris QC per field
A per-field, per-channel pass over `stack/*.npy` — pre-normalisation, while the raw dtype
ceiling still exists — producing a `field_qc` table with: saturated-pixel fraction, whole-
field focus score as a ratio to the plate median, a **broadband debris mask** (pixels
simultaneously in the top percentile of *every* channel, morphologically opened — real
fluorophores are channel-specific, dust and fibres and plate-bottom scratches are not),
and a low-frequency anomaly score against the plate flatfield, which is what a bubble or a
meniscus looks like. Optionally write the debris mask as an extra plane in `merged/` so
`measure_crop` can exclude those pixels. Broadband autofluorescent debris is the most
common source of a false hit in an intensity screen: bright in the readout channel,
segments as an object, lands in one well — the exact signature of a real hit. Grep
confirms **no saturation check anywhere in the package**; `validate.py` reads `.npy`
headers only, `seg_qc` starts from label masks, `plate_qc` starts from well aggregates.
**Builds on:** `io.preprocess_img_data:1806` as the pre-normalisation hook;
`measure._estimate_blur:1211`; `zstack._focus_scores:445`; `seg_qc.format_scorecard:932` /
`write_scorecard:1001` so the output reads like the QC the user already knows;
`errors.RunLedger` for the verdict stamp. **Effort:** M. **Risk:** the broadband rule
fires on genuinely bright cells when channels bleed into each other, so it must be
rank-based rather than absolute. And the saturation check only works pre-normalisation —
point it at `merged/` by mistake and it silently always reports zero.

### 14. `coloc` — Costes-thresholded colocalisation with a per-object null
A second colocalisation estimator beside the shipped one: Costes' automatic threshold
(regress channel j on channel i over the object's pixels, walk the threshold down the
regression line until the below-threshold pixels are uncorrelated), Manders M1/M2 at *that*
threshold, and a per-object p-value from Costes block-scramble randomisation. Deliberately
descoped from the original proposal: the separate observation that the shipped
`manders_thresholds` default `[15, 85, 95]` makes M1 ≈ 1 for essentially every object is a
**settings bug to file**, not part of this feature. What earns the feature is the two
things a percentile threshold can never give: a data-derived threshold, and a null. A PVM
ring and a mitochondrial network in the same small region correlate simply because both
are non-uniform, and the shipped Pearson has no way to say a per-object value is more than
geometry.
**Builds on:** `measure._calculate_correlation_object_level:1166` and its call loop
(`measure.py:828-837`); `settings['manders_thresholds']` (`settings.py:593`);
`feature_dict.FEATURE_FAMILIES` `'correlation'` for the dictionary entries. numpy/scipy
only. **Effort:** M. **Risk:** runtime. The coloc loop is already O(channels² × 5 masks)
per field; 100 randomisation draws multiplies it. Default the null off, vectorise the
block scramble, and restrict it to `pathogen` and `cytoplasm`, where the question is
actually asked.

## Toxoplasma-specific measurement and assays (4)

### 15. `pvm_profile` — signed, physically-anchored intensity profile across the vacuole membrane
For each pathogen object, compute a signed Euclidean distance transform (negative inside
the vacuole, positive outside) restricted to the parent cell, and bin every channel's
intensity into **fixed physical shells** (default −5 to +15 px at 1 px steps, converted to
µm when the measurement stamp says so). Fit four numbers per vacuole per channel: peak
offset, peak height, FWHM and `recruitment_index` = peak / host-cytosol plateau, with a
per-object null from re-sampling the same profile around the vacuole mask displaced to a
random in-cell position. This is the central Toxoplasma assay — is host protein X at the
PVM? — and the three shipped columns that gesture at it each fail specifically.
`periphery_mean` is boundary pixels only, one shell, no baseline. `outside_mean` is a
single 5 px shell that averages a sharp PVM ring together with 4 px of host cytosol.
And the radial-distribution bins are **not usable for this**: verified at
`measure.py:1159` the distance map is `distance_transform_edt(~object_boundary)` —
*unsigned*, so bin 0 mixes intra-vacuolar with peri-vacuolar pixels — and the bin edges
are `max_distance/num_bins` where max_distance is *that host cell's* extent, so
`rad_dist_..._bin_3` is a different physical distance in every row of the table. A
fixed-shell profile separates "ring at the PVM" from "accumulated in the lumen" from
"generally brighter cell", per vacuole, as a continuous score a regression can use.
**Builds on:** `measure._outside_intensity:1032` and `_calculate_radial_distribution:1074`
for the pattern; `scipy.ndimage.distance_transform_edt` with `sampling` from
`measure.resolve_measurement_spacing`; `crops.MergedField.mask_plane:489` for memory-mapped
access; `pathogen.cell_id` (`measure.py:804-826`); `MEASUREMENT_STAMP_COLUMNS`
(`measure.py:90`) so the shells are physical. Note the existing `rad_dist_*` columns are
read by nothing but `feature_dict` — a "plot what you already have" panel is a free
no-reprocessing fallback, provided it is labelled as per-cell-normalised and not pooled
across wells. **Effort:** L. **Risk:** at 20× / 0.325 µm px the PVM is ~1 px wide, so the
peak is 1–2 samples and the fit is fragile; it also effectively *depends* on `chanreg`,
since an unregistered recruitment channel smears the peak. Report the profile itself, not
only the fit, and refuse to fit below a minimum vacuole diameter.

### 16. `pvm_contact` — how much of the vacuole perimeter is actually touched
Pixel-level rather than column-level. For each pathogen object, memory-map its field, take
the pathogen mask plane, extract the boundary, walk it, and sample a marker channel in a
1–3 px ring, thresholding positivity against that same host cell's own
`cytoplasm_channel_C_percentile_75` from the database row. Report **fraction of PV
perimeter positive**, **number of contiguous contact arcs**, **longest arc in degrees**,
and mean intensity inside vs. outside the arcs. Host mitochondrial association (HMA/MAF1)
and host ER wrapping are *coverage* phenotypes: a PV with one mitochondrion hugging 40 % of
its circumference and a PV sitting in uniformly bright cytosol have identical
`pathogen_channel_C_periphery_mean` — and that is exactly the column `analyze_recruitment`
divides by cytoplasm mean. Contiguity is destroyed the moment the ring is averaged, which
makes this orthogonal to every radial measurement including #15.
**Builds on:** `crops.MergedField.mask_plane:489` / `read_window` / `label_index`,
`crops.DEFAULT_MASK_DIMS:133`; the `path_name` column written at `utils.py:1934` as the
DB→pixels join key; `measure._periphery_intensity` as the reference ring;
`cytoplasm_channel_C_percentile_75` (`measure.py:969`). `skimage.segmentation.find_boundaries`.
**Effort:** L. **Risk:** needs `merged/*.npy` still on disk with correct `*_mask_dim`
settings, and the positivity threshold is the whole ballgame — it needs a control-well
default and per-plate calibration. On a 3-D run "perimeter" becomes a surface and the arc
logic does not generalise; refuse volumetric input rather than reporting a wrong number.

### 17. `egress` — detect the event, not just the calcium signal that precedes it
Over a tracked timelapse, for every host-cell track find the frame where its pathogen
count goes from N ≥ 2 to 0, or total pathogen area drops > 80 % within two frames, with the
host cell simultaneously changing area and solidity. Report per well: egress events per
infected cell per hour, **parasite burden at the last frame before the drop**, and
time-from-first-detection to egress for tracks followed from the start. This is the direct
readout for egress mutants (CDPK3, PLP1, DGK2) and for compound-induced egress. spaCR
already analyses calcium oscillations — the *signal* that precedes egress — but nothing
scores the event. "Burden at egress" is the measurement that separates premature from
delayed egress, and it cannot be obtained from a fixed endpoint at all.
**Builds on:** `timelapse._process_merged_group:3183`, which already emits per-cell-per-frame
rows with aggregated pathogen features and a per-parent child count from
`_summarise_child_features_per_parent:2852`; `_relabelled_stack_to_tracks_df:673` for
`frame`/`track_id`/`x`/`y`; `_filter_short_tracks:950` as the existing quality gate;
`analyze_calcium_oscillations:1563` for the signal cross-reference. **Effort:** L.
**Risk:** a count dropping to zero is also what a tracking failure, a cell drifting out of
frame and photobleaching all look like. Border-touching and last-frame track exclusion, a
minimum track length, and a per-field photobleaching control are mandatory work, not
optional polish.

### 18. `plaque_screen` — wire the plaque assay into the screen, with lytic-cycle shape metrics
Parse the Yokogawa filename into `plateID/rowID/columnID/fieldID`, aggregate to per-well
**plaque number** and **plaque size distribution as separate outputs**, and take the shape
properties from the `regionprops` call that today reads only `.area` — `solidity`,
`eccentricity`, `perimeter`, `equivalent_diameter_area` — plus monolayer **clearance
fraction**. Write a `perform_regression`-shaped CSV. Verified: `analyze_plaques`
(`submodules.py:905`) records only `file` / `object_count` / `average_size` /
`std_dev_size` and writes `plaques_analysis.db` with **no `plateID`/`rowID`/`columnID`**,
so plaque results cannot be joined to the barcode tables and a plaque-based CRISPR screen
is a dead end in this codebase. Scientifically, plaque number (initiation: invasion plus
first replication), plaque size (lytic cycle rate over 5–7 days) and plaque raggedness
(coherent front vs. satellite micro-plaques — motility/egress vs. replication mutants) are
three different phenotypes that `average_size` collapses into one.
**Builds on:** `submodules.analyze_plaques:905`, the bundled `toxo_plaque_cyto_e25000`
Cellpose model it already downloads, `spacr_cellpose.identify_masks_finetune`, the
well-parsing helpers around `utils._map_wells`, `ml.perform_regression`'s CSV contract.
**Effort:** M. **Risk:** plaque plates are often imaged as one whole-well montage rather
than numbered fields, so the parser must handle a naming convention the rest of spaCR
never sees. Low-mag plaque images also vignette badly, which biases solidity and clearance
fraction — this one wants `illum` first. Honest caveat: the metadata gap is a bug and
should be filed as one regardless of whether the rest is built.

## Screen statistics and hit calling (6)

### 19. `screen_fdr` — FDR-calibrated hit calling with a non-targeting empirical null
Replace the raw cut with error-controlled calling: add `p_value_adj` (BH / Storey q) to
`coef_df`, and in parallel build an empirical null from the ~30 non-targeting control
gRNAs already listed in `settings['controls']`, whose coefficients give the distribution of
"no effect" under the actual design. Emit a p-value histogram, a control-uniformity
calibration plot and an FDR-vs-hit-count curve so the threshold is chosen knowingly. A
5,000-gRNA screen at raw p ≤ 0.05 returns ~250 gRNAs by chance and the current hit list has
no way to say how many of its entries are noise. Verified: `ml.py:1483` is
`significant = coef_df[coef_df['p_value']<= 0.05]`, and `multipletests` — imported at
`utils.py:29` and used in `utils`, `sp_stats` and `submodules` — appears **nowhere in
`ml.py`**. The existing control threshold filters on effect size *after* the p-cut, so it
controls nothing. The calibration plot also tests whether OLS p-values are even valid
under the collinear `fraction:grna + gene_fraction:gene` design.
**Builds on:** `ml.py:1483`; `process_model_coefficients:505`, which already tags each row
`condition` ∈ nc/pc/control/other; `settings.py:1063`; `sp_stats.choose_p_adjust_method:10`.
**Effort:** S–M. **Risk:** if control gRNAs are not exchangeable with targeting ones
(different `fraction` distributions, different well occupancy) the empirical null is
biased — match on `fraction` and `grna_well_count` before forming it.

### 20. `assay_window` — Z′-factor and SSMD from the control wells the pipeline deletes
Per plate, compute Z′-factor, SSMD, robust MAD-based z′ and positive-vs-negative control
AUC for the chosen phenotype, using the c1/c2/c3 control wells that `clean_controls`
removes at `ml.py:1269` (`filter_value=['c1','c2','c3']`, verified at `settings.py:1081`).
Output a per-plate table, a traffic-light panel and control-well strip plots. Answers "is
this plate good enough to screen on?" before it enters hit calling — Z′ < 0 means the assay
cannot separate the strongest positive control from the negative, yet the pipeline
currently regresses on it silently. It also lets the phenotype *definition* be chosen on
evidence: compare `pred` against a recruitment ratio against
`pathogen_channel_2_periphery_mean / _outside_mean` by which gives the widest assay window
on the same plate. That choice is guesswork today. Grep confirms **no Z′-factor, SSMD or
B-score anywhere in the repo**.
**Builds on:** `ml.clean_controls:1751` and its call site; the
`positive_control='c2'` / `negative_control='c1'` convention in `ml_analysis:2127`;
`plate_qc.load_plate_frame`; `plot.plot_plates`. **Effort:** S–M. **Risk:** control-well
identity is a convention, not recorded data. If a plate deviates the numbers are
meaningless, so the layout must be an explicit setting with a fail-loud check that both
control classes are present.

### 21. `gene_level` — real gene-level aggregation from sgRNA statistics
Take the per-gRNA table and aggregate member gRNAs into a gene call with three selectable
estimators: inverse-variance-weighted meta-analysis (gene effect + CI + heterogeneity I²),
Stouffer's weighted z, and an α-RRA rank statistic with permutation-derived gene p-values.
Report n_gRNA per gene, sign agreement, and flag genes driven by one outlier gRNA. This is
genuinely not what ships: gene terms currently come from the *same* regression as gRNA
terms via `gene_fraction`, so gene and member-gRNA coefficients are collinear by
construction and compete for identical variance, and `results_gene.csv` is a row subset of
one joint fit, not an aggregation. A gene where 4/4 gRNAs shift vacuole recruitment the
same way is a very different claim from one where 1/4 does — currently indistinguishable.
**Builds on:** `results_path_grna` (`ml.py:997`); `coef_df` columns
`grna`/`gene`/`coefficient`/`p_value`/`n_grna`/`n_gene`; `prepare_formula:230`;
`grna_metricks:1041` for coverage-based weights. **Prerequisite (verified):** `std_err`
is populated only in the `beta` branch (`ml.py:510-517`); the OLS/GLM branches write
coefficient and p_value only, so `model.bse` must be carried through before
inverse-variance weighting is possible. Small change, but it comes first.
**Effort:** M. **Risk:** RRA permutations at library scale are the expensive part; cache
the null by gRNA-count stratum.

### 22. `concordance` — do my plates agree, and is plate 2 a dud?
Fit the regression per plate (or per user-defined replicate group) instead of on the
pooled concatenation, then report per-gRNA and per-gene effect-size scatter across
replicates with Pearson/Spearman/CCC, rank-rank plots, hit-list overlap (Jaccard +
hypergeometric p) and sign-replication rate, ranking each replicate by agreement with the
consensus of the others. Each flagged plate or well links to a cause column pulled from
existing QC — median object blur, cell count, edge-ring membership. Unanswerable today:
`_perform_regression_read_data` concatenates all plates and the default formula
(`prepare_formula:235`) carries `rowID + columnID` but **no `plateID` term at all**, so
plate offsets are unmodelled and one bad plate silently drags the hit list. Replicate
concordance is also the honest denominator for believing a hit and the first thing a
reviewer asks for.
**Builds on:** the existing multi-file plate assignment (`plates_score`/`plates_count`/
`plate_from_order`); `regression():717`; `prepare_formula:230`; `plate_qc.detect_edge_effect:1240`
and `row_column_trends:1019` for the cause columns. **Effort:** M. **Risk:** per-plate
fits have far fewer wells, so gRNAs present in one or two wells per plate become
non-estimable. A minimum-wells guard on `grna_well_count` is required, and the number of
dropped gRNAs must be reported rather than quietly comparing a shrunken set.

### 23. `plate_norm` — B-score / median-polish correction before hit calling
Insert a normalisation step between `process_scores` and `regression`: median polish
(B-score), per-plate robust z to the plate median or to negative-control wells, and a
plate-median offset for cross-plate pooling, writing corrected well values beside the raw
ones with before/after heatmaps and the variance explained by row/column/plate. Toxoplasma
plates evaporate at the edges and outer wells routinely give different infection rates;
that gradient is currently absorbed by `rowID + columnID` as ~24 additive fixed-effect
levels sitting in the same design matrix as the gRNA terms, which burns degrees of freedom
and can only represent additive row + column, never the ring structure
`plate_qc.detect_edge_effect` already shows is there. **Explicitly distinct from
`plate_qc`:** verified that module diagnoses (`RingStats`, `GradientStats`,
`EdgeEffectReport`, `row_column_trends`) and stops — nothing in it corrects a value or
touches the screen readout. This is the correction step, and it consumes `plate_qc`'s
report to decide whether correction is warranted at all.
**Builds on:** `plate_qc.detect_edge_effect:1240`, `row_column_trends:1019`,
`layout_matrix:906`; `ml.process_scores:1774`; `prepare_formula:230`; `plot.plot_plates`.
**Effort:** M. **Risk:** median polish assumes hits are a small, spatially unstructured
minority. Where a whole column is one condition, or a strong-phenotype gene is regionally
over-represented, B-score normalises away real signal. Opt-in only, with a diagnostic
showing how many hits change status — never default-on.

### 24. `cell_level_screen` — hit calling on per-cell scores instead of well means
Replace the collapse-to-well-mean with a cell-level mixture likelihood: each well holds a
known mixture of gRNAs at known proportions, so a cell from that well carries gRNA *g* with
probability ≈ `fraction_g`; per-cell phenotype is modelled as a mixture over the well's
gRNAs and effects are estimated by EM (or a hierarchical model) over all cells at once,
reporting per-gRNA effect + CI, per-cell posterior responsibilities, and the montage of the
cells most likely to be knockouts of gene X. `process_scores:1774` currently discards every
per-cell value, reducing thousands of cells to one number, which dilutes a gene affecting
only ~15 % of cells by roughly 7× — the difference between detecting a vacuole-recruitment
gene and missing it. This is the single proposal that exploits what makes spaCR different
from a bulk screen. Note: it was independently proposed three times across the ideation
batches; this is the canonical version, and the other two are cut.
**Builds on:** `ml.process_scores:1774` (the collapse point); the per-object score frames
in `_perform_regression_read_data`; `png_list.pred` / `cv_predictions`
(`deep_spacr.merge_predictions_into_db:1919`); the `fraction` column from
`process_reads:1690`; the existing `agg_type=None` path, which already passes per-object
rows through unaggregated; `submodules.compare_reads_to_scores:1075`, whose hard-coded
r1–r16 PC/NC titration (90:10 … 20:80) is a ready-made ground truth with known mixing
proportions. **Depends on `seq_link` (#30)** for the genotype↔cell bridge.
**Effort:** L. **Risk:** identifiability. With 5–20 gRNAs per well and few wells per gRNA,
EM will happily converge to a confident wrong answer. It needs an up-front
co-occurrence/design-rank check, a shrinkage prior toward the NC baseline, a per-gRNA
effective-N column and a refusal to report below a minimum well count. Ship alongside the
well-mean path, never replacing it, and validate on the titration rows.

## Sequencing (6)

### 25. `seq_qc` — read-attrition funnel and per-cycle quality scorecard
Instrument `process_chunk` to return counters for every stage a read can die at — reads
in, `target_sequence` found (R1 / R2 / both), window long enough after the slice,
`re.match` succeeded, and each of rowID/columnID/grna resolved to a name — plus the modal
anchor position and its spread, and per-cycle mean Phred and N-rate from the quality
strings `process_chunk` already parses and throws away. Emit a `seg_qc`-style scorecard
plus a per-sample PDF. Today `qc.csv` structurally cannot tell you a run failed: verified
at `sequencing.py:339`, `qc_df['total_reads'] = len(df)` where `df` is built only from
reads that already matched the regex, so the denominator is "reads that worked" and
attrition is invisible. A run where 96 % of reads never found the anchor and one where 96 %
matched produce the same-shaped QC file, and the user discovers the difference from a
`print` inside a worker process.
**Builds on:** `sequencing.process_chunk` (its two inner `*_find_sequence_in_chunk_reads`
closures already compute `r1_pos`/`r2_pos`); `save_qc_df_to_csv`, which already sums
element-wise via `.add(fill_value=0)` so new counter columns aggregate for free;
`seg_qc.format_scorecard`; `report.SECTION_KEYS` for a new `sequencing_qc` section.
**Effort:** S–M. **Risk:** changing `process_chunk`'s 3-tuple return breaks
`saver_process`, both chunked processors and `tests/test_sequencing.py`; and appending
columns to an existing `qc.csv` mid-project makes `.add(fill_value=0)` produce a ragged
frame.

### 26. `seq_layout` — infer the read layout and write the regex for you
Take the first ~100k reads and, for each of `row_csv` / `column_csv` / `grna_csv`, search
every read for exact occurrences of any reference sequence in both orientations, anchored
on `target_sequence`, then histogram (start offset, orientation, barcode set). The modal
offsets give the true layout; emit a ready-to-paste `regex`, `target_sequence`,
`offset_start`, `expected_end` and a per-set reverse-complement verdict, with the fraction
of reads consistent with the proposal and the runner-up layout. These four are the most
error-prone settings in spaCR — the shipped default regex was itself unusable until
recently because it named its groups `column`/`row` while `process_chunk` reads
`match.group('columnID')` — and getting them right today is trial-and-error with
`test=True`, one 10k-read chunk at a time, per sample. Orientation is a separate manual
guess (`barecodes_reverse_complement`) with no way to know which set needs it.
**Builds on:** `sequencing.map_sequences_to_names:20`, `barecodes_reverse_complement:706`,
`process_chunk`'s anchor logic, `io.parse_gz_files:3206`, and `validate.py:875`, which
already checks the three CSVs exist — this extends it from "the file is there" to "the
file matches your reads". **Effort:** M. **Risk:** 8-nt well barcodes match by chance at
many offsets. Without a background-rate correction the modal offset can be noise — anchor
the scan on `target_sequence` and refuse to propose a layout when the peak is not clearly
above background.

### 27. `barcode_resolve` — error-tolerant barcode assignment with a collision guard
Three parts: audit each reference CSV (minimum pairwise Hamming distance, distance
histogram, duplicate sequences or names) and state how many substitution errors the set
can correct; build a distance-1 (and distance-2 where the set allows) neighbour index and
rescue a read only when its correction is *unique*, counting ambiguous corrections
separately; and report per-set rescue yield ("1-mismatch recovery returns 11 % of reads on
the row barcodes, 0.4 % on the protospacers"). Verified: `map_sequences_to_names` is a bare
`csv_sequences.get(sequence, pd.NA)` (`sequencing.py:63`), so one base error anywhere in
an 8-nt well barcode throws the read away — and the `fill_na=True` escape hatch is worse,
substituting the raw sequence as the ID, which becomes a phantom well or phantom guide in
`unique_combinations.csv` and flows straight into `ml.process_reads`, inflating the
per-well denominator. At Q30 across three barcode blocks the loss is large, systematic and
currently unmeasured.
**Builds on:** `map_sequences_to_names`, `process_chunk`'s `fill_na` branch, and
`rapidfuzz` — **verified declared at `setup.py:60` and `requirements.txt:30` and imported
by nothing in `spacr/`**, so this costs no new dependency. **Effort:** M. **Risk:** a
barcode set with minimum Hamming distance 1 or 2 is not error-correcting and rescuing on
it mis-assigns wells, which is worse than dropping reads. The audit must gate the rescue
and the tool must be loud about refusing. Distance-2 maps over a 20–21 nt protospacer
library must be seed-and-extend, not full enumeration.

### 28. `library_qc` — representation, skew and reference dropout
Use the full `grna_csv` as the *denominator* — today it is only ever a lookup, never a
completeness check — and report per sample and per plate: how many reference guides were
never observed at all, observed at < 10 and < 50 reads; how many genes lost all their
guides; Gini coefficient and Lorenz curve of guide counts; top-10 % / bottom-10 % skew;
reads per guide. Then the within-well version: for each well, the skew across the guides
actually present, rendered as a `plot_plates` heatmap of per-well Gini and depth beside
the existing `unique_counts` map. Answers "did my library arrive intact and stay even"
before any hit calling. A gene that lost all its guides in the prep is a guaranteed false
negative and nothing currently names those genes.
**Builds on:** `sequencing.graph_sequencing_stats:735`, which already computes
`total_count` and `fraction` per `prc`; `unique_combinations.csv`; `settings['grna_csv']`;
`plot.plot_plates:2214`; the `org_gene_grna` split in `ml.process_reads:1711`. Not a
duplicate of `graph_sequencing_stats`, which picks a *threshold*; this reports *coverage
against the reference*. **Effort:** M. **Risk:** a whole-library Gini is meaningless for
an arrayed-pool design where each well is meant to hold ~5 guides. The feature must compute
skew within the intended pool per well and be explicit about which denominator each number
uses, or it produces alarming, meaningless statistics.

### 29. `seq_chimera` — index-hopping and template-switching estimate
Every read carries a (rowID, columnID, grna) triple from one amplicon, so a chimera is
directly observable as a triple whose row and column come from different wells. For each
guide, fit the independence null — expected hopped count in well (r,c) ≈
n(r,·)·n(·,c)/n(·,·) — and report per plate the overall hopping rate, per well the fraction
of reads explained by row/column bleed, and per guide a "cross" diagnostic showing whether
its apparent multi-well presence lies along its own row and column. Let the user declare
deliberately-empty wells; reads landing there are an assumption-free hopping-rate
measurement. Ship a corrected `unique_combinations.csv` as an opt-in second output. At
`target_unique_count = 5`, even 1 % hopping puts spurious guides in every well;
`graph_sequencing_stats` currently handles this by picking a global fraction cutoff, which
discards low-abundance real guides along with the contamination.
**Builds on:** `unique_combinations.csv` — the row × column × guide triple exists in no
other spaCR output; `graph_sequencing_stats`' `prc`/`fraction` construction;
`ml.grna_metricks:1041`, which already counts distinct wells per guide; `plot.plot_plates`.
**Effort:** M–L. **Risk:** if a library really is laid out so a guide occupies many wells
in the same row, the independence null over-corrects and deletes real signal. Diagnostic
first, applied correction only on request, and empty-well calibration before it is trusted.

### 30. `seq_link` — put genotype in `measurements.db`
Write the well→guide table (guide, count, fraction, the threshold used, source sample)
into the plate's `measurements/measurements.db` as a `sequencing` table keyed on `prc`,
with an explicit, recorded FASTQ-sample ↔ `plateID` mapping. Deliberately descoped: the
per-cell EM layer that was originally bundled here is `cell_level_screen` (#24), and this
is its prerequisite. Verified: grep finds **no `sequencing` table anywhere in
`measurements.db`** — barcode data lives only in per-sample `unique_combinations.csv`
files, and the sample↔plate correspondence is *positional* in `perform_regression`
(`plates_count` aligned by order with `count_data`). That means genotype cannot currently
be joined to a cell at all. Once the table exists, `umap`, `annotate`, `activation`,
`db_browser` and `plate_view` can all consume it immediately: colour a UMAP by gene, filter
the annotation queue to one guide, stratify Grad-CAM by genotype.
**Builds on:** `sequencing.generate_barecode_mapping` output; `ml.process_reads:1598` for
the fractions; `utils._create_database` / `_merge_and_save_to_database` for the write;
`ml.grna_metricks:1041` for the per-guide well counts that belong in the same table.
**Effort:** S. **Risk:** the sample↔plate mapping is the entire risk. It must be an
explicit user-supplied mapping with a fail-loud check, not inferred from file order, or
every downstream join is silently wrong in a way nothing will catch.

---

# The five I would build first, and why

1. **`illum` (#11).** It is the only proposal that corrupts *everything else* if left
   undone. A 15–30 % corner-to-centre gradient is currently inside every intensity column,
   every recruitment ratio, `frac_high90`, and therefore inside every hit list — and the
   invasion module already works around it per field, which is a confession. Fix it once
   and every past and future measurement gets better without anyone opening a new app.

2. **`screen_fdr` (#19) with `assay_window` (#20) as a pair.** The headline output of the
   whole tool is currently uncalibrated: raw p ≤ 0.05 over thousands of gRNAs, on plates
   nobody checked were separable, using control wells the pipeline deletes. These are the
   two smallest-effort features in the list and together they change a hit list from "here
   are some numbers" to "here are N hits at 10 % FDR on a plate with Z′ = 0.61". Build them
   in the same week.

3. **`bystander` (#1).** One join away, and it unlocks the question the lab actually
   studies — is this effect cell-autonomous effector biology or a paracrine host response?
   Right now the answer is not merely hard, it is impossible, because the uninfected host
   cells are deleted by an inner merge before analysis begins. Highest science-per-line of
   anything here.

4. **`burden_adjust` (#2).** The dominant false-positive class in a pooled host-factor
   screen is a gene that only changes invasion masquerading as a recruitment hit, because
   nearly every host readout scales with parasites per cell. Giving the researcher the raw
   and burden-residualised well mean side by side, and letting them see which hits survive,
   is the cheapest large improvement in hit quality available.

5. **`object_qc` (#3), tier 1 only to start.** Retroactive, runs on every database already
   on disk, no reprocessing, and it hands every other module one agreed definition of a
   usable cell instead of each analysis inventing its own filter. It is also the answer to
   the first question a referee asks. Add the Cellpose-flow tier later, when masks are next
   re-run.

*(Honourable mention: `seq_qc` (#25) and `barcode_resolve` (#27) are both S–M and both fix
things that are silently invisible today. If the next thing you do is a sequencing run
rather than an imaging run, promote them ahead of #3.)*

---

# Cut, and why (14 of 44)

**Duplicates — the capability survives elsewhere, build it once.** Single-cell mixture
deconvolution and the EM half of `genotype_deconvolve` were the same model as
`cell_level_screen` (#24), proposed three times; `pv_halo` was the same signed-shell
profile as `pvm_profile` (#15); "radial recruitment profiles" proposed plotting the
existing `rad_dist_*` bins, which are unsigned and per-cell-scaled and so cannot be pooled
across wells (kept only as a labelled fallback panel inside #15); "field neighbourhood
context" was `spatial` (#10) less well grounded; "replicate concordance" was
`assay_window` + `concordance` in one wrapper. The object-QC gate and `segconf` were
**merged** into #3, and the materialised `vacuole` table was **merged** into #8.

**Cut on merit.** `explain-this-cell` — a composition of shipped `attribution` plus a
surrogate model whose entire value is its R², strictly downstream of `feature_rank`.
`phenotype card` — pure assembly, depended on two cut proposals and one broken function;
revisit as a `report` section once #5 and #15 exist. `screen_power` — the canonical
open-once-never-again calculator, and its variance estimate does not transfer to the next
parasite prep. `unmix` (bleed-through) — the estimator cannot separate GFP→RFP leakage
from a host protein genuinely concentrating at the vacuole, which is precisely the biology
being measured, and no settings key marks a single-stain control well, so the one
defensible mode has no data to trigger on. `guide_dropout` — needs a second sequenced
sample spaCR never captures in a normal run. `well_map_check` — the plate-rotation audit
can legitimately conclude "cannot distinguish" on a real layout, and a rotated plate also
surfaces in `assay_window`. `public_ref` — **cut on a factual check**: the GT1 CRISPR
fitness columns do not ship with the package (`toxoplasma_metadata.csv` has no
`T.gondii GT1 CRISPR Phenotype` column; `settings.py:1084` points at a user's local
`TGGT1_Summary.csv`), so its stated data foundation does not exist, and its GO half calls a
broken function.

---

# Not features — bugs found while checking (for the FIXES list)

- `toxo.py:275` — `go_term_enrichment_by_column` builds its hit list as
  `significant_df['n_gene'].to_list()` and matches it against `metadata['gene_nr']`. But
  `n_gene` is a **count** column (`df['gene'].value_counts()`), not a gene ID, so it
  compares occurrence counts (12, 40) against ToxoDB gene numbers (292020). The hit set is
  ~always empty and the enrichment is vacuous. No multiple-testing correction across GO
  terms either.
- `measure.MORPHOLOGICAL_PROPS` (`measure.py:60`) omits both `'centroid'` and
  `'orientation'`. Geometric object position is only recoverable via a per-channel
  *intensity-weighted* centroid, which is why `submodules._find_centroid_columns` needs a
  three-step fallback. `centroid` is valid in 2-D and 3-D; `orientation` belongs in
  `PROPS_2D_ONLY`. Prerequisite for #8 and #10, and it would unlock rosette-angle metrics.
- `sequencing.py:466-468` and `:564-566` — both chunked processors do
  `pool.apply_async(process_chunk, ...)` immediately followed by `result.get()` inside the
  read loop. Only one chunk is ever in flight, so a `Pool(cpu_count()-3)` runs effectively
  serial. Fixing it is a several-fold speedup on every sequencing run.
- `sequencing.py:103` — `unique_combinations.to_csv(csv_file, index=True)` after a
  `groupby(..., as_index=False)` writes a `RangeIndex` column; the next append reads it
  back as `Unnamed: 0` and sums it, so the file accumulates a meaningless growing integer.
- `settings.py:593` — `manders_thresholds` default `[15, 85, 95]`. At threshold 15, M1 is
  "the fraction of total intensity above the object's own 15th percentile", which is ≈ 1
  for essentially every object. The columns are near-vacuous at the default and a screen
  may be making calls on them.
- `submodules.py:905` — `analyze_plaques` writes `plaques_analysis.db` with no
  `plateID`/`rowID`/`columnID`, so plaque results cannot be joined to anything. Worth
  fixing whether or not #18 gets built.
- `ml.py:510-517` — `std_err` is written only in the `beta` branch; OLS/GLM discard
  `model.bse`. Blocks any inverse-variance weighting (#21).
