---
name: spacr-graphs
description: The maintainer's rules for graphs and figures (spaCR, paper figures, slides, reports), including the reference-paper house style and the matplotlib helper spacr_figstyle.py (fixed bar width/gap, fixed axis width per condition count, 60 % alpha, grey control, auto box/violin above 20 points). Use whenever making or reviewing any chart, plot or figure panel. Pair with spacr-presentations for slides.
---

# Graph rules (maintainer, collected 2026-10 to 2026-11-12)

## Chart type
- Multi-condition comparisons: bar graphs plus jittered replicate points, both at about 60 % opacity (with more than 20 points per group: box or violin instead, see 2026-11-18). No paired-dot plots and no trade-off scatter plots for comparisons.
- Per-class results: heatmaps with control rows.
- Show significance with brackets (Holm-corrected unless stated otherwise).
- Show only what answers the question. For example, plate averages in one graph rather than every well's bar.

## Axes
- Start the y-axis where the data live, not at 0 by habit, when that makes the differences readable (e.g. 50 % for genotyping rate; 750 or 90.0 when asked). State this in the legend.
  - (2026-11-18) When the maintainer asks for an "honest" axis (e.g. the footprint sweep line graph), start at 0. Lines and sweeps start at 0 by default; a cut axis is allowed only where the maintainer asked for it, and is stated in the legend.
- Explain every x-axis item and every y-axis metric, in the legend or on the slide.
- Spell out abbreviations.

- Legend colours must match the plotted colours exactly; no clipped labels.
- Too many points: show per-well means or a density/hexbin, not tens of thousands of dots.
- When one method detects more objects, show a representative FOV segmented by both (same image, same display range) so viewers can judge real cells vs artefacts.
- Zooms of detections: intensity-normalise the display (same stretch for both methods) so the detected objects are visible.

- Axis labels centred on their axes.
- Insets: mark their source region in the full image with a thin white box (0.5 pt lines).
- Outline overlays: spaCR teal, competitor magenta. Objects found by only one method: highlight their outline in magenta (or the agreed colour), never with a centre dot.

- Any graph whose conditions are not labelled in the graph itself gets a legend box (e.g. hatched = shuffled genes, teal = spaCR, blue = Brieflow). Outline overlays always carry a colour legend.
- Centroids on overlays: crosses (+) in the method colour.

- Medians on jitter plots: a short black horizontal line across the points, not an arrow. Every marker (arrow, triangle) is explained in the legend.

## Style
- Font: Open Sans (Light text, Regular titles, SemiBold emphasis), black.
- Palette: light, semi-transparent blue, teal and gray (spaCR cool palette). Never brown. **For figures this is refined by the reference-paper palette below (2026-11-18); never brown and control grey still hold.**
- Minimal: no chart junk, no shadows, legends that don't overlap the data.
- Titles go in the legend for paper figures; minimal whitespace and text.
- Images: the same background and the same display range and gamma for all compared images. Document every image-processing step.
- Model outlines from a maintainer-trained Cellpose model come with that model's scorecard.

## Content
- Fair comparisons only: same data, n, genes, scoring and controls. State the matching in the legend. Unmatched comparisons are not shown.
- Every chart has a control or reference row (shuffled, chance, NT).
- Representative images: say where the shown example sits in the distribution.
- Look at every rendered figure before handing it over.
- Save general graph feedback from the maintainer into this skill.

# New rules (maintainer, 2026-11-18)

## Geometry and symmetry
- **Same bar width, same gap.** Every bar in every graph has the same width and the same gap between bars. (2026-11-18)
- **More than 20 data points per group: box or violin**, not a bar with whiskers. Bars, boxes and violins all have the same width and spacing, so every element has the same size and the figure is symmetrical. (2026-11-18)
- **Same number of conditions, same graph width.** All 3-condition graphs share one width, all 4-condition graphs another, and so on. (2026-11-18)
- **Control or reference is always grey.** (2026-11-18)
- **Every graph element is at 60 % opacity**: bars, boxes, violins, points, data lines, histogram fills, heatmap cells. Text, axes and tick marks stay at 100 %. (2026-11-18)
- How to get it: `spacr_figstyle.py` in this folder. One condition = one slot of `SLOT_IN` = 0.25 in; axis width = n × 0.25 in; bar/box/violin width = 0.6 slot (0.15 in), gap 0.4 slot (0.10 in). Graphs with a continuous x axis (histograms, scatter, lines, sweeps) are not slot-bound: they are square (see "Axis aspect" below; `square_axes`). Heatmaps: `heatmap_axes`, square cells.
- Grouped bars (e.g. guide vs decoy at distance 2 and 3) are just consecutive slots, labelled with group ticks under the axis. No narrower "dodged" bars.

## Reference-paper house style (2026-11-18)
Studied on 2026-11-18 from the figures of five of the six reference papers the maintainer chose (all from the Lourido lab and collaborators):
1. Nat Commun 2026, s41467-026-73079-z (SWI/SNF complexes in *Toxoplasma*), Figs 1-7.
2. Nat Microbiol 2024, s41564-024-01754-2 (in vivo CRISPR screen in *Toxoplasma*), Figs 1-5.
3. eLife 93877 (SPARK and AGC kinases), Figs 1-7.
4. bioRxiv 2026.09.01.748362 (proteome-wide crosslinking MS in *Toxoplasma*), Figs 1-6.
5. EMBO J 2026, s44318-026-00911-z (MIC1/4/6 and host N-glycans), Figs 1-6.
6. Cell S0092-8674(26)00827-5: **not seen.** cell.com refused automated fetches (HTTP 403), so nothing here comes from it.

Where the papers differ, the majority rule is taken and the split is noted as (k/5).

### Palette (decision 2026-11-18)
The maintainer asked to copy the papers, so for **figures** this palette replaces the pastel spaCR palette. "Never brown" and "control grey" stay. The colours were measured from the figure pixels.

| Role | Hex | Source |
|---|---|---|
| Control / reference condition | `#8C8C8C` mid grey | all 5 (WT, parental, vehicle, TIR1, "random" in grey) |
| Background points (not highlighted) | `#C8C8C8` light grey | NC, NM, EMBO, bioRxiv screen scatters |
| Primary accent (the treatment or method of interest) | `#0A77B8` blue | NM, eLife, NC (3/5); bioRxiv/EMBO use magenta (2/5) |
| Second series | `#0A9878` teal-green | eLife; same family as spaCR teal |
| Third series | `#283868` navy | NC, eLife dark blues |
| Paired lighter tint of the accent (e.g. −drug / +drug) | `#78AED2` | NM, eLife, EMBO pair a saturated and a light tint of one hue |
| Fourth series | `#C878A8` magenta | EMBO, bioRxiv |
| Fifth series only | `#F0A830` amber | NC, NM, EMBO; never darkened towards brown |
| Reference line | black, 0.6 pt, dotted, 60 % | EMBO dotted 100 % line, bioRxiv dashed 30 Å line |

- One accent per comparison: the papers colour only what the reader should look at; everything else is grey (5/5).
- Sequential maps: white → blue (`Blues`) for fractions. Diverging data such as log fold change: blue-white-red, centred at 0 (NC, eLife). Never jet or rainbow, except a flow-cytometry density (EMBO).
- spaCR teal stays the outline colour on images. Competitor outlines stay magenta.

### Text, lines and markers (measured at final print size)
- **Font:** the papers use a Helvetica-type grotesque (5/5). We keep **Open Sans** (existing rule). Sizes: 7 pt for axis labels and text, 6 pt for tick labels, 5–5.5 pt for value labels and n, 6.5–7 pt for in-graph titles. **SUPERSEDED 2026-11-18: minimum font size is 8 pt for every text element (ticks, labels, values); size panels from that.**
- **Panel letters:** bold capitals (A, B, C), 10 pt, top left, outside the axes (4/5; only Nat Microbiol uses lower case, as Nature house style). If a journal demands lower case, switch all of them.
- **Axes:** full box frame, all four spines (4/5; EMBO uses open L-axes). 0.6 pt black, ticks 2.5 pt pointing outward. Heatmaps have no frame.
- **Data lines:** 1.0 pt. Medians: 0.9 pt black horizontal line across the element. Reference lines: 0.6 pt dotted or dashed.
- **Markers:** replicate points are 4 pt² filled circles (n ≤ 20); points over a box or violin are 1–1.5 pt². Replicate means in a superplot (eLife) are larger black markers with a different shape per replicate.
- **Stats:** a bracket with the exact p value above it (5/5), e.g. "p = 0.011"; "n.s." only when the test is stated. Group labels go under the axis: a thin line with the group name below (5/5), or a +/− condition table (EMBO, eLife "h + IAA" rows).
- **Legends:** label lines directly with coloured text at their end, or put a frameless legend in an empty corner inside the axes (4/5), never over data. A colour key shared by several panels goes once, outside the panels. The title goes in the figure legend (existing rule); in-graph titles only name the dataset.
- **Schematics:** flat shapes with no gradients or shadows. Light-grey bodies (`#D0D0D0`–`#E4E4E4`) with thin darker-grey outlines, and one or two accent colours for the element under study. Rounded rectangles for genes, constructs and categories. Black arrows with small heads. Short labels next to the objects, not in boxes (5/5).
- **Images:** a row of images per condition, a white scale bar on the last image, channel names in the channel colour above them (NC, NM, eLife, EMBO).

### Data type → graph type (what the papers use; our rules where they differ)
| Data | Graph | Seen in |
|---|---|---|
| A few replicates (n ≤ 20) per condition | bar = mean, plus every replicate as a point; no whiskers or SD whiskers | EMBO 4, NC 1G/I, eLife 2E, NM 3i |
| Many values per condition (cells, fields, crosslinks; n > 20) | **box** (median, IQR, 5–95 % whiskers) with points, or **violin** with a median line; replicate means as large markers (superplot) | bioRxiv 1F/2A, eLife 2F/I/J and 4G, NM 1h/1j/3o |
| Thousands of values per group | violin, no points | bioRxiv 2A/E, eLife 4G |
| Time course or dose | line with points; shaded 95 % band when replicates exist | NC 1E, eLife 2D, eLife 4A/B |
| Parameter sweep (n settings) | line with points on a square plot, honest y axis, reference in grey | our rule (2026-11-18) |
| Survival | step line, control grey | NM 3c/f |
| Composition, fractions of a whole | stacked bar, one shade family per category | NM 3i/k, bioRxiv 2B, NC 1G |
| A distribution of one continuous value, 2+ groups | overlaid filled histograms or densities, control grey, both at 60 % | eLife 4E/F, NM 1b |
| Genome-wide screen, two scores | scatter: grey cloud, hits in accent colour, y = x diagonal, dotted thresholds, labels with leader lines | NC 2F/5F, EMBO 1B, NM 1e |
| Ranked scores | rank plot, grey line of all genes, labelled hits in the accent | NM 3a |
| Matrix (correlation, substitutions, distances) | heatmap with a colour bar or printed values; dendrogram when clustered | NC 5C, bioRxiv 2I/K |
| Enrichment results | dot plot: size = count, colour = p | NM 1f, EMBO 1C |
| Workflow or method | flat schematic (see above) | all 5 |

## Helper: spacr_figstyle.py (2026-11-18)
`/home/olafsson/.claude/skills/spacr-graphs/spacr_figstyle.py`. Use the `spacr` conda env, `python -I`, and `sys.path.insert(0, <skill dir>)`. Do not name a data variable `fs` or `sfs` in the same script; import it as `sfs`.
- `apply_style()`: sets the rcParams (fonts, sizes, boxed axes, outward ticks, 300 dpi, editable SVG text).
- `figure(w, h)`, `cond_axes(fig, left, bottom, n, height)` (width = n × SLOT_IN), `square_axes(fig, left, bottom, size)` for every continuous-axis plot, `make_square(ax)` for an existing one, `heatmap_axes(fig, left, bottom, ncols, nrows, cell_in)` with square cells, and `free_axes(..., kind="other")` only for schematics and images; all in inches.
- `cond_plot(ax, groups, colors, control=i)`: one element per slot. It switches automatically to `box` (or `many="violin"`) above 20 points, applies 60 % alpha, and forces the control to grey. Set the y limits first; points above them become open triangles on the top edge.
- `ci_marker` (whole-sample value with 95 % CI: white circle, black edge), `hist_overlay` (overlaid filled histograms), `bracket` (p value), `group_line`, `panel_letter`, `dev_mark`.
- `check_figure(fig)`: lists overlapping texts, text cut off at the canvas edge, condition axes whose width breaks n × SLOT_IN, continuous-axis plots that are not square, heatmaps whose cells are not square, legend entries named after an identity ("guide", "decoy") drawn in another colour (`IDENTITY`, `GUIDE`, `DECOY`, `GUIDE_LIGHT`), and schematics (`free_axes(..., kind="schematic")`) with more than 10 words. Run it, print it, fix it.

## Checklist before any figure is called done (2026-11-18)
1. Run `sfs.check_figure(fig)`: no overlaps, no clipped text, no width violations. False positives (e.g. a label meant to sit on a line) must be checked by eye.
2. Open every PNG and look at it at 100 %: no overlaps, no clipped labels, no legend over data, nothing cut at the edges.
3. Symmetry: all bars, boxes and violins are the same width with the same gaps; graphs with the same n are the same width; panels in a row share their top and bottom edges; panel letters are aligned.
4. Control or reference is grey in every panel, and named "(reference)" or "(ref.)" in the key.
5. Every data element is at 60 % opacity; text and axes at 100 %.
6. Groups with more than 20 points are boxes or violins; groups with 20 or fewer are bars plus every point.
7. Honest y axis: a line or sweep starts at 0 unless the legend says why not; any cut axis is stated.
8. Palette: only the colours in the table above; no brown; one accent per comparison; legend colours match the plotted colours.
9. Every chart has a control or reference row; every abbreviation is spelled out in the legend; the title is in the legend.
10. Comparators: our run of their code **and** their published value with its source (existing rule).
11. Development figures carry "DEVELOPMENT PANEL – made by a script" and are saved as PNG (300 dpi), SVG and a legend .md. A new version gets a `_vN` suffix and never overwrites the old one.
12. Source data: recompute read-only and check against the stored CSVs; do not rewrite them.

## Axis aspect (maintainer, 2026-11-18)
- A graph takes no more space than it needs. Continuous-axis plots (histograms, density plots, scatter plots, line graphs, sweeps, curves) are SQUARE by default: plot area width = height.
- Only graphs with many categorical conditions along the x axis may be wide, and then the width follows the fixed per-condition slot (spacr_figstyle).
- Heatmaps use square cells.
- Checklist item: before calling a figure done, check that every non-categorical axis box is square.

## Schematics and text (maintainer, 2026-11-18)
- Schematics are small: no larger than one data panel unless they are the main subject of the figure.
- Concepts are ILLUSTRATED, not described: use shapes, arrows, colour and icons. No explanatory sentences inside a schematic; at most a few single-word labels.
- Minimise text everywhere in a figure; explanations go in the legend.
- Checklist item: count the words inside each schematic (aim for under 10) and remove any sentence.

## Colour per identity (maintainer, 2026-11-18)
- One colour per identity across ALL panels and figures: when an entity is the control throughout (for example decoys), it is grey everywhere; the real entity keeps its colour (guides teal) in every panel. "Reference grey" never flips an identity's colour between panels.

## Maintainer rules, 2026-11-18 (round 4)
- Error bars are ALWAYS black.
- Axis labels sit close to the axis (labelpad about 2 pt).
- Space efficiency: minimum font size 8 pt, and the panel size follows from that. Panels are as small as 8 pt text allows; no panel is larger than needed.
- Naming: write "Funk et al." (not "Funk 2022") and "<first author> et al." for PERISCOPE (Ramezani et al.); the same applies to every published method.
- Shared axes: when panels show similar data that will be compared, use the SAME axis limits so differences are not hidden by different scales. FDR (%) axes use (0, 1) unless the data need otherwise.
- No unexplained elements: every visual element must be explained by the legend; remove anything that only repeats an expected value (for example a uniform-reference row in a heatmap; put the expected value in the colour bar or legend instead).
- Schematics: tight bounding boxes, no wasted white space.
- Related panels are combined into one figure rather than spread over several.
- Every analysis choice (threshold, default, rule) must be backed by a figure panel; keep a choice-to-panel register.
- (2026-11-18, round 4b) **A heatmap uses only colours from its own colour bar.** No extra grey or coloured rows or cells. An expected value goes on the colour bar as a labelled tick (e.g. "0.33 (uniform)"). Cells that cannot hold a value (e.g. true = called base for a mismatch) are marked with a thin black "×", never left white or grey (white reads as a value), and the "×" is explained in the legend. When the question is deviation from an expected value, use a diverging map centred on it (`diverging_cmap()` + `centred_norm(vmin, expected, vmax)`: pale magenta below, white at expected, blue above). `check_figure()` flags filled patches inside a heatmap.


# Maintainer rules, 2026-11-19 (round 5, ParaScape paper figures)
These were given after a set of ParaScape panels broke rules that were already written above. They are restated here because each one was violated in a way the earlier wording did not stop.
## Nothing is written on the plotting area
- **Text appears ONLY in an axis title or in a legend.** Not a value above a bar, not a median beside a line, not a baseline label next to its dashes, not a note in a corner, not a count of anything. A schematic may carry short labels (see Schematics); a data panel may not.
- **n is never stated on a graph.** It goes in the figure legend. The same applies to medians, totals, "x of y studies", and any sentence explaining what a line means — a reference line is identified by a **legend entry**, never by text floating beside it.
## Axis titles
- **Strictly objective: the quantity, and its unit.** Name what was measured, not what the measurement is for and not what was done to it.
- **No superfluous words.** These were all rejected: "share of the module", "after removing the cell cycle axis", "contrasts reaching it (of 201)", "right when it answers", "above guessing", "never measured".
- **Give the unit.** A share is `(%)`. A correlation is named: `Spearman (ρ)` or `Spearman (r²)`, not "pair similarity (|Spearman|)". A count needs no unit but needs the noun: "Contrasts".
- An axis whose quantity DIFFERS between categories cannot be given an objective title; that is a sign the panel is mixing two measurements and must be split or relabelled per category.
## Limits
- **No data element may touch or cross the axis limits.** Violins, boxes, error bars and points are padded inside the frame; a violin drawn to the exact data range sits on the spine and reads as data running off the panel.
## Schematics — the point is INFORMATION, not neatness
- A schematic is judged on **what it tells the reader about the system**, not on whether its boxes are the same height. Tidying the geometry of an uninformative schematic does not improve it.
- A capability schematic must **cover what the program can actually do**, not a four-box left-to-right pipeline. If the reader cannot tell from it what questions the program answers, it has failed regardless of how it looks.
- Still true from 2026-11-18: illustrate, do not describe; under ten words; no sentences; tight bounding box; flat shapes, light-grey bodies with thin darker outlines, one or two accents.
## Figures, not panels
- Deliver **figures**. A panel is an ingredient. Assemble the panels, letter them, and present the figure with its legend.
- A figure with a single panel carries no panel letter.
### Wide graphs (Einar, 2026-11-19; REVISED 2026-11-20)
The 2026-11-19 advice here (named dot rows / dumbbells / maps) was REJECTED on 2026-11-20. Use the spaCR software paper figures as the model (github.com/EinarOlafsson/spacr-software-paper, "mostly great"):
- **vertical bars**, slot 0.22 in, grouped by a **group line with its label under the tick labels** (`spacr_figstyle.group_line`), e.g. method pairs under each threshold;
- one lettered panel may hold **two or three small sub-plots side by side**, one per metric, sharing the category axis order, instead of one wide plot;
- **replicate points over bars** when n <= 20; error bars as thin black lines;
- **hatched bars for published / external values** next to values recomputed here;
- a reference is a **dotted line** (`REFLINE`) or a tick, identified in the legend;
- missing values are written in grey **inside the empty slot** ("n/s", "pending", "not run") rather than leaving a gap - this is the one place a short word may sit in the plot;
- a status tag ("PRELIMINARY", "PENDING") in grey at the top right of a panel that is not final.
### Page size (Einar, 2026-11-19)
Paper figures fit an **A4 page, 210 x 297 mm, at true size** (8 pt text stays 8 pt). Pack panels into rows no wider than 210 mm; if the figure is taller than 297 mm, shrink panels (legends below or inside, shorter value axes) or move panels to a supplementary figure. Never scale the composed figure down.
### Round 6 (Einar, 2026-11-20): graph types, whitespace, alignment, narrative
Rejected forms (do not use again without asking):
- **Named dot rows, dumbbells and lollipops** (categories down the left, a dot or a pair of dots per row, IQR whiskers as horizontal lines). Introduced in round 5 to replace wide bar rows; Einar rejected them across figures 1, 4, 6-12 and the supplements. "Wide graphs need a different form" did NOT mean "turn them on their side as dots".
- The labelled-point scatter of figure 6a for supplementary panels (text on points).
- Histograms of counts as a main panel (null distributions, effectors-per-gene) where a simpler statement would do.
- Pastel / low-contrast categorical colour maps on UMAPs (figure 5a).
Required:
- **No dead whitespace; panels aligned**: panels in one row share a height and their axes line up; a figure is not a grid of differently sized cells with gaps.
- **All UMAPs in a figure the same size**, and large enough to read.
- **A prediction must look like a prediction** (distinct marker, label or style from measured values), e.g. candidate drugs in figure 5F.
- **Tell a story**: every figure has one point and panels are ordered to make it.
- **Write the results first.** The results text sets the order of figures and panels (they must be referenced in order) and shows which supporting data are still missing.
- When a panel form has been rejected, draw several candidate forms from the same data and ASK which one before redrawing all panels (memory: offer-plot-options-before-redrawing).
- **Decided 2026-11-20:** "value per category against its reference" = vertical BARS with a black TICK for each category's reference (candidate 1 of the candidate sheet). Distributions of many genes per category stay violins/boxes. Paper figures are numbered in Results order.
### Keep this skill evolving (Einar, 2026-11-20)
"Make sure your skills keep evolving with my feedback." Every round of figure feedback is written here THE SAME TURN it is given, superseded advice is marked REVISED / REJECTED rather than left standing, and the copy in each paper repository (`skills/`) is refreshed with it.
- **Short category names** (<= ~12 characters: "dense gran.", "rhop./mic.", "mito."), full names in the legend text and data.csv. Long tick names widen every panel and break row packing.
- Paper panels share one plot-area height (40 mm) and compose lines up the plot tops of a row; rows are packed by width, not fixed counts.
- A zero or missing value in a bar group is written "none" / "n/s" / "pending" in grey inside its slot (spaCR paper); never a sliver of bar clipped at the log floor.
### Round 7 (Einar, 2026-11-20, after round 6 was implemented)
- **Too many bar graphs.** Round 6 chose bars for "value per category against a reference"; it was applied to nearly every panel. A bar is ONE option. For each panel ask what the reader must see (a difference, a trend, a relationship, a distribution, a composition, a position on a map) and pick the clearest form for THAT; vary forms across a figure as the questions vary.
- **Every figure needs an obvious point; every panel a purpose** in a logical progression. If a reader cannot say what a figure shows without the legend, it is not done. Storyboard first: figure claim -> panel questions in order -> form -> axis titles; then draw.
- **Axis titles and labels use the nomenclature of the field**, taken from the papers the data come from (e.g. the CRISPR screen's "phenotype score", hyperLOPIT's compartment names, "hours post-invasion"), not invented phrases. Look them up and cite where each term comes from.
- **Language on panels must change**: no home-made jargon ("kinds of knowledge", "layers", "wave-built", "lift over guessing", "named module"); use the plain field term.
- **Much less whitespace between panels**; panels sit close, figures read as one composition.
- **A schematic is a FLOWCHART, not text.** Figure 1A was rejected twice (boxes of words; then text lists). It must show the components of the package: the data it takes as input -> what it does with them -> what they are used for, as boxes/icons joined by arrows, few words.
- 2026-11-20: Figure 1A flowchart (inputs -> ParaScape steps -> uses, boxes joined by arrows) accepted as the direction: "much better ... now it just needs to look better". Storyboard in parascape-paper/manuscript/storyboard.md approved; figures 4-5 panels each, method checks to supplements.
- **Approved 2026-11-20 ("figure 1 is so much better now")**: flowchart with three coloured header bars (input -> tool -> uses), equal boxes, grey connectors; then VARIED forms, each chosen for its question: stacked area (datasets over time), presence matrix with a % margin (which genes are measured), histogram (data types per gene), two-row heat strip (secreted vs all). Use this figure as the model for the rest.

## Maintainer rules, 2026-11-20 (living figures)
- **Three tiers per pipeline step,** each kept up to date as results arrive:
  - Main: spaCR vs Brieflow vs PERISCOPE, plus new spaCR features.
  - Supplementary: the data behind every design choice and speed test, intermediate comparisons, and cross runs (Brieflow on PERISCOPE data, PERISCOPE on Funk data).
  - Development: the same panels at the well level, with plate heat maps of the analysed wells and their metrics. Wells without data are marked.
- **Formats:** every figure and every panel is saved as PDF (vector elements wherever possible; raster only for micrographs) AND PNG.
- **Per panel folder (`panel_x/`):**
  - `panel_x.png` and `panel_x.pdf`;
  - `data.csv`: the data the panel is drawn from;
  - `stats.csv`: every test, with n, statistic, p, CI and correction;
  - `source_data/`, or `SOURCE_DATA_LOCATION.txt` when the data is too large.
- **Status:** every legend starts with FINAL, PRELIMINARY, SUSPECT (with the reason) or PENDING (labelled placeholder). Never hide unfinished work.
- **Fonts:** bundle the font files so the light weight never falls back. A missing Open Sans Light once broke the panel layout.
- **No segmentation timing panels** (2026-11-19).
- **Readiness:** a status table must separate code status (general, not per dataset) from "full data analyzed" (per dataset).
- **Showing progress:** when asked how things stand, show the figures themselves (send the PNGs), not only tables.

## Maintainer rules, 2026-11-20 (absolute numbers, representative images)
- **Absolute numbers too:** keep percentages, and add absolute counts next to them, for example cells genotyped, reads, nuclei and calls per well, on a second axis, panel or annotated n.
- **Representative images:** add them wherever they show what the numbers mean. Examples:
  - corner illumination: spaCR captures cells at field corners;
  - objects in stitching seams: spaCR joins nuclei across seams and keeps every nucleus except at the well edge;
  - cell-cycle classes as Brieflow/Funk call them, with their accuracy, next to spaCR's classes and accuracy for each model (DNA only, DNA + tubulin, …), with one image gallery per class.
- **Unfinished results:** still add the image panels, with real examples where they exist and a labelled placeholder otherwise.
- **Image style:** same display range per channel within a panel, fair normalisation, and a scale bar.
- **Confirmed (2026-11-20, "I love your figures"):** the 1590244 build style is approved. That includes the three tiers, status-labelled panels, count axes beside percentages, fixed-range image panels with scale bars and outlines (spaCR teal, Brieflow magenta, method-unique objects amber), and per-class galleries. Keep it.

- 2026-11-20 maintainer: comparator/stock outlines on images are magenta even when the same model is grey in bar charts; say both in the legend.
