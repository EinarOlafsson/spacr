---
name: apicomplexan-figures
description: Build publication figures in the visual idiom of the Lourido lab and comparable Cell / Nature Microbiology / Cell Host & Microbe apicomplexan papers. Use whenever making any figure or panel for a manuscript, poster or grant in parasitology or adjacent cell/molecular biology — before writing the first line of plotting code. Covers the palette, the grey-plus-highlight rule, panel proportions, which plot type fits which data, when a schematic earns its place, and which statistics get drawn versus stated.
---

# Figures in the apicomplexan-genomics idiom

Derived by direct inspection of published figures, not from general design taste:

- **Waldman BS, Schwarz D, Wadsworth MH II, Saeij JP, Shalek AK, Lourido S.** *Identification of
  a master regulator of differentiation in Toxoplasma.* **Cell** 2020. Figures 1 and 3 examined at
  source resolution; complete legends for Figures 1–6.
- **Giuliano CJ, Wei KJ, Harling FM, … Dvorin JD, Lourido S.** *CRISPR-based functional profiling of
  the Toxoplasma gondii genome during acute murine infection.* **Nature Microbiology** 2024. Figure 1
  examined at 1800×1620; legend for Figure 1.

Where the two disagree (axis framing), both are given. Everything else below was consistent.

---

## 1. The one rule that matters most

**Everything is grey except what the sentence is about.**

In Giuliano Fig 1E, ~8,000 genes are plotted in light grey; ~100 are blue (apicoplast-localised) and
~400 orange (delayed phenotype). In Fig 1L, "this study" is blue and the four comparison studies are
grey. Grey is the default ink for data; colour is an argument.

Consequences:
- Never colour by category when the category is not the point. Twelve-colour categorical palettes
  appear only where the categories genuinely are the data (alluvial band plots).
- A highlight set should be a small minority of the marks. If half the points are coloured, the
  figure has no claim.
- Reference lines, thresholds and limits of detection are grey, thin, dashed or dotted — never bold.

## 2. Palette

Sampled from the two figures. Use these; do not invent hues.

```
GREY        #B4B4B4   default data, non-significant, comparison studies
GREY_DARK   #7F7F7F   secondary series, secondary axis, mean bars
INK         #231F20   text, axes, spines, brackets
BLUE        #2E77BC   primary highlight / "this study" / the gene of interest
BLUE_LIGHT  #7FB3E0   second strain (e.g. a knockout)
GREEN       #2E7D4F   wild type, upregulated, tissue-cyst-forming
RUST        #C4441C   downregulated, delayed phenotype, the other highlight
CORAL       #E8A88C   density and histogram fills
GOLD        #E8C33A   third category
OCHRE       #C87A28   fourth category
PURPLE      #8B4A82   complemented strain
NAVY        #1F3F6E   fifth category / non-cutting controls
```

Strain and condition colours are **fixed across every panel of a paper**. In Waldman Fig 3, mock is
grey, WT green, ΔBFD1 light blue and the complement purple in the weight curve, the survival curve
and the cyst dot plot. Assign once, at the top of the figure script, and never re-map.

Sequential encodings (a *p*-value, a score) use a single-hue blue ramp, light→dark. Diverging
colormaps appear only for genuinely signed quantities.

**Opacity.** Scatter marks are opaque with no edge; overplotting is handled by point size and by
greying, not by alpha. The exceptions observed:
- SEM/CI bands around a line: same hue as the line at ~0.25.
- Superplot small points: ~0.5 so the large mean points read on top.
- Density/histogram fills: solid but pale (CORAL), not a translucent saturated colour.

## 3. Proportions and typography

Measured from Giuliano Fig 1 (1800 px ≈ 180 mm double column) and Waldman Fig 1 (single column).

| element | size relative to axis-label text | notes |
|---|---|---|
| panel letter | **1.9–2.2×** | bold, sans-serif, upper-case, no period |
| panel descriptor (optional) | 1.0× | small, centred above axes, lower-case, e.g. `nontarget gRNA` |
| axis label | **1.0×** (the reference) | lower-case, spelled out, no units in brackets unless needed |
| tick label | 0.85–0.9× | |
| in-panel legend / annotation | 0.85× | no frame, no box |
| statistics annotation | 0.9× | |

Absolute anchors that reproduce the look at 300 dpi: axis label 7 pt, tick 6.2 pt, panel letter
13–14 pt bold, annotation 6 pt. Line widths: spines and ticks 0.6–0.7 pt, data lines 1.1–1.4 pt,
reference lines 0.6 pt.

- **Sans-serif throughout** (Helvetica/Arial).
- **Axis labels are lower-case** — `days post infection`, `fraction of shared UMIs`,
  `fold enrichment`, `cysts per brain`. Not Title Case, not sentence case.
- **Gene and protein names italic**, with Δ prefixes and superscripts kept exact:
  *ΔBFD1::BFD1*^WT^-Ty.
- **No panel titles as sentences.** The axis labels carry the content. If a panel needs a
  descriptor it is 2–4 words above the axes.
- **No gridlines. Ever.**
- Long categorical tick labels rotate **45°**, right-aligned.

**Axis framing.** Nature Microbiology figure: full four-spine box, thin. Cell figure: left and
bottom spines only. Pick one per manuscript and hold it. Box framing reads better when panels are
small and dense; L-framing when panels are sparse.

**Secondary axes take the colour of their series** (Giuliano Fig 1G: the evenness axis and its line
are both grey).

## 4. Which plot for which data

Follow this table; it reflects what these papers actually do, including what they refuse to do.

| data | use | never use |
|---|---|---|
| n = 2–8 replicates or animals | **individual points** with a horizontal line at the mean; error bar only if SD/SEM stated | bar chart — a bar for n = 3 is not done in these papers |
| nested / hierarchical (cells within animals, gRNAs within mice) | **superplot**: small semi-transparent points coloured by unit, large opaque points for unit means, black mean ± SEM over the top | pooling everything into one distribution |
| two scores per gene, genome-wide | **scatter, grey, with a highlighted subset**, dotted 1:1 diagonal | colouring all points |
| differential expression | **volcano**, grey / GREEN up / RUST down, dotted threshold lines, a handful of genes labelled | heatmap of everything |
| enrichment across ordered categories | **bubble plot**: size = count, fill = −log₁₀ *p* on a blue ramp, categories sorted by effect | bar chart of *p*-values |
| time course with replicates | **line + same-hue band** at 0.25 alpha; legend as coloured *text*, no markers | error bars at every point |
| survival | **Kaplan–Meier step function**, no band, strain colours | smoothed curve |
| counts per animal spanning orders of magnitude | **log-scale dot plot**, mean line, limit of detection as a thick grey band labelled in-plot | linear axis with the LOD implicit |
| composition across ordered samples | **alluvial / stacked band**, many pale hues, with a grey summary line on a secondary axis | pie chart |
| similarity between many pairs | **dot strip per pair**, mean bar | correlation heatmap without the underlying points |
| relatedness | **dendrogram** with a scale bar in distance units | unlabelled cladogram |
| micrographs | greyscale per channel in a row, **channel name coloured to match its channel** as the column header, row labels italic at left, one white scale bar in a single panel | rainbow LUTs, per-panel scale bars |

Second axis encodings (open vs filled circles) carry a real second variable, explained in the
legend — Waldman Fig 3E uses open for animals sacrificed early, filled for those surviving to week 5.

## 5. When a schematic earns its place

Both papers open with one, and in both cases it does work that no plot can:

- **Waldman Fig 1E** — the screen's logic *and its expected outcome*: a workflow with a small inset
  axis showing where an impaired-differentiation mutant should fall, plus a compact table of library
  composition. A reader can predict the result before seeing it.
- **Giuliano Fig 1A/D** — the construct, then the experiment as numbered steps (circled ①②③), a
  passage timeline with curved recycling arrows, a mouse, and outgrowth durations as horizontal bars.

Rules extracted:
1. A schematic is justified when the **experimental logic is not inferable from the data panels** —
   a construct design, a selection scheme, a sampling timeline.
2. Draw process in **grey arrows**; draw biological material in **colour**.
3. Use **the same colours as the data panels** (Waldman: RFP red, mNG green, Cas9 blue-grey, and
   those channels stay those colours in the micrographs).
4. Number the steps if there is an order.
5. If the expected result has a shape, **show that shape as a mini-axis inside the schematic**.
6. Do not decorate. No shadows, no gradients, no 3-D, no clip-art cells.

A timeline schematic (Waldman Fig 3A) is a plain horizontal axis with ticks, down-arrows at events,
and a grey bar spanning a sampling window. It is one of the cheapest, highest-value panels available.

## 6. Statistics: what is drawn, what is stated

**Drawn on the panel:** a horizontal bracket spanning the compared groups with asterisks or `n.s.`
above it. Nothing else.

**Stated in the legend, always:** the test by name, what n counts, and the asterisk convention. Real
example (Waldman Fig 1B): *"The mean was plotted for n = 3 biological replicates; 92–102 vacuoles
were scored for each replicate; ****p < 0.0001 by Student's two-tailed t test."* Note that it says
**how many vacuoles per replicate** — the unit of replication is never left ambiguous.

Conventions observed:
- `*` <0.05, `**` <0.01, `***` <0.001, `****` <0.0001, and `n.s.` written out.
- Non-significant comparisons are shown, not omitted (Waldman Fig 3E labels `n.s.`).
- Correlations report the coefficient in-panel (`Pearson's r = 0.844`) and the test in the legend.
- Non-parametric tests (Mann–Whitney) for distributions of fold changes; *t* tests for replicate
  means; hypergeometric for category enrichment; BH or explicit adjusted-*p* thresholds for
  genome-wide sets.
- Effect size and n accompany every *p*. A bare *p* is not acceptable.
- Where a comparison is under-powered, the effect is reported and the claim is labelled directional.

## 7. Composition of a figure

- **6–12 panels** for a main figure in these journals; up to ~14 when panels are small and paired.
- Panels are laid out so that **reading order matches the argument**: design → validation → result →
  mechanism. Waldman Fig 1 runs reporter → control → dose-response → transcriptome → screen design →
  screen result at gene level → at gRNA level.
- Related panels share axes and scales, and are placed adjacent.
- Micrograph rows and their quantification sit next to each other, never on separate figures.
- White space between panel groups is larger than within a group; that grouping is the only
  hierarchy cue used.

## 8. Using the companion style module

`apicomplexan_style.py` in this skill directory implements the above. Import it before plotting:

```python
import sys; sys.path.insert(0, "<skill dir>")
from apicomplexan_style import use, C, panel_letter, dots_with_mean, superplot, highlight_scatter, \
    bracket, stat_note, micrograph_row
use(frame="box")          # or frame="L"
```

Provided helpers, each encoding a rule above: `dots_with_mean` (never a bar for small n),
`superplot` (nested data), `highlight_scatter` (grey + highlight), `volcano`, `bubble_enrichment`,
`km_survival`, `line_with_band`, `log_dotplot_with_lod`, `bracket` (statistics annotation),
`panel_letter`, `rotate_ticks`.

## 9. Checklist before saving a figure

1. Is everything grey except the claim? Is the highlight a minority of marks?
2. Are strain/condition colours identical to every other figure in the manuscript?
3. Any bar chart standing in for n ≤ 8? Replace with dots.
4. Is nested data superplotted rather than pooled?
5. Panel letters bold, upper-case, top-left, ~2× the axis-label size, no periods?
6. Axis labels lower-case and spelled out? No gridlines? No sentence titles?
7. Does every drawn statistic have a bracket, and every legend state test + n + convention?
8. Is the unit of replication unambiguous from the legend alone?
9. Does the schematic (if any) show experimental logic that the data panels cannot?
10. Does reading order follow the argument?
