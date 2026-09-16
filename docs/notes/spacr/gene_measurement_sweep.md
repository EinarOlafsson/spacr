# Notes from `spacr/gene_measurement_sweep.py`

Prose lifted out of `spacr/gene_measurement_sweep.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [measurement_columns](#measurement_columns) (1 entry)
- [gene_of_guide](#gene_of_guide) (2 entries)
- [sweep](#sweep) (12 entries)
- [HOUSE](#house) (13 entries)
- [_write](#_write) (3 entries)
- [_readable](#_readable) (4 entries)
- [plot_sweep](#plot_sweep) (3 entries)
- [plot_effect_against_representation](#plot_effect_against_representation) (6 entries)
- [plot_measurement_families](#plot_measurement_families) (2 entries)
- [plot_guide_concordance](#plot_guide_concordance) (5 entries)
- [plot_grid_volcano](#plot_grid_volcano) (6 entries)
- [plot_gene_profile](#plot_gene_profile) (3 entries)
- [plot_gene_similarity](#plot_gene_similarity) (1 entry)
- [plot_measurement_hits](#plot_measurement_hits) (3 entries)
- [plot_circularity](#plot_circularity) (2 entries)
- [plot_calibration](#plot_calibration) (2 entries)

## Module level

### lines 38-41

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE, AT MODULE SCOPE. `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time which is why every `plt` in this file is still imported lazily and this is not.

## measurement_columns

### lines 102-104

```python
usable = [c for c in named
```

RANK ONCE, THEN ONE MATMUL -- not a correlation per pair. The pairwise version was 715 x 70 = 50,000 spearman calls and did not finish in two minutes, which is the very thing this module's docstring warns about.

## gene_of_guide

### lines 258-262

```python
if len(parts) >= 2:
```

THE ORGANISM IS ALREADY GONE, so the shape rule must not take a second component off. What is left is `<gene>_<guide>`, and a gene id that carries an underscore of its own -- `ROP18_kinase` -- keeps all of it. Applying the three-component rule here as well dropped the gene's first component and split its guides across two "genes".

### lines 268-269  _(unsure)_

```python
gene = "_".join(parts[1:-1]).strip()
```

`<organism>_<gene>_<guide>`: the gene is the middle component, and anything between it and the guide number belongs to it.

## sweep

### lines 372-374

```python
fractions = pd.concat(
```

Suffixed so a gene and one of its guides cannot collide in the index -- `233460` the gene and `233460_1` the guide are different rows and a reader must be able to tell which is which.

### lines 390-392

```python
if drop_measurements:
```

NAMED EXCLUSIONS, applied after the automatic ones. A user who knows a column is wrong -- a stale plate id, a measurement they no longer trust -- should not have to enumerate the seven hundred that are fine.

### lines 403-405

```python
excluded: Dict[str, Tuple[str, ...]] = {}
```

THE THREE GUIDE FILTERS, and each is recorded rather than silent: a sweep that quietly dropped a gene the user was looking for would send them hunting through the table for a row that was never computed.

### lines 451-452

```python
R = (F.T @ M) / max(n - n_blocks, 1)
```

ONE MATMUL. Every guide against every measurement, as a correlation after the block means are gone.

### lines 456-467

```python
sq = F * F
```

THE DEGREES OF FREEDOM ARE THE GUIDE'S, NOT THE SCREEN'S.

A guide present in 7 of 1,366 wells has a fraction vector that is zero almost everywhere, and its correlation is carried by those 7 points. Using n - blocks - 1 for it reported p = 0.0 from seven wells, which is the most confident possible statement of almost nothing, and it sat at the top of the table.

The participation ratio (sum x^2)^2 / sum x^4 is the effective number of wells actually carrying a predictor: it equals n for a dense one and collapses to the count of non-zero wells for a sparse one. Cheap -- two column sums -- and conservative in the direction that matters.

### lines 473-475

```python
presence = (fractions[guides] > 0)
```

Representation, reported rather than corrected for: the right response to a gene that is everywhere is to SEE that it is, not to have its numbers quietly adjusted.

### lines 485-487

```python
circular = np.full(len(chosen), np.nan)
```

NaN, NOT ZERO, when it was never computed. A column of 0.00 reads as "nothing here is circular", which is the most confident possible way to say nothing -- and it is what the panel displayed before this line.

### line 491  _(unsure)_

```python
s = pd.Series(np.asarray(list(scores), dtype=float), index=common).rank()
```

Ranked once and correlated as a matrix, for the same reason.

### lines 500-505

```python
overlap = int(pd.Series(np.asarray(list(scores), dtype=float),
```

A SCORE THAT JOINED TO NOTHING MUST NOT READ AS "NOT CIRCULAR". The score CSVs of a real screen carry `pplate1` where the measurement databases carry `plate1`, so an un-canonicalised join matches no well at all -- and the resulting all-NaN column was reported as "0 of 5,959 hits are circular", which is the most confident possible way to say nothing.

### lines 512-515

```python
effects = pd.DataFrame(
```

THE SUFFIX GOES FROM BOTH, or the table and the grid disagree about what a row is called and `plot_sweep` looks a gene up under a name only the table uses. It exists only to keep a gene and its guides apart while they are concatenated, and a gene id never equals a guide id anyway.

### lines 528-534

```python
"share": np.repeat(share_of, len(chosen)),
```

HOW MUCH OF THE SCREEN THIS GENE IS. Measured on the maintainer's own: 220950 sits in ALL 1,536 wells at a median fraction of 0.176 17.6% of every well -- while the median gene is in 73. With that many wells a partial correlation of 0.396 is overwhelming, and a 73-well gene needs a far larger effect to clear the same bar. So ranking by q ranks by REPRESENTATION as much as by biology, and a reader cannot see that unless it is on the row.

### lines 542-543  _(unsure)_

```python
"effective_wells": np.repeat(np.round(n_eff, 1), len(chosen)),
```

What the P VALUE was actually computed on, which is not the same number and is the one a reader needs to judge it.

## HOUSE

### line 579, trailing  _(unsure)_

```python
GREY = "#B4B4B4"
```

default data, non-significant

### line 580, trailing  _(unsure)_

```python
GREY_DARK = "#7F7F7F"
```

secondary series, mean bars

### line 581, trailing  _(unsure)_

```python
BLUE = "#2E77BC"
```

the primary highlight / the gene of interest

### line 582, trailing  _(unsure)_

```python
BLUE_LIGHT = "#7FB3E0"
```

a second series beside the first

### line 583, trailing  _(unsure)_

```python
GREEN = "#2E7D4F"
```

up

### line 584, trailing  _(unsure)_

```python
RUST = "#C4441C"
```

down / the other highlight

### line 585, trailing  _(unsure)_

```python
CORAL = "#E8A88C"
```

density and histogram fills

### line 586, trailing  _(unsure)_

```python
GOLD = "#E8C33A"
```

third category

### line 587, trailing  _(unsure)_

```python
OCHRE = "#C87A28"
```

fourth category

### line 588, trailing  _(unsure)_

```python
PURPLE = "#8B4A82"
```

fifth category

### line 589, trailing  _(unsure)_

```python
NAVY = "#1F3F6E"
```

sixth category / controls

### line 590, trailing  _(unsure)_

```python
SEQ = "Blues"
```

single-hue ramp for a p-value or a score

### line 591, trailing  _(unsure)_

```python
DIVERGING = "RdBu_r"
```

ONLY for a genuinely signed quantity

## _write

### line 635, trailing  _(unsure)_

```python
except Exception:
```

style absent

### lines 646-652

```python
for axes in figure.axes:
```

AND EACH AXES' OWN GROUND. The figure patch is the margin; the axes patch is the PAGE the data sits on, and on a heatmap it is very nearly the whole image. Flipping only the figure was invisible while rcParams were matplotlib's defaults -- the axes were already white -- and showed up the moment this module drew inside the house style, where the screen palette colours them dark. Measured: `plot_sweep` saved onto (141, 12, 37).

### lines 670-675

```python
figure.savefig(path, dpi=200, bbox_inches="tight",
```

NOT `plot.save_figure`, and deliberately (108 point 6). This function IS the export rule for the sweep's figures: it reads `saved_figure_appearance` itself and has already flipped the figure's ground, each axes' ground and every piece of chrome above. Routing it through the shared writer would apply the same repaint a second time, on artists this function is holding the undo for.

## _readable

### line 741, trailing  _(unsure)_

```python
except Exception:
```

style absent

### lines 743-745

```python
try:
```

TRANSPARENT, not a colour of our own: the page the figure lands on is the application's, and painting white behind it is what makes a dark theme look broken.

### line 748, trailing  _(unsure)_

```python
except Exception:
```

defensive

### line 774, trailing  _(unsure)_

```python
except Exception:
```

defensive

## plot_sweep

### lines 802-806

```python
drawn = str(level or "").strip().lower()
```

ONE LEVEL PER PICTURE. At `level='both'` the table holds a gene row and a row for each of its guides, and drawn together they are the same effect counted several times -- a block of near-identical rows that reads as agreement between independent things. Genes by default, because that is the question the sweep is usually asked.

### lines 824-827

```python
grid = _order_like_neighbours(grid)
```

ORDERED SO NEIGHBOURS ARE ALIKE. A heatmap whose rows are in the order they happened to arrive hides every block structure in it; the measurements of one compartment belong together and a reader looking for "what kind of thing does this gene move" is looking for exactly that.

### lines 832-835

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## plot_effect_against_representation

### lines 967-970

```python
return None
```

NOTHING, rather than every gene on a flat line at zero. That picture is not false -- no gene passed -- but it reads as a measured absence of effect when it is an absence of evidence, and the two are the thing this whole module tries to keep apart.

### lines 983-986

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 991-993

```python
axes.scatter(rest["weight"], rest["hits"], s=9, color=HOUSE.GREY,
```

OPAQUE, NO EDGE. The skill: overplotting is handled by point size and by greying, not by alpha -- a translucent mark makes density and value the same channel.

### lines 997-999

```python
axes.scatter(controls["weight"], controls["hits"], s=22,
```

THE CONTROLS ARE THE OTHER HIGHLIGHT, opaque and small like every other mark. A hollow diamond at s=54 read as an annotation rather than as data.

### lines 1003-1005

```python
if len(per_gene) >= 8 and per_gene["weight"].nunique() > 2:
```

THE TREND, and only when there is something to fit. Two points make a line through themselves and say nothing; drawing it anyway would put a confident diagonal on a plot that has no evidence for one.

### lines 1013-1014

```python
rho = float(np.corrcoef(x, y)[0, 1]) if len(set(x)) > 1 else np.nan
```

Named on the plot, because the number is the answer to the question and a reader should not have to eyeball the slope.

## plot_measurement_families

### lines 1079-1082

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1087-1090

```python
for family in families:
```

FAMILY COLOURS ASSIGNED ONCE AND NEVER RE-MAPPED, the rule Waldman Fig 3 keeps for strains across three different panels. Here the categories genuinely ARE the data, which is the one case the skill allows a categorical palette.

## plot_guide_concordance

### line 1143  _(unsure)_

```python
per_gene_guides = table.groupby("gene")["guide"].nunique()
```

A gene needs TWO guides to agree or disagree about anything.

### lines 1170-1178

```python
with figure_style(theme_target()):
```

DOTS, NEVER A BAR. A gene has two to four guides, and the skill is explicit: "n = 2-8 replicates ... individual points with a horizontal line at the mean; NEVER a bar chart -- a bar for n = 3 is not done in these papers". The old bar hid exactly what this panel exists to show: whether the guides agree, or whether one of them carries the gene. THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1186-1187

```python
spread = rng.uniform(-0.13, 0.13, len(values))
```

Jitter is deterministic: a figure that moves its points between two renders of the same data is a figure a reader cannot check.

### lines 1192-1193

```python
colour = (HOUSE.BLUE if mean >= 0.99 else
```

EVERYTHING GREY EXCEPT THE CLAIM: the mean is coloured only where it says something -- complete agreement, or a real split.

### lines 1208-1209

```python
axes.text(0.02, 0.02, "one point per measurement · line is the mean",
```

A LEGEND AS COLOURED TEXT, no frame and no markers -- the Waldman Fig 3B/C idiom, which costs no space and needs no key to decode.

## plot_grid_volcano

### lines 1235-1241

```python
def plot_grid_volcano(result: "SweepResult", path: Optional[str] = None, *,
```

The other six views

Ten ways of looking at one grid, each answering a question the others cannot. The list is kept HERE rather than in a message, because the first four were built from a conversation and the other six nearly were not.

### lines 1265-1267

```python
effect_values = pd.to_numeric(keep["effect"], errors="coerce")
```

COERCED, not assumed numeric: an empty frame built from a column list carries object dtype, and `np.isfinite` on that raises a TypeError rather than returning an empty mask.

### lines 1277-1285

```python
with figure_style(theme_target()):
```

GREY / GREEN UP / RUST DOWN, the skill's volcano exactly. Colour is an ARGUMENT here: the grey is every pair tested and the coloured minority is the claim. Circularity is not a colour ramp over all of them any more -- a sequential ramp over 900 grey points is a texture, and it spent the one channel that could carry the finding. THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1298-1300

```python
if result.circularity_known:
```

A CIRCULAR HIT IS RINGED, NOT RECOLOURED. It is still a hit; what the ring says is that the classifier already tracks that measurement, so it cannot corroborate anything derived from the classifier.

### lines 1311-1313

```python
cut = float(keep.loc[passed, "p"].max())
```

The BH line falls where the correction actually landed, not at a nominal 0.05 -- drawing the nominal one puts the threshold in the wrong place on every corrected screen.

### lines 1318-1320

```python
if passed.any():
```

A HANDFUL LABELLED, not all of them: the skill labels "a handful of genes" on a volcano and nothing else, because a label on every hit is a wall of text with no claim in it.

## plot_gene_profile

### lines 1372-1374

```python
shown = passed if len(passed) else mine
```

Fall back to the strongest effects when nothing cleared the correction: "this gene has no significant measurement" is worth SEEING, and an empty axis does not say it.

### lines 1381-1385

```python
families = [measurement_family(m) for m in shown["measurement"]]
```

GREY EXCEPT THE CLAIM. This drew every bar in a tab10 family colour, which spends the colour channel on a grouping the reader can already see in the labels and leaves nothing to say which effects are real. Significance is the claim here; family is context, and it goes on the tick labels.

### lines 1393-1396

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## plot_gene_similarity

### lines 1467-1470

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## plot_measurement_hits

### lines 1515-1519

```python
share = counts.to_numpy(dtype=float) / max(total, 1)
```

A BUBBLE PLOT, which is what the skill prescribes for "enrichment across ordered categories": size = count, fill = the evidence on a single-hue ramp, categories sorted by effect. The bar chart it replaces carried ONE number per measurement; this carries three in the same space -- how many genes move it, how strongly, and how sure.

### lines 1526-1529

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1538-1540

```python
loud = share >= 0.5
```

PROMISCUOUS MEASUREMENTS ARE RINGED, not recoloured: a measurement half the library moves is a plate effect wearing a measurement's name, and it will put a hit on every gene in the screen.

## plot_circularity

### lines 1584-1586

```python
return None
```

NOT AN EMPTY AXIS. The column is NaN, and a scatter of NaN is a blank panel that reads as "nothing is circular" -- which is the exact misreading this whole column exists to prevent.

### lines 1597-1600

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

## plot_calibration

### lines 1655-1658

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 1667-1669

```python
from scipy.stats import chi2
```

THE INFLATION FACTOR, named. A number beats an eyeballed slope, and this one has a standard meaning: lambda near 1 is calibrated, and well above it means something systematic is inflating every test.
