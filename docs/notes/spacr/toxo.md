# Notes from `spacr/toxo.py`

Prose lifted out of `spacr/toxo.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [custom_volcano_plot](#custom_volcano_plot) (13 entries)
- [_fit_outside_legend](#_fit_outside_legend) (2 entries)
- [go_term_enrichment_by_column](#go_term_enrichment_by_column) (24 entries)
- [plot_gene_phenotypes.extract_gene_id](#plot_gene_phenotypesextract_gene_id) (1 entry)
- [plot_gene_phenotypes](#plot_gene_phenotypes) (12 entries)
- [plot_gene_heatmaps.extract_gene_id](#plot_gene_heatmapsextract_gene_id) (1 entry)
- [plot_gene_heatmaps](#plot_gene_heatmaps) (10 entries)

## Module level

### line 24, trailing  _(unsure)_

```python
from . import tabular
```

one reader: spacr.tabular is the funnel

### lines 25-30

```python
from .plot import save_figure  # noqa: F401
```

The module-scope routing contract: every figure this module keeps reaches the format and DPI preferences through `spacr.plot.save_figure`. NOTHING HERE CALLS IT DIRECTLY -- a save goes through `_write_the_figure` -> `figures.scene.write_figure`, which draws the scene the screen would show and falls back to this helper -- so spying on THIS name sees no call while the file is written exactly as asked.

## custom_volcano_plot

### lines 211-215

```python
metadata = metadata_path.copy()
```

.copy() for the same reason `data` above takes one: the next line rewrites 'gene_nr' to str in place, and without the copy that edit landed in the CALLER's frame. A caller that plots two volcanoes from one metadata table got its integer gene numbers silently retyped by the first call.

### lines 223-232

```python
try:
```

many_to_one: `data` holds one row per regression *feature*, and several features share a gene -- a gRNA-level fit contributes one row per guide, so gene_nr repeats on the left by design. `metadata` is a lookup table: one localisation per gene, which is what the shipped resources/data/lopit.csv is (3832 rows, 3832 distinct gene_nr). A duplicated gene_nr on the right is therefore not a legitimate shape here, it is a fan-out: every affected gene gets plotted twice and appended to the returned hit list twice, which then propagates into plot_gene_phenotypes and plot_gene_heatmaps as duplicate genes. Declaring the relationship turns that into a stop rather than a wrong figure.

### lines 245-247

```python
raise
```

MergeError also covers things this message would misdescribe a colliding suffix, for one. Only claim the cardinality story when the duplicates that would justify it are actually there.

### lines 261-264

```python
called = ((merged_data['p_value'] <= 0.05)
```

ONE RULE, ONE PLACE. The hit list and the colouring below read the same mask, so a gene the volcano marks and a gene the phenotype plot reports cannot disagree -- they used to be a vectorised expression and a row-by-row `if` written separately.

### lines 276-278

```python
with _house(figsize) as (ink, scale):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS: rcParams colour an artist when it is CREATED, so a context opened after plt.subplots would leave the spines, ticks and text at the caller's global style.

### lines 296-297

```python
on_upper = (neg_log_p > upper_lim[0]) if is_broken \
```

Which panel each point belongs to, decided once for the whole column rather than by a function call per row.

### lines 306-309

```python
layers = [(~called_mask, ROLES['data'], 1, None),
```

ONE SCATTER CALL PER ARGUMENT, not one per point. The old loop ran `ax.scatter` once for each of the ~3,800 rows and then a second full pass to build the hit list; both are gone. zorder puts the claim on top of the grey rather than leaving it to row order.

### lines 325-326  _(unsure)_

```python
axis.scatter(coefficient[take], neg_log_p[take], color=colour,
```

Opaque, no edge: the published figures handle overplotting with size and with grey, not with alpha.

### lines 342-343

```python
ax_upper.spines['bottom'].set_visible(False)
```

The break itself: the two panels face each other across a gap, so the spines that would draw a line through it come off.

### lines 346-347  _(unsure)_

```python
for axis in all_axes:
```

Threshold lines

Grey, thin, dashed, behind the data. A reference is not a result.

### lines 353-354

```python
reference_line(axis, x=0.0)
```

threshold=0 means "no coefficient cut": the two lines would land on top of each other, so draw the zero once.

### lines 384-389

```python
if entries:
```

Legend

THREE LINES OF TEXT, INSIDE THE PANEL. It was 27 framed swatches anchored outside the axes, which is why `_fit_outside_legend` had to exist and why the data was squeezed into a strip beside it. A legend is an index; with grey as the default ink there are only the called directions to index.

### lines 394-396

```python
save_path = _write_the_figure(fig, save_path,
```

Saved INSIDE the context: savefig.transparent and savefig.facecolor are read at write time, so a save outside it would put the default white ground back under the figure.

## _fit_outside_legend

### line 464

```python
fig.subplots_adjust(right=max(min(right, 0.98), min_axes_width))
```

A legend wider than the figure would invert the axes; clamp instead.

### line 466, trailing

```python
except Exception:
```

layout is never worth an exception

## go_term_enrichment_by_column

### line 530  _(unsure)_

```python
metadata = tabular.read_table(metadata_path, report=None)
```

Load metadata

### line 535  _(unsure)_

```python
hits_metadata = metadata.loc[
```

Create a subset of metadata with only the rows that contain genes in gene_list (hits)

### line 539  _(unsure)_

```python
combined_results = []
```

Create a list to hold results from all columns

### line 543  _(unsure)_

```python
go_terms = []
```

Initialize lists to store results

### line 555  _(unsure)_

```python
all_go_term_counts = all_go_terms.value_counts()
```

Count occurrences of each GO term in hits and total metadata

### line 559  _(unsure)_

```python
for go_term in all_go_term_counts.index:
```

Perform enrichment analysis for each GO term

### line 564  _(unsure)_

```python
total_genes = len(metadata)
```

Calculate the total number of genes and hits

### line 568  _(unsure)_

```python
contingency_table = [[hits_with_go_term, total_hits - hits_with_go_term],
```

Perform Fisher's exact test

### line 580  _(unsure)_

```python
if enrichment_score > 0.0:
```

Store the results only if enrichment score is non-zero

### line 586  _(unsure)_

```python
results_df = pd.DataFrame({
```

Create a results DataFrame for this GO term column

### line 591, trailing  _(unsure)_

```python
'GO Column': go_term_column
```

Track the GO term column for final combined plot

### line 597  _(unsure)_

```python
combined_results.append(results_df)
```

Append this DataFrame to the combined list

### lines 600-609

```python
with _house(10) as (ink, scale):
```

Plot the enrichment results for each individual column

GREY, WITH THE CALLED TERMS COLOURED. `hue='GO Term'` gave every term of the ontology its own hue and its own legend row -- the 27-colour failure again, and worse here, because a GO column carries hundreds of terms and the legend was anchored outside the axes. The sentence is "these terms are enriched among the hits and the enrichment is significant", so significance is what carries colour, and the terms that carry it are named on the points instead. ONE scatter call, in the frame's own row order, so the points are the same points in the same order as before.

### lines 616-617

```python
ax.scatter(enrichment, significance, s=_scaled_sizes(enrichment),
```

Size still reads the effect, as `sizes=(50, 200)` did; only the hue moved.

### lines 623-624  _(unsure)_

```python
reference_line(ax, x=1.0)
```

Enrichment of 1 is "as common among the hits as in the background", which is the null this panel is read against.

### line 627  _(unsure)_

```python
ax.set_title(f'GO Term Enrichment Analysis for {go_term_column}')
```

Set plot labels and title

### lines 632-633

```python
texts = [ax.text(enrichment[i], significance[i],
```

The terms that cleared p <= 0.05 are named on the panel, which is what the every-term legend was there to do and could not.

### line 649  _(unsure)_

```python
print(f'Results for {go_term_column}')
```

Optionally return or save the results for each column

### line 652  _(unsure)_

```python
combined_df = pd.concat(combined_results)
```

Combine results from all columns into a single DataFrame

### line 655  _(unsure)_

```python
with _house(12) as (ink, scale):
```

Plot the combined results with text labels

### lines 665-668

```python
for index, column in enumerate(dict.fromkeys(combined_df['GO Column'])):
```

WHICH ONTOLOGY A TERM CAME FROM IS A REAL SECOND VARIABLE, so it keeps its encoding -- as marker shape, which is what `style='GO Column'` already used and what the style spends no colour on. One collection per shape, in frame order within each.

### line 679  _(unsure)_

```python
ax.set_title('Combined GO Term Enrichment Analysis')
```

Set plot labels and title for the combined graph

### line 684  _(unsure)_

```python
texts = [ax.text(enrichment[i], significance[i],
```

Annotate the points with labels and connecting lines

### line 690

```python
adjust_text(texts, ax=ax, arrowprops=_LEADER)
```

Adjust text to avoid overlap

## plot_gene_phenotypes.extract_gene_id

### line 721  _(unsure)_

```python
def extract_gene_id(gene):
```

Ensure x_column is properly processed

## plot_gene_phenotypes

### lines 728-730

```python
data = data.copy()
```

The caller's table is not ours to retype. `data.loc[:, col] = ...` below writes through to whatever frame was passed, and `ml.perform_regression` passes the GT1 metadata table it read once and uses again.

### line 744  _(unsure)_

```python
x = data['rank']
```

Prepare the x, y, and error values for plotting

### line 749  _(unsure)_

```python
with _house(10) as (ink, scale):
```

Create the plot

### lines 753-755

```python
plt.plot(x, y, label='Mean Phenotype', color=Palette.GREY_DARK,
```

Plot the mean phenotype with standard error shading. The band takes the line's own hue at 0.25, which is the only alpha the published figures use on a curve.

### line 764  _(unsure)_

```python
texts = []  # Store text objects for adjustment
```

Prepare for adjustText

### line 765, trailing  _(unsure)_

```python
texts = []
```

Store text objects for adjustment

### line 779, trailing  _(unsure)_

```python
zorder=3
```

Ensure the points are on top

### line 781  _(unsure)_

```python
texts.append(
```

Add the text label next to the highlighted gene

### line 793

```python
adjust_text(texts, arrowprops=_LEADER)
```

Adjust text to avoid overlap with lines drawn from points to text

### line 796  _(unsure)_

```python
plt.xlabel('Rank')
```

Label the plot

### line 799, trailing  _(unsure)_

```python
plt.legend().remove()
```

Remove the legend if not needed

### line 802  _(unsure)_

```python
if save_path:
```

Save the plot if a path is provided

## plot_gene_heatmaps.extract_gene_id

### line 831  _(unsure)_

```python
def extract_gene_id(gene):
```

Ensure x_column is properly processed

## plot_gene_heatmaps

### lines 838-839

```python
data = data.copy()
```

`data['x'] = ...` is a new column on the caller's table otherwise, and `ml.perform_regression` reuses the ME49 frame it passes here.

### line 843  _(unsure)_

```python
filtered_data = data[data['x'].isin(gene_list)].set_index('x')[columns]
```

Filter the data to only include the specified genes

### line 846  _(unsure)_

```python
if normalize:
```

Normalize each gene's values between 0 and 1 if normalize=True

### line 850  _(unsure)_

```python
width = len(columns) * 4
```

Define the figure size dynamically based on the number of genes and columns

### line 854  _(unsure)_

```python
with _house(width, frame='box') as (ink, scale):
```

Create the heatmap

### lines 859-861

```python
ax = sns.heatmap(
```

Plot the heatmap with genes on the y-axis and columns on the x-axis linewidths=0: the white rules between cells were a grid, and the rule the style states is no gridlines ever.

### lines 871-872

```python
rotate_ticks(ax, 45)
```

Long column names rotate 45 and anchor right, as every categorical axis in the style does.

### line 874, trailing  _(unsure)_

```python
plt.yticks(rotation=0)
```

Keep y-axis labels horizontal

### line 882  _(unsure)_

```python
plt.tight_layout()
```

Adjust layout to ensure the plot fits well

### line 885  _(unsure)_

```python
if save_path:
```

Save the plot if a path is provided
