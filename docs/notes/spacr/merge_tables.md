# Notes from `spacr/merge_tables.py`

Prose lifted out of `spacr/merge_tables.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (5 entries)
- [roll_up](#roll_up) (3 entries)
- [_columns_agree](#_columns_agree) (1 entry)
- [reconcile_duplicates](#reconcile_duplicates) (2 entries)
- [merge_tables](#merge_tables) (6 entries)
- [_apply_na_policy](#_apply_na_policy) (2 entries)
- [reduce_dimensions](#reduce_dimensions) (6 entries)
- [group_variance_share](#group_variance_share) (1 entry)

## Module level

### lines 66-72

```python
(r"(^|_)(object_label|label|id|cell_id|nucleus_id|pathogen_id|"
```

Identity first, because it must beat every rule below it. A label is a NAME, not a quantity: averaging the three pathogen labels 1, 2 and 3 in a cell produces `object_label_pathogen` = 2.0, which looks like a measurement, plots like one, and can be handed to a model as a feature while naming an object that may not exist. The parent already learns how many children it had from the count column, so the label is carried only so a row can be traced back, and it is carried verbatim.

### lines 80-84

```python
(r"(^|_)(area|volume|convex_area|filled_area)(_|$)", SUM),
```

Extent: four objects' AREAS add up. Their LENGTHS do not -- two nuclei each 10 units long are not one nucleus 20 units long, so an axis length, a perimeter and an equivalent diameter are shape descriptors of an individual object and the parent gets the typical one. Volume adds for the same reason area does. (Maintainer's call, 2026-08-11.)

### line 88  _(unsure)_

```python
(r"(^|_)(integrated|total|sum|integral)(_|$)", SUM),
```

Anything already integrated over an object is a total.

### lines 90-91  _(unsure)_

```python
(r"(^|_)(std|stdev|var|variance|mad|iqr|percentile|quantile|"
```

Spread and shape are properties of each object; the parent gets the typical one.

### lines 745-747  _(unsure)_

```python
REDUCTIONS: Tuple[str, ...] = ("pca", "umap", "tsne")
```

Dimensional reduction -- what xD means

## roll_up

### lines 344-345  _(unsure)_

```python
out["count"] = grouped.size()
```

The count is the one measurement the child table does not carry and the parent almost always wants: "how many pathogens are in this cell".

### lines 348-360

```python
measured_columns = [c for c in plan
```

HOW MANY CHILDREN ACTUALLY CONTRIBUTED, which is not always `count`.

pandas skips NaN in every aggregation and says nothing. A cell with three pathogens, one of whose area could not be measured, reports `pathogen_area` as the sum of TWO while `pathogen_count` says three measured: areas [10, NaN, 30] give 40.0 with a count of 3. The sum is not wrong so much as answering a question nobody asked, and the two columns disagree with nothing to reveal it.

`measured` is the smallest number of non-null values behind any measurement in the row. `measured < count` is the flag: some child contributed nothing to at least one column. Downstream can test it, and the log below names the columns so a user does not have to.

### lines 383-387

```python
renamed = {c: f"{name}_{c}" for c in out.columns
```

Measure already writes its columns prefixed -- the nucleus table holds `nucleus_area`, not `area` -- so prefixing unconditionally would hand the user `nucleus_nucleus_area` for a measurement they know by another name. Prefix only what needs it, matching the one-row-per-cell branch in `merge_tables` so a column has ONE name whichever way it was joined.

## _columns_agree

### lines 475-476

```python
same = pd.Series(
```

Two tables can reach the same number by different arithmetic, so float noise is not a conflict; a real disagreement is never 1e-9.

## reconcile_duplicates

### line 523, trailing  _(unsure)_

```python
continue
```

a genuinely new column that happens to end so

### lines 530-531  _(unsure)_

```python
continue
```

Two measurements that happen to share a name. They describe different objects, so they differ by design and both are kept.

## merge_tables

### lines 580-582

```python
if PNG_TABLE in tables and PNG_TABLE not in wanted and PNG_TABLE in available:
```

png_list holds one row per CROP, not per object, and its object key is text. It is merged like any child -- the keys are reconciled below but it has no measurements to aggregate, so it contributes its paths.

### lines 604-616

```python
anchor = anchor_column(table) if table in ANCHOR_COLUMN else PARENT_LINK
```

THE ANCHOR HAS TWO NAMES. cell and cytoplasm carry it as

`object_label` -- a cytoplasm is the cell minus its interior objects, so its own label IS the cell's -- while nucleus, pathogen, organelle and png_list carry the parent's label in `cell_id`.

This assumed `cell_id` for every non-primary table, so CYTOPLASM WAS SILENTLY DROPPED: it logged a line about an unlinkable table and returned a frame with no cytoplasm columns at all.

One row per cell also means no roll-up. Aggregating a table that already has one row per cell is not wrong so much as meaningless, and it would put the cytoplasm's own measurements through the sum/mean rules meant for a group of children.

### lines 619-621

```python
LOG.info("%s carries no %s, so it cannot be joined onto %s; "
```

Measured without a parent mask: the roll-up is not empty, it is UNDEFINED. Named and skipped, exactly as io.py does -- one unlinkable table must not cost the user the others.

### lines 627-631

```python
skip = set(_keys_in(child)) | {anchor, "prcf", "prcfo"}
```

Prefixed like the primary table above, with two exceptions that would otherwise produce nonsense: the join keys keep their names, and a column that ALREADY carries the table's name is left alone -- `cytoplasm_area` must not become `cytoplasm_cytoplasm_area`.

### lines 646-651

```python
how = policy.how_for(table)
```

`how="left"` used to be hard-coded here, so this reader and

`io._read_and_join_tables` -- which has always called `join_how` disagreed about which objects exist. Two readers of the same tables giving different populations is the defect instruction 77 item (c) fixed for `_merge_grouped`; this was the same bug in the third reader. See `MergePolicy.how_for` for what decides it.

### lines 656-658

```python
LOG.info(
```

Never silent. An inner join is a filter, and a filter that removes a third of the population without saying so is how a result gets reported for a subgroup nobody chose.

## _apply_na_policy

### lines 719-735

```python
if not counts:
```

DROPPED FOR HAVING NO CHILD, not for having an unmeasurable one.

This used to drop on every column with an underscore in its name i.e. every measurement the child contributed. Measured on three cells: one with a pathogen and a good correlation, one WITH A PATHOGEN whose correlation came back NaN, and one with no pathogen at all. Only the first survived. The second has a pathogen; it is a unit of analysis; its area and count are real numbers. It was removed from the denominator because one correlation could not be computed -- and a correlation is NaN whenever a channel is flat inside the object, which is common and says nothing about whether the object exists.

The count is the column that answers "does this cell have one", which is the question this policy is documented to ask. Roll-up counts are NaN exactly when the child table contributed no row, so they are the correct and only subset.

### lines 737-739

```python
return frame.reset_index(drop=True)
```

Nothing was rolled up, so there is no childlessness to test. Dropping on the measurements here would silently narrow the population on a merge that has no children in it at all.

## reduce_dimensions

### lines 804-812

```python
coverage = data.notna().mean()
```

DROPPING every row with any missing value does not work on a real measurement table. With 60 columns at 2% missing each, two rows in three are lost; with the several hundred columns spaCR actually writes, none survive -- which is why xD looked like it had never been implemented.

So: drop the columns that are mostly empty, then fill what is left with the column median. A median fill moves an object to the middle of an axis it had no value on, which is the least it can be moved; discarding the object instead loses every measurement it DID have.

### line 828

```python
usable = usable.dropna(axis=1, how="any")
```

A column that is entirely NaN has no median; it cannot contribute.

### lines 854-855  _(unsure)_

```python
out.attrs["explained_variance"] = list(
```

Explained variance is the only honest label for a PC axis: "PC1" alone says nothing about whether it is the data or the noise.

### lines 860-863

```python
from .utils import umap
```

spacr.utils' lazy loader, not a bare `import umap`: the package's __init__ reaches umap.parametric_umap -> tensorflow, and spaCR's standing rule is that nothing drags TF in. The loader imports umap.umap_ with the TF-backed roots blocked.

### lines 879-881

```python
bounded = max(5.0, min(float(perplexity), (len(values) - 1) / 3.0))
```

sklearn RAISES when perplexity >= n_samples, so a perfectly reasonable default becomes a failed projection the moment the selection is small. Clamped rather than passed through.

### lines 889-891

```python
return out.reindex(frame.index)
```

Reindexed to the WHOLE frame: objects that could not be projected keep their row and get NaN, so the components can be added to the table without silently dropping rows out from under every other column.

## group_variance_share

### lines 895-914

```python
def group_variance_share(frame: pd.DataFrame,
```

Diagnostics: is the projection about what the user thinks it is about?

Both of these come from starplast, which built the same picker for a different table and learned two things worth not re-learning:

a group can be named as an input and contribute almost nothing starplast caught one carrying 1.1% -- and nobody notices, because a projection always produces a picture; a projection can separate objects on WHETHER THEY WERE MEASURED rather than on what was measured, and that reads as a phenotype.

The second matters more in spaCR than it did there. A cell with no pathogen has NaN for every pathogen measurement, and `reduce_dimensions` median-fills rather than dropping the row -- deliberately, because dropping loses every measurement the object DID have. But a median fill puts all the uninfected cells at the same point on those axes, so an embedding can split infected from uninfected on missingness alone. That split is real, reproducible, and not a phenotype.
