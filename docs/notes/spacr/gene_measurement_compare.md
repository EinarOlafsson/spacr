# Notes from `spacr/gene_measurement_compare.py`

Prose lifted out of `spacr/gene_measurement_compare.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [control_wells](#control_wells) (1 entry)
- [Comparison](#comparison) (1 entry)
- [build](#build) (8 entries)
- [with_statistics](#with_statistics) (1 entry)
- [plot](#plot) (4 entries)
- [save](#save) (5 entries)
- [object_identity](#object_identity) (1 entry)
- [join_measurements](#join_measurements) (4 entries)
- [ComparisonStyle](#comparisonstyle) (1 entry)
- [render_comparison](#render_comparison) (2 entries)

## Module level

### lines 15-17

```python
from .figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

### lines 852-854

```python
LABEL_COLUMNS: Tuple[str, ...] = (
```

187 A: every measurement in the database, not only the ones on png_list

### lines 971-974

```python
"pred", "score", "test", "cv_predicted_class",
```

FROM `png_list`, and the reason it is worth joining at all: the score and the call are there and in no object table. `png_path` stays out a path is not a measurement and it would be offered in the comparison chooser.

## control_wells

### lines 223-226

```python
wanted = [typed] if isinstance(typed, str) else list(typed or ())
```

ONE NAME IS A NAME, NOT FOUR LETTERS. `resolve_controls` iterates its argument, so a bare string arrives as a sequence of characters and every one of them resolves to nothing. The control field hands over a list; a caller with one control in hand should not have to know that.

## Comparison

### line 300, trailing  _(unsure)_

```python
frame: pd.DataFrame
```

long: group, value, and the unit id

## build

### lines 392-397

```python
contrast = str(contrast or "")
```

the contrast

WHICH ROWS ARE "the rest" IS THE QUESTION (187 B). The same annotated cells against the rest of their own well, against the controls, and against every other well are three different experiments, and the default -- everything else, wherever it is -- is the one that mixes all three together.

### lines 414-415

```python
if wells is not None:
```

THE CHOSEN WELLS FIRST, because every contrast below is defined against the annotation that is actually being used.

### lines 431-434

```python
keep &= annotated | (rest & ~where.astype(str).isin(theirs))
```

`theirs`, NOT `mine`: a well excluded from the annotation is excluded, full stop. Letting it back in as "another well" would put the very rows the user threw out on the other side of the comparison.

### lines 465-468

```python
work[key] = objects[key].astype(str).loc[work.index]
```

`.loc[work.index]` because the contrast may have dropped rows: assigning the FULL column onto a shorter frame aligns on the index and leaves NaN wherever the two disagree, which then joins every such row into one well named "nan".

### lines 472-479

```python
if not len(work):
```

NOTHING LEFT IS AN ANSWER, NOT A CRASH. Every other empty case in this function returns a Comparison carrying the reason, and this one used to raise instead: on an empty frame `agg("_".join, axis=1)` hands back an empty DATAFRAME of the key columns rather than a Series, and assigning that to one column is a ValueError -- "Cannot set a DataFrame with multiple columns to the single column unit". It reached the user as a traceback out of the Cells tab's Compare dialog, from a measurement that simply held no numbers.

### line 490

```python
work["unit"] = work[keys].agg("_".join, axis=1)
```

ONE ROW PER (unit, group), not per unit: see the docstring.

### lines 496-499

```python
said = contrast_note(contrast) if contrast else ""
```

THE CONTRAST IS NAMED FIRST, ahead of every other caveat, because it decides what the p-value below is a p-value ABOUT. A number from "within the well" and a number from "against every other well" are not comparable, and nothing else on the panel distinguishes them.

### lines 507-509

```python
note = ((note + " · ") if note else "") + (
```

SAID, ALWAYS. A comparison quietly computed on fewer rows than the user thinks is the kind of result that survives review and is wrong.

## with_statistics

### lines 555-557

```python
row["Normality"] = _summarise(normal)
```

THE ASSUMPTION CHECKS TRAVEL WITH THE RESULT. "where the variance and normality and n is noted and the correct test chosen" -- a test name without the checks that produced it cannot be reported.

## plot

### lines 634-636

```python
highlight = [HOUSE.BLUE, HOUSE.RUST, HOUSE.GREEN, HOUSE.PURPLE,
```

GREY IS THE REST, COLOUR IS THE CLAIM. More than one gene gets the palette in its fixed order rather than a colormap, so the same gene is the same colour in every panel of a figure.

### lines 642-645

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### lines 701-702

```python
axes.text(0.02, 0.98,
```

Preserve the requested bar plot but identify when the sample is so small that showing individual observations would be more informative.

### lines 722-723

```python
from .plot import save_figure
```

108 point 6: the one writer, so a comparison saved from this panel is in the format every other kept figure is in.

## save

### lines 771-774

```python
written[suffix] = save_figure(
```

BOTH FORMATS ON PURPOSE (the folder is a deliverable), so `fmt` is the loop's; everything else `save_figure` does -- the DPI rule and the repaint for paper -- is gained.

### lines 780-783

```python
plt.close(figure)
```

``save`` returns paths rather than the figure, so there is no caller that can own this pyplot registration.  Keeping it open accumulated one figure per saved comparison in long-lived GUI and batched-test processes.

### lines 788-790

```python
comparison.frame.to_csv(path, index=False)
```

THE DATA BEHIND THE GRAPH, which is the frame that was PLOTTED not the objects it came from. A reader checking the figure needs the numbers the figure drew.

### lines 809-810

```python
record["regression_settings"] = {
```

THE SETTINGS THAT GENERATED THE REGRESSION AND THE GRAPH, together under separate keys, so it is never ambiguous which produced which.

### lines 833-834

```python
continue
```

A crop that will not encode costs one image, never the save: the figure and the numbers are the point.

## object_identity

### lines 897-905

```python
head = frame[FIELD_KEY[0]].astype(str)
```

THE FOUR COLUMNS ARE THE FIELD KEY. `prcf` is nothing but plateID_rowID_columnID_fieldID pasted together -- `plate1_r5_c1_f16` and `png_list` carries the four without ever carrying the paste. So the crop table, which is the object table the Compare panel starts from, had "no object identity" and the join refused it: "these object rows carry no object identity (prcfo, or prcf and an object label)". Every morphological measurement in the screen was unreachable from that panel because of a column that was not written rather than data that was not there.

## join_measurements

### lines 1059-1061

```python
have = set(map(str, objects.columns))
```

ONLY WHAT IS NEW. A column present on both sides is already the value the montage selected on, and replacing it here would move the cells under the user's feet.

### lines 1075-1093

```python
out = pd.concat(
```

ASSIGNED BY POSITION, not joined. `_all_objects` concatenates one frame per plan, so the index can repeat -- and a join on a repeated label is a cartesian product of the matching rows on both sides, which turns 20,000 cells into 40,000 and every count on the panel with it. ALL AT ONCE, not column by column. Inserting them in a loop makes a new block per column, and a measurement table brings hundreds -- which is O(n^2) copying and a PerformanceWarning per column, hundreds of identical lines in the user's terminal for one merge.

RESET BOTH INDEXES FIRST, and that is the positional contract above, not tidying: `concat(axis=1)` ALIGNS ON THE INDEX, so with a repeated label -- which `_all_objects` produces, one frame per plan -- it would do exactly the cartesian product the loop existed to avoid. Two clean RangeIndexes make the concatenation positional, and the real index goes back on afterwards.

`reset_index` rather than `to_numpy`, so each column keeps its own dtype. One array for the block would cast an integer count to float because some other column beside it is float.

### lines 1102-1111

```python
return objects, (
```

A SILENT ZERO IS THE REAL FAULT (instruction 203). There is no run where NONE of the objects have a measurement, so zero matches is a join key that does not line up -- and a merge that matched nothing looks exactly like a merge that worked on an empty column once it reaches a panel that draws it.

THE ORIGINAL ROWS GO BACK, not the widened ones. Handing on a frame of all-NaN measurement columns is how the empty plot gets drawn; returning what the caller already had leaves them exactly where they were, with a sentence saying why.

### lines 1120-1123

```python
note = (f"{len(objects) - matched:,} of {len(objects):,} object "
```

SAID, ALWAYS -- the same rule the dropped denominators follow. A measurement that is missing on a third of the cells produces a comparison on a third fewer cells, and the panel has to be able to say so rather than quietly shrinking.

## ComparisonStyle

### lines 1132-1134

```python
@dataclass
```

108 points 1 and 2: a second style on the shared base, and the contract

## render_comparison

### lines 1293-1295

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS, and that is why the `figure=` branch is inside the context too: an axes added to an existing figure creates its own spines and ticks at that moment.

### lines 1344-1346

```python
axes.set_ylabel(str(comparison.measurement))
```

THE DEFAULT LABEL IS THE MEASUREMENT'S NAME, and a style that names one wins -- `apply_page` sets it after this and only when it is not blank, so "leave it alone" and "set it to nothing" stay different.
