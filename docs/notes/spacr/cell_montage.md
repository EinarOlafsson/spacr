# Notes from `spacr/cell_montage.py`

Prose lifted out of `spacr/cell_montage.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [effects_grid_from_results](#effects_grid_from_results) (1 entry)
- [write_effects_grid](#write_effects_grid) (1 entry)
- [_guide_of_term](#_guide_of_term) (1 entry)
- [normalised_share](#normalised_share) (1 entry)
- [score_window](#score_window) (1 entry)
- [MontagePlan.arithmetic](#montageplanarithmetic) (1 entry)
- [_well_labels](#_well_labels) (1 entry)
- [_sudoku_calls](#_sudoku_calls) (4 entries)
- [select_montage](#select_montage) (26 entries)
- [read_well_guide_fractions](#read_well_guide_fractions) (1 entry)
- [fractions_from_counts](#fractions_from_counts) (6 entries)
- [load_montage_objects](#load_montage_objects) (5 entries)
- [montage_route_requirements](#montage_route_requirements) (1 entry)
- [resolve_montage_crop_source](#resolve_montage_crop_source) (1 entry)

## effects_grid_from_results

### lines 304-307

```python
return None
```

Silent, like `effects_from_results` beside it: an unreadable grid means "no sweep to use", and the caller's own message says what that costs. Raising here would take the montage down over a file it can do without.

## write_effects_grid

### lines 333-334

```python
return ""
```

A sweep that produced its answer has not failed because the grid could not be filed beside it.

## _guide_of_term

### line 385  _(unsure)_

```python
if inside.startswith("T."):
```

statsmodels writes `C(rowID)[T.r2]` for a factor level.

## normalised_share

### lines 438-439

```python
return min(share * factor, 1.0), factor
```

Capped at 1: a share above 1 would mean this guide is more than all of the well, which is a join that did not line up rather than a number.

## score_window

### lines 654-657

```python
scale = float(np.std(values))
```

Every score identical, or a distribution so concentrated the MAD underflows. Fall back to the plain standard deviation rather than inventing a width; if that is zero too the window is degenerate and says so instead of admitting nothing.

## MontagePlan.arithmetic

### lines 1109-1115

```python
crowded = sum(w.n_selected for w in self.wells
```

THE GUIDES SHARE CELLS, AND A READER SHOULD NOT FIND THAT OUT BY ARITHMETIC (172, last open item). Every guide in a well is given `round(n x share)` of the SAME top-ranked cells, so the counts of a well's guides can add to more than the well holds -- measured on the maintainer's four plates, 190 of 1,366 wells. It is correct for this heuristic and it is exactly the flaw instruction 173 exists to fix, but unsaid it reads as a bug in the count.

## _well_labels

### lines 1177-1180

```python
labels = [
```

pandas 3 preserves missing values while casting string-dtype columns to ``str``. Joining the whole frame can therefore hand ``"_".join`` a float NaN. Exclude incomplete identities before conversion: those rows are deliberately ``None`` and never have a join key.

## _sudoku_calls

### lines 1246-1249

```python
numeric = frame.select_dtypes(include=[np.number])
```

THE FEATURES ARE THE MEASUREMENTS, NOT THE SCORE. The anchors are chosen BY the score, so a graph built on it would place every high-scoring cell beside every guide's anchors and affirm all of them. `spacr.sudoku` leaves the score out for the same reason.

### lines 1273-1297

```python
wanted = {str(g) for g in (guides or ())} or {str(name)}
```

SCOPE: THIS GUIDE'S WELLS, PLUS THE WELLS THAT ANCHOR ITS RIVALS.

THE FIRST VERSION OF THIS TRIMMED TO THIS GUIDE'S WELLS ALONE AND THAT WAS WRONG, on an argument that conflated two different parts of the method. The WELL CONSTRAINT is per well, so a guide absent from a well genuinely cannot claim cells there -- that much held. The GRAPH is not per well at all: it links cells by how they look, across the screen, and that is where a guide's appearance is learned.

So trimming the cells also trimmed the ANCHORS. `anchors_for` takes a guide's examples from wells where that guide DOMINATES, and a rival's best wells are usually not this guide's wells -- so the rivals were being characterised from whatever share they happened to have here, which is the weakest sample available rather than the strongest. Raised by the maintainer: "other guides in the chosen guides wells do [share wells], so that information can be used, no?" -- yes, and it was being thrown away.

The scope is therefore the union of this guide's wells and, for every guide that appears in them, the wells where that guide is large enough to anchor. Those extra wells inform the graph and the anchors; only this guide's wells are drawn. THE COEFFICIENT'S GUIDES, not its name. A gene is never a key in `here`, which is guide -> fraction.

### lines 1334-1344

```python
return dict(zip(frame.index, result.guides))
```

KEYED BY THE FRAME'S INDEX, NOT BY POSITION.

THIS WAS THE BUG, and it was silent: the caller matches these against `ranked`, which is the well's rows SORTED BY SCORE, while this list was in the order the rows arrived. Same length, different cells -- so the calls landed on the wrong objects, and where the lengths disagreed nothing was marked at all. A real run reported "59 of 1,076 cells annotated" and then highlighted zero in every well.

An index cannot be misaligned by a sort.

### line 1347

```python
notes.append(f"sudoku could not run ({type(exc).__name__}: {exc}); "
```

A PICKER THAT CANNOT RUN SAYS SO AND THE MONTAGE STILL DRAWS.

## select_montage

### lines 1460-1475

```python
excluded_note = ""
```

THE EXCLUSION HAPPENS FIRST, ONCE, AND ON THE COUNT TABLE ITSELF. Asked for 2026-08-21: "make sure it is removed first. right?" -- and right, for a reason that a later per-call filter would not have fixed. `well_totals` below sums the fraction column over the FULL table, and `normalised_share` divides by that sum. A contaminant removed further downstream would still be sitting in that denominator, holding every real guide's share down by its own -- which on one real plate was a fifth of all the reads.

So there is exactly one exclusion point and it is above every path: the ranking, the per-well fractions, the posteriors and the totals all read a table the contaminant has already left.

GUIDES AND GENES, several of each, resolved by `control_names` -- the same resolver `controls` and `positive_control_id` use, so an exclusion is typed in the spelling those already accept.

### lines 1491-1493

```python
missing = unmatched_exclusions(exclude_grnas, names, genes)
```

A MISSPELLED EXCLUSION EXCLUDES NOTHING AND LOOKS LIKE IT

WORKED, which is how a known contaminant survives the filter that was meant to remove it.

### lines 1506-1509

```python
grouped: Dict[Tuple[Any, ...], pd.DataFrame] = {}
```

Always key on a tuple. ``groupby(['prc'])`` yields a bare string on pandas 1.x and a 1-tuple on 2.2+, so a lookup written for either one silently matches nothing on the other -- and "nothing matched" here is an empty montage with a caption that still reads as if it worked.

### lines 1513-1517

```python
well_totals: Dict[str, float] = {}
```

EVERY GUIDE'S FRACTION IN EACH WELL, which is what the normalisation divides by (instruction 172). It comes from the FULL count table and not from `selected_counts`: the latter holds only the chosen coefficient, so summing it would always give that guide's own fraction and a factor of exactly 1 -- a normalisation that never normalised.

### lines 1535-1548

```python
if str(picking or "rank") in ("attributed", "assigned", "multivariate") \
```

CAN THIS GUIDE BE ATTRIBUTED AT ALL -- asked BEFORE any cell is attributed, which is instruction 173's own wording. Until now this was a library function with no caller, so the answer existed and nobody saw it.

IT IS THE DIFFERENCE BETWEEN AN EMPTY MONTAGE AND AN EXPLAINED ONE. A guide whose effect is too small against the spread of scores can never reach the threshold in any well, so the attributed picker selects nothing and the montage comes back blank -- which reads as a bug in the viewer rather than as arithmetic about the guide.

The scale and centre are taken over ALL the objects, not per well: centring per well destroys the between-well signal that identifies the effect at all.

### lines 1563-1566

```python
targets = covered if resolved_level == "gene" else [name]
```

Fractions and effects are guide-keyed.  A gene-level coefficient therefore has to pre-flight the guides it covers; asking for the gene name itself always produces the false verdict "no well carries it".

### lines 1576-1577

```python
pass
```

A pre-flight is a courtesy, not a precondition. It must never be the reason a montage does not draw.

### lines 1580-1584

```python
sudoku_calls = None
```

SUDOKU RUNS ONCE, OVER THE WHOLE SCREEN, BEFORE THE WELL LOOP. Every other picker decides a well from that well's own cells, so it can be computed inside the loop. Sudoku cannot: what a guide's cells look like is learned from every well the guide is in, and running it per well would throw away the one thing it is for.

### lines 1610-1614

```python
total_here = float(well_totals.get(label, 0.0))
```

HOW MANY (instruction 172). The guide's share of what the count table still holds for this well, times the number of cells that actually carry a classification score -- not the number of rows. An object with no score cannot be ranked, so counting it would promise cells the ranking cannot deliver.

### lines 1620-1631

```python
share, factor = float(fraction), 1.0
```

THE RAW FRACTION, WHICH IS THE CONSERVATIVE ONE (207 D). Normalising divides by what SURVIVED the threshold, so the reads of every filtered-out guide are redistributed onto the ones that remain -- measured on a real screen, the filtered sums fall to a median of 0.5526, so each survivor is inflated by about 1.8x. A guide with few reads in a well can come out of that with a high share, and the ranking then takes cells on the strength of a number normalisation created.

Raw keeps the discarded reads in the denominator, where they dilute a marginal guide instead of being handed to it. Neither is right for every screen, which is why it is a choice.

### lines 1639-1647

```python
note = (f"round({share:.4g} x {n_classified}) rounds to zero, so "
```

THE SAME FOUR NUMBERS AS THE NON-ZERO CASE. This said round(n_objects x fraction), and the count is round(n_classified x share) -- a different count off a different base. It named the UN-NORMALISED fraction, which is the one number instruction 172 exists to stop anyone reading: on the maintainer's four plates the factor runs to 6.6x, so a reader checking this line would work out a share up to six times too small and conclude the montage had dropped their well. It covers 153 of 5,615 guide-well pairs that hold cells.

### lines 1660-1667

```python
ranked = here.assign(_montage_score=here_scores)
```

WHICH ONES (instruction 172). "rank all cells by classefication score and take the top x cells".

THE DIRECTION FOLLOWS THE COEFFICIENT. Highest scores for a positive effect, lowest for a negative one, because the cells a coefficient points at are the ones whose phenotype moved the way it says. Always-descending would show a negative coefficient the cells LEAST consistent with it.

### lines 1674-1676

```python
picked_by = "rank"
```

THE OTHER TWO PICKERS (instruction 173). Both need every guide's fraction AND effect in this well, because a posterior is a comparison: comparing a guide against nothing returns the prior.

### lines 1680-1684

```python
wanted = "attributed"
```

SAID, NOT SUBSTITUTED SILENTLY. Option C needs one effect per MEASUREMENT per guide, which is the gene x measurement sweep's grid; a run that has not swept has nothing to read. Falling back to the single-score attribution is the right answer, and a montage that quietly changed how it chose its cells is not.

### lines 1711-1718

```python
if sudoku_calls:
```

ACROSS WELLS, WHICH IS THE WHOLE POINT (209), and also the reason it cannot be computed inside this per-well loop the way the others are: a guide's appearance is learned from every well it is in. `_sudoku_calls` runs once for the whole screen before the loop and this reads its answer for these rows.

BY INDEX. `ranked` is sorted by score, so a positional lookup would put each cell's call on a different cell.

### lines 1720-1723

```python
_wanted = {str(g) for g in (covered or ())} or {str(name)}
```

AGAINST THE COEFFICIENT'S GUIDES. `sudoku` calls a cell for a GUIDE; a gene-level montage is named for the gene, and comparing the two matched nothing -- which is a montage of unringed cells and no error anywhere.

### lines 1742-1745

```python
outcome = assign_well(values, here_fractions, effects,
```

EVERY cell in the well gets exactly one guide and each guide gets exactly the cells its reads imply, so this picker's count is x by construction rather than by rounding.

### lines 1759-1772

```python
picked_by = (f"rank ({wanted} needs more than one guide in a "
```

A WELL WITH ONE GUIDE CANNOT BE ATTRIBUTED, and until now it did not say so. Attribution is a comparison; with a single guide there is nothing to compare against, so control fell through to rank with `picked_by` still "rank" and no note. Both sibling pickers disclose their fallback -- multivariate above, sudoku below -- so in a montage mixing single-guide and multi-guide wells, two wells were chosen by different rules and only one of them said which. The prefix is "rank" ON PURPOSE. `_by_rank` below keys off it, and rank arithmetic is genuinely what chose these cells, so the caption must still show the round(share x n) line that explains them. Saying "fell back to rank" instead would both suppress the true explanation and read "...fell back to rank chose 6 of 20 classified cell(s)".

### lines 1776-1786

```python
_by_rank = str(picked_by).startswith("rank")
```

THE NOTE MUST DESCRIBE THE PICKER THAT RAN. The fraction arithmetic below is how `rank` decides; every other picker decides some other way and does not consult it. Printing it regardless reported a calculation that did not happen -- observed on a sudoku montage that highlighted nothing and still said "round(0.1267 x 187) = 24", which is a number the run never used.

`expected` stays computed either way: it is the count the

SEQUENCING supports, which is worth stating next to what the picker actually chose, because the gap between them is the interesting part.

### lines 1788-1793

```python
_fallback = str(picked_by)[len("rank"):].strip() if _by_rank else ""
```

A RANK FALLBACK STILL OWES THE REASON IT FELL BACK, and the two are not alternatives. `_by_rank` decides whether the round(share x n) arithmetic is shown, and for a fallback that arithmetic IS what chose the cells -- so it has to stay. But the branch that shows it never printed `picked_by`, so a qualifier attached there vanished. It rides with the arithmetic instead of replacing it.

### lines 1810-1812

```python
arithmetic = (f"round({share:.4g} x {n_classified}) = {expected}, "
```

SAY WHICH FRACTION IT USED. The two differ by the normalisation factor, and a count that cannot be traced to a fraction is a count nobody can check.

### lines 1820-1822

```python
take = ranked.copy()
```

EVERY CELL IN THE WELL, with the chosen ones marked rather than the rest removed. "show all the images from each well and highlight the cells most likely to be whatever gene is picked".

### lines 1826-1834

```python
take[ANNOTATION_COLUMN] = np.where(
```

AND THE REST ARE NAMED, NOT MERELY UNMARKED (207 B). Asked for 2026-08-21: "i an the non annotated datapoints to be annotated as Non_annotated and shown".

A CELL THAT IS SHOWN AND CARRIES NO LABEL IS COUNTED BY THE

EYE AND BY NOTHING ELSE. Giving it a name puts it in the legend, in the group-by, and in the denominator -- which is where it has to be, because a fraction computed over only the annotated cells is the fraction that came out as 1.

### lines 1842-1844

```python
marked = (f"{_n_marked} are highlighted and the rest are "
```

NOT "the N with the highest scores" -- a picker that is not

`rank` did not choose by score, and saying it did explains the picture with the wrong rule.

### lines 1855-1856  _(unsure)_

```python
take[ANNOTATION_COLUMN] = str(name)
```

The same column either way, so a consumer does not have to know which view produced the frame.

### lines 1899-1902

```python
if "montage_candidate" not in picked.columns:
```

WHICH OF THEM THE COEFFICIENT POINTS AT, kept as a column so the panel can mark them. In the filtered view every row is a candidate by construction; in the show-all view it is the distinction the whole option exists for.

## read_well_guide_fractions

### lines 2019-2024

```python
frame = _read_table(target, report=None)
```

THROUGH THE FUNNEL (145). `_well_key` composes prc from plateID, rowID and columnID, and a results folder written by an older spaCR or by a plate whose png_list spells them row_name / column_name -- gave it none of the three. It raised nothing; it produced a frame with no well key, and the montage then drew nothing for wells holding 244 objects.

## fractions_from_counts

### lines 2054-2060

```python
for index, path in enumerate(paths):
```

ENUMERATED, and the index is used even for the files that are skipped. A count CSV names its plate in a column or not at all -- the real ones carry `row_name, column_name, grna_name, count` and nothing else -- so the plate comes from WHICH FILE it is, exactly as `ml.load_regression_input_pairs` resolves it: own column, then pair-row order. Letting an unreadable file collapse the numbering would shift every later plate's label by one and silently mislabel the wells.

### lines 2066-2069

```python
frame = _read_table(text, report=None)
```

The count CSVs are the case 145 measured: `row_name`, `column_name`, `grna_name` and NO plate column at all, so four plates' r1/c1 pooled into one well -- 384 wells instead of 1,536 -- and the fractions still summed to 1.

### lines 2075-2082

```python
try:
```

CANONICALISE FIRST, the way every other reader in spaCR does. Reported 2026-08-18: "the cell montage failed because the column grna was not found in any of the count tables" -- and the tables had the identifier under one of the spellings `correct_metadata_column_names` exists to absorb (`grna_name`, and whatever `schema.canonicalise_frame` maps). Reading the CSV raw made this function the ONE reader that did not, which is exactly the "one vocabulary" failure instruction 145 is about, introduced while fixing something else.

### lines 2089-2093

```python
if "grna" not in frame.columns:
```

AND THE ALIASES THE REST OF THE PROJECT ALREADY ACCEPTS. `utils` looks for a gRNA identifier under seven spellings when reading metadata; a count table is the same identifier in the same shape, so refusing it here for its header would be this module inventing a stricter rule than the code around it.

### lines 2106-2110

```python
if "plateID" not in frame.columns or frame["plateID"].isna().all():
```

THE PLATE, from the pair row, and only when the file does not say. Without this the four plates' wells pool: `prc` composed from row and column alone makes plate1 r1/c1 and plate2 r1/c1 ONE well, and every fraction below is then a share of four plates' reads. That is a wrong number that looks right -- the fractions still sum to 1.

### lines 2116-2119

```python
problems.append(
```

NAME WHAT THE FILE ACTUALLY HAS. "column grna was not found" is true and unactionable: the user cannot tell whether they picked the wrong file or whether their header is spelled differently, and those have different answers.

## load_montage_objects

### lines 2236-2252

```python
try:
```

CANONICALISE FIRST, the way every other reader in spaCR does -- and the way `fractions_from_counts` was taught to earlier the same day, for the same reason and by the same failure. Instruction 145.

Measured on the maintainer's four plates, `png_list`:

plate1  rowID / columnID   and plateID = 'pplate1' plate2  row_name / column_name plate3  row_name / column_name plate4  row_name / column_name

So plates 2-4 could not compose a `prc` at all, and plate1 composed one against a doubled plate name that matches nothing in the counts. Every well then reported "no object in the imported databases comes from this well" and the montage drew nothing, while the baseline was happily computed over all 226,467 objects -- which is what made it look like a selection problem rather than a join one.

### lines 2267-2276

```python
from .predictions import attach_predictions
```

THE SCORES THE RUN ALREADY HAS (instruction 167). A screen whose png_list has no `pred` is not a screen without scores: the score CSVs the regression module is holding carry one row per cell, and the fit was run on exactly those numbers. Joined through `predictions.attach_predictions`, which is the SAME key choice `merge_prediction_results` makes, so a montage reading them here and a database that had them merged in cannot disagree.

NOTHING IS WRITTEN. A montage is a read, which is the same rule the crop-path re-rooting follows.

### lines 2302-2303

```python
joined = frame.copy()
```

The join keeps only rows that can be cut from merged/. A PNG folder alone is still a montage, so fall back rather than returning none.

### lines 2306-2310

```python
root = src or portable_paths.source_root_for_database(db_path)
```

RE-ROOT BEFORE ANYTHING READS A PATH. The crop source is resolved from the folder the user is looking at, so it is found correctly; it was the per-object rows that still pointed at the machine the screen was measured on, and a montage over 60,000 dead paths draws nothing and blames the crops.

### lines 2314-2320

```python
if report.partial or (report.moved and verbose) or (
```

SAID WHEN IT CANNOT, not only when it can -- a crop that could not be placed is returned unchanged and fails later as a missing file, somewhere with less context (instruction 155 F). But a column where NOTHING resolved is a route that is not on this machine, not 60,816 failures: a screen with PNG crops and no `merged/` folder is healthy, and saying otherwise is the false alarm that teaches a reader to ignore the true one.

## montage_route_requirements

### lines 2490-2491  _(unsure)_

```python
has_mask = "cell" in mask_dims and bool(
```

Derived as cell minus nucleus/pathogen/organelle, so it needs the cell plane and at least one to subtract.

## resolve_montage_crop_source

### lines 2616-2619

```python
root = src.get("src") if isinstance(src, Mapping) else src
```

WHETHER A CHANNEL LIST EXISTS AT ALL, which is a different question from whether the spec has channels: `crop_spec_from_settings` always produces some, so an unrecorded run silently draws planes 0,1,2 and looks like a deliberate choice.
