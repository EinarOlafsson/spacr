# Notes from `spacr/settings_advisor.py`

Prose lifted out of `spacr/settings_advisor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Reading](#reading) (1 entry)
- [read_the_counts](#read_the_counts) (4 entries)
- [read_the_response](#read_the_response) (4 entries)
- [Module level](#module-level) (4 entries)
- [_family_and_transform](#_family_and_transform) (2 entries)
- [_plate](#_plate) (3 entries)
- [_significance](#_significance) (1 entry)
- [_thresholds](#_thresholds) (2 entries)
- [advise](#advise) (2 entries)
- [refusals](#refusals) (6 entries)
- [advise_that_runs](#advise_that_runs) (1 entry)

## Reading

### line 120

```python
run_folder: str = ""
```

what only a finished fit knows (instruction 226)

## read_the_counts

### lines 249-251

```python
parts = wells.str.split("_")
```

r1/c1 out of `plate1_r1_c1`, so a one-row or one-column screen can be recognised -- `model_plate_position` has nothing to model on one of either.

### lines 255-257

```python
try:
```

THE FRACTION DISTRIBUTION ITSELF, not only its shape. It was computed here and thrown away, and it is the only thing that can say whether a `fraction_threshold` keeps a library or deletes it.

### lines 266-267

```python
out["kept_at_two_percent"] = float((share >= 0.02).mean())
```

What the usual default would cost THIS screen, which is the number a user can act on.

### lines 277-280

```python
prefix = common_prefix([str(g) for g in guides.unique()])
```

THE GENE OF A GUIDE THROUGH THE ONE READER (145/184). `gene_of_guide` measures the organism prefix itself rather than assuming `TGGT1_`, so a Plasmodium or a human library is not pooled into one gene; the prefix is measured here too, for the note this reading carries.

## read_the_response

### lines 353-357

```python
here = _columns_of(path)
```

EACH FILE'S OWN HEADER. Taking the columns off the FIRST file and asking every other file for them is how plates 2, 3 and 4 of the reference screen were dropped: plate 1 carries `col` and the others do not, so `usecols` raised on each of them and the response was measured from one plate while the reading said four.

### lines 365-367

```python
piece = read_table(path, usecols=[wanted] + keys,
```

THE ONE READER (145), with `usecols` and `nrows` passed through: `read_table` forwards its kwargs to pandas, so the cap this module needs costs nothing to keep.

### lines 382-384

```python
frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 \
```

THE COLUMNS DIFFER BETWEEN PLATES, so the concatenation is on the union and a key missing from one file is NaN there rather than an error. `_well_key` picks whichever spelling is complete.

### lines 392-395

```python
where = _well_key(frame)
```

THE WELL IS THE UNIT THE FIT SEES. A per-object response is aggregated to wells before the model touches it, so the family question is about the WELL means -- and the object-level spread, which is much wider, is not the distribution being modelled.

## Module level

### lines 443-445

```python
QUESTIONS: Tuple[Question, ...] = (
```

The questions the data cannot answer

### lines 967-969

```python
_RUN_NUMBERS: Dict[str, Tuple[str, ...]] = {
```

226: what only a finished run knows

### lines 1310-1311  _(unsure)_

```python
"analysis_mode": "regression",
```

The permutation test works well by well; only the model can read objects.

### line 1314  _(unsure)_

```python
"agg_type": None,
```

One row per object already: there is nothing to aggregate.

## _family_and_transform

### lines 587-588

```python
chosen.append(Choice(
```

`check_distribution`'s own answer, and the reason names the number it turns on rather than repeating the recommendation.

### lines 621-623

```python
family = next(c.value for c in chosen if c.key == "regression_type")
```

THE TRANSFORM IS PART OF THE SAME DECISION. Every bounded family above carries its own link, so a transform on top of it is the double one 182 exists to prevent.

## _plate

### lines 698-700

```python
chosen.append(Choice(
```

PER-PLATE CENTRING, which needs nothing but the plate. It removes the plate's own mean and estimates NOTHING from the residuals, so there is no design for it to mistake signal for noise against.

### lines 706-709

```python
undecided.append(Undecided(
```

`control_center` IS THE BETTER ONE AND IS NOT PROPOSED, because it centres each plate on its CONTROL WELLS -- values in a column and which wells those are is not in the count or score tables. Naming it here is the difference between a default and a ceiling.

### lines 718-729

```python
if reading.plates > 1:
```

NOT ComBat, AND THAT IS A DECISION (196). ComBat estimates the plate effect from whatever the design does not explain, and it REFUSES to run until the caller says which biology to protect from that -- correctly, because in a pooled screen the biology is the per-well GUIDE COMPOSITION, which is continuous and is not a categorical covariate column. There is nothing honest to pass, so proposing ComBat means proposing a run that either refuses or removes the effects being looked for.

This module used to propose it anyway, with no covariate. The proposal was accepted, the run was pressed, and it failed on the refusal which is the whole reason 196 exists.

## _significance

### lines 794-796

```python
alpha = 0.05 if share >= 0.10 else (0.01 if share <= 0.01 else 0.05)
```

A LOW PRIOR MAKES EVERY DISCOVERY MORE LIKELY TO BE FALSE at the same alpha, which is the whole argument for moving it: at 5 real hits in 1,000, a 0.1 FDR is a list that is mostly noise.

## _thresholds

### lines 890-891

```python
proposal = float(f"{median / 10.0:.1g}")
```

A tenth of the typical share: low enough to keep the library, which is the failure that matters, and still above nothing.

### line 910  _(unsure)_

```python
per_guide = reading.wells_per_guide
```

MIN N -- how many observations a hit must rest on.

## advise

### lines 1158-1159  _(unsure)_

```python
_thresholds(reading, chosen, undecided)
```

Add evidence thresholds and the aggregation unit after the model family, transform, and batch method have been selected.

### lines 1162-1163

```python
_from_the_run(reading, chosen, undecided)
```

Completed-run diagnostics are applied last because they can supersede recommendations inferred from input structure alone.

## refusals

### lines 1199-1201

```python
def refusals(settings: Mapping[str, Any]) -> Tuple[str, ...]:
```

196 B: a proposal that the run would refuse is not a proposal

### line 1224  _(unsure)_

```python
if str(got.get("batch_correction") or "").lower() == "combat":
```

1. ComBat without a covariate. The one that was actually hit.

### lines 1244-1246

```python
kind = str(got.get("regression_type") or "").lower()
```

3. A setting the chosen estimator cannot read. `perform_regression` REFUSES these rather than ignoring them, so a number left on the panel from another model stops the run.

### lines 1258-1262

```python
mode = str(got.get("analysis_mode") or "").lower()
```

4. THE PERMUTATION TEST CANNOT SEE OBJECTS. Hit live on 2026-08-21: a run reached "permuting the guides" thirty-one seconds in -- after the filters, the plots and two saved CSVs -- and only then raised. The incompatibility is knowable from the settings alone and had no business waiting for the data.

### lines 1274-1277

```python
if (str(got.get("inference") or "").lower() in ("nonparametric",
```

5. THE SAME COMBINATION REACHED THROUGH `inference`, which is the door a user actually walks through: 'nonparametric' SELECTS guide_permutation, so the refusal has to recognise it under both names or it fires for the setting nobody typed.

### lines 1289-1292

```python
if unit == "cell" and got.get("agg_type"):
```

6. AN AGGREGATION THAT WILL NOT BE READ. `analysis_unit='cell'` keeps every object, so an `agg_type` set beside it is a control the user changed and the run ignored -- which is how somebody concludes the setting does nothing.

## advise_that_runs

### lines 1367-1368

```python
chosen, withdrawn = [], list(advice.undecided)
```

WHICH SETTING TO WITHDRAW. Named from the sentence rather than guessed: every refusal above quotes the key it is about.
