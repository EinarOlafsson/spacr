# Notes from `spacr/qt/widgets/power_design.py`

Prose lifted out of `spacr/qt/widgets/power_design.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [DesignSpec](#designspec) (3 entries)
- [power_curve](#power_curve) (1 entry)
- [plain_sentence](#plain_sentence) (1 entry)

## DesignSpec

### lines 442-446

```python
effect_fold: float = 6.667
```

0.80 / 0.12 to three decimals — the form carries three, so this is the value a freshly-opened screen reads back and `DesignSpec() PowerScreen().spec()` holds. The rounding puts the hit-cell rate at 0.80004 rather than 0.80; expressing a pair of rates as their ratio cannot do better, and 4e-5 of a probability changes no power.

### line 451  _(unsure)_

```python
gene_abundance_alpha: float = 0.6
```

held at the real screen's fitted values

### lines 461-465

```python
sequencing_error_rate: float = 0.0
```

Both default to the R behaviour rather than to the realistic value. A simulator whose baseline moved under a version bump would make every power figure already quoted from this screen wrong, and "the number changed because spaCR got more honest" is indistinguishable from "the number changed because something broke" to the person reading it.

## power_curve

### lines 750-753

```python
detected = ok & (auroc.fillna(-np.inf).to_numpy() >= threshold)
```

A detection needs BOTH an ok status and a score over the bar. The `fillna(-inf)` is not cosmetic: a NaN comparison is False in numpy too, but writing it down is what stops a later refactor from "tidying" this into a dropna() and quietly changing the denominator.

## plain_sentence

### lines 799-800

```python
return (f"The sweep did not include {target:g} cells per well, so "
```

Not interpolated. A power curve read between its own points is a number the simulation never produced.
