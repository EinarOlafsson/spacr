# Notes from `spacr/sequencing_qc.py`

Prose lifted out of `spacr/sequencing_qc.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_resolve_column](#_resolve_column) (1 entry)
- [load_count_table](#load_count_table) (4 entries)
- [position_effects](#position_effects) (2 entries)
- [library_depth](#library_depth) (1 entry)
- [unmapped_read_fractions](#unmapped_read_fractions) (1 entry)
- [_read_reference](#_read_reference) (1 entry)
- [barcode_collisions](#barcode_collisions) (3 entries)
- [WellFractions](#wellfractions) (1 entry)
- [derive_threshold.choice_at](#derive_thresholdchoice_at) (3 entries)
- [derive_threshold](#derive_threshold) (3 entries)
- [sweep_grid](#sweep_grid) (1 entry)
- [recommend_threshold](#recommend_threshold) (2 entries)
- [_save_figure](#_save_figure) (1 entry)
- [plot_threshold_sweep](#plot_threshold_sweep) (4 entries)
- [plot_barcode_qc](#plot_barcode_qc) (5 entries)
- [plot_barcode_qc._natural](#plot_barcode_qc_natural) (1 entry)
- [barcode_qc](#barcode_qc) (4 entries)

## Module level

### lines 65-67

```python
from .figures.style import ROLES, figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## _resolve_column

### lines 114-116  _(unsure)_

```python
def _resolve_column(df: pd.DataFrame, canonical: str) -> Optional[str]:
```

Loading and normalising the count table

## load_count_table

### lines 174-179

```python
from .tabular import read_table
```

THROUGH THE FUNNEL (145). These are the COUNT CSVs, and they are the case that instruction measured: `row_name`, `column_name`, `grna_name` and NO plate column at all, so four plates' r1/c1 pooled into one well -- 384 wells instead of 1,536 -- with the fractions still summing to 1, so nothing downstream could notice.

### lines 216-219

```python
df = df[df["count"] > 0]
```

A zero or negative count is not a call — it is an artefact of a table that was merged or hand-edited. Keeping it would put a gRNA in a well at fraction 0.0 and inflate every "gRNAs per well" count below every threshold.

### lines 233-234  _(unsure)_

```python
counts = (counts.groupby(["prc", "plateID", "rowID", "columnID", "grna"],
```

Sum first: two sources may legitimately hold the same well (a resequenced lane), and their reads belong to the same well total.

### lines 237-240

```python
counts["well_reads"] = counts.groupby("prc")["count"].transform("sum")
```

No zero-total well can reach this line: every surviving row carries a positive count, so every well's sum is positive and the division below is always defined. That is why there is no guard here — one would be unreachable, and unreachable guards get believed.

## position_effects

### lines 347-350

```python
fold = median / plate_median if plate_median else np.inf
```

A plate median of zero cannot happen (load_count_table rejects empty wells), but a defensive guard here keeps the ratio finite for a caller that built the frame by hand.

### lines 363-364  _(unsure)_

```python
order = np.abs(np.log2(out["ratio_to_plate"].replace(0, np.nan)))
```

Worst first: distance from parity on a log scale, so a half-depth row and a double-depth row rank equally badly.

## library_depth

### lines 421-423

```python
skew = float(p90 / p10) if p10 > 0 else float("inf")
```

p10 of a library where a tenth of the guides are absent is 0. Reporting inf is honest — the skew is unbounded — and is what a user needs to see rather than a silently clipped ratio.

## unmapped_read_fractions

### lines 505-507

```python
"unmapped_fraction_lower": max(per_field.values()) if per_field else 0.0,
```

A read is lost if ANY field failed. At best the failures all coincide on the same reads (lower bound = the worst field); at worst they are disjoint (upper bound = their sum).

## _read_reference

### lines 545-548

```python
df = pd.read_csv(path)
```

RAW, DELIBERATELY. This is a barcode table -- `name` and `sequence` and carries no plate, row, column, field or well. Canonicalising it would be a no-op with an import behind it, and 145's rule is about readers of METADATA-bearing tables.

## barcode_collisions

### line 605  _(unsure)_

```python
by_sequence: Dict[str, List[str]] = {}
```

Exact duplicates, by sequence.

### lines 619-623

```python
buckets: Dict[Tuple[int, str], List[int]] = {}
```

Hamming-1 neighbours without the N^2 comparison: two equal- length sequences differ in at most one position exactly when they agree after masking that position. A pooled gRNA library is 10^4-10^5 barcodes, where N^2 is not affordable and this is linear in N.

### lines 647-649

```python
arrays = [np.frombuffer(s.encode(), dtype=np.uint8) for s in seqs]
```

Beyond one substitution the masking trick no longer applies and the honest implementation is the pairwise one. It is only reachable when the caller asks for it.

## WellFractions

### lines 721-723  _(unsure)_

```python
class WellFractions:
```

The threshold: derive it from the target, then sweep around it

## derive_threshold.choice_at

### lines 934-935  _(unsure)_

```python
low_index = 0
```

Last candidate whose statistic is still strictly above the one we settled on; everything after it is on the plateau.

### lines 946-948

```python
low = float(candidates[low_index - 1] if low_index > 0
```

Thresholds below the smallest observed fraction all behave identically, so that fraction — not zero — is the meaningful bottom of an open-ended plateau.

### lines 951-952

```python
middle = float(np.sqrt(low * high)) if high > low else high
```

Geometric middle: abundances are ratios spanning orders of magnitude, so halfway between 0.004 and 0.22 is 0.03, not 0.11.

## derive_threshold

### lines 962-964

```python
return choice_at(0, lowest, attainable=False)
```

Even keeping every observed gRNA does not reach the target. No threshold can; say so instead of returning the smallest number in the table as though it were a choice.

### lines 967-969

```python
low, high = 0, int(candidates.size) - 1
```

Largest candidate whose statistic still meets the target. The statistic is non-increasing, so the predicate is monotone and a bisection is exact.

### lines 979-982

```python
if low + 1 < candidates.size:
```

The next candidate up is the first that falls short. When it lands closer to the target than the one that meets it, it is the better answer; on a tie the one that meets the target wins, because a well short of its guides has lost power that no later step recovers.

## sweep_grid

### lines 1031-1035

```python
grid = grid[~np.isclose(grid, threshold, rtol=1e-9, atol=0.0)]
```

Drop grid points that merely round to the centre before inserting it. np.unique compares bit patterns, so geomspace's 0.21999999999997 would survive next to an inserted 0.22 as a second, near-identical row of the sweep — two lines the user cannot tell apart reporting different numbers.

## recommend_threshold

### lines 1177-1179

```python
here = float(at["grnas_per_well"])
```

Quote the thresholds where the answer actually CHANGES, not the ends of the sweep: on a wide plateau the ends both report the same gRNAs-per-well and the sentence says nothing.

### lines 1209-1213

```python
below = sweep[sweep["threshold"] <= t].sort_values("threshold")
```

The knee: the adjacent pair of thresholds below the derived one across which the collision rate climbs fastest per octave. That is the sentence a methods section wants — "below X, collisions rise sharply" — so it is quoted as the two measured values, not as a slope the reader has to integrate.

## _save_figure

### line 1241

```python
from .plot import save_figure
```

108 point 6: the format and the DPI are the user's, not this line's.

## plot_threshold_sweep

### lines 1277-1280

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 1291

```python
top.axhline(choice.target, color=ROLES["reference"], ls="--", lw=1,
```

178 A: the reference role, so the line is visible in both themes.

### lines 1294-1295  _(unsure)_

```python
top.set_yscale("symlog", linthresh=1, linscale=0.35)
```

linscale keeps the 0-1 linear band from eating a third of the panel; "no guides left" needs to be visible, not prominent.

### lines 1316-1318

```python
top.annotate(f"derived: {choice.threshold:.4f}",
```

Anchored in axes coordinates on the y and data coordinates on the x, so the label rides the line at a fixed height whatever the symlog axis does with its limits.

## plot_barcode_qc

### lines 1354-1357

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.

### line 1378  _(unsure)_

```python
ax = axes[0][1]
```

2 — position effects.

### lines 1406-1407  _(unsure)_

```python
labels = [str(b) if str(b).lower().startswith(a[0])
```

The axis initial is only worth a prefix when the label does not already carry it.

### line 1420  _(unsure)_

```python
ax = axes[1][0]
```

3 — library coverage, as a Lorenz curve.

### line 1437  _(unsure)_

```python
ax = axes[1][1]
```

4 — read fate.

## plot_barcode_qc._natural

### lines 1383-1385

```python
def _natural(value):
```

Natural order, so c2 sits between c1 and c10 rather than after c12 — a position-effect panel whose columns are out of plate order cannot be read against the plate.

## barcode_qc

### lines 1679-1681

```python
population = keep or None
```

Never hand an empty population to the derivation: when every well is below the cut the run is starved as a whole, and the honest answer is to fit on what there is and say so in the QC.

### lines 1687-1691

```python
tail = float(np.quantile(counts["fraction"].to_numpy(float), 0.01))
```

Reach down into the bleed-through tail even when the derived threshold sits far above it, so the collision knee is on the curve. Floored at a thousandth of the derived value: a run with one freakishly small fraction should not stretch the plot over six decades of empty space.

### lines 1696-1698

```python
high=choice.interval_high * 1.5)
```

...and up past the top of the plateau, so the cost of tightening is on the curve too rather than sitting just off the right-hand edge of it.

### lines 1751-1754

```python
plt.close(figure)
```

Closed rather than shown: this runs inside a mapping pipeline and inside the Qt worker thread, neither of which owns a GUI event loop, and an accumulating figure stack is a memory leak over a plate's worth of samples.
