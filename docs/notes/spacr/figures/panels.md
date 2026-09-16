# Notes from `spacr/figures/panels.py`

Prose lifted out of `spacr/figures/panels.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [control_threshold](#control_threshold) (1 entry)
- [volcano](#volcano) (7 entries)
- [effect_rank](#effect_rank) (1 entry)
- [p_histogram](#p_histogram) (1 entry)
- [control_separation](#control_separation) (3 entries)
- [guide_agreement](#guide_agreement) (1 entry)
- [available](#available) (1 entry)

## Module level

### lines 19-21

```python
from ..figures.style import figure_style, theme_target
```

THE HOUSE STYLE (136). `figures.style` imports matplotlib only inside its own functions, so naming it here costs nothing at import time.

## control_threshold

### lines 215-217

```python
finite = values[np.isfinite(values)]
```

No usable controls: the all-guide MAD, which is the same statistic over a family that includes the hits. Named differently so a reader is never told a number is control-based when it is not.

## volcano

### lines 272-276

```python
baseline = resolve_baseline(frame, baseline_kind or "zero",
```

BEFORE THE THRESHOLD IS COMPUTED, not after. The effect-size cut is derived from the control guides' spread, so a baseline that moved the controls to zero and a cut placed on the unshifted column would be a line drawn in the wrong place -- the one failure mode this whole panel is meant to avoid.

### lines 288-290

```python
rule = ""
```

THE EFFECT-SIZE CUT IS ON BY DEFAULT NOW. "in your versions i have never seen the effect size threshold" -- because it defaulted to None and drew no line at all.

### lines 305-308

```python
in_compartment = None
```

COMPARTMENT COLOURING REPLACES THE UP/DOWN COLOURING RATHER THAN JOINING IT. Both are "the thing the sentence is about", and a volcano carrying two of those has no sentence -- a reader cannot tell whether a coloured dot is coloured for being called or for being a rhoptry.

### lines 331-332

```python
reference_line(ax, x=sign * abs(effect_threshold),
```

The rule is named on the line, once. A threshold a reader cannot attribute is a threshold they cannot report.

### lines 338-339

```python
order = np.argsort(-np.nan_to_num(np.where(called, y, 0.0), nan=0.0))
```

Label the strongest, and only where a label would not land on another one. A volcano with every hit labelled is a word cloud.

### lines 345-347

```python
name = str(names.iloc[index]).strip()
```

Pandas 2 turns a missing label into the string ``"nan"`` here, while pandas 3 can preserve the float NaN. Normalise before any string operation so the same coefficient table draws on both.

### line 370

```python
text_legend(ax, [
```

TWO ENTRIES, which is the whole point of one compartment at a time.

## effect_rank

### lines 448-452

```python
ax.set_yticks([])
```

NAMES INSIDE THE PANEL, not on the axis. A y-tick label is drawn outside the axes, so a long gene id in one cell of a sheet reaches into the cell to its left -- which is what the first pass did to panel A. Inside, each name sits against its own dot and cannot collide with a neighbouring panel at any width.

## p_histogram

### line 501  _(unsure)_

```python
upper = float(np.mean(values > 0.5)) * 2.0
```

The shape, stated. A reader should not have to judge flatness by eye.

## control_separation

### lines 607-611

```python
ax.set_xticklabels([f"{label}\n(n={len(g)})"
```

THE COUNT BESIDE THE LABEL. It was in the annotation below the panel and the axis said only "pc" -- so the reader had to carry three numbers from one line to another to know that one of these groups is three points. Same request, same fix, as the interactive plot; the two must not disagree about what they show.

### lines 615-616

```python
annotate(ax, "  ".join(
```

The annotation keeps what the axis cannot: the MEDIAN of each group, which is the number the panel is actually comparing.

### lines 628-631

```python
groups={label: values for label, values
```

THE UNIT HERE IS THE COEFFICIENT, not the cell and not the well: each point is one fitted guide or gene effect. Named so the exported stats table cannot claim otherwise -- a test across the wrong unit returns p < 1e-10 on noise.

## guide_agreement

### lines 668-670

```python
rng = np.random.default_rng(0)
```

JITTERED, because guides-per-gene is an integer and agreement is a small set of fractions: without it several hundred genes stack into a dozen dots and the panel looks like it has no data in it.

## available

### lines 758-761

```python
with figure_style(theme_target()):
```

THE STYLE HAS TO BE ON BEFORE THE FIGURE EXISTS:

rcParams reach an artist when it is CREATED, so a context opened after `plt.subplots` would leave the spines, ticks and labels at the caller's globals.
