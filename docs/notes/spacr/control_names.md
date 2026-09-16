# Notes from `spacr/control_names.py`

Prose lifted out of `spacr/control_names.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [common_prefix](#common_prefix) (1 entry)
- [resolve_control](#resolve_control) (2 entries)
- [matches](#matches) (3 entries)
- [rows_for](#rows_for) (1 entry)

## common_prefix

### lines 84-93

```python
if len(name.split(SEPARATOR)) >= 3:
```

THREE COMPONENTS OR IT IS NOT A SPECIES TAG. Caught by pointing this at a one-guide library: "000000_1" made "000000" look like a prefix carried by 100% of the names, so the GENE was read as the organism and the control then matched nothing.

spaCR's own convention is <org>_<gene>_<guide>, and `process_reads` splits on exactly that -- three parts, no fewer. So the gene is always the SECOND-TO-LAST component, and a two-part name's head is a gene, never a species. Only three-part names can contribute a candidate prefix.

## resolve_control

### lines 124-127

```python
prefix = common_prefix(() if names is None else names)
```

NOT `names or ()`. A pandas Series is the natural thing to pass here -- guide names live in a column -- and its truthiness raises "The truth value of a Series is ambiguous" rather than falling back. Caught the first time this was pointed at a real library.

### lines 133-134  _(unsure)_

```python
return ControlSpec(text, GUIDE, SEPARATOR.join(parts[1:]), prefix)
```

Everything after the FIRST underscore is the guide, which is exactly how process_reads stores a three-component name.

## matches

### lines 191-195

```python
head = f"{spec.prefix}{SEPARATOR}" if spec.prefix else ""
```

THE PREFIX AGAIN. `resolve_control` strips the measured organism prefix off what the user typed, so `TGGT1_000000_11` becomes the guide `000000_11` -- and the count table spells that guide `TGGT1_000000_11`. Comparing the stripped value against the unstripped column matched nothing, for either spelling.

### lines 201-213

```python
head = f"{spec.prefix}{SEPARATOR}" if spec.prefix else ""
```

THE PREFIX HERE TOO, AND THIS BRANCH IS THE ONE THAT WAS MISSED. The two branches on either side of it carry long comments about `resolve_control` stripping the measured organism prefix off what the user typed; both were fixed and this one, between them, was not. A gene column that keeps its prefix -- `TGGT1_220950` against a spec whose value is `220950` -- matched NOTHING.

It is the branch a screen WITH a gene column takes, so this is not an edge: `controls`, `positive_control_id` and `negative_control_id` all arrive through `rows_for`, and a gene-level control on such a screen selected zero rows in silence. Found on 2026-08-21 while excluding a contaminant by gene name, which is the same code path.

### lines 217-235

```python
head = f"{spec.prefix}{SEPARATOR}" if spec.prefix else ""
```

No gene column: a guide belongs to the gene its name CONTAINS as its middle component.

THE PREFIX IS PART OF THE NAME AND THIS IGNORED IT. `startswith` alone asks whether the guide begins `000000_`, and the guides of a real count table begin `TGGT1_000000_` -- the organism prefix that `resolve_control` has already MEASURED and is carrying on the spec. So a gene control matched nothing on every library whose names keep their prefix, which is every count table `process_reads` writes.

Measured on the example screen: `rows_for('000000', guides)` reported "resolved to gene '000000': 0 guide(s)" against 30 guides that are exactly that gene's. 184 recorded "all four spellings reach the same 28 guides" and that was measured WITH a gene column beside the guides; this path -- the one a count table actually takes -- was never exercised on prefixed names.

AND ZERO IS NOT LOUD. A control that selects nothing leaves the thresholds to fall back and the baseline at zero, and the run finishes.

## rows_for

### lines 299-312

```python
if not found and SEPARATOR in spec.typed:
```

THE DATA HAS ALREADY DROPPED THE PREFIX, and the user has not. `process_reads` stores `TGGT1_000000_11` as guide `000000_11`, so the names in hand carry no organism token at all and `common_prefix` measures "" from them -- correctly. A control pasted from the LIBRARY file still says `TGGT1_000000`, whose head is then not a known prefix and which reads as a two-part guide that matches nothing.

Measured on the maintainer's screen: 'TGGT1_000000' found 0 where '000000' found 28, which is the same control written the way the library writes it.

So: when nothing matched and the first token is absent from every name in hand, try again without it. Still WHOLE values -- this drops a leading token, it does not go back to substring matching.
