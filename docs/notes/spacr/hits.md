# Notes from `spacr/hits.py`

Prose lifted out of `spacr/hits.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [family_labels](#family_labels) (1 entry)
- [Module level](#module-level) (1 entry)
- [load_results](#load_results) (2 entries)
- [load_gene_metadata](#load_gene_metadata) (2 entries)
- [join_metadata](#join_metadata) (1 entry)
- [build_hit_list](#build_hit_list) (4 entries)

## family_labels

### lines 188-189  _(unsure)_

```python
token = series.str.extract(_BRACKET.pattern, expand=False)
```

Keep this vectorized for interactive plot restyling. Explicit term labels take precedence over the identifier suffix, matching ``guide_of``.

## Module level

### line 286  _(unsure)_

```python
_GENE_ID_PREFIXES = ("TGGT1", "TGME49", "TGVEG", "TGRH88", "TGARI",
```

Known VEuPathDB prefixes whose underscore is part of the gene accession.

## load_results

### lines 435-437

```python
def load_results(folder: Union[str, os.PathLike]) -> Dict[str, pd.DataFrame]:
```

Reading what a regression run wrote

### lines 458-460

```python
found[role] = tabular.read_table(path, report=None)
```

ONE READER, so a results CSV whose header says `column` is offered to the caller as `columnID` -- the name the joins in this module key on.

## load_gene_metadata

### lines 506-509

```python
frame = tabular.read_table(target, canonicalise=False, report=None)
```

canonicalise=False: `key` is the caller's column name in a curated third-party export ('Gene ID'), not spaCR metadata, and a header the vocabulary renamed out from under the caller would raise the KeyError below on a file that was fine.

### lines 520-522

```python
unparsed = int(frame["gene"].isna().sum())
```

A NaN key must never act as a join key: pandas treats NaN as equal to NaN, so every unparsable metadata row would fan out against every unparsable result row.

## join_metadata

### line 571, trailing  _(unsure)_

```python
if len(joined) != before:
```

validate="many_to_one" already raises

## build_hit_list

### lines 1103-1107

```python
alpha = DEFAULT_SELECTION_THRESHOLD
```

The threshold means the opposite thing on this branch, so the default has to change with it: 0.05 as a SELECTION FREQUENCY is "chosen in one bootstrap in twenty", which would report almost the whole design as significant. An alpha the caller passed explicitly is theirs and is left alone.

### lines 1132-1135

```python
from .annotation import annotate
```

AFTER the user's files, deliberately. `annotate` leaves a column that is already there alone, so this order means a name the user supplied always wins over the bundle's -- which is the precedence anybody would expect from a file they passed by hand.

### line 1139, trailing  _(unsure)_

```python
if len(joined) != before:
```

annotate's many_to_one join raises

### line 1147, trailing  _(unsure)_

```python
if len(joined) != len(table):
```

validate="many_to_one" already raises
