# Notes from `spacr/annotation.py`

Prose lifted out of `spacr/annotation.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_keyed](#_keyed) (2 entries)
- [annotate](#annotate) (2 entries)
- [Module level](#module-level) (1 entry)
- [annotate_from_uniprot](#annotate_from_uniprot) (3 entries)

## _keyed

### lines 110-112

```python
out["_key"] = out[source_column].map(gene_number)
```

A SCRATCH NAME, not "gene_nr" directly: three of the five bundled tables already spell their key `gene_nr`, and writing the parsed key over the source column meant the very next line dropped it.

### lines 118-121

```python
return out.drop_duplicates(subset="gene_nr", keep="first")
```

KEEP THE FIRST. The alternative -- letting duplicates through -- is a row-multiplying join, and the alternative to that -- averaging text columns -- is not defined. The tables that carry numbers are already collapsed on disk; what reaches here is a handful of split models.

## annotate

### lines 275-281

```python
keyed = right[["gene_nr"] + new].rename(
```

THE JOIN KEY IS RENAMED BEFORE THE MERGE, not dropped after it. Merging on a right column called "gene_nr" while the caller's own table already has one makes pandas suffix BOTH into gene_nr_x and gene_nr_y, so the drop below found no "gene_nr" and raised KeyError. That is not a rare table: "gene_nr" is the FIRST name _key_column looks for, so it is what spaCR's own annotated output is keyed on, and re-annotating a table this function had already written crashed.

### lines 294-300

```python
hit = int(out[added].notna().any(axis=1).sum()) if len(out) else 0
```

ANY ADDED COLUMN, NOT THE FIRST ONE. This read

`out[added[0]]`, which is `gene_name` -- and a gene can be in every bundled table while having no NAME: `TGME49_200130` is one, and it carries a product description, an in-vivo fitness score and a UniProt accession. The line said "1 matched" of three rows when two had matched, which understates the join in exactly the direction that makes a reader distrust it.

## Module level

### lines 367-369  _(unsure)_

```python
UNIPROT_COLUMNS = {
```

Any organism, not only this one

## annotate_from_uniprot

### lines 454-456

```python
asked = _uniprot_key_column(frame) if key_column is None else key_column
```

THE GENES THIS TABLE NAMES, so the query can ask for them rather than for a whole proteome. Read before the key column is resolved because the key finder is cheap and the fetch is not.

### lines 475-477

```python
table["_key"] = _uniprot_keys(table.get("gene_name", ""))
```

ONE ROW PER KEY. A gene name that appears on two entries -- an isoform pair, a duplicated locus -- would otherwise turn one coefficient into two rows, which is the failure this module was written to prevent.

### line 489  _(unsure)_

```python
new = [c for c in keep if c not in frame.columns]
```

Columns the caller already computed are theirs, not UniProt's.
