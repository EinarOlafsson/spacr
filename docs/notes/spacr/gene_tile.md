# Notes from `spacr/gene_tile.py`

Prose lifted out of `spacr/gene_tile.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [uniprot_accessions](#uniprot_accessions) (1 entry)
- [Module level](#module-level) (1 entry)
- [uniprot_reference](#uniprot_reference) (1 entry)
- [GeneTile.sections](#genetilesections) (1 entry)
- [_parse](#_parse) (1 entry)
- [_screen_numbers](#_screen_numbers) (1 entry)
- [gene_tile](#gene_tile) (4 entries)

## uniprot_accessions

### lines 116-118

```python
out.setdefault(gene, accession)
```

Keyed on the bare number, and the file is written with leading zeros preserved (039160), so both spellings resolve.

## Module level

### lines 154-156

```python
_GENE_TILE_UI_SOURCES = frozenset({
```

Canonical English used when a structured gene record is rendered in Qt. These strings live in the headless data model rather than in literal Qt calls, so the runtime catalog builder imports this inventory explicitly.

## uniprot_reference

### lines 368-370

```python
number = str(accession).strip()
```

THE BUNDLED MAPPING. Keyed on the gene NUMBER, so a full ToxoDB accession (`TGGT1_224750`) and a bare number (`224750`) both resolve the tile is handed either depending on where the click came from.

## GeneTile.sections

### lines 816-818

```python
rest = tuple(n for n in self.unresolved if n != self.subtitle)
```

A note already showing as the subtitle is not repeated: the tile would otherwise print its own headline twice, which reads as two different problems rather than one.

## _parse

### lines 909-915

```python
if "gene_fraction:gene[" in text or ":gene[" in text:
```

THE TERM SAYS WHICH IT IS, not the shape of the token inside it. `guide_of` returns the WHOLE bracketed token, so on a numeric screen a gene term's token has no underscore and guide == "" -- which is how "guide if there is a guide" happened to work. It stops working the moment the id itself contains an underscore: `gene_fraction:gene[TGGT1_231640]` is a GENE term whose token is `TGGT1_231640`, and it was reported as a guide of gene TGGT1.

## _screen_numbers

### lines 1084-1086

```python
frame = results
```

Deliberately not `results.copy()`: a click must not duplicate the whole coefficient table to read one row out of it. The feature column is taken as strings once and used as the index for every lookup below.

## gene_tile

### line 1213

```python
is_control = (numbers["condition"].lower() == "control"
```

the control block, which is not a gene and must not pretend to be

### lines 1244-1249

```python
numbers["gene_effect"] = float("nan")
```

The control block is fitted as if it were one gene, so a "gene-level effect" and a sign agreement exist for it arithmetically. Neither means anything — there is no gene for the guides to agree ABOUT — and printing them would dress the null distribution up as a result. The sibling guides stay: they ARE that null, and seeing the other 23 sit near zero is how a user reads one control that did not.

### lines 1305-1306  _(unsure)_

```python
genes.sort(key=lambda item: (item[0] != gene, item[0]))
```

The gene the counts were attributed to leads, then the rest in a stable order, so two runs of the same click read the same.

### lines 1317-1321

```python
if not is_toxoplasma_gene_id(gene) and not any(
```

A candidate nothing is known about, for an id that is not even shaped like a Toxoplasma gene, is not a gene record — it is the clicked string echoed back under a heading that says "identity". Drop it, so the tile says plainly that it resolved nothing instead of implying it resolved something empty.
