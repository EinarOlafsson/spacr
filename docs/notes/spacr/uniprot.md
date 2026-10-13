# Notes from `spacr/uniprot.py`

Prose lifted out of `spacr/uniprot.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [near_misses](#near_misses) (1 entry)
- [_next_page](#_next_page) (1 entry)
- [fetch_genes](#fetch_genes) (1 entry)
- [annotation_for](#annotation_for) (2 entries)

## Module level

### line 146, trailing  _(unsure)_

```python
"toxoplasma gondii": 508771,
```

ME49

### line 157, trailing  _(unsure)_

```python
"plasmodium falciparum": 36329,
```

3D7

### lines 181-183

```python
"sarcocystis neurona": 42890,
```

The three tissue-cyst and intestinal apicomplexans the genus name alone was missing. Each points at the species a screen would be run on rather than at the genus taxon, which carries no proteome.

## near_misses

### lines 290-291

```python
head = name.split()[0]
```

A GENUS ON ITS OWN is the common miss -- "plasmodium", "leishmania" and difflib does not rate a prefix highly against a two-word name.

## _next_page

### lines 356-362

```python
found = re.search(r'<([^>]+)>\s*;\s*rel="next"', str(link_header or ""))
```

NOT split(","). A Link header is comma-separated, but the URL inside it carries `fields=accession,id,protein_name,...` with literal commas, so splitting cut the URL into fragments and the one holding `rel="next"` had lost its opening bracket. The header parsed as having no next page, every fetch stopped at 500 rows, and a "proteome" was the first five hundred entries of one -- which is why a human screen looking for TP53 matched nothing.

## fetch_genes

### lines 486-487

```python
clauses = " OR ".join(
```

gene: AND accession, because a screen library names its targets one way or the other and neither spelling is wrong.

## annotation_for

### lines 533-535

```python
frame = fetch_genes(resolution, genes, cache_dir=cache_dir)
```

ASK FOR WHAT IS NEEDED FIRST. A table naming a few genes gets a query naming those genes, which is seconds; only a table naming more than `TARGETED_MAX` of them pulls the whole proteome.

### lines 541-550

```python
frame = fetch(resolution, cache_dir=cache_dir, reviewed=False)
```

NO REVIEWED ENTRIES IS NORMAL for most parasites. Swiss-Prot has a few hundred proteins for Plasmodium and none at all for some strain-level taxa -- Babesia bovis and Leishmania major both came back empty on the reviewed query while having thousands of unreviewed entries. Refusing to annotate those organisms because nobody has curated them by hand would make the field useless for exactly the organisms it was asked for.

So the fallback is taken and SAID, because a TrEMBL annotation is a prediction and the reader is entitled to know which they have.
