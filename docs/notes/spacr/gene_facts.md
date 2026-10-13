# Notes from `spacr/gene_facts.py`

Prose lifted out of `spacr/gene_facts.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_show](#_show) (1 entry)
- [facts_for](#facts_for) (1 entry)

## Module level

### lines 118-120

```python
"fit_invitro_hff": "in vitro (HFF)",
```

The seven screens bundled today, respelled but NOT interpreted: "PE" stays "PE" because expanding an abbreviation the source file does not expand is how a caption ends up asserting an experiment nobody ran.

## _show

### lines 211-213

```python
return f"{int(value):d}"
```

`n_transmembrane` and `signal_peptide_length` arrive as floats because their column has blanks in it. "7.0 helices" is a pandas detail leaking onto a figure caption.

## facts_for

### lines 488-492

```python
joined = annotation.annotate(pd.DataFrame({"gene": wanted}),
```

`gene`, NOT `gene_nr`: `annotate` merges every source with right_on="gene_nr", so a left column of that name collides, pandas renames both halves, and the tidy-up then drops a column that is no longer there. Reported for a fix; avoided here so the tile does not have to wait for one.
