# Notes from `spacr/regex_infer.py`

Prose lifted out of `spacr/regex_infer.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [hint_for](#hint_for) (1 entry)
- [_proposal_for](#_proposal_for) (2 entries)
- [_assign_roles](#_assign_roles) (3 entries)
- [propose](#propose) (2 entries)

## hint_for

### lines 81-84

```python
run = str(before or "")
```

THE TRAILING ALPHA RUN ONLY, stopping at the first non-letter. Reading every letter in the run made `plate1_` end in "PLATE" and claim the well letter after it as a plate id -- the separator is exactly what says the word is not about this slot.

## _proposal_for

### line 317  _(unsure)_

```python
pieces: List[dict] = []          # {"literal": str} or {"slot": FieldEvidence}
```

pass one: what varies, and what literal sits in front of it

### line 318, trailing  _(unsure)_

```python
pieces: List[dict] = []
```

{"literal": str} or {"slot": FieldEvidence}

## _assign_roles

### line 462  _(unsure)_

```python
for info in slots:
```

1. a well is a well, whatever is in front of it.

### line 469  _(unsure)_

```python
for info in slots:
```

2. the vendor letters, which are the strongest thing a name carries.

### line 477  _(unsure)_

```python
for info in slots:
```

3. and only then the shape of the values themselves.

## propose

### lines 531-533

```python
if sum(1 for info in proposal.fields.values() if not info.numeric
```

TWO MICROSCOPES ARE NOT ONE FAMILY. A mask family varying in many non-digit positions has merged unrelated names, and its regex would match everything while meaning nothing.

### lines 539-541

```python
proposals.sort(key=lambda p: (p.matched, len(p.fields)), reverse=True)
```

By coverage, then by how many groups it found: between two patterns that match every file, the one that pulled out more metadata is the more useful answer and the one a user can always simplify.
