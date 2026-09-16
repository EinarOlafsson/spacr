# Notes from `spacr/infection.py`

Prose lifted out of `spacr/infection.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [border_rules_agree](#border_rules_agree) (1 entry)
- [_load](#_load) (1 entry)
- [parasites_per_cell](#parasites_per_cell) (2 entries)
- [infection_report](#infection_report) (4 entries)

## border_rules_agree

### lines 197-200

```python
if cells is None or pathogens is None:
```

ONE RECORDED AND ONE NOT is not "they agree". The missing one defaults to False in `settings.py`, but a database that recorded only half of the pair is one we cannot make that assumption about: it may predate the other key entirely.

## _load

### lines 263-269

```python
from .tabular import _read_query
```

THROUGH THE FUNNEL. `read_query` is the canonical reader for a connection that is already open -- which is what this helper is handed, because its callers read several tables off one connection and reopening per table would change the transaction each read sees. `report=None` because an absent-or-odd column here is a fact about the run, not something to print into a report the caller is assembling.

## parasites_per_cell

### lines 307-310

```python
return out
```

No pathogen table, or one that never recorded its host: every cell is uninfected as far as this database can say. That is a real answer for an uninfected control plate and a loud one for a plate that should have parasites.

### line 319  _(unsure)_

```python
for frame in (out, counted):
```

The host key is written as a float when it arrives through a merge.

## infection_report

### lines 350-354

```python
measured_uninfected = uninfected_cells_were_measured(db_path)
```

WHETHER THE CELL TABLE IS A POPULATION OR A SELECTION, asked once per report rather than per group. False means Measure never wrote the uninfected cells, so the rate below has no denominator to be a rate over; the counts are still true and are relabelled to say what they actually counted.

### lines 360-363

```python
if border_rules_agree(db_path) is False:
```

AND WHETHER THE TWO POPULATIONS WERE FILTERED THE SAME WAY. A rate whose numerator and denominator obeyed different border rules is wrong by an amount nothing in the table reveals, so the denominator says so rather than the value being quietly off. See `border_rules_agree`.

### lines 401-405

```python
if cells_are_all_infected:
```

REFUSED, NOT PRINTED AS 1.000. Measured on the TSG101 plates, which were run with include_uninfected=False: every well reported an infection rate of exactly 1.0, which is what `infected / cells` must give when the only cells in the table are the infected ones. A reader sees a column of 1.000 and reads 100% infection.

### lines 409-415

```python
add("infection_rate", infected / cells if cells else float("nan"),
```

`cell_population`, NOT THE LITERAL IT USED TO REPEAT. This branch already knows the population is the segmented one, so the two strings were equal and the duplication was invisible until the border warning was appended to the variable and the ONE ROW THAT MOST NEEDED IT went on saying the old words. Caught by the test rather than by reading, which is the argument for naming a value once.
