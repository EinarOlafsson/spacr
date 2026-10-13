# Notes from `spacr/organelle_types.py`

Prose lifted out of `spacr/organelle_types.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (4 entries)
- [preset_for](#preset_for) (1 entry)
- [organelle_number](#organelle_number) (1 entry)
- [organelle_role_of](#organelle_role_of) (1 entry)
- [_count_implied_by_the_slots](#_count_implied_by_the_slots) (1 entry)

## Module level

### lines 118-121

```python
ORGANELLE_TYPES: Dict[str, OrganelleType] = {
```

The table. One row per category the maintainer listed, with the rows where SIZE DECIDES marked, and the rows with no dedicated detector said out loud.

### lines 139-141

```python
size_split=("spots", "ring"),
```

Split, and this is the row that proves the mapping is not one-to-one: a transport vesicle is a dot and a vacuole is a ring, and both are on the maintainer's own Vesicular list.

### line 231  _(unsure)_

```python
"toroidal": OrganelleType(
```

no dedicated detector, and the file says so

### lines 423-443

```python
NUMBER_OF_ORGANELLES = "number_of_organelles"
```

THE SLOTS. How many organelles a run has, and what each one's keys are called.

An organelle SLOT is one segmented object with its own channel, its own type preset and its own copy of every detection setting. How many exist is a setting -- :data:`NUMBER_OF_ORGANELLES` -- and the keys of each slot are GENERATED from it rather than written out, so two gives two slots and seven gives seven.

LOWERING THE NUMBER HIDES SLOTS, IT DOES NOT DELETE THEM, which is why there are two answers to "which slots are there" below. :func:`active_organelle_roles` is what a panel shows and :func:`declared_organelle_roles` is what a settings dict must keep: a file written at seven and opened at two shows two and carries seven, so the values ride along untouched and putting the number back brings the old answers with it. A count that erased what it could not currently show would punish a user for trying a smaller number, which is the opposite of a setting worth exploring. `spacr.settings` generates its type, tooltip and category registries for every slot :data:`MAX_ORGANELLES` allows, for the same reason: a hidden slot still has to be readable.

## preset_for

### lines 322-325

```python
if preset.method and preset.method in LEGAL_METHODS[morphology]:
```

The method has to be legal for the morphology that SIZE just chose, not for the one the table lists first. A Vesicular preset recommending 'log' is fine for spots and fine for ring; one recommending 'ridge' would raise the moment the user's diameter tipped it into a ring.

## organelle_number

### lines 573-574  _(unsure)_

```python
offset = 0
```

The inverse of `_carried_suffix`: earlier lengths are used up before this one starts, so their totals are added back.

## organelle_role_of

### lines 603-609

```python
head, separator, _rest = text.partition("_")
```

THE ROLE IS READ OFF THE KEY, not searched for among every role there could be. Scanning `_ROLE_MATCH` was O(roles x keys), and with the ceiling raised from 26 to 702 that is a million string comparisons to answer a question about one settings dict -- 4 ms per call, on a path the settings form takes repeatedly. A key is `<role>_<question>`, so the role is the text before the first underscore and `organelle_number` is what decides whether it is a real one.

## _count_implied_by_the_slots

### lines 690-691

```python
highest = 0
```

ONE PASS OVER THE KEYS, not one pass per role. See

`organelle_role_of` for why: the roles are no longer a short list.
