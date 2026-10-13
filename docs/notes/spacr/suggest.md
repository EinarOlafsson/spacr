# Notes from `spacr/suggest.py`

Prose lifted out of `spacr/suggest.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [suggest_from_scores](#suggest_from_scores) (1 entry)
- [write_suggestions](#write_suggestions) (2 entries)

## suggest_from_scores

### lines 161-162  _(unsure)_

```python
seen = sorted({int(v) for v in labels.dropna().unique()
```

The retrain encoded classes from the sorted annotation values, and a suggestion offset is not one of them.

## write_suggestions

### lines 211-226

```python
if "suggested" in suggestions.columns:
```

379-C, ANSWERED 2026-09-10 AND NOT THE WAY IT WAS ASKED. The question was "should this extend past two classes"; the answer is that it already does. `suggest_from_scores` argmaxes across N score columns and nothing in it counts to two, so classes 1 to 9 round-trip today driven in `tests/test_the_suggestion_offset_holds_more_than_two_classes.py`.

THE LIMIT IS A CLASS VALUE, NOT A CLASS COUNT, and that is what the refusal below is for. The maintainer confirmed he does not use class values of ten or more, so the offset stays at 10 and this guard stays a refusal rather than becoming a bound. The offset assumes real class values stay below it, and with a class 11 a suggested 1 and an answered 11 are the same integer -- so a bulk KEEP would rewrite somebody's class 11 to a 1 and nothing would ever say so. Refused by name rather than guarded downstream, because by the time the value is in the column the two are indistinguishable.

### lines 238-254

```python
with _connect_read_only(db_path) as db:
```

379-C, THE OTHER DIRECTION, and the one the guard above does not reach. That one refuses to SUGGEST a class at or above the offset. This refuses to write suggestions into a column that ALREADY HOLDS one, which is the case that loses a person's work rather than merely confusing a number.

`pending_suggestions` counts every value above the offset as outstanding, and `is_suggestion` reads one the same way, so a human who annotated class 11 has a row that the bulk KEEP will rewrite to a 1. PART 2's first rule is that a suggestion must never overwrite a human annotation, and this is the only route by which it still could.

SAFE TO REFUSE HERE because the screen clears outstanding suggestions before it asks for new ones -- `resolve_suggestions(keep=False)` in `annotate._on_suggest` -- so anything at or above the offset that survives to this point is a real answer and not a stale proposal. A caller that has not cleared gets told to, which is the same sentence.
