# Notes from `spacr/barcode_search.py`

Prose lifted out of `spacr/barcode_search.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [_decide](#_decide) (1 entry)

## Module level

### lines 124-133

```python
MIN_ENRICHMENT = 3.0
```

THE THRESHOLDS, AND THE MEASUREMENTS THAT SET THEM.

These were chosen against a real paired run whose two mates were measured read by read, rather than picked for roundness.  On that run the tables that were genuinely part of the library reached enrichments of nineteen, twenty nine and several million over their own chance rates, while every table that was truly absent sat between a fifth of its chance rate and one and a fraction times it, the highest coincidence reaching one point zero two.  A cut at three leaves the coincidence ceiling far below it and the weakest true signal far above it, so neither side is close to the boundary.

### lines 136-142

```python
MIN_USABLE_RATE = 0.10
```

The same run also carried a small number of reads in the wrong orientation, about one guide barcode in six hundred, which index hopping and chimeric fragments produce in every pooled run.  Those hits are thousands of times above chance and completely real, and mapping from them would still be hopeless.  A barcode that belongs to the construct appears in most reads, so a table found in under a tenth of them is reported honestly as enriched but not usable rather than being offered as a source of settings.

### lines 145-150

```python
MAX_OFFSET_SPAN = 6
```

The narrowest run of offsets accounting for most of the hits was one to three bases wide for every true finding, the width above one coming from guide sequences of twenty or twenty one bases shifting everything downstream of them by a base.  For the adapter that masqueraded as a column table the same measurement needed forty five bases.  Six bases allows a construct with more length variation than this one while still refusing anything adapter shaped.

## _decide

### lines 950-953

```python
return ABSENT, (
```

Deciding this by the ratio alone fails for a table whose coincidence rate rounds to nothing, such as a single long anchor sequence, because the bar the observation has to clear is then also nothing and no observation can fall below it.  Nothing matched, so nothing is there.
