# Notes from `spacr/ops_stitch.py`

Prose lifted out of `spacr/ops_stitch.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## stitch_well

### lines 248-253

```python
LOG.debug("could not register the pair %s-%s", a, b,
```

ONE UNREADABLE TILE IS NOT A FAILED WELL -- and the pair is

RECORDED AS REFUSED rather than dropped. A pair that failed is a fact about the acquisition and belongs in `ops_geometry`; leaving it out of the table would make "624 of 624 edges" mean two different things depending on whether a file was readable.
