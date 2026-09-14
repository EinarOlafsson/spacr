# Notes from `spacr/report.py`

Prose lifted out of `spacr/report.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_read_csv_head](#_read_csv_head) (1 entry)
- [_find_artifacts](#_find_artifacts) (1 entry)
- [_settings_point_at](#_settings_point_at) (1 entry)
- [_read_stamps](#_read_stamps) (1 entry)
- [_field_qcs_from_csv](#_field_qcs_from_csv) (1 entry)
- [_collect_plate_qc](#_collect_plate_qc) (1 entry)
- [_collect_appendix](#_collect_appendix) (1 entry)
- [collect_report](#collect_report) (2 entries)
- [_table_text](#_table_text) (1 entry)
- [write_pdf](#write_pdf) (2 entries)

## _read_csv_head

### lines 403-406

```python
if any("\x00" in cell for cell in row):
```

Python's CSV reader no longer rejects embedded NUL bytes on every supported version. Treat them as the documented half-written-file boundary before counting or displaying the corrupt row.

## _find_artifacts

### lines 605-606  _(unsure)_

```python
roots: List[Tuple[Path, bool]] = [(src, False)]
```

Top level + the folders that hold results. measurements/ holds the database; qc/ the scorecards; settings/ the settings CSVs.

## _settings_point_at

### line 729  _(unsure)_

```python
try:
```

A CSV round-trip turns a list of plates into its repr.

## _read_stamps

### lines 741-743

```python
def _read_stamps(paths: Sequence[Path]) -> Tuple[List[Tuple[Path, Dict[str, Any]]], List[str]]:
```

Section: run status  (first, always, because failure must not be buried)

## _field_qcs_from_csv

### lines 1152-1155

```python
return [], f"{path.name} is not readable as CSV ({exc})"
```

A damaged scorecard is reported as unreadable rather than summarised from the rows that did parse: a plate verdict derived from half a scorecard is a different verdict, and this module does not invent one.

## _collect_plate_qc

### lines 1343-1345

```python
pass
```

Damaged past some row: the wells counted so far are real, and a layout export that cannot be parsed to the end must not take the whole report down with it.

## _collect_appendix

### lines 1738-1740

```python
have_something = True
```

THE FILE EXISTS, BUT MEASURE WROTE NO FEATURES. This is neither an absent database nor a describer error, and omitting it makes a run interrupted before its first table look complete.

## collect_report

### lines 1925-1928

```python
pass
```

RuntimeError is what pathlib raises for a symlink loop (ELOOP), and this function promises never to raise for bad input: an unresolvable folder is reported as "does not exist", not as a traceback in the caller's face.

### lines 1947-1949

```python
try:
```

Third-party report chapters are inserted relative to stable core keys. A failing builder becomes a visible problem chapter rather than silently disappearing from the report.

## _table_text

### line 2347

```python
while sum(widths) + 2 * n_columns > width and max(widths) > 8:
```

Squeeze the widest column when the row would overflow the page.

## write_pdf

### lines 2440-2442

```python
import matplotlib.image as mpimage
```

`import matplotlib` does NOT bind `matplotlib.image`; reaching it that way raises AttributeError, which the per-figure guard below would swallow into "could not be drawn" on *every* page. Import it by name.

### lines 2469-2472

```python
pdf.savefig(figure)
```

`PdfPages.savefig`, NOT a Figure's (108 point 6). These are pages appended to a multi-page book, and the book's format is named by the caller that asked for a report; there is no single file here for a format preference to rename.
