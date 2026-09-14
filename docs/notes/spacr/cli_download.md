# Notes from `spacr/cli_download.py`

Prose lifted out of `spacr/cli_download.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [resolve_selection](#resolve_selection) (2 entries)
- [_is_interactive](#_is_interactive) (1 entry)
- [fetch_piece](#fetch_piece) (2 entries)
- [download](#download) (2 entries)
- [cmd_download](#cmd_download) (1 entry)

## resolve_selection

### lines 243-247

```python
take_screen = screen is not None or bool(plates)
```

NAMING A SCREEN FILTER IS NAMING THE SCREEN, read from both ends. `--screen crops` on its own must not fall through to the default and download the examples instead, and `mask --screen crops` must not silently drop the filter -- both would give a user who typed the word "screen" the one thing they cannot have meant.

### lines 276-278

```python
wanted_examples.sort(key=EXAMPLE_SETS.index)
```

Listing order, not typing order: the sets read as a pipeline and the screen reads as a table, and neither should be shuffled by the order somebody happened to type two names in.

## _is_interactive

### line 477, trailing  _(unsure)_

```python
except (AttributeError, ValueError):
```

a closed or exotic stdin

## fetch_piece

### lines 551-553

```python
chunk_size=1 << 20)
```

A MEGABYTE, not the 32 KB the per-file demo pull uses: these are archives of gigabytes, and the small chunk buys nothing on a body that is never displayed as it arrives.

### line 558  _(unsure)_

```python
expand_measure_arrays(piece.folder / "merged")
```

The .npz compression is a transport detail; Measure reads .npy.

## download

### lines 596-600

```python
failures.append((piece, exc))
```

STOP, rather than skip to the next piece. Ctrl-C on the second of eight plates means all eight, not "abandon this 8 GB and start the next 8 GB". Nothing partial is kept: the archive is still a `.part` file, which `download_archive` removes on its way out.

### lines 618-622

```python
try:
```

LAST, AND ONCE PER FOLDER. A measurements database stores absolute paths to its crops; the published copy stores them relative to the dataset root so it is portable. This is what turns them back into paths that open -- and it has to run after the crops are there, not between two pieces of the same plate.

## cmd_download

### lines 701-703

```python
folders: List[Path] = []
```

THE PATHS, NAMED. What a user does next is put one of these in `src`, and a summary that said only how many gigabytes arrived would leave them to work out where from the flags they typed.
