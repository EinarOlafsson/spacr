# Notes from `spacr/qt/crash_recovery.py`

Prose lifted out of `spacr/qt/crash_recovery.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [note_that_a_launch_began](#note_that_a_launch_began) (3 entries)
- [take_the_backdrop_out_of_this_launch](#take_the_backdrop_out_of_this_launch) (1 entry)

## note_that_a_launch_began

### lines 90-99

```python
if os.path.isfile(marker):
```

isfile, NOT exists. `exists` is also True for a DIRECTORY at this path, and `os.remove` cannot delete one -- it raises IsADirectoryError, which `note_a_clean_shutdown` swallows with everything else. A stray directory here (a botched restore, a sync tool that materialises a name as a folder) was therefore read as "the last run died" on every launch, and no clean shutdown could ever clear it: the marker cannot be written either, so the next launch found the same directory and counted again. The user lost the backdrop permanently with no crash and no setting to point at. Confining the test to files we could actually have written confines the mechanism to what it is evidence of.

### line 101

```python
unclean += 1
```

The last run wrote this and never removed it.

### lines 105-110

```python
LOG.warning(
```

Something that is not a file is sitting on the name. Neither counting it nor ignoring it silently is right: this launch cannot write a marker, so a REAL crash after it will leave no evidence and recovery is off until the obstruction goes. Say so at WARNING the whole failure this entry records is a mechanism that was wrong about itself and never mentioned it.

## take_the_backdrop_out_of_this_launch

### lines 158-162

```python
os.environ["SPACR_NO_GL"] = "1"
```

ONLY THE BACKDROP. Reading every preference as its default -- what safe mode does -- would be the right response to "a saved setting is killing it" and the wrong one here: the evidence points at the backdrop, and silently resetting the user's language, theme and paths to diagnose a driver crash is a bigger surprise than the crash.
