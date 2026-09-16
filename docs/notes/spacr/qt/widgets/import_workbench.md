# Notes from `spacr/qt/widgets/import_workbench.py`

Prose lifted out of `spacr/qt/widgets/import_workbench.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ImportWorkbench.add_files](#importworkbenchadd_files) (2 entries)
- [ImportWorkbench._files_found](#importworkbench_files_found) (1 entry)
- [ImportWorkbench._show](#importworkbench_show) (1 entry)
- [ImportWorkbench._rebuild_roles](#importworkbench_rebuild_roles) (1 entry)
- [ImportWorkbench.refresh](#importworkbenchrefresh) (1 entry)
- [ImportWorkbench._fill_the_table](#importworkbench_fill_the_table) (1 entry)
- [ImportWorkbenchDialog.done](#importworkbenchdialogdone) (1 entry)

## ImportWorkbench.add_files

### lines 277-285

```python
self._scan_trouble = ""
```

SAID BEFORE THE WALK, not after: on a sleeping share the walk is the part that takes seconds, and a panel that says nothing in the meantime looks like a drop that was ignored. `refresh` replaces it with the plan summary the moment there is one -- and `_walk` makes sure there IS one even when the walk fails.

THE CATALOGS ALREADY CARRY "Working…" in all nine languages. A wordier caption invented here would be English-only until someone noticed, and `tests/qt/test_i18n_caption_ratchet.py` fails on it.

### lines 289-293

```python
self.refresh()
```

`submit` answers False only for a job that ran INLINE -- a runner built `threaded=False`, which is how some tests drive this panel -- and whose handler raised. Nothing is coming to replace the caption above, so put the summary back rather than leave the panel claiming to be working.

## ImportWorkbench._files_found

### lines 306-307

```python
self._show(self._files + [p for p in (found or ()) if p not in seen])
```

`_show`, NOT `set_files`: `set_files` cancels, and a second drop landing must not abandon the first drop's walk.

## ImportWorkbench._show

### lines 366-368

```python
if self._files and not self.regex.text().strip():
```

A REGEX PROPOSED FOR THE OLD SET IS NOT PROPOSED FOR THIS ONE, so the first drop offers one and a later drop does not overwrite what the user has since edited.

## ImportWorkbench._rebuild_roles

### lines 429-431

```python
chosen = self._roles.get(group, group if group in known else "")
```

THE GROUP'S OWN NAME IS THE DEFAULT when it is already a role: a proposal that named its groups `wellID` should not make the user say so again.

## ImportWorkbench.refresh

### lines 473-475

```python
said += (f" · Could not read what you dropped: "
```

SAID, NOT SWALLOWED. A walk that failed leaves the table short by a whole folder, and a summary that counts only what did arrive reads as if that folder had held nothing.

## ImportWorkbench._fill_the_table

### lines 507-509

```python
for offset, name in enumerate(missed):
```

UNMATCHED LAST AND NAMED, never dropped in silence: "412 of 480 matched" with the other 68 listed is an answer, and 412 files appearing without comment is how half a plate goes missing.

## ImportWorkbenchDialog.done

### line 579, trailing  _(unsure)_

```python
def done(self, result: int) -> None:
```

Qt override
