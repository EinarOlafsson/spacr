# Notes from `spacr/qt/module_hints.py`

Prose lifted out of `spacr/qt/module_hints.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [module_hint_text](#module_hint_text) (2 entries)
- [_ModuleHints._show](#_modulehints_show) (1 entry)
- [_ModuleHints.eventFilter](#_modulehintseventfilter) (2 entries)

## module_hint_text

### line 74, trailing  _(unsure)_

```python
except RuntimeError:
```

the C++ half has gone

### lines 83-85

```python
cut = line[:MAX_HINT_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:—-")
```

Cut on a word so the tail is not half a word, and mark it so the reader knows there is more rather than thinking the sentence ends oddly.

## _ModuleHints._show

### line 113, trailing  _(unsure)_

```python
except RuntimeError:
```

the C++ half has gone

## _ModuleHints.eventFilter

### lines 162-165

```python
if not self._shows_its_own_name(watched):
```

AN ICON-ONLY BUTTON KEEPS ITS POPUP. The description still goes to the status bar -- that is what was asked for -- but the popup is not suppressed, because a button with no label has nothing else to identify it with.

### lines 168-170

```python
return landed
```

SUPPRESSED ONLY IF IT LANDED SOMEWHERE. A window with no status bar would otherwise lose the description entirely, which is worse than the popup this replaces.
