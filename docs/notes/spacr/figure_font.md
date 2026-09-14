# Notes from `spacr/figure_font.py`

Prose lifted out of `spacr/figure_font.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## use_open_sans_for_figures

### lines 75-77

```python
return _resolved
```

ALREADY DONE THIS PROCESS. The check below is a set comprehension over every font matplotlib knows, which is not free either, and this function is called before every figure is styled.

### lines 90-94

```python
if FAMILY not in available:
```

ASK BEFORE ADDING. `addfont` is expensive -- it reads the file, builds a FontProperties and resolves alternative family names, and measured on the Mask screen the eight bundled faces cost 23 SECONDS of a 13-second module open, because this ran while the settings panel was being built. A machine that already has Open Sans installed needs none of it.

### line 100

```python
continue
```

One unreadable face must not cost the other seven.
