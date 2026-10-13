# Notes from `spacr/resources/home/versions/_generators/parts.py`

Prose lifted out of `spacr/resources/home/versions/_generators/parts.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [elide_to_lines](#elide_to_lines) (1 entry)
- [extra_qss](#extra_qss) (2 entries)
- [resume_banner](#resume_banner) (1 entry)
- [scroll_area](#scroll_area) (1 entry)

## elide_to_lines

### lines 37-39

```python
def elide_to_lines(text: str, font: QFont, width: int, lines: int) -> str:
```

Text that never clips

## extra_qss

### lines 127-129  _(unsure)_

```python
def extra_qss(ctx: Ctx) -> str:
```

Page-level QSS for the widgets invented here

### lines 134-138

```python
page_bg = P["bg"] if ctx.theme != "space" else (
```

The Space theme paints its sky on QMainWindow, which these pages are not. Reproduce the offline fallback sky (a deep-space gradient) so a Space render is not a flat near-black rectangle. These renders never load the generated star image — it is cached per user and would make the output non-deterministic.

## resume_banner

### lines 825-827

```python
def resume_banner(ctx: Ctx, *, width: int = 0) -> QWidget:
```

Elements that do not exist on the home screen today

## scroll_area

### lines 1195-1197

```python
area.setStyleSheet("QScrollArea { background: transparent; "
```

Scoped to the scroll area itself. An unscoped `background:

transparent` here cascades to every descendant and silently strips the fill off the buttons and panels inside it.
