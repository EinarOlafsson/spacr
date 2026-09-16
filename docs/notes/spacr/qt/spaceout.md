# Notes from `spacr/qt/spaceout.py`

Prose lifted out of `spacr/qt/spaceout.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## main

### lines 21-28

```python
from . import _prefer_a_context_the_shaders_can_run_on
```

Before the application: `run` builds the stylesheet from

`theme.palette_for`, and the palette has to already be re-hued by then or the first window paints in the undressed colours and only later screens pick the new ones up. BEFORE THE THEME, which resolves fonts: Qt reports "OpenType support missing for Open Sans" while a face is being loaded, and `run` does not install the filter until later -- so the warnings that leaked were the ones emitted on the way in.
