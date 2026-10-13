# Notes from `spacr/qt/widgets/timelapse_movie.py`

Prose lifted out of `spacr/qt/widgets/timelapse_movie.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [FilmStrip.__init__](#filmstrip__init__) (1 entry)
- [FovMovie.__init__](#fovmovie__init__) (1 entry)
- [FovMovie.show_frame](#fovmovieshow_frame) (1 entry)
- [TimelapseMoviePanel.__init__](#timelapsemoviepanel__init__) (1 entry)

## FilmStrip.__init__

### lines 114-116

```python
self.viewport().setAutoFillBackground(False)
```

Scaffolding: it positions thumbnails and must paint nothing, or it is one more opaque rectangle over the page. See `spacr.qt.theme.make_transparent`.

## FovMovie.__init__

### lines 196-197  _(unsure)_

```python
self._strip = FilmStrip(self)
```

The strip lives ABOVE the movie and starts collapsed: the request was that clicking the movie "expands upwards into a row of frames".

## FovMovie.show_frame

### line 358  _(unsure)_

```python
self._scrub.blockSignals(True)
```

Without the guard this re-enters through `valueChanged`.

## TimelapseMoviePanel.__init__

### lines 521-523

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.
