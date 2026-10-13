# Notes from `spacr/qt/screens/embeddings.py`

Prose lifted out of `spacr/qt/screens/embeddings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [EmbeddingsScreen.__init__](#embeddingsscreen__init__) (4 entries)
- [EmbeddingsScreen._on_embedded](#embeddingsscreen_on_embedded) (1 entry)
- [EmbeddingsScreen._fill_preview](#embeddingsscreen_fill_preview) (2 entries)
- [Module level](#module-level) (1 entry)

## EmbeddingsScreen.__init__

### lines 125-133

```python
self._policy.addItem("Per channel (one pass per stain)", "per_channel")
```

THE COST IS IN THE CAPTION, not only the tooltip. 386 asks for this choice to be explicit; a user who cannot see what it costs will pick whichever is first and never revisit it. SHORT ENOUGH TO TRANSLATE. These are runtime catalog rows, and the machine translator returns long clause-heavy English unchanged the earlier five-line tooltip failed the zh_CN gate outright. One idea per sentence, which a tooltip wants anyway. The PASS COUNT stays in the caption: 386 asks for the cost to be visible where the choice is made, and a test pins it.

### lines 136-141

```python
self._policy.setToolTip(
```

VERIFIED AGAINST THE zh_CN MODEL BEFORE BEING WRITTEN. The M2M checkpoint returns a string UNCHANGED -- not an error -- when "channel" and "stain" appear in the same row, and the catalog audit then reports it as "remains exact English". Six variants were run through `_translate_batches` to find that; this wording avoids the pair and comes back as Chinese. See instruction 394.

### lines 177-181

```python
self._table = install_sorting(QTableWidget(0, 0, self))
```

install_sorting + table_item, like every other view in the app. A preview of eight dimensions is exactly the table someone sorts "which objects score highest on dimension 3" is the only question a raw embedding column can answer by eye -- and Qt's default sort is lexicographic, so -0.0412 would rank above 0.9031.

### lines 199-203

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into: a tooltip that only appears over the control is one the user meets after they have already decided what to put in it. One post-pass rather than a convention every hand-built row has to remember -- the same call `live_preview.py` ends with.

## EmbeddingsScreen._on_embedded

### lines 284-288

```python
frame = pd.DataFrame(np.asarray(result.values),
```

`EmbeddingResult` carries the matrix and the names separately and offers `to_frame(object_ids)`. The screen has no object ids -- it was handed a crop stack, not a table -- so it builds the frame from the two directly rather than inventing ids that would then look like a join key.

## EmbeddingsScreen._fill_preview

### lines 311-327

```python
self._table.setSortingEnabled(False)
```

SORTING OFF ACROSS THE FILL, which is what every other table in the app does (project_browser.py:427, run_history.py:447) and what this one was missing.

The table is sorted, and with a sort active every `setItem` into the sorted column does a sorted RE-INSERTION -- it moves the row it was just handed. The loop then writes that row's remaining columns at an index now holding a different object, so one line ends up carrying dimensions from two objects, and a cell from the PREVIOUS run survives where nothing was written. Nothing on screen says so.

`_SortState` does try to suspend itself during a fill, but only on `rowsInserted`/`rowsRemoved`, and a second Embed always repeats the shape -- rows is `min(len(frame), 50)` over the same crops, columns are capped at PREVIEW_DIMENSIONS -- so those never fire. Clearing first is not enough either, measured: the sort is re-applied as the rows go back in. Turning sorting off is the only thing that holds.

### lines 336-338

```python
value = float(frame.iloc[row][column])
```

The displayed text is rounded to four places; the SORT

KEY is the float, so two dimensions that both print -0.0000 still order by what they actually are.

## Module level

### lines 376-378

```python
_ROW = declared_app(APP_KEY)
```

The row is declared in `spacr.qt.app_catalog`, read back here rather than restated, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.
