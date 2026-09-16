# Notes from `spacr/resources/home/versions/_generators/variants.py`

Prose lifted out of `spacr/resources/home/versions/_generators/variants.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_patch_startup_determinism](#_patch_startup_determinism) (2 entries)
- [v01](#v01) (2 entries)
- [v02](#v02) (1 entry)
- [v20](#v20) (1 entry)
- [v26](#v26) (1 entry)

## _patch_startup_determinism

### line 150  _(unsure)_

```python
H.SystemPanel.gpu_util = staticmethod(lambda: "41%")
```

Staticmethods: assign plain functions, not lambdas taking self.

### lines 154-155  _(unsure)_

```python
H.QueuedPanel.queue_items = lambda self: []
```

An empty queue is what a fresh install shows, and it is the only queue state that does not depend on the reviewer's ~/.spacr.

## v01

### lines 183-186

```python
argument="It is the thing every other variant has to beat, and it "
```

Do not re-add "the last apps are cut off with no way to scroll to them". That was true of the pre-QScrollArea Sidebar and is now contradicted by finding 1 in VARIANTS.md, three paragraphs above where this text lands — one artefact cannot say both.

### lines 202-205

```python
from spacr.qt.app import make_home_page
```

make_home_page() rather than HomePage(...): the grouping, the stages, the notes and the icon provider are four arguments that have to agree, and a baseline that assembles its own HomePage is a render of a page that does not ship.

## v02

### lines 234-238

```python
for title, keys in CATS_STAGE5:
```

Seven columns are the measured compromise for this fixed canvas. Six columns create extra rows; eight make the tiles too narrow for current app names. The layout audit below records any resulting wrapping or elision against the current registry rather than against a historical app count.

## v20

### lines 1043-1058

```python
page = Page(ctx, margins=MARGINS, spacing=9)
```

The rent went up. This variant spends its vertical budget on the release panel and pays for it with `cats_current()` — one caption plus one grid per LIVE section — so a section costs a caption AND a full tile row even when it holds one app. The spacing below is the measured fit for the current registry on the 900 px canvas.

Paid out of tile height and inter-block spacing rather than by dropping the panel, which is the only thing this variant is for, or by cutting the captions, which is how it replaces the tabs. Widening the grid was measured and refused: at nine columns the row count does fall by one, but the tile falls to 146 px with it, and v02's note already records that a name elides below 166.

The smaller icon also returns width to the label. Growth within a partially filled row is cheap; another section costs a caption and a complete row and must be measured again.

## v26

### lines 1349-1350  _(unsure)_

```python
sec = Section(f"{title.replace('&', '&&')}  ({len(keys)})")
```

QToolButton reads a lone "&" as a mnemonic and swallows it

("Data & batch runs" renders as "Data _batch runs").
