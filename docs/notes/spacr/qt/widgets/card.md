# Notes from `spacr/qt/widgets/card.py`

Prose lifted out of `spacr/qt/widgets/card.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Card.__init__

### lines 65-67

```python
self.body.setObjectName("CardBody")
```

The global `QWidget { background: bg }` rule would paint the body solid black over the card's rounded surface. Make it transparent so the card colour shows behind the content (bars, etc.).

### lines 78-80

```python
self.folder = make_foldable(title_label, self.body, name=title,
```

The BODY folds, not the card: the title has to stay to be clicked again, which is what makes the folded state a strip that names itself rather than a disappearance.
