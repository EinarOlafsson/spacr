# Notes from `spacr/qt/widgets/gene_panel.py`

Prose lifted out of `spacr/qt/widgets/gene_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [warm_annotation](#warm_annotation) (1 entry)
- [GenePanel.__init__](#genepanel__init__) (2 entries)
- [GenePanel.warm_for](#genepanelwarm_for) (1 entry)
- [GenePanel._annotation_loaded](#genepanel_annotation_loaded) (1 entry)
- [GenePanel._render_known](#genepanel_render_known) (1 entry)
- [GenePanel._shut_down_warming](#genepanel_shut_down_warming) (1 entry)

## warm_annotation

### lines 102-104

```python
LOG.debug("gene panel: could not warm the gene_tile indices",
```

The warm-up is an optimisation. A reference file this install does not have must not stop the panel from opening -- the click path says what it could not resolve, which is the answer either way.

## GenePanel.__init__

### lines 170-171

```python
self._known.setProperty("i18nSkipText", True)
```

A gene id must survive translation intact: TGGT1_239740 is not a phrase, and a catalog that "translated" it would be renaming a gene.

### lines 208-212

```python
application = QApplication.instance()
```

BELT AND BRACES ON THE THREAD'S LIFETIME. Qt aborts the process if a running QThread is destroyed, and a panel can be dropped without ever being closed -- a tab rebuilt, a screen replaced, an interpreter shutting down. `closeEvent` covers the ordinary path; this covers the one where nobody closed anything.

## GenePanel.warm_for

### lines 297-302

```python
self._pending_warm = terms
```

HELD UNTIL THE PANEL IS SHOWN. A frame can arrive before anybody looks at the tab -- the screen loads a run's results into every tab at once -- and starting a thread for a panel that is never shown is what aborted the process: Qt calls abort() when a running QThread is destroyed, and a panel that is never shown is never closed either, so neither `closeEvent` nor `aboutToQuit` ever fires.

## GenePanel._annotation_loaded

### lines 314-315

```python
self.show_feature(self.summary.feature)
```

A click that beat the warm-up is re-answered rather than left showing "loading" over a tile that is already on screen.

## GenePanel._render_known

### lines 389-392

```python
parts.append("<h3 style='margin-bottom:0'>what spaCR knows "
```

Named per gene, because the whole point of the ambiguous case is that these blocks are alternatives and not one record: three products under one heading would read as one protein with three names.

## GenePanel._shut_down_warming

### lines 517-518  _(unsure)_

```python
pass
```

The C++ half can already be gone when a whole window closes at once; there is nothing left to shut down and nothing to report.
