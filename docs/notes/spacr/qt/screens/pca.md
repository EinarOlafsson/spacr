# Notes from `spacr/qt/screens/pca.py`

Prose lifted out of `spacr/qt/screens/pca.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [PCAScreen.__init__](#pcascreen__init__) (6 entries)
- [PCAScreen.load_path](#pcascreenload_path) (1 entry)
- [PCAScreen.closeEvent](#pcascreencloseevent) (1 entry)
- [Module level](#module-level) (1 entry)

## PCAScreen.__init__

### lines 105-107

```python
self._jobs = JobRunner(self, threaded=threaded, app_key="pca")
```

Every table read goes through here, so it never runs on the GUI thread and always shows up in the run registry (and so in the background-activity spinner).

### lines 162-163

```python
self.pca = PCAPanel(self, link=link, threaded=threaded)
```

The sklearn fit is 1.63 s on a 200 000-row table; the panel runs it on a worker when the screen does its reads on one.

### lines 168-175

```python
from ..preferences import scaled_px
```

SCALED, NOT A DEVICE-PIXEL CONSTANT. This cap exists to stop the settings column eating the figure beside it, and 320 px is the right answer at 100 %% -- and only there. The glyphs inside it double at 200 %% and the box did not, which is the same defect instruction 350 already fixed on UsageBar's fixed 48 px caption column. Measured on Control Charts: the column's own sizeHint wants 586 px at 100 %%, 707 at 125 %% and 1107 at 200 %%, against a cap that stayed 330 in all three.

### lines 187-188

```python
self._refilter = QTimer(self)
```

The filter is upstream of the maths, so the screen listens for it itself rather than leaving the canvas to redraw stale components.

### lines 195-196  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 199-201

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## PCAScreen.load_path

### lines 269-272

```python
self._jobs.cancel()
```

A second load supersedes the first. Without this, switching table twice in quick succession delivers the frames in whatever order the reads happen to finish, and the picker ends up disagreeing with the panel below it.

## PCAScreen.closeEvent

### lines 428-436

```python
"""Stop background work and unlink before going away.
```

Abandon an in-flight read rather than let it outlive the screen: Qt aborts the process if a running QThread is destroyed, and a worker that delivers into a closed widget is a use-after-free.

The panel's decomposition too. `close()` on a child widget does not reliably reach its `closeEvent`, and the panel's runner is the one holding the long job — leaving it out is exactly the leak this line exists to prevent.

## Module level

### lines 493-502

NO REGISTRY ROW. PCA is reached as a button on Image UMAP's masthead :data:`spacr.qt.screens.image_umap.FOLDED_APPS` -- which builds it through :func:`make_pca_screen` and then loads the measurements database the UMAP screen is already reading. The three are projections of one table, so the source travelling with the press is what makes them one module rather than three screens that read the same file.

The strings above are kept because they are this module's public description the fold button's name and sentence are asserted against them, and the i18n catalogs carry the translations.
