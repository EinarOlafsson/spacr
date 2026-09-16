# Notes from `spacr/resources/home/versions/_generators/common.py`

Prose lifted out of `spacr/resources/home/versions/_generators/common.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (14 entries)
- [bootstrap](#bootstrap) (2 entries)
- [_registry](#_registry) (1 entry)
- [Ctx.apply_theme](#ctxapply_theme) (1 entry)
- [cats_current](#cats_current) (1 entry)
- [_with_late_registrations](#_with_late_registrations) (2 entries)

## Module level

### lines 80-82

```python
_prefer_checkout_package()
```

The tables below read the registry during module import, before :func:`bootstrap` is called. Select the checkout now so those tables and the later renderer cannot disagree about which spaCR tree they represent.

### lines 326-330

```python
MOCK = {
```

Mock content for the elements that do not exist yet

Fixed literals, never live state — see the module docstring. Anything drawn from these is *proposed* UI, not something spaCR reports today.

### lines 385-390

```python
USE_COUNTS.setdefault(_key, UNUSED_APP_COUNT)
```

Variant 14 reads ``USE_COUNTS[k]`` for the badge on every tile, so a key missing here is not "sorts to the bottom", it is a KeyError that takes all thirty variants down. That is a hand-edit a module which registers itself from its own file cannot make, so the table fills itself and the literals above stay a statement about the apps somebody actually had an opinion on.

### lines 471-473

```python
("Prepare", ["power", "experiment_design", "convert", "align",
```

Power / Design is the only app in the registry that runs BEFORE the images exist. "Prepare" is the closest of these three to that, and it is where a screener would look for it.

### lines 494-495

```python
("Acquire", ["power", "experiment_design", "convert", "align", "foreign",
```

Design precedes acquisition; project conversion, dispatch and storage management all prepare inputs rather than interpret results.

### lines 499-500  _(unsure)_

```python
("Segment", ["mask", "make_masks", "layer_viewer"]),
```

Mask creation, manual correction and registered layer inspection are one segmentation stage; folded model tools are reached through Mask.

### line 502

```python
("Measure", ["measure", "annotate", "lineage", "analyze_plaques",
```

These applications quantify, label or summarize measured objects.

### lines 506-507

```python
("Analyse", ["classify_merged", "map_barcodes", "regression", "umap",
```

Classification, barcode mapping, regression and exploratory model interrogation produce analytical results.

### lines 511-512

```python
("Report", ["plate_view", "train_compare", "run_history", "run_compare",
```

Provenance, QC, comparisons and export determine whether a result can be reported and preserve the evidence used to reach that decision.

### lines 520-521

```python
("Segment",          ["mask", "make_masks"]),
```

The bands stay deliberately narrow. Folded capabilities remain on their host screens and therefore do not receive standalone entries.

### lines 526-528

```python
("Screens & reports", ["map_barcodes", "regression",
```

The Prediction Profiler goes here rather than under "Classify": what it sweeps is a screen's regression, which is this band's subject, while Classify contains the classifier and training review.

### lines 546-548

```python
("I have images. Where are my objects?",
```

Power / Design answers the question BEFORE the first one here — "do I have enough images?" — and the honest place for it is the band about getting images, since that is the decision it feeds.

### lines 556-558

```python
("I have a screen. Which genes matter?",
```

Hit List answers this band's question in the most direct way there is — it IS the list of genes that matter — and the Prediction Profiler is how you interrogate the model that produced it.

### lines 564-567

```python
("Should I believe any of this?",
```

Pipeline Graph belongs here for the literal reason: it marks the outputs that no longer follow from their inputs, which is the question in the heading. Methods & Results is the other half — what you write down once you have decided you do believe it.

## bootstrap

### lines 104-109

```python
global _WE_OWN_THE_APP
```

QSettings.setDefaultFormat / setPath are PROCESS-GLOBAL. Redirecting them when we did not create the QApplication reaches into a host that is already running: under pytest-qt it repoints every other test's preferences at a temp directory mid-session, which is how this file took the whole tests/qt suite down with a segfault. Only isolate when this really is our own standalone process.

### lines 113-117

```python
sandbox = tempfile.mkdtemp(prefix="spacr-home-variants-")
```

NativeFormat as well as Ini. `preferences._settings()` builds `QSettings("spacr", "qt")`, which is a NativeFormat object and ignores setDefaultFormat/setPath(IniFormat, ...) — redirecting only Ini left every render reading the operator's own saved font scale and theme, so "deterministic" renders differed per machine.

## _registry

### lines 158-160

```python
import spacr.qt
```

``spacr.qt.run`` performs these registrations before constructing MainWindow. Home and the sidebar therefore show this launched registry, not the shorter import-time table from ``app.py`` alone.

## Ctx.apply_theme

### line 275

```python
if target is not None:
```

Guest inside someone else's QApplication: never touch it.

## cats_current

### lines 395-397  _(unsure)_

```python
def cats_current() -> "List[Tuple[str, List[str]]]":
```

Categorisations — every one covers every real app key exactly once

## _with_late_registrations

### lines 441-444

```python
result = [(title, list(keys)) for title, keys in cats]
```

New registrations still enter the declared fallback so the review surface remains buildable. Retired rows are deliberately different: retaining one would present a standalone Home tile that no longer exists, so the explicit failure above makes that drift visible.

### lines 454-462

```python
raise AssertionError(
```

THE SAFETY NET HAS TO SAY WHEN IT IS NOT THERE. The fallback is matched by exact title, so a rename or a typo in the caller's table turns this whole mechanism off. Silently returning the unrepaired table does not avoid the failure, it MOVES it: the uncategorised keys then hit `check_coverage`, which raises "keys not categorised: [...]" at module import of the variant generators and takes all thirty Home renders down -- blaming the registry for a mistake in the band title. Raising here names the actual cause, at the line that can see it.
