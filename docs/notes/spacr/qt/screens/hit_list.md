# Notes from `spacr/qt/screens/hit_list.py`

Prose lifted out of `spacr/qt/screens/hit_list.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [HitListScreen.__init__](#hitlistscreen__init__) (2 entries)
- [HitListScreen._build_ui](#hitlistscreen_build_ui) (3 entries)
- [HitListScreen._on_hits_ready](#hitlistscreen_on_hits_ready) (1 entry)
- [HitListScreen.current_filters](#hitlistscreencurrent_filters) (1 entry)
- [HitListScreen._on_export_csv](#hitlistscreen_on_export_csv) (1 entry)

## Module level

### lines 161-163

```python
register_widget_qss("HitListFilters", _hit_list_qss, replace=True)
```

``replace=True`` because this module owns the name: a reimport (a test that reloads it, a plugin that pulls it in twice) must re-register the same block rather than raise on the duplicate and leave the screen unstyled.

### lines 709-724

NO REGISTRY ROW. The hit list is not a tile: it arrives as the **Hits tab** on Regression's results panel, loaded with the run whose coefficients are on screen, and as a button on that masthead which raises the tab -- :data:`spacr.qt.screens.regression.FOLDED_APPS` and :class:`spacr.qt.screens.regression.HitsOpener`. A tile would have been a second front door onto the same table, opening it empty and asking the user to find the results folder the host already knows.

Everything the row used to fan out has a home that outlives it: the button's name, sentence and alpha maturity colour in :data:`spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, the API link in ``settings_model._APP_API_MODULE``, the headless answer in :data:`spacr.cli.INTERACTIVE_ONLY`, and the nine translated names in the shipped i18n catalogs. The strings above stay because they are this module's own description, and because those homes are asserted against them.

## HitListScreen.__init__

### lines 220-221  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 224-226

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## HitListScreen._build_ui

### lines 280-284

```python
self._q_spin.setValue(1.0)
```

Opens at 1.0 — the whole ranked list — rather than at the FDR. The ranking is the deliverable; the cut is the user's decision, and a screen that silently opens pre-filtered hides both the controls (whose position IS the QC) and the near-misses a reader wants to see. The summary strip still reports how many clear the FDR.

### lines 379-380

```python
install_sorting(self._table)
```

The list arrives ranked, and a third click on a header brings that ranking back -- so sorting costs the default order nothing.

### lines 385-386

```python
mark_surface(self._table)
```

The hit list IS the page below the filter bar. The bar is the only panel on this screen and the tree does not sit on it.

## HitListScreen._on_hits_ready

### lines 438-441

```python
self._set_summary("The hit list could not be built.", problem=True)
```

THE WORKER FAILED. This runs on the GUI thread from a finished signal, so an AttributeError here surfaces as an unhandled exception in the Qt event loop and leaves the screen showing the last list it had.

## HitListScreen.current_filters

### lines 462-464

```python
arguments["min_selection"] = float(self._q_spin.value())
```

The same dial means the opposite thing for a backend that ranks by selection frequency: there is no q-value to be under, and a threshold of 0.6 is a floor on how often the guide was chosen.

## HitListScreen._on_export_csv

### lines 568-571

```python
def _on_export_csv(self) -> None:
```

MODAL IS A REASON NOT TO OPEN ONE IN A TEST, not a reason to leave these untested: everything that matters happens after the dialog returns. Driven by stubbing the Qt static, in tests/qt/test_the_modal_slots_do_what_the_dialog_returns.py.
