# Notes from `spacr/qt/screens/methods_export.py`

Prose lifted out of `spacr/qt/screens/methods_export.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [MethodsExportScreen.__init__](#methodsexportscreen__init__) (2 entries)
- [MethodsExportScreen._on_digest_ready](#methodsexportscreen_on_digest_ready) (1 entry)
- [MethodsExportScreen._on_browse](#methodsexportscreen_on_browse) (1 entry)

## Module level

### lines 143-144

```python
register_widget_qss("MethodsExportSources", _methods_qss, replace=True)
```

``replace=True`` because this module owns the name: a reimport must re-register the same block rather than raise and leave the screen unstyled.

### lines 571-586

NO REGISTRY ROW. Methods & Results is not a tile: it is a button on Regression's masthead that opens the module seeded with the project and the results folder that screen is already pointed at :data:`spacr.qt.screens.regression.FOLDED_APPS` and :data:`spacr.qt.screens.regression.BUILDERS`. That seeding is what makes the fold a superset of the tile, which opened on two empty path boxes and asked the user to type what the host already knew.

Everything the row used to fan out has a home that outlives it: the button's name, sentence and alpha maturity colour in :data:`spacr.qt.screens.map_barcodes.FOLD_FALLBACK`, the API link in ``settings_model._APP_API_MODULE``, the headless answer in :data:`spacr.cli.INTERACTIVE_ONLY`, and the nine translated names in the shipped i18n catalogs. The strings above stay because they are this module's own description, and because those homes are asserted against them.

## MethodsExportScreen.__init__

### lines 211-212  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 215-217

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## MethodsExportScreen._on_digest_ready

### lines 351-352

```python
self._set_provenance("The digest could not be built.",
```

None or empty -- both are as useless as each other here, and saying nothing would leave the previous digest up.

## MethodsExportScreen._on_browse

### lines 472-475

```python
def _on_browse(self, key: str, is_folder: bool) -> None:
```

MODAL IS A REASON NOT TO OPEN ONE IN A TEST, not a reason to leave these untested: everything that matters happens after the dialog returns. Driven in tests/qt/test_the_modal_slots_do_what_the_dialog_returns.py.
