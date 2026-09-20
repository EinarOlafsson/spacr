# Notes from `spacr/qt/settings_search.py`

Prose lifted out of `spacr/qt/settings_search.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [SettingsSearchBar.__init__](#settingssearchbar__init__) (6 entries)
- [SettingsSearchBar.apply](#settingssearchbarapply) (1 entry)
- [SettingsSearchBar._build_index](#settingssearchbar_build_index) (1 entry)
- [SettingsSearchBar._apply_section_state](#settingssearchbar_apply_section_state) (2 entries)
- [SettingsSearchBar._compose_count](#settingssearchbar_compose_count) (1 entry)
- [_form_of](#_form_of) (1 entry)
- [_set_row_visible](#_set_row_visible) (1 entry)
- [install](#install) (4 entries)
- [install_window_hooks](#install_window_hooks) (1 entry)
- [Module level](#module-level) (1 entry)

## SettingsSearchBar.__init__

### lines 193-196

```python
self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
```

Fixed height, explicitly. The strip is two rows tall and the scroll area under it wants everything else; without this the two share the pane by their stretch factors and the search box ends up 800 pixels high on first layout.

### lines 201-202

```python
self._index: Dict[str, Tuple[QWidget, QWidget]] = {}
```

key -> (section, field widget). Built once, from the rendered form, so the filter never has to guess which section a key ended up in.

### lines 206-207

```python
self._restore_expanded: Optional[Dict[int, bool]] = None
```

Which sections the user had open before a filter took over, so clearing the box puts the form back rather than leaving it splayed.

### lines 234-236

```python
self._modified_label = QLabel(self)
```

A `Toggle`, not a QCheckBox: every boolean control in the shell is a switch, and `tests/qt/test_widgets.py` bans the plain checkbox outright so one panel cannot quietly reintroduce it.

### lines 262-265

```python
self._controls_row = row
```

Where other modules hang their own controls — see

`spacr.qt.recipes.install`. Kept as the row itself rather than a nested container so a trailing button lines up with the ones above it rather than being visibly bolted on.

### lines 268-269  _(unsure)_

```python
self._count = QLabel(self)
```

The count line sits under the controls, so the whole strip is one widget the host inserts in one place.

## SettingsSearchBar.apply

### lines 350-366

```python
hidden_by_run = getattr(model, "keys_hidden_by_the_run", None)
```

THE OBJECT RULE OUTRANKS THE INDEX, and is subtracted BEFORE the level and the query rather than undone afterwards. This strip indexed every row on the panel, including rows belonging to objects the run does not have, so "All settings" filled Mask with nucleus and pathogen settings while every channel was None -- the maintainer's report, and what `test_the_object_rows_stay_off_the_form` guards.

SUBTRACTED, NOT RE-HIDDEN. Showing them and hiding them again in the same pass leaves `visible_keys()` disagreeing with the form for as long as it takes the second write to land, and it is the strip's own index that answers that question. Removing them from `wanted` means the row is never shown, so there is one answer throughout.

Asked of the model, because the strip has no idea which objects a run has and teaching it would put the same rule in two places.

## SettingsSearchBar._build_index

### lines 479-485

```python
for child in field.findChildren(QWidget):
```

THE FIELD IN THE ROW IS NOT ALWAYS THE FIELD. A setting that takes a Cellpose checkpoint sits in a little holder beside its "Model zoo…" button, so the form's row is the HOLDER and matching on it alone left `cell_model_name` out of the index entirely -- typing "model" on Mask found nothing and the row could not be reached from the search at all.

## SettingsSearchBar._apply_section_state

### lines 512-514

```python
continue
```

Not narrowing means nothing is filtered, and a section with no rows was already invisible for its own reasons (maturity). Leave that judgement alone.

### lines 527-528  _(unsure)_

```python
refresh = getattr(self._screen, "refresh_maturity_visibility", None)
```

Hand maturity visibility back to the screen, which is the only thing that knows why a section was hidden in the first place.

## SettingsSearchBar._compose_count

### lines 539-542

```python
"""Build the line under the form saying how much of it is showing.
```

COMPOSED FROM TRANSLATED PARTS. The catalog is keyed on the sentence with its numbers as placeholders; a line built out of f-strings first and looked up after matches nothing, and this line sits under every settings panel in the program.

## _form_of

### lines 572-579

```python
def _form_of(section: QWidget) -> Optional[QFormLayout]:
```

Row visibility

`Section` builds its rows with `QFormLayout.addRow`, and the label side is a wrapper widget it builds itself and does not hand back. `setRowVisible` keyed on the FIELD widget therefore reaches both halves, which nothing outside the section can do by hand.

## _set_row_visible

### lines 612-614

```python
field.setVisible(visible)
```

Qt < 6.4 has no setRowVisible. Hiding the field alone leaves an orphaned label, but a stranded label is a far smaller problem than a settings panel that will not draw.

## install

### lines 669-678

```python
container = QWidget()
```

NO PARENT HERE. `insertWidget` below parents this container to the splitter, and handing it the same parent at construction parents it twice -- Shiboken then releases the wrapper twice when the screen's children are deleted, and the process dies inside QObjectPrivate::deleteChildren.

It is a SEGFAULT, so it does not arrive as a failing assertion: the test body passes and the process dies afterwards, which xdist reports as a failed test with no message and which takes the rest of that shard with it.

### lines 685-686  _(unsure)_

```python
column.addWidget(scroll, 1)
```

addWidget re-parents the scroll area out of the splitter, which is what frees the slot the container then takes.

### lines 689-694

```python
container.show()
```

`setParent` hides a widget, and a hidden widget is one a layout skips. Without these two the QVBoxLayout saw no visible children, left the scroll area at the geometry it had as a splitter pane, and centred the strip on top of the settings form -- both of them drawing over each other at full pane height. Nothing that reads form-row visibility notices, which is why it has a geometry test.

### lines 704-708

```python
try:
```

The strip captions itself in the user's language, but it is also an extension point — `spacr.qt.recipes` hangs a button on it — and it is built from `stack.currentChanged`, long after the window has run its one language pass over this screen. A pass over the finished strip is idempotent and means nothing added here can be left in English.

## install_window_hooks

### line 773  _(unsure)_

```python
QTimer.singleShot(0, watcher.install_current)
```

The first module may already be on screen by the time hooks run.

## Module level

### lines 843-846

```python
try:
```

AT IMPORT TIME, so the failure is not a missing background it is the module not importing, which takes down whatever imports it. Driven in tests/qt/test_a_theme_that_refuses_does_not_stop_an_import.py.

## SettingsSearchBar._grid_section

### added 2026-09-19 (431)

```python
shown_per_section[id(grid_section)] = len(in_the_grid)
```

With the per-object table on, the flat rows it answers for -- every object's channel among them -- are hidden, and the table's section has no form rows for this strip to count. Under Essentials every section with nothing counted is hidden, so the table went too: measured on a built Mask screen with the preference on, no channel could be set anywhere on the form. The section is now counted by the keys the table answers for, filtered exactly as rows are (query, Modified, level) and less the keys whose object the run lacks.

The table can be mounted or taken down by Preferences after this strip was built, so the section is looked up on every `apply()` and the strip's list of sections is kept in step: a section taken down is dropped before it can be touched, which matters because it is deleted with `deleteLater()` and would raise on the next `setVisible`.

Seen and not changed, measured 2026-09-19 on a built Mask screen under All settings: a section with no rows of its own and only sub-headings counts zero, so any narrowing hides it with everything under it. Searching "remove border objects" reports one match, `cell_remove_border_objects`, whose row is visible on its form -- while "Object Filtration (all objects)" and the "Advanced settings" umbrella above it are both hidden, so nothing is on screen. A count rolled up from each sub-heading to the headings above it would fix it; it is recorded in 431 rather than fixed there, because it is the search's own defect and not the channels'.

## SettingsSearchBar.apply

### added 2026-09-19 (431, from review)

```python
if visible and (reopen or kept_before is None
                or id(section) not in kept_before):
```

431 made the screen re-apply this filter after every pass of the object rule, and under Essentials every call counts as narrowing, which opens every kept section. Measured in review on a fresh Mask screen: shut every section, change `metadata_type` (a dependency source, so it runs the object rule), and all four sections opened again. On the branch base they stayed shut. The screen's two re-applications, after the object rule (`AppScreen._refilter_the_settings_search`) and after laying out rows that arrived late (`AppScreen._the_rows_moved`), now pass `reopen=False`. A section the user shut then stays shut, and a section that call brings back onto the form, such as Pathogen Segmentation after a pathogen channel is committed, is still opened, because it is not in the set the previous call kept. A change to the query, the Modified switch or the level still opens everything it keeps, as before. The previous call's kept sections are recorded instead of read from `isHidden()`, because the object rule shows and hides headings itself before this runs. Held by `test_a_section_the_user_shut_stays_shut`.

## SettingsSearchBar._show_all_without_remembering

### added 2026-09-19 (422, from review)

```python
blocked = self._disclosure.blockSignals(True)
```

`reveal` raises the disclosure level when the row it was asked for is one Essentials hides, and it used to do that with `set_level(ALL)`. That sets the toggle, which emits `toggled`, which runs `_on_disclosure_toggled`, which calls `remember_disclosure` -- a persistent `QSettings` write. So looking a setting up from the Help search moved the module out of Essentials permanently, and since most settings are not essentials (Mask renders 190 rows against a handful), that was the common path rather than the rare one. The commit that first narrowed this said it had fixed it; it had only made it conditional.

Showing the row needs the level raised on the FORM. It does not need the raise written to the store, so the toggle's signal is blocked while the button, the level and the caption are moved, and the one caller of `remember_disclosure` never runs. Clicking the switch still remembers, because that is the user choosing. One direction only: there is no reason to lower the level behind the user's back at all, and a `level` parameter would have added a branch nothing takes -- which `spacr/qt/settings_search.py` cannot afford, since its coverage baseline records zero uncovered branches.

Held by `test_arriving_here_does_not_rewrite_the_essentials_choice`, which now reads the store back with `disclosure_for` rather than only asking the strip what level it is on.
