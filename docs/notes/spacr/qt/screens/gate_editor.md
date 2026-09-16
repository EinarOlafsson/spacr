# Notes from `spacr/qt/screens/gate_editor.py`

Prose lifted out of `spacr/qt/screens/gate_editor.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [GateEditorScreen.__init__](#gateeditorscreen__init__) (13 entries)
- [GateEditorScreen.choose_table](#gateeditorscreenchoose_table) (1 entry)
- [GateEditorScreen.load_paths](#gateeditorscreenload_paths) (3 entries)
- [GateEditorScreen._record_merge](#gateeditorscreen_record_merge) (1 entry)
- [GateEditorScreen.load_path](#gateeditorscreenload_path) (2 entries)
- [GateEditorScreen._on_frame_loaded](#gateeditorscreen_on_frame_loaded) (1 entry)
- [GateEditorScreen._install_graph_context_menu](#gateeditorscreen_install_graph_context_menu) (1 entry)
- [GateEditorScreen.graph_menu_items](#gateeditorscreengraph_menu_items) (1 entry)
- [GateEditorScreen.set_axis_scale](#gateeditorscreenset_axis_scale) (1 entry)
- [GateEditorScreen._show_graph_menu](#gateeditorscreen_show_graph_menu) (1 entry)
- [GateEditorScreen._on_projection_requested](#gateeditorscreen_on_projection_requested) (1 entry)
- [GateEditorScreen.apply_settings](#gateeditorscreenapply_settings) (1 entry)
- [GateEditorScreen.reduce_to_components](#gateeditorscreenreduce_to_components) (1 entry)
- [GateEditorScreen.save_graph](#gateeditorscreensave_graph) (1 entry)
- [GateEditorScreen.annotate_from_gates](#gateeditorscreenannotate_from_gates) (1 entry)
- [GateEditorScreen.database_labels](#gateeditorscreendatabase_labels) (1 entry)
- [GateEditorScreen.choose_save_filters](#gateeditorscreenchoose_save_filters) (1 entry)
- [Module level](#module-level) (1 entry)

## GateEditorScreen.__init__

### lines 247-248

```python
self._save_filters = QPushButton("Save filters…", self)
```

Beside the gate buttons, because a filter set is the same kind of decision: which rows this analysis is about.

### lines 290-292

```python
outer.addLayout(head)
```

The Settings button lives on the gates panel's tool row, left of Cluster, where the rest of the gating controls are. Two buttons opening one window is one too many.

### lines 295-298

```python
self._db_chips = QHBoxLayout()
```

The DATABASE working set, one removable chip per source. Same idiom as the table chips below it, because it is the same idea: a combination the user assembled, which has to be visible and has to be editable a member at a time (instruction 109, point 1).

### line 309  _(unsure)_

```python
self._chips = QHBoxLayout()
```

The working set, one removable chip per table.

### lines 332-334

```python
self._z_label = QLabel("Z", self)
```

Z, shown only in 3D/xD. Hidden rather than absent in 2D: the third measurement is remembered while the user works in 2D, so switching back does not lose it.

### lines 349-354

```python
body.setChildrenCollapsible(True)
```

Collapsible, deliberately. The console carries a width floor

(CONSOLE_MIN_WIDTH), so with collapsing DISABLED the splitter would be forced to hand it 320px on every screen -- and the console is meant to start out of the way. Allowing collapse gives it two honest states, hidden or readable, instead of the third one the user actually met: open but too narrow to read.

### lines 371-380

```python
side_body = QWidget(self)
```

ONE section, not two tabs. Filter and Columns were separate tabs inside a QTabWidget capped at 340px, and a panel whose content needs more than that had nowhere to put it -- which is what read as elements overlapping. They are also the same job: both narrow what the scatter shows, so hiding one behind the other meant neither could be checked while using the other.

A scroll area rather than a taller widget: the content is unbounded (a table can have hundreds of columns) and the panel is not, so something has to scroll or something has to clip.

### lines 402-414

```python
self.side_tabs = QTabWidget(self)
```

FILTER AND SEARCH AS TABS, which is the last item of instruction 31.

The search is a thing you ITERATE ON -- change a parameter, look, change it again -- and it lived behind a modal, so looking meant closing the dialog and reopening it to change anything. Beside the filter it is one click away and the plot stays visible while it is adjusted.

Filter and COLUMNS are NOT the pair that becomes tabs, and they were deliberately merged into one page earlier: they are the same job both narrow what the scatter shows -- so hiding one behind the other meant neither could be checked while using the other. Search is a different job, which is what makes it a different tab.

### lines 426-428

```python
side = self.side_tabs
```

The tab strip's QSS is registered at import -- see `_side_tabs_qss`. It used to be registered from here, against a name the theme has never exported, so it never was.

### lines 433-435

```python
side.setMinimumWidth(260)
```

The width is the SPLITTER's to decide now. A hard maximum is what made the cap unescapable: the user could not widen the column even when the content plainly needed it.

### lines 445-447

```python
body.addWidget(self.console)
```

The console goes in the splitter too, so the user decides how much room a transcript deserves. Collapsed by default: it is a thing you reach for, not a thing you look past.

### lines 454-455  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 458-460

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## GateEditorScreen.choose_table

### lines 560-562

```python
"""Ask which table in the project to gate."""
```

getOpenFileNames, plural: a screen acquired as three plates is three databases, and comparing them used to mean three sessions (instruction 109). One file behaves exactly as it did before.

## GateEditorScreen.load_paths

### lines 606-609

```python
plan = None
```

The plan first and on its own, because it is the thing that has to survive a refusal: a merge that is refused still has to be able to say WHICH plates clashed and in which databases, and that answer comes from the plan rather than from the read that refused.

### lines 615-619

```python
self._source.setText(str(exc))
```

REFUSED, AND WRITTEN DOWN. The refusal is the screen telling the user; the record is what makes the decision they then take dropping one of the two databases, usually -- answerable six months later, when the surviving frame can no longer say which plate1 it is.

### lines 632-635

```python
self._table_picker.blockSignals(True)
```

The table pickers follow the merge. Without this a multi-database session had no table working set at all: the picker was never filled, so nucleus could not be added to a merged frame, and a later reload would have read only the FIRST database.

## GateEditorScreen._record_merge

### line 684

```python
LOG.info("could not record the merge decision", exc_info=True)
```

An audit line must never be the reason a screen fails to load.

## GateEditorScreen.load_path

### lines 691-694

```python
if self._paths != [path]:
```

One database is a working set of one, not a different mode. Keeping the two in one list is what lets `_reload_working_set` stay a single code path -- and what stopped a merged session silently reloading only its first database when a table was added to it.

### lines 721-724

```python
self._tables = [chosen]
```

A table the working set does not have means a NEW database or a deliberate switch, so the set restarts. A table it already has means a reload -- a settings change, say -- and the set has to survive it, or every sampling change silently unmerges.

## GateEditorScreen._on_frame_loaded

### lines 759-760

```python
head = (f"{len(self._paths)} databases" if len(self._paths) > 1
```

WHAT THE FRAME ACTUALLY IS. Naming one file after a merge of three would be the screen saying something untrue about the numbers on it.

## GateEditorScreen._install_graph_context_menu

### lines 787-791

```python
canvas.rendered.connect(self._narrow_to_cutoffs)
```

Cutoffs are re-applied after EVERY render, because a render is what undoes them: the limits are computed from the data each time, and a render happens on every gate edit. Riding the canvas's own `rendered` signal is what makes a cutoff a state of the view rather than a gesture that survives until the next click.

## GateEditorScreen.graph_menu_items

### line 817, trailing  _(unsure)_

```python
(None, True, None, ""),
```

separator

## GateEditorScreen.set_axis_scale

### lines 908-911

```python
self.apply_settings(self._settings.replaced(
```

`log_x` / `log_y` are the retired spelling of the same choice, and `scale_for` prefers the scale only while the scale is linear. Left set, an old log flag would put the axis back on log the moment the menu chose linear.

## GateEditorScreen._show_graph_menu

### lines 1063-1064  _(unsure)_

```python
if why:
```

A greyed row with no reason is a dead end that looks like a bug, so disabled items say why.

## GateEditorScreen._on_projection_requested

### line 1160  _(unsure)_

```python
self.gates.set_projection_active(False)
```

The button claimed something that did not happen.

## GateEditorScreen.apply_settings

### line 1185  _(unsure)_

```python
if len(self._tables) > 1:
```

Through the working set, so a reload keeps every merged table.

## GateEditorScreen.reduce_to_components

### lines 1215-1216  _(unsure)_

```python
groups = getattr(self._settings, "reduction_groups", None) or {}
```

Nothing picked means every numeric column -- what xD did before there was a picker, so an existing session is unchanged.

## GateEditorScreen.save_graph

### lines 1412-1415

```python
from ..widgets.figure_queue import render_figure_to_png
```

`render_figure_to_png` writes the PNG and, in PDF mode, a vector PDF beside it. Asking it for a .png next to the chosen .pdf is how the vector file gets made, so point it at the sibling and hand back whichever one the user asked for.

## GateEditorScreen.annotate_from_gates

### lines 1535-1536

```python
self._source.setText(f"{mode} annotation — {summary} "
```

Still useful without a database: the counts are the answer, and refusing outright would hide them.

## GateEditorScreen.database_labels

### line 1631

```python
def database_labels(self) -> List[str]:
```

the database working set (instruction 109)

## GateEditorScreen.choose_save_filters

### lines 1814-1815  _(unsure)_

```python
def choose_save_filters(self) -> None:
```

the strategy filter sets, saved the way gates already are

## Module level

### lines 1916-1920

```python
_ROW = declared_app(APP_KEY)
```

The row this screen puts in the registry is declared in

`spacr.qt.app_catalog`, which is what lets the app be registered without importing this module -- the launch reads the table, not the screen. These read the same row back rather than restating it, so the name, the blurb and the nine translations have one spelling and no second copy to drift from.
