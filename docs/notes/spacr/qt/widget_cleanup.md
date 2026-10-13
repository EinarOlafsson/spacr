# Notes from `spacr/qt/widget_cleanup.py`

Prose lifted out of `spacr/qt/widget_cleanup.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [retire_pyqtgraph_menus](#retire_pyqtgraph_menus) (3 entries)
- [retire_pyqtgraph_menus.retire](#retire_pyqtgraph_menusretire) (4 entries)

## retire_pyqtgraph_menus

### lines 37-39

```python
pyqtgraph = sys.modules.get("pyqtgraph")
```

Cleanup must not be the event that loads an optional plotting stack. A real pyqtgraph view can only already exist after its package has been imported; unrelated QGraphicsViews need no work from this helper.

### lines 46-51

```python
menus: dict[int, Any] = {}
```

Keep the wrappers alive until every root has been detached from pyqtgraph and queued.  ``id(menu)`` by itself is not an ownership token: Shiboken may release a wrapper as soon as PlotItem/ViewBox drops its Python reference even though the C++ object is waiting for a deferred delete.  On a large Qt session the next wrapper can then reuse that id and be mistaken for a root already visited.

### line 53  _(unsure)_

```python
owned_widgets: list[Any] = []
```

The same applies to C++-owned submenu/control wrappers discovered below.

## retire_pyqtgraph_menus.retire

### lines 62-67

```python
if (isinstance(owner, QWidget)
```

Repair the ownership hole before relying on the event loop. The closing plot now provides a final, synchronous destruction boundary if a platform defers (or coalesces) deleteLater events differently. Reparenting may clear the menu's window flag, which is harmless after close has begun. This is deliberately the supplied owner, never a QApplication-wide retirement bin.

### lines 73-74  _(unsure)_

```python
pass
```

Deletion is still queued below if a binding rejects the ownership transfer during its own close notification.

### lines 76-79

```python
for child in reversed(menu.findChildren(QWidget)):
```

ViewBoxMenu embeds spin boxes and combo-box popup views that Qt promotes to top-level windows. Delete the whole QObject-owned widget tree before deleting the menu root, not just submenus, or those controls are reparented to ``None`` and survive it.

### lines 84-87

```python
for controls in getattr(menu, "ctrl", ()):
```

pyqtgraph's generated ``Ui_Form`` control holders are ordinary Python objects, not QObject children of ViewBoxMenu. Closing the menu reparents a few of their editors/popups to ``None``; retire those widgets while the menu still owns the holders.
