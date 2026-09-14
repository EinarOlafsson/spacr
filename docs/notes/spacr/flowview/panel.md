# Notes from `spacr/flowview/panel.py`

Prose lifted out of `spacr/flowview/panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [FlowGraphicsView.__init__](#flowgraphicsview__init__) (1 entry)
- [_PanelHeightGrip.__init__](#_panelheightgrip__init__) (1 entry)
- [FlowViewPanel.__init__](#flowviewpanel__init__) (8 entries)

## FlowGraphicsView.__init__

### lines 180-185

```python
self.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
```

TRANSPARENT, not CANVAS. This brush is the near-black rectangle reported on 2026-09-01, and it is set HERE -- clearing the scene's brush in the panel left this one painting over it, which is why the box stayed black through the first attempt. The viewport must stop filling itself as well, or Qt paints the palette colour underneath before either brush is consulted.

## _PanelHeightGrip.__init__

### lines 243-245

```python
self.setStyleSheet(
```

Drawn here rather than by the app theme: FlowView renders standalone too, and must not need the Qt palette to have a visible edge.

## FlowViewPanel.__init__

### lines 346-356

```python
panel_surface = (
```

NO RIM, and the reason is worth keeping: the border here read `#FFFFFF1A`, which in a CSS file means white at 10% alpha (#RRGGBBAA) but in a QT STYLESHEET is parsed as #AARRGGBB opaque rgb(255, 255, 26). That is the bright yellow rectangle around the inspector, reported on 2026-09-01 as "the yellow rim". The same literal in `export.py` is correct, because that one really is CSS and a browser really does read #RRGGBBAA.

It is removed rather than corrected to a faint white, which is what was asked for: the panel sits inside a section that already draws the only box this needs.

### lines 366-368

```python
"QPlainTextEdit {"
```

TRANSPARENT, ROUNDED, RIMLESS. The inspector was a black rectangle with the yellow border above; it now shows the page behind it like every other surface on the screen.

### lines 410-413

```python
self.scene.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
```

THE OTHER BLACK BOX. The scene painted CANVAS (#0E1216), which is a near-black rectangle sitting on top of whatever the screen behind it is showing. Transparent lets the page through, and the nodes carry their own fills so nothing becomes unreadable.

### lines 416-419

```python
self.view.setStyleSheet(
```

THE SCENE BRUSH IS NOT ENOUGH. A QGraphicsView paints its own widget background and its viewport's before the scene is drawn, so clearing only the brush left the same near-black rectangle on screen. All three have to give way for the page to show through.

### lines 428-431

```python
self.inspector.setMinimumHeight(self.INSPECTOR_MIN_HEIGHT)
```

TALLER TO START, AND FREE TO GROW. 118 px showed about four lines, so every stage worth inspecting needed scrolling immediately. The splitter below gives it a real share of the height rather than the sliver a minimum alone would earn it.

### lines 436-440

```python
splitter.setStretchFactor(0, 3)
```

THE INSPECTOR GETS A REAL SHARE. At 4:1 it was a sliver that collapsed to its minimum the moment the graph had anything in it; the graph still leads, but the pane underneath is now a place text can actually be read, and the splitter handle stays so either can be given the whole height.

### lines 444-449

```python
splitter.setHandleWidth(_PanelHeightGrip.HEIGHT)
```

THE HANDLE HAS TO BE VISIBLE TO BE FOUND. A QSplitter draws nothing by default on this style, so the divider between the graph and the inspector was a 6 px band of nothing -- draggable, but only by somebody who already knew it was there. It is drawn as the same hairline the console's resize handle uses, so the two affordances on this screen do not look like two.

### lines 460-463

```python
self._bottom_grip = _PanelHeightGrip(self)
```

AND ONE BELOW THE TEXT BOX. The splitter divides the panel's own height between graph and inspector; this one changes how much height the panel has at all, so the pair reads as "the graph against the text" and "the two of them against the page".
