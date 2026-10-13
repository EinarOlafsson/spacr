# Notes from `spacr/qt/widgets/hover_tooltip.py`

Prose lifted out of `spacr/qt/widgets/hover_tooltip.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_anchor_setting_key](#_anchor_setting_key) (1 entry)
- [_AnimationView.load](#_animationviewload) (1 entry)
- [_AnimationView._schedule](#_animationview_schedule) (1 entry)
- [HoverTooltip.__init__](#hovertooltip__init__) (5 entries)
- [HoverTooltip._apply_theme](#hovertooltip_apply_theme) (4 entries)
- [HoverTooltip.show_for](#hovertooltipshow_for) (1 entry)
- [HoverTooltip.toggle_animation](#hovertooltiptoggle_animation) (1 entry)
- [HoverTooltip._set_animation](#hovertooltip_set_animation) (5 entries)
- [HoverTooltip._resize_text_column](#hovertooltip_resize_text_column) (3 entries)
- [HoverTooltip._position_under](#hovertooltip_position_under) (1 entry)
- [HoverTooltip._claim_anchor](#hovertooltip_claim_anchor) (2 entries)
- [HoverTooltip._pointer_is_on_me](#hovertooltip_pointer_is_on_me) (1 entry)
- [HoverTooltip._maybe_hide](#hovertooltip_maybe_hide) (2 entries)

## _anchor_setting_key

### line 192  _(unsure)_

```python
return ""
```

The anchor's C++ half is gone; there is nothing to read.

## _AnimationView.load

### line 332

```python
self.play()
```

Same setting hovered again: keep playing rather than restart.

## _AnimationView._schedule

### lines 393-394  _(unsure)_

```python
"""Arm the timer for the current frame's own delay.
```

One delay per frame, guaranteed by `read_frames`; a still image has nothing to schedule.

## HoverTooltip.__init__

### lines 466-475

```python
self._label.setAlignment(Qt.AlignJustify | Qt.AlignTop)
```

Belt and braces with the layout's AlignTop below. If the label is ever stretched to the height of the animation, QLabel's default AlignVCenter would float the prose down to the middle of the square while the widget's top edge stayed put — top-aligned by geometry and centred to the eye, which is not what was asked for. JUSTIFIED. Asked for 2026-08-28. A tooltip is a paragraph of prose in a narrow fixed-width popup, which is where a ragged right edge is most visible: every line ends somewhere different and the block reads as an offcut rather than as a paragraph. Qt justifies rich text, and the popup is rich text already.

### lines 484-500

```python
self._api_link = _LinkWord(API_MARK, "HoverTooltipApiLink",
```

The drawn text and the announced text are the same again, which they were not while the marks were drawn. The accessible names are set explicitly all the same: they were correct throughout and are what a screen reader has always said, so leaving them to be inferred from the label would be trading a guarantee for a coincidence.

NO `setToolTip` ON EITHER WORD, reported 2026-09-03: "for some reason when i hover API links they get a tooltip themeselves upon hover. remove this." These two words live INSIDE this panel, and this panel is itself the tooltip -- so hovering one raised a second, native tooltip on top of the help the reader was already reading. The strings said nothing the words do not say.

The `setAccessibleDescription` calls stay. A screen reader reads those, not the tooltip, so removing the popup costs a sighted reader nothing and costs a screen-reader user nothing either.

### lines 519-520  _(unsure)_

```python
column.addStretch(1)
```

Holds the prose and the two words at the TOP of a column that is stretched to the height of the square beside it.

### lines 527-532

```python
self._setting_key = ""
```

The reveal, in two fields: which setting the reader pressed

Animation** on, and what they pressed it to. It applies to that setting and to nothing else, so hovering anything else falls back to the preference. A session-wide flag here is precisely the behaviour that was rejected — one press must not put every later hover back on the decode path.

### lines 542-543

```python
lay.addWidget(self._text_column, 0, Qt.AlignTop)
```

AlignTop on both, so the text starts level with the first frame instead of floating to the middle of a 220-pixel square.

## HoverTooltip._apply_theme

### lines 555-557

```python
palette = active_palette()
```

This widget is a separate top-level window, so app-level QSS does not reliably reach it. It is also a singleton that survives a Preferences theme switch, hence this must be refreshed on show.

### lines 565-579

```python
f"QWidget#HoverTooltipTextColumn,"
```

The two layout containers paint NOTHING. Both are plain

`QWidget`s, so without this they inherit the application sheet's blanket `QWidget { background-color: bg }` — and `bg` is the WINDOW colour, #000000 in the dark theme, not a surface. The result was a black slab covering all but a 6-pixel margin of the popup's own rounded grey: 20669 black pixels inside a #161719 frame. `theme.clear_container_surfaces` exists for exactly this and could not help — it only tags ANONYMOUS widgets as scaffolding, and both of these are named.

Transparent rather than re-filled with the frame's colour on purpose: one surface, one alpha. Painting the same grey twice is what left the System panel's meters unable to thin out when the page-opacity slider moved, because the two translucent layers composited into something darker than either.

### lines 589-591

```python
f"QLabel#SettingTooltipAnimation {{"
```

Transparent, not black: the rounded corners are cut into the frames themselves, and a background painted by the sheet would square them off again.

### lines 595-602

```python
f"QLabel#HoverTooltipApiLink {{"
```

Two marks, two colours, no underline anywhere. Teal DOT for the API, purple SQUARE for the animation -- the colours the maintainer named, and the shapes carry the same distinction for a reader who cannot separate the colours.

A LARGER FONT THAN THE PROSE: these glyphs are drawn at the label's font size, and at the popup's small size a circle reduces to a few pixels. They are targets as well as marks.

## HoverTooltip.show_for

### lines 646-648

```python
self._setting_key = _anchor_setting_key(anchor)
```

Which setting this tooltip is for, which is what the reveal is scoped to. Read before `_set_animation`, because that is what asks `animations_shown()` whether this particular setting was pressed.

## HoverTooltip.toggle_animation

### lines 760-762

```python
wanted = not self.animations_shown()
```

Read BEFORE the key is claimed: afterwards `animations_shown` would answer with the state being written here rather than the one being inverted.

## HoverTooltip._set_animation

### lines 807-808  _(unsure)_

```python
showing = self._animation_view.load(animation)
```

The only line in this class that reads a GIF, and it is reached only from a press or from the preference being on.

### lines 812-815

```python
self._animation_view.stop()
```

Folded away while still on the same setting: pause, keep the finished pixmaps (~3.5 MB for one animation), so pressing again costs nothing. Bounded at one animation and never filled before a press -- moving to any other setting hits the branch below.

### lines 819-821

```python
self._animation_view.clear_animation()
```

Nothing to show, or not asked for: decode nothing and drop the previous setting's frames. This is the default path, and it is why a plain hover costs no decode and holds no pixmaps.

### lines 826-829

```python
self._animation_link.setVisible(
```

Offered but hidden -> the word is the invitation. Showing -> the word folds it away again. Revealed but undecodable -> hide it: a word that visibly does nothing is worse than no word. No animation for this setting at all -> nothing to say.

### lines 832-834

```python
self._links.setVisible(
```

Hidden, not merely empty: a zero-height row still costs the layout its spacing, which is exactly the slack a text-only popup was asked to lose.

## HoverTooltip._resize_text_column

### lines 884-885  _(unsure)_

```python
self._label.setMaximumWidth(self.TEXT_WIDTH)
```

Nothing pinned: the layout takes the popup down to what the prose and the two words actually occupy.

### lines 890-892

```python
prose = max(0, self._label.heightForWidth(width))
```

QLabel's own size hint for wrapped rich text is derived from its preferred, not its actual, width; without this the popup opens tall enough for a couple of lines and clips the rest.

### lines 898-900

```python
self._text_column.setFixedHeight(max(self.ANIMATION_SIZE, needed))
```

Exactly the height of the square. `max` only matters for prose too long to fit even at the widest step, where the alternative would be truncating the user's help text.

## HoverTooltip._position_under

### line 921

```python
try:
```

Not enough space below — flip above

## HoverTooltip._claim_anchor

### line 947  _(unsure)_

```python
return
```

The anchor's C++ half is gone; there is no tooltip to suppress.

### lines 949-950

```python
QToolTip.hideText()
```

A native tooltip already on screen — from the widget the pointer crossed on its way here — would otherwise sit over this one.

## HoverTooltip._pointer_is_on_me

### lines 973-983

```python
try:
```

THE GEOMETRY DECIDES, and `underMouse()` is not consulted at all. It was, as a first answer with the geometry as a fallback, and that inherited exactly the unreliability it was added to work around: measured here, `underMouse()` reports True with the cursor demonstrably outside the popup's rectangle, and elsewhere it reports False with the pointer on it. A source that is wrong in both directions cannot improve an answer by being consulted first.

`frameGeometry`, not `geometry`: the popup is frameless so they agree, and if a platform ever adds a frame the pointer is still on the popup when it is over that frame.

## HoverTooltip._maybe_hide

### lines 1002-1009

```python
anchor = self._anchor
```

`self._anchor is not None` is not enough. The tooltip is a process-wide singleton holding a plain reference to a widget it does not own, and the hide is deferred by a timer -- so hovering a settings label and switching module inside the delay destroys the anchor's C++ object while this timer is still pending. The Python wrapper survives, so the None check passes, and underMouse() then raises RuntimeError('Internal C++ object already deleted') inside the Qt event loop, where there is nobody to catch it.

### line 1019  _(unsure)_

```python
self._anchor = None
```

The anchored widget is gone; nothing can be hovering it.
