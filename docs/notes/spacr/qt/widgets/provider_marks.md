# Notes from `spacr/qt/widgets/provider_marks.py`

Prose lifted out of `spacr/qt/widgets/provider_marks.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (2 entries)
- [gemini_path](#gemini_path) (1 entry)
- [ProviderMark](#providermark) (1 entry)
- [ProviderMark.__init__](#providermark__init__) (1 entry)
- [ProviderMark.mousePressEvent](#providermarkmousepressevent) (1 entry)
- [ProviderMark.paintEvent](#providermarkpaintevent) (1 entry)
- [ProviderMark._ready_ink](#providermark_ready_ink) (1 entry)
- [ProviderMark._colours](#providermark_colours) (1 entry)
- [ProviderMark._paint](#providermark_paint) (2 entries)

## Module level

### lines 33-40

```python
from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, Signal
```

QEvent AT MODULE SCOPE, NOT INSIDE THE CALLBACK. A function-local import in an event handler is not lazy loading: this module is a QWidget module and cannot load without QtCore, so the import bought nothing but a sys.modules lookup on every event -- and it put an EXCEPTION SITE on a path with no way to report one. The same shape in `ModuleHintBar.event` produced 419 errors in one sweep when a test stubbed PySide6.QtCore out of sys.modules and teardown then delivered a paint event.

### lines 55-60

```python
"github": "#181717",
```

GitHub's mark is monochrome, so "in colour" means in the theme's own ink rather than in a brand hue -- "if the colour of the icon is black and white go from grey to black and white". Signed out it takes the muted ink like any other unavailable mark. The value here is the LIGHT-theme ink; on a dark theme `_ready_ink` inverts it, because GitHub's own near-black on a near-black card is an invisible mark.

## gemini_path

### lines 126-128

```python
between = 2.0 * math.pi * (index + 0.5) / 4.0 - math.pi / 2.0
```

The control point sits near the centre, which is what pulls each side inward and makes the four points read as a spark rather than as a plain diamond.

## ProviderMark

### lines 345-347

```python
chosen = Signal(str)
```

Gating the choice on availability was reported 2026-08-22 -- "for the ai assistant i can only click claude" -- and the report is right: a preference is not a launch.

## ProviderMark.__init__

### line 396

```python
self.setCursor(Qt.PointingHandCursor)
```

EVERY MARK IS CLICKABLE, so every one gets the hand.

## ProviderMark.mousePressEvent

### lines 415-417

```python
"""Open the provider's page.
```

AVAILABILITY DOES NOT GATE THE CHOICE. It is drawn -- brand colour when the CLI is here, muted ink when it is not -- and said in the tooltip, which is information rather than obstruction.

## ProviderMark.paintEvent

### lines 458-459

```python
pass
```

Decoration is never load-bearing: an unpainted mark is still a control that answers the question when it is clicked.

## ProviderMark._ready_ink

### lines 472-475

```python
return QColor(palette.get("fg", BRAND["github"]))
```

THE THEME'S OWN INK, which is the end of monochrome that can be read against the card it sits on: white on a dark theme, near black on a light one. GitHub's #181717 IS the light-theme value; painted on a dark card it is a signed-in state nobody can see.

## ProviderMark._colours

### lines 484-489

```python
ink = QColor(palette.get("fg_muted", palette["fg"]))
```

THE BRAND FILL IS WHAT "READY" LOOKS LIKE, so a provider that is not installed does not get one: "GPT and Gemini should only get their color fill when they are installed". The mark is muted ink -- legible, at alpha 190 rather than the 110 that made it a ghost -- and HOVER gives the brand background, so the colour still tells you which provider you are pointing at.

## ProviderMark._paint

### lines 539-540

```python
painter.setPen(QColor(palette["fg"]))
```

THE NAME IS ALWAYS LEGIBLE. It used to fade with the mark, so a provider that was not set up had no readable name either.

### lines 546-548

```python
note = self.STATUS_TEXT.get(self.status, "")
```

AND WHAT TO DO ABOUT IT, under the name, in the brand colour. A greyed control that does not say why is the thing this replaces.
