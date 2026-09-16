# Notes from `spacr/qt/widgets/availability_panel.py`

Prose lifted out of `spacr/qt/widgets/availability_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [AvailabilityPanel.__init__](#availabilitypanel__init__) (2 entries)
- [AvailabilityPanel._make_link](#availabilitypanel_make_link) (1 entry)
- [AvailabilityPanel._render](#availabilitypanel_render) (1 entry)
- [AvailabilityPanel._maybe_hide](#availabilitypanel_maybe_hide) (1 entry)
- [AvailabilityPanel._on_link](#availabilitypanel_on_link) (1 entry)
- [run_install_offer](#run_install_offer) (3 entries)

## AvailabilityPanel.__init__

### lines 157-158

```python
"""Build the availability popup.
```

Qt.Tool rather than Qt.ToolTip: a ToolTip window can never take focus, and the keyboard route in `open_for` needs it to.

### lines 184-186

```python
self._links = QWidget(self)
```

THE TWO LINK WORDS. Separate labels rather than one, so "INSTALL to the right of the API link" is a fact about geometry a test can measure rather than a claim about a string.

## AvailabilityPanel._make_link

### lines 253-255

```python
label.setTextInteractionFlags(Qt.TextBrowserInteraction)
```

The measured route: `linkActivated` carries the href, and

TextBrowserInteraction includes LinksAccessibleByKeyboard, which is what puts the word in the tab order once the panel has focus.

## AvailabilityPanel._render

### lines 380-382

```python
parts = [reason]
```

The refusal first, then what would fix it. Two sentences that say the same thing are collapsed to one -- `backend_status` and `backend_install_offer` genuinely agree on some entries.

## AvailabilityPanel._maybe_hide

### lines 503-504

```python
self._hide_timer.start(int(self.HIDE_DELAY_MS))
```

Still travelling. Re-arm rather than close, or the Install link can never be reached.

## AvailabilityPanel._on_link

### line 599, trailing  _(unsure)_

```python
self._pinned = True
```

the dialog steals the pointer; stay put

## run_install_offer

### lines 630-632  _(unsure)_

```python
def run_install_offer(parent, offer, *, confirm=None, inform=None,
```

What pressing the word actually does

### lines 682-684

```python
inform(title, offer.as_text())
```

RUNS NOTHING. This is the branch instruction 158 B exists for: a prompt that runs pip here either fails, or succeeds at breaking the install, and the second is worse.

### lines 709-711

```python
if not confirm("This moves packages spaCR depends on",
```

THE SECOND CONFIRMATION NAMES WHAT MOVES. Not "are you sure" -- the packages, with their versions, in the sentence the user has to agree to.
