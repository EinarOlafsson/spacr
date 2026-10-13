# Notes from `spacr/qt/widgets/activity_spinner.py`

Prose lifted out of `spacr/qt/widgets/activity_spinner.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ActivitySpinner.__init__](#activityspinner__init__) (1 entry)
- [ActivitySpinner._connect_registry](#activityspinner_connect_registry) (1 entry)
- [ActivitySpinner._sync](#activityspinner_sync) (1 entry)
- [ActivitySpinner._show_now](#activityspinner_show_now) (1 entry)
- [ActivitySpinner.paintEvent](#activityspinnerpaintevent) (1 entry)
- [attach_activity_spinner](#attach_activity_spinner) (2 entries)

## ActivitySpinner.__init__

### lines 171-172

```python
self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
```

Nothing here reacts to the mouse, and a transparent-for-mouse widget cannot swallow a click meant for the button beside it.

## ActivitySpinner._connect_registry

### lines 219-221

```python
self._auto = False
```

A spinner that cannot find the registry is a spinner that never turns. It must never be a spinner that stops the screen it lives on from opening.

## ActivitySpinner._sync

### lines 294-295

```python
self._delay.stop()
```

Hiding is immediate and unconditional. Anything else would leave the widget saying something that is not true.

## ActivitySpinner._show_now

### lines 334-337

```python
if self.isVisible() and not self._timer.isActive():
```

``isVisible`` is False while an ancestor is hidden, and the whole idle-costs-zero claim rests on never running the animation timer for pixels nobody can see. ``showEvent`` starts it if and when the screen this lives on comes back.

## ActivitySpinner.paintEvent

### lines 426-428

```python
for strand, phase in zip(strands, (0.0, math.pi)):
```

The two strands are the same wave half a turn apart -- which is what makes it read as a double helix rather than as a wobbling line.

## attach_activity_spinner

### lines 481-482  _(unsure)_

```python
pass
```

Its C++ half is gone (the screen was rebuilt); fall through and install a fresh one.

### lines 498-499  _(unsure)_

```python
spinner.setParent(None)
```

Not a box layout. Nothing sensible to insert into; the caller gets None and the screen opens exactly as before.
