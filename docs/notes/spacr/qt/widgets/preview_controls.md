# Notes from `spacr/qt/widgets/preview_controls.py`

Prose lifted out of `spacr/qt/widgets/preview_controls.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_flat_qss](#_flat_qss) (1 entry)
- [_FlatStyleMixin.showEvent](#_flatstylemixinshowevent) (1 entry)
- [FlatComboBox.__init__](#flatcombobox__init__) (1 entry)
- [FlatSpinBox.__init__](#flatspinbox__init__) (1 entry)
- [ImageSet](#imageset) (1 entry)
- [_get_regex_callable](#_get_regex_callable) (1 entry)
- [_acquisition_regex](#_acquisition_regex) (1 entry)
- [enumerate_image_sets](#enumerate_image_sets) (5 entries)
- [ImageSetSampler.set_max](#imagesetsamplerset_max) (1 entry)

## _flat_qss

### lines 165-167

```python
f"{selector}#{FLAT_CONTROL_NAME}::up-button,"
```

QSpinBox draws two framed arrow buttons of its own. Left alone they are the only chrome in a row that is otherwise pure text, so strip them the same way the combo's drop-down is stripped.

## _FlatStyleMixin.showEvent

### lines 187-189

```python
"""Rebuild the flat style each time the control comes back on screen.
```

Preferences can change the theme while this panel is hidden; the widget stylesheet keeps whatever palette it was born with until it is rebuilt, so rebuild it every time the panel comes back.

## FlatComboBox.__init__

### lines 226-229

```python
self.setProperty("i18nSkipItems", True)
```

The entries are *data* (file names, channel indices), not prose. Letting the language pass rewrite them would break every lookup that reads ``currentText()`` back — the same trap that silently reverted the live preview's outline colour to its default.

## FlatSpinBox.__init__

### lines 299-300  _(unsure)_

```python
self.setMaximum(10_000_000)
```

Wide open until a folder is enumerated; configure_max_sets_box then clamps it to the number of sets that actually exist.

## ImageSet

### lines 436-438  _(unsure)_

```python
@dataclass(frozen=True)
```

Enumerating image sets without loading them

## _get_regex_callable

### line 532  _(unsure)_

```python
spec = importlib.util.find_spec("spacr.utils")
```

find_spec locates the file without executing it.

## _acquisition_regex

### lines 569-570

```python
return None
```

An unparsable custom regex must degrade to "one set per file", never take the panel down.

## enumerate_image_sets

### lines 599-602

```python
patterns: Dict[str, "re.Pattern"] = {}
```

One pattern per file extension, looked up by the extension rather than tried in turn: the acquisition regex back-tracks heavily, and on a 98 304-file plate trying all five suffix variants per name cost 713 ms against 368 ms for the single right one (os.scandir alone is 40 ms).

### lines 632-635

```python
slice_id = str(groups.get("sliceID") or "")
```

sliceID is what separates the planes of a stack. Under cellvoyager/cq1 the regex has this group and every plane is its own name; under metadata_type='auto' it does not, and a field is one file per channel already.

### lines 639-642

```python
key = ("", "", name)
```

Not an acquisition name: one set per file, labelled with the file name exactly as the dropdown always showed it. Keyed on the *name*, not the stem, so ``a.tif`` and ``a.tiff`` stay two entries rather than colliding.

### lines 645-648

```python
grouped.setdefault(key, {}).setdefault(chan, []).append(
```

Collect every plane rather than keeping the first and dropping the rest. `setdefault(chan, name)` silently threw away 20 planes of a 21-plane stack, and which one survived was decided by directory order.

### lines 658-659

```python
ordered[chan] = [name for _sort, name in sorted(entries)]
```

Acquisition order, by sliceID where the regex reports one and by name otherwise, so plane 2 does not sort between 19 and 20.

## ImageSetSampler.set_max

### line 872  _(unsure)_

```python
self._pinned = None
```

A new cap is a new draw; the old pin has no claim on it.
