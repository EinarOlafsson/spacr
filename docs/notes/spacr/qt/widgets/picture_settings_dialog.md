# Notes from `spacr/qt/widgets/picture_settings_dialog.py`

Prose lifted out of `spacr/qt/widgets/picture_settings_dialog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [picture_defaults](#picture_defaults) (1 entry)
- [_editor](#_editor) (2 entries)
- [_value_of](#_value_of) (1 entry)
- [PictureSettingsDialog.__init__](#picturesettingsdialog__init__) (10 entries)
- [PictureSettingsDialog.set_mode](#picturesettingsdialogset_mode) (2 entries)
- [PictureSettingsDialog._say_what_the_cap_costs](#picturesettingsdialog_say_what_the_cap_costs) (3 entries)

## picture_defaults

### lines 45-51

```python
out = {}
```

OWN_DEFAULTS WINS WHERE IT SPEAKS, and that is not a preference for our own table: it is where a shipped default is the wrong TYPE for a control. `set_annotate_default_settings` ships the STRING 'False' for `edge_image`, and a non-empty string is TRUE -- so the flag read as on everywhere it was used as one, and this dialog drew a text box containing the word False instead of a checkbox. The annotator's value is still what fills every key OWN_DEFAULTS does not name.

## _editor

### lines 86-89

```python
combo = QComboBox(parent)
```

BUILT FROM THE SCREEN, not typed. Offering `object_array` as free text asks the user to remember what their own screen contains and to spell it the way `measure` did -- and every other chooser in spaCR is built from the data.

### lines 92-99

```python
if isinstance(option, tuple) and len(option) == 2:
```

A chooser may offer (value, label) or a bare value. The STORED value is always the first, so a label can be renamed without changing what any settings file already on disk means.

`stored`, NOT `value`: the first version of this loop unpacked into `value` and so clobbered the parameter it was about to search for -- every dropdown then opened on its LAST entry, whatever the setting actually was.

## _value_of

### lines 148-150

```python
return widget.value()
```

THE PAIR STAYS A PAIR. Read through `text()` like any other unfamiliar editor it would come back as the string "2, 98", and every settings file already on disk holds a two-element list.

## PictureSettingsDialog.__init__

### lines 235-238

```python
from .channel_picker import ChannelPicker
```

THE R,G,B SYSTEM FOR THE TWO CHANNEL SETTINGS (188 B). A dropdown of the eight combinations made "which channels are on" a question you had to open a list to answer, and turning one channel off the thing a user does constantly here -- two clicks.

### lines 242-246

```python
for title, keys in categories():
```

ONE TAB PER GROUP OF QUESTIONS. Twenty-eight controls in one column made the reader scroll past every question they were not asking to reach the one they were; the module screens group their settings the same way and `picture_settings.categories` is the one table both read.

### lines 254-258

```python
value = int(value)
```

CAP IS A COUNT even when a settings table round-trip stores it as a float. Normalise at the boundary, using the same truncation the montage already applies, so the dialog cannot hand a fractional object count back to its caller and the control remains the live-wired QSpinBox.

### lines 263-267

```python
allow_none=(key != "channels"))
```

`channels` with nothing on is a blank picture; `normalize_channels` with nothing on means "normalise nothing" and `outline` with nothing on means "outline nothing" -- both real answers, and `outline`'s default is off.

### lines 276-284

```python
_attach_picking_help(editor, key)
```

THE TOOLTIP IS ON THE LABEL, not the field: a tooltip on the control fires while the user is editing it, which is the one moment they did not ask for it. EACH ANNOTATION METHOD EXPLAINS ITSELF, ON ITS OWN ENTRY (208 C). "The API should be verry specific and evplain exactly how cells are cjhoosen for annotation" -- and five methods cannot be explained in one tooltip that has to fit in 600 characters. Per-entry help is the only place with room, and it is where the choice is actually made.

### lines 293-295

```python
cap = self._editors.get("cap")
```

THE COST FOLLOWS THE NUMBER. Written once at build time it would describe the cap the dialog opened on, which is the one value the user is not asking about while they change it.

### lines 301-307

```python
source_editor = self._editors.get("crop_source")
```

AND THE GREYING FOLLOWS THE MODE CHOSEN *HERE*. `crop_source` is one of the annotator's own controls, so it is a control IN this window as well as on the toolbar that opened it -- and a mode read once at build time described the mode the window OPENED on. A user who switched to streaming here was then told the array and channel selectors their chosen mode does use were unavailable, with a reason naming the mode they had just left.

### lines 321-326

```python
try:
```

THE SAME API HELP THE SETTINGS PANEL GIVES, not a plainer copy. Reported: "there are still no tooltips with api guides". These labels carried the description string and nothing else -- no `settingKey`, no rendered `apiTooltipHtml`, so no link to the API page and none of the typed metadata every other reader keys on.

### lines 334-335

```python
pass
```

A dialog that cannot decorate its help is still a dialog that sets the picture. The plain descriptions installed below remain.

### lines 339-342

```python
from ..screens.settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box you type in. Built here on the field, it is moved onto the label as the last step, so every panel in the application explains itself the same way.

## PictureSettingsDialog.set_mode

### lines 363-365

```python
label.setToolTip(why_not(key, self._mode))
```

THE REASON BEATS THE DESCRIPTION when the control is greyed: what the user is asking at that moment is why they cannot touch it, not what it would have done.

### lines 368-373

```python
rich = str(label.property("apiTooltipHtml") or "")
```

THE RICH API HELP IF IT WAS INSTALLED, and the plain description otherwise. This line used to write the plain text unconditionally, straight over the HTML `install_api_tooltips` had just rendered -- so every label carried the API metadata and showed none of it. Reported as "there are still no tooltips with api guides".

## PictureSettingsDialog._say_what_the_cap_costs

### lines 395-397

```python
base = self._cap_help
```

THE HELP IS REMEMBERED, not read back off the label. Re-reading it appends the new sentence to the last one, so a reader who tried three caps got three of them.

### lines 404-406

```python
label.setToolTip(base)
```

ZERO HAS NO PRICE. Remove the previous count's sentence rather than leaving a tooltip that describes a value no longer in the control; the setting's original help remains truthful at zero.

### lines 409-411

```python
joiner = "<p>{0}</p>" if base.lstrip().startswith("<") else "\n\n{0}"
```

PLAIN OR RICH, whichever the label ended up with: the API help is HTML, so the sentence is appended as a paragraph there and as a blank line on a plain tooltip.
