# Notes from `spacr/picture_settings.py`

Prose lifted out of `spacr/picture_settings.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (20 entries)
- [applies_to](#applies_to) (3 entries)
- [bounding_box_only](#bounding_box_only) (1 entry)
- [_as_channel_mapping](#_as_channel_mapping) (1 entry)
- [to_crop_settings](#to_crop_settings) (10 entries)
- [draw_crop](#draw_crop) (1 entry)
- [available_coordinate_columns](#available_coordinate_columns) (1 entry)
- [offered_values](#offered_values) (5 entries)

## Module level

### lines 28-32

```python
"crop_shape",
```

THE SHAPE OF THE CUT, and it left STREAM_ONLY because a crop already on disk still has a mask to cut against -- so an object-shaped crop is a real choice there too. The one route that cannot offer it is the database one, which has coordinates and no outline, and that is said in `applies_to` rather than by filing it under a mode.

### lines 44-46

```python
"channels",
```

Which channels are shown. It means "of the PNG" when loading and "of the merged array" when streaming, which is the same question asked of two sources -- not two settings.

### lines 48-53

```python
"show_all_in_well",
```

THE WELL, OR THE CANDIDATES. Asked for 2026-08-19: "an option to show all the images from each well and highlight the cells most likely to be whatever gene is picked". Not the default, because the two answer different questions -- the filtered view asks which cells look like the effect, this asks what the whole well looks like and where they are in it -- and a reader who cannot see the well cannot judge the window.

### lines 55-59

```python
"crop_source",
```

THE MONTAGE'S OWN CONTROLS, moved off the toolbar 2026-08-19: "the half-width baseline, score column, and max objects can be moved to the settings panel", and the three combos beside them read as stray labels once the row was crowded. A toolbar is for what you change constantly; these are set once for a screen.

### lines 65-69

```python
"cell_picking",
```

WHICH CELLS BELONG TO THE COEFFICIENT (instructions 172 and 173). 'rank' is heuristic 1 -- the top x by score. 'attributed' is each cell's posterior of carrying the guide. 'assigned' is the constrained assignment, where every cell in the well gets exactly one guide and each guide gets exactly the cells its reads imply.

### lines 77-85

```python
"show_all_in_well": True,
```

ON BY DEFAULT (207). Asked for 2026-08-21: "change that to default by the way", and the reason is the report it came from -- with it OFF the only objects on screen are the ones already annotated to the guide, so every visible object is a hit and every visible fraction is 1. That is what produced "a tone of dotts at 1 and a tone of datapoints at 0".

A VIEW THAT SHOWS ONLY THE CELLS AGREEING WITH THE ANNOTATION CANNOT DISAGREE WITH IT, which makes it useless as a check and misleading as a picture.

### lines 89-90

```python
"object_array": "",
```

EMPTY MEANS THE OBJECT TYPE'S OWN PLANE. A number names a specific plane of the merged array instead.

### lines 92-95

```python
"red_channel": DEFAULT_PNG_CHANNEL_MAPPING.get("r", 2),
```

THE SHIPPED MAPPING, so a panel nobody touches cuts what it always cut. spaCR's own default is r=2, g=1, b=0 -- the inverted order the PNG path has always used -- and stating it here is what makes it visible and changeable instead of implicit.

### lines 105-107

```python
"crop_shape": "object",
```

NAMED, not left to fall out of the dialog. Without an entry here the settings dict said None while the settings WINDOW showed "object" the user reads one thing and the crop is cut by another.

### lines 109-114

```python
"normalize_channels": "",
```

"NOTHING NORMALISED" AND "NOTHING OUTLINED", spelled the way the chooser spells them. The annotator ships None for both and `_as_channel_list` reads None and '' identically, so this changes no behaviour -- it makes the dialog's default one of the options it offers, instead of a value that matches none of them and so opened the chooser on an entry the settings did not hold.

### lines 117-120

```python
"edge_image": False,
```

A REAL BOOLEAN. `set_annotate_default_settings` ships the STRING 'False' for this, and a non-empty string is TRUE -- so "draw the outline over the picture" was on by default everywhere it was read as a flag, and the settings window drew a text box saying False.

### lines 142-150

```python
"object_array": (
```

THE PLANE THE ARRAY ROUTE READS ITS LABELS FROM. There used to be two fields for this -- `object_array` and `mask_array` -- described differently and doing the same job, which left the panel asking the same question twice and the two able to disagree. One field, and it is this one.

The database route has no use for it: it locates a row by the coordinate columns its object type names, and a crop already on disk was located when it was written.

### lines 154-163

```python
"red_channel": (
```

WHICH SOURCE CHANNEL FEEDS EACH COLOUR. Cutting from merged/*.npy means choosing planes out of an array that may hold any number of them; a crop already on disk was made from a choice taken when it was written.

THE PROBLEM THESE SOLVE: the mapping used to be fixed, so an array whose nucleus is plane 0 came out with the nucleus in whichever colour the default put plane 0 in -- "with stream i get the nucleus red". They also let a plane be picked that is not one of the first three: 1, 2 and 4 out of five is a mapping, not a slice.

### line 222

```python
("Source", ("crop_source", "image_type", "object_type", "object_array",
```

WHERE THE PIXELS COME FROM, and how the object is found in them.

### lines 225-226

```python
("Channels", ("channels", "red_channel", "green_channel",
```

WHICH SOURCE PLANE IS DRAWN IN WHICH COLOUR, and which colours survive into the picture. Four controls that are one question.

### line 229

```python
("Picture", ("crop_size", "object_size", "normalize_channels",
```

HOW THE OBTAINED CROP IS DRAWN: its size and its contrast.

### lines 232-233  _(unsure)_

```python
("Outline", ("outline", "outline_threshold_factor", "outline_sigma",
```

The outline is six controls of its own and nothing else depends on them, which is exactly what a tab is for.

### lines 236-238

```python
("Which cells", ("cell_picking", "picking_threshold", "show_all_in_well",
```

WHICH OBJECTS ARE DRAWN AT ALL -- the montage's own question, and the only group here that changes what is in the picture rather than how it looks.

### lines 600-602

```python
DRAW_SETTINGS: Tuple[str, ...] = (
```

Drawing a crop the way the annotator would

### lines 933-935

```python
MEASURED_PAGE_SIZES: Tuple[int, int, int] = (24, 36, 60)
```

What a cap actually costs

## applies_to

### lines 329-331

```python
return chosen == STREAM_IMAGES
```

ONLY THE ARRAY ROUTE READS A PLANE. The database route locates a row by the coordinate columns its object type names, so a plane index there would be a setting that changes nothing.

### lines 334-336

```python
return chosen == STREAM_FROM_DB
```

ONLY THE DATABASE ROUTE ASKS WHICH OBJECT. The type is how that route finds its coordinates -- it names the columns -- so it is the question there and silent everywhere else.

### lines 339-342

```python
return chosen != STREAM_FROM_DB
```

THE DATABASE ROUTE HAS NOTHING TO FOLLOW. Coordinates give a rectangle; an outline needs the labelled plane the array route reads, or the mask a written crop was cut against. So the shape is a real choice in both of those and a box in the third.

## bounding_box_only

### lines 389-392

```python
return chosen == STREAM_FROM_DB
```

THE DATABASE ROUTE HAS NOTHING TO FOLLOW. Coordinates give a rectangle; only the labelled plane the array route reads carries the object's own outline. The other two routes both have one, so the box is a choice there rather than the only answer.

## _as_channel_mapping

### lines 433-435

```python
return {key: (base.get(key) if key in parts else None)
```

A COLOUR THE USER DID NOT PICK IS BLANK, not absent: an absent key would fall back to the default and quietly put a plane back that they turned off.

## to_crop_settings

### lines 489-500

```python
mapping = _as_channel_mapping(value, items)
```

COLOUR LETTERS ARE THE VOCABULARY, and they are translated rather than dropped. Asked for 2026-08-19: "in the anotation app, r,g,b is used. i want this to be consistent, so use r,g,b in the regression cell feature."

THROUGH THE SCREEN'S OWN MAPPING, never by position. spaCR's default is {r: 2, g: 1, b: 0} -- 'r' is source channel TWO so reading 'r,g,b' as 0,1,2 would hand the streamer the planes in reverse and produce a crop that looks plausible and is wrong. `png_channel_mapping` is emitted rather than a `png_dims` list because the list's positional convention is the legacy inverted one; the mapping says what it means.

### lines 505-521

```python
indices = _as_indices(value)
```

THE ANNOTATOR'S `channels` IS NOT THE CROP LAYER'S `png_dims`, and this mapping treated them as one thing. `channels` defaults to 'r,g,b' -- which COLOUR PLANES OF AN EXISTING PNG to show, a display choice the renderer makes with the annotator's own `filter_channels_pil`. `png_dims` is which SOURCE ARRAY CHANNELS to cut, and it must be indices.

Handed the letters, `resolve_png_channel_mapping` reached int('r') and the montage died with "invalid literal for int() with base 10: 'r'" -- from inside the worker, so what the user saw was "The montage load failed" with no mention of a setting. This is 145 exactly: one idea, two vocabularies, and the code in between assuming they agree.

So only an INDEX form crosses over. Letters stay a display setting and reach the renderer, which is where they mean something.

### lines 528-531

```python
value = [int(value), int(value)]
```

`crop_size` (called `img_size` until 2026-09-19) is ONE number -- a single spin box -- and `png_size` is a (width, height) pair. Handed the scalar, `crop_spec_from_settings` raised "'int' object is not subscriptable" from inside the montage worker.

### lines 534-535  _(unsure)_

```python
if applies_to("crop_shape", mode):
```

The SHAPE of the cut, which only streaming decides -- a crop already written to disk was cut when it was written.

### lines 544-551

```python
if mode == STREAM_FROM_DB:
```

WHICH OBJECT, AND HOW IT IS FOUND -- and the two routes answer that with different things, which is why each is asked for in exactly one mode.

The DATABASE route is located by object type: the type names the coordinate columns through `stream_dataset.coordinate_column`, so asking for the columns as well would be asking the user to repeat what they just said, and to be wrong about it.

### lines 556-559

```python
chosen = {}
```

THE COLOURS THE USER MAPPED, which beat the letters: a mapping says which plane is red, and the letters can only say whether red is drawn at all. A field left blank means that colour is not drawn, which is how a two-channel picture is asked for.

### lines 575-576

```python
if mode == STREAM_FROM_DB:
```

THE SOURCE IS THE METHOD. Two entries in one list rather than a mode and a second setting that has to agree with it.

### lines 580-582

```python
out["stream_method"] = "array"
```

The ARRAY route is located by a labelled plane, and

`object_array` names it. Blank means the object type's own plane, which is what a panel nobody touched should cut.

### lines 586-591

```python
out["object_array"] = plane
```

ALWAYS `object_array`, WHICH IS WHAT THE CROP LAYER READS. This used to write an integer plane index to `mask_array`, a setting `stream_dataset` never consulted -- it takes the plane from `object_array` for both methods -- so a panel that named a plane by index was answered by silence. `mask_array` was retired on 2026-09-09 (357-Q4).

### lines 593-594

```python
if bounding_box_only(items):
```

AND THE BOX WINS WHERE IT IS THE ONLY CUT AVAILABLE, rather than the panel promising an outline the route cannot follow.

## draw_crop

### lines 718-719

```python
shown = _as_channel_list(items.get("channels"))
```

LAST, because zeroing a channel before the outline is computed would outline a channel that is no longer there.

## available_coordinate_columns

### lines 760-761

```python
names = getattr(frame, "columns", None)
```

`or ()` on a pandas Index raises -- an Index has no truth value. The guard has to be an explicit None check.

## offered_values

### lines 862-866

```python
if name == "crop_shape":
```

`coordinate_columns` is deliberately not offered: the database route derives its columns from the object type, and a second control for the same fact is one the user can set to disagree with the first. `available_coordinate_columns` stays -- it is what the derivation checks against.

### lines 870-875

```python
return modes()
```

THROUGH `modes()`, NOT A SECOND LIST. This built its own pair and dropped the database route entirely, so the panel offered two of the three modes it implements and no amount of filling in the database settings could reach one. `modes()` is the table; a second copy of it is a mode that exists everywhere except where the user can choose it.

### lines 878-881

```python
return (
```

WHICH PLANES SURVIVE INTO THE PICTURE -- the annotator's own

`filter_channels_pil` question. Offered for the same reason as the two below: "showing only one channel ... none of this works" was a free-text box with no statement of what it wanted.

### lines 892-900

```python
what = ("normalised" if name == "normalize_channels" else "outlined")
```

OFFERED, NOT TYPED. These were blank QLineEdits: nothing on screen said that the answer is a channel list, so a user who typed nothing got nothing and a user who typed "0,1,2" -- which is what every other channel setting in spaCR takes -- got a control that accepted their input and did nothing. Reported twice.

The stored value stays the annotator's comma-separated string, so a settings CSV written before this still means what it meant, and free text is still accepted by `_as_channel_list`.

### lines 913-914  _(unsure)_

```python
return ("cell", "nucleus", "pathogen", "cytoplasm",
```

Every object a measure run can write a plane for, in the order the pipeline names them.
