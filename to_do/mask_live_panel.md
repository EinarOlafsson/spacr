# Mask live panel — what is left

Status as of 1.5.0.2. Everything under "Done" is verified against
cellvoyager names (three fields, channels 01/02/04, six z planes) through
the load path a folder drop actually uses, not by calling the functions
directly.

## Done

- **Regex confirmed on every import**, not only when validation fails. A
  regex can capture every required field and still capture the *wrong*
  one; a well ID read as a field ID validates and mislabels the plate.
- **Every z-plane kept.** `enumerate_image_sets` used
  `setdefault(chan, name)`, so a 21-plane stack became one arbitrary plane
  chosen by directory order. `ImageSet.planes` now holds all of them in
  acquisition order, and `z_count` reports the depth.
- **The configured regex reaches the preview.** It always enumerated with
  the default dialect, so confirming a regex had no effect on the panel
  beside it. The sampler cache key includes the dialect, so changing it
  re-enumerates.
- **MIP switch**, beside the set controls. Projects per field and channel,
  the same reduction `_rename_and_organize_image_files` applies before
  masking. Disabled with a reason when there are no stacks; states the
  plane count when there are; projects within a timepoint only for 4-D and
  never infers axis order.
- **MIP is a mode.** It holds across field and channel changes instead of
  needing a re-click per image.
- **Set table** — one row per field, one column per channel, `(nz)` per
  cell. Column header changes channel and keeps the field; row header
  changes field and keeps the channel.
- **Both dropdowns gone** from the row. They stay constructed and hidden
  because `apply_sample_to_combo` fills one, the saved view state names a
  field through it, and `selected_channel()` reads the other.
- **Table and images share a splitter**, table stretched to full width.
- **Choose image** pins its set into the table even when the random sample
  did not draw it.
- **Shift selection and an image cap.** Shift-click extends; shift on a row
  header takes every channel of that field, on a column header every field
  of that channel. The cap field sits between MIP and the set count. Over
  the cap the most recent survive. Last selected is the active one.

## Not done

### 1. Draw more than one image  — the big one

`_selected_cells` is tracked, capped and correct, but **only the active
image is drawn**. Selecting four still shows one.

The panel is built around `self._image` and `self._masks` as singletons and
every render path reads them. This is a refactor, not an addition:

- replace the fixed `_src_view` / `_mask_view` pair with a grid driven by
  `_selected_cells`
- give the active pane a blue border, and route live settings to it
- show **one** intensity view per field when more than one image is shown,
  rather than the current two-per-field
- decide what a mask overlay means per pane, and keep per-pane mask state

### 2. Right-click cross-overlay

Not started. Right-click any shown image and overlay objects or outlines
from another image **in the same set**:

- outlines of different object types in different colours
- overlaid objects in the random-colour cmap
- an opacity control

Needs a compositor that does not exist yet, a context menu, and per-overlay
colour/opacity state.

### 3. Smaller things

- The channel dropdown is hidden, but while it existed it labelled channels
  `Ch 0, Ch 1, Ch 2` from a count while the table reads `ch 01, ch 02,
  ch 04`. If it is ever shown again it should use the real IDs.
- Verify the console/chat splitter reads as a hairline after pulling. The
  handle is 1px and the theme styles all handles at 1px, so it should
  already match the settings/console line.
- Clicking a table cell is verified to load the right file, but only
  through the synchronous path. The async reload branch is untested.
- Nothing here has been tried on a real plate. The fixtures are tens of
  files; a 10 752-file plate exercises sampling and enumeration costs these
  do not.

## Notes for whoever picks this up

The recurring failure in this work was testing a function by calling it
directly, which proves the function works and proves nothing about whether
anything calls it. Every bug the user hit was in the wiring:
`_populate_set_table` hung off the wrong caller, `DEFAULT_METADATA_TYPE`
never imported, cell clicks routed through a set-keyed dropdown that could
not express a channel change.

Drive `load_source_async()` with a real `QEventLoop` and inspect the
widgets afterwards. That is the only check here that has caught anything.
