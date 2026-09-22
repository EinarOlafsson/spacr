# Notes from `spacr/qt/screens/make_masks.py`

## Magnifier integration notes (2026-09-15)

- `_MaskCanvas.wheelEvent`: Shift + wheel changes the magnifier size, never
  the view zoom. Some platforms report a shifted wheel as horizontal, so a
  horizontal notch has the same meaning. With no Shift modifier the wheel
  changes the magnifier zoom while the magnifier is enabled.
- `_LiveMagnifier` worker delivery: when a model fails to load, the newly
  computed settings key identifies the Classical fallback that actually ran.
- `MakeMasksScreen._note_curated`: read `self._queue.folder` outside the
  write-error handler. Only a failure to persist the record (for example a
  full disk or a read-only sync folder) should be swallowed after the mask
  is safe. A missing queue is a programming error; catching its AttributeError
  here would incorrectly report it as a record-write failure.

## Item 419, points 1-3 (2026-09-19)

- `_MaskCanvas.update_readout` / `readout_text` / `_paint_readout`: the
  readout in the image's top-left corner. It is fixed in the corner and its
  CONTENT follows the mouse, which is what "in the top left corner" asked
  for; a label that followed the cursor would sit on the object being read.
  The numbers come from `mask_engine.ObjectLookup`, which measures exactly as
  `filter_objects` does (canonical id, pixel count, float32 mean over the
  object's pixels in raster order), so a bound typed from the readout
  predicts the filter. The mean is written to the two decimals the filter's
  intensity boxes take; a bound within 0.005 of a mean whose third decimal is
  not zero can land either side of it, because the box rounds what is typed.
- `_MaskCanvas._object_lookup`: a refresh marks the lookup stale, and a stale
  lookup is compared with a copy of the mask it was built from before it is
  rebuilt. Zoom, pan and resize all refresh without changing a label, and a
  rebuild costs tens of milliseconds on a 2048 px field (measured: 65-90 ms
  at 2048 x 2048 with 400 objects; 15 ms at 1024 x 1024 with 200), where the
  comparison costs about 2 ms.
- `_MaskCanvas.mouseMoveEvent`: while a button is held the readout reports
  only the pixel. A brush stroke changes the mask on every move, and
  re-measuring each time would put a lookup rebuild on every mouse event of
  the stroke. `mouseReleaseEvent` and `refresh` queue a re-read for when the
  button is up, so an object erased under a resting mouse stops being
  reported without the mouse moving.
- `MakeMasksScreen._build_ui`: the settings are the splitter's FIRST pane and
  the views the second. `SETTINGS_GAP` is the splitter handle's width, so the
  gap the maintainer asked for is also where the pane is dragged wider.
  `_on_toggle_settings` reads and writes index 0 accordingly.
- `MakeMasksScreen._build_ui`: the masthead has no description. That sentence
  was also the masthead's only route to the API page (its hover help); the
  maintainer asked for it to go, and every setting's label still links its
  own API entry.
- `MakeMasksScreen._build_tool_row` / `add_toolbar_action`: the Magnifier is
  inserted directly before Settings, and later actions are inserted before
  the Magnifier, so the pair stays adjacent whatever is added; the stretch is
  after Settings, which keeps the row against the left edge above the
  settings it toggles.
- `MakeMasksScreen._offer_backend_install`: replaced the synchronous
  `subprocess.run` of c60d48e35, which froze the window for the whole pip run
  ("the window will not respond while it runs"). The install is now
  `model_install.PackageInstall`, a QProcess watched from the event loop,
  with a busy bar and pip's latest line under Mode. The box goes back to the
  mode it was on when a missing row is chosen, and a finished install selects
  the new mode. Whether the package can be found is asked again after pip
  exits 0, because pip can succeed and leave nothing importable.
- `MakeMasksScreen._fill_zoo_models` / `_keep_model_loadable` /
  `_on_model_activated` / `download_zoo_model`: a zoo Cellpose model not on
  this machine was a DISABLED row, which cannot be chosen at all. It is now a
  greyed row that stores no path; `activated` (a click) starts the download,
  and `_keep_model_loadable` puts the box back if the keyboard or the wheel
  lands on such a row, so `_magnifier_context()["model_name"]` never falls
  back to cpsam behind the user's back.
- `_cp_fetched`: `model_zoo.fetch` files a download under
  `versioned_path(folder, entry.name)`, which strips a trailing `_v1`
  (`foo_v1.cp_model` lands as `foo.cp_model`). `_zoo_cellpose_models` looks
  for `folder/entry.name`, as the picker's `_local_path_on_disk` does, so a
  zoo name ending in `_v1` never shows as downloaded after a restart. None of
  today's zoo names end that way; the session map covers the rest of the
  session, and the lasting fix belongs in `model_zoo` (a lookup of where an
  entry was installed), which this item does not own.

## Item 419, points 4-6 (2026-09-19)

- `SHORTCUT_HINTS` / `_build_shortcut_panel` / `_build_view_pane`: the
  shortcut list the maintainer asked for "to the right of the mak, cell
  probability, Flows". Those three are the view TABS, so the splitter's
  right-hand pane became the tabs and the list side by side rather than the
  tabs alone; `_view_pane` is that pane, and the two layout tests that read
  `indexOf(self._view_tabs) == 1` now read the pane and assert the tabs are
  inside it. The width is fixed (`SHORTCUTS_WIDTH`) so a wider window gives
  its pixels to the image, and the Settings toggle does NOT hide the list:
  it is not settings, and a shortcut list that disappears exactly when the
  screen is cleared for work is one you can only read when you do not need
  it.
- Every line of `SHORTCUT_HINTS` is a gesture this module implements, and
  `tests/qt/test_make_masks_shortcuts_otsu_and_object_edits.py` drives each
  one on the canvas rather than reading the table back. A panel of plausible
  sentences is the failure mode here: the keys change, the panel does not,
  and nothing is red. Writing the tests moved one line --
  "Choose every object the drag passes" became "Add the objects it passes
  over", because a drag across two objects with background between them
  added only the first, and the panel must not claim more than the gesture
  does. What a drag adds in each save mode is item 417's file to state.
- `canonical_magnifier_mode` / `_MAGNIFIER_MODE_ALIASES`: point 5 renamed the
  magnifier's `classical` mode to `otsu`. A mode name reaches this module
  from a script, a test and (through `set_mode`) anything that remembered
  one, so the old key still runs and is translated to the new one at the
  door: `_MAGNIFIER_SEGMENTERS` is keyed on the new name only, and
  `_segment_region`, `set_mode` and `_model_settings` all canonicalise first.
  `mask_engine._classical_region_labels` keeps its name -- it is another
  module's private function, with its own tests, and renaming it would have
  been a second change dressed as this one.
- `_RENAMED_CATEGORIES`: the folded-categories preference is a list of
  TITLES, so renaming "Cellpose-SAM" to "Object detection" would have quietly
  un-folded it for every user who had folded it. The stored list is read
  through the map; what is written back is the new title, so the migration
  happens once.
- `_build_otsu_card` / `_otsu_settings`: the threshold correction was the
  only Otsu setting there was and it sat at the bottom of the Cellpose-SAM
  category, where nobody looking for Otsu would open it. It moves to a
  category of its own with the Bright switch and four new settings, and ONE
  reader (`_otsu_settings`) feeds both the button and, through
  `_magnifier_context`, the magnifier.
- THE OTSU DEFAULTS MOVE THE BUTTON, on purpose. Before this the magnifier's
  Otsu mode smoothed by 1 px, filled holes and split a blob with two centres,
  and Otsu detect did none of the three: the box under the mouse and the
  button disagreed about the same field and nothing said so. The boxes start
  where the preview has always been, so the box is a PREVIEW of the button;
  three clicks put the plain threshold back.
  `mask_engine._otsu_instances`'s own defaults are unchanged, so nothing that
  calls it directly moved.
- "PREVIEW" AND NOT "THE SAME FUNCTION", and the difference is measured,
  because an earlier draft of this note claimed the button "now gives what
  the box showed". Closing those three gaps does not make the two one
  routine: `_classical_region_labels` still opens the binary image, still
  offsets Otsu's level by the magnifier's own Sensitivity, and still falls
  back to a noise-floor cut where a region holds no two clear populations,
  and `_otsu_instances` does none of the three. Driven from the panel
  defaults over twelve synthetic 96x96 fields of three to six bright discs
  on noise (2026-09-19) the two agreed on the object COUNT twelve times out
  of twelve and were pixel-identical in two, the other ten differing by 3 to
  16 boundary pixels in 9,216. The box tells a curator what the button is
  about to do; it does not promise the same array.
- `OTSU_SMOOTHING` is a hand-written copy of the private
  `mask_engine._CLASSICAL_SMOOTHING`, because a screen importing a private
  name from the engine is worse than a duplicated float -- but an unpinned
  copy is the same silent disagreement one level up, so a test holds the two
  equal, along with `_MagnifierRequest.otsu_smoothing`'s default.
- `_MODEL_SETTING_FIELDS`: the three new Otsu settings are APPENDED. A
  magnifier request key is this tuple positionally after `(field, box)`, and
  an insertion in the middle would make every cached key mean something else.
- `_on_dilate` / `_on_shrink` / `_on_clear_mask`: point 6's three buttons.
  Clear already existed and already confirmed; what changed is that it says
  how many objects are about to go, which is the one fact that decides the
  question. Shrink reports how many objects it erased, because erosion
  deletes anything thinner than twice the step and a curator who has just
  lost eleven objects needs to be told while Undo is still the obvious thing
  to do. Both go through `_apply_op`, so each is one undo step and one
  ledger entry carrying its step.
- BOTH NUMBERS IN SHRINK'S SENTENCE ARE THE COUNT BEFORE THE EDIT, and the
  first is deliberately the count the button was pressed on and not the
  count that survived. The first draft reported the survivors, which made
  ten objects with seven erased read "Shrank 3 object(s) ... 7 object(s) are
  gone" -- an arithmetic puzzle over a field where all ten were eroded --
  and made emptying the field read "Shrank 0 object(s)".
## Item 407, the magnifier's last status lines (2026-09-19)

- `_LiveMagnifier.click`, `_on_delivered` and `_note_fallback`: the three
  sentences item 407 left behind now go through `tr` with NAMED values
  rather than being built as f-strings. An f-string is assembled before
  anything can translate it, so the region mode's two failures -- the model
  that could not segment the region, and the model that could not load at
  all -- reached a reader in Japanese in English. The error and the mode are
  values in the template (`{error}`, `{mode}`, `{reason}`), which is what
  lets a catalog reorder them: a locale that puts the reason first can, and
  one that drops the brackets can.
- `_magnifier_mode_label`: the fallback note names the mode the way the Mode
  box names it. It used to print the internal key, so a user who had chosen
  "Cellpose 3 · cyto3" was told `cellpose3:cyto3` could not run and had no
  row to look for. The caption itself is translated, so the sentence and the
  box agree in every language. An unknown key is handed back unchanged: a
  mode that has lost its caption still reads as itself rather than vanishing.
- `MakeMasksScreen._on_toggle_magnifier` and `_commit_magnifier_result`: the
  toggle's two sentences and the "nothing to add" refusal were already
  collected as catalog sources -- the generator reads a `setText` literal --
  and were shown without `tr`, so the rows existed and nothing used them.
  Wrapping them changes no source and owes no translation.

## Item 435 (2026-09-20)

- `_MaskCanvas.displayed_source` / `invert_display`: the complement is
  computed at the one place the canvas paints and nowhere else, so the
  corner readout, the object filter, both detect buttons, the live
  magnifier's crop and the saved mask all go on reading `canvas.image`. A
  view inversion that leaked into what is measured would be worse than the
  defect it fixes, because a curator would be filtering on numbers that are
  not the data; the tests drive each of those five paths with Invert on.
- The complement is cached against the identity of `image` because
  `refresh` runs on every point of a brush stroke, and re-subtracting a
  megapixel field per point would be felt on the brush.
- "Invert image" is in the DISPLAY category, beside the two contrast
  percentiles, because that is what it is. Item 419 point 9 is a SECOND,
  separate invert, in Object detection, which makes the detectors work on
  inverted pixels on purpose and carries a warning saying so; the two must
  not be confused, and putting this one where the contrast lives is what
  keeps them apart on the panel.
- "Invert mask" in Object operations is now "Swap object and background",
  and it reports its own result. The thing that made it look broken is that
  its result is invisible: one field-sized object reads as one flat wash.
  Nothing on the panel called "Invert" flips the mask any more.
- The Otsu category's three new controls -- `Classes`, `Foreground class`
  and the local threshold with its `Local window` -- are the detect
  BUTTON's only, on the precedent item 419 set with "Drop objects the image
  border cuts". They are judgements about a whole field: a 64 px magnifier
  box rarely holds three populations, and a window the size of the box is
  the box's own threshold, so offering either there would be offering a
  control that does nothing, which is the defect this item exists for.
  `_classical_region_labels`, the magnifier's own routine, is untouched.
- `_sync_otsu_controls` greys out what is not being read: the foreground
  class before there is more than one band to choose from, the window while
  the local threshold is off, and the class count while it is on. A local
  level and a split into several bands have no joint meaning and the engine
  refuses the pair rather than dropping one of the two quietly.
- The minimum object area after the threshold was ALREADY THERE -- item
  419's `Min area` in Object operations, read by `_detect_min_area` and
  passed through to `connected_instances` -- and a second box in the Otsu
  category could only disagree with it. The card points at it in a line
  instead.
- `_OtsuHistogramDialog` is modeless, because the point of a preview is to
  change a setting and look again, and because a static modal runs its
  event loop in C++ and hangs a headless run. It is painted rather than
  plotted: a few hundred bars and two or three lines do not justify
  importing a chart library onto the path that opens Make Masks.

## Item 407, a field is segmented whole once a session (2026-09-19)

- `_LiveMagnifier._keep_image_result` / `_cached_image_result` /
  `_trim_image_cache`: leaving a field and coming back re-ran the whole-image
  model on it. Measured on the RTX 3090 for item 407 that is 3.7-10.3 s a
  field; on a CPU it is minutes, and a curator walking a plate goes back and
  forth constantly. What is kept is the label image, under a name made of
  the FIELD and every setting a model reads, so a run under a new
  Sensitivity neither matches the old one nor evicts it.
- Why the name is not the run key: a run key opens with `_field`, which
  counts LOADS rather than fields, so the same field opened twice carries two
  different keys and could never match itself. `set_field` gives the
  magnifier the image file's path, and `_cache_key` swaps it in for the
  counter. A canvas handed an array with no file behind it has no name and is
  never kept -- nothing could tell two such arrays apart.
- Why `memory_budget.what_to_drop` and not a size of its own: it is what
  every other cache in the application is trimmed by, so the user's idle
  timeout and cache ceiling reach this one too. The ceiling applied is the
  LOWER of that preference and `_MAGNIFIER_IMAGE_CACHE_MB`, because 2 GB of
  label images is not what a user who raised the ceiling for image caches was
  asking for. The field on screen is held by `_image_result` as well, so a
  trim can never take the objects out from under the box.
- A cache hit says exactly what a finished run says -- "{n} object(s) found
  in the whole image" -- because it is true and because it owes no new
  translation. What the user sees instead is the busy bar not appearing.

## Item 407, the box shows what a click would add (2026-09-20)

- `_LiveMagnifier._overlap_preview`: the box outlined what the MODEL found,
  and a click adds what the Overlap rule and Min area leave of it. A click
  that added half an object, or nothing at all, said so only afterwards in
  the status line. The pixels the rule takes away now keep a quarter of
  their alpha, so they read as "found, not yours" beside the solid objects a
  click commits, and the promise the box makes is the one the click keeps.
- It is computed on the GUI thread and not on the worker because what it
  depends on is the MASK, and the mask is the GUI thread's. The answer is
  kept until the result, the mask or the rule changes -- identity, not a
  hash: a mask edit replaces the array, and the magnifier owns the left
  button while it is on, so nothing mutates the mask under the cache.
  Nothing is computed at all under Replace, which takes nothing away, or
  over an empty mask, which has nothing to take.
- Only under "Region under the mouse". Under Whole image a click adds one
  whole object, most of which can lie outside the box, and the rule's answer
  for it cannot be read off the box's slice: Skip asks whether the object
  touches the mask ANYWHERE, and Clip keeps the object's largest surviving
  piece, which may be outside. A preview there needs the object's whole
  extent, and that is left undone rather than approximated.
- `set_overlap` does NOT refresh: the rule is applied to what was found, not
  by the thing that finds it, so changing it asks no model and does not
  discard the whole-image objects. It matches the Size and Zoom rows in that
  and is why it is a setter of its own rather than part of the request key.

## Item 407, the busy bar says how long (2026-09-20)

- `remaining_seconds` / `_note_pace` / `MakeMasksScreen._tick_magnifier_eta`:
  neither Otsu nor Cellpose reports steps, and tiling the field to get real
  progress would cut objects at the seams (that decision is item 407's, from
  2026-09-15, and stands). What CAN be known without either is what the LAST
  run under this mode and model cost per megapixel.
- So the first run of a session promises nothing and the bar stays
  indeterminate. That is the honest answer, and it is also the right one: on
  a cold model most of a first run is the load -- item 407 measured 10.3 s
  against 3.9 s for the same field with the model cached -- so a first run
  would over-promise by a factor of three.
- The pace is kept against the mode and the model and NOT against the field,
  because the whole point is to answer for a field nothing has been measured
  on. Sensitivity and Min area are left out: they move the objects found
  rather than the work done.
- An estimate that runs out goes back to the indeterminate bar rather than
  counting past zero or sitting at 99%. A bar that has stopped being able to
  say how long should stop saying it.

## Item 419, points 7-9 (2026-09-19)

- `_build_tools_panel`'s filter category is called "Filter", as the
  maintainer named it, and is in `_RENAMED_CATEGORIES` beside Cellpose-SAM
  for the same reason: the folded-categories preference stores TITLES, so a
  user who had folded "Auto-filter objects" away would find it open again
  and have to fold it a second time.
- `_filter_log` / `_set_filter_log` / `_filter_removal_line`: the removal
  ledger point 7 asks for, one row per object. The row carries the bound's
  own NUMBER as well as its name -- "removed by minimum area 20" -- because
  a row naming only the bound leaves the reader hunting for the box it came
  from. It is cleared at the START of every run, including the run that
  found nothing and the automatic run a field load makes: the rows name
  objects in the mask ON SCREEN, and a row left over from the last field
  names an object that is not there. Empty shows a placeholder rather than a
  blank red box.
- The log is fixed at `FILTER_LOG_ROWS` rows with the rest a scroll away. A
  box that grew with its contents would push every other category off the
  panel the first time a bound was set too tight.
- THE ROWS ARE DROPPED BY THE NEXT EDIT, in `_record`, which is the one
  place every edit passes through. They promise to name objects in the mask
  on screen and the next edit breaks that promise whatever it was: Ctrl+Z
  puts a removed object back under the id a row still lists, and a detect in
  replace mode rebuilds the mask around it. Undo was the cheap one to reach
  -- one keystroke after a filter left a red row naming an object that was
  visibly back on the canvas.
- `_make_masks_qss` / `MAKE_MASKS_QSS_NAME`: the log's red and the Invert
  warning's amber are REGISTERED QSS, not colours set on the widgets. A
  colour written onto a widget at build time is the colour it keeps when the
  theme changes under it; registered, both follow the user's theme, and the
  test reads the resolved palette rather than the string that was asked for.
- `_MaskCanvas._ctrl_edit_at` and the Ctrl branch of `mousePressEvent`:
  point 8. CTRL IS TESTED FIRST, before the magnifier and before the pan, or
  the two edits would exist only while the magnifier was off -- its press
  handler takes every left click and its whole-image mode takes every right
  one.
- `_MaskCanvas._ctrl_click` exists because of what the RELEASE does. The
  magnifier's `release()` with no stroke open IS a click, so without it a
  Ctrl+click that split one object would commit every object in the box on
  the way back up. It also stops a move with the button still down turning
  the finished edit into a drag.
- It holds the BUTTON rather than a yes/no, and the edit only starts when
  nothing else is down (`event.buttons() == event.button()`). Two buttons
  at once is where a click-shaped gesture goes wrong: Ctrl+left pressed
  during a right-button sweep used to close the SWEEP's stroke and label it
  a split, and then the sweep's own release was swallowed as the Ctrl
  click's, leaving `_sweeping` set with no ledger entry for what had already
  been erased. Now the second button starts nothing and ends nothing, and
  only the button that opened the edit closes it.
- `_MaskCanvas.status`: the canvas had no way to say anything. These two
  gestures are the first that can decline to act for a reason worth telling
  -- a click on background, and an object with no waist -- and a shortcut
  that does nothing and says nothing reads as a broken shortcut.
- `split_min_area` / `_on_min_area_changed`: the split reads the Min area
  box, which is the same judgement about debris the detectors read. An
  object the screen would not keep is not one the gesture cuts in two.
- Every `QCheckBox` on this screen is the package's `Toggle` (point 9a). It
  is a `QCheckBox` subclass, so `setChecked`/`isChecked` and every existing
  connection are unchanged and nothing that found these controls by type
  moved.
- `_cp_invert` / `_detector_image` / `_LiveMagnifier.region_for`: point 9's
  inversion is what the DETECTOR sees, not a display trick, so one function
  on each side produces the array that is segmented. The canvas's own array
  is never inverted in place, which is what keeps the hover readout and the
  Filter category reporting the field's real values -- point 9's own note
  asks for that, since a user filtering by intensity would otherwise be
  judging inverted numbers.
- THREE THINGS ON THIS SCREEN ARE CALLED INVERT AND THE CAPTION IS WHERE
  THEY ARE TOLD APART. Item 435 landed two of them and the module docstring
  named two; this is the third, and it is the only one that changes what a
  detector reads, so it is the one that had to say so in its own name. It is
  "Invert for detection", which is the maintainer's own phrase for it ("a
  new boolean slider ... invert for object detection"), and it carries the
  warning banner. "Invert image" in Display is a view and promises in its
  tooltip that nothing measured moves; "Swap object and background" in
  Object operations flips the LABELS. A panel with two switches both reading
  "Invert", one of which silently redirected both detect buttons, is the
  defect item 435 was filed about wearing different clothes.
- The arithmetic is `mask_engine.invert_for_detection` and NOT 435's
  `invert_intensity`, although the two differ only by which ends they
  reflect about. The detect button thresholds ABSOLUTE intensity and
  multiplies the level by item 417's threshold correction, so an inversion
  that moves the field's span moves what that dial means: on a 12-bit field
  the dtype complement makes a correction of 0.8 label the whole frame and
  1.3 label nothing. The measurement is in `invert_for_detection`'s
  docstring and in the note for `mask_engine`.
- `_LiveMagnifier.inverted_field` caches the WHOLE field inverted, on the
  array's identity, and `region_for` and `paint` both cut from that one
  array. A region reflected about its OWN extremes is reflected differently
  wherever the box is put, so the box would stop being a preview of the
  button; and item 407 moved the box's paint onto `_stretch_for_box`, which
  indexes the field by absolute coordinates and so needs a field rather than
  a crop. One cached array answers both. A full-field pass on every hover
  would be work on the GUI thread for an answer that cannot have changed; a
  new field is a new array and asks again, which is how the canvas already
  caches item 435's display complement.
- `invert` is APPENDED to `_MODEL_SETTING_FIELDS`, like the three settings
  point 5 added, because a request key is that tuple positionally and an
  insertion in the middle would make every cached key mean something else.
  It is on the request although no segmenter reads it: the same box under
  the same settings with Invert on and off are two different questions.
- `_invert_warning` sits between the tool row and the image and NOT in the
  settings panel (point 9e). The Settings toggle hides that panel to give
  the image the width, and the magnifier goes on inverting while it is
  hidden; a warning you can only see when you are not working is not one.

Prose lifted out of `spacr/qt/screens/make_masks.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (8 entries)
- [_MaskCanvas](#_maskcanvas) (3 entries)
- [_MaskCanvas.__init__](#_maskcanvas__init__) (8 entries)
- [_MaskCanvas.set_image_and_mask](#_maskcanvasset_image_and_mask) (2 entries)
- [_MaskCanvas.refresh](#_maskcanvasrefresh) (1 entry)
- [_MaskCanvas._canvas_to_image](#_maskcanvas_canvas_to_image) (3 entries)
- [_MaskCanvas.paintEvent](#_maskcanvaspaintevent) (2 entries)
- [_MaskCanvas._paint_recrop_boxes](#_maskcanvas_paint_recrop_boxes) (1 entry)
- [_MaskCanvas.mousePressEvent](#_maskcanvasmousepressevent) (3 entries)
- [_MaskCanvas.mouseMoveEvent](#_maskcanvasmousemoveevent) (3 entries)
- [_MaskCanvas.mouseReleaseEvent](#_maskcanvasmousereleaseevent) (3 entries)
- [_MaskCanvas._finish_region_gesture](#_maskcanvas_finish_region_gesture) (2 entries)
- [flow_rgb](#flow_rgb) (2 entries)
- [load_cellpose_model](#load_cellpose_model) (1 entry)
- [cellpose_detect](#cellpose_detect) (2 entries)
- [_FlowPane.__init__](#_flowpane__init__) (1 entry)
- [_FlowPane.show_rgb](#_flowpaneshow_rgb) (1 entry)
- [FoldedModulePanel.__init__](#foldedmodulepanel__init__) (2 entries)
- [NapariBridgeScreen.__init__](#naparibridgescreen__init__) (2 entries)
- [NapariBridgeScreen.open_in_napari](#naparibridgescreenopen_in_napari) (1 entry)
- [NapariBridgeScreen.take_mask_back](#naparibridgescreentake_mask_back) (2 entries)
- [MakeMasksScreen.__init__](#makemasksscreen__init__) (2 entries)
- [MakeMasksScreen._build_ui](#makemasksscreen_build_ui) (6 entries)
- [MakeMasksScreen._restate_fold_button](#makemasksscreen_restate_fold_button) (1 entry)
- [MakeMasksScreen._build_folded_screen](#makemasksscreen_build_folded_screen) (2 entries)
- [MakeMasksScreen.seed_folded](#makemasksscreenseed_folded) (1 entry)
- [MakeMasksScreen._build_tool_row](#makemasksscreen_build_tool_row) (6 entries)
- [MakeMasksScreen._on_toggle_settings](#makemasksscreen_on_toggle_settings) (1 entry)
- [MakeMasksScreen._build_tools_panel](#makemasksscreen_build_tools_panel) (5 entries)
- [MakeMasksScreen._install_shortcuts](#makemasksscreen_install_shortcuts) (1 entry)
- [MakeMasksScreen._on_wand_salvage_changed](#makemasksscreen_on_wand_salvage_changed) (1 entry)
- [MakeMasksScreen._on_undo](#makemasksscreen_on_undo) (1 entry)
- [MakeMasksScreen._on_detect_otsu](#makemasksscreen_on_detect_otsu) (1 entry)
- [MakeMasksScreen._build_view_tabs](#makemasksscreen_build_view_tabs) (1 entry)
- [MakeMasksScreen._build_cellpose_card](#makemasksscreen_build_cellpose_card) (1 entry)
- [MakeMasksScreen.run_cellpose](#makemasksscreenrun_cellpose) (2 entries)
- [MakeMasksScreen._warn](#makemasksscreen_warn) (1 entry)
- [MakeMasksScreen._should_background_load](#makemasksscreen_should_background_load) (1 entry)
- [MakeMasksScreen._handle_load_failure](#makemasksscreen_handle_load_failure) (1 entry)
- [MakeMasksScreen._apply_loaded_pair](#makemasksscreen_apply_loaded_pair) (4 entries)
- [MakeMasksScreen._open_ledger](#makemasksscreen_open_ledger) (1 entry)
- [MakeMasksScreen._on_recrop_requested](#makemasksscreen_on_recrop_requested) (1 entry)
- [MakeMasksScreen.recrop](#makemasksscreenrecrop) (3 entries)
- [MakeMasksScreen.finish_recrop](#makemasksscreenfinish_recrop) (2 entries)
- [MakeMasksScreen._on_prev](#makemasksscreen_on_prev) (1 entry)
- [MakeMasksScreen._on_next](#makemasksscreen_on_next) (1 entry)
- [MakeMasksScreen._on_stroke_started](#makemasksscreen_on_stroke_started) (1 entry)
- [MakeMasksScreen._sync_button_states](#makemasksscreen_sync_button_states) (1 entry)

## Module level

### lines 189-193

```python
"model_compare": (
```

STABLE, not alpha: `spacr.qt.maturity` promoted both at launch on the evidence in its own table, and it is the promoted stage the tile lit in. A fallback copied from `app.py`'s literal records the colour before that rewrite, which is a button lighting green-cyan where the tile it replaced lit blue.

### lines 209-211

```python
"napari_bridge": (
```

THE ONLY SOURCE, not a fallback: the bridge registered its own row until the screen folded in here, so nothing puts one in the registry any more and this is what the button reads.

### line 225  _(unsure)_

```python
_HEADLESS_PLATFORMS = ("offscreen", "minimal", "minimalegl", "vnc")
```

Qt platform plugins that have no way for a human to click a dialog button.

### lines 256-258  _(unsure)_

```python
MODE_NONE = "none"
```

Canvas — image + mask overlay with brush/erase mouse handling

### lines 295-299

```python
(MODE_DRAW,         "Draw",         "draw"),
```

THE TWO REGION TOOLS SIT BESIDE THE WAND, not after Zoom. All three answer the same question -- which pixels are one object -- where brush and erase answer it a pixel at a time, and Zoom is not a tool for changing a mask at all. Reaching the row through the fallback put them last in alphabetical order; named here they are placed.

### lines 303-306

```python
(MODE_RECROP,       "Recrop",       "recrop"),
```

RECROP IS LAST, past the tools that change a mask, because it is not one of them: every button left of it edits the field in view, and this one replaces the field in view with the several fields it should have been. Beside Divide it would read as another way to split an object.

### lines 336-340

```python
PAN_MODIFIERS = Qt.ShiftModifier | Qt.AltModifier
```

Held with the left button, these pan from ANY tool. Two of them because window managers eat one or the other: Alt+drag moves the window on most Linux desktops, and Shift+drag is taken by some tablet drivers. Whichever one survives on this machine, panning still works without putting the brush down.

### lines 1187-1189  _(unsure)_

```python
CELLPROB_THRESHOLD = 0.0
```

Cellpose-SAM: the segmentation, and its two intermediate outputs

## _MaskCanvas

### line 419, trailing  _(unsure)_

```python
stroke_started = Signal()
```

emitted just before self.mask is mutated

### line 420, trailing  _(unsure)_

```python
stroke_finished = Signal()
```

emitted after a stroke completes

### line 421, trailing  _(unsure)_

```python
zoom_changed = Signal(bool)
```

emitted with True when zoom entered / False on reset

## _MaskCanvas.__init__

### line 431, trailing  _(unsure)_

```python
self.image: Optional[np.ndarray] = None
```

uint16 grayscale

### line 432, trailing  _(unsure)_

```python
self.mask: Optional[np.ndarray] = None
```

uint8 labels

### lines 457-459

```python
follow_device_ratio(self, self.refresh)
```

The field is composited once and stays up between edits, and a window dragged to another screen fires no resize -- so the recomposite has to be asked for.

### line 468  _(unsure)_

```python
self._zoom_x0: Optional[int] = None
```

Zoom viewport in image coords; None = full-image view.

### lines 474-477

```python
self._zoom_drag_start: Optional[QPoint] = None
```

Zoom-rectangle drag state (widget-local pixel coords). The recrop box is dragged the same way and reuses them, so the two rectangle tools cannot get out of step with each other; which one is being aimed is `self.mode`.

### lines 488-490

```python
self._gesture_points: List[QPoint] = []
```

The draw outline / divide line in flight, in widget coords. Both gestures change nothing until the button comes up, so the path is collected here and converted to image pixels once, on release.

### lines 500-501  _(unsure)_

```python
self._sweeping = False
```

Right-button sweep-delete: one gesture, one undo step, one ledger entry naming every object it took out.

### line 505  _(unsure)_

```python
self._pan_from: Optional[QPoint] = None
```

Shift/Alt + left-drag pan, in widget coords.

## _MaskCanvas.set_image_and_mask

### lines 519-522

```python
self._gesture_points = []
```

A gesture belongs to the field it was started on. The arrow keys move to the next field from anywhere, including the middle of a traced outline, and the points collected on the old field name nothing on the new one.

### lines 524-525

```python
self.recrop_boxes = []
```

The boxes belong to the field they were cut out of; on the next field they would be rectangles drawn over unrelated pixels.

## _MaskCanvas.refresh

### lines 572-576

```python
pixmap = scaled_for(pixmap, self, avail_w, avail_h)
```

Composited at the panel's real pixel density. Everything below that maps a mouse position onto this picture therefore asks `logical_size`, not `pixmap.width()`: the two differ by the device pixel ratio, and a drawn outline that is out by that factor lands on the wrong object.

## _MaskCanvas._canvas_to_image

### lines 580-582  _(unsure)_

```python
def _canvas_to_image(self, x: float, y: float) -> Optional[tuple]:
```

Coordinate mapping (widget-local px  ↔  full image px)

### lines 584-585

```python
"""Widget coordinates to IMAGE pixel coordinates, or ``None``.
```

NB: QLabel.pixmap() returns a *null* QPixmap (never None) when no pixmap is set, so the emptiness test has to be isNull().

### line 612  _(unsure)_

```python
img_x = max(0, min(self.mask.shape[1] - 1, img_x))
```

Clamp to image bounds

## _MaskCanvas.paintEvent

### lines 775-777  _(unsure)_

```python
def paintEvent(self, event):
```

Painting (adds a zoom-rectangle overlay while dragging)

### lines 786-789

```python
self._paint_recrop_boxes()
```

The boxes already cut are drawn under everything else and in every mode: they are the record of what this field has already given up, and they have to be visible while the next box is being aimed as well as after the tool has been put down.

## _MaskCanvas._paint_recrop_boxes

### lines 816-821

```python
rendered = self.pixmap()
```

The pixmap is checked here rather than per box, because it is what every box is mapped through: a paint that arrives before refresh() has composited anything (a resize on a screen that has not loaded a field yet) has nothing to place a rectangle against, and boxes placed at the widget origin instead would each be a blue square over an object they name nothing about.

## _MaskCanvas.mousePressEvent

### lines 967-971

```python
self._gesture_points = [event.position().toPoint()]
```

No stroke is opened here: neither tool touches the mask until the button comes up, and an outline that encloses nothing or a line that separates nothing must leave no undo step and no ledger entry behind it — the same rule the sweep-delete follows in :meth:`_sweep_delete_at`.

### lines 994-997

```python
self._emit_stroke_end(
```

The report goes in the ledger with the click: which way the flood leaked, what tolerance the rescue settled on and whether the budget stopped it are the reasons the wand took what it took, and a mask nobody can explain is a mask nobody trusts.

### line 1005  _(unsure)_

```python
radius = self._mask_radius_for_brush()
```

Brush / erase strokes

## _MaskCanvas.mouseMoveEvent

### lines 1025-1027

```python
if (dx or dy) and self.pan_by(dx, dy):
```

Only re-anchor once the drag has actually moved the view:

discarding sub-pixel drags instead of accumulating them is what makes a slow pan at high zoom stall completely.

### lines 1041-1042

```python
self._gesture_points = [self._gesture_points[0], now]
```

A divide is one straight cut, so the drag moves the far end of the line instead of adding a bend to it.

### lines 1052-1054

```python
self._emit_stroke_start()
```

A drag that *began* outside the pixmap never fired stroke_started, so without this the resulting edit would never be pushed onto the undo history. Idempotent.

## _MaskCanvas.mouseReleaseEvent

### lines 1072-1073

```python
self._emit_stroke_end(kind="sweep_delete", target=list(labels),
```

ONE entry for the whole sweep. Six deletes in the ledger would say six decisions were made; the user made one.

### line 1084  _(unsure)_

```python
p0 = self._canvas_to_image(self._zoom_drag_start.x(),
```

Convert both endpoints to image coords and commit

### lines 1092-1095

```python
if p0 is not None and p1 is not None:
```

Handed on rather than acted on, and handed on even when it is obviously too small: the screen owns the refusal, so the user gets the same sentence for every box that will not be cut instead of silence for some of them.

## _MaskCanvas._finish_region_gesture

### lines 1138-1142

```python
image_points = [p for p in
```

Points that left the pixmap mid-drag are dropped rather than clamped to its edge, which would drag the outline onto the border of the image. A path that lost every point this way arrives as an empty list and is refused below by the same guards that refuse a click: two points do not make a cut, three do not make an outline.

### lines 1157-1159

```python
self._emit_stroke_end(
```

The ledger names both ends of the split: which object was cut and which id the piece that came off it was given, so a later reader can follow one object through the division.

## flow_rgb

### lines 1276-1280

```python
if array.ndim == 3 and array.shape[0] == 2:
```

THE VECTOR SHAPE IS TESTED FIRST. A `(2, H, W)` field also satisfies "three dimensions with at least three along the last one" whenever the image is three pixels wide or more, so testing for a picture first slices the vectors as though they were one and produces a 2-pixel-tall smear.

### lines 1282-1283  _(unsure)_

```python
dy, dx = stretch_to_uint8(array[0]), stretch_to_uint8(array[1])
```

(dY, dX) -> two colour channels plus their magnitude, each stretched on its own so a weak field is still visible.

## load_cellpose_model

### lines 1338-1340

```python
from ...accelerator import cellpose_kwargs
```

gpu= AND device= from the one resolver. Mask Generation is the module people open first, so leaving it CUDA-only while every other entry point took any accelerator was the confusing half-state.

## cellpose_detect

### lines 1387-1390

```python
try:
```

Cellpose has removed eval arguments between minor versions (4.2 has no `invert`, 3.x had no `max_size_fraction`). Offering only what THIS install accepts keeps the screen working across the versions spaCR supports instead of raising TypeError on the one it was written on.

### lines 1397-1401

```python
kwargs = {k: v for k, v in kwargs.items() if k in params}
```

Only filter against a signature that LISTS what it takes. An eval declared `(self, x, **kw)` names nothing, and filtering against it drops every setting the user chose while the run still succeeds -- the thresholds on the panel would then do nothing at all, silently.

## _FlowPane.__init__

### lines 1439-1441

```python
follow_device_ratio(self, self._rescale)
```

The flow picture is composited once per run and then left up, so a move onto a denser screen has to redraw it or it stays soft for the rest of the session.

## _FlowPane.show_rgb

### lines 1449-1451

```python
image = QImage(data.data, width, height, 3 * width,
```

The QImage borrows the buffer, so it is copied before `data` goes out of scope and the pixmap is left pointing at freed memory — which shows up as a garbled pane, not as a crash.

## FoldedModulePanel.__init__

### lines 1557-1559

```python
button.clicked.connect(
```

The bool ``clicked`` emits is swallowed here rather than in every callback: these are the host's own methods, and one that took a stray positional would fail only when pressed.

### lines 1564-1565

```python
self.buttons.setVisible(bool(self.actions))
```

An empty row would be a strip of padding under the module saying nothing; it appears the moment there is a button to put in it.

## NapariBridgeScreen.__init__

### lines 1673-1674

```python
mark_surface(self.status)
```

The log IS this screen's body — nothing is behind it — so it keeps a surface where the sweep would leave it see-through.

### lines 1677-1678  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

## NapariBridgeScreen.open_in_napari

### line 1783

```python
self.say(str(exc))
```

The one refusal that is an instruction rather than an error.

## NapariBridgeScreen.take_mask_back

### lines 1815-1817

```python
self.say(str(exc))
```

Every refusal `to_spacr_mask` raises is written to be read by the person who has to act on it, so it is shown verbatim rather than replaced with a house apology.

### lines 1828-1830

```python
self._handoff = self._reloaded(result)
```

The handoff now holds what is on disk, so pressing the button twice reports "unchanged" rather than recording the same edit a second time.

## MakeMasksScreen.__init__

### line 1920  _(unsure)_

```python
try:
```

Drag & drop — accepts a folder of images to fine-tune against.

### lines 1927-1930

```python
from .settings_model import retarget_field_tooltips
```

HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box you type in. Built here on the field, it is moved onto the label as the last step, so every panel in the application explains itself the same way.

## MakeMasksScreen._build_ui

### lines 1942-1943  _(unsure)_

```python
self._header = ModuleHeader(
```

Masthead — the module's own name and blurb, the folder in force, and the strip of modules that fold into this one.

### lines 1952-1954

```python
self._src_label.setSizePolicy(QSizePolicy.Maximum,
```

A deep folder path must never widen the window or push the fold buttons off the end of the row: the label may shrink below its ideal width, and the tooltip carries what is cut off.

### lines 1964-1968

```python
self._tool_row = self._build_tool_row()
```

The one row of tools, across the top of the body. It is above the canvas and the settings both, so the settings toggle at its far end cannot hide the button that brings the settings back. `_tool_row` is the scroller the row rides in; the row itself is `_tool_row_layout`.

### lines 1998-2001

```python
self._settings_scroll = QScrollArea()
```

THE SETTINGS, AS ONE GROUP. Everything the settings button toggles is inside this one scroll area, so hiding them is one call and the canvas — the splitter's other child — takes the width they give up.

### lines 2012-2013  _(unsure)_

```python
self._body_stack.currentChanged.connect(self._sync_tool_row_visibility)
```

The row belongs to the editor, not to the empty state: there is nothing to brush before a folder is open.

### line 2019  _(unsure)_

```python
nav = QWidget()
```

Bottom nav bar

## MakeMasksScreen._restate_fold_button

### lines 2096-2098

```python
button.style().unpolish(button)
```

A property the stylesheet selects on is only read at polish, so a button already on screen keeps the old colour until it is polished again.

## MakeMasksScreen._build_folded_screen

### lines 2136-2138

```python
screen.compare_requested.connect(self._on_zoo_compare_requested)
```

The zoo's "compare these two" hand-off is wired by whoever hosts it. Folded, that is this screen, or the button would select two models and open nothing.

### lines 2146-2149

```python
from .app_screen import AppScreen
```

A module with no screen of its own gets the generic settings page — the same page its tile opened. Every key this screen folds today has a screen; this is what the next one gets if it does not.

## MakeMasksScreen.seed_folded

### lines 2229-2230  _(unsure)_

```python
screen.apply_settings_dict({"src": self._folder})
```

A module with no screen of its own: a settings page, whose one path is the folder this screen already has open.

## MakeMasksScreen._build_tool_row

### lines 2376-2378  _(unsure)_

```python
def _build_tool_row(self) -> QWidget:
```

The toolbar row and the settings toggle

### lines 2425-2426

```python
self._btn_recrop.setToolTip(RECROP_TOOLTIP)
```

The one tool in the row whose result is not on the canvas, so it is the one that has to say what it does before it is pressed.

### lines 2429-2431

```python
row.addWidget(Divider(Qt.Vertical))
```

Reset zoom, undo and redo ride in the same row: they are pressed between strokes, so hiding them with the settings would hide the two buttons a correction session leans on hardest.

### lines 2463-2465

```python
self._btn_settings.setChecked(True)
```

Checked before it is connected: the settings start on screen and the toggle starts lit, and neither half announces a change that did not happen.

### lines 2470-2477

```python
scroller = QScrollArea()
```

A ROW THAT CANNOT FORCE THE WINDOW WIDER THAN THE DISPLAY. Measured with every tool in it, the row asks for well over 1300px, and a layout minimum that large is not a wide toolbar — it is a window that refuses to be narrowed, so the canvas and the settings go off the right edge with it on a 1366px laptop. Inside a scroll area the row keeps its natural width and the viewport gives up first: a scrollbar on a narrow display, and on a wide one the whole set visible at once, which is the point.

### lines 2485-2488

```python
scroller.setFixedHeight(
```

The bar's own height plus room for the scrollbar that appears when it does not fit: reserved always, so the row does not grow a pixel taller the moment a tool is added and shove the canvas down with it.

## MakeMasksScreen._on_toggle_settings

### lines 2538-2541

```python
sizes = splitter.sizes()
```

A splitter that has never been laid out reports zero for everything; splitting nothing gives the panel a negative width and Qt clamps it to a pane the user cannot see. Fall back to the widths it was born with.

## MakeMasksScreen._build_tools_panel

### line 2557  _(unsure)_

```python
brush_card = Card(title="Brush")
```

Brush size slider

### lines 2644-2647

```python
runaway = QGroupBox("Trim a runaway flood")
```

The three rescues for a flood that escapes down a bright seam. Grouped and defaulted so the panel does not open as a wall of knobs: the group's own checkbox is the master switch, and the numbers under it only matter when the detector misjudges an image.

### line 2790  _(unsure)_

```python
norm_card = Card(title="Display")
```

Display card — contrast percentiles and wheel-zoom speed.

### lines 2794-2797

```python
self._norm_lo.setDecimals(PERCENTILE_DECIMALS)
```

setDecimals BEFORE setRange/setValue: a QDoubleSpinBox rounds both to the precision it has at the time, so setting 99.9999 against the default two decimals stores 100.0 and the control looks broken rather than imprecise.

### lines 2923-2927

```python
for _mode in ("replace", "merge"):
```

THE MODE IS THE ITEM'S DATA, NOT ITS LABEL. `replace` and `merge` are shown to the user and a language switch rewrites the item text in place; reading the mode back off that text would hand `engine.combine_masks` a translated word it has never heard of, so the untranslated key travels with the item instead.

## MakeMasksScreen._install_shortcuts

### lines 2967-2968  _(unsure)_

```python
QShortcut(QKeySequence("R"), self, lambda: self._set_mode(MODE_RECROP))
```

R for recrop. Free: B/E/W/D/V/Z are the other six tools and

Ctrl+S / Ctrl+Z / Ctrl+Y / Escape / the arrows are the rest.

## MakeMasksScreen._on_wand_salvage_changed

### lines 3040-3042

```python
def _on_wand_salvage_changed(self, on: bool):
```

Rescue controls. Each writes one canvas attribute; the canvas builds the dict the flood reads in wand_rescue_settings(), so a control is wired by setting the attribute it names and nothing else.

## MakeMasksScreen._on_undo

### lines 3165-3168

```python
changed = self._diff(self._canvas.mask, prev)
```

Diffed against what is ON the canvas, not against the history head: undo() has already popped, so the head IS `prev` by now and comparing the two would measure every undo as having changed nothing — which is exactly how they went unrecorded.

## MakeMasksScreen._on_detect_otsu

### lines 3303-3305

```python
self._status_label.setText(
```

Replacing with nothing would silently wipe the mask on a flat field, or on one where the minimum area rejected everything. Clearing a mask is what the Clear button is for, and it asks.

## MakeMasksScreen._build_view_tabs

### lines 3330-3332  _(unsure)_

```python
def _build_view_tabs(self) -> QTabWidget:
```

Cellpose-SAM on the open field, and its two intermediates

## MakeMasksScreen._build_cellpose_card

### lines 3398-3401

```python
for name in cellpose_model_choices():
```

THE NAME IS THE ITEM'S DATA, not its label, for the same reason the replace/merge combo carries its mode that way: a language switch rewrites item text in place, and Cellpose has never heard of a translated model name.

## MakeMasksScreen.run_cellpose

### lines 3536-3538

```python
app.processEvents()
```

The button is disabled first, so painting the status line cannot let a second click start a second run on top of this one.

### lines 3564-3565

```python
self._status_label.setText(
```

Replacing with nothing would wipe a mask the user may have spent an hour on, over a threshold that was one notch out.

## MakeMasksScreen._warn

### lines 3598-3600  _(unsure)_

```python
def _warn(self, title: str, text: str) -> None:
```

User messaging (headless-safe — see :func:`is_headless`)

## MakeMasksScreen._should_background_load

### line 3696  _(unsure)_

```python
return False
```

Let the real loader report corrupt/unreadable inputs.

## MakeMasksScreen._handle_load_failure

### lines 3771-3772

```python
self._canvas.image = None
```

Leaving the previous field visible while _current_index names the failed file would let Save write the old mask under a new filename.

## MakeMasksScreen._apply_loaded_pair

### lines 3794-3796

```python
self._recrop_children = []
```

In lockstep with the canvas clearing its own boxes: the cuts belong to the field they were made on, and carrying them onto the next one would retire the wrong file.

### lines 3798-3800

```python
self._reset_flow_panes()
```

The probability and flow panes described the LAST field's

Cellpose run; on this one they would be a picture of the wrong image with nothing on screen saying so.

### line 3802

```python
self._history.clear()
```

Reset undo history for the new image and seed with the loaded mask

### lines 3812-3813

```python
self.apply_object_filter(on_load=True)
```

Last, so its status message and its undo step sit on top of the freshly seeded history rather than being wiped by it.

## MakeMasksScreen._open_ledger

### lines 3830-3832

```python
LOG.warning("Unreadable curation ledger beside %s: %s",
```

A damaged sidecar must not cost the user the edit they are about to make. Start a fresh log, and say so rather than quietly overwriting a record nobody can read.

## MakeMasksScreen._on_recrop_requested

### lines 3841-3843  _(unsure)_

```python
def _on_recrop_requested(self, x0: int, y0: int, x1: int, y1: int) -> None:
```

Recrop — one field becoming the several fields it should have been

## MakeMasksScreen.recrop

### lines 3880-3882

```python
self._image_files.insert(
```

Straight after the field it came from, and after any sibling already cut out of it, so the children come out in the order they were drawn rather than in reverse.

### lines 3886-3888

```python
area = (box[2] - box[0]) * (box[3] - box[1])
```

On the PARENT's ledger, because this is something that was done to the parent: an area of it left. The child's own ledger says the other half of it — see :func:`mask_engine.write_recrop`.

### lines 3893-3896

```python
self._status_label.setText(
```

The object COUNT is the half of this the user cannot see: a box drawn a little too tight round two touching cells cuts both of them and writes a field with nothing in it, and the box on screen looks the same either way.

## MakeMasksScreen.finish_recrop

### lines 3920-3922

```python
if self._canvas.mask is not None:
```

The parent's mask and ledger are written before it is moved, so the record of the boxes travels into the archive with the file they were cut out of rather than being lost with the session.

### lines 3928-3929

```python
LOG.warning("Could not save %s before retiring it: %s",
```

The archive is the recovery, so a mask that will not write must not also stop the original being put somewhere safe.

## MakeMasksScreen._on_prev

### lines 3961-3963

```python
"""Go to the previous field, retiring this one if it was cut up."""
```

Leaving the field retires it if it was cut up, whichever way the user leaves: the parent must not be reachable again as though it were still a field to curate.

## MakeMasksScreen._on_next

### lines 3972-3973

```python
"""Go to the next field, retiring this one if it was cut up."""
```

A retirement has already moved the queue onto the first child, so Next has done what Next does and must not step past it.

## MakeMasksScreen._on_stroke_started

### lines 4054-4056

```python
"""Snapshot the mask before a stroke mutates it in place.
```

Brush/erase strokes mutate the mask in place; nothing to record until the stroke ends. History already has the pre-stroke mask from the previous op/load.

## MakeMasksScreen._sync_button_states

### lines 4086-4089

```python
for b in (self._btn_prev, self._btn_next, self._btn_save,
```

EVERY tool in the row, read off the row itself rather than listed here: a tool added to the mode table is disabled until a folder is open like the rest of them, without anyone having to remember this method exists.
