# Notes from `spacr/qt/sound_preferences.py`

The module carries no comments; this is where its reasons live. Item 427,
part A, built 2026-09-19.

## Why the tab is built last in the dialog, after the Fractal tab

"A Sound tab in Preferences, LAST tab" (427, as filed): it is the least
important thing in Preferences. The launcher-only Fractal tab is appended
after AI when spaceout is on, so the Sound page is created at the very end
of `PreferencesDialog._build_the_dialog`, after it, and
`test_sound_stays_last_when_the_fractal_tab_is_offered` holds that.

## Why every caption is written out at its own line

The first draft kept the five event rows in a table of `(event, label,
object name, tooltip)` and built them in a loop. The runtime-catalog
extractor reads string literals at the calls it knows (`tr`, `setToolTip`,
...), so every caption in that table would have been invisible to it and
stayed English in all nine languages -- the defect
`tests/test_a_helper_does_not_hide_a_caption_from_the_catalog.py` exists
for, in a shape that test does not catch (a table, not a helper). The rows
are therefore written out, and `_event_row` builds only the widgets; its
caller sets the caption and the tooltip as literals.

The sound SET names and descriptions are the exception and are listed as
owed: they live on `SoundTheme` objects in `spacr/qt/sound_synth.py` and
reach `tr()` as attributes. Part C adds ten sets, so the extractor rule
belongs with that change.

## Why Preview is disabled while the master switch is off

"Nothing plays when disabled" (the decision, 2026-09-19). A Preview press is
the user asking to hear a sound, but only once they have switched sound on
in the same dialog; before that the button is greyed and the engine is
never built (`test_preview_is_dead_while_sound_is_off`). A preview uses the
page's own, unsaved, volume and set, so the user can hear a choice before
committing to it; Cancel still writes nothing.

GREYING THE BUTTONS IS NOT THE SAME AS STOPPING THE SOUND, found in review
2026-09-19. `_follow_the_master(False)` disabled the rows and left a music
bed preview -- nine seconds of it -- playing on under a switch that now
said sound was off. Every other sound on this page is over before the user
can reach the switch; the bed is the one that outlives it. So switching the
master off also calls `_end_any_preview`, which is what closing the dialog
already did.

`_end_any_preview` asks `sys.modules` before importing `spacr.qt.sound`,
because this page is built every single time anybody opens Preferences,
sound or no sound, and an unimported engine is an engine with nothing
playing. `_follow_the_master` runs once during `__init__` with the stored
value, which for a fresh install is off -- so without the guard, merely
opening Preferences would import the sound engine.

## Why the Preview buttons say `spacrSilentPress`

With sound saved on, the app-wide input filter hears every press in the
application, this dialog included. A Preview press would therefore answer
with a click sound AND the sound being previewed; on the Click row those
are the same pluck, twice. The buttons carry
`spacr.qt.sound.SILENT_PRESS_PROPERTY` so the filter passes over them. The
name is spelled out here rather than imported, for the reason above, and
`test_a_preview_press_is_not_also_a_click` is what stops the two spellings
drifting apart: it drives the real dialog with sound on, presses Preview,
and then presses the switch beside it to show the filter is still awake.
