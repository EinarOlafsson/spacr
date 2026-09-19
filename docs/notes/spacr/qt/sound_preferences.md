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
