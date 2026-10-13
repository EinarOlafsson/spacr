# Notes from `spacr/qt/preferences_navigation.py`

Written by hand on 2026-09-19 for instruction 422.

## Why this is not in `preferences.py`

`spacr/qt/preferences.py` is imported headless — the getters and setters are
read by code that never builds a widget, and `PreferencesDialog` is a factory
precisely so that importing the module costs no QtWidgets. Everything here
needs live widgets, so it lives beside that module rather than inside it.

## Object names, not indices and not captions

A tab INDEX moves the moment a tab is added, and it has: the Fractal tab
appears between Appearance and Performance only when the fractal backdrop is
on, so "Preferences tab 4" is two different pages on two machines.

A tab CAPTION is translated. Matching `"Figures"` against `tabText(i)` works
in English and matches nothing in Korean, which would have made every
preference result in the search field a no-op for eight of the nine shipped
languages.

Every page already sets `PreferencesTabGeneral`, `PreferencesTabAnimation`
and so on, and `tools/build_help_search_index.py` reads those same names off
the same dialog. The two ends of the hand-off are the same string by
construction rather than by a second list.

## Row captions go through `tr` in the same direction

The index stores the ENGLISH caption, because the generator built the dialog
in English. The live dialog shows whatever the interface language is. So the
match is made by translating the index's English with the same `tr` the
dialog used, not by translating the dialog's caption back. Translation is not
reversible; the forward direction is the only one that is defined.

`&` is stripped from both sides: a caption that has picked up a mnemonic is
the same row.

## Reading the finished dialog finds no tooltips at all

Worth recording because it is surprising and it cost a rebuild of the
generator. `explain_every_row` moves a row's sentence from its field onto its
label, and `_everything_explains_itself_in_the_strip` then moves every
remaining one into the hint bar — deliberately, so a control explains itself
once rather than twice, and so the popup cannot cover the strip it
duplicates. By the time `PreferencesDialog(...)` returns, **not one widget in
it has a tooltip**. The generator wraps `explain_every_row` and keeps a copy
of each sentence on the way past; that is what took the preference rows'
descriptions from 0 of 121 to 121 of 121.
