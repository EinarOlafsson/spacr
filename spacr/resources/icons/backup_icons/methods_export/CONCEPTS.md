# methods_export - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **methods_export_01** - A paragraph with two markers, each on a leader line down to its bar.
2. **methods_export_02** - Blanks in the draft filled from the cells of the results table.
3. **methods_export_03** - Methods on one page, results on the other, one figure feeding both.
4. **methods_export_04** - A sentence's number opened up to show the value behind it.
5. **methods_export_05** - A page whose lower half is the figure the paragraph is talking about.
6. **methods_export_06** - A thread from the marker in the text back down to the plate it came from.
7. **methods_export_07** - A footnote rule, with the source line the paragraph is standing on.
8. **methods_export_08** - The draft being written straight off the stack of run outputs.
9. **methods_export_09** - Numbered references down the margin, each tied to a bar in the chart.
10. **methods_export_10** - One claim in the text branching down onto the three numbers behind it.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_data_io.py`
