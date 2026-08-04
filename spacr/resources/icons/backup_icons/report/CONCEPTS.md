# report - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **report_01** - A page carrying a bar figure and three lines of text.
2. **report_02** - A finished page with a share arrow springing off it.
3. **report_03** - A page dropping into an export tray.
4. **report_04** - Two pages fanned out: the figures sheet over the write-up.
5. **report_05** - The QC verdict stamped at the head of the page, findings beneath.
6. **report_06** - The HTML version: a browser window with a figure inside it.
7. **report_07** - A page published behind a shareable link.
8. **report_08** - The report sent out: a page tucked into an envelope.
9. **report_09** - One page holding all of it: a figure, a table and the settings.
10. **report_10** - A page sealed with a stamp: the versions and settings on record.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
