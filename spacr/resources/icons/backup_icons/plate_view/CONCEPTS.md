# plate_view - candidate concepts

White-on-transparent, 1024x1024 RGBA, spaCR house style
(flat, thin outlines + solid fills, no colour).

1. **plate_view_01** - Wells graded solid to empty across the plate: a heatmap gradient.
2. **plate_view_02** - A hot outer ring: the edge wells solid, the interior barely marked.
3. **plate_view_03** - A magnifier over the plate, the well under it read as a solid disc.
4. **plate_view_04** - The plate beside its value key: solid, outlined and empty wells.
5. **plate_view_05** - A column profile drawn under the plate, rising at both edges.
6. **plate_view_06** - A measured column of bars poured into the plate as well values.
7. **plate_view_07** - A diagonal gradient: one corner of the plate hot, the far one cold.
8. **plate_view_08** - The plate rendered as heat tiles: filled squares against empty ones.
9. **plate_view_09** - A callout hanging off one well, showing the value it holds.
10. **plate_view_10** - One column of the plate reading hot: a plating artefact spotted.

See `_sheet_dark.png` / `_sheet_light.png` for a numbered contact sheet;
each cell also shows the icon at 48 px.

`_sheet_light.png` recolours the artwork through its alpha channel to dark ink.
The PNGs themselves are pure white, so on a light background they are invisible
(the known light-theme bug) - the tinted sheet lets the *shape* be judged there.

Regenerate with:
`QT_QPA_PLATFORM=offscreen python3 _generators/group_results_qc.py`
